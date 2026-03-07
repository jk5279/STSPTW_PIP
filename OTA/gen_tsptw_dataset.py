"""
Generate offline RL dataset for OTA training on TSPTW (deterministic).

Produces:
  data/TSPTW/tsptw{N}_{hardness}_train.pkl     — 2000 training instances
  data/TSPTW/tsptw{N}_{hardness}_val.pkl       — 500 val instances
  data/TSPTW/tsptw{N}_{hardness}_buffer.pkl    — 10000 trajectories (5 × 2000)
  data/TSPTW/tsptw{N}_{hardness}_val_opt.pkl   — brute-force optimal for all 500 val

Instance format (each element of train/val pkl):
  (node_xy, service_time, tw_start, tw_end)
  node_xy      : list/array, shape (N, 2), coords in [0, coord_factor=100]
  service_time : list/array, shape (N,), all zeros
  tw_start     : list/array, shape (N,)
  tw_end       : list/array, shape (N,)

Buffer format (each element):
  dict with keys:
    instance_idx  : int   — index into train instances
    slot          : int   — solver slot 0-4
    tour          : list  — [c1, c2, ..., cK, 0]  (customers then depot)
    reward        : float — normalized reward
    feasible      : bool
    violations    : int
    distance      : float — raw Euclidean tour length
    edge_costs    : list  — per-step normalized distances (len = N-1)
    first_action  : int   — first customer visited
    instance_data : dict  — {node_xy, service_time, tw_start, tw_end, distances}

Val optimal format (each element):
  dict with keys:
    instance_idx : int
    tour         : list  — [c1, ..., cK, 0]
    distance     : float
    feasible     : bool
    violations   : int

5-slot trajectory layout (deterministic, Euclidean only):
  Slot 0 — Brute-force optimal             → true optimal (guaranteed feasible for "hard")
  Slot 1 — Greedy NN + 2-opt (seed 0)     → good quality
  Slot 2 — Greedy NN + 2-opt (seed 1)     → diversity via randomised tie-breaking
  Slot 3 — Greedy NN (no 2-opt)            → medium quality, may violate TW
  Slot 4 — Random permutation              → almost always infeasible; teaches penalty cliff
"""

import argparse
import itertools
import os
import pickle
import random
import sys
from multiprocessing import Pool, cpu_count
import numpy as np
from tqdm import tqdm

# ── path setup ─────────────────────────────────────────────────────────────
_OTA_DIR = os.path.abspath(os.path.dirname(__file__))
if _OTA_DIR not in sys.path:
    sys.path.insert(0, _OTA_DIR)

from envs.TSPTWEnv import TSPTWEnv  # noqa: E402

# ── constants ───────────────────────────────────────────────────────────────
REWARD_SCALE     = 300.0   # normalises raw Euclidean distance (~avg feasible tour n=10)
VIOLATION_PENALTY = -10_000.0


# ═══════════════════════════════════════════════════════════════════════════
# Deterministic helper functions
# ═══════════════════════════════════════════════════════════════════════════

def _dist_matrix(node_xy_arr: np.ndarray) -> np.ndarray:
    """Return (N, N) Euclidean distance matrix."""
    diff = node_xy_arr[:, None, :] - node_xy_arr[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def _eval_det(customers: list, dist: np.ndarray,
              tw_start: np.ndarray, tw_end: np.ndarray) -> dict:
    """
    Deterministic evaluation of a customer ordering.

    Travel time = Euclidean distance (speed = 1, no noise, no cyclic wrap).
    Waits if arriving before window opens.
    Violation if arrival > tw_end (after optional wait).

    Args:
        customers : ordered list of customer node indices (no depot sentinels)
        dist      : (N, N) distance matrix
        tw_start  : (N,) time-window start
        tw_end    : (N,) time-window end

    Returns:
        dict with tour, reward, feasible, violations, distance, edge_costs, first_action
    """
    n_cust    = len(customers)
    t         = 0.0
    total_d   = 0.0
    violations = 0
    edge_costs = []
    prev       = 0  # depot

    for node in customers:
        step = dist[prev, node]
        total_d += step
        t       += step
        if t < tw_start[node]:
            t = tw_start[node]        # wait
        if t > tw_end[node] + 1e-8:
            violations += 1
        edge_costs.append(step / REWARD_SCALE)
        prev = node

    total_d += dist[prev, 0]   # return to depot

    feasible  = (violations == 0)
    raw_r     = -total_d if feasible else (-total_d + VIOLATION_PENALTY)

    return {
        'tour':        customers + [0],
        'reward':      raw_r / REWARD_SCALE,
        'feasible':    feasible,
        'violations':  violations,
        'distance':    total_d,
        'edge_costs':  edge_costs,
        'first_action': customers[0] if customers else -1,
    }


# ── Solver 1: Brute-force optimal ──────────────────────────────────────────

def brute_force_optimal(dist: np.ndarray,
                        tw_start: np.ndarray,
                        tw_end: np.ndarray) -> list:
    """
    Pruned brute-force over all customer permutations.
    Feasibility-first, then minimum distance.
    Safe for n ≤ 12 customers (~2 s for n=9, 11 nodes total).

    Returns ordered customer list (no depot).
    """
    n = dist.shape[0]
    customers = list(range(1, n))
    best_feas_tour = None;  best_feas_d = 1e18
    best_inf_tour  = None;  best_inf_v  = n + 1;  best_inf_d = 1e18

    for perm in itertools.permutations(customers):
        t = 0.0; viols = 0; d = 0.0; cur = 0; prune = False
        for node in perm:
            step = dist[cur, node]
            d   += step
            t   += step
            if t < tw_start[node]:
                t = tw_start[node]
            if t > tw_end[node] + 1e-8:
                viols += 1
                if viols > best_inf_v:
                    prune = True; break
            cur = node
        if prune:
            continue
        d += dist[cur, 0]
        if viols == 0:
            if d < best_feas_d:
                best_feas_d = d; best_feas_tour = list(perm)
        else:
            if viols < best_inf_v or (viols == best_inf_v and d < best_inf_d):
                best_inf_v = viols; best_inf_d = d; best_inf_tour = list(perm)

    return best_feas_tour if best_feas_tour is not None else best_inf_tour


# ── Solver 2: Greedy NN construction ───────────────────────────────────────

def greedy_nn(dist: np.ndarray,
              tw_start: np.ndarray,
              tw_end: np.ndarray,
              rng: np.random.Generator = None) -> list:
    """
    Time-aware nearest-neighbour construction.
    Breaks distance ties randomly (controlled by rng).
    Does NOT prune on TW violations — always produces a full permutation.
    """
    n = dist.shape[0]
    unvisited = list(range(1, n))
    if rng is not None:
        rng.shuffle(unvisited)

    tour  = []
    t     = 0.0
    cur   = 0

    while unvisited:
        # Pick nearest by Euclidean distance; break ties by tw_start
        nearest = min(unvisited, key=lambda j: (dist[cur, j], tw_start[j]))
        step    = dist[cur, nearest]
        t      += step
        if t < tw_start[nearest]:
            t = tw_start[nearest]
        tour.append(nearest)
        unvisited.remove(nearest)
        cur = nearest

    return tour


# ── Solver 3: 2-opt improvement ────────────────────────────────────────────

def two_opt(tour: list,
            dist: np.ndarray,
            tw_start: np.ndarray,
            tw_end: np.ndarray,
            max_iter: int = 400) -> list:
    """
    Standard 2-opt local search on the customer sequence.
    Evaluates tour quality deterministically and accepts improving swaps.
    """
    best   = list(tour)
    best_d = _eval_det(best, dist, tw_start, tw_end)['distance']
    improved = True
    iters    = 0

    while improved and iters < max_iter:
        improved = False
        iters   += 1
        n = len(best)
        for i in range(n - 1):
            for j in range(i + 2, n):
                candidate = best[:i+1] + best[i+1:j+1][::-1] + best[j+1:]
                r = _eval_det(candidate, dist, tw_start, tw_end)
                # Accept if strictly better distance (ignore feasibility —
                # IQL can learn from both feasible and infeasible tours)
                if r['distance'] < best_d - 1e-8:
                    best   = candidate
                    best_d = r['distance']
                    improved = True
    return best


def nn2opt(dist: np.ndarray,
           tw_start: np.ndarray,
           tw_end: np.ndarray,
           rng: np.random.Generator = None) -> list:
    """NN construction followed by 2-opt improvement."""
    init = greedy_nn(dist, tw_start, tw_end, rng=rng)
    return two_opt(init, dist, tw_start, tw_end)


# ── Solver 4: Random permutation ───────────────────────────────────────────

def random_perm(n: int, rng: np.random.Generator) -> list:
    """Shuffle all customer indices uniformly."""
    customers = list(range(1, n))
    rng.shuffle(customers)
    return customers


# ═══════════════════════════════════════════════════════════════════════════
# Trajectory bundle builder
# ═══════════════════════════════════════════════════════════════════════════

def _make_idata(node_xy_arr, tw_start_arr, tw_end_arr, dist):
    return {
        'node_xy':      node_xy_arr,
        'service_time': np.zeros(len(node_xy_arr), dtype=np.float32),
        'tw_start':     tw_start_arr,
        'tw_end':       tw_end_arr,
        'distances':    {(i, j): float(dist[i, j])
                         for i in range(len(node_xy_arr))
                         for j in range(len(node_xy_arr))},
    }


def generate_trajectories(problem: tuple, instance_idx: int, seed: int = 0) -> list[dict]:
    """
    Generate 5-slot trajectory bundle for one TSPTW instance.

    All solvers use deterministic Euclidean travel times.

    Slot layout:
      0 — Brute-force optimal   (true optimum; feasible for 'hard' instances)
      1 — NN + 2-opt, seed 0    (good quality, diverse start)
      2 — NN + 2-opt, seed 1    (diversity via different random tie-breaking)
      3 — Greedy NN (no 2-opt)  (medium quality, possible TW violations)
      4 — Random permutation    (intentionally bad; trains penalty-cliff value)

    Args:
        problem      : (node_xy, service_time, tw_start, tw_end) in raw [0,100] coords
        instance_idx : index of this instance in the training set
        seed         : base RNG seed for reproducibility

    Returns:
        List of 5 trajectory dicts, each containing instance_data.
    """
    node_xy, _, tw_start, tw_end = problem
    node_xy_arr  = np.asarray(node_xy,    dtype=np.float64)
    tw_start_arr = np.asarray(tw_start,   dtype=np.float64)
    tw_end_arr   = np.asarray(tw_end,     dtype=np.float64)
    n            = node_xy_arr.shape[0]
    dist         = _dist_matrix(node_xy_arr)
    idata        = _make_idata(node_xy_arr, tw_start_arr.astype(np.float32),
                                tw_end_arr.astype(np.float32), dist)

    rng0 = np.random.default_rng(seed)
    rng1 = np.random.default_rng(seed + 1)

    trajs = []
    for slot, customers in enumerate([
        brute_force_optimal(dist, tw_start_arr, tw_end_arr),   # 0 — optimal
        nn2opt(dist, tw_start_arr, tw_end_arr, rng=rng0),      # 1 — NN+2opt s0
        nn2opt(dist, tw_start_arr, tw_end_arr, rng=rng1),      # 2 — NN+2opt s1
        greedy_nn(dist, tw_start_arr, tw_end_arr),              # 3 — greedy NN
        random_perm(n, np.random.default_rng(seed + 2)),        # 4 — random
    ]):
        result = _eval_det(customers, dist, tw_start_arr, tw_end_arr)
        result['instance_idx']  = instance_idx
        result['slot']          = slot
        result['instance_data'] = idata
        trajs.append(result)

    return trajs


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# Top-level worker functions (must be picklable for multiprocessing.Pool)
# ═══════════════════════════════════════════════════════════════════════════

def _train_worker(args_tuple):
    """Pool worker: generate 5 trajectories for one training instance."""
    idx, problem, seed = args_tuple
    return generate_trajectories(problem, instance_idx=idx, seed=seed)


def _val_worker(args_tuple):
    """Pool worker: compute brute-force optimal for one val instance."""
    idx, problem = args_tuple
    node_xy, _, tw_start, tw_end = problem
    node_xy_arr  = np.asarray(node_xy,  dtype=np.float64)
    tw_start_arr = np.asarray(tw_start, dtype=np.float64)
    tw_end_arr   = np.asarray(tw_end,   dtype=np.float64)
    dist         = _dist_matrix(node_xy_arr)
    customers    = brute_force_optimal(dist, tw_start_arr, tw_end_arr)
    result       = _eval_det(customers, dist, tw_start_arr, tw_end_arr)
    return {
        'instance_idx': idx,
        'tour':         result['tour'],
        'distance':     result['distance'],
        'feasible':     result['feasible'],
        'violations':   result['violations'],
    }


def parse_args():
    p = argparse.ArgumentParser(description="Generate TSPTW dataset for OTA training")
    p.add_argument('--problem_size', type=int, default=10,     help='Number of customers (depot not counted)')
    p.add_argument('--hardness',   type=str, default='hard',  choices=['hard', 'medium', 'easy'])
    p.add_argument('--n_train',    type=int, default=2000,    help='Number of training instances')
    p.add_argument('--n_val',      type=int, default=500,     help='Number of validation instances')
    p.add_argument('--traj_per',   type=int, default=5,       help='Trajectories per training instance')
    p.add_argument('--output_dir', type=str,
                   default=os.path.join(_OTA_DIR, 'data', 'TSPTW'),
                   help='Output directory')
    p.add_argument('--seed',       type=int, default=2024)
    return p.parse_args()


def _save(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved → {path}  ({len(obj)} items)")


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    n_nodes = args.problem_size + 1  # includes depot
    out_dir = args.output_dir
    tag     = f"tsptw{args.problem_size}_{args.hardness}"

    # ── paths ───────────────────────────────────────────────────────────────
    train_path   = os.path.join(out_dir, f"{tag}_train.pkl")
    val_path     = os.path.join(out_dir, f"{tag}_val.pkl")
    buffer_path  = os.path.join(out_dir, f"{tag}_buffer.pkl")
    val_opt_path = os.path.join(out_dir, f"{tag}_val_opt.pkl")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1: Generate instances
    # ─────────────────────────────────────────────────────────────────────────
    total = args.n_train + args.n_val
    print(f"\n{'='*60}")
    print(f"Generating {total} instances  (n={args.problem_size}, hardness={args.hardness})")
    print(f"{'='*60}")

    env = TSPTWEnv(
        problem_size=n_nodes,
        pomo_size=n_nodes,
        hardness=args.hardness,
        k_sparse=n_nodes + 1,   # no sparsity
        device='cpu',
    )

    # Generate in one batch (TSPTWEnv returns tensors; convert to numpy lists)
    node_xy_t, svc_t, tws_t, twe_t = env.get_random_problems(
        total, n_nodes,
        coord_factor=100, max_tw_size=100
    )

    instances = []
    for i in range(total):
        instances.append((
            node_xy_t[i].numpy().tolist(),
            svc_t[i].numpy().tolist(),
            tws_t[i].numpy().tolist(),
            twe_t[i].numpy().tolist(),
        ))

    train_instances = instances[:args.n_train]
    val_instances   = instances[args.n_train:]

    print(f"  Training: {len(train_instances)}  |  Val: {len(val_instances)}")
    _save(train_instances, train_path)
    _save(val_instances,   val_path)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2: Collect trajectories for training buffer
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Collecting trajectories  (5 × {len(train_instances)} = "
          f"{5 * len(train_instances)} total)")
    print(f"{'='*60}")
    print("  Slot 0: brute-force optimal")
    print("  Slot 1: NN + 2-opt (seed 0)")
    print("  Slot 2: NN + 2-opt (seed 1)")
    print("  Slot 3: greedy NN")
    print("  Slot 4: random permutation")

    n_workers = min(cpu_count(), 8)
    print(f"  Using {n_workers} workers")
    work_items = [(idx, problem, args.seed + idx)
                  for idx, problem in enumerate(train_instances)]

    buffer = []
    feasible_counts = [0] * 5
    with Pool(n_workers) as pool:
        for trajs in tqdm(pool.imap(_train_worker, work_items, chunksize=4),
                          total=len(work_items), desc="Buffer"):
            for t in trajs:
                buffer.append(t)
                if t['feasible']:
                    feasible_counts[t['slot']] += 1

    print(f"\n  Buffer size: {len(buffer)}")
    print(f"  Feasibility by slot:")
    for slot, cnt in enumerate(feasible_counts):
        print(f"    Slot {slot}: {cnt}/{len(train_instances)} "
              f"({cnt/len(train_instances)*100:.1f}%)")
    best_dist = min((t['distance'] for t in buffer if t['feasible']), default=None)
    print(f"  Best feasible distance in buffer: "
          f"{best_dist:.2f}" if best_dist else "  No feasible tour found!")
    _save(buffer, buffer_path)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3: Brute-force optimal for val instances
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Computing brute-force optimal for {len(val_instances)} val instances")
    print(f"{'='*60}")

    val_work = list(enumerate(val_instances))
    val_opt  = []
    n_feas   = 0

    with Pool(n_workers) as pool:
        for entry in tqdm(pool.imap(_val_worker, val_work, chunksize=4),
                          total=len(val_work), desc="Val opt"):
            val_opt.append(entry)
            if entry['feasible']:
                n_feas += 1

    print(f"  Feasible optimal: {n_feas}/{len(val_instances)} "
          f"({n_feas/len(val_instances)*100:.1f}%)")
    avg_opt = np.mean([e['distance'] for e in val_opt if e['feasible']])
    print(f"  Mean optimal distance (feasible): {avg_opt:.2f}")
    _save(val_opt, val_opt_path)

    # ─────────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Done!")
    print(f"  Training instances : {train_path}")
    print(f"  Val instances      : {val_path}")
    print(f"  Training buffer    : {buffer_path}")
    print(f"  Val optimal        : {val_opt_path}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
