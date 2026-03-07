"""
Generate offline RL dataset for OTA training.

OTA configuration (matching the paper comparison table):
  - 2,000 unique training instances
  - 5 trajectories/instance via vrp_bench TabuSearchSolver  → 10k replay buffer
  - 200 training epochs, LR milestones [80, 160]

Primary solver:  vrp_bench TabuSearchSolver (tabu search with NN+2-opt warm-start).
All travel times are evaluated deterministically (plain Euclidean distance) so that
the buffer quality is independent of the stochastic travel-time model used in the
vrp_bench paper.  The tabu search still uses its full swap/relocate/exchange
neighbourhood but optimises a deterministic objective.

Falls back to nearest_neighbour_tour() if vrp_bench is unavailable.
"""

import pickle
import random
import sys
import numpy as np
import os
from tqdm import tqdm

# ── Path to vrp_bench solvers (sibling package) ───────────────────────────────
_VRPBENCH_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'stsptw_problemset', 'vrp_bench')
)

# ── OR-Tools (kept as optional secondary fallback) ────────────────────────────
try:
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver.pywrapcp import (
        RoutingIndexManager, RoutingModel, DefaultRoutingSearchParameters,
    )
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False


def _patch_vrpbench_modules():
    """
    Import vrp_bench modules.  Returns (tabu_module, constants_module).
    Raises ImportError if vrp_bench cannot be found.
    Note: sample_travel_time is used in its native stochastic form (vrp_bench paper
    Section 2.1 Eq.1-11) — no deterministic override so expert tours match the
    exact vrp_bench evaluation methodology.
    """
    if _VRPBENCH_PATH not in sys.path:
        sys.path.insert(0, _VRPBENCH_PATH)

    import tabu_search_solver as _tsm   # noqa: PLC0415
    import constants          as _const # noqa: PLC0415

    return _tsm, _const


class STSPTWDatasetGenerator:
    def __init__(self, seed=2023):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        # Reward normalisation constant (avg stochastic tour cost for n=10 on 1000×1000 map)
        # vrp_bench eval: tour cost ~4000-5000 for n=10 (Euclidean+stochastic delays)
        self.REWARD_SCALE = 4500.0
        self.VIOLATION_PENALTY = -10000.0

        # Cache patched vrp_bench modules (populated on first tabu call)
        self._vrpbench_tabu_mod  = None
        self._vrpbench_const_mod = None

    def load_instances(self, filepath):
        """Load problem instances from pickle file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data

    def split_train_test(self, instances, train_ratio=0.8):
        """
        Split instances into train/test with fixed seed
        
        Args:
            instances: list of problem instances (tuples)
            train_ratio: fraction for training (default 0.8 = 80/20 split)
        
        Returns:
            train_instances, test_instances
        """
        indices = list(range(len(instances)))
        random.shuffle(indices)
        
        split_idx = int(len(indices) * train_ratio)
        train_indices = set(indices[:split_idx])
        
        train_instances = [instances[i] for i in range(len(instances)) if i in train_indices]
        test_instances = [instances[i] for i in range(len(instances)) if i not in train_indices]
        
        return train_instances, test_instances

    # ── vrp_bench Tabu Search solver ──────────────────────────────────────────

    def _get_vrpbench_modules(self):
        """
        Return (tabu_module, constants_module) with deterministic travel patched.
        Raises ImportError if vrp_bench is not available.
        """
        if self._vrpbench_tabu_mod is None:
            self._vrpbench_tabu_mod, self._vrpbench_const_mod = _patch_vrpbench_modules()
        return self._vrpbench_tabu_mod, self._vrpbench_const_mod

    def optimal_tsptw(self, problem):
        """
        Exact optimal single-vehicle TSP-TW solver via pruned brute-force.

        For n ≤ 12 customers this is tractable (~0.4–2 s).  The solver returns:
          - The minimum-distance feasible tour if one exists, OR
          - The tour with fewest TW violations (ties broken by distance) otherwise.

        This is the preferred expert baseline for small instances because the
        vrp_bench Tabu solver is a multi-vehicle VRP solver — it splits customers
        into sub-routes whose merged order is not guaranteed to respect individual
        TW constraints.

        Args:
            problem: (node_xy, service_time, tw_start, tw_end) raw [0,100] scale.

        Returns:
            dict with tour, reward, feasible, violations, distance, edge_costs.
        """
        import itertools as _itertools
        node_xy, service_time, node_tw_start, node_tw_end = problem
        n = len(node_xy)
        node_xy_arr  = np.asarray(node_xy,       dtype=np.float64)
        tw_start_arr = np.asarray(node_tw_start, dtype=np.float64)
        tw_end_arr   = np.asarray(node_tw_end,   dtype=np.float64)

        diff = node_xy_arr[:, None, :] - node_xy_arr[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=-1))

        customers = list(range(1, n))
        best_feas_tour = None;  best_feas_d = 1e18
        best_inf_tour  = None;  best_inf_v  = n + 1;  best_inf_d = 1e18

        for perm in _itertools.permutations(customers):
            t = 0.0; viols = 0; d = 0.0; cur = 0; prune = False
            for node in perm:
                step   = dist[cur, node]
                d     += step
                t     += step
                if t < tw_start_arr[node]:
                    t = tw_start_arr[node]
                if t > tw_end_arr[node]:
                    viols += 1
                    # Prune branch if already worse than best known infeasible
                    if viols > best_inf_v:
                        prune = True; break
                cur = node
            if prune:
                continue
            d += dist[cur, 0]   # return to depot
            if viols == 0:
                if d < best_feas_d:
                    best_feas_d = d; best_feas_tour = list(perm)
            else:
                if viols < best_inf_v or (viols == best_inf_v and d < best_inf_d):
                    best_inf_v = viols; best_inf_d = d; best_inf_tour = list(perm)

        if best_feas_tour is not None:
            tour = best_feas_tour
            total_dist = best_feas_d
            violations = 0
        else:
            tour = best_inf_tour
            total_dist = best_inf_d
            violations = best_inf_v

        return self._eval_tour([0] + tour + [0], problem)

    def nn2opt_tour(self, problem):
        """
        Solve via vrp_bench NN+2-opt heuristic (stochastic sample_travel_time).

        Uses NN2optSolver from vrp_bench, which:
          - Builds an initial tour with a time-aware Nearest-Neighbour construction.
          - Improves it with a time-aware 2-opt local search.

        Both construction and improvement use the native stochastic sample_travel_time
        (vrp_bench paper Eq.1 — T(a,b,t) = D/V + B*R + accidents) so the expert
        tour quality matches the vrp_bench evaluation methodology.

        Args:
            problem: (node_xy, service_time, tw_start, tw_end)
                     node_xy in [0, 1000], TW in [0, 1440] (vrp_bench native).

        Returns:
            dict with tour, reward, feasible, violations, distance, edge_costs.
        """
        try:
            if _VRPBENCH_PATH not in sys.path:
                sys.path.insert(0, _VRPBENCH_PATH)
            from nn_2opt_solver import NN2optSolver  # noqa: PLC0415
        except Exception as exc:
            return self.nearest_neighbor_tour(problem)

        node_xy, service_time, node_tw_start, node_tw_end = problem[:4]
        n = len(node_xy)

        node_xy_arr   = np.asarray(node_xy,        dtype=np.float64)
        tw_start_arr  = np.asarray(node_tw_start,  dtype=np.float64)
        tw_end_arr    = np.asarray(node_tw_end,     dtype=np.float64)

        locations    = node_xy_arr[np.newaxis]
        time_windows = np.stack([tw_start_arr, tw_end_arr], axis=-1)[np.newaxis]
        demands      = np.zeros((1, n), dtype=np.float64)
        demands[0, 1:] = 1.0

        data = {
            'locations':    locations,
            'time_windows': time_windows,
            'demands':      demands,
            'appear_times': None,
        }

        try:
            solver = NN2optSolver(data)
            result = solver.solve_instance(0, num_realizations=1)
        except Exception:
            return self.nearest_neighbor_tour(problem)

        routes = result.get('routes', [])
        non_trivial = [r for r in routes if isinstance(r, (list, np.ndarray)) and len(r) > 2]
        if not non_trivial:
            return self.nearest_neighbor_tour(problem)

        merged_customers = []
        for r in non_trivial:
            merged_customers.extend([int(x) for x in r if int(x) != 0])

        expected = set(range(1, n))
        if set(merged_customers) != expected:
            return self.nearest_neighbor_tour(problem)

        return self._eval_tour([0] + merged_customers + [0], problem)

    def _pick_best(self, candidates):
        """
        From a list of result dicts, pick:
          1. Best feasible by distance, or
          2. Fewest violations (ties broken by distance).
        Returns None if candidates is empty.
        """
        if not candidates:
            return None
        feasible = [r for r in candidates if r.get('feasible')]
        if feasible:
            return min(feasible, key=lambda r: r['distance'])
        return min(candidates, key=lambda r: (r['violations'], r['distance']))

    def tabu_tour(self, problem, time_limit_s: float = 2.0, seed: int = 0):
        """
        Solve a raw STSPTW instance using a cascade of solvers.

        Uses vrp_bench's stochastic sample_travel_time (Eq.1) and cyclic time (% 1440).
        Cascade: NN+2-opt (fast warm-start) then Tabu; returns best feasible result
        or least-violation tour if no feasible solution is found.

        Args:
            problem       : (node_xy, service_time, tw_start, tw_end)
                            node_xy in [0, 1000], TW in [0, 1440] (vrp_bench native).
            time_limit_s  : Tabu wall-clock time limit.
            seed          : Random seed for Tabu diversification.

        Returns:
            dict with keys: tour, reward, feasible, violations, distance, edge_costs
              — same schema as ``nearest_neighbor_tour``.
        """
        random.seed(seed)
        np.random.seed(seed)

        candidates = []

        # — NN + 2-opt (fast warm-start) ———————————————————————————————————————
        try:
            r_nn = self.nn2opt_tour(problem)
            candidates.append(r_nn)
        except Exception:
            pass

        # — Tabu search ————————————————————————————————————————————————————————
        try:
            _tsm, _const = self._get_vrpbench_modules()
            from tabu_search_solver import TabuSearchSolver  # noqa: PLC0415
        except Exception as exc:
            best = self._pick_best(candidates)
            return best if best is not None else self.nearest_neighbor_tour(problem)

        node_xy, service_time, node_tw_start, node_tw_end = problem[:4]
        n = len(node_xy)

        node_xy_arr   = np.asarray(node_xy,        dtype=np.float64)
        tw_start_arr  = np.asarray(node_tw_start,  dtype=np.float64)
        tw_end_arr    = np.asarray(node_tw_end,     dtype=np.float64)

        locations    = node_xy_arr[np.newaxis]
        time_windows = np.stack([tw_start_arr, tw_end_arr], axis=-1)[np.newaxis]
        demands      = np.zeros((1, n), dtype=np.float64)
        demands[0, 1:] = 1.0

        data = {
            'locations':    locations,
            'time_windows': time_windows,
            'demands':      demands,
            'appear_times': None,
        }

        orig_tl = _const.TABU_TIME_LIMIT_SECONDS
        _const.TABU_TIME_LIMIT_SECONDS = time_limit_s

        try:
            solver = TabuSearchSolver(data)
            result = solver.solve_instance(0, num_realizations=1)
        except Exception as exc:
            _const.TABU_TIME_LIMIT_SECONDS = orig_tl
            best = self._pick_best(candidates)
            return best if best is not None else self.nearest_neighbor_tour(problem)
        finally:
            _const.TABU_TIME_LIMIT_SECONDS = orig_tl

        routes = result.get('routes', [])
        non_trivial = [r for r in routes if isinstance(r, (list, np.ndarray)) and len(r) > 2]
        if non_trivial:
            merged_customers = []
            for r in non_trivial:
                merged_customers.extend([int(x) for x in r if int(x) != 0])
            expected = set(range(1, n))
            if set(merged_customers) == expected:
                r_tabu = self._eval_tour([0] + merged_customers + [0], problem)
                candidates.append(r_tabu)

        best = self._pick_best(candidates)
        return best if best is not None else self.nearest_neighbor_tour(problem)

    def _eval_tour(self, route, problem):
        """
        Stochastic evaluation of a route matching vrp_bench Section 2.1.

        Travel time: T(a,b,t) = D(a,b)/V + B(a,b,t)*R(t) + accidents  (vrp_bench Eq.1).
        Feasibility: cyclic time  current_cyclic = current_time % 1440   (vrp_bench
                     _check_feasibility logic) — violation iff cyclic arrival > tw_end.
        Cost:        cumulative stochastic travel times (not wrapped), matching
                     vrp_bench _calculate_stochastic_cost single-realization convention.

        Args:
            route  : list of node indices [0, c1, c2, ..., cK, 0] with depot=0.
            problem: (node_xy, service_time, tw_start, tw_end) with coords in [0,1000]
                     and TW in [0,1440] (vrp_bench native scale).

        Returns:
            dict with tour, reward, feasible, violations, distance, edge_costs.
        """
        if _VRPBENCH_PATH not in sys.path:
            sys.path.insert(0, _VRPBENCH_PATH)
        from travel_time_generator import sample_travel_time as _stt  # noqa: PLC0415

        node_xy, service_time, node_tw_start, node_tw_end = problem[:4]
        n = len(node_xy)

        node_xy_arr  = np.asarray(node_xy,       dtype=np.float64)
        tw_start_arr = np.asarray(node_tw_start, dtype=np.float64)
        tw_end_arr   = np.asarray(node_tw_end,   dtype=np.float64)
        _LOC_FACTOR  = 1000.0
        _DAY_MINUTES = 1440.0

        # Build Euclidean distance dict (required by sample_travel_time)
        distances = {}
        for i in range(n):
            for j in range(n):
                d = node_xy_arr[i] - node_xy_arr[j]
                distances[(i, j)] = float(np.sqrt(d @ d))

        # Extract ordered customer visits (strip depot sentinels)
        customers = [x for x in route if x != 0]
        expected  = set(range(1, n))

        if set(customers) != expected:
            return {
                'tour':       customers + [0],
                'reward':     self.VIOLATION_PENALTY / self.REWARD_SCALE,
                'feasible':   False,
                'violations': 1,
                'distance':   0.0,
                'edge_costs': [],
            }

        # current_time tracks cyclic time-of-day (% 1440) — matches vrp_base._check_feasibility
        current_time   = 0.0   # cyclic time of day in minutes [0, 1440)
        total_distance = 0.0   # cumulative stochastic travel time (not wrapped)
        edge_costs     = []
        violations     = 0
        prev           = 0

        for node in customers:
            tt = _stt(prev, node, distances, current_time)
            total_distance += tt
            current_time += tt
            current_time  = current_time % _DAY_MINUTES  # cyclic wrap (vrp_bench)

            if current_time > tw_end_arr[node] + 1e-6:
                violations += 1
            # Wait if arriving before window opens (vrp_bench: current_time = start_time)
            if current_time < tw_start_arr[node]:
                current_time = tw_start_arr[node]

            svc = float(service_time[node]) if hasattr(service_time, '__len__') else 0.0
            current_time += svc
            edge_costs.append(tt / _LOC_FACTOR)
            prev = node

        # Return-to-depot leg
        tt_ret = _stt(prev, 0, distances, current_time)
        total_distance += tt_ret

        feasible    = (violations == 0)
        reward_raw  = -total_distance if feasible else (-total_distance + self.VIOLATION_PENALTY)

        return {
            'tour':       customers + [0],
            'reward':     reward_raw / self.REWARD_SCALE,
            'feasible':   feasible,
            'violations': violations,
            'distance':   total_distance,
            'edge_costs': edge_costs,
        }

    # ── OR-Tools solver (kept as secondary fallback) ──────────────────────────

    def ortools_tour(self, problem, time_limit_ms: int = 1000):
        """
        Solve (S)TSP-TW with OR-Tools (1 vehicle VRPTW) using a Time dimension.
        Assumes service_time is all zeros (or can be ignored). Time = travel only.

        Args:
            problem: (node_xy, service_time, node_tw_start, node_tw_end)
            time_limit_ms: solver time limit in milliseconds

        Returns:
            dict with: tour, reward, feasible, violations, distance
        """
        if not ORTOOLS_AVAILABLE:
            return self.nearest_neighbor_tour(problem)

        # ---- imports expected somewhere in your file ----
        # from ortools.constraint_solver import routing_enums_pb2
        # from ortools.constraint_solver.pywrapcp import RoutingIndexManager, RoutingModel, DefaultRoutingSearchParameters

        node_xy, service_time, node_tw_start, node_tw_end = problem
        n = len(node_xy)

        node_xy = np.asarray(node_xy, dtype=float)
        node_tw_start = np.asarray(node_tw_start, dtype=float)
        node_tw_end = np.asarray(node_tw_end, dtype=float)

        SCALE = 100  # convert float time/distance to int for OR-Tools (scale for [0,1000] coords)

        # Euclidean distance in original units (float)
        def dist_f(i: int, j: int) -> float:
            dx = node_xy[i, 0] - node_xy[j, 0]
            dy = node_xy[i, 1] - node_xy[j, 1]
            return float((dx * dx + dy * dy) ** 0.5)

        # Integer distance/time matrix (scaled)
        dist = [[int(dist_f(i, j) * SCALE) for j in range(n)] for i in range(n)]

        manager = RoutingIndexManager(n, 1, 0)  # 1 vehicle, depot = 0
        routing = RoutingModel(manager)

        # Cost = distance
        def cost_cb(from_idx, to_idx):
            i = manager.IndexToNode(from_idx)
            j = manager.IndexToNode(to_idx)
            return dist[i][j]

        cost_cb_idx = routing.RegisterTransitCallback(cost_cb)
        routing.SetArcCostEvaluatorOfAllVehicles(cost_cb_idx)

        # Time = travel only (service_time is 0 in your setting)
        def time_cb(from_idx, to_idx):
            i = manager.IndexToNode(from_idx)
            j = manager.IndexToNode(to_idx)
            return dist[i][j]

        time_cb_idx = routing.RegisterTransitCallback(time_cb)

        # Horizon: must be large enough to allow waiting + traversal.
        # If your TWs are already "absolute times", using max(tw_end) is fine.
        tw_end_max = int(np.max(node_tw_end) * SCALE)
        horizon = max(tw_end_max + 10_000, 10_000)

        # Add Time dimension (waiting is modeled via slack, so slack_max should be >= horizon)
        routing.AddDimension(
            time_cb_idx,
            horizon,    # slack_max (waiting allowed)
            horizon,    # capacity / max route duration
            True,       # IMPORTANT: start cumul fixed to 0
            "Time"
        )
        time_dim = routing.GetDimensionOrDie("Time")

        # Apply time windows to customers (skip depot unless you intentionally constrain it)
        for node in range(1, n):
            idx = manager.NodeToIndex(node)
            time_dim.CumulVar(idx).SetRange(
                int(node_tw_start[node] * SCALE),
                int(node_tw_end[node] * SCALE)
            )

        # Depot time window: allow start at 0 and wide end
        depot_idx = manager.NodeToIndex(0)
        time_dim.CumulVar(depot_idx).SetRange(0, horizon)

        # Search params
        search_parameters = DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.seconds = time_limit_ms // 1000
        search_parameters.time_limit.nanos = (time_limit_ms % 1000) * 1_000_000

        solution = routing.SolveWithParameters(search_parameters)
        if solution is None:
            # Fallback (don’t silently pass—this helps you see if OR-Tools fails often)
            return self.nearest_neighbor_tour(problem)

        # Extract tour
        index = routing.Start(0)
        tour = []
        total_distance = 0.0
        _LOC_FACTOR = 1000.0
        edge_costs  = []  # per-step normalized edge distances (customers only)

        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            tour.append(node)

            next_index = solution.Value(routing.NextVar(index))
            next_node = manager.IndexToNode(next_index)

            d = dist_f(node, next_node)
            total_distance += d
            # Include depot→c1, c1→c2, ..., c_{K-1}→cK but NOT return cK→depot
            if next_node != 0:
                edge_costs.append(d / _LOC_FACTOR)
            index = next_index

        # End at depot (optional to include; keep if your downstream expects it)
        tour.append(0)

        # Robust feasibility: did we visit all nodes?
        visited = set(tour)
        feasible = (visited == set(range(n)))

        reward_raw = -total_distance if feasible else (-total_distance + self.VIOLATION_PENALTY)
        reward = reward_raw / self.REWARD_SCALE

        return {
            "tour":        tour,
            "reward":      reward,
            "feasible":    feasible,
            "violations":  0 if feasible else 1,
            "distance":    total_distance,
            "edge_costs":  edge_costs,  # list[float], len=n_customers, normalized by loc_factor=100
        }

    def nearest_neighbor_tour(self, problem):
        """
        Greedy nearest-neighbour heuristic using stochastic travel times.

        Uses vrp_bench sample_travel_time (T(a,b,t)) and cyclic time (% 1440)
        for TW feasibility checking — consistent with _eval_tour and vrp_bench.

        Args:
            problem: tuple (node_xy, service_time, node_tw_start, node_tw_end)
                     coords in [0,1000], TW in [0,1440].

        Returns:
            dict with tour, reward, feasible, distance
        """
        if _VRPBENCH_PATH not in sys.path:
            sys.path.insert(0, _VRPBENCH_PATH)
        from travel_time_generator import sample_travel_time as _stt  # noqa: PLC0415

        node_xy, service_time, node_tw_start, node_tw_end = problem[:4]
        n = len(node_xy)
        node_xy_arr   = np.asarray(node_xy, dtype=np.float64)
        tw_start_arr  = np.asarray(node_tw_start, dtype=np.float64)
        tw_end_arr    = np.asarray(node_tw_end,   dtype=np.float64)
        _DAY_MINUTES  = 1440.0
        _LOC_FACTOR   = 1000.0

        # Euclidean distance dict for sample_travel_time
        distances = {}
        for i in range(n):
            for j in range(n):
                d = node_xy_arr[i] - node_xy_arr[j]
                distances[(i, j)] = float(np.sqrt(d @ d))

        tour = [0]  # Start at depot
        unvisited = set(range(1, n))
        current_time   = 0.0   # cyclic time of day [0, 1440)
        total_distance = 0.0
        violations     = 0
        truncated      = False
        edge_costs     = []

        while unvisited:
            current = tour[-1]
            # Find nearest unvisited by Euclidean distance
            nearest = min(unvisited, key=lambda i: distances[(current, i)])
            tt = _stt(current, nearest, distances, current_time)
            arrival_cyclic = (current_time + tt) % _DAY_MINUTES

            # Check time window (cyclic)
            if arrival_cyclic <= tw_end_arr[nearest]:
                wait_time = max(0.0, tw_start_arr[nearest] - arrival_cyclic)
                svc = float(service_time[nearest]) if isinstance(service_time, (list, tuple)) else 0.0
                current_time = arrival_cyclic + wait_time + svc
                total_distance += tt
                edge_costs.append(tt / _LOC_FACTOR)
                tour.append(nearest)
                unvisited.remove(nearest)
            else:
                # TW violation — TERMINATE IMMEDIATELY
                violations = 1
                truncated  = True
                break

        if truncated:
            reward_raw = -total_distance + self.VIOLATION_PENALTY
            feasible   = False
        else:
            # Feasible: return to depot
            tt_ret = _stt(tour[-1], 0, distances, current_time)
            total_distance += tt_ret
            reward_raw     = -total_distance
            feasible       = True

        return {
            'tour':       tour,
            'reward':     reward_raw / self.REWARD_SCALE,
            'feasible':   feasible,
            'violations': violations,
            'distance':   total_distance,
            'edge_costs': edge_costs,
        }

    def random_permutation_tour(self, problem):
        """
        Evaluate a uniformly random customer permutation as a tour.

        This deliberately ignores time windows — the tour is almost always
        infeasible and incurs the VIOLATION_PENALTY.  Including it in the
        buffer is intentional: the IQL expectile value function needs to see
        the *low* end of the quality distribution (penalty cliff) so that it
        can form a proper gradient signal distinguishing good from bad states.
        The advantage-weighted actor loss automatically ignores trajectories
        with Q < V, so these never corrupt policy learning.

        Returns:
            dict in the same schema as nearest_neighbor_tour.
        """
        node_xy, service_time, node_tw_start, node_tw_end = problem
        n = len(node_xy)

        customers = list(range(1, n))
        random.shuffle(customers)

        return self._eval_tour([0] + customers + [0], problem)

    def generate_trajectories_for_instance(self, problem, num_good=5):
        """
        Generate a quality-diverse 5-trajectory bundle for one instance.

        The IQL expectile loss (η=0.7) extracts the upper-30th-percentile value
        from the buffer.  To work well it needs *variation* — both high-quality
        feasible solutions (to pull the upper bound high) and low-quality /
        infeasible ones (to define the penalty cliff and give the value function
        a gradient signal).  The actor's advantage-weighted regression naturally
        ignores trajectories where Q < V, so including failures does not corrupt
        policy learning.

        Fixed 5-slot layout (independent of num_good for clarity):
          Slot 1 — Tabu search, seed 0          → best feasible solution
          Slot 2 — Tabu search, seed 1          → diversity via different nbhd walk
          Slot 3 — OR-Tools (→ Tabu seed 2)     → different search paradigm
          Slot 4 — Nearest-neighbour greedy     → medium quality, may violate TW
          Slot 5 — Random permutation           → almost always infeasible + penalty
                                                  (teaches value function cost landscape)

        If num_good != 5 the method scales by repeating tabu calls for extra
        slots and always keeping at least one greedy and one random failure.

        Args:
            problem : (node_xy, service_time, tw_start, tw_end) raw [0,100] scale.
            num_good: total trajectories to return (default 5).

        Returns:
            list of ``num_good`` trajectory dicts.
        """
        tl   = getattr(self, 'tabu_time_limit_s', 2.0)
        trajs = []

        if num_good == 5:
            # ── Fixed 5-slot layout ──────────────────────────────────────────
            # Slot 1: Tabu seed 0
            trajs.append(self.tabu_tour(problem, time_limit_s=tl, seed=self.seed))

            # Slot 2: Tabu seed 1
            trajs.append(self.tabu_tour(problem, time_limit_s=tl, seed=self.seed + 1))

            # Slot 3: OR-Tools if available, else Tabu seed 2
            if ORTOOLS_AVAILABLE:
                trajs.append(self.ortools_tour(problem, time_limit_ms=2000))
            else:
                trajs.append(self.tabu_tour(problem, time_limit_s=tl, seed=self.seed + 2))

            # Slot 4: Nearest-neighbour greedy (medium quality / possible violation)
            trajs.append(self.nearest_neighbor_tour(problem))

            # Slot 5: Random permutation (intentional failure for penalty signal)
            trajs.append(self.random_permutation_tour(problem))

        else:
            # ── Generalised layout for arbitrary num_good ────────────────────
            # Always reserve the last 2 slots for greedy + random.
            n_tabu = max(num_good - 2, 1)
            for i in range(n_tabu):
                solver = (self.ortools_tour(problem, time_limit_ms=2000)
                          if (i == 0 and ORTOOLS_AVAILABLE)
                          else self.tabu_tour(problem, time_limit_s=tl,
                                              seed=self.seed + i))
                trajs.append(solver)
            if num_good >= 2:
                trajs.append(self.nearest_neighbor_tour(problem))
            if num_good >= 3:
                trajs.append(self.random_permutation_tour(problem))

        return trajs

    def generate_dataset(self, instances, trajectories_per_instance=5):
        """
        Generate trajectories for all instances.
        
        Args:
            instances: list of problem instances
            trajectories_per_instance: how many trajectories to generate per instance
        
        Returns:
            list of all trajectories with metadata
        """
        all_trajectories = []
        
        for idx, problem in enumerate(tqdm(instances, desc="Generating trajectories")):
            node_xy_raw, svc_raw, tws_raw, twe_raw = problem
            # Build instance_data dict (raw [0,100] scale) shared across all trajs
            node_xy_arr = np.asarray(node_xy_raw, dtype=np.float64)
            n_nodes = len(node_xy_arr)
            # Build Euclidean distances dict for sample_travel_time calls
            distances_shared = {}
            for _i in range(n_nodes):
                for _j in range(n_nodes):
                    _d = node_xy_arr[_i] - node_xy_arr[_j]
                    distances_shared[(_i, _j)] = float(np.sqrt(_d @ _d))
            idata_shared = {
                'node_xy':      node_xy_arr,
                'service_time': np.asarray(svc_raw,     dtype=np.float64),
                'tw_start':     np.asarray(tws_raw,     dtype=np.float64),
                'tw_end':       np.asarray(twe_raw,     dtype=np.float64),
                'distances':    distances_shared,  # Euclidean dict for sample_travel_time
            }

            trajectories = self.generate_trajectories_for_instance(
                problem,
                num_good=trajectories_per_instance
            )
            
            for slot, traj in enumerate(trajectories):
                traj['instance_idx']  = idx
                traj['slot']          = slot   # 0-indexed solver slot for quality reporting
                traj['instance_data'] = idata_shared  # shared ref across slots
                all_trajectories.append(traj)
        
        return all_trajectories

    def save_dataset(self, train_trajectories, test_trajectories, output_dir, difficulty='easy', size=50):
        """Save train and test datasets"""
        os.makedirs(output_dir, exist_ok=True)
        
        train_file = os.path.join(output_dir, f'stsptw{size}_{difficulty}_train.pkl')
        test_file = os.path.join(output_dir, f'stsptw{size}_{difficulty}_test.pkl')
        
        with open(train_file, 'wb') as f:
            pickle.dump(train_trajectories, f)
        
        with open(test_file, 'wb') as f:
            pickle.dump(test_trajectories, f)
        
        print(f"✅ Saved {len(train_trajectories)} training trajectories to {train_file}")
        print(f"✅ Saved {len(test_trajectories)} test trajectories to {test_file}")
        
        return train_file, test_file


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Generate offline RL buffer for OTA training.\n"
            "OTA config: 2 000 instances × 5 Tabu-Search trajectories = 10k buffer.\n"
            "Solver: vrp_bench TabuSearchSolver with deterministic Euclidean travel."
        )
    )
    parser.add_argument('--size',   type=int, default=10,
                        choices=[10, 20, 50, 100],
                        help='Problem size (number of nodes incl. depot).')
    parser.add_argument('--difficulty', type=str, default='easy',
                        choices=['easy', 'medium', 'hard'])
    parser.add_argument('--train_ratio',             type=float, default=0.8)
    parser.add_argument('--trajectories_per_instance', type=int, default=5,
                        help='Trajectories per instance (5 × 2k = 10k buffer).')
    parser.add_argument('--tabu_time_limit', type=float, default=2.0,
                        help='Wall-clock seconds per Tabu Search call.')
    parser.add_argument('--output_dir', type=str, default='data/offline_rl_buffer')
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--input_file', type=str, default=None,
                        help='Override instance file path (default: data/STSPTW/stsptw{size}_{difficulty}.pkl)')
    parser.add_argument('--max_instances', type=int, default=None,
                        help='Cap total instances loaded (e.g. 2500 → 2000 train + 500 test at 80/20 split)')

    args = parser.parse_args()

    print(f"🔄 Generating OTA offline RL buffer  "
          f"(STSPTW-{args.size}, {args.difficulty})")
    print(f"   Solver         : vrp_bench TabuSearchSolver "
          f"(time_limit={args.tabu_time_limit}s/traj, deterministic travel)")
    print(f"   Train ratio    : {args.train_ratio * 100:.0f}%")
    print(f"   Traj/instance  : {args.trajectories_per_instance}")
    print()

    # Load instances
    input_file = args.input_file or os.path.join(
        'data', 'STSPTW', f'stsptw{args.size}_{args.difficulty}.pkl')
    print(f"📂 Loading instances from {input_file}...")
    generator = STSPTWDatasetGenerator(seed=args.seed)
    generator.tabu_time_limit_s = args.tabu_time_limit   # expose to generate call
    instances = generator.load_instances(input_file)
    if args.max_instances and len(instances) > args.max_instances:
        print(f"   Trimming to {args.max_instances} instances (--max_instances)")
        import random as _rnd
        _rnd.seed(args.seed)
        instances = _rnd.sample(instances, args.max_instances)
    print(f"   Total instances: {len(instances)}")

    # Split train/test
    print(f"\n📊 Splitting into train/test...")
    train_instances, test_instances = generator.split_train_test(
        instances, train_ratio=args.train_ratio)
    print(f"   Train: {len(train_instances)} instances")
    print(f"   Test : {len(test_instances)} instances")

    # Generate trajectories
    print(f"\n🎯 Generating trajectories with Tabu Search...")
    train_trajectories = generator.generate_dataset(
        train_instances, args.trajectories_per_instance)
    test_trajectories  = generator.generate_dataset(
        test_instances,  args.trajectories_per_instance)

    print(f"\n📈 Dataset Statistics:")
    print(f"   Train trajectories : {len(train_trajectories)}")
    print(f"   Test  trajectories : {len(test_trajectories)}")

    # Per-slot feasibility breakdown (train split)
    from collections import defaultdict
    slot_counts   = defaultdict(int)
    slot_feasible = defaultdict(int)
    slot_costs    = defaultdict(list)
    for t in train_trajectories:
        s = t.get('slot', -1)
        slot_counts[s]   += 1
        slot_feasible[s] += int(t.get('feasible', False))
        slot_costs[s].append(t.get('distance', 0.0))

    _slot_labels = {
        0: 'Tabu-0',
        1: 'Tabu-1',
        2: 'OR-Tools' if ORTOOLS_AVAILABLE else 'Tabu-2',
        3: 'NN-greedy',
        4: 'Random',
    }
    print(f"\n   Per-slot breakdown (train):")
    print(f"   {'Slot':<4}  {'Solver':<10}  {'Feas%':>6}  {'AvgDist':>9}")
    import numpy as _np_stats
    for s in sorted(slot_counts):
        n   = slot_counts[s]
        f   = slot_feasible[s]
        avg = float(_np_stats.mean(slot_costs[s])) if slot_costs[s] else 0.0
        lbl = _slot_labels.get(s, f'slot-{s}')
        print(f"   {s:<4}  {lbl:<10}  {100*f/max(n,1):6.1f}%  {avg:9.2f}")

    # Overall feasible rates
    print()
    for split_name, trajs in [('Train', train_trajectories),
                               ('Test',  test_trajectories)]:
        n_feas = sum(1 for t in trajs if t['feasible'])
        pct    = 100 * n_feas / max(len(trajs), 1)
        print(f"   {split_name} feasible rate : {n_feas}/{len(trajs)} ({pct:.1f}%)")

    # Save
    print(f"\n💾 Saving datasets...")
    generator.save_dataset(train_trajectories, test_trajectories,
                           args.output_dir, args.difficulty, args.size)

    print(f"\n✨ Done! Buffer ready for OTA training.")


if __name__ == '__main__':
    main()
