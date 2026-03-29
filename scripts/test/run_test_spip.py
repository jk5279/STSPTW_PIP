#!/usr/bin/env python3
"""
S-PIP (TSPTW_SPIP + POMO_STAR_PIP) evaluation sweep for n=10/50/100, easy/medium/hard.

Checkpoint directory mapping:
  n=10 easy stochastic : results/spip_tsptw10_100M_stochastic/
  n=10 medium/hard     : results/spip_tsptw10_100M/   (no stochastic run exists)
  n=50 all             : results/tsptw_spip50_{hardness}_stochastic/
  n=100 all            : results/tsptw_spip100_{hardness}_stochastic/

Checkpoint selection: trained_model_best.pt if present, else highest-numbered epoch-*.pt.

Run from src/:
    python ../scripts/test/run_test_spip.py
    python ../scripts/test/run_test_spip.py --sizes 100
    python ../scripts/test/run_test_spip.py --sizes 10 --hardness easy
    python ../scripts/test/run_test_spip.py --aug_factor 1

Chunked eval (save after each chunk, resume on re-run):
    python ../scripts/test/run_test_spip.py --sizes 100 --hardness medium --chunk_episodes 500
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from glob import glob

METRIC_KEYS = ("score", "aug_score", "sol_infeasible_pct", "ins_infeasible_pct")


def weighted_merge(merged, merged_n, chunk, chunk_n):
    """Combine per-episode mean metrics across contiguous chunks."""
    if merged is None or merged_n <= 0:
        return dict(chunk), float(chunk_n)
    out = {}
    for k in METRIC_KEYS:
        out[k] = (merged[k] * merged_n + chunk[k] * chunk_n) / (merged_n + chunk_n)
    return out, merged_n + chunk_n


def progress_path_default(csv_path: str, N: int, hardness: str) -> str:
    d = os.path.dirname(os.path.abspath(csv_path)) or "."
    base = os.path.splitext(os.path.basename(csv_path))[0]
    return os.path.join(d, f"{base}_n{N}_{hardness}_progress.json")


def load_progress(path):
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_progress(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def csv_row_exists(csv_path: str, N: int, hardness: str, ckpt_base: str) -> bool:
    if not os.path.isfile(csv_path):
        return False
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if int(row["N"]) == N and row["hardness"] == hardness and row["checkpoint"] == ckpt_base:
                    return True
            except (KeyError, ValueError):
                continue
    return False

# ------------------------------------------------------------------ #
#  Fixed test settings                                                #
# ------------------------------------------------------------------ #
TEST_EPISODES = 10000
AUG_FACTOR    = 8

# PIP mask allocates (batch × aug_factor × pomo × N × N).
# aug_factor=8 multiplies effective batch 8×, easily causing OOM on a shared GPU.
# Default to aug_factor=1 for n≥50; override with --aug_factor 8 on a dedicated node.
BATCH_BY_SIZE = {
    10:  {"test_batch_size": 1000, "aug_batch_size": 2500, "aug_factor": 8},
    50:  {"test_batch_size":  100, "aug_batch_size":  100, "aug_factor": 8},
    100: {"test_batch_size":   25, "aug_batch_size":   25, "aug_factor": 8},
}

SIZES    = [10, 50, 100]
HARDNESS = ["easy", "medium", "hard"]

# (N, hardness) → checkpoint root dir.
# Always evaluate on the stochastic env regardless of how the model was trained.
CONFIGS = {
    (10,  "easy"):   {"root": "results/spip_tsptw10_100M_stochastic",    "stochastic": True},
    (10,  "medium"): {"root": "results/spip_tsptw10_100M",               "stochastic": True},
    (10,  "hard"):   {"root": "results/spip_tsptw10_100M",               "stochastic": True},
    (50,  "easy"):   {"root": "results/tsptw_spip50_easy_stochastic",    "stochastic": True},
    (50,  "medium"): {"root": "results/tsptw_spip50_medium_stochastic",  "stochastic": True},
    (50,  "hard"):   {"root": "results/tsptw_spip50_hard_stochastic",    "stochastic": True},
    (100, "easy"):   {"root": "results/tsptw_spip100_easy_stochastic",   "stochastic": True},
    (100, "medium"): {"root": "results/tsptw_spip100_medium_stochastic", "stochastic": True},
    (100, "hard"):   {"root": "results/tsptw_spip100_hard_stochastic",   "stochastic": True},
}


def find_run_dir(root, N, hardness):
    """Return the most recently modified run dir matching *TSPTW_SPIP{N}_{hardness}*."""
    pattern = os.path.join(root, f"*TSPTW_SPIP{N}_{hardness}*")
    run_dirs = sorted(glob(pattern), key=os.path.getmtime)
    return run_dirs[-1] if run_dirs else None


def find_checkpoint(run_dir):
    """Return trained_model_best.pt if present, else the highest-numbered epoch-*.pt."""
    best = os.path.join(run_dir, "trained_model_best.pt")
    if os.path.exists(best):
        return best
    candidates = glob(os.path.join(run_dir, "epoch-*.pt"))
    if not candidates:
        return None
    # Sort by epoch number, not modification time, to get the highest epoch
    def epoch_num(p):
        base = os.path.basename(p)          # epoch-1234.pt
        try:
            return int(base.split("-")[1].split(".")[0])
        except (IndexError, ValueError):
            return -1
    return max(candidates, key=epoch_num)


def test_data_path(N, hardness):
    if N == 10:
        return f"../data/TSPTW/test_tsptw10_{hardness}.pkl"
    return f"../data/TSPTW/tsptw{N}_{hardness}.pkl"


def parse_metrics(output: str) -> dict:
    """Extract key metrics from test.py stdout."""
    metrics = {}
    m = re.search(r"NO-AUG SCORE:\s*([\d.]+)", output)
    if m:
        metrics["score"] = float(m.group(1))
    m = re.search(r"AUGMENTATION SCORE:\s*([\d.]+)", output)
    if m:
        metrics["aug_score"] = float(m.group(1))
    m = re.search(r"Solution level Infeasible rate:\s*([\d.]+)%", output)
    if m:
        metrics["sol_infeasible_pct"] = float(m.group(1))
    m = re.search(r"Instance level Infeasible rate:\s*([\d.]+)%", output)
    if m:
        metrics["ins_infeasible_pct"] = float(m.group(1))
    return metrics


def run_and_stream(cmd, env) -> tuple[int, str]:
    """Run cmd, stream output to stdout line by line, return (returncode, full_output)."""
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    proc.wait()
    return proc.returncode, "".join(lines)


def build_test_cmd(N, hardness, cfg, ckpt, test_bs, aug_fact, aug_bs, test_episodes, episode_start=0):
    cmd = [
        sys.executable, "test.py",
        "--problem", "TSPTW_SPIP",
        "--problem_size", str(N),
        "--pomo_size", str(N),
        "--hardness", hardness,
        "--generate_PI_mask",
        "--test_episodes", str(test_episodes),
        "--test_batch_size", str(test_bs),
        "--aug_factor", str(aug_fact),
        "--aug_batch_size", str(aug_bs),
        "--test_set_path", test_data_path(N, hardness),
        "--no_opt_sol",
        "--checkpoint", ckpt,
    ]
    if episode_start:
        cmd += ["--test_episode_start", str(episode_start)]
    if cfg["stochastic"]:
        cmd += ["--spip_stochastic_transition", "1"]
    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all S-PIP checkpoints for n=10/50/100 and easy/medium/hard."
    )
    parser.add_argument("--sizes",      type=int,   nargs="+", default=SIZES)
    parser.add_argument("--hardness",               nargs="+", default=HARDNESS)
    parser.add_argument("--test_episodes", type=int, default=TEST_EPISODES,
                        help=f"Total test instances per config (default: {TEST_EPISODES})")
    parser.add_argument("--aug_factor", type=int,   default=None,
                        help="Override aug_factor for all sizes (default: size-dependent: 8 for n=10, 1 for n≥50)")
    parser.add_argument("--aug_batch_size", type=int, default=None,
                        help="Override aug_batch_size (default: size-dependent, see BATCH_BY_SIZE)")
    parser.add_argument("--results_csv", type=str,
                        default=f"../results/spip_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        help="Path to save results CSV (default: results/spip_eval_<timestamp>.csv)")
    parser.add_argument("--chunk_episodes", type=int, default=None,
                        help="If set, evaluate in chunks of this many episodes, save JSON progress after each "
                             "chunk, and resume from last offset on re-run (same --results_csv).")
    parser.add_argument("--progress_json", type=str, default=None,
                        help="Override path for progress JSON (default: <results_csv_basename>_n<N>_<hardness>_progress.json)")
    parser.add_argument("--no_resume", action="store_true",
                        help="Ignore existing progress JSON and start from episode 0.")
    args = parser.parse_args()

    csv_path = args.results_csv
    csv_dir = os.path.dirname(os.path.abspath(csv_path))
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    csv_fields = ["N", "hardness", "stochastic", "aug_factor", "checkpoint",
                  "score", "aug_score", "sol_infeasible_pct", "ins_infeasible_pct"]

    use_append = bool(args.chunk_episodes) and os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0
    csv_file = open(csv_path, "a" if use_append else "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    if not use_append:
        writer.writeheader()

    skipped, ran = [], []
    total_eps = args.test_episodes
    multi_pair = len(args.sizes) * len(args.hardness) > 1
    progress_json_warned = False

    for N in args.sizes:
        for hardness in args.hardness:
            cfg = CONFIGS.get((N, hardness))
            if cfg is None:
                skipped.append(f"N={N} {hardness}: no config entry")
                continue

            if not os.path.isdir(cfg["root"]):
                skipped.append(f"N={N} {hardness}: root dir not found ({cfg['root']})")
                continue

            run_dir = find_run_dir(cfg["root"], N, hardness)
            if run_dir is None:
                skipped.append(f"N={N} {hardness}: no matching run dir in {cfg['root']}")
                continue

            ckpt = find_checkpoint(run_dir)
            if ckpt is None:
                skipped.append(f"N={N} {hardness}: no checkpoint in {run_dir}")
                continue

            stoch_label = "stoch" if cfg["stochastic"] else "det"
            print(f"\n{'='*60}")
            print(f"N={N}  {hardness}  [{stoch_label}]  [{os.path.basename(run_dir)}]")
            print(f"Checkpoint: {ckpt}")
            print(f"{'='*60}")

            batch_cfg = BATCH_BY_SIZE.get(N, BATCH_BY_SIZE[10])
            test_bs  = batch_cfg["test_batch_size"]
            aug_bs   = args.aug_batch_size if args.aug_batch_size is not None else batch_cfg["aug_batch_size"]
            aug_fact = args.aug_factor     if args.aug_factor     is not None else batch_cfg["aug_factor"]

            env = {"PYTORCH_ALLOC_CONF": "expandable_segments:True"}
            env.update(os.environ)

            ckpt_base = os.path.basename(ckpt)

            if args.chunk_episodes:
                if args.progress_json and not multi_pair:
                    prog_path = args.progress_json
                else:
                    if args.progress_json and multi_pair and not progress_json_warned:
                        print("Warning: --progress_json ignored when multiple (N, hardness) pairs; "
                              "using per-pair <results_basename>_n<N>_<hardness>_progress.json")
                        progress_json_warned = True
                    prog_path = progress_path_default(csv_path, N, hardness)
                state = None if args.no_resume else load_progress(prog_path)
                if state:
                    if state.get("checkpoint") != ckpt_base:
                        print(f"Warning: progress checkpoint {state.get('checkpoint')} != {ckpt_base}; "
                              f"starting fresh (use --no_resume or delete {prog_path})")
                        state = None
                    elif state.get("N") != N or state.get("hardness") != hardness:
                        state = None
                    elif int(state.get("total_episodes", total_eps)) != total_eps:
                        print(f"Warning: total_episodes in progress != {total_eps}; starting fresh.")
                        state = None

                if state is None:
                    state = {
                        "version": 1,
                        "N": N,
                        "hardness": hardness,
                        "checkpoint": ckpt_base,
                        "total_episodes": total_eps,
                        "chunk_episodes": args.chunk_episodes,
                        "chunks": [],
                        "merged_metrics": None,
                        "merged_n": 0,
                        "next_offset": 0,
                    }

                merged = state.get("merged_metrics")
                merged_n = float(state.get("merged_n", 0))
                next_off = int(state.get("next_offset", 0))

                if next_off >= total_eps and merged and all(k in merged for k in METRIC_KEYS):
                    print(f"Already complete ({total_eps} episodes). Progress: {prog_path}")
                    if not csv_row_exists(csv_path, N, hardness, ckpt_base):
                        writer.writerow({
                            "N": N, "hardness": hardness,
                            "stochastic": cfg["stochastic"], "aug_factor": aug_fact,
                            "checkpoint": ckpt_base,
                            **merged,
                        })
                        csv_file.flush()
                    ran.append(f"N={N} {hardness}")
                    continue

                while next_off < total_eps:
                    chunk_n = min(args.chunk_episodes, total_eps - next_off)
                    cmd = build_test_cmd(N, hardness, cfg, ckpt, test_bs, aug_fact, aug_bs, chunk_n, episode_start=next_off)
                    print(f"\n--- Chunk episodes [{next_off}, {next_off + chunk_n}) / {total_eps} ---")
                    returncode, output = run_and_stream(cmd, env)
                    if returncode != 0:
                        skipped.append(f"N={N} {hardness}: test.py exited {returncode} at offset {next_off}")
                        break

                    metrics = parse_metrics(output)
                    if not all(k in metrics for k in METRIC_KEYS):
                        skipped.append(
                            f"N={N} {hardness}: could not parse metrics at offset {next_off} "
                            f"(got keys {list(metrics.keys())})"
                        )
                        break

                    merged, merged_n = weighted_merge(merged, merged_n, metrics, float(chunk_n))
                    state["chunks"].append({
                        "start": next_off,
                        "end": next_off + chunk_n,
                        "metrics": metrics,
                    })
                    state["merged_metrics"] = merged
                    state["merged_n"] = merged_n
                    next_off += chunk_n
                    state["next_offset"] = next_off
                    save_progress(prog_path, state)
                    print(f"Saved progress -> {prog_path}  (next_offset={next_off}/{total_eps})")

                if next_off < total_eps:
                    continue

                if merged and not csv_row_exists(csv_path, N, hardness, ckpt_base):
                    writer.writerow({
                        "N": N, "hardness": hardness,
                        "stochastic": cfg["stochastic"], "aug_factor": aug_fact,
                        "checkpoint": ckpt_base,
                        **merged,
                    })
                    csv_file.flush()
                ran.append(f"N={N} {hardness}")
                continue

            cmd = build_test_cmd(N, hardness, cfg, ckpt, test_bs, aug_fact, aug_bs, total_eps, episode_start=0)
            returncode, output = run_and_stream(cmd, env)
            if returncode != 0:
                skipped.append(f"N={N} {hardness}: test.py exited with code {returncode}")
                continue

            metrics = parse_metrics(output)
            row = {
                "N": N, "hardness": hardness,
                "stochastic": cfg["stochastic"], "aug_factor": aug_fact,
                "checkpoint": ckpt_base,
                **metrics,
            }
            writer.writerow(row)
            csv_file.flush()
            ran.append(f"N={N} {hardness}")

    csv_file.close()
    print(f"\nDone. Ran: {len(ran)}  Skipped: {len(skipped)}")
    print(f"Results saved to: {csv_path}")
    if skipped:
        print("Skipped:")
        for s in skipped:
            print(f"  {s}")


if __name__ == "__main__":
    main()
