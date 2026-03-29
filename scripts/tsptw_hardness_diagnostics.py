#!/usr/bin/env python3
"""
Per-hardness TSPTW diagnostics (plan section 6).
For easy, medium, hard: generate a batch of instances and report
  (1) mean and min of tw_end - tw_start per node
  (2) mean travel time between consecutive nodes in a random tour
  (3) ratio (slack / travel)
Run from repo root: python scripts/tsptw_hardness_diagnostics.py
"""
import os
import sys
import torch
import numpy as np

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# Run from POMO+PIP so env imports work
PIPO = os.path.join(REPO_DIR, "POMO+PIP")
os.chdir(PIPO)
sys.path.insert(0, PIPO)

from envs.TSPTWEnv_SPIP import TSPTWEnv_SPIP

BATCH_SIZE = 100
PROBLEM_SIZE = 10
SPEED = 1.0


def travel_time_segments(node_xy, tour, speed=1.0):
    """node_xy: (n, 2), tour: (n,) indices. Return list of travel times between consecutive nodes."""
    n = node_xy.size(0)
    times = []
    for i in range(n - 1):
        a, b = tour[i].item(), tour[i + 1].item()
        d = (node_xy[a] - node_xy[b]).norm(p=2).item()
        times.append(d / speed)
    return times


def stats_for_batch(node_xy, tw_start, tw_end, speed=1.0):
    """node_xy (B,n,2), tw_start (B,n), tw_end (B,n). Return dict of aggregates."""
    B, n = node_xy.size(0), node_xy.size(1)
    duration = (tw_end - tw_start).float()
    mean_dur = duration.view(-1).mean().item()
    min_dur = duration.view(-1).min().item()

    travel_means = []
    ratios = []
    rng = np.random.default_rng(42)
    for b in range(B):
        xy = node_xy[b]
        tw_s = tw_start[b]
        tw_e = tw_end[b]
        perm = rng.permutation(n)
        segs = travel_time_segments(xy, torch.tensor(perm), speed)
        if not segs:
            continue
        mean_travel = np.mean(segs)
        travel_means.append(mean_travel)
        mean_slack = (tw_e - tw_s).float().mean().item()
        if mean_travel > 1e-10:
            ratios.append(mean_slack / mean_travel)
        else:
            ratios.append(float("nan"))

    return {
        "mean_tw_duration": mean_dur,
        "min_tw_duration": min_dur,
        "mean_travel_per_segment": np.mean(travel_means) if travel_means else float("nan"),
        "mean_slack_over_travel": np.nanmean(ratios) if ratios else float("nan"),
    }


def main():
    device = torch.device("cpu")
    print("TSPTW per-hardness diagnostics (n={}, batch={})".format(PROBLEM_SIZE, BATCH_SIZE))
    print("=" * 70)

    for hardness in ["easy", "medium", "hard"]:
        env_params = {
            "problem_size": PROBLEM_SIZE,
            "pomo_size": 1,
            "hardness": hardness,
            "device": device,
        }
        env = TSPTWEnv_SPIP(**env_params)
        node_xy, service_time, tw_start, tw_end = env.get_random_problems(BATCH_SIZE, PROBLEM_SIZE)
        s = stats_for_batch(node_xy, tw_start, tw_end, speed=SPEED)
        print("\n{}:".format(hardness.upper()))
        print("  mean(tw_end - tw_start):     {:.2f}".format(s["mean_tw_duration"]))
        print("  min(tw_end - tw_start):      {:.2f}".format(s["min_tw_duration"]))
        print("  mean travel (random tour):    {:.4f}".format(s["mean_travel_per_segment"]))
        print("  mean(slack / travel):         {:.2f}".format(s["mean_slack_over_travel"]))

    print("\n" + "=" * 70)
    print("Medium has much smaller tw duration and slack/travel than easy; compare with infeasibility rates.")


if __name__ == "__main__":
    main()
