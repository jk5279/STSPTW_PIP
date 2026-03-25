#!/usr/bin/env python3
"""
Generate STSPTW_v2 test datasets with pre-sampled N×N stochastic travel time matrices.

For each (N, hardness, noise_type, cv) combination, loads the base TSPTW test
instances (test_tsptw{N}_{h}.pkl, seed 2026) and pre-samples one stochastic
travel time matrix per instance.  Travel times depend only on pairwise L2 distance
(v2 is state-independent), so the full N×N matrix is fixed for the entire rollout.
Matrices are computed on normalized coordinates (divided by 100, same as the env).

Noise matrix seed: 3000 (distinct from val seed 2025 and base test seed 2026).

Output (one file per config):
    data/TSPTW/test_stsptw_v2_{N}_{h}_{noise_type}_cv{cv}.pkl
Each file stores:
    [(node_xy, service_time, tw_start, tw_end, travel_matrix), ...]
    travel_matrix: np.float32 array of shape (N, N), normalized travel times.
"""

import argparse
import os
import pickle
import sys
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(SCRIPT_DIR, "..")
DATA_DIR = os.path.join(REPO_DIR, "data", "TSPTW")

NOISE_SEED = 3000


def sample_travel_times(distances: np.ndarray, noise_type: str, cv: float,
                         two_point_delta: float = 0.3, two_point_p: float = 0.5,
                         rng: np.random.Generator = None) -> np.ndarray:
    """
    Sample stochastic travel time matrix from pairwise distances.
    distances: (N, N) float32, normalized.
    Returns: (N, N) float32 travel times with E[T_ij] = distances[i,j].
    """
    if rng is None:
        rng = np.random.default_rng()

    safe_dist = np.where(distances < 1e-8, 1e-8, distances)

    if noise_type == "gamma":
        k = 1.0 / (cv ** 2)
        # Gamma(k, scale=d/k) → E[X] = d, Var[X] = d²/k
        scale = safe_dist / k
        travel = rng.gamma(shape=k, scale=scale).astype(np.float32)

    elif noise_type == "two_point":
        delta = two_point_delta
        p = two_point_p
        epsilon = p * delta / (1 - p)  # mean-preserving
        coin = rng.random(size=distances.shape).astype(np.float32)
        travel = np.where(
            coin < p,
            safe_dist * (1 - delta),
            safe_dist * (1 + epsilon),
        ).astype(np.float32)
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")

    # Zero distance → zero travel
    travel = np.where(distances < 1e-8, 0.0, travel).astype(np.float32)
    return travel


def generate(N: int, hardness: str, noise_type: str, cv: float,
             num_samples: int = 10000, data_dir: str = DATA_DIR):
    base_path = os.path.join(data_dir, f"test_tsptw{N}_{hardness}.pkl")
    if not os.path.exists(base_path):
        print(f"  [SKIP] Base file not found: {base_path}")
        return

    with open(base_path, "rb") as f:
        raw = pickle.load(f)[:num_samples]

    rng = np.random.default_rng(NOISE_SEED)

    out = []
    for inst in raw:
        node_xy_raw = np.array(inst[0], dtype=np.float32)   # (N, 2), in [0, 100]
        service_time = inst[1]
        tw_start = inst[2]
        tw_end = inst[3]

        # Normalize to [0, 1] — same as env's load_problems(normalize=True)
        node_xy_norm = node_xy_raw / 100.0  # (N, 2)

        # Pairwise L2 distances on normalized coordinates (N×N)
        diff = node_xy_norm[:, None, :] - node_xy_norm[None, :, :]  # (N, N, 2)
        distances = np.sqrt((diff ** 2).sum(axis=-1)).astype(np.float32)  # (N, N)

        # Sample stochastic travel time matrix
        travel_matrix = sample_travel_times(distances, noise_type, cv, rng=rng)

        out.append((inst[0], inst[1], inst[2], inst[3], travel_matrix))

    cv_str = f"{cv:g}"
    out_name = f"test_stsptw_v2_{N}_{hardness}_{noise_type}_cv{cv_str}.pkl"
    out_path = os.path.join(data_dir, out_name)
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    print(f"  Saved {len(out)} instances → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate STSPTW_v2 test datasets with pre-sampled noise matrices.")
    parser.add_argument("--sizes", type=int, nargs="+", default=[10, 50, 100])
    parser.add_argument("--hardness", nargs="+", default=["easy", "medium", "hard"])
    parser.add_argument("--noise_types", nargs="+", default=["gamma", "two_point"])
    parser.add_argument("--cvs", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    args = parser.parse_args()

    print(f"Noise matrix seed: {NOISE_SEED}")
    print(f"Configs: N={args.sizes}, hardness={args.hardness}, "
          f"noise={args.noise_types}, cv={args.cvs}")
    total = len(args.sizes) * len(args.hardness) * len(args.noise_types) * len(args.cvs)
    print(f"Total files: {total}\n")

    for N in args.sizes:
        for h in args.hardness:
            for nt in args.noise_types:
                for cv in args.cvs:
                    cv_str = f"{cv:g}"
                    print(f"N={N} {h} {nt} cv={cv_str} ...", flush=True)
                    generate(N, h, nt, cv,
                             num_samples=args.num_samples,
                             data_dir=args.data_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
