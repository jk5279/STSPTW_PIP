#!/usr/bin/env python3
"""
STSPTW_v2 evaluation sweep.

Loops over all (N, hardness, noise_type, cv, model) configs, finds the
matching checkpoint under results/sweep_v2_n{N}/, and evaluates with a
fixed noise seed so all models see identical noise realizations.

Run from STSPTW_PIP/POMO+PIP/:
    python ../scripts/test/run_test_v2.py
    python ../scripts/test/run_test_v2.py --sizes 50 100
"""

import argparse
import os
import subprocess
import sys
from glob import glob

# ------------------------------------------------------------------ #
#  Fixed test settings — do not change                                #
# ------------------------------------------------------------------ #
TEST_NOISE_SEED = 42   # all models see identical noise realizations
AUG_FACTOR      = 1   # must be 1 with pre-sampled noise matrix
TEST_EPISODES   = 10000
TEST_BATCH_SIZE = 1000

SIZES       = [10, 50, 100]
HARDNESS    = ["easy", "medium", "hard"]
NOISE_TYPES = ["gamma", "two_point"]
CVS         = [0.25, 0.5, 1.0]


def find_checkpoint(run_dir, checkpoint_name=None):
    if checkpoint_name:
        p = os.path.join(run_dir, checkpoint_name)
        return p if os.path.exists(p) else None
    candidates = glob(os.path.join(run_dir, "*.pt"))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes",       type=int,   nargs="+", default=SIZES)
    parser.add_argument("--hardness",                nargs="+", default=HARDNESS)
    parser.add_argument("--noise_types",             nargs="+", default=NOISE_TYPES)
    parser.add_argument("--cvs",         type=float, nargs="+", default=CVS)
    parser.add_argument("--checkpoint_name", type=str, default=None,
                        help="Checkpoint filename to use (e.g. 'epoch-10000.pt'). "
                             "If omitted, picks the most recently modified .pt file.")
    args = parser.parse_args()

    skipped, ran = [], []

    for N in args.sizes:
        results_root = f"./results/sweep_v2_n{N}"
        if not os.path.isdir(results_root):
            print(f"[SKIP] {results_root} not found")
            continue

        for hardness in args.hardness:
            for noise_type in args.noise_types:
                for cv in args.cvs:
                    cv_str = f"{cv:g}"
                    pattern = os.path.join(
                        results_root,
                        f"*STSPTW_v2{N}_{hardness}_{noise_type}_cv{cv_str}*"
                    )
                    run_dirs = sorted(glob(pattern))
                    if not run_dirs:
                        skipped.append(f"N={N} {hardness} {noise_type} cv={cv_str} (no run dir)")
                        continue

                    for run_dir in run_dirs:
                        ckpt = find_checkpoint(run_dir, args.checkpoint_name)
                        if ckpt is None:
                            skipped.append(f"{run_dir} (no checkpoint)")
                            continue

                        pip_flag = ["--generate_PI_mask"] if "PIMask" in run_dir else []

                        print(f"\n{'='*60}")
                        print(f"N={N} {hardness} {noise_type} cv={cv_str}  [{os.path.basename(run_dir)}]")
                        print(f"{'='*60}")

                        cmd = [
                            sys.executable, "test.py",
                            "--problem", "STSPTW_v2",
                            "--problem_size", str(N), "--pomo_size", str(N),
                            "--hardness", hardness,
                            "--noise_type", noise_type,
                            "--cv", cv_str,
                            "--reveal_delay_before_action",
                            "--test_noise_seed", str(TEST_NOISE_SEED),
                            "--aug_factor", str(AUG_FACTOR),
                            "--test_episodes", str(TEST_EPISODES),
                            "--test_batch_size", str(TEST_BATCH_SIZE),
                            "--test_set_path", f"../data/TSPTW/test_tsptw{N}_{hardness}.pkl",
                            "--no_opt_sol",
                            "--checkpoint", ckpt,
                        ] + pip_flag

                        subprocess.run(cmd, check=True)
                        ran.append(run_dir)

    print(f"\nDone. Ran: {len(ran)}  Skipped: {len(skipped)}")
    if skipped:
        print("Skipped:")
        for s in skipped:
            print(f"  {s}")


if __name__ == "__main__":
    main()
