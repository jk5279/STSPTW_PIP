#!/usr/bin/env python3
"""
Test 90 STSPTW-trained models each on their matching STSPTW env (same hardness, same delay_weight).
Total: 90 experiments. Results written to CSV.
Run from STSPTWenv/, or set POMO_PIP_DIR to POMO+PIP directory.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
POMO_PIP_DIR = Path(os.environ.get("POMO_PIP_DIR", SCRIPT_DIR / "POMO+PIP")).resolve()
assert POMO_PIP_DIR.is_dir(), f"POMO+PIP dir not found: {POMO_PIP_DIR}"

# 90 configs: (hardness, model_type, delay_weight); delay_weight 0.1 .. 1.0 step 0.1
HARDNESSES = ("easy", "medium", "hard")
MODEL_TYPES = ("POMO", "POMO_STAR", "POMO_STAR_PIP")
DELAY_WEIGHTS = [round(0.1 * i, 1) for i in range(1, 11)]  # 0.1, 0.2, ..., 1.0

# path suffix per model type (dir name ends with _STSPTW10_{h}_dw{dw}{suffix})
SUFFIX_BY_MODEL = {
    "POMO": "",
    "POMO_STAR": "_LM",
    "POMO_STAR_PIP": "_LM_PIMask_1Step",
}

RESULTS_DIR = POMO_PIP_DIR / "results"


def find_stsptw_checkpoint(hardness: str, delay_weight: float, model_type: str) -> Path:
    """Find epoch-10000.pt under results/*_STSPTW10_{hardness}_dw{dw}{suffix}/."""
    suffix = SUFFIX_BY_MODEL[model_type]
    # Match dir name ending, e.g. _STSPTW10_easy_dw0.1 or _STSPTW10_easy_dw0.1_LM
    target_end = f"_STSPTW10_{hardness}_dw{delay_weight}{suffix}"
    for d in RESULTS_DIR.iterdir():
        if d.is_dir() and d.name.endswith(target_end):
            ckpt = d / "epoch-10000.pt"
            if ckpt.is_file():
                return ckpt
    raise FileNotFoundError(f"No checkpoint for STSPTW10 {hardness} dw{delay_weight} {model_type}")


def run_one_test(hardness: str, model_type: str, checkpoint: Path, delay_scale: float) -> dict:
    """Run test.py once; return dict with score, gap, infeasible rates."""
    cmd = [
        sys.executable,
        "test.py",
        "--problem", "STSPTW",
        "--problem_size", "10",
        "--hardness", hardness,
        "--checkpoint", str(checkpoint),
        "--reveal_delay_before_action",
        "--delay_scale", f"{delay_scale:.1f}",
        "--no_opt_sol",
        "--aug_factor", "8",
    ]
    if model_type == "POMO_STAR_PIP":
        cmd += ["--generate_PI_mask", "--pip_step", "1"]

    proc = subprocess.run(
        cmd,
        cwd=POMO_PIP_DIR,
        capture_output=True,
        text=True,
        timeout=600,
    )
    out = proc.stdout + "\n" + proc.stderr
    if proc.returncode != 0:
        return {"error": f"exit {proc.returncode}", "stderr": out[-2000:]}

    no_aug_score = no_aug_gap = aug_score = aug_gap = sol_infeasible = ins_infeasible = None
    for line in out.splitlines():
        m = re.search(r"NO-AUG SCORE:\s*([\d.]+),\s*Gap:\s*([\d.]+)", line)
        if m:
            no_aug_score, no_aug_gap = float(m.group(1)), float(m.group(2))
            continue
        m = re.search(r"AUGMENTATION SCORE:\s*([\d.]+),\s*Gap:\s*([\d.]+)", line)
        if m:
            aug_score, aug_gap = float(m.group(1)), float(m.group(2))
            continue
        m = re.search(r"Solution level Infeasible rate:\s*([\d.]+)%", line)
        if m:
            sol_infeasible = float(m.group(1))
            continue
        m = re.search(r"Instance level Infeasible rate:\s*([\d.]+)%", line)
        if m:
            ins_infeasible = float(m.group(1))
            continue

    if no_aug_score is None or aug_score is None:
        return {"error": "parse_failed", "stdout_tail": out[-1500:]}

    return {
        "no_aug_score": no_aug_score,
        "no_aug_gap": no_aug_gap if no_aug_gap is not None else float("nan"),
        "aug_score": aug_score,
        "aug_gap": aug_gap if aug_gap is not None else float("nan"),
        "sol_infeasible_pct": sol_infeasible if sol_infeasible is not None else float("nan"),
        "ins_infeasible_pct": ins_infeasible if ins_infeasible is not None else float("nan"),
    }


def main():
    import csv
    import argparse
    ap = argparse.ArgumentParser(description="Test 90 STSPTW models on matching STSPTW env")
    ap.add_argument("--dry_run", action="store_true", help="only resolve checkpoints and print first run cmd")
    ap.add_argument("--limit", type=int, default=None, help="limit to first N configs (for testing)")
    args = ap.parse_args()

    configs = []
    for h in HARDNESSES:
        for mt in MODEL_TYPES:
            for dw in DELAY_WEIGHTS:
                configs.append((h, mt, dw))
    if args.limit:
        configs = configs[: args.limit]

    out_csv = SCRIPT_DIR / "../../results/csv/test_stsptw_matched.csv"
    total = len(configs)

    if args.dry_run:
        for i, (h, mt, dw) in enumerate(configs[:5]):
            try:
                ckpt = find_stsptw_checkpoint(h, dw, mt)
                print(f"  {h} {mt} dw{dw} -> {ckpt.name}")
            except FileNotFoundError as e:
                print(f"  {h} {mt} dw{dw} -> NOT FOUND: {e}")
        if configs:
            h, mt, dw = configs[0]
            try:
                ckpt = find_stsptw_checkpoint(h, dw, mt)
                cmd = [sys.executable, "test.py", "--problem", "STSPTW", "--problem_size", "10", "--hardness", h, "--checkpoint", str(ckpt), "--reveal_delay_before_action", "--delay_scale", f"{dw:.1f}", "--no_opt_sol", "--aug_factor", "8"]
                if mt == "POMO_STAR_PIP":
                    cmd += ["--generate_PI_mask", "--pip_step", "1"]
                print("Example cmd (from POMO+PIP):", " ".join(cmd))
            except FileNotFoundError:
                pass
        return

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "hardness", "model_type", "delay_weight",
            "no_aug_score", "no_aug_gap_pct", "aug_score", "aug_gap_pct",
            "sol_infeasible_pct", "ins_infeasible_pct", "error"
        ])

        for idx, (hardness, model_type, delay_weight) in enumerate(configs):
            print(f"[{idx+1}/{total}] {hardness} {model_type} dw{delay_weight} ...", flush=True)
            try:
                checkpoint = find_stsptw_checkpoint(hardness, delay_weight, model_type)
            except FileNotFoundError as e:
                writer.writerow([hardness, model_type, delay_weight, "", "", "", "", "", "", str(e)])
                f.flush()
                continue

            row = run_one_test(hardness, model_type, checkpoint, delay_weight)
            if "error" in row:
                writer.writerow([hardness, model_type, delay_weight, "", "", "", "", "", "", row.get("error", "")])
                print(f"  -> error: {row.get('error')}", flush=True)
            else:
                writer.writerow([
                    hardness, model_type, delay_weight,
                    row["no_aug_score"], row["no_aug_gap"], row["aug_score"], row["aug_gap"],
                    row["sol_infeasible_pct"], row["ins_infeasible_pct"], ""
                ])
            f.flush()

    print(f"Done. Results: {out_csv}")


if __name__ == "__main__":
    main()
