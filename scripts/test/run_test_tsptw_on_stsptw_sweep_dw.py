#!/usr/bin/env python3
"""
Test 9 TSPTW-trained models on STSPTW env with delay_scale sweep 0.01 ~ 1.0 (step 0.01).
Total: 9 × 100 = 900 experiments. Results written to CSV.
Run from project root (STSPTWenv), or set POMO_PIP_DIR to POMO+PIP directory.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

# Default: script lives in STSPTWenv, so POMO+PIP is STSPTWenv/POMO+PIP
SCRIPT_DIR = Path(__file__).resolve().parent
POMO_PIP_DIR = Path(os.environ.get("POMO_PIP_DIR", SCRIPT_DIR / "POMO+PIP")).resolve()
assert POMO_PIP_DIR.is_dir(), f"POMO+PIP dir not found: {POMO_PIP_DIR}"

# 9 TSPTW checkpoints: (hardness, model_type, path_suffix)
# path_suffix: "" -> *_TSPTW10_{h}/, "_LM" -> *_TSPTW10_{h}_LM/, "_LM_PIMask_1Step" -> *_TSPTW10_{h}_LM_PIMask_1Step/
TSPTW_CONFIGS = [
    ("easy", "POMO", ""),
    ("easy", "POMO_STAR", "_LM"),
    ("easy", "POMO_STAR_PIP", "_LM_PIMask_1Step"),
    ("medium", "POMO", ""),
    ("medium", "POMO_STAR", "_LM"),
    ("medium", "POMO_STAR_PIP", "_LM_PIMask_1Step"),
    ("hard", "POMO", ""),
    ("hard", "POMO_STAR", "_LM"),
    ("hard", "POMO_STAR_PIP", "_LM_PIMask_1Step"),
]

# Resolve checkpoint paths (first matching directory per config)
RESULTS_DIR = POMO_PIP_DIR / "results"


def find_tsptw_checkpoint(hardness: str, suffix: str) -> Path:
    """Find epoch-10000.pt under results/*_TSPTW10_{hardness}{suffix}/."""
    pattern = f"*_TSPTW10_{hardness}{suffix}"
    for d in RESULTS_DIR.iterdir():
        if d.is_dir() and d.name.endswith(f"_TSPTW10_{hardness}{suffix}"):
            ckpt = d / "epoch-10000.pt"
            if ckpt.is_file():
                return ckpt
    raise FileNotFoundError(f"No checkpoint for TSPTW10 {hardness} {suffix}")


def run_one_test(hardness: str, model_type: str, checkpoint: Path, delay_scale: float) -> dict:
    """Run test.py once; return dict with score, gap, infeasible rates (and raw stdout on parse failure)."""
    cmd = [
        sys.executable,
        "test.py",
        "--problem", "STSPTW",
        "--problem_size", "10",
        "--hardness", hardness,
        "--checkpoint", str(checkpoint),
        "--reveal_delay_before_action",
        "--delay_scale", f"{delay_scale:.2f}",
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

    # Parse final lines, e.g.:
    #  NO-AUG SCORE: 123.456, Gap: 1.23
    #  AUGMENTATION SCORE: 123.456, Gap: 1.23
    # 123.456 (1.230%)
    # 123.456 (1.230%)
    # Solution level Infeasible rate: 0.000%
    # Instance level Infeasible rate: 0.000%
    no_aug_score = no_aug_gap = aug_score = aug_gap = sol_infeasible = ins_infeasible = None
    for line in out.splitlines():
        # some runs (medium/hard POMO) can print 'nan' as score; accept it
        m = re.search(r"NO-AUG SCORE:\s*([\d.]+|nan),\s*Gap:\s*([\d.]+)", line)
        if m:
            no_aug_score, no_aug_gap = float(m.group(1)), float(m.group(2))
            continue
        m = re.search(r"AUGMENTATION SCORE:\s*([\d.]+|nan),\s*Gap:\s*([\d.]+)", line)
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

    # if we at least parsed infeasible rates, treat as success even if score is nan
    if sol_infeasible is None or ins_infeasible is None:
        return {"error": "parse_failed", "stdout_tail": out[-1500:]}

    return {
        "no_aug_score": float("nan") if no_aug_score is None else no_aug_score,
        "no_aug_gap": float("nan") if no_aug_gap is None else no_aug_gap,
        "aug_score": float("nan") if aug_score is None else aug_score,
        "aug_gap": float("nan") if aug_gap is None else aug_gap,
        "sol_infeasible_pct": sol_infeasible if sol_infeasible is not None else float("nan"),
        "ins_infeasible_pct": ins_infeasible if ins_infeasible is not None else float("nan"),
    }


def main():
    import csv
    import argparse
    ap = argparse.ArgumentParser(description="TSPTW-on-STSPTW delay_scale sweep (9×100=900 runs)")
    ap.add_argument("--dry_run", action="store_true", help="only resolve checkpoints and print first run cmd")
    ap.add_argument("--limit_models", type=int, default=None, help="limit to first N model configs (for testing)")
    ap.add_argument("--limit_dw", type=int, default=None, help="limit to first N delay values, e.g. 3 for 0.01,0.02,0.03")
    args = ap.parse_args()

    delay_values = [round(0.01 * i, 2) for i in range(1, 101)]  # 0.01 .. 1.0
    if args.limit_dw:
        delay_values = delay_values[: args.limit_dw]
    configs = TSPTW_CONFIGS if not args.limit_models else TSPTW_CONFIGS[: args.limit_models]

    out_csv = SCRIPT_DIR / "../../results/csv/test_tsptw_on_stsptw_dw_sweep.csv"
    total = len(configs) * len(delay_values)
    run_idx = 0

    if args.dry_run:
        for hardness, model_type, suffix in configs:
            try:
                ckpt = find_tsptw_checkpoint(hardness, suffix)
                print(f"  {hardness} {model_type} -> {ckpt}")
            except FileNotFoundError as e:
                print(f"  {hardness} {model_type} -> NOT FOUND: {e}")
        dw = delay_values[0]
        h, mt, suf = configs[0]
        ckpt = find_tsptw_checkpoint(h, suf)
        cmd = [sys.executable, "test.py", "--problem", "STSPTW", "--problem_size", "10", "--hardness", h, "--checkpoint", str(ckpt), "--reveal_delay_before_action", "--delay_scale", f"{dw:.2f}", "--no_opt_sol", "--aug_factor", "8"]
        if mt == "POMO_STAR_PIP":
            cmd += ["--generate_PI_mask", "--pip_step", "1"]
        print("Example cmd (from POMO+PIP):", " ".join(cmd))
        return

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "hardness", "model_type", "delay_scale",
            "no_aug_score", "no_aug_gap_pct", "aug_score", "aug_gap_pct",
            "sol_infeasible_pct", "ins_infeasible_pct", "error"
        ])

        for hardness, model_type, suffix in configs:
            try:
                checkpoint = find_tsptw_checkpoint(hardness, suffix)
            except FileNotFoundError as e:
                print(e, file=sys.stderr)
                for dw in delay_values:
                    writer.writerow([hardness, model_type, dw, "", "", "", "", "", "", str(e)])
                    run_idx += 1
                continue

            for dw in delay_values:
                run_idx += 1
                print(f"[{run_idx}/{total}] {hardness} {model_type} delay_scale={dw:.2f} ...", flush=True)
                row = run_one_test(hardness, model_type, checkpoint, dw)
                if "error" in row:
                    writer.writerow([hardness, model_type, dw, "", "", "", "", "", "", row.get("error", "")])
                    print(f"  -> error: {row.get('error')}", flush=True)
                    continue
                writer.writerow([
                    hardness, model_type, dw,
                    row["no_aug_score"], row["no_aug_gap"], row["aug_score"], row["aug_gap"],
                    row["sol_infeasible_pct"], row["ins_infeasible_pct"], ""
                ])
                f.flush()

    print(f"Done. Results: {out_csv}")


if __name__ == "__main__":
    main()
