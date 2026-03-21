#!/usr/bin/env python3
"""
Test POMO* checkpoints with PIP applied at inference time (post-hoc),
comparing against jointly-trained POMO*+PIP.

For each POMO_STAR checkpoint from sweep_v1 and sweep_v2, runs:
  - POMO_STAR (baseline, no PIP)          -- already in test_sweep_v1_v2.csv
  - POMO_STAR + PIP post-hoc              -- new: this script
  - POMO_STAR_PIP (jointly trained)       -- already in test_sweep_v1_v2.csv

Results written to: results/csv/test_pomo_star_posthoc_pip.csv
Run from project root (STSPTWenv), or set POMO_PIP_DIR.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
POMO_PIP_DIR = Path(os.environ.get("POMO_PIP_DIR", PROJECT_ROOT / "POMO+PIP")).resolve()
assert POMO_PIP_DIR.is_dir(), f"POMO+PIP dir not found: {POMO_PIP_DIR}"

RESULTS_DIR = POMO_PIP_DIR / "results"
SWEEP_V1_DIR = RESULTS_DIR / "sweep_v1"
SWEEP_V2_DIR = RESULTS_DIR / "sweep_v2"
SLURM_LOG_DIR = PROJECT_ROOT / "logs" / "slurm"


def parse_sweep_v1_pomo_star() -> list[tuple]:
    """Return (hardness, delay_scale, reveal, checkpoint_path) for all POMO_STAR v1 checkpoints."""
    configs = []
    if not SLURM_LOG_DIR.exists():
        return configs
    logs = sorted(SLURM_LOG_DIR.glob("sweep_v1_*_*.out"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not logs:
        return configs
    job_tasks = {}
    for p in logs:
        m = re.match(r"sweep_v1_(\d+)_(\d+)\.out", p.name)
        if m:
            job, task = m.group(1), int(m.group(2))
            if job not in job_tasks:
                job_tasks[job] = {}
            job_tasks[job][task] = p
    if not job_tasks:
        return configs
    latest_job = max(job_tasks.keys(), key=lambda j: max(p.stat().st_mtime for p in job_tasks[j].values()))
    task_logs = job_tasks[latest_job]
    for task_id in range(54):
        if task_id not in task_logs:
            continue
        text = task_logs[task_id].read_text(errors="ignore")
        m = re.search(r"=== v1\s+model=(\w+)\s+(post|pre)-decision\s+dw=([\d.]+)\s+(\w+)\s+===", text)
        m2 = re.search(r">> Log Path:\s*\.?/?results/sweep_v1/([^\s]+)", text)
        if not m or not m2:
            continue
        model_type, decision, dw, hardness = m.group(1), m.group(2), m.group(3), m.group(4)
        if model_type != "POMO_STAR":
            continue
        dir_name = m2.group(1).strip()
        if not dir_name.endswith("_LM"):
            continue
        ckpt = SWEEP_V1_DIR / dir_name / "epoch-10000.pt"
        if not ckpt.is_file():
            continue
        configs.append((hardness, float(dw), decision == "pre", str(ckpt)))
    return configs


def parse_sweep_v2_pomo_star() -> list[tuple]:
    """Return (hardness, noise_type, cv, checkpoint_path) for all POMO_STAR v2 checkpoints."""
    configs = []
    if not SWEEP_V2_DIR.exists():
        return configs
    for d in SWEEP_V2_DIR.iterdir():
        if not d.is_dir():
            continue
        ckpt = d / "epoch-10000.pt"
        if not ckpt.is_file():
            continue
        # Match POMO_STAR only: ends with _LM but NOT _LM_PIMask_1Step
        m = re.match(r".*_STSPTW_v210_(easy|medium|hard)_(gamma|two_point)_cv([\d.]+)_LM$", d.name)
        if not m:
            continue
        hardness, noise_type, cv = m.group(1), m.group(2), float(m.group(3))
        configs.append((hardness, noise_type, cv, str(ckpt)))
    return configs


def run_test_v1(hardness: str, delay_scale: float, reveal: bool, checkpoint: str) -> dict:
    """Run POMO_STAR checkpoint with PIP applied post-hoc (v1 env)."""
    cmd = [
        sys.executable, "test.py",
        "--problem", "STSPTW",
        "--problem_size", "10",
        "--hardness", hardness,
        "--checkpoint", checkpoint,
        "--delay_scale", f"{delay_scale:.1f}",
        "--no_opt_sol",
        "--aug_factor", "8",
        "--generate_PI_mask",
        "--pip_step", "1",
    ]
    if reveal:
        cmd.append("--reveal_delay_before_action")
    return _run_test(cmd)


def run_test_v2(hardness: str, noise_type: str, cv: float, checkpoint: str) -> dict:
    """Run POMO_STAR checkpoint with PIP applied post-hoc (v2 env)."""
    cmd = [
        sys.executable, "test.py",
        "--problem", "STSPTW_v2",
        "--problem_size", "10",
        "--hardness", hardness,
        "--checkpoint", checkpoint,
        "--noise_type", noise_type,
        "--cv", f"{cv}",
        "--reveal_delay_before_action",
        "--no_opt_sol",
        "--aug_factor", "8",
        "--generate_PI_mask",
        "--pip_step", "1",
    ]
    return _run_test(cmd)


def _run_test(cmd: list) -> dict:
    proc = subprocess.run(cmd, cwd=POMO_PIP_DIR, capture_output=True, text=True, timeout=600)
    out = proc.stdout + "\n" + proc.stderr
    if proc.returncode != 0:
        return {"error": f"exit {proc.returncode}", "stderr": out[-2000:]}
    no_aug_score = no_aug_gap = aug_score = aug_gap = sol_infeasible = ins_infeasible = None
    for line in out.splitlines():
        m = re.search(r"NO-AUG SCORE:\s*([\d.]+|nan),\s*Gap:\s*([\d.]+)", line)
        if m:
            no_aug_score = float(m.group(1)) if m.group(1) != "nan" else float("nan")
            no_aug_gap = float(m.group(2))
            continue
        m = re.search(r"AUGMENTATION SCORE:\s*([\d.]+|nan),\s*Gap:\s*([\d.]+)", line)
        if m:
            aug_score = float(m.group(1)) if m.group(1) != "nan" else float("nan")
            aug_gap = float(m.group(2))
            continue
        m = re.search(r"Solution level Infeasible rate:\s*([\d.]+)%", line)
        if m:
            sol_infeasible = float(m.group(1))
            continue
        m = re.search(r"Instance level Infeasible rate:\s*([\d.]+)%", line)
        if m:
            ins_infeasible = float(m.group(1))
            continue
    if sol_infeasible is None or ins_infeasible is None:
        return {"error": "parse_failed", "stdout_tail": out[-1500:]}
    return {
        "no_aug_score": no_aug_score if no_aug_score is not None else float("nan"),
        "no_aug_gap": no_aug_gap if no_aug_gap is not None else float("nan"),
        "aug_score": aug_score if aug_score is not None else float("nan"),
        "aug_gap": aug_gap if aug_gap is not None else float("nan"),
        "sol_infeasible_pct": sol_infeasible,
        "ins_infeasible_pct": ins_infeasible,
    }


def main():
    import csv
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--v1_only", action="store_true")
    ap.add_argument("--v2_only", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    v1_configs = parse_sweep_v1_pomo_star() if not args.v2_only else []
    v2_configs = parse_sweep_v2_pomo_star() if not args.v1_only else []
    if args.limit:
        v1_configs = v1_configs[:args.limit]
        v2_configs = v2_configs[:args.limit]

    if args.dry_run:
        print(f"sweep_v1 POMO_STAR configs: {len(v1_configs)}")
        for c in v1_configs[:3]:
            print(f"  {c}")
        print(f"sweep_v2 POMO_STAR configs: {len(v2_configs)}")
        for c in v2_configs[:3]:
            print(f"  {c}")
        return

    out_csv = PROJECT_ROOT / "results" / "csv" / "test_pomo_star_posthoc_pip.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    total = len(v1_configs) + len(v2_configs)
    run_idx = 0

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sweep", "hardness", "model_type", "delay_scale", "reveal", "noise_type", "cv",
            "no_aug_score", "no_aug_gap", "aug_score", "aug_gap",
            "sol_infeasible_pct", "ins_infeasible_pct", "error"
        ])

        for h, dw, rev, ckpt in v1_configs:
            run_idx += 1
            print(f"[{run_idx}/{total}] v1 {h} POMO_STAR+PIP(posthoc) dw{dw} {'pre' if rev else 'post'} ...", flush=True)
            row = run_test_v1(h, dw, rev, ckpt)
            if "error" in row:
                writer.writerow(["v1", h, "POMO_STAR_PIP_posthoc", dw, rev, "", "",
                                  "", "", "", "", "", "", row.get("error", "")])
                print(f"  -> error: {row.get('error')}", flush=True)
            else:
                writer.writerow(["v1", h, "POMO_STAR_PIP_posthoc", dw, rev, "", "",
                                  row["no_aug_score"], row["no_aug_gap"],
                                  row["aug_score"], row["aug_gap"],
                                  row["sol_infeasible_pct"], row["ins_infeasible_pct"], ""])
            f.flush()

        for h, noise, cv, ckpt in v2_configs:
            run_idx += 1
            print(f"[{run_idx}/{total}] v2 {h} POMO_STAR+PIP(posthoc) {noise} cv{cv} ...", flush=True)
            row = run_test_v2(h, noise, cv, ckpt)
            if "error" in row:
                writer.writerow(["v2", h, "POMO_STAR_PIP_posthoc", "", "", noise, cv,
                                  "", "", "", "", "", "", row.get("error", "")])
                print(f"  -> error: {row.get('error')}", flush=True)
            else:
                writer.writerow(["v2", h, "POMO_STAR_PIP_posthoc", "", "", noise, cv,
                                  row["no_aug_score"], row["no_aug_gap"],
                                  row["aug_score"], row["aug_gap"],
                                  row["sol_infeasible_pct"], row["ins_infeasible_pct"], ""])
            f.flush()

    print(f"Done. Results: {out_csv}")


if __name__ == "__main__":
    main()
