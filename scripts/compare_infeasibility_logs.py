#!/usr/bin/env python3
"""
Compare infeasibility rates across S-PIP job logs (easy / medium / hard).
Parses "Infeasible_rate: [sol%, ins%]" and "HARDNESS: ..." from log files.
Usage:
  python scripts/compare_infeasibility_logs.py spip_n10_*.out
  python scripts/compare_infeasibility_logs.py --dir . --pattern "spip_n10*.out"
"""
import argparse
import re
import os
from collections import defaultdict


def parse_log(path):
    """Extract hardness and list of (epoch, sol_rate, ins_rate) from a single log file."""
    hardness = None
    rates = []  # (epoch, sol_pct, ins_pct)
    epoch_re = re.compile(r"Epoch\s+(\d+):")
    rate_re = re.compile(r"Infeasible_rate:\s*\[\s*([\d.]+)\s*%?\s*,\s*([\d.]+)\s*%?\s*\]")
    hard_re = re.compile(r"HARDNESS:\s*(\w+)")
    with open(path, "r") as f:
        for line in f:
            m = hard_re.search(line)
            if m:
                hardness = m.group(1).strip()
            m = epoch_re.search(line)
            if m:
                epoch = int(m.group(1))
            else:
                epoch = None
            m = rate_re.search(line)
            if m:
                sol_pct = float(m.group(1))
                ins_pct = float(m.group(2))
                if epoch is not None:
                    rates.append((epoch, sol_pct, ins_pct))
    return hardness, rates


def infer_hardness_from_filename(path):
    """Infer hardness from log filename, e.g. spip_n10_medium_12345.out -> medium."""
    name = os.path.basename(path)
    if "medium" in name:
        return "medium"
    if "easy" in name:
        return "easy"
    if "hard" in name or "spip_n10_" in name:
        return "hard"
    return None


def main():
    ap = argparse.ArgumentParser(description="Compare infeasibility rates across S-PIP logs")
    ap.add_argument("logs", nargs="*", help="Log files (e.g. spip_n10_*.out)")
    ap.add_argument("--dir", default=None, help="Directory to search for logs (use with --pattern)")
    ap.add_argument("--pattern", default="spip_n10*.out", help="Glob pattern when using --dir")
    ap.add_argument("--epochs", type=str, default="1000,5000", help="Comma-separated epochs to sample (e.g. 1000,5000)")
    args = ap.parse_args()

    if args.dir:
        import glob
        path_glob = os.path.join(args.dir, args.pattern)
        paths = sorted(glob.glob(path_glob))
    else:
        paths = args.logs

    if not paths:
        print("No log files found. Provide paths or use --dir and --pattern.")
        return

    epochs_wanted = [int(x) for x in args.epochs.split(",")]
    by_hardness = defaultdict(list)  # hardness -> list of (path, hardness_parsed, rates)

    for path in paths:
        if not os.path.isfile(path):
            continue
        hardness_parsed, rates = parse_log(path)
        if not rates:
            continue
        hardness = hardness_parsed or infer_hardness_from_filename(path) or "unknown"
        by_hardness[hardness].append((path, hardness_parsed, rates))

    if not by_hardness:
        print("No Infeasible_rate lines found in any log.")
        return

    print("=" * 60)
    print("Infeasibility rate comparison (solution-level %, instance-level %)")
    print("=" * 60)

    for hardness in ["easy", "medium", "hard"]:
        if hardness not in by_hardness:
            continue
        entries = by_hardness[hardness]
        for path, _, rates in entries:
            name = os.path.basename(path)
            if not rates:
                continue
            # Sample near requested epochs (closest epoch for each)
            rates_by_epoch = {r[0]: (r[1], r[2]) for r in rates}
            epochs_avail = sorted(rates_by_epoch.keys())
            print("\n{} (from {}):".format(hardness.upper(), name))
            for target in epochs_wanted:
                closest = min(epochs_avail, key=lambda e: abs(e - target))
                sol_pct, ins_pct = rates_by_epoch[closest]
                print("  Epoch ~{}: sol_infeasible {:.2f}%, ins_infeasible {:.2f}%".format(closest, sol_pct, ins_pct))
            # Latest
            last_epoch = max(epochs_avail)
            sol_pct, ins_pct = rates_by_epoch[last_epoch]
            print("  Epoch {} (latest): sol_infeasible {:.2f}%, ins_infeasible {:.2f}%".format(last_epoch, sol_pct, ins_pct))

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
