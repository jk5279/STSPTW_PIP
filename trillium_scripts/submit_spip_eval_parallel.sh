#!/usr/bin/env bash
# Submit ONE eval allocation and run all (SIZE, HARDNESS) pairs in parallel inside that node.
# Run from repo root: bash trillium_scripts/submit_spip_eval_parallel.sh
#
# Options:
#   --sizes "10 50 100"          subset of sizes (default: all three)
#   --hardness "easy medium hard" subset of hardness levels (default: all three)
#   --time "01:00:00"            SLURM walltime for the single allocation (default: 1 hr)
#   --cpus 96                    CPUs for this job (default: 96). Trillium GPU full-node jobs get
#                                 96 cores per node; CPU whole-node jobs use 192 (Alliance/SciNet quickstart).
#
# Example (only n=100, easy/medium):
#   bash trillium_scripts/submit_spip_eval_parallel.sh --sizes "100" --hardness "easy medium"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SIZES=(10 50 100)
HARDNESS=(easy medium hard)
TIME_LIMIT="01:00:00"
CPUS_PER_TASK=96

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sizes)    read -ra SIZES    <<< "$2"; shift 2 ;;
    --hardness) read -ra HARDNESS <<< "$2"; shift 2 ;;
    --time)     TIME_LIMIT="$2"; shift 2 ;;
    --cpus)     CPUS_PER_TASK="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

echo "=========================================="
echo "S-PIP parallel eval submission"
echo "Sizes:    ${SIZES[*]}"
echo "Hardness: ${HARDNESS[*]}"
echo "Time:     ${TIME_LIMIT}"
echo "CPUs:     ${CPUS_PER_TASK}"
echo "=========================================="

sbatch \
  --account="def-cglee" \
  --partition="compute_full_node" \
  --job-name="spip-eval-parallel" \
  --nodes=1 \
  --ntasks=1 \
  --gpus-per-node=4 \
  --cpus-per-task="${CPUS_PER_TASK}" \
  --time="${TIME_LIMIT}" \
  --output="spip_eval_parallel_%j.out" \
  --error="spip_eval_parallel_%j.err" \
  --export=ALL,SIZES="${SIZES[*]}",HARDNESS="${HARDNESS[*]}" \
  "${SCRIPT_DIR}/run_spip_eval_parallel.slurm"

echo "=========================================="
echo "Single allocation submitted. Check with: squeue -u \$USER"
echo "=========================================="
