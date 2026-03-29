#!/bin/bash
#SBATCH --job-name=spip-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --account=rrg-cglee
#SBATCH --output=spip_eval_%j.out
#SBATCH --error=spip_eval_%j.err
#
# Evaluate one (SIZE, HARDNESS) pair. Called by submit_spip_eval_parallel.sh.
# Required env vars (passed via --export):  SIZE (10|50|100)  HARDNESS (easy|medium|hard)
# To submit manually:
#   sbatch --export=ALL,SIZE=100,HARDNESS=hard trillium_scripts/run_spip_eval_single.slurm

set -e

SIZE="${SIZE:?SIZE must be set (10|50|100)}"
HARDNESS="${HARDNESS:?HARDNESS must be set (easy|medium|hard)}"

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  PROJECT_DIR="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
ENV_DIR="${ENV_DIR:-${PROJECT_DIR}/venv}"
POMO_PIP_DIR="${PROJECT_DIR}/POMO+PIP"

module purge 2>/dev/null || true
module load StdEnv/2023
module load python/3.11.5
module load cuda/12.6 2>/dev/null || module load cuda 2>/dev/null || true

if [ -d "$ENV_DIR" ] && [ -f "${ENV_DIR}/bin/activate" ]; then
  source "${ENV_DIR}/bin/activate"
else
  echo "WARNING: No venv at $ENV_DIR; using system Python."
fi

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

mkdir -p "${POMO_PIP_DIR}/logs/slurm"

CSV_PATH="${PROJECT_DIR}/results/spip_eval_n${SIZE}_${HARDNESS}_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}.csv"

echo "=========================================="
echo "S-PIP eval — N=${SIZE} ${HARDNESS} — Node: $(hostname)  JobId: ${SLURM_JOB_ID:-N/A}"
echo "=========================================="

cd "$POMO_PIP_DIR"
python ../scripts/test/run_test_spip.py \
  --sizes "$SIZE" \
  --hardness "$HARDNESS" \
  --results_csv "$CSV_PATH"

echo "CSV saved to: $CSV_PATH"
echo "=== Done at $(date) ==="
