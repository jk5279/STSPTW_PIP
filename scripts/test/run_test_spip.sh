#!/bin/bash
#SBATCH --job-name=test_spip
#SBATCH --output=logs/slurm/test_spip_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --account=rrg-cglee
#
# Evaluate all S-PIP checkpoints (n=10/50/100, easy/medium/hard).
# Submit from repo root: sbatch scripts/test/run_test_spip.sh
# Pass extra args: sbatch scripts/test/run_test_spip.sh --sizes 100 --aug_factor 1

set -e

# Under SLURM the batch script runs from /var/spool/slurm; use submit dir instead.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  POMO_PIP_DIR="${SLURM_SUBMIT_DIR}"
  PROJECT_DIR="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
  POMO_PIP_DIR="${PROJECT_DIR}/POMO+PIP"
fi
ENV_DIR="${ENV_DIR:-${PROJECT_DIR}/venv}"

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

echo "=========================================="
echo "S-PIP eval sweep — Node: $(hostname)  JobId: ${SLURM_JOB_ID:-N/A}"
echo "=========================================="

CSV_PATH="${PROJECT_DIR}/results/spip_eval_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}.csv"

cd "$POMO_PIP_DIR"
python ../scripts/test/run_test_spip.py --results_csv "$CSV_PATH" "$@"

echo "CSV saved to: $CSV_PATH"

echo "=== Done at $(date) ==="
