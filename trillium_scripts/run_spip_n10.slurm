#!/bin/bash
#SBATCH --job-name=spip-n10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --account=YOUR_ACCOUNT_HERE
#SBATCH --output=spip_n10_%j.out
#SBATCH --error=spip_n10_%j.err
#
# S-PIP TSPTW n=10, 10k epochs x 10k train_episodes, batch 128.
# Submit from repo root: sbatch trillium_scripts/run_spip_n10.slurm
# For stochastic: sbatch --export=ALL,STOCHASTIC=1 trillium_scripts/run_spip_n10.slurm
# Set #SBATCH --account=... to your cluster account (required on most HPC systems).

set -e

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_PATH/.." && pwd)"
ENV_DIR="${ENV_DIR:-${PROJECT_DIR}/venv}"

cd "$SCRIPT_PATH"

# Intentional suppression: some clusters emit warnings to stderr on module purge even on success
module purge 2>/dev/null || true
module load StdEnv/2023
module load python/3.11.5
# Wrong CUDA version can cause silent issues; run "module avail cuda" once after StdEnv to verify
module load cuda/12.6 2>/dev/null || module load cuda 2>/dev/null || true

if [ -d "$ENV_DIR" ] && [ -f "${ENV_DIR}/bin/activate" ]; then
  source "${ENV_DIR}/bin/activate"
else
  echo "WARNING: No venv at $ENV_DIR; using system Python."
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export STOCHASTIC="${STOCHASTIC:-0}"

echo "=========================================="
echo "S-PIP n=10 job — Node: $(hostname) JobId: ${SLURM_JOB_ID:-N/A}"
echo "=========================================="
echo "Project: $PROJECT_DIR"
echo "Scripts: $SCRIPT_PATH"
echo "STOCHASTIC: $STOCHASTIC"
echo "=========================================="

bash "$SCRIPT_PATH/run_train_spip_n10.sh"

echo "Job finished."
