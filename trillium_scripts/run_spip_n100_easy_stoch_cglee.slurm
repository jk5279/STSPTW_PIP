#!/bin/bash
#SBATCH --job-name=spip-n100-easy-stoch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --account=rrg-cglee
#SBATCH --output=spip_n100_easy_stoch_%j.out
#SBATCH --error=spip_n100_easy_stoch_%j.err
#
# S-PIP TSPTW n=100, hardness=easy, stochastic transitions, POMO_STAR_PIP.
# Submit from repo root: sbatch trillium_scripts/run_spip_n100_easy_stoch_cglee.slurm

set -e

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  PROJECT_DIR="${SLURM_SUBMIT_DIR}"
  SCRIPT_PATH="${SLURM_SUBMIT_DIR}/trillium_scripts"
else
  SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_DIR="$(cd "$SCRIPT_PATH/.." && pwd)"
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

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export PROBLEM_SIZE=100
export HARDNESS=easy
export STOCHASTIC=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

echo "=========================================="
echo "S-PIP n=100 EASY stoch job — Node: $(hostname) JobId: ${SLURM_JOB_ID:-N/A}"
echo "=========================================="

bash "$SCRIPT_PATH/run_train_spip.sh"

echo "Job finished."
