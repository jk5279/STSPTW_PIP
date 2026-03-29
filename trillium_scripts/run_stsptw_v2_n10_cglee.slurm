#!/bin/bash
#SBATCH --job-name=stsptw-v2-n10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --account=rrg-cglee
#SBATCH --output=stsptw_v2_n10_%j.out
#SBATCH --error=stsptw_v2_n10_%j.err
#
# STSPTW v2 (mean-preserving) training, n=10.
# Submit from repo root:
#   sbatch --export=ALL,HARDNESS=medium trillium_scripts/run_stsptw_v2_n10_cglee.slurm
#
# Key env vars (optional overrides):
#   HARDNESS=easy|medium|hard   (default: hard)
#   NOISE_TYPE=gamma|two_point  (default: gamma)
#   CV=0.5                      (default: 0.5; ignored for two_point)
#   REVEAL_DELAY=1              (default: 1; 1 => --reveal_delay_before_action)
#   TRAIN_BATCH_SIZE=1024       (default: 1024)

set -e

# Under SLURM the script runs from job spool dir; use submit dir so run_train_spip.sh is found
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  PROJECT_DIR="${SLURM_SUBMIT_DIR}"
  SCRIPT_PATH="${SLURM_SUBMIT_DIR}/trillium_scripts"
else
  SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_DIR="$(cd "$SCRIPT_PATH/.." && pwd)"
fi
ENV_DIR="${ENV_DIR:-${PROJECT_DIR}/venv}"

cd "$SCRIPT_PATH"

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
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

export PROBLEM="STSPTW_v2"
export PROBLEM_SIZE=10
export HARDNESS="${HARDNESS:-hard}"
export NOISE_TYPE="${NOISE_TYPE:-gamma}"
export CV="${CV:-0.5}"
export REVEAL_DELAY="${REVEAL_DELAY:-1}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1024}"

echo "=========================================="
echo "STSPTW_v2 n=10 job — Node: $(hostname) JobId: ${SLURM_JOB_ID:-N/A}"
echo "=========================================="
echo "Project: $PROJECT_DIR"
echo "Scripts: $SCRIPT_PATH"
echo "PROBLEM: $PROBLEM"
echo "PROBLEM_SIZE: $PROBLEM_SIZE"
echo "HARDNESS: $HARDNESS"
echo "NOISE_TYPE: $NOISE_TYPE"
echo "CV: $CV"
echo "REVEAL_DELAY: $REVEAL_DELAY"
echo "TRAIN_BATCH_SIZE: $TRAIN_BATCH_SIZE"
echo "=========================================="

bash "$SCRIPT_PATH/run_train_spip.sh"

echo "Job finished."

