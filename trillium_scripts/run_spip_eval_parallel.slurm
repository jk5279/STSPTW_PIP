#!/bin/bash
#SBATCH --job-name=spip-eval-parallel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=compute_full_node
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=96
#SBATCH --time=01:00:00
#SBATCH --account=def-cglee
#SBATCH --output=spip_eval_parallel_%j.out
#SBATCH --error=spip_eval_parallel_%j.err
#
# Evaluate all requested (SIZE, HARDNESS) pairs in parallel inside one node allocation.
# Required env vars (passed via --export):
#   SIZES    (e.g., "10 50 100")
#   HARDNESS (e.g., "easy medium hard")

set -euo pipefail

SIZES_STR="${SIZES:-10 50 100}"
HARDNESS_STR="${HARDNESS:-easy medium hard}"
read -ra SIZES_ARR <<< "${SIZES_STR}"
read -ra HARDNESS_ARR <<< "${HARDNESS_STR}"

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
# Prevent CPU oversubscription when launching multiple Python processes.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "${POMO_PIP_DIR}/logs/slurm"
mkdir -p "${PROJECT_DIR}/results"

echo "=========================================="
echo "S-PIP parallel eval in one node"
echo "Node: $(hostname)"
echo "JobId: ${SLURM_JOB_ID:-N/A}"
echo "Sizes: ${SIZES_ARR[*]}"
echo "Hardness: ${HARDNESS_ARR[*]}"
echo "=========================================="

cd "$POMO_PIP_DIR"

# One eval per GPU at a time: each task is heavy (esp. n=100 + PIP mask); sharing GPU 0 causes OOM.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -ra GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
else
  GPU_IDS=(0 1 2 3)
fi
NGPUS=${#GPU_IDS[@]}
echo "Parallel eval: at most ${NGPUS} concurrent tasks (GPUs: ${GPU_IDS[*]})"

declare -a TASK_SIZE TASK_H TASK_TAG
for SIZE in "${SIZES_ARR[@]}"; do
  for H in "${HARDNESS_ARR[@]}"; do
    TASK_SIZE+=("$SIZE")
    TASK_H+=("$H")
    TASK_TAG+=("n${SIZE}_${H}")
  done
done

ntasks=${#TASK_TAG[@]}
fail_count=0
idx=0
while (( idx < ntasks )); do
  wave_pids=()
  wave_tags=()
  for ((slot = 0; slot < NGPUS && idx < ntasks; slot++, idx++)); do
    SIZE="${TASK_SIZE[idx]}"
    H="${TASK_H[idx]}"
    TAG="${TASK_TAG[idx]}"
    gid="${GPU_IDS[slot]}"
    CSV_PATH="${PROJECT_DIR}/results/spip_eval_${TAG}_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}.csv"
    LOG_PATH="${POMO_PIP_DIR}/logs/slurm/spip_eval_${TAG}_${SLURM_JOB_ID:-local}.log"
    echo "[launch] ${TAG} on GPU ${gid} -> ${CSV_PATH}"

    (
      export CUDA_VISIBLE_DEVICES="${gid}"
      export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
      python ../scripts/test/run_test_spip.py \
        --sizes "$SIZE" \
        --hardness "$H" \
        --results_csv "$CSV_PATH" \
      >"$LOG_PATH" 2>&1
    ) &
    wave_pids+=("$!")
    wave_tags+=("${TAG}")
  done

  for i in "${!wave_pids[@]}"; do
    if wait "${wave_pids[i]}"; then
      echo "[ok]   ${wave_tags[i]}"
    else
      echo "[fail] ${wave_tags[i]}"
      fail_count=$((fail_count + 1))
    fi
  done
done

if [ "$fail_count" -gt 0 ]; then
  echo "Completed with failures: ${fail_count}"
  exit 1
fi

echo "All parallel eval tasks completed successfully."
