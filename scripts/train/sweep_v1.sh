#!/bin/bash
# ============================================================
# STSPTW v1 (STSPTWEnv) full training sweep
# Axes: model_type(POMO,POMO_STAR,POMO_STAR_PIP)
#       × decision_mode(post,pre)
#       × delay_scale(0.1,0.3,0.5)
#       × hardness(easy,medium,hard)
# Total: 3 × 2 × 3 × 3 = 54 runs  (array 0-53)
# Index layout (fastest→slowest): hardness, dw, decision, model
# ============================================================
#SBATCH --job-name=sv1
#SBATCH --output=logs/slurm/sweep_v1_%A_%a.out
#SBATCH --array=0-53
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=5:00:00
#SBATCH --partition=compute

set -e

module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cd "$SCRATCH/RL4Research/STSPTWenv/POMO+PIP"

# ---- Parameter grid ----
HARDNESS_LIST=(easy medium hard)     # 3
DW_LIST=(0.1 0.3 0.5)               # 3
DECISION_LIST=(post pre)             # 2
MODEL_LIST=(POMO POMO_STAR POMO_STAR_PIP)  # 3

idx=$SLURM_ARRAY_TASK_ID
h_idx=$(( idx % 3 ))
dw_idx=$(( (idx / 3) % 3 ))
dec_idx=$(( (idx / 9) % 2 ))
m_idx=$(( idx / 18 ))

HARDNESS=${HARDNESS_LIST[$h_idx]}
DW=${DW_LIST[$dw_idx]}
DECISION=${DECISION_LIST[$dec_idx]}
MODEL=${MODEL_LIST[$m_idx]}

PRE_FLAG=""
if [ "$DECISION" == "pre" ]; then
    PRE_FLAG="--reveal_delay_before_action"
fi

echo "============================================"
echo "=== v1  model=${MODEL}  ${DECISION}-decision  dw=${DW}  ${HARDNESS} ==="
echo "=== Array task ${SLURM_ARRAY_TASK_ID} / Job ${SLURM_ARRAY_JOB_ID} ==="
echo "=== Started at $(date) ==="
echo "============================================"

python train.py \
    --problem STSPTW \
    --problem_size 10 --pomo_size 10 \
    --hardness "$HARDNESS" \
    --delay_scale "$DW" \
    $PRE_FLAG \
    --model_type "$MODEL" \
    --epochs 10000 \
    --train_episodes 10000 --train_batch_size 1024 \
    --val_episodes 10000 --validation_batch_size 1000 \
    --validation_interval 500 --model_save_interval 50 \
    --log_dir "./results/sweep_v1"

echo "=== Finished at $(date) ==="
