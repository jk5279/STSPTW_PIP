#!/bin/bash
# ============================================================
# STSPTW v2 (STSPTWEnv_v2) full training sweep
# Pre-decision noise only
# Axes: model_type(POMO,POMO_STAR,POMO_STAR_PIP)
#       × noise_type(gamma,two_point)
#       × cv(0.25,0.5,1.0)
#       × hardness(easy,medium,hard)
# Total: 3 × 2 × 3 × 3 = 54 runs  (array 0-53)
# Index layout (fastest→slowest): hardness, cv, noise, model
# ============================================================
#SBATCH --job-name=sv2
#SBATCH --output=logs/slurm/sweep_v2_%A_%a.out
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
HARDNESS_LIST=(easy medium hard)           # 3
CV_LIST=(0.25 0.5 1.0)                    # 3
NOISE_LIST=(gamma two_point)              # 2
MODEL_LIST=(POMO POMO_STAR POMO_STAR_PIP) # 3

idx=$SLURM_ARRAY_TASK_ID
h_idx=$(( idx % 3 ))
cv_idx=$(( (idx / 3) % 3 ))
n_idx=$(( (idx / 9) % 2 ))
m_idx=$(( idx / 18 ))

HARDNESS=${HARDNESS_LIST[$h_idx]}
CV=${CV_LIST[$cv_idx]}
NOISE=${NOISE_LIST[$n_idx]}
MODEL=${MODEL_LIST[$m_idx]}

echo "============================================"
echo "=== v2  model=${MODEL}  pre-decision  ${NOISE}  cv=${CV}  ${HARDNESS} ==="
echo "=== Array task ${SLURM_ARRAY_TASK_ID} / Job ${SLURM_ARRAY_JOB_ID} ==="
echo "=== Started at $(date) ==="
echo "============================================"

python train.py \
    --problem STSPTW_v2 \
    --problem_size 10 --pomo_size 10 \
    --hardness "$HARDNESS" \
    --noise_type "$NOISE" \
    --cv "$CV" \
    --reveal_delay_before_action \
    --model_type "$MODEL" \
    --epochs 10000 \
    --train_episodes 10000 --train_batch_size 1024 \
    --val_episodes 10000 --validation_batch_size 1000 \
    --validation_interval 500 --model_save_interval 50 \
    --log_dir "./results/sweep_v2"

echo "=== Finished at $(date) ==="
