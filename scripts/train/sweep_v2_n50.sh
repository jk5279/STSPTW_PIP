#!/bin/bash
# ============================================================
# STSPTW_v2 N=50 full training sweep (pre-decision noise only)
# Axes: model_type(POMO,POMO_STAR,POMO_STAR_PIP)
#       × noise_type(gamma,two_point)
#       × cv(0.25,0.5,1.0)
#       × hardness(easy,medium,hard)
# Total: 3 × 2 × 3 × 3 = 54 runs  (array 0-53)
# Index layout (fastest→slowest): hardness, cv, noise, model
# Submit: sbatch scripts/train/sweep_v2_n50.sh
# ============================================================
#SBATCH --job-name=sv2_n50
#SBATCH --output=logs/slurm/sweep_v2_n50_%A_%a.out
#SBATCH --array=0-53%8
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --partition=compute

set -e

echo "=== SLURM array job (v2 n50) started: A=${SLURM_ARRAY_JOB_ID}, a=${SLURM_ARRAY_TASK_ID} at $(date) ==="

module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cd "$SCRATCH/RL4Research/STSPTW_PIP/POMO+PIP"

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
echo "=== v2 n50  model=${MODEL}  pre-decision  ${NOISE}  cv=${CV}  ${HARDNESS} ==="
echo "=== Array task ${SLURM_ARRAY_TASK_ID} / Job ${SLURM_ARRAY_JOB_ID} ==="
echo "=== Started at $(date) ==="
echo "============================================"

srun python train.py \
    --problem STSPTW_v2 \
    --problem_size 50 --pomo_size 50 \
    --hardness "$HARDNESS" \
    --noise_type "$NOISE" \
    --cv "$CV" \
    --reveal_delay_before_action \
    --model_type "$MODEL" \
    --epochs 10000 \
    --train_episodes 10000 --train_batch_size 1024 \
    --val_episodes 10000 --validation_batch_size 1000 \
    --validation_interval 500 --model_save_interval 50 \
    --log_dir "./results/sweep_v2_n50"

echo "=== Finished at $(date) ==="
