#!/bin/bash
#SBATCH --job-name=stsp_n10_90
#SBATCH --output=logs/slurm/stsp_n10_90_%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1          # single H100 on Trillium GPU
#SBATCH --cpus-per-task=24         # 1/4 node (24 cores) with 1 GPU
#SBATCH --time=4:00:00            # adjust if needed
#SBATCH --partition=compute

set -e

echo "=== SLURM job started at $(date) ==="
echo "Submit dir: $SLURM_SUBMIT_DIR"

# Work from submission directory (should be under $SCRATCH)
cd "$SLURM_SUBMIT_DIR"

echo "=== Loading Trillium modules ==="
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

echo "=== Changing directory to STSPTWenv/POMO+PIP ==="
cd "$SCRATCH/RL4Research/STSPTWenv/POMO+PIP"

# Common training args: 10 epochs, 10k instances, batch_size=1024
COMMON_ARGS="--problem STSPTW --problem_size 10 --pomo_size 10 \
  --epochs 10000 --train_episodes 10000 --train_batch_size 1024 \
  --val_episodes 10000 --validation_batch_size 1000 \
  --validation_interval 500 --model_save_interval 50"

DELAY_WEIGHTS="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"

echo "=== Starting 90 training runs (POMO / POMO* / POMO*+PIP via --model_type) ==="

MODEL_TYPES=("POMO" "POMO_STAR" "POMO_STAR_PIP")

for model_type in "${MODEL_TYPES[@]}"; do
  for hardness in easy medium hard; do
    for dw in $DELAY_WEIGHTS; do
      echo "=============================="
      echo "=== STSPTW N=10 $hardness delay_weight=$dw model_type=$model_type $(date) ==="
      echo "=============================="
      srun python train.py $COMMON_ARGS \
        --model_type "$model_type" \
        --hardness "$hardness" \
        --delay_scale "$dw"
    done
  done
done

echo "=============================="
echo "=== ALL STOCHASTIC N=10 DONE $(date) ==="
echo "=============================="

