#!/bin/bash
#SBATCH --job-name=stsp_n10_arr
#SBATCH --output=logs/slurm/stsp_n10_arr_%A_%a.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1          # single H100 on Trillium GPU
#SBATCH --cpus-per-task=24         # 1/4 node with 1 GPU
#SBATCH --time=04:00:00            # per-task walltime; adjust if needed
#SBATCH --partition=compute

set -e

echo "=== SLURM array job started: A=${SLURM_ARRAY_JOB_ID}, a=${SLURM_ARRAY_TASK_ID} at $(date) ==="

module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cd "$SCRATCH/RL4Research/STSPTWenv/POMO+PIP"

COMMON_ARGS="--problem STSPTW --problem_size 10 --pomo_size 10 \
  --epochs 10000 --train_episodes 10000 --train_batch_size 1024 \
  --val_episodes 10000 --validation_batch_size 1000 \
  --validation_interval 500 --model_save_interval 50"

# Index mapping: 3 model_types × 3 hardness × 10 delay weights = 90 tasks (0..89)
# Using train.py --model_type presets: POMO / POMO_STAR / POMO_STAR_PIP
MODES=("POMO" "POMO_STAR" "POMO_STAR_PIP")          # 0: POMO, 1: POMO*, 2: POMO*+PIP
HARDNESSES=("easy" "medium" "hard")
DELAY_WEIGHTS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

TASK_ID=${SLURM_ARRAY_TASK_ID}

mode_idx=$(( TASK_ID / (3 * 10) ))
rem=$(( TASK_ID % (3 * 10) ))
hard_idx=$(( rem / 10 ))
dw_idx=$(( rem % 10 ))

MODE=${MODES[$mode_idx]}
HARDNESS=${HARDNESSES[$hard_idx]}
DW=${DELAY_WEIGHTS[$dw_idx]}

echo "=== TASK $TASK_ID: mode=$MODE, hardness=$HARDNESS, delay_weight=$DW ==="

srun python train.py $COMMON_ARGS \
  --model_type "$MODE" \
  --hardness "$HARDNESS" \
  --delay_scale "$DW"

echo "=== TASK $TASK_ID FINISHED at $(date) ==="

