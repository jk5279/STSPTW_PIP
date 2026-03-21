#!/bin/bash
# STSPTW n=10 full sweep (pre-decision noise): easy/medium/hard × POMO, POMO_STAR, POMO_STAR_PIP × 10 delay weights = 90 runs.
# Uses --reveal_delay_before_action so travel times are sampled before action selection.
# SLURM array throttled to 8 concurrent tasks.
# Submit: sbatch run_train_stochastic_n10_array_reveal.sh
#SBATCH --job-name=stsp_n10_reveal
#SBATCH --output=logs/slurm/stsp_n10_reveal_%A_%a.out
#SBATCH --array=0-89%8
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=04:00:00
#SBATCH --partition=compute

set -e

echo "=== SLURM array job (reveal_delay_before_action) started: A=${SLURM_ARRAY_JOB_ID}, a=${SLURM_ARRAY_TASK_ID} at $(date) ==="

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
MODES=("POMO" "POMO_STAR" "POMO_STAR_PIP")
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

echo "=== TASK $TASK_ID: mode=$MODE, hardness=$HARDNESS, delay_weight=$DW (reveal_delay_before_action) ==="

srun python train.py $COMMON_ARGS \
  --reveal_delay_before_action \
  --model_type "$MODE" \
  --hardness "$HARDNESS" \
  --delay_scale "$DW"

echo "=== TASK $TASK_ID FINISHED at $(date) ==="
