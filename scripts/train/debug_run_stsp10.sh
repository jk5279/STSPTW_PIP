#!/bin/bash
set -e

# Lightweight debug run for STSPTW N=10 on Trillium GPU
# Use inside a GPU debugjob session: debugjob -g 1

echo "=== Loading Trillium modules ==="
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

echo "=== Changing directory to STSPTWenv/POMO+PIP ==="
cd "$SCRATCH/RL4Research/STSPTWenv/POMO+PIP"

COMMON_ARGS="--problem STSPTW --problem_size 10 --pomo_size 10 \
  --epochs 1 --train_episodes 10 --train_batch_size 4 \
  --val_episodes 10 --validation_batch_size 10 \
  --validation_interval 1 --model_save_interval 1"

HARDNESS="easy"
DELAY_WEIGHT="0.5"

echo "=== Debug run: POMO (hardness=${HARDNESS}, delay_scale=${DELAY_WEIGHT}) ==="
python train.py $COMMON_ARGS --hardness "$HARDNESS" --delay_scale "$DELAY_WEIGHT"

echo "=== Debug run: POMO* (hardness=${HARDNESS}, delay_scale=${DELAY_WEIGHT}) ==="
python train.py $COMMON_ARGS --hardness "$HARDNESS" --delay_scale "$DELAY_WEIGHT" --pomo_start True

echo "=== Debug run: POMO*+PIP (hardness=${HARDNESS}, delay_scale=${DELAY_WEIGHT}) ==="
python train.py $COMMON_ARGS --hardness "$HARDNESS" --delay_scale "$DELAY_WEIGHT" --pomo_start True --generate_PI_mask

echo "=== DEBUG STSPTW N=10 RUNS COMPLETED ==="

