#!/bin/bash
# Sanity check: STSPTW_v2 pre-decision, N=50 and N=100.
# Runs 4 short jobs (gamma + two_point × N=50 + N=100) with POMO_STAR_PIP.
# Verifies the full pipeline works before launching the real sweeps.
# Submit: sbatch scripts/train/sanity_check_v2_n50_n100.sh
#SBATCH --job-name=sanity_v2_n50_n100
#SBATCH --output=logs/slurm/sanity_v2_n50_n100_%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:30:00
#SBATCH --partition=debug

set -e

echo "=== Sanity check (v2 N=50 & N=100) started at $(date) ==="

module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cd "$SCRATCH/RL4Research/STSPTW_PIP/POMO+PIP"

COMMON_N50="--problem STSPTW_v2 --problem_size 50 --pomo_size 50 \
  --epochs 3 --train_episodes 100 --train_batch_size 64 \
  --val_episodes 100 --validation_batch_size 64 \
  --validation_interval 1 --model_save_interval 10 \
  --hardness easy --model_type POMO_STAR_PIP \
  --reveal_delay_before_action --cv 0.5 \
  --log_dir ./results_sanity"

COMMON_N100="--problem STSPTW_v2 --problem_size 100 --pomo_size 100 \
  --epochs 3 --train_episodes 100 --train_batch_size 64 \
  --val_episodes 100 --validation_batch_size 64 \
  --validation_interval 1 --model_save_interval 10 \
  --hardness easy --model_type POMO_STAR_PIP \
  --reveal_delay_before_action --cv 0.5 \
  --simulation_stop_epoch 100 --pip_update_epoch 20 \
  --log_dir ./results_sanity"

echo "============================================"
echo "=== [1/4] N=50  gamma  cv=0.5  pre-decision ==="
echo "============================================"
python train.py $COMMON_N50 --noise_type gamma

echo "============================================"
echo "=== [2/4] N=50  two_point  cv=0.5  pre-decision ==="
echo "============================================"
python train.py $COMMON_N50 --noise_type two_point

echo "============================================"
echo "=== [3/4] N=100  gamma  cv=0.5  pre-decision ==="
echo "============================================"
python train.py $COMMON_N100 --noise_type gamma

echo "============================================"
echo "=== [4/4] N=100  two_point  cv=0.5  pre-decision ==="
echo "============================================"
python train.py $COMMON_N100 --noise_type two_point

echo "============================================"
echo "=== ALL 4 SANITY CHECKS PASSED at $(date) ==="
echo "============================================"
echo "Clean up with: rm -rf POMO+PIP/results_sanity/"
