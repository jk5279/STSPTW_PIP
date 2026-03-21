#!/bin/bash
# Sanity check: 10 epochs each for STSPTWEnv (v1) and STSPTWEnv_v2.
# Verifies the full pipeline works before launching the real sweep.
# Submit from GPU login node: sbatch scripts/train/sanity_check_v1_v2.sh
#SBATCH --job-name=sanity_v1v2
#SBATCH --output=logs/slurm/sanity_v1v2_%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:30:00
#SBATCH --partition=debug

set -e

echo "=== Sanity check started at $(date) ==="

module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cd "$SCRATCH/RL4Research/STSPTWenv/POMO+PIP"

COMMON="--problem_size 10 --pomo_size 10 \
  --epochs 10 --train_episodes 100 --train_batch_size 32 \
  --val_episodes 100 --validation_batch_size 32 \
  --validation_interval 5 --model_save_interval 10 \
  --hardness easy --model_type POMO_STAR_PIP \
  --log_dir ./results_sanity"

echo "============================================"
echo "=== [1/6] STSPTW v1  post-decision  dw=0.3 ==="
echo "============================================"
python train.py $COMMON --problem STSPTW --delay_scale 0.3

echo "============================================"
echo "=== [2/6] STSPTW v1  pre-decision   dw=0.3 ==="
echo "============================================"
python train.py $COMMON --problem STSPTW --delay_scale 0.3 --reveal_delay_before_action

echo "============================================"
echo "=== [3/6] STSPTW_v2  gamma  cv=0.5  post   ==="
echo "============================================"
python train.py $COMMON --problem STSPTW_v2 --noise_type gamma --cv 0.5

echo "============================================"
echo "=== [4/6] STSPTW_v2  gamma  cv=0.5  pre    ==="
echo "============================================"
python train.py $COMMON --problem STSPTW_v2 --noise_type gamma --cv 0.5 --reveal_delay_before_action

echo "============================================"
echo "=== [5/6] STSPTW_v2  two_point cv=0.5 post ==="
echo "============================================"
python train.py $COMMON --problem STSPTW_v2 --noise_type two_point --cv 0.5

echo "============================================"
echo "=== [6/6] STSPTW_v2  two_point cv=0.5 pre  ==="
echo "============================================"
python train.py $COMMON --problem STSPTW_v2 --noise_type two_point --cv 0.5 --reveal_delay_before_action

echo "============================================"
echo "=== ALL 6 SANITY CHECKS PASSED at $(date) ==="
echo "============================================"
echo "Clean up with: rm -rf POMO+PIP/results_sanity/"
