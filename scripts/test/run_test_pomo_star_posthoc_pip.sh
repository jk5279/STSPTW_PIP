#!/bin/bash
# Test POMO* checkpoints with PIP applied post-hoc at inference time.
# Compares: POMO* (no PIP) vs POMO*+PIP(post-hoc) vs POMO*+PIP(jointly trained)
# Usage: cd STSPTWenv && sbatch scripts/test/run_test_pomo_star_posthoc_pip.sh
#SBATCH --job-name=posthoc_pip
#SBATCH --output=logs/slurm/posthoc_pip_%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --partition=compute

set -e
cd "$SCRATCH/RL4Research/STSPTWenv"
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

python scripts/test/run_test_pomo_star_posthoc_pip.py
echo "Done at $(date)"
