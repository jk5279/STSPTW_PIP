#!/bin/bash
# Test sweep_v1 (54) + sweep_v2 (54) = 108 models on matching env.
# Usage: cd STSPTWenv && sbatch scripts/test/run_test_sweep_v1_v2.sh
#SBATCH --job-name=test_sweep_v1v2
#SBATCH --output=logs/slurm/test_sweep_v1v2_%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --partition=compute

set -e
cd "$SCRATCH/RL4Research/STSPTWenv"
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

python scripts/test/run_test_sweep_v1_v2.py
echo "Done at $(date)"
