#!/bin/bash
# Submit: sbatch scripts/test/run_test_v2.sh
#SBATCH --job-name=test_v2
#SBATCH --output=logs/slurm/test_v2_%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --partition=compute

set -e
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cd "$SCRATCH/RL4Research/STSPTW_PIP/POMO+PIP"
python ../scripts/test/run_test_v2.py
echo "=== Done at $(date) ==="
