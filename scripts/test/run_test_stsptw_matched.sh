#!/bin/bash
# Test 90 STSPTW models on their matching env. Single job.
# Usage: sbatch run_test_stsptw_matched.sh
#SBATCH --job-name=stsptw_matched
#SBATCH --output=logs/slurm/stsptw_matched_%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --partition=compute

set -e
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

python run_test_stsptw_matched.py
echo "Done at $(date)"
