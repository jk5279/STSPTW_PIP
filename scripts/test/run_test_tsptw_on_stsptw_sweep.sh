#!/bin/bash
# Run full TSPTW-on-STSPTW delay sweep (900 experiments). Single job; inference is fast.
# Usage: sbatch run_test_tsptw_on_stsptw_sweep.sh
#SBATCH --job-name=tsptw_stsptw_sweep
#SBATCH --output=logs/slurm/tsptw_on_stsptw_sweep_%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --partition=compute

set -e
# Use submit dir so we run in project dir when job runs from spool (e.g. compute node)
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

python run_test_tsptw_on_stsptw_sweep_dw.py
echo "Done at $(date)"
