#!/bin/bash
# Setup script for STSPTW_PIP on a compute cluster.
# Loads modules (aligned with parent STSPTW slurm jobs), creates a venv, and installs dependencies.
# Run from repo root: bash trillium_scripts/setup_venv.sh
# Optional: set VENV_DIR to a custom venv path (default: repo root venv).

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${REPO_DIR}/venv}"

echo "=========================================="
echo "STSPTW_PIP environment setup"
echo "=========================================="
echo "Repo:  $REPO_DIR"
echo "Venv:  $VENV_DIR"
echo "=========================================="

# 1. Load cluster modules (same as parent STSPTW slurm scripts)
module purge 2>/dev/null || true
module load StdEnv/2023
module load python/3.11.5
module load cuda/12.6 2>/dev/null || module load cuda 2>/dev/null || true

echo "Loaded modules:"
module list 2>/dev/null || true

# 2. Create venv if it does not exist
if [ -d "$VENV_DIR" ] && [ -f "${VENV_DIR}/bin/activate" ]; then
  echo "Venv already exists at $VENV_DIR"
else
  echo "Creating venv at $VENV_DIR ..."
  python -m venv "$VENV_DIR"
fi
source "${VENV_DIR}/bin/activate"

# 3. Upgrade pip and install dependencies
pip install --upgrade pip

# PyTorch (use CUDA build when available; adjust index-url for your cluster CUDA version)
# For CUDA 12.x:
pip install torch torchvision torchaudio

# Core deps (README + S-PIP plan: scipy for z_factor)
pip install "scipy>=1.7"
pip install matplotlib tqdm pytz scikit-learn pandas wandb
pip install tensorboard_logger
pip install ortools

echo "=========================================="
echo "Installation complete. Activate with:"
echo "  source ${VENV_DIR}/bin/activate"
echo "Then run from POMO+PIP: cd POMO+PIP && python train.py --problem TSPTW_SPIP ..."
echo "=========================================="
