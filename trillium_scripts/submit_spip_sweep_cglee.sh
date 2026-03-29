#!/usr/bin/env bash
# Submit all S-PIP sweep jobs: n={50,100} x hardness={easy,medium,hard} x mode={det,stoch}.
# Run from repo root: bash trillium_scripts/submit_spip_sweep_cglee.sh
# To submit only deterministic jobs: bash ... --det-only
# To submit only stochastic jobs:    bash ... --stoch-only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DET_ONLY=0
STOCH_ONLY=0
for arg in "$@"; do
  [ "$arg" = "--det-only"   ] && DET_ONLY=1
  [ "$arg" = "--stoch-only" ] && STOCH_ONLY=1
done

submit() {
  local slurm_file="$1"
  echo "Submitting: $slurm_file"
  sbatch "$SCRIPT_DIR/$slurm_file"
}

echo "=========================================="
echo "S-PIP sweep submission (n=50, n=100)"
echo "DET_ONLY=$DET_ONLY  STOCH_ONLY=$STOCH_ONLY"
echo "=========================================="

# --- n=50 ---
if [ "$STOCH_ONLY" = "0" ]; then
  submit run_spip_n50_easy_cglee.slurm
  submit run_spip_n50_medium_cglee.slurm
  submit run_spip_n50_hard_cglee.slurm
fi
if [ "$DET_ONLY" = "0" ]; then
  submit run_spip_n50_easy_stoch_cglee.slurm
  submit run_spip_n50_medium_stoch_cglee.slurm
  submit run_spip_n50_hard_stoch_cglee.slurm
fi

# --- n=100 ---
if [ "$STOCH_ONLY" = "0" ]; then
  submit run_spip_n100_easy_cglee.slurm
  submit run_spip_n100_medium_cglee.slurm
  submit run_spip_n100_hard_cglee.slurm
fi
if [ "$DET_ONLY" = "0" ]; then
  submit run_spip_n100_easy_stoch_cglee.slurm
  submit run_spip_n100_medium_stoch_cglee.slurm
  submit run_spip_n100_hard_stoch_cglee.slurm
fi

echo "=========================================="
echo "All jobs submitted. Check with: squeue -u \$USER"
echo "=========================================="
