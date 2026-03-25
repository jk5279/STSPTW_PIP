#!/bin/bash
# Generate separate STSPTW test datasets for N=10, 50, 100.
# STSPTW uses TSPTW base graphs with stochastic delays applied at runtime,
# so test sets are TSPTW instances generated with a different seed (2026)
# from the validation sets (seed 2025).
#
# Output: data/TSPTW/test_tsptw{10,50,100}_{easy,medium,hard}.pkl
#
# Usage: bash scripts/generate_stsptw_test_data.sh
#   (run from STSPTW_PIP/ root, or submit as a short CPU job)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POMO_PIP_DIR="$SCRIPT_DIR/../POMO+PIP"
DATA_DIR="$SCRIPT_DIR/../data"

cd "$POMO_PIP_DIR"

echo "=== Generating STSPTW test datasets (seed=2026) at $(date) ==="

for N in 10 50 100; do
    for hardness in easy medium hard; do
        echo "--- test_tsptw${N}_${hardness}.pkl ---"
        python generate_data.py \
            --problem TSPTW \
            --problem_size "$N" \
            --pomo_size "$N" \
            --hardness "$hardness" \
            --num_samples 10000 \
            --seed 2026 \
            --prefix test_ \
            --dir "$DATA_DIR"
    done
done

echo "=== Done at $(date) ==="
echo "Generated files:"
ls "$DATA_DIR/TSPTW/test_tsptw"*.pkl
