#!/usr/bin/env bash
# Run the test pipeline on medium (and optionally easy/hard) TSPTW instances to report
# baseline feasibility. Use this to check whether medium instances are inherently
# infeasible or the policy is to blame (plan section 5).
#
# Usage (from repo root):
#   bash scripts/run_baseline_feasibility_medium.sh
#   bash scripts/run_baseline_feasibility_medium.sh  [path/to/checkpoint.pt]
#
# If checkpoint is omitted, test.py uses its default (may not exist). For a quick
# feasibility check, use a trained S-PIP checkpoint or any TSPTW checkpoint.
# Output: Solution-level and instance-level infeasible rate for each hardness.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPO_DIR="${REPO_DIR}/src"
DATA_DIR="${REPO_DIR}/data"
CHECKPOINT="${1:-}"

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cd "$PIPO_DIR"

# Default test set: 500 episodes per hardness for a quick comparison
TEST_EPISODES="${TEST_EPISODES:-500}"

run_one() {
  local hardness=$1
  echo "=========================================="
  echo "Baseline feasibility: hardness=$hardness (test_episodes=$TEST_EPISODES)"
  echo "=========================================="
  local test_set="${DATA_DIR}/TSPTW/tsptw10_${hardness}.pkl"
  if [[ ! -f "$test_set" ]]; then
    echo ">> Data file not found: $test_set (generate with: python generate_data.py --problem TSPTW_SPIP --problem_size 10 --hardness $hardness --num_samples $((TEST_EPISODES+10)) --dir $DATA_DIR)"
    return 1
  fi
  local extra=""
  [[ -n "$CHECKPOINT" ]] && extra="--checkpoint $CHECKPOINT"
  python test.py --problem TSPTW_SPIP --hardness "$hardness" --problem_size 10 \
    --test_set_path "$test_set" \
    --test_episodes "$TEST_EPISODES" \
    --generate_PI_mask \
    $extra 2>&1 | tee "/tmp/baseline_feasibility_${hardness}.log" || true
  echo ""
}

for h in medium easy hard; do
  run_one "$h" || true
done

echo "=========================================="
echo "Summary: check above for 'Solution level Infeasible rate' and 'Instance level Infeasible rate' per hardness."
echo "If medium shows much higher rates than easy, compare with LKH/exact solver if available (see plan section 5)."
echo "=========================================="
