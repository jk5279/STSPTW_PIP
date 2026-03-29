#!/usr/bin/env bash
# S-PIP 10-city tester: run test.py for easy, medium, hard with matching checkpoint and test set.
# Run from repo root: bash trillium_scripts/run_spip_n10_test.sh
# Uses batch size 512 and test_episodes 10000 (original POMO+PIP default).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PIPO_DIR="${REPO_DIR}/POMO+PIP"
RESULTS_BASE="${REPO_DIR}/POMO+PIP/results/spip_tsptw10_100M"
# Prefer val-best, then train-best, then latest epoch checkpoint (some runs only have epoch-*.pt)
CKPT_NAMES="trained_model_val_best.pt trained_model_best.pt"

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
cd "$PIPO_DIR"

# Resolve latest checkpoint per hardness (latest run dir by mtime containing hardness in name)
get_checkpoint() {
  local hardness="$1"
  local dir
  # cd to RESULTS_BASE so glob expands; ls -td gives newest first (disable exit-on-error for no-match)
  dir=$(cd "${RESULTS_BASE}" 2>/dev/null && ls -td *${hardness}* 2>/dev/null | head -1) || true
  if [ -n "${dir:-}" ]; then
    dir="${RESULTS_BASE}/${dir}"
  fi
  if [ -z "${dir:-}" ] || [ ! -d "${dir}" ]; then
    echo ""
    return
  fi
  for ckpt in $CKPT_NAMES; do
    if [ -f "${dir}/${ckpt}" ]; then
      echo "${dir}/${ckpt}"
      return
    fi
  done
  # Fallback: latest epoch-*.pt (e.g. epoch-10000.pt)
  latest_epoch=$(ls -t "${dir}"/epoch-*.pt 2>/dev/null | head -1)
  if [ -n "${latest_epoch}" ]; then
    echo "${latest_epoch}"
  else
    echo ""
  fi
}

RUN_EASY=$(get_checkpoint easy)
RUN_MEDIUM=$(get_checkpoint medium)
RUN_HARD=$(get_checkpoint hard)

echo "=========================================="
echo "S-PIP n=10 test — batch 512, test_episodes 10000"
echo "=========================================="
echo "Checkpoint easy:   ${RUN_EASY:- (not found)}"
echo "Checkpoint medium: ${RUN_MEDIUM:- (not found)}"
echo "Checkpoint hard:   ${RUN_HARD:- (not found)}"
echo "=========================================="

run_test() {
  local hardness="$1"
  local checkpoint="$2"
  if [ -z "$checkpoint" ] || [ ! -f "$checkpoint" ]; then
    echo ">> Skipping $hardness: no checkpoint found."
    return 0
  fi
  echo ">> Testing $hardness on tsptw10_${hardness}.pkl with checkpoint $checkpoint"
  python test.py --problem TSPTW_SPIP --problem_size 10 --hardness "$hardness" \
    --checkpoint "$checkpoint" \
    --test_batch_size 512 --test_episodes 10000 \
    --generate_PI_mask --fsb_dist_only True
}

run_test easy   "$RUN_EASY"
run_test medium "$RUN_MEDIUM"
run_test hard   "$RUN_HARD"

echo ">> All S-PIP n=10 tests finished."
