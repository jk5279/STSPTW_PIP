#!/usr/bin/env bash
# S-PIP training for TSPTW_SPIP, n=10, 100M steps (10k epochs x 10k train_episodes), batch 128.
# Deterministic by default; use --stochastic or STOCHASTIC=1 for stochastic transitions.
# Run from repo root: bash trillium_scripts/run_train_spip_n10.sh [--stochastic]
# Or from SLURM: STOCHASTIC is set by the job script (see run_spip_n10.slurm).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PIPO_DIR="${REPO_DIR}/POMO+PIP"

DATA_DIR="${REPO_DIR}/data"
# Hardness level: easy / medium / hard (default: hard)
HARDNESS="${HARDNESS:-hard}"
# Validation file: generate_data with TSPTW_SPIP writes to TSPTW/ (env.problem = "TSPTW")
VAL_DATA_FILE="${DATA_DIR}/TSPTW/tsptw10_${HARDNESS}.pkl"
LOG_BASE="${REPO_DIR}/POMO+PIP/results"
SAVED_MODELS_BASE="${REPO_DIR}/POMO+PIP/saved_models"

# Stochastic mode: env first, then argv overrides (so --stochastic wins over STOCHASTIC=0)
STOCHASTIC="${STOCHASTIC:-0}"
for arg in "$@"; do
  [ "$arg" = "--stochastic" ] && STOCHASTIC=1
done

if [ "$STOCHASTIC" = "1" ]; then
  LOG_SUBDIR="${LOG_BASE}/spip_tsptw10_100M_stochastic"
  SAVED_SUBDIR="${SAVED_MODELS_BASE}/spip_tsptw10_100M_stochastic"
else
  LOG_SUBDIR="${LOG_BASE}/spip_tsptw10_100M"
  SAVED_SUBDIR="${SAVED_MODELS_BASE}/spip_tsptw10_100M"
fi
# Concurrent runs share LOG_SUBDIR; train.py creates a timestamped subdir under --log_dir.

EPOCHS=10000
TRAIN_EPISODES=10000
VAL_EPISODES=10000
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
MODEL_SAVE_INTERVAL=50
VALIDATION_INTERVAL=500
PROBLEM_SIZE=10
# Validation file must have at least val_episodes instances; +10 buffer
GEN_SAMPLES=$((VAL_EPISODES + 10))

[[ ":${PYTHONPATH:-}:" != *":${REPO_DIR}:"* ]] && export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
# Work around tensorboard_logger + protobuf 4+ incompatibility (Descriptors cannot be created directly)
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
cd "$PIPO_DIR"

echo "=========================================="
echo "S-PIP TSPTW n=10 — deterministic=$([ "$STOCHASTIC" = "1" ] && echo "no" || echo "yes")"
echo "=========================================="
echo ">> REPO_DIR=$REPO_DIR"
echo ">> PIPO_DIR=$PIPO_DIR"
echo ">> HARDNESS=$HARDNESS"
echo ">> LOG_SUBDIR=$LOG_SUBDIR"
echo "=========================================="

# Generate validation data (idempotent: skip if file exists)
# Existence check does not verify integrity; if you suspect a truncated pkl, delete the file and re-run.
echo ">> Generating validation data..."
if [ -f "$VAL_DATA_FILE" ]; then
  echo ">> Validation data exists, skipping generation"
else
  python generate_data.py --problem TSPTW_SPIP --problem_size "$PROBLEM_SIZE" --hardness "$HARDNESS" \
    --num_samples "$GEN_SAMPLES" --dir "$DATA_DIR"
fi

# Train
echo ">> Starting S-PIP training..."
TRAIN_OPTS=(
  --problem TSPTW_SPIP
  --problem_size "$PROBLEM_SIZE"
  --hardness "$HARDNESS"
  --epochs "$EPOCHS"
  --train_episodes "$TRAIN_EPISODES"
  --val_episodes "$VAL_EPISODES"
  --train_batch_size "$TRAIN_BATCH_SIZE"
  --model_save_interval "$MODEL_SAVE_INTERVAL"
  --validation_interval "$VALIDATION_INTERVAL"
  --generate_PI_mask
  --log_dir "$LOG_SUBDIR"
)
if [ "$STOCHASTIC" = "1" ]; then
  TRAIN_OPTS+=(--spip_stochastic_transition True)
fi
python train.py "${TRAIN_OPTS[@]}"

# Checkpoint copy (required, error-tolerant): do not let copy failure obscure training outcome
echo ">> Copying checkpoints..."
set +e
LATEST_RUN=$(ls -td "$LOG_SUBDIR"/*/ 2>/dev/null | head -1)
LATEST_RUN="${LATEST_RUN%/}"
if [ -d "$LOG_SUBDIR" ] && [ -n "$LATEST_RUN" ]; then
  mkdir -p "$SAVED_SUBDIR"
  [ -f "${LATEST_RUN}/trained_model_val_best.pt" ] && cp "${LATEST_RUN}/trained_model_val_best.pt" "$SAVED_SUBDIR/"
  [ -f "${LATEST_RUN}/trained_model_best.pt" ] && cp "${LATEST_RUN}/trained_model_best.pt" "$SAVED_SUBDIR/"
  EPOCH_PT=$(ls -t "${LATEST_RUN}"/epoch-*.pt 2>/dev/null | head -1)
  [ -n "$EPOCH_PT" ] && cp "$EPOCH_PT" "$SAVED_SUBDIR/$(basename "$EPOCH_PT")"
  echo ">> Copied checkpoints to $SAVED_SUBDIR"
else
  echo ">> No run dir found, skipping checkpoint copy"
fi
set -e

echo ">> Done. Logs: $LOG_SUBDIR"
