#!/usr/bin/env bash
# Generic training script for TSPTW_SPIP / STSPTW / STSPTW_v2, parameterized by env vars.
# Controlled by:
#   PROBLEM          (default: TSPTW_SPIP)  TSPTW_SPIP | STSPTW | STSPTW_v2
#   PROBLEM_SIZE     (required, e.g. 50 or 100)
#   HARDNESS         (default: hard)
#   TRAIN_BATCH_SIZE (default: 256; override with env)
#
#   TSPTW_SPIP only:
#     STOCHASTIC     (default: 0; set to 1 for stochastic transitions)
#
#   STSPTW only:
#     DELAY_SCALE    (default: 0.3)
#
#   STSPTW_v2 only:
#     NOISE_TYPE     (default: gamma)  gamma | two_point
#     CV             (default: 0.5)
#     REVEAL_DELAY   (default: 0; set to 1 for --reveal_delay_before_action)
#
# Run from repo root: PROBLEM=STSPTW PROBLEM_SIZE=50 HARDNESS=easy bash trillium_scripts/run_train_spip.sh
# Or invoked by a slurm file via sbatch.
#
# Resume (optional — Slurm scripts unchanged; set env before sbatch or export in your shell):
#   RESUME_CHECKPOINT=/abs/path/to/.../epoch-3134.pt   # required to resume
#   RESUME_PATH=/abs/path/to/.../20260321_..._LM_PIMask_1Step   # log dir; default: dirname(RESUME_CHECKPOINT)
# Or fully automatic:
#   AUTO_RESUME=1   # picks newest run dir under LOG_SUBDIR and highest-numbered epoch-*.pt
# Optional:
#   NO_OPT_SOL=1      # append --no_opt_sol (match test.py-style eval without gap)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PIPO_DIR="${REPO_DIR}/POMO+PIP"
DATA_DIR="${REPO_DIR}/data"

PROBLEM="${PROBLEM:-TSPTW_SPIP}"
PROBLEM_SIZE="${PROBLEM_SIZE:?PROBLEM_SIZE must be set}"
HARDNESS="${HARDNESS:-hard}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"

POMO_SIZE="$PROBLEM_SIZE"
# All three problems use TSPTW-format validation data
VAL_DATA_FILE="${DATA_DIR}/TSPTW/tsptw${PROBLEM_SIZE}_${HARDNESS}_val.pkl"
VAL_SEED="${VAL_SEED:-2027}"

EPOCHS=10000
TRAIN_EPISODES=10000
VAL_EPISODES=10000
MODEL_SAVE_INTERVAL=50
VALIDATION_INTERVAL=500
GEN_SAMPLES=$((VAL_EPISODES + 10))

# Build a path-safe run label that encodes all problem-specific hyperparams
case "$PROBLEM" in
  TSPTW_SPIP)
    STOCHASTIC="${STOCHASTIC:-0}"
    if [ "$STOCHASTIC" = "1" ]; then
      RUN_LABEL="tsptw_spip${PROBLEM_SIZE}_${HARDNESS}_stochastic"
    else
      RUN_LABEL="tsptw_spip${PROBLEM_SIZE}_${HARDNESS}"
    fi
    ;;
  STSPTW)
    DELAY_SCALE="${DELAY_SCALE:-0.3}"
    RUN_LABEL="stsptw${PROBLEM_SIZE}_${HARDNESS}_dw${DELAY_SCALE}"
    ;;
  STSPTW_v2)
    NOISE_TYPE="${NOISE_TYPE:-gamma}"
    CV="${CV:-0.5}"
    REVEAL_DELAY="${REVEAL_DELAY:-0}"
    RUN_LABEL="stsptw_v2${PROBLEM_SIZE}_${HARDNESS}_${NOISE_TYPE}_cv${CV}"
    [ "$REVEAL_DELAY" = "1" ] && RUN_LABEL="${RUN_LABEL}_pre"
    ;;
  *)
    echo "ERROR: unknown PROBLEM=$PROBLEM" >&2; exit 1 ;;
esac

LOG_SUBDIR="${REPO_DIR}/POMO+PIP/results/${RUN_LABEL}"
SAVED_SUBDIR="${REPO_DIR}/POMO+PIP/saved_models/${RUN_LABEL}"

[[ ":${PYTHONPATH:-}:" != *":${REPO_DIR}:"* ]] && export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
cd "$PIPO_DIR"

echo "=========================================="
echo "Training: PROBLEM=${PROBLEM} n=${PROBLEM_SIZE} hardness=${HARDNESS}"
echo "batch=${TRAIN_BATCH_SIZE} pomo=${POMO_SIZE}"
echo "RUN_LABEL=${RUN_LABEL}"
echo ">> REPO_DIR=$REPO_DIR"
echo ">> LOG_SUBDIR=$LOG_SUBDIR"
echo "=========================================="

# Generate validation data (idempotent: skip if file exists)
echo ">> Checking validation data..."
if [ -f "$VAL_DATA_FILE" ]; then
  echo ">> Validation data exists, skipping generation"
else
  echo ">> Generating validation data..."
  # STSPTW/STSPTW_v2 use TSPTW-format data; TSPTW_SPIP has its own generator
  GEN_PROBLEM="$PROBLEM"
  [ "$PROBLEM" = "STSPTW" ] || [ "$PROBLEM" = "STSPTW_v2" ] && GEN_PROBLEM="TSPTW"
  python generate_data.py --problem "$GEN_PROBLEM" --problem_size "$PROBLEM_SIZE" \
    --hardness "$HARDNESS" --num_samples "$GEN_SAMPLES" --seed "$VAL_SEED" --suffix "_val" --dir "$DATA_DIR"
fi

# Build training options
TRAIN_OPTS=(
  --problem "$PROBLEM"
  --problem_size "$PROBLEM_SIZE"
  --pomo_size "$POMO_SIZE"
  --hardness "$HARDNESS"
  --model_type POMO_STAR_PIP
  --epochs "$EPOCHS"
  --train_episodes "$TRAIN_EPISODES"
  --val_episodes "$VAL_EPISODES"
  --train_batch_size "$TRAIN_BATCH_SIZE"
  --model_save_interval "$MODEL_SAVE_INTERVAL"
  --validation_interval "$VALIDATION_INTERVAL"
  --log_dir "$LOG_SUBDIR"
  --val_dataset "tsptw${PROBLEM_SIZE}_${HARDNESS}_val.pkl"
)

case "$PROBLEM" in
  TSPTW_SPIP)
    [ "$STOCHASTIC" = "1" ] && TRAIN_OPTS+=(--spip_stochastic_transition True)
    ;;
  STSPTW)
    TRAIN_OPTS+=(--delay_scale "$DELAY_SCALE")
    ;;
  STSPTW_v2)
    TRAIN_OPTS+=(--noise_type "$NOISE_TYPE" --cv "$CV")
    [ "$REVEAL_DELAY" = "1" ] && TRAIN_OPTS+=(--reveal_delay_before_action)
    ;;
esac

# --- Optional resume (same training hyperparams; continues in existing log dir) ---
if [ "${AUTO_RESUME:-0}" = "1" ]; then
  LATEST_RUN=""
  if [ -d "$LOG_SUBDIR" ]; then
    # Newest run directory by modification time (same idea as checkpoint copy at end of script)
    LATEST_RUN=$(ls -td "$LOG_SUBDIR"/*/ 2>/dev/null | head -1)
    LATEST_RUN="${LATEST_RUN%/}"
  fi
  if [ -z "$LATEST_RUN" ] || [ ! -d "$LATEST_RUN" ]; then
    echo "ERROR: AUTO_RESUME=1 but no run directory found under $LOG_SUBDIR" >&2
    exit 1
  fi
  RESUME_PATH="$LATEST_RUN"
  best_n=-1
  RESUME_CHECKPOINT=""
  for f in "$RESUME_PATH"/epoch-*.pt; do
    [ -f "$f" ] || continue
    base="${f##*/}"
    n="${base#epoch-}"
    n="${n%.pt}"
    if [ "$n" -gt "$best_n" ] 2>/dev/null; then
      best_n="$n"
      RESUME_CHECKPOINT="$f"
    fi
  done
  if [ -z "$RESUME_CHECKPOINT" ] || [ ! -f "$RESUME_CHECKPOINT" ]; then
    echo "ERROR: AUTO_RESUME=1 but no epoch-*.pt under $RESUME_PATH" >&2
    exit 1
  fi
fi

if [ -n "${RESUME_CHECKPOINT:-}" ]; then
  if [ ! -f "$RESUME_CHECKPOINT" ]; then
    echo "ERROR: RESUME_CHECKPOINT does not exist: $RESUME_CHECKPOINT" >&2
    exit 1
  fi
  if [ -z "${RESUME_PATH:-}" ]; then
    RESUME_PATH="$(cd "$(dirname "$RESUME_CHECKPOINT")" && pwd)"
  fi
  TRAIN_OPTS+=(--resume_path "$RESUME_PATH" --checkpoint "$RESUME_CHECKPOINT")
  echo ">> Resuming: checkpoint=$RESUME_CHECKPOINT"
  echo ">> Resuming: log dir=$RESUME_PATH"
fi

if [ "${NO_OPT_SOL:-0}" = "1" ]; then
  TRAIN_OPTS+=(--no_opt_sol)
  echo ">> Validation: --no_opt_sol (no gap vs LKH)"
fi

echo ">> Starting training..."
python train.py "${TRAIN_OPTS[@]}"

# Checkpoint copy (error-tolerant: do not let copy failure obscure training outcome)
echo ">> Copying checkpoints..."
set +e
LATEST_RUN=$(ls -td "$LOG_SUBDIR"/*/ 2>/dev/null | head -1)
LATEST_RUN="${LATEST_RUN%/}"
if [ -d "$LOG_SUBDIR" ] && [ -n "${LATEST_RUN:-}" ]; then
  mkdir -p "$SAVED_SUBDIR"
  [ -f "${LATEST_RUN}/trained_model_val_best.pt" ] && cp "${LATEST_RUN}/trained_model_val_best.pt" "$SAVED_SUBDIR/"
  [ -f "${LATEST_RUN}/trained_model_best.pt" ]     && cp "${LATEST_RUN}/trained_model_best.pt"     "$SAVED_SUBDIR/"
  EPOCH_PT=$(ls -t "${LATEST_RUN}"/epoch-*.pt 2>/dev/null | head -1)
  [ -n "${EPOCH_PT:-}" ] && cp "$EPOCH_PT" "$SAVED_SUBDIR/$(basename "$EPOCH_PT")"
  echo ">> Copied checkpoints to $SAVED_SUBDIR"
else
  echo ">> No run dir found, skipping checkpoint copy"
fi
set -e

echo ">> Done. Logs: $LOG_SUBDIR"
