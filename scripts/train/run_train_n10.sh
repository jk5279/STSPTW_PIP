#!/bin/bash
set -e

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
source /home/ramy/Desktop/Temp/PIP-constraint/venv/bin/activate
cd /home/ramy/Desktop/Temp/STSPTWenv/POMO+PIP

# Train on stochastic TSPTW with pre-decision noise (reveal_delay_before_action)
COMMON_ARGS="--problem STSPTW --problem_size 10 --pomo_size 10 \
  --epochs 10000 --train_episodes 10000 --train_batch_size 1024 \
  --val_episodes 10000 --validation_batch_size 1000 \
  --validation_interval 500 --model_save_interval 1000 \
  --reveal_delay_before_action"

for hardness in easy medium hard; do
  echo "=============================="
  echo "=== POMO (POMO) STSPTW $hardness  $(date) ==="
  echo "=============================="
  CKPT_PATTERN="results/*_STSPTW10_${hardness}_dw0.1/epoch-10000.pt"
  if ls ${CKPT_PATTERN} >/dev/null 2>&1; then
    echo ">> Skip POMO ${hardness} (found ${CKPT_PATTERN})"
  else
    python train.py $COMMON_ARGS --hardness $hardness --model_type POMO
  fi
done

for hardness in easy medium hard; do
  echo "=============================="
  echo "=== POMO* (POMO_STAR) STSPTW $hardness  $(date) ==="
  echo "=============================="
  CKPT_PATTERN="results/*_STSPTW10_${hardness}_dw0.1_LM/epoch-10000.pt"
  if ls ${CKPT_PATTERN} >/dev/null 2>&1; then
    echo ">> Skip POMO* (POMO_STAR) ${hardness} (found ${CKPT_PATTERN})"
  else
    python train.py $COMMON_ARGS --hardness $hardness --model_type POMO_STAR
  fi
done

for hardness in easy medium hard; do
  echo "=============================="
  echo "=== POMO*+PIP (POMO_STAR_PIP) STSPTW $hardness  $(date) ==="
  echo "=============================="
  CKPT_PATTERN="results/*_STSPTW10_${hardness}_dw0.1_LM_PIMask_1Step/epoch-10000.pt"
  if ls ${CKPT_PATTERN} >/dev/null 2>&1; then
    echo ">> Skip POMO*+PIP (POMO_STAR_PIP) ${hardness} (found ${CKPT_PATTERN})"
  else
    python train.py $COMMON_ARGS --hardness $hardness --model_type POMO_STAR_PIP
  fi
done

echo "=============================="
echo "=== ALL DONE  $(date) ==="
echo "=============================="
