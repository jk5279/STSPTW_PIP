#!/bin/bash
# TSPTW n=10 full sweep (paired with STSPTW pre-decision experiments):
# easy/medium/hard × POMO, POMO_STAR, POMO_STAR_PIP (9 runs total).
# Passes --reveal_delay_before_action for interface consistency with STSPTW runs.
# (TSPTWEnv has no delay, so the flag has no effect, but keeps the interface uniform.)
#
# Submit: sbatch run_train_tsptw_n10_batch_reveal.sh
#SBATCH --job-name=tsptw_n10_all_reveal
#SBATCH --output=logs/slurm/tsptw_n10_all_reveal_%A_%a.out
#SBATCH --array=0-8
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=04:00:00
#SBATCH --partition=compute

set -e

echo "=== SLURM array task started: A=${SLURM_ARRAY_JOB_ID}, a=${SLURM_ARRAY_TASK_ID} at $(date) ==="

module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cd "$SCRATCH/RL4Research/STSPTWenv/POMO+PIP"

COMMON_ARGS="--problem TSPTW --problem_size 10 --pomo_size 10 \
  --epochs 10000 --train_episodes 10000 --train_batch_size 1024 \
  --val_episodes 10000 --validation_batch_size 1000 \
  --validation_interval 500 --model_save_interval 1000"

# Task 0-2: easy (POMO, POMO_STAR, POMO_STAR_PIP)
# Task 3-5: medium
# Task 6-8: hard
HARDNESSES=("easy" "easy" "easy" "medium" "medium" "medium" "hard" "hard" "hard")
MODES=("POMO" "POMO_STAR" "POMO_STAR_PIP" "POMO" "POMO_STAR" "POMO_STAR_PIP" "POMO" "POMO_STAR" "POMO_STAR_PIP")

# Checkpoint patterns (match existing run script)
# POMO: *_TSPTW10_${hardness}/epoch-10000.pt
# POMO_STAR: *_TSPTW10_${hardness}_LM/epoch-10000.pt
# POMO_STAR_PIP: *_TSPTW10_${hardness}_LM_PIMask_1Step/epoch-10000.pt
get_ckpt_pattern() {
  local h="$1"
  local m="$2"
  case "$m" in
    POMO)            echo "results/*_TSPTW10_${h}/epoch-10000.pt" ;;
    POMO_STAR)       echo "results/*_TSPTW10_${h}_LM/epoch-10000.pt" ;;
    POMO_STAR_PIP)   echo "results/*_TSPTW10_${h}_LM_PIMask_1Step/epoch-10000.pt" ;;
    *)               echo "" ;;
  esac
}

TASK_ID=${SLURM_ARRAY_TASK_ID}
HARDNESS=${HARDNESSES[$TASK_ID]}
MODE=${MODES[$TASK_ID]}

CKPT_PATTERN=$(get_ckpt_pattern "$HARDNESS" "$MODE")

echo "=== TASK $TASK_ID: TSPTW $HARDNESS $MODE (reveal_delay_before_action flag passed but unused) ==="

if [ -n "$CKPT_PATTERN" ] && ls $CKPT_PATTERN >/dev/null 2>&1; then
  echo ">> Skip $MODE $HARDNESS (found $CKPT_PATTERN)"
else
  srun python train.py $COMMON_ARGS --hardness "$HARDNESS" --model_type "$MODE" --reveal_delay_before_action
fi

echo "=== TASK $TASK_ID FINISHED at $(date) ==="

