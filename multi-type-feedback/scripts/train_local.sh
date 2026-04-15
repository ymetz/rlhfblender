#!/bin/bash
# Local parallel equivalent of train_baseline_agents.sh (no Slurm).
# Run from the multi-type-feedback/ directory:
#   bash scripts/train_local.sh
#
# Adjust ENVS, SEEDS, SAVE_FREQ, and ALGO below as needed.

set -euo pipefail

ALGO="sac"
ENVS=("metaworld-sweep-into-v3")
SEEDS=(1789) #1687123 12 912391 330)
SAVE_FREQ=50000          # checkpoints every 10k steps; CartPole trains for 100k total
LOG_FOLDER="train_baselines/gt_agents"
FIX_START_STATE=true     # set to true to fix starting/object/goal positions every episode
STATE_SEED=0             # seed used ONLY for sampling the fixed start state;
                         # must match --state-seed in run_simulated_phases.py

mkdir -p logs

PIDS=()

for ENV in "${ENVS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        LOG_FILE="logs/train_${ALGO}_${ENV}_seed${SEED}.log"
        echo "Starting: algo=$ALGO  env=$ENV  seed=$SEED  → $LOG_FILE"
        FIX_FLAG=""
        if [ "$FIX_START_STATE" = "true" ]; then FIX_FLAG="--fix-start-state --state-seed $STATE_SEED"; fi
        python -u train_baselines/train.py \
            --algo "$ALGO" \
            --env  "$ENV" \
            --seed "$SEED" \
            --save-freq "$SAVE_FREQ" \
            --log-folder "$LOG_FOLDER" \
            --env-kwargs "max_episode_steps:150" \
            $FIX_FLAG \
            > "$LOG_FILE" 2>&1 &
        PIDS+=($!)
        sleep 3  # let SB3 claim its run directory before the next process starts
    done
done

echo ""
echo "Launched ${#PIDS[@]} training runs (PIDs: ${PIDS[*]})"
echo "Waiting for all to finish..."

FAILED=0
for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
        echo "  [WARN] Process $PID exited with error"
        FAILED=$((FAILED + 1))
    fi
done

if [ "$FAILED" -eq 0 ]; then
    echo "All training runs completed successfully."
else
    echo "$FAILED run(s) failed — check logs/ for details."
    exit 1
fi
