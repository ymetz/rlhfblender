#!/bin/bash
# ============================================================
# SLURM ablation launcher: Supervised feedback baseline sweep
#
# Comprehensive hyperparameter search for the supervised (GT reward)
# feedback type. Ablates both RLHF pipeline params and PPO params.
#
# Usage (from repo root):
#   bash scripts/submit_supervised_ablations.sh [--dry-run] [--group=L] [--label=L_steps_50k] [--seed=42]
#
# --dry-run:      print sbatch commands without submitting
# --group=LETTER: only submit configs whose label starts with LETTER_ (e.g. --group=L)
# --label=NAME:   only submit the single named config
# --seed=N:       only submit runs with this specific seed
# ============================================================

set -euo pipefail

# ── Cluster settings ─────────────────────────────────────────────────────────
PARTITION="cpu"
ACCOUNT=""                          # leave blank if not required
CPUS=2
MEM="4G"
TIME="24:00:00"                     # 500K+ total RL steps on CPU needs generous wall time

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXPERT_MODEL_PATH="multi-type-feedback/train_baselines/gt_agents"
LOG_DIR="$REPO_ROOT/slurm_logs"

# ── Fixed settings (shared across all runs) ───────────────────────────────────
ENV="metaworld-sweep-into-v3"
ALGO="ppo"
EXPERT_ALGO="sac"
MAX_EPISODE_STEPS=150
STATE_SEED=0
N_TRAJECTORIES=10
SEGMENT_LEN=150
DEVICE="cpu"
SEEDS=(42 123)                      # 2 seeds (many configs)

# ── Reference values for the supervised baseline ──────────────────────────────
# These are the "default" values when an axis is not being swept.
REF_RL_STEPS=50000
REF_PHASES=10
REF_BUDGET=500
REF_EPOCHS=20
REF_INITIAL=50

# All configs use supervised feedback only
FB_TYPES="--feedback-types supervised"

# ── Config format ─────────────────────────────────────────────────────────────
# 8 pipe-separated fields:
#   label | --rl-steps | --num-phases | --feedback-budget |
#   --reward-epochs | --initial-feedback | EXTRA_ARGS | HYPERPARAMS
#
# EXTRA_ARGS:   additional flags (buffer size, model type, etc.)
# HYPERPARAMS:  PPO overrides as KEY:VALUE (passed via --hyperparams)

# ── Optimized PPO baseline (from round 1 ablation results) ────────────────────
# With n_envs=1, the ppo.yml defaults (tuned for n_envs=16) are broken.
# These values are derived from single-axis winners in round 1:
#   n_steps=512 (S, 112.4), gamma=0.9 (R, 61.5), gae_lambda=0.9 (W, 57.4),
#   clip_range=0.2 (U, 45.5), ent_coef=0.01 (X, 42.3), n_epochs=10 (Z_aggressive),
#   learning_rate=1e-4 (Z_conservative), batch_size=128 (Z_aggressive)
TUNED_PPO="n_steps:512 gamma:0.9 gae_lambda:0.9 clip_range:0.2 ent_coef:0.01 learning_rate:1e-4 batch_size:128 n_epochs:10"

CONFIGS=(

# ══════════════════════════════════════════════════════════════════════════════
# Group L2: RL training steps per phase (with tuned PPO)
# ══════════════════════════════════════════════════════════════════════════════
    "L2_steps_10k  |--rl-steps  10000|--num-phases 10|--feedback-budget 500|--reward-epochs 10|--initial-feedback  50||${TUNED_PPO}"
    "L2_steps_20k  |--rl-steps  20000|--num-phases 10|--feedback-budget 500|--reward-epochs 10|--initial-feedback  50||${TUNED_PPO}"
    "L2_steps_50k  |--rl-steps  50000|--num-phases 10|--feedback-budget 500|--reward-epochs 10|--initial-feedback  50||${TUNED_PPO}"
    "L2_steps_100k |--rl-steps 100000|--num-phases 10|--feedback-budget 500|--reward-epochs 10|--initial-feedback  50||${TUNED_PPO}"

# ══════════════════════════════════════════════════════════════════════════════
# Group M2: Phase granularity (total RL = 500K steps, with tuned PPO)
# ══════════════════════════════════════════════════════════════════════════════
    "M2_grain_p05_s100k|--rl-steps 100000|--num-phases  5|--feedback-budget 500|--reward-epochs 10|--initial-feedback  50||${TUNED_PPO}"
    "M2_grain_p10_s50k |--rl-steps  50000|--num-phases 10|--feedback-budget 500|--reward-epochs 10|--initial-feedback  50||${TUNED_PPO}"
    "M2_grain_p20_s25k |--rl-steps  25000|--num-phases 20|--feedback-budget 500|--reward-epochs 10|--initial-feedback  50||${TUNED_PPO}"
    "M2_grain_p50_s10k |--rl-steps  10000|--num-phases 50|--feedback-budget 500|--reward-epochs 10|--initial-feedback  50||${TUNED_PPO}"

# ══════════════════════════════════════════════════════════════════════════════
# Group N2: Feedback budget (with tuned PPO)
# ══════════════════════════════════════════════════════════════════════════════
    "N2_budget_100  |--rl-steps 50000|--num-phases 10|--feedback-budget  100|--reward-epochs 10|--initial-feedback  25||${TUNED_PPO}"
    "N2_budget_250  |--rl-steps 50000|--num-phases 10|--feedback-budget  250|--reward-epochs 10|--initial-feedback  50||${TUNED_PPO}"
    "N2_budget_500  |--rl-steps 50000|--num-phases 10|--feedback-budget  500|--reward-epochs 10|--initial-feedback  50||${TUNED_PPO}"
    "N2_budget_1000 |--rl-steps 50000|--num-phases 10|--feedback-budget 1000|--reward-epochs 10|--initial-feedback 100||${TUNED_PPO}"
    "N2_budget_2000 |--rl-steps 50000|--num-phases 10|--feedback-budget 2000|--reward-epochs 10|--initial-feedback 200||${TUNED_PPO}"

# ══════════════════════════════════════════════════════════════════════════════
# Group O2: Initial feedback (with tuned PPO)
# ══════════════════════════════════════════════════════════════════════════════
    "O2_init_10   |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 10|--initial-feedback   10||${TUNED_PPO}"
    "O2_init_25   |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 10|--initial-feedback   25||${TUNED_PPO}"
    "O2_init_50   |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 10|--initial-feedback   50||${TUNED_PPO}"
    "O2_init_100  |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 10|--initial-feedback  100||${TUNED_PPO}"
    "O2_init_250  |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 10|--initial-feedback  250||${TUNED_PPO}"

# ══════════════════════════════════════════════════════════════════════════════
# Group P2: Reward model training epochs (with tuned PPO)
# ══════════════════════════════════════════════════════════════════════════════
    "P2_epochs_5   |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs  5|--initial-feedback  50||${TUNED_PPO}"
    "P2_epochs_10  |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 10|--initial-feedback  50||${TUNED_PPO}"
    "P2_epochs_20  |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||${TUNED_PPO}"
    "P2_epochs_50  |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 50|--initial-feedback  50||${TUNED_PPO}"
    "P2_epochs_100 |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 100|--initial-feedback  50||${TUNED_PPO}"

# ══════════════════════════════════════════════════════════════════════════════
# Group Q2: PPO learning rate (fine-tune around 1e-4 baseline)
# ══════════════════════════════════════════════════════════════════════════════
    "Q2_lr_3e5  |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 10|--initial-feedback  50||n_steps:512 gamma:0.9 gae_lambda:0.9 clip_range:0.2 ent_coef:0.01 batch_size:128 n_epochs:10 learning_rate:3e-5"
    "Q2_lr_5e5  |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 10|--initial-feedback  50||n_steps:512 gamma:0.9 gae_lambda:0.9 clip_range:0.2 ent_coef:0.01 batch_size:128 n_epochs:10 learning_rate:5e-5"
    "Q2_lr_1e4  |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 10|--initial-feedback  50||n_steps:512 gamma:0.9 gae_lambda:0.9 clip_range:0.2 ent_coef:0.01 batch_size:128 n_epochs:10 learning_rate:1e-4"
    "Q2_lr_3e4  |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 10|--initial-feedback  50||n_steps:512 gamma:0.9 gae_lambda:0.9 clip_range:0.2 ent_coef:0.01 batch_size:128 n_epochs:10 learning_rate:3e-4"
    "Q2_lr_5e4  |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 10|--initial-feedback  50||n_steps:512 gamma:0.9 gae_lambda:0.9 clip_range:0.2 ent_coef:0.01 batch_size:128 n_epochs:10 learning_rate:5e-4"

# ══════════════════════════════════════════════════════════════════════════════
# Group R2: PPO gamma (fine-tune around 0.9 baseline)
# ══════════════════════════════════════════════════════════════════════════════
    "R2_gamma_080 |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 10|--initial-feedback  50||n_steps:512 gae_lambda:0.9 clip_range:0.2 ent_coef:0.01 learning_rate:1e-4 batch_size:128 n_epochs:10 gamma:0.8"
    "R2_gamma_085 |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 10|--initial-feedback  50||n_steps:512 gae_lambda:0.9 clip_range:0.2 ent_coef:0.01 learning_rate:1e-4 batch_size:128 n_epochs:10 gamma:0.85"
    "R2_gamma_090 |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 10|--initial-feedback  50||n_steps:512 gae_lambda:0.9 clip_range:0.2 ent_coef:0.01 learning_rate:1e-4 batch_size:128 n_epochs:10 gamma:0.9"
    "R2_gamma_095 |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 10|--initial-feedback  50||n_steps:512 gae_lambda:0.9 clip_range:0.2 ent_coef:0.01 learning_rate:1e-4 batch_size:128 n_epochs:10 gamma:0.95"
    "R2_gamma_098 |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 10|--initial-feedback  50||n_steps:512 gae_lambda:0.9 clip_range:0.2 ent_coef:0.01 learning_rate:1e-4 batch_size:128 n_epochs:10 gamma:0.98"

# ══════════════════════════════════════════════════════════════════════════════
# Group S2: PPO n_steps (fine-tune around 512 baseline)
# ══════════════════════════════════════════════════════════════════════════════
    "S2_nsteps_256  |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 10|--initial-feedback  50||gamma:0.9 gae_lambda:0.9 clip_range:0.2 ent_coef:0.01 learning_rate:1e-4 batch_size:128 n_epochs:10 n_steps:256"
    "S2_nsteps_512  |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 10|--initial-feedback  50||gamma:0.9 gae_lambda:0.9 clip_range:0.2 ent_coef:0.01 learning_rate:1e-4 batch_size:128 n_epochs:10 n_steps:512"
    "S2_nsteps_1024 |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 10|--initial-feedback  50||gamma:0.9 gae_lambda:0.9 clip_range:0.2 ent_coef:0.01 learning_rate:1e-4 batch_size:128 n_epochs:10 n_steps:1024"
    "S2_nsteps_2048 |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 10|--initial-feedback  50||gamma:0.9 gae_lambda:0.9 clip_range:0.2 ent_coef:0.01 learning_rate:1e-4 batch_size:128 n_epochs:10 n_steps:2048"
    "S2_nsteps_4096 |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 10|--initial-feedback  50||gamma:0.9 gae_lambda:0.9 clip_range:0.2 ent_coef:0.01 learning_rate:1e-4 batch_size:128 n_epochs:10 n_steps:4096"

# ══════════════════════════════════════════════════════════════════════════════
# Group Z2: Best-guess combos (round 2)
# ══════════════════════════════════════════════════════════════════════════════
    # Tuned PPO baseline (the reference point for all round 2)
    "Z2_baseline         |--rl-steps  50000|--num-phases 10|--feedback-budget 500|--reward-epochs 10|--initial-feedback  50||${TUNED_PPO}"
    # More total RL (1M steps)
    "Z2_long_train       |--rl-steps 100000|--num-phases 10|--feedback-budget 500|--reward-epochs 10|--initial-feedback  50||${TUNED_PPO}"
    # Fine phases + tuned PPO (50 phases × 10K = 500K total, more reward model updates)
    "Z2_fine_phases       |--rl-steps  10000|--num-phases 50|--feedback-budget 500|--reward-epochs 10|--initial-feedback  50||${TUNED_PPO}"
    # Rich data + tuned PPO
    "Z2_rich_data         |--rl-steps  50000|--num-phases 10|--feedback-budget 1000|--reward-epochs 10|--initial-feedback 100||${TUNED_PPO}"
    # Fine phases + rich data + tuned PPO
    "Z2_fine_rich         |--rl-steps  10000|--num-phases 50|--feedback-budget 1000|--reward-epochs 10|--initial-feedback 100||${TUNED_PPO}"
    # Larger rollout buffer (n_steps=1024) with tuned PPO
    "Z2_big_rollout       |--rl-steps  50000|--num-phases 10|--feedback-budget 500|--reward-epochs 10|--initial-feedback  50||n_steps:1024 gamma:0.9 gae_lambda:0.9 clip_range:0.2 ent_coef:0.01 learning_rate:1e-4 batch_size:128 n_epochs:10"
    # Smaller policy net (faster learning with less data)
    "Z2_small_net         |--rl-steps  50000|--num-phases 10|--feedback-budget 500|--reward-epochs 10|--initial-feedback  50||n_steps:512 gamma:0.9 gae_lambda:0.9 clip_range:0.2 ent_coef:0.01 learning_rate:1e-4 batch_size:128 n_epochs:10 policy_kwargs:dict(net_arch=dict(pi=[64,64],vf=[64,64]))"
    # Even lower gamma (very short horizon)
    "Z2_very_short_horizon|--rl-steps  50000|--num-phases 10|--feedback-budget 500|--reward-epochs 10|--initial-feedback  50||n_steps:512 gamma:0.8 gae_lambda:0.9 clip_range:0.2 ent_coef:0.01 learning_rate:1e-4 batch_size:128 n_epochs:10"
    # Higher entropy for more exploration
    "Z2_high_entropy      |--rl-steps  50000|--num-phases 10|--feedback-budget 500|--reward-epochs 10|--initial-feedback  50||n_steps:512 gamma:0.9 gae_lambda:0.9 clip_range:0.2 ent_coef:0.05 learning_rate:1e-4 batch_size:128 n_epochs:10"

)

# ── Helpers ───────────────────────────────────────────────────────────────────
DRY_RUN=false
ONLY_LABEL=""
ONLY_SEED=""
ONLY_GROUP=""
for arg in "$@"; do
    case "$arg" in
        --dry-run)  DRY_RUN=true ;;
        --label=*)  ONLY_LABEL="${arg#--label=}" ;;
        --seed=*)   ONLY_SEED="${arg#--seed=}" ;;
        --group=*)  ONLY_GROUP="${arg#--group=}" ;;
    esac
done

mkdir -p "$LOG_DIR"

ACCOUNT_FLAG=""
if [[ -n "$ACCOUNT" ]]; then ACCOUNT_FLAG="#SBATCH --account=$ACCOUNT"; fi

n_submitted=0

# ── Submit ────────────────────────────────────────────────────────────────────
for config_str in "${CONFIGS[@]}"; do
    IFS='|' read -r label rl_steps_arg num_phases_arg budget_arg epochs_arg initial_arg extra_arg hp_arg <<< "$config_str"

    # Strip leading/trailing whitespace from label
    label="$(echo "$label" | xargs)"

    seed_list=("${SEEDS[@]}")

    # Filter by --group / --label / --seed if provided
    [[ -n "$ONLY_GROUP" && "$label" != "${ONLY_GROUP}_"* ]] && continue
    [[ -n "$ONLY_LABEL" && "$label" != "$ONLY_LABEL" ]] && continue

    for SEED in "${seed_list[@]}"; do
        [[ -n "$ONLY_SEED" && "$SEED" != "$ONLY_SEED" ]] && continue
        EXP_NAME="${ENV//metaworld-/mw_}_${ALGO}_${label}_s${SEED}"

        # Build the python command
        PYTHON_CMD="python scripts/run_simulated_phases.py"
        PYTHON_CMD+=" --env ${ENV}"
        PYTHON_CMD+=" --algorithm ${ALGO}"
        PYTHON_CMD+=" --expert-algorithm ${EXPERT_ALGO}"
        PYTHON_CMD+=" --expert-model-path ${EXPERT_MODEL_PATH}"
        PYTHON_CMD+=" --exp-name ${EXP_NAME}"
        PYTHON_CMD+=" --seed ${SEED}"
        PYTHON_CMD+=" --device ${DEVICE}"
        PYTHON_CMD+=" --max-episode-steps ${MAX_EPISODE_STEPS}"
        PYTHON_CMD+=" --n-trajectories ${N_TRAJECTORIES}"
        PYTHON_CMD+=" --segment-len ${SEGMENT_LEN}"
        PYTHON_CMD+=" --fix-start-state"
        PYTHON_CMD+=" --state-seed ${STATE_SEED}"
        PYTHON_CMD+=" --skip-projections"
        PYTHON_CMD+=" --results-only"
        PYTHON_CMD+=" --n-envs 1"
        PYTHON_CMD+=" ${rl_steps_arg}"
        PYTHON_CMD+=" ${num_phases_arg}"
        PYTHON_CMD+=" ${budget_arg}"
        PYTHON_CMD+=" --uncertainty-penalty 0.0"
        PYTHON_CMD+=" ${epochs_arg}"
        PYTHON_CMD+=" ${initial_arg}"
        PYTHON_CMD+=" ${FB_TYPES}"
        [[ -n "$extra_arg" ]] && PYTHON_CMD+=" ${extra_arg}"

        # PPO hyperparameter overrides
        if [[ -n "$hp_arg" ]]; then
            PYTHON_CMD+=" --hyperparams ${hp_arg}"
        fi

        JOB_SCRIPT=$(cat <<SLURM
#!/bin/bash
#SBATCH --job-name=sv_${label}_s${SEED}
#SBATCH --partition=${PARTITION}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem-per-cpu=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${LOG_DIR}/${EXP_NAME}_%j.out
#SBATCH --error=${LOG_DIR}/${EXP_NAME}_%j.err
${ACCOUNT_FLAG}

set -euo pipefail
cd "${REPO_ROOT}"

${PYTHON_CMD}
SLURM
)

        if $DRY_RUN; then
            echo "=== DRY RUN: $EXP_NAME ==="
            echo "$JOB_SCRIPT"
            echo ""
        else
            JOB_ID=$(echo "$JOB_SCRIPT" | sbatch --parsable)
            echo "Submitted [$label | seed=$SEED] → job $JOB_ID"
        fi

        n_submitted=$((n_submitted + 1))
    done
done

echo ""
if $DRY_RUN; then
    echo "Dry run: would submit $n_submitted jobs (${#SEEDS[@]} seeds each)."
else
    echo "Submitted $n_submitted jobs. Logs → $LOG_DIR/"
fi
