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

CONFIGS=(

# ══════════════════════════════════════════════════════════════════════════════
# Group L: RL training steps per phase
# ══════════════════════════════════════════════════════════════════════════════
# Question: How many RL steps does the supervised reward model need?
# Total RL = rl_steps × phases. Sweeps from 100K total to 1M total.
    "L_steps_10k  |--rl-steps  10000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||"
    "L_steps_20k  |--rl-steps  20000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||"
    "L_steps_50k  |--rl-steps  50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||"
    "L_steps_100k |--rl-steps 100000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||"

# ══════════════════════════════════════════════════════════════════════════════
# Group M: Phase granularity (total RL = 500K steps)
# ══════════════════════════════════════════════════════════════════════════════
# Question: Frequent reward model updates vs fewer longer training bursts?
    "M_grain_p05_s100k|--rl-steps 100000|--num-phases  5|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||"
    "M_grain_p10_s50k |--rl-steps  50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||"
    "M_grain_p20_s25k |--rl-steps  25000|--num-phases 20|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||"
    "M_grain_p50_s10k |--rl-steps  10000|--num-phases 50|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||"

# ══════════════════════════════════════════════════════════════════════════════
# Group N: Feedback budget (supervised = per-step GT rewards, ~150 items/traj)
# ══════════════════════════════════════════════════════════════════════════════
# Question: How much GT reward data does the reward model need?
    "N_budget_100  |--rl-steps 50000|--num-phases 10|--feedback-budget  100|--reward-epochs 20|--initial-feedback  25||"
    "N_budget_250  |--rl-steps 50000|--num-phases 10|--feedback-budget  250|--reward-epochs 20|--initial-feedback  50||"
    "N_budget_500  |--rl-steps 10000|--num-phases 10|--feedback-budget  500|--reward-epochs 20|--initial-feedback  50||"
    "N_budget_1000 |--rl-steps 50000|--num-phases 10|--feedback-budget 1000|--reward-epochs 20|--initial-feedback 100||"
    "N_budget_2000 |--rl-steps 50000|--num-phases 10|--feedback-budget 2000|--reward-epochs 20|--initial-feedback 200||"

# ══════════════════════════════════════════════════════════════════════════════
# Group O: Initial feedback (phase 0 warmup)
# ══════════════════════════════════════════════════════════════════════════════
# Question: How much warmup data does the supervised reward model need?
    "O_init_10   |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback   10||"
    "O_init_25   |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback   25||"
    "O_init_50   |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback   50||"
    "O_init_100  |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  100||"
    "O_init_250  |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  250||"

# ══════════════════════════════════════════════════════════════════════════════
# Group P: Reward model training epochs
# ══════════════════════════════════════════════════════════════════════════════
# Question: How much reward model training per phase?
    "P_epochs_5   |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs  5|--initial-feedback  50||"
    "P_epochs_10  |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 10|--initial-feedback  50||"
    "P_epochs_20  |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||"
    "P_epochs_50  |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 50|--initial-feedback  50||"
    "P_epochs_100 |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 100|--initial-feedback  50||"

# ══════════════════════════════════════════════════════════════════════════════
# Group Q: PPO learning rate
# ══════════════════════════════════════════════════════════════════════════════
# Baseline: 3e-4 (from ppo.yml). Sweep around it.
    "Q_lr_1e5  |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||learning_rate:1e-5"
    "Q_lr_5e5  |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||learning_rate:5e-5"
    "Q_lr_1e4  |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||learning_rate:1e-4"
    "Q_lr_3e4  |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||learning_rate:3e-4"
    "Q_lr_5e4  |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||learning_rate:5e-4"
    "Q_lr_1e3  |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||learning_rate:1e-3"

# ══════════════════════════════════════════════════════════════════════════════
# Group R: PPO gamma (discount factor)
# ══════════════════════════════════════════════════════════════════════════════
# Baseline: 0.99. Critical for reward shaping — learned rewards may need
# different discounting than GT env rewards.
    "R_gamma_090 |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||gamma:0.9"
    "R_gamma_095 |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||gamma:0.95"
    "R_gamma_098 |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||gamma:0.98"
    "R_gamma_099 |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||gamma:0.99"
    "R_gamma_0995|--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||gamma:0.995"
    "R_gamma_0999|--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||gamma:0.999"

# ══════════════════════════════════════════════════════════════════════════════
# Group S: PPO n_steps (rollout buffer length)
# ══════════════════════════════════════════════════════════════════════════════
# Baseline: 256 (× 16 envs = 4096 samples/update). With n_envs=1 in ablations,
# n_steps directly controls samples/update.
    "S_nsteps_64   |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||n_steps:64"
    "S_nsteps_128  |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||n_steps:128"
    "S_nsteps_256  |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||n_steps:256"
    "S_nsteps_512  |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||n_steps:512"
    "S_nsteps_1024 |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||n_steps:1024"
    "S_nsteps_2048 |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||n_steps:2048"

# ══════════════════════════════════════════════════════════════════════════════
# Group T: PPO batch_size (minibatch size for SGD updates)
# ══════════════════════════════════════════════════════════════════════════════
# Baseline: 64. Must be <= n_steps (with n_envs=1).
# Using n_steps=256 (default) as constraint.
    "T_batch_16  |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||batch_size:16"
    "T_batch_32  |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||batch_size:32"
    "T_batch_64  |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||batch_size:64"
    "T_batch_128 |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||batch_size:128"
    "T_batch_256 |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||batch_size:256"

# ══════════════════════════════════════════════════════════════════════════════
# Group U: PPO clip_range
# ══════════════════════════════════════════════════════════════════════════════
# Baseline: 0.4 (aggressive). Reward model noise may benefit from tighter clipping.
    "U_clip_01  |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||clip_range:0.1"
    "U_clip_02  |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||clip_range:0.2"
    "U_clip_03  |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||clip_range:0.3"
    "U_clip_04  |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||clip_range:0.4"

# ══════════════════════════════════════════════════════════════════════════════
# Group V: PPO n_epochs (SGD passes per rollout)
# ══════════════════════════════════════════════════════════════════════════════
# Baseline: 20. With noisy reward signal, fewer epochs may prevent overfitting.
    "V_nepochs_5  |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||n_epochs:5"
    "V_nepochs_10 |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||n_epochs:10"
    "V_nepochs_20 |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||n_epochs:20"
    "V_nepochs_30 |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||n_epochs:30"

# ══════════════════════════════════════════════════════════════════════════════
# Group W: PPO gae_lambda
# ══════════════════════════════════════════════════════════════════════════════
# Baseline: 0.92. Controls bias-variance tradeoff in advantage estimation.
    "W_gae_080 |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||gae_lambda:0.8"
    "W_gae_090 |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||gae_lambda:0.9"
    "W_gae_092 |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||gae_lambda:0.92"
    "W_gae_095 |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||gae_lambda:0.95"
    "W_gae_098 |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||gae_lambda:0.98"
    "W_gae_100 |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||gae_lambda:1.0"

# ══════════════════════════════════════════════════════════════════════════════
# Group X: PPO entropy coefficient
# ══════════════════════════════════════════════════════════════════════════════
# Baseline: 0.0. Exploration bonus may help with noisy reward landscape.
    "X_ent_000  |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||ent_coef:0.0"
    "X_ent_001  |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||ent_coef:0.01"
    "X_ent_005  |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||ent_coef:0.05"
    "X_ent_010  |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||ent_coef:0.1"

# ══════════════════════════════════════════════════════════════════════════════
# Group Y: PPO network architecture
# ══════════════════════════════════════════════════════════════════════════════
# Baseline: [256, 256, 256]. Smaller/larger networks for the policy.
    "Y_arch_64x2    |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||policy_kwargs:dict(net_arch=dict(pi=[64,64],vf=[64,64]))"
    "Y_arch_128x2   |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||policy_kwargs:dict(net_arch=dict(pi=[128,128],vf=[128,128]))"
    "Y_arch_256x2   |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||policy_kwargs:dict(net_arch=dict(pi=[256,256],vf=[256,256]))"
    "Y_arch_256x3   |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||policy_kwargs:dict(net_arch=dict(pi=[256,256,256],vf=[256,256,256]))"
    "Y_arch_512x2   |--rl-steps 50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||policy_kwargs:dict(net_arch=dict(pi=[512,512],vf=[512,512]))"

# ══════════════════════════════════════════════════════════════════════════════
# Group Z: Best-guess combinations
# ══════════════════════════════════════════════════════════════════════════════
# Promising multi-axis combos informed by single-axis intuitions.

    # High RL budget + generous feedback
    "Z_high_budget       |--rl-steps  50000|--num-phases 10|--feedback-budget 1000|--reward-epochs 20|--initial-feedback 100||"
    # High RL budget + fine phases
    "Z_high_fine         |--rl-steps  10000|--num-phases 50|--feedback-budget 1000|--reward-epochs 20|--initial-feedback 100||"
    # High RL + conservative PPO (lower lr, tighter clip, more exploration)
    "Z_conservative_ppo  |--rl-steps  50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||learning_rate:1e-4 clip_range:0.2 ent_coef:0.01"
    # High RL + aggressive PPO (higher lr, big batches)
    "Z_aggressive_ppo    |--rl-steps  50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||learning_rate:5e-4 batch_size:128 n_epochs:10"
    # Long rollouts + high gamma (maximize long-horizon credit)
    "Z_long_horizon      |--rl-steps  50000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||n_steps:1024 gamma:0.995 gae_lambda:0.98"
    # Minimal architecture + lots of training
    "Z_small_net_long    |--rl-steps 100000|--num-phases 10|--feedback-budget 500|--reward-epochs 20|--initial-feedback  50||policy_kwargs:dict(net_arch=dict(pi=[64,64],vf=[64,64]))"
    # Kitchen sink: high budget + fine phases + tuned PPO
    "Z_kitchen_sink      |--rl-steps  20000|--num-phases 25|--feedback-budget 1000|--reward-epochs 30|--initial-feedback 100||learning_rate:1e-4 clip_range:0.2 n_steps:512 gamma:0.995"

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
