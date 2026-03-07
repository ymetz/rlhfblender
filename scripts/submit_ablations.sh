#!/bin/bash
# ============================================================
# SLURM ablation launcher for run_simulated_phases.py
#
# Usage (from repo root):
#   bash scripts/submit_ablations.sh [--dry-run]
#
# --dry-run: print sbatch commands without submitting
# ============================================================

set -euo pipefail

# ── Cluster settings ─────────────────────────────────────────────────────────
PARTITION="cpu"
ACCOUNT="yametz"                          # leave blank if not required
CPUS=2
MEM="4G"
TIME="06:00:00"

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_ENV="rlhf"                    # conda env name; leave blank to skip activation
EXPERT_MODEL_PATH="multi-type-feedback/train_baselines/gt_agents"
LOG_DIR="$REPO_ROOT/slurm_logs"

# ── Fixed settings (shared across all runs) ───────────────────────────────────
ENV="metaworld-sweep-into-v3"
ALGO="ppo"
EXPERT_ALGO="sac"
MAX_EPISODE_STEPS=150               # matches train_local.sh
STATE_SEED=0
N_TRAJECTORIES=10
SEGMENT_LEN=50
DEVICE="cpu"
SEEDS=(42 123 456)                  # 3 seeds for A–H groups
SEEDS_FB=(42 123)                   # 2 seeds for Group I (feedback type combos — more configs)

# Reference values for axes not under study in each group
REF_RL_STEPS=2000
REF_PHASES=10
REF_BUDGET=500
REF_PENALTY=0.0
REF_EPOCHS=20
REF_INITIAL=50
REF_BUFFER=""                       # empty = default (equals budget)

# ── Ablation grid ─────────────────────────────────────────────────────────────
# Format: "label|--rl-steps N|--num-phases P|--feedback-budget B|
#          --uncertainty-penalty U|--reward-epochs E|--initial-feedback I|BUFFER_EXTRA"
# BUFFER_EXTRA is either empty or "--feedback-buffer-size N"
#
# Each group varies ONE axis; everything else is held at REF_* values.
# Total: ~32 configs × 3 seeds = 96 jobs.

CONFIGS=(
# Config format (10 pipe-separated fields):
#   label | --rl-steps | --num-phases | --feedback-budget | --uncertainty-penalty
#         | --reward-epochs | --initial-feedback | BUFFER_EXTRA | FB_TYPES_EXTRA | SEG_LEN_EXTRA
#
# BUFFER_EXTRA  : empty OR "--feedback-buffer-size N"
# FB_TYPES_EXTRA: empty (use default 3 types) OR "--feedback-types TYPE [TYPE ...]"
# SEG_LEN_EXTRA : empty (use SEGMENT_LEN=50) OR "--segment-len N"  (overrides fixed default)
#
# Groups A–I: all use penalty=0.0 (penalty ablation deferred).
# Each group varies ONE axis; all others fixed at reference values.

# ── Group A: RL steps per phase ───────────────────────────────────────────────
# Question: How much does per-phase exploitation matter?
# Fixed: phases=10, budget=500, seg=50, epochs=20, initial=50
    "A_steps_0500|--rl-steps    500|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.0|--reward-epochs 20|--initial-feedback  50|||"
    "A_steps_1000|--rl-steps   1000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.0|--reward-epochs 20|--initial-feedback  50|||"
    "A_steps_2000|--rl-steps   2000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.0|--reward-epochs 20|--initial-feedback  50|||"
    "A_steps_5000|--rl-steps   5000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.0|--reward-epochs 20|--initial-feedback  50|||"
    "A_steps_10k |--rl-steps  10000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.0|--reward-epochs 20|--initial-feedback  50|||"

# ── Group B: Phase granularity (total RL = 20k steps) ────────────────────────
# Question: More-frequent reward model updates vs fewer longer bursts?
# Fixed: total_rl=20k, budget=500, seg=50, epochs=20, initial=50
# A_steps_2000 (phases=10, steps=2000) already provides the 20k midpoint.
    "B_grain_p05_s4000|--rl-steps  4000|--num-phases  5|--feedback-budget 500|--uncertainty-penalty 0.0|--reward-epochs 20|--initial-feedback  50|||"
    "B_grain_p20_s1000|--rl-steps  1000|--num-phases 20|--feedback-budget 500|--uncertainty-penalty 0.0|--reward-epochs 20|--initial-feedback  50|||"
    "B_grain_p40_s0500|--rl-steps   500|--num-phases 40|--feedback-budget 500|--uncertainty-penalty 0.0|--reward-epochs 20|--initial-feedback  50|||"

# ── Group C: Feedback budget ──────────────────────────────────────────────────
# Question: More oracle data per run → better reward model?
# Fixed: phases=10, steps=2000, seg=50, epochs=20, initial=50
    "C_budget_0250|--rl-steps 2000|--num-phases 10|--feedback-budget  250|--uncertainty-penalty 0.0|--reward-epochs 20|--initial-feedback  50|||"
    # budget=500 ← covered by A_steps_2000
    "C_budget_0750|--rl-steps 2000|--num-phases 10|--feedback-budget  750|--uncertainty-penalty 0.0|--reward-epochs 20|--initial-feedback  50|||"
    "C_budget_1000|--rl-steps 2000|--num-phases 10|--feedback-budget 1000|--uncertainty-penalty 0.0|--reward-epochs 20|--initial-feedback  50|||"
    "C_budget_1500|--rl-steps 2000|--num-phases 10|--feedback-budget 1500|--uncertainty-penalty 0.0|--reward-epochs 20|--initial-feedback  50|||"

# ── Group D: Reward model training epochs ─────────────────────────────────────
# Question: How much does reward model training depth matter?
# Fixed: phases=10, steps=2000, budget=500, seg=50, initial=50
    "D_epochs_05|--rl-steps 2000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.0|--reward-epochs  5|--initial-feedback  50|||"
    "D_epochs_10|--rl-steps 2000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.0|--reward-epochs 10|--initial-feedback  50|||"
    # epochs=20 ← covered by A_steps_2000
    "D_epochs_30|--rl-steps 2000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.0|--reward-epochs 30|--initial-feedback  50|||"
    "D_epochs_50|--rl-steps 2000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.0|--reward-epochs 50|--initial-feedback  50|||"

# ── Group E: Initial feedback count (phase 0 warmup) ─────────────────────────
# Question: Does a better phase-0 reward model bootstrap subsequent phases?
# Fixed: phases=10, steps=2000, budget=500, seg=50, epochs=20
    "E_init_0025|--rl-steps 2000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.0|--reward-epochs 20|--initial-feedback   25|||"
    # initial=50 ← covered by A_steps_2000
    "E_init_0100|--rl-steps 2000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.0|--reward-epochs 20|--initial-feedback  100|||"
    "E_init_0250|--rl-steps 2000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.0|--reward-epochs 20|--initial-feedback  250|||"
    "E_init_0500|--rl-steps 2000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.0|--reward-epochs 20|--initial-feedback  500|||"

# ── Group F: Feedback buffer size (staleness) ─────────────────────────────────
# Question: Should old off-distribution feedback expire faster?
# Fixed: phases=10, steps=2000, budget=500, seg=50, epochs=20, initial=50
    "F_buf_100|--rl-steps 2000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.0|--reward-epochs 20|--initial-feedback  50|--feedback-buffer-size  100||"
    "F_buf_250|--rl-steps 2000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.0|--reward-epochs 20|--initial-feedback  50|--feedback-buffer-size  250||"
    # no buffer flag (keep all) ← covered by A_steps_2000

# ── Group G: Segment length ───────────────────────────────────────────────────
# Question: What feedback granularity suits this task?
# Shorter segments = more clips per episode, finer reward signal but less context.
# Longer segments = richer context but fewer clips per budget.
# Fixed: phases=10, steps=2000, budget=500, epochs=20, initial=50
# max_episode_steps=150, so seg=150 = full episode; seg=10 = 15 clips/episode.
    "G_seg_010|--rl-steps 2000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.0|--reward-epochs 20|--initial-feedback  50|||--segment-len  10"
    "G_seg_025|--rl-steps 2000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.0|--reward-epochs 20|--initial-feedback  50|||--segment-len  25"
    # seg=50 ← covered by A_steps_2000
    "G_seg_075|--rl-steps 2000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.0|--reward-epochs 20|--initial-feedback  50|||--segment-len  75"
    "G_seg_100|--rl-steps 2000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.0|--reward-epochs 20|--initial-feedback  50|||--segment-len 100"
    "G_seg_150|--rl-steps 2000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.0|--reward-epochs 20|--initial-feedback  50|||--segment-len 150"

# ── Group H: Best-guess combinations ─────────────────────────────────────────
# Promising multi-axis combos using findings from the single-axis sweeps above.
# Fill in the best seg_len after Group G results; using 50 as placeholder.
    # Fine-grained phases + rich budget
    "H_fine_rich|--rl-steps 1000|--num-phases 20|--feedback-budget 1000|--uncertainty-penalty 0.0|--reward-epochs 20|--initial-feedback  50|||"
    # Very fine phases + larger budget + fresh buffer
    "H_fine_fresh|--rl-steps  500|--num-phases 40|--feedback-budget 500|--uncertainty-penalty 0.0|--reward-epochs 20|--initial-feedback  50|--feedback-buffer-size  200||"
    # Strong warmup + fine phases
    "H_warmup_fine|--rl-steps 1000|--num-phases 20|--feedback-budget 500|--uncertainty-penalty 0.0|--reward-epochs 20|--initial-feedback 500|||"
    # Generous: budget + phases + epochs + warmup
    "H_generous|--rl-steps 1000|--num-phases 20|--feedback-budget 1000|--uncertainty-penalty 0.0|--reward-epochs 30|--initial-feedback 100|||"
    # Short segment + fine phases (bet: short seg better for dense manipulation)
    "H_short_seg_fine|--rl-steps 1000|--num-phases 20|--feedback-budget 500|--uncertainty-penalty 0.0|--reward-epochs 20|--initial-feedback  50|||--segment-len  25"

# ── Group I: Feedback type combinations ───────────────────────────────────────
# Question: Which feedback types contribute signal vs noise?
# Fixed: phases=10, steps=2000, budget=500, seg=50, epochs=20, initial=50
# "demonstrative" requires expert models (auto-enabled when --expert-model-path present).
# Uses SEEDS_FB (2 seeds) to limit job count.
    "I_fb_eval      |--rl-steps 2000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.0|--reward-epochs 20|--initial-feedback  50||--feedback-types evaluative|"
    "I_fb_comp      |--rl-steps 2000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.0|--reward-epochs 20|--initial-feedback  50||--feedback-types comparative|"
    "I_fb_desc      |--rl-steps 2000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.0|--reward-epochs 20|--initial-feedback  50||--feedback-types descriptive|"
    "I_fb_demo      |--rl-steps 2000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.0|--reward-epochs 20|--initial-feedback  50||--feedback-types demonstrative|"
    "I_fb_eval_comp |--rl-steps 2000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.0|--reward-epochs 20|--initial-feedback  50||--feedback-types evaluative comparative|"
    "I_fb_eval_demo |--rl-steps 2000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.0|--reward-epochs 20|--initial-feedback  50||--feedback-types evaluative demonstrative|"
    "I_fb_comp_demo |--rl-steps 2000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.0|--reward-epochs 20|--initial-feedback  50||--feedback-types comparative demonstrative|"
    "I_fb_no_demo   |--rl-steps 2000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.0|--reward-epochs 20|--initial-feedback  50||--feedback-types evaluative comparative descriptive|"
    "I_fb_all       |--rl-steps 2000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.0|--reward-epochs 20|--initial-feedback  50||--feedback-types evaluative comparative descriptive demonstrative|"
    "I_fb_all_fine  |--rl-steps 1000|--num-phases 20|--feedback-budget 500|--uncertainty-penalty 0.0|--reward-epochs 20|--initial-feedback  50||--feedback-types evaluative comparative descriptive demonstrative|"

)

# ── Helpers ───────────────────────────────────────────────────────────────────
DRY_RUN=false
ONLY_LABEL=""
ONLY_SEED=""
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --label=*) ONLY_LABEL="${arg#--label=}" ;;
        --seed=*)  ONLY_SEED="${arg#--seed=}" ;;
    esac
done

mkdir -p "$LOG_DIR"

ACCOUNT_FLAG=""
if [[ -n "$ACCOUNT" ]]; then ACCOUNT_FLAG="#SBATCH --account=$ACCOUNT"; fi

CONDA_INIT=""
if [[ -n "$CONDA_ENV" ]]; then
    CONDA_INIT="source \"\$(conda info --base)/etc/profile.d/conda.sh\" && conda activate $CONDA_ENV"
fi

n_submitted=0

# ── Submit ────────────────────────────────────────────────────────────────────
for config_str in "${CONFIGS[@]}"; do
    IFS='|' read -r label rl_steps_arg num_phases_arg budget_arg penalty_arg epochs_arg initial_arg buffer_arg fb_types_arg seg_len_arg <<< "$config_str"

    # Strip leading/trailing whitespace from label (some have padding for alignment)
    label="$(echo "$label" | xargs)"

    # Group I (feedback type combos) uses fewer seeds to stay under ~100 total jobs
    if [[ "$label" == I_* ]]; then
        seed_list=("${SEEDS_FB[@]}")
    else
        seed_list=("${SEEDS[@]}")
    fi

    # Filter by --label / --seed if provided
    [[ -n "$ONLY_LABEL" && "$label" != "$ONLY_LABEL" ]] && continue

    for SEED in "${seed_list[@]}"; do
        [[ -n "$ONLY_SEED" && "$SEED" != "$ONLY_SEED" ]] && continue
        EXP_NAME="${ENV//metaworld-/mw_}_${ALGO}_${label}_s${SEED}"

        JOB_SCRIPT=$(cat <<SLURM
#!/bin/bash
#SBATCH --job-name=rlhf_${label}_s${SEED}
#SBATCH --partition=${PARTITION}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${LOG_DIR}/${EXP_NAME}_%j.out
#SBATCH --error=${LOG_DIR}/${EXP_NAME}_%j.err
${ACCOUNT_FLAG}

set -euo pipefail
cd "${REPO_ROOT}"
${CONDA_INIT}

python scripts/run_simulated_phases.py \\
    --env               "${ENV}" \\
    --algorithm         "${ALGO}" \\
    --expert-algorithm  "${EXPERT_ALGO}" \\
    --expert-model-path "${EXPERT_MODEL_PATH}" \\
    --exp-name          "${EXP_NAME}" \\
    --seed              ${SEED} \\
    --device            "${DEVICE}" \\
    --max-episode-steps ${MAX_EPISODE_STEPS} \\
    --n-trajectories    ${N_TRAJECTORIES} \\
    --segment-len       ${SEGMENT_LEN} \\
    ${seg_len_arg} \\
    --fix-start-state \\
    --state-seed        ${STATE_SEED} \\
    --skip-projections \\
    ${rl_steps_arg} \\
    ${num_phases_arg} \\
    ${budget_arg} \\
    ${penalty_arg} \\
    ${epochs_arg} \\
    ${initial_arg} \\
    ${buffer_arg} \\
    ${fb_types_arg}
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
    echo "Dry run: would submit $n_submitted jobs (Groups A–H: ${#SEEDS[@]} seeds; Group I: ${#SEEDS_FB[@]} seeds)."
else
    echo "Submitted $n_submitted jobs. Logs → $LOG_DIR/"
fi
