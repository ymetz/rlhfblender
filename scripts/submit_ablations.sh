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

# ── Group A: RL steps per phase ───────────────────────────────────────────────
# Question: How much does per-phase exploitation matter?
# Fixed: phases=10, budget=500, penalty=0.0, epochs=20, initial=50
    "A_steps_0500|--rl-steps    500|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.0|--reward-epochs 20|--initial-feedback  50|"
    "A_steps_1000|--rl-steps   1000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.0|--reward-epochs 20|--initial-feedback  50|"
    "A_steps_2000|--rl-steps   2000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.0|--reward-epochs 20|--initial-feedback  50|"
    "A_steps_5000|--rl-steps   5000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.0|--reward-epochs 20|--initial-feedback  50|"
    "A_steps_10k |--rl-steps  10000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.0|--reward-epochs 20|--initial-feedback  50|"

# ── Group B: Phase granularity (total RL = 20k steps) ────────────────────────
# Question: More-frequent reward model updates vs fewer, longer training bursts?
# Fixed: total_rl=20k, budget=500, penalty=0.0, epochs=20, initial=50
    "B_grain_p05_s4000|--rl-steps  4000|--num-phases  5|--feedback-budget 500|--uncertainty-penalty 0.0|--reward-epochs 20|--initial-feedback  50|"
    # A_steps_2000 / phases=10 / steps=2000  ← already in Group A (20k total)
    "B_grain_p20_s1000|--rl-steps  1000|--num-phases 20|--feedback-budget 500|--uncertainty-penalty 0.0|--reward-epochs 20|--initial-feedback  50|"
    "B_grain_p40_s0500|--rl-steps   500|--num-phases 40|--feedback-budget 500|--uncertainty-penalty 0.0|--reward-epochs 20|--initial-feedback  50|"

# ── Group C: Uncertainty penalty ─────────────────────────────────────────────
# Question: Does penalising ensemble disagreement prevent reward hacking?
# Fixed: phases=10, steps=2000, budget=500, epochs=20, initial=50
    "C_pen_0.05|--rl-steps 2000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.05|--reward-epochs 20|--initial-feedback  50|"
    "C_pen_0.10|--rl-steps 2000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.10|--reward-epochs 20|--initial-feedback  50|"
    "C_pen_0.20|--rl-steps 2000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.20|--reward-epochs 20|--initial-feedback  50|"
    "C_pen_0.30|--rl-steps 2000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.30|--reward-epochs 20|--initial-feedback  50|"
    "C_pen_0.50|--rl-steps 2000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.50|--reward-epochs 20|--initial-feedback  50|"

# ── Group D: Feedback budget ──────────────────────────────────────────────────
# Question: More oracle data per run → better reward model?
# Fixed: phases=10, steps=2000, penalty=0.1, epochs=20, initial=50
    "D_budget_0250|--rl-steps 2000|--num-phases 10|--feedback-budget  250|--uncertainty-penalty 0.1|--reward-epochs 20|--initial-feedback  50|"
    # budget=500 covered by C_pen_0.10
    "D_budget_0750|--rl-steps 2000|--num-phases 10|--feedback-budget  750|--uncertainty-penalty 0.1|--reward-epochs 20|--initial-feedback  50|"
    "D_budget_1000|--rl-steps 2000|--num-phases 10|--feedback-budget 1000|--uncertainty-penalty 0.1|--reward-epochs 20|--initial-feedback  50|"
    "D_budget_1500|--rl-steps 2000|--num-phases 10|--feedback-budget 1500|--uncertainty-penalty 0.1|--reward-epochs 20|--initial-feedback  50|"

# ── Group E: Reward model training epochs ────────────────────────────────────
# Question: How much does reward model training depth matter?
# Fixed: phases=10, steps=2000, budget=500, penalty=0.1, initial=50
    "E_epochs_05|--rl-steps 2000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.1|--reward-epochs  5|--initial-feedback  50|"
    "E_epochs_10|--rl-steps 2000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.1|--reward-epochs 10|--initial-feedback  50|"
    # epochs=20 covered by C_pen_0.10
    "E_epochs_30|--rl-steps 2000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.1|--reward-epochs 30|--initial-feedback  50|"
    "E_epochs_50|--rl-steps 2000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.1|--reward-epochs 50|--initial-feedback  50|"

# ── Group F: Initial feedback count (phase 0 warmup) ─────────────────────────
# Question: Does a better phase-0 reward model bootstrap subsequent phases?
# Fixed: phases=10, steps=2000, budget=500, penalty=0.1, epochs=20
    "F_init_0025|--rl-steps 2000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.1|--reward-epochs 20|--initial-feedback   25|"
    # initial=50 covered by C_pen_0.10
    "F_init_0100|--rl-steps 2000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.1|--reward-epochs 20|--initial-feedback  100|"
    "F_init_0250|--rl-steps 2000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.1|--reward-epochs 20|--initial-feedback  250|"
    "F_init_0500|--rl-steps 2000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.1|--reward-epochs 20|--initial-feedback  500|"

# ── Group G: Feedback buffer size ─────────────────────────────────────────────
# Question: Should old (potentially off-distribution) feedback expire faster?
# Fixed: phases=10, steps=2000, budget=500, penalty=0.1, epochs=20, initial=50
    "G_buf_100|--rl-steps 2000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.1|--reward-epochs 20|--initial-feedback  50|--feedback-buffer-size  100"
    "G_buf_250|--rl-steps 2000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.1|--reward-epochs 20|--initial-feedback  50|--feedback-buffer-size  250"
    # no buffer flag = keep all (budget=500 used by C_pen_0.10)

# ── Group H: Best-guess combinations ─────────────────────────────────────────
# Promising multi-axis combinations informed by the above groups
    # Fine-grained + penalty: frequent updates to limit hacking, penalty as backup
    "H_fine_pen|--rl-steps 1000|--num-phases 20|--feedback-budget 500|--uncertainty-penalty 0.2|--reward-epochs 20|--initial-feedback  50|"
    # Rich oracle data + fine phases
    "H_rich_fine|--rl-steps 1000|--num-phases 20|--feedback-budget 1000|--uncertainty-penalty 0.1|--reward-epochs 20|--initial-feedback  50|"
    # Max conservative: very short exploitation + strong penalty + fresh buffer
    "H_max_safe|--rl-steps  500|--num-phases 40|--feedback-budget 500|--uncertainty-penalty 0.3|--reward-epochs 20|--initial-feedback  50|--feedback-buffer-size  200"
    # Strong warmup: lots of initial feedback so phase-0 reward model is solid
    "H_warmup|--rl-steps 2000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.1|--reward-epochs 20|--initial-feedback 500|"
    # Generous everything: budget + phases + penalty + good warmup
    "H_generous|--rl-steps 1000|--num-phases 20|--feedback-budget 1000|--uncertainty-penalty 0.2|--reward-epochs 30|--initial-feedback 100|"

# ── Group I: Feedback type combinations ───────────────────────────────────────
# Question: Which feedback types contribute signal vs noise?
# Fixed: phases=10, steps=2000, budget=500, penalty=0.0, epochs=20, initial=50
# Note: "demonstrative" requires expert models (enabled when --expert-model-path is given)
# Uses SEEDS_FB (2 seeds) to stay within ~100 total jobs.
# Single-type runs establish individual signal; combo runs test complementarity.
    "I_fb_eval       |--rl-steps 2000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.0|--reward-epochs 20|--initial-feedback  50||--feedback-types evaluative"
    "I_fb_comp       |--rl-steps 2000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.0|--reward-epochs 20|--initial-feedback  50||--feedback-types comparative"
    "I_fb_desc       |--rl-steps 2000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.0|--reward-epochs 20|--initial-feedback  50||--feedback-types descriptive"
    "I_fb_demo       |--rl-steps 2000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.0|--reward-epochs 20|--initial-feedback  50||--feedback-types demonstrative"
    "I_fb_eval_comp  |--rl-steps 2000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.0|--reward-epochs 20|--initial-feedback  50||--feedback-types evaluative comparative"
    "I_fb_eval_demo  |--rl-steps 2000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.0|--reward-epochs 20|--initial-feedback  50||--feedback-types evaluative demonstrative"
    "I_fb_comp_demo  |--rl-steps 2000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.0|--reward-epochs 20|--initial-feedback  50||--feedback-types comparative demonstrative"
    "I_fb_no_demo    |--rl-steps 2000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.0|--reward-epochs 20|--initial-feedback  50||--feedback-types evaluative comparative descriptive"
    "I_fb_all        |--rl-steps 2000|--num-phases 10|--feedback-budget 500|--uncertainty-penalty 0.0|--reward-epochs 20|--initial-feedback  50||--feedback-types evaluative comparative descriptive demonstrative"
    # All types + conservative settings: best of both worlds
    "I_fb_all_safe   |--rl-steps 1000|--num-phases 20|--feedback-budget 500|--uncertainty-penalty 0.2|--reward-epochs 20|--initial-feedback  50||--feedback-types evaluative comparative descriptive demonstrative"

)

# ── Helpers ───────────────────────────────────────────────────────────────────
DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=true; fi

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
    IFS='|' read -r label rl_steps_arg num_phases_arg budget_arg penalty_arg epochs_arg initial_arg buffer_arg fb_types_arg <<< "$config_str"

    # Strip leading/trailing whitespace from label (some have padding for alignment)
    label="$(echo "$label" | xargs)"

    # Group I (feedback type combos) uses fewer seeds to stay under ~100 total jobs
    if [[ "$label" == I_* ]]; then
        seed_list=("${SEEDS_FB[@]}")
    else
        seed_list=("${SEEDS[@]}")
    fi

    for SEED in "${seed_list[@]}"; do
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
