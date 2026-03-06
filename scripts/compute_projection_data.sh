#!/bin/bash

# Exit on any error
set -e

# Default values
EXPERIMENT_NAME="Metaworld-With-States"
ENV_NAME="metaworld-sweep-into-v3"
CHECKPOINTS=(0)
PROJECTION_METHOD="PCA"
OUTPUT_DIR="data/saved_projections"
NUM_EPISODES=10
ALGORITHM="ppo"
EXP_ID=""
FEEDBACK_TYPE="evaluative"
REWARD_MODEL_DIR="multi-type-feedback/reward_models/checkpoints"
POLICY_MODEL_DIR="multi-type-feedback/train_baselines/dynamic_rlhf_agents"
REWARD_MODEL_TYPE="separate"
# Legacy static paths (used only when --exp-id is not supplied)
EXPER_MODEL_PATH="multi-type-feedback/train_baselines/gt_agents/ppo/metaworld-sweep-into-v3_1"
STATIC_REWARD_MODEL="multi-type-feedback/reward_models/ppo_metaworld-sweep-into-v3_12_evaluative_12.ckpt"
STATIC_POLICY_MODEL="multi-type-feedback/train_baselines/gt_agents/ppo/metaworld-sweep-into-v3_1/best_model.zip"
ADDITIONAL_GYM_PACKAGES="metaworld"

# Help function
function show_help {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -e,   --experiment-name NAME     Experiment name (default: $EXPERIMENT_NAME)"
    echo "  -env, --environment NAME         RL environment name (default: $ENV_NAME)"
    echo "  -c,   --checkpoints LIST         Space-separated checkpoints (default: ${CHECKPOINTS[0]})"
    echo "  -p,   --projection-method METHOD PCA | TSNE | UMAP (default: $PROJECTION_METHOD)"
    echo "  -n,   --num-episodes NUM         Episodes to run for data gen (default: $NUM_EPISODES)"
    echo "  -a,   --algorithm ALGO           ppo | sac (default: $ALGORITHM)"
    echo "  -id,  --exp-id ID                Experiment DB id (enables dynamic model paths)"
    echo "  -ft,  --feedback-type TYPE       Reward model feedback type (default: $FEEDBACK_TYPE)"
    echo "  -rmd, --reward-model-dir DIR     Directory containing reward .ckpt files"
    echo "  -pmd, --policy-model-dir DIR     Directory containing policy .zip files"
    echo "  -rmt, --reward-model-type TYPE   separate | unified (default: $REWARD_MODEL_TYPE)"
    echo "  -h,   --help                     Show this help"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -e|--experiment-name)
            EXPERIMENT_NAME="$2"; shift 2 ;;
        -env|--environment)
            ENV_NAME="$2"; shift 2 ;;
        -c|--checkpoints)
            IFS=' ' read -ra CHECKPOINTS <<< "$2"; shift 2 ;;
        -p|--projection-method)
            PROJECTION_METHOD="$2"; shift 2 ;;
        -n|--num-episodes)
            NUM_EPISODES="$2"; shift 2 ;;
        -a|--algorithm)
            ALGORITHM="$2"; shift 2 ;;
        -id|--exp-id)
            EXP_ID="$2"; shift 2 ;;
        -ft|--feedback-type)
            FEEDBACK_TYPE="$2"; shift 2 ;;
        -rmd|--reward-model-dir)
            REWARD_MODEL_DIR="$2"; shift 2 ;;
        -pmd|--policy-model-dir)
            POLICY_MODEL_DIR="$2"; shift 2 ;;
        -rmt|--reward-model-type)
            REWARD_MODEL_TYPE="$2"; shift 2 ;;
        -h|--help)
            show_help ;;
        *)
            echo "Unknown option: $1"; show_help ;;
    esac
done

# Derive the underscored env name used in checkpoint filenames
# e.g. metaworld-sweep-into-v3 → metaworld_sweep_into_v3
ENV_UNDERSCORED=$(echo "$ENV_NAME" | tr '-' '_')

echo "========================================================="
echo "Experiment : $EXPERIMENT_NAME"
echo "Environment: $ENV_NAME  (underscored: $ENV_UNDERSCORED)"
echo "Algorithm  : $ALGORITHM"
echo "Exp ID     : ${EXP_ID:-<static paths>}"
echo "Checkpoints: ${CHECKPOINTS[*]}"
echo "Projection : $PROJECTION_METHOD"
echo "Feedback   : $FEEDBACK_TYPE  (reward model type: $REWARD_MODEL_TYPE)"
echo "========================================================="

# ---------------------------------------------------------------------------
# Joint projection (optional – uncomment if you have a trained reference agent)
# ---------------------------------------------------------------------------
: '
python rlhfblender/generate_data.py \
  --exp "$EXPERIMENT_NAME" \
  --env "$ENV_NAME" \
  --checkpoints "${CHECKPOINTS[@]}" \
  --model-path "$EXPER_MODEL_PATH" \
  --num-episodes "$NUM_EPISODES" \
  --additional-gym-packages "$ADDITIONAL_GYM_PACKAGES" \
  --env-kwargs camera_name:corner

python scripts/compute_joint_obs_state_projection.py \
  --experiment-name "$EXPERIMENT_NAME" \
  --checkpoints "${CHECKPOINTS[@]}" \
  --projection-method "$PROJECTION_METHOD" \
  --additional-gym-packages "$ADDITIONAL_GYM_PACKAGES" \
  --state-epochs 500
'

# Look for an existing joint projection to use as shared reference space
JOINT_PROJECTION_PATH=""
JOINT_METADATA_PATTERN="data/saved_projections/joint_obs_state/*_joint_obs_state_${PROJECTION_METHOD}_*_metadata.json"
JOINT_PROJECTION_PATH=$(ls $JOINT_METADATA_PATTERN 2>/dev/null | tail -1)
if [ -z "$JOINT_PROJECTION_PATH" ]; then
    JOINT_OBS_PATTERN="data/saved_projections/joint/*_joint_${PROJECTION_METHOD}_*_metadata.json"
    JOINT_PROJECTION_PATH=$(ls $JOINT_OBS_PATTERN 2>/dev/null | tail -1)
fi
if [ -n "$JOINT_PROJECTION_PATH" ]; then
    echo "Using joint projection: $JOINT_PROJECTION_PATH"
else
    echo "No joint projection found – individual projections will be computed independently."
fi

# ---------------------------------------------------------------------------
# Per-checkpoint projection + reward/uncertainty prediction
# ---------------------------------------------------------------------------
for CHECKPOINT in "${CHECKPOINTS[@]}"; do
    echo ""
    echo "========================================================="
    echo "Processing checkpoint: $CHECKPOINT"
    echo "========================================================="

    # ------------------------------------------------------------------
    # Resolve reward model and policy model paths
    # ------------------------------------------------------------------
    if [ -n "$EXP_ID" ]; then
        # Dynamic paths derived from the simulated training naming convention:
        #   {algo}_{env_underscored}_{exp_id}_{feedback_type}_{ckpt}.ckpt
        #   {algo}_{env_underscored}_{exp_id}_{ckpt}.zip
        CKPT_REWARD_MODEL="${REWARD_MODEL_DIR}/${ALGORITHM}_${ENV_UNDERSCORED}_${EXP_ID}_${FEEDBACK_TYPE}_${CHECKPOINT}.ckpt"
        CKPT_POLICY_MODEL="${POLICY_MODEL_DIR}/${ALGORITHM}_${ENV_UNDERSCORED}_${EXP_ID}_${CHECKPOINT}.zip"

        # Fall back to the highest available checkpoint if this one is missing
        if [ ! -f "$CKPT_REWARD_MODEL" ]; then
            echo "  Reward model for checkpoint $CHECKPOINT not found, looking for latest available..."
            CKPT_REWARD_MODEL=$(ls "${REWARD_MODEL_DIR}/${ALGORITHM}_${ENV_UNDERSCORED}_${EXP_ID}_${FEEDBACK_TYPE}_"*.ckpt 2>/dev/null | sort -V | tail -1)
        fi
        if [ ! -f "$CKPT_POLICY_MODEL" ]; then
            echo "  Policy model for checkpoint $CHECKPOINT not found, looking for latest available..."
            CKPT_POLICY_MODEL=$(ls "${POLICY_MODEL_DIR}/${ALGORITHM}_${ENV_UNDERSCORED}_${EXP_ID}_"*.zip 2>/dev/null | sort -V | tail -1)
        fi

        REWARD_MODEL="$CKPT_REWARD_MODEL"
        POLICY_MODEL="$CKPT_POLICY_MODEL"
    else
        # Legacy: use the static paths supplied at the top of the script
        REWARD_MODEL="$STATIC_REWARD_MODEL"
        POLICY_MODEL="$STATIC_POLICY_MODEL"
    fi

    echo "  Reward model : ${REWARD_MODEL:-<none>}"
    echo "  Policy model : ${POLICY_MODEL:-<none>}"

    # ------------------------------------------------------------------
    # Step 1: Generate projections
    # ------------------------------------------------------------------
    echo "Step 1: Generating projections..."
    if [ -n "$JOINT_PROJECTION_PATH" ]; then
        python rlhfblender/projections/generate_projections.py \
          --experiment-name "$EXPERIMENT_NAME" \
          --compute-inverse \
          --auto-grid-range \
          --checkpoint "$CHECKPOINT" \
          --projection-method "$PROJECTION_METHOD" \
          --joint-projection-path "$JOINT_PROJECTION_PATH"
    else
        python rlhfblender/projections/generate_projections.py \
          --experiment-name "$EXPERIMENT_NAME" \
          --compute-inverse \
          --auto-grid-range \
          --checkpoint "$CHECKPOINT" \
          --projection-method "$PROJECTION_METHOD"
    fi

    # ------------------------------------------------------------------
    # Step 2: Generate reward/uncertainty predictions
    # ------------------------------------------------------------------
    if [ -n "$REWARD_MODEL" ] && [ -f "$REWARD_MODEL" ] && \
       [ -n "$POLICY_MODEL" ] && [ -f "$POLICY_MODEL" ]; then
        echo "Step 2: Generating reward/uncertainty predictions..."
        python rlhfblender/projections/predict_reward_and_uncertainty.py \
          --experiment-name "$EXPERIMENT_NAME" \
          --checkpoint "$CHECKPOINT" \
          --reward-model "$REWARD_MODEL" \
          --reward-model-type "$REWARD_MODEL_TYPE" \
          --projection-method "$PROJECTION_METHOD" \
          --output-dir "$OUTPUT_DIR" \
          --policy-algorithm "$ALGORITHM" \
          --policy-model "$POLICY_MODEL"
    else
        echo "Step 2: Skipping reward/uncertainty prediction (model files not found)."
    fi
done

echo ""
echo "All tasks completed successfully!"
