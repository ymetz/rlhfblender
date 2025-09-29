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
EXPER_MODEL_PATH="multi-type-feedback/train_baselines/gt_agents/ppo/metaworld-sweep-into-v3_1"
REWARD_MODEL="multi-type-feedback/reward_models/ppo_metaworld-sweep-into-v3_12_evaluative_12.ckpt"
POLICY_MODEL="multi-type-feedback/train_baselines/gt_agents/ppo/metaworld-sweep-into-v3_1/best_model.zip"
ADDITIONAL_GYM_PACKAGES="metaworld"

# Help function
function show_help {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -e, --experiment-name NAME   Specify experiment name (default: $EXPERIMENT_NAME)"
    echo "  -env, --environment NAME     Specify name of RL environment"
    echo "  -c, --checkpoints LIST       Space-separated list of checkpoints (default: ${CHECKPOINTS[0]})"
    echo "  -p, --projection-method METHOD  Projection method to use (default: $PROJECTION_METHOD)"
    echo "  -n, --num-episodes NUM       Number of episodes to run (default: $NUM_EPISODES)"
    echo "  -h, --help                   Show this help message"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -e|--experiment-name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        -env|--environment)
            ENV_NAME="$2"
            shift 2
            ;;
        -c|--checkpoints)
            IFS=' ' read -ra CHECKPOINTS <<< "$2"
            shift 2
            ;;
        -p|--projection-method)
            PROJECTION_METHOD="$2"
            shift 2
            ;;
        -n|--num-episodes)
            NUM_EPISODES="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

echo "Generating data for experiment: $EXPERIMENT_NAME"

echo "Generated data for experiment: $EXPERIMENT_NAME"
echo "========================================================="


echo "Compute joint observation and state projection for experiment: $EXPERIMENT_NAME"

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
  --max-trajectories 50 \
  --max-steps 100 \
  --state-epochs 20
  
echo "Generated joint projections for experiment: $EXPERIMENT_NAME"
echo "========================================================="

echo "Starting projection generation and reward prediction script..."
echo "Experiment: $EXPERIMENT_NAME"
echo "Checkpoints: ${CHECKPOINTS[*]}"
echo "Projection method: $PROJECTION_METHOD"

# Convert experiment name to lowercase and replace spaces with hyphens for filenames
EXPERIMENT_NAME_LOWER=$(echo "$EXPERIMENT_NAME" | tr '[:upper:]' '[:lower:]' | tr ' ' '-')

# Find the joint projection metadata file that was just created
JOINT_METADATA_PATTERN="data/saved_projections/joint_obs_state/*_joint_obs_state_${PROJECTION_METHOD}_*_metadata.json"
JOINT_PROJECTION_PATH=$(ls $JOINT_METADATA_PATTERN 2>/dev/null | tail -1)

if [ -z "$JOINT_PROJECTION_PATH" ]; then
    echo "Warning: No joint projection metadata found. Looking for observation-only joint projections..."
    JOINT_OBS_PATTERN="data/saved_projections/joint/*_joint_${PROJECTION_METHOD}_*_metadata.json"
    JOINT_PROJECTION_PATH=$(ls $JOINT_OBS_PATTERN 2>/dev/null | tail -1)
fi

if [ -n "$JOINT_PROJECTION_PATH" ]; then
    echo "Using joint projection: $JOINT_PROJECTION_PATH"
else
    echo "Warning: No joint projection metadata found. Individual projections will be computed without joint reference."
fi

# Process each checkpoint
for CHECKPOINT in "${CHECKPOINTS[@]}"; do
    echo "Processing checkpoint: $CHECKPOINT"
    
    # Step 1: Generate projections
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
    
    # Step 2: Generate predictions for reward and uncertainty
    echo "Step 2: Generating predictions for reward and uncertainty..."
    python rlhfblender/projections/predict_reward_and_uncertainty.py \
      --experiment-name "$EXPERIMENT_NAME" \
      --checkpoint "$CHECKPOINT" \
      --reward-model "$REWARD_MODEL" \
      --projection-method "$PROJECTION_METHOD" \
      --output-dir "$OUTPUT_DIR" \
      --policy-model "$POLICY_MODEL"
done

echo "All tasks completed successfully!"