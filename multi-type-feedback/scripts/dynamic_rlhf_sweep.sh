#!/bin/bash

# Single environment for faster sweeping
envs=("HalfCheetah-v5")
seeds=(330 1687123 1789 330 1687123) #(1789 12 912391 330 1687123)
# Basic feedback types for initial sweep
feedback_types=("evaluative comparative demonstrative corrective descriptive descriptive_preference" "evaluative comparative demonstrative corrective")
#feedback_types=("evaluative" "comparative" "demonstrative" "corrective")
#feedback_types=("supervised")
# Hyperparameter ranges
n_feedback_per_iteration=(50)
reward_training_epochs=(15)
feedback_buffer_size=(5000)

# Create a directory for log files and scripts if they don't exist
mkdir -p logs
mkdir -p job_scripts

# Counter for job scripts
job_counter=0

# Generate all combinations and create individual job scripts
for seed in "${seeds[@]}"; do
    for env in "${envs[@]}"; do
        for feedback in "${feedback_types[@]}"; do
            for n_feedback in "${n_feedback_per_iteration[@]}"; do
                for epochs in "${reward_training_epochs[@]}"; do
                    for buffer_size in "${feedback_buffer_size[@]}"; do
                        # Create a unique job script for this combination
                        job_script="job_scripts/job_${job_counter}.sh"
                        
                        # Create the job script with SLURM directives
                        cat <<EOT > $job_script
#!/bin/bash
#SBATCH --partition=single
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --job-name=hp_sweep_${job_counter}
#SBATCH --time=04:30:00
#SBATCH --output=logs/hp_sweep_${job_counter}_%j.out

# Load necessary modules or activate environments
source /pfs/data5/home/kn/kn_kn/kn_pop257914/ws_feedback_querying/venv/bin/activate

# Set environment variable to force CPU usage
export CUDA_VISIBLE_DEVICES=""

# Run the training job
python multi_type_feedback/dynamic_rlhf.py \
    --algorithm ppo \
    --environment $env \
    --feedback-types $feedback \
    --reward-model-type unified \
    --seed $seed \
    --n-feedback-per-iteration $n_feedback \
    --reward-training-epochs $epochs \
    --feedback-buffer-size $buffer_size \
    --reference-data-folder ../multi-type-feedback_iclr2025/rlhf/feedback \
    --expert-model-base-path ../multi-type-feedback_iclr2025/main/gt_agents \
    --wandb-project-name dynamic_rlhf_mujoco_fixed_length
EOT

                        # Make the job script executable
                        chmod +x $job_script
                        
                        # Submit the job
                        sbatch $job_script
                        
                        # Increment the counter
                        ((job_counter++))
                    done
                done
            done
        done
    done
done

echo "All $job_counter jobs have been submitted."