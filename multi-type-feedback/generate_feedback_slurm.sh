#!/bin/bash

# Set the experiment parameters
#envs=("Swimmer-v5" "HalfCheetah-v5" "Walker2d-v5")
#envs=("Swimmer-v5" "HalfCheetah-v5" "Walker2d-v5")
envs=("LunarLander-v3" "Acrobat-v1" "MountainCar-v0")
seeds=(1789 1687123 12 912391 330)

# Create a directory for log files if it doesn't exist
mkdir -p logs

# Prepare an array to hold all combinations
declare -a combinations

# Generate all combinations
for seed in "${seeds[@]}"; do
    for env in "${envs[@]}"; do
        combinations+=("$seed $env")
    done
done

# Set the batch size (number of jobs per GPU)
batch_size=5
total_combinations=${#combinations[@]}

# Loop over the combinations in batches
for ((i=0; i<$total_combinations; i+=$batch_size)); do
    batch=("${combinations[@]:$i:$batch_size}")
    batch_id=$((i / batch_size))

    # Create a temporary Slurm job script for this batch
    sbatch_script="batch_job_$batch_id.sh"
    cat <<EOT > $sbatch_script
#!/bin/bash
#SBATCH --partition=gpu_4,gpu_8,gpu_4_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --ntasks=1
#SBATCH --job-name=generate_feedback_$batch_id
#SBATCH --time=02:30:00
#SBATCH --output=logs/train_generate_feedback_${batch_id}_%j.out

# Load any necessary modules or activate environments here
# module load python/3.9

# Run the training jobs in background
EOT

    # Add each task to the Slurm script
    for combination in "${batch[@]}"; do
        read seed env <<< $combination
        echo "python multi_type_feedback/generate_feedback.py --algorithm ppo --environment $env --seed $seed --n-feedback 10000 --save-folder feedback &" >> $sbatch_script
    done

    # Wait for all background jobs to finish
    echo "wait" >> $sbatch_script

    # Submit the Slurm job script
    sbatch $sbatch_script

    # Optional: Remove the temporary Slurm script
    rm $sbatch_script
done

echo "All jobs have been submitted."
