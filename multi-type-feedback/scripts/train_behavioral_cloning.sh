#!/bin/bash

envs=("Ant-v5" "Hopper-v5" "Humanoid-v5" "HalfCheetah-v5" "Walker2d-v5" "Swimmer-v5")
#envs=("metaworld-button-press-v2" "metaworld-sweep-into-v2" "metaworld-pick-place-v2")
#envs=("merge-v0" "highway-fast-v0" "roundabout-v0")
#seeds=(1789 1687123 12 912391 330)
seeds=(1789 1687123 12)
noise_levels=(0.0)
#algs=("sac" "sac" "sac" "ppo" "ppo" "ppo")
algs=("sac" "sac" "sac" "ppo" "ppo" "ppo")
n_feedbacks=(5000 2500 1250 750)

# Create a directory for log files if it doesn't exist
mkdir -p logs

# Prepare an array to hold all combinations
declare -a combinations

# Generate all combinations
for seed in "${seeds[@]}"; do
    for i in "${!envs[@]}"; do
        for noise in "${noise_levels[@]}"; do
            for n_feedback in "${n_feedbacks[@]}"; do
                combinations+=("$seed ${envs[$i]} $noise ${algs[$i]} $n_feedback")
            done
        done
    done
done

# Set the batch size (number of jobs per GPU)
batch_size=4
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
#SBATCH --cpus-per-task=10
#SBATCH --ntasks=1
#SBATCH --job-name=train_behavioral_cloning_$batch_id
#SBATCH --time=02:30:00
#SBATCH --output=logs/train_behavioral_cloning_${batch_id}_%j.out

# Load any necessary modules or activate environments here
source /pfs/data5/home/kn/kn_kn/kn_pop257914/multi-type-feedback/venv/bin/activate

# Run the training jobs in background
EOT

    # Add each task to the Slurm script
    for combination in "${batch[@]}"; do
        read seed env noise algo n_feedback <<< $combination
        echo "python multi_type_feedback/train_bc.py --environment $env --seed $seed --algo $algo --n-feedback $n_feedback --noise-level $noise &" >> $sbatch_script
    done

    # Wait for all background jobs to finish
    echo "wait" >> $sbatch_script

    # Submit the Slurm job script
    sbatch $sbatch_script

    # Optional: Remove the temporary Slurm script
    rm $sbatch_script
done

echo "All jobs have been submitted."