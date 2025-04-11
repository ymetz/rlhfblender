#!/bin/bash

# Define arrays
envs=("Acrobot-v1")
seeds=(1789 1687123 12 912391 330)
save_freqs=(50000 50000 50000)

# Create a directory for log files if it doesn't exist
mkdir -p logs

# Prepare an array to hold all combinations
declare -a combinations

# Generate combinations with matched save frequencies
for seed in "${seeds[@]}"; do
    for i in "${!envs[@]}"; do
        combinations+=("$seed ${envs[$i]} ${save_freqs[$i]}")
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
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --job-name=train_agents_$batch_id
#SBATCH --time=02:00:00
#SBATCH --output=logs/train_agents_${batch_id}_%j.out

EOT
    
    # Add each task to the Slurm script
    for combination in "${batch[@]}"; do
        read seed env save_freq <<< "$combination"
        echo "python train_baselines/train.py --algo ppo --env $env --verbose 0 --save-freq $save_freq --seed $seed --log-folder train_baselines/gt_agents &" >> $sbatch_script
    done
    
    # Wait for all background jobs to finish
    echo "wait" >> $sbatch_script
    
    # Submit the Slurm job script
    sbatch $sbatch_script
    
    # Remove the temporary Slurm script
    rm $sbatch_script
done

echo "All jobs have been submitted."