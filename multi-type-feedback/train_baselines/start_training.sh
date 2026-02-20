#!/bin/bash

# Set the experiment parameters
envs=("Ant-v5" "Swimmer-v5" "HalfCheetah-v5" "Hopper-v5" "procgen-coinrun-v0" "procgen-miner-v0" "ALE/MsPacman-v5" "ALE/BeamRider-v5" "ALE/Enduro-v5" "ALE/Pong-v5")
cuda_devices=(0 0 0 1 1 2 2 3 3 4 4)
save_freqs=(50000 50000 50000 50000 1250000 1250000 500000 500000 500000 500000)
seeds=(1789 1687123 12 912391 330)

# Loop over the environments and CUDA devices
for seed in "${!seeds[@]}"; do
    for i in "${!envs[@]}"; do
      export CUDA_VISIBLE_DEVICES=${cuda_devices[$i]}
      echo "Starting training for ${envs[$i]} with CUDA_VISIBLE_DEVICES=${cuda_devices[$i]}"
      python train_baselines/train.py --algo ppo --env ${envs[$i]} --verbose 0 --save-freq ${save_freqs[$i]} --seed ${seeds[$seed]} --gym-packages procgen ale_py --log-folder gt_agents  &
    done
    
    # Wait for all training processes to finish
    wait
done

echo "All training runs completed."

python train_baselines/benchmark_env.py --algo ppo --model-base-path gt_agents