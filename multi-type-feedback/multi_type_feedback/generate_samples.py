import bisect
import os
import pickle
import random
import re
from pathlib import Path
from typing import List, Type, Union

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from multi_type_feedback.utils import TrainingUtils


def one_hot_vector(k, max_val):
    """Convert action index to one-hot vector."""
    vec = np.zeros(max_val)
    np.put(vec, k, 1)
    return vec


def create_segments(arr, start_indices, done_indices, segment_length):
    """
    Creates array segments with target length (segment_length),
    selects the longest contiguous array within a segment [start_indices: start_indices+segment_length]
    """
    segments = []

    for start in start_indices:
        end = start + segment_length

        # Find the position of the first done_index greater than or equal to start
        insert_pos = bisect.bisect_left(done_indices, start)

        # Collect all done_indices within the range [start, end)
        relevant_done_indices = []
        while insert_pos < len(done_indices) and done_indices[insert_pos] < end:
            relevant_done_indices.append(done_indices[insert_pos])
            insert_pos += 1

        # If there are no relevant done_indices, take the full segment
        if not relevant_done_indices:
            segment = arr[start:end]
        else:
            # Consider the segment before the first done_index
            longest_segment = arr[start : relevant_done_indices[0]]

            # Consider segments between each pair of consecutive done_indices
            for i in range(len(relevant_done_indices) - 1):
                segment = arr[relevant_done_indices[i] : relevant_done_indices[i + 1]]
                if len(segment) > len(longest_segment):
                    longest_segment = segment

            # Consider the segment after the last done_index
            segment = arr[relevant_done_indices[-1] : end]
            if len(segment) > len(longest_segment):
                longest_segment = segment

            # Use the longest valid segment
            segment = longest_segment

        segments.append(segment)

    return segments


def discounted_sum_numpy(rewards, gamma):
    """Calculate discounted sum of rewards."""
    return np.sum(rewards * (gamma ** np.arange(len(rewards))))


def generate_samples(
    model_class: Type[Union[PPO, SAC]],
    expert_models: List[Union[PPO, SAC]],
    environment: gym.Env,
    environment_name: str = "HalfCheetah-v5",
    checkpoints_path: str = "gt_agents",
    total_steps_factor: int = 5,
    n_samples: int = 1000,
    segment_len: int = 50,
    algorithm: str = "ppo",
    device: str = "cuda",
    action_one_hot: bool = False,
    random_sample: bool = False,
) -> dict:
    """Generate agent's observations and samples in the training environment."""

    env_name = environment_name if "ALE" not in environment_name else environment_name.replace("/", "-")

    if not random_sample:
        # Get available checkpoint indices
        possible_checkpoint_indices = [
            str(model_dir.split("_")[-1])
            for model_dir in os.listdir(os.path.join(checkpoints_path, algorithm))
            if env_name in model_dir
        ]

        checkpoints_dir = os.path.join(checkpoints_path, algorithm, f"{env_name}_1")

        print(f"Generating samples for: {algorithm}_{env_name}")

        # Get checkpoint files
        checkpoint_files = [file for file in os.listdir(checkpoints_dir) if re.search(r"rl_model_.*\.zip", file)] or [
            f"{env_name}.zip"
        ]

        total_steps = n_samples * total_steps_factor
        num_checkpoints = len(checkpoint_files) + 1
        steps_per_checkpoint = total_steps // num_checkpoints
        samples_per_checkpoint = n_samples // num_checkpoints
        gamma = expert_models[0][0].gamma

        checkpoint_files = ["random"] + sorted(checkpoint_files, key=lambda x: int(re.search(r"\d+", x).group()))

    else:
        print(f"Generating random samples for: {environment_name}")
        checkpoint_files = ["random"]
        total_steps = n_samples * total_steps_factor
        num_checkpoints = 1
        steps_per_checkpoint = total_steps
        samples_per_checkpoint = n_samples
        gamma = expert_models[0][0].gamma

    print(
        f"""
    Sample Generation Info:
      Environment: {environment_name}
      Algorithm: {algorithm}
      Number of Checkpoints: {num_checkpoints}
      Checkpoint Files: {checkpoint_files}
      Total Steps: {total_steps}
      Steps per Checkpoint: {steps_per_checkpoint}
      Target Samples: {n_samples}
      Samples per Checkpoint: {samples_per_checkpoint}
      Env. Gamma: {gamma}
    """
    )

    if action_one_hot:
        one_hot_dim = environment.action_space.n  # only works for discrete spaces

    segments = []
    state_copies = []

    for model_file in checkpoint_files:
        trajectory = []
        sample_indices = random.sample(
            range(steps_per_checkpoint - segment_len + 1),
            k=min(samples_per_checkpoint + 1, steps_per_checkpoint - segment_len + 1),
        )
        final_segment_indices = sorted(set(sample_indices))

        if model_file != "random":
            # Use specified checkpoint index
            checkpoint_
            model_path = checkpoints_dir.replace("_1", f"_{random.choice(possible_checkpoint_indices)}")
            model = model_class.load(
                os.path.join(model_path, model_file),
                custom_objects={"learning_rate": 0.0, "lr_schedule": lambda _: 0.0},
            )
            norm_env_path = os.path.join(model_path, env_name, "vecnormalize.pkl")
            norm_env = (
                VecNormalize.load(norm_env_path, DummyVecEnv([lambda: environment])) if os.path.isfile(norm_env_path) else None
            )
        else:
            model = None
            norm_env = None

        observation, _ = environment.reset()

        for step in range(steps_per_checkpoint):
            if step in final_segment_indices:
                state_copies.append(environment.save_state(observation=observation))

            if model is not None:
                actions, _ = model.predict(
                    norm_env.normalize_obs(observation) if norm_env else observation,
                    deterministic=False,
                )
            else:
                actions = environment.action_space.sample()

            next_observation, reward, terminated, truncated, _ = environment.step(actions)
            done = terminated or truncated

            if action_one_hot:
                actions = one_hot_vector(actions, one_hot_dim)

            trajectory.append((np.expand_dims(observation, axis=0), actions, reward, done))

            observation = next_observation if not done else environment.reset()[0]

            if step % 1000 == 0:
                print(f"Generated {step} steps out of {steps_per_checkpoint}")

        segments.extend(
            create_segments(
                trajectory,
                final_segment_indices,
                np.where([t[3] for t in trajectory])[0],
                segment_len,
            )
        )

        print(f"Generated segments: {len(segments)} of target {n_samples}")

    return {"segments": segments}


def main():
    parser = TrainingUtils.setup_base_parser()
    parser.add_argument(
        "--n-steps-factor",
        type=int,
        default=5,
        help="Number of steps sampled for each sample instance",
    )
    parser.add_argument(
        "--segment-len",
        type=int,
        default=50,
        help="Segment length for sample generation",
    )
    parser.add_argument("--save-folder", type=str, default="samples", help="Save folder for samples")
    parser.add_argument("--top-n-models", type=int, default=3, help="Top N models to use")
    parser.add_argument(
        "--expert-model-base-path",
        type=str,
        default="gt_agents",
        help="Expert model base path",
    )
    parser.add_argument(
        "--checkpoint-index",
        type=int,
        default=5,
        help="Specific checkpoint index to use",
    )
    parser.add_argument("--random", action="store_true", help="Generate random samples only")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of samples to generate",
    )

    args = parser.parse_args()

    TrainingUtils.set_seeds(args.seed)
    device = TrainingUtils.get_device()

    feedback_id, _ = TrainingUtils.get_model_ids(args)

    if not args.random:
        sample_path = Path(args.save_folder) / f"{feedback_id}_{args.seed}.pkl"
    else:
        sample_path = Path(args.save_folder) / f"random_{args.environment}.pkl"

    environment = TrainingUtils.setup_environment(
        args.environment, args.seed, env_kwargs=TrainingUtils.parse_env_kwargs(args.environment_kwargs)
    )

    # Load expert models for gamma value
    expert_models = TrainingUtils.load_expert_models(
        args.environment,
        args.algorithm,
        str(args.expert_model_base_path),
        environment,
        args.top_n_models,
    )

    # Generate samples
    samples = generate_samples(
        model_class=PPO if args.algorithm == "ppo" else SAC,
        expert_models=expert_models,
        environment=environment,
        environment_name=args.environment,
        checkpoints_path=str(args.expert_model_base_path),
        total_steps_factor=args.n_steps_factor,
        n_samples=args.n_samples,
        segment_len=args.segment_len,
        algorithm=args.algorithm,
        device=device,
        random_sample=args.random,
        action_one_hot=isinstance(environment.action_space, gym.spaces.Discrete),
    )

    # Save samples
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sample_path, "wb") as f:
        pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Samples saved to: {sample_path}")


if __name__ == "__main__":
    main()
