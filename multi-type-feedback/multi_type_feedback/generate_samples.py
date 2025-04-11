import argparse
import bisect
import os
import pickle
import random
import re
from pathlib import Path
from typing import List, Type, Union

# necessary to import ale_py/procgen, otherwise it will not be found
import ale_py
import gymnasium as gym
import highway_env
import numpy as np
import pandas as pd
import procgen
import torch
from gymnasium.wrappers.stateful_observation import FrameStackObservation
from gymnasium.wrappers.transform_observation import TransformObservation
from minigrid.wrappers import FlatObsWrapper
from procgen import ProcgenGym3Env
from train_baselines.utils import ppo_make_metaworld_env
from train_baselines.wrappers import Gym3ToGymnasium
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from torch import Tensor
from multi_type_feedback.save_reset_wrapper import SaveResetEnvWrapper


def one_hot_vector(k, max_val):
    vec = np.zeros(max_val)
    np.put(vec, k, 1)
    return vec


def create_segments(arr, start_indices, done_indices, segment_length):
    """
    Creates array segments with target length (segment_length) and minimum length min_segment_len,
    selects the longest contiguous array within a segment [start_indices: start_indeces+segment_length]
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
    return np.sum(rewards * (gamma ** np.arange(len(rewards))))


def generate_feedback(
    model_class: Type[Union[PPO, SAC]],
    expert_models: List[Union[PPO, SAC]],
    environment: gym.Env,
    environment_name: str = "HalfCheetah-v5",
    checkpoints_path: str = "rl_checkpoints",
    total_steps_factor: int = 50,
    n_feedback: int = 100,
    segment_len: int = 50,
    min_segment_len: int = 25,
    algorithm: str = "sac",
    device: str = "cuda",
    binning_type: str = "width",
    action_one_hot: bool = False,
    random_sample: bool = False,
) -> dict:
    """Generate agent's observations and feedback in the training environment."""
    feedback_id = f"{algorithm}_{environment_name.replace('/', '-')}"

    if not random_sample:
        possible_checkpoint_indices = [
            str(model_dir.split("_")[-1])
            for model_dir in os.listdir(os.path.join(checkpoints_path, algorithm))
            if f"{environment_name.replace('/', '-')}" in model_dir
        ]
        checkpoints_dir = os.path.join(
            checkpoints_path, algorithm, f"{environment_name.replace('/', '-')}_1"
        )

        print(f"Generating feedback for: {feedback_id}")

        # Adaptive oversampling
        oversampling_factor = 1.0
        target_n_feedback = int(n_feedback * oversampling_factor)

        checkpoint_files = [
            file
            for file in os.listdir(checkpoints_dir)
            if re.search(r"rl_model_.*\.zip", file)
        ] or [f"{environment_name}.zip"]

        total_steps = n_feedback * total_steps_factor
        num_checkpoints = len(checkpoint_files) + 1
        steps_per_checkpoint = total_steps // num_checkpoints
        feedback_per_checkpoint = target_n_feedback // num_checkpoints
        gamma = expert_models[0][0].gamma

        checkpoint_files = ["random"] + sorted(
            checkpoint_files, key=lambda x: int(re.search(r"\d+", x).group())
        )

    else:
        print(f"Generating random samples for: {environment_name}")
        checkpoint_files = ["random"]
        target_n_feedback = n_feedback
        total_steps = n_feedback * total_steps_factor
        num_checkpoints = 1
        steps_per_checkpoint = total_steps
        feedback_per_checkpoint = target_n_feedback
        gamma = expert_models[0][0].gamma

    print(
        f"""
    Feedback Generation Debug Info:
      Feedback ID: {feedback_id}
      Number of Checkpoints: {num_checkpoints}
      Checkpoint Files: {checkpoint_files}
      Total Steps: {total_steps}
      Steps per Checkpoint: {steps_per_checkpoint}
      Target Feedback: {n_feedback}
      Oversampled Generated Feedback Instances: {target_n_feedback}
      Feedback per Checkpoint: {feedback_per_checkpoint}
      Env. Gamma: {gamma}
    """
    )

    if action_one_hot:
        one_hot_dim = environment.action_space.n  # only works for discrete spaces

    segments = []
    state_copies = []
    for model_file in checkpoint_files:
        feedback = []
        fb_indices = random.sample(
            range(steps_per_checkpoint - segment_len + 1), k=feedback_per_checkpoint + 1
        )
        final_segment_indices = sorted(set(fb_indices))

        if model_file != "random":
            # replace the _1 index by other possible indices, this only works if all models have exactly the same number of checkpoints
            # model_path = checkpoints_dir.replace("_1", f"_{random.choice(possible_checkpoint_indices)}")
            model_path = checkpoints_dir.replace("_1", "_5")
            model = model_class.load(
                os.path.join(model_path, model_file),
                custom_objects={"learning_rate": 0.0, "lr_schedule": lambda _: 0.0},
            )
            norm_env_path = os.path.join(
                model_path, environment_name, "vecnormalize.pkl"
            )
            norm_env = (
                VecNormalize.load(norm_env_path, DummyVecEnv([lambda: environment]))
                if os.path.isfile(norm_env_path)
                else None
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

            next_observation, reward, terminated, truncated, _ = environment.step(
                actions
            )
            done = terminated or truncated

            if action_one_hot:
                actions = one_hot_vector(actions, one_hot_dim)

            feedback.append(
                (np.expand_dims(observation, axis=0), actions, reward, done)
            )

            observation = next_observation if not done else environment.reset()[0]

            if step % 100 == 0:
                print(f"Generate {step} steps ouf of {total_steps}")

        segments.extend(
            create_segments(
                feedback,
                final_segment_indices,
                np.where([f[3] for f in feedback])[0],
                segment_len,
            )
        )
        print(len(segments))

        print(f"Generated segments: {len(segments)} of target {target_n_feedback}")

    return {"segments": segments}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=int, default=0, help="Experiment number")
    parser.add_argument("--algorithm", type=str, default="ppo", help="RL algorithm")
    parser.add_argument(
        "--environment", type=str, default="HalfCheetah-v5", help="Environment"
    )
    parser.add_argument(
        "--n-steps-factor",
        type=int,
        default=int(20),
        help="Number of steps sampled for each feedback instance",
    )
    parser.add_argument(
        "--n-feedback",
        type=int,
        default=int(1000),
        help="How many feedback instances should be generated",
    )
    parser.add_argument(
        "--seed", type=int, default=6389, help="TODO: Seed for env and stuff"
    )
    parser.add_argument(
        "--segment-len",
        type=int,
        default=50,
        help="How long is the segment we generate feedback for",
    )
    parser.add_argument(
        "--save-folder", type=str, default="samples", help="Where to save the feedback"
    )
    parser.add_argument("--top-n-models", type=int, default=3)
    parser.add_argument("--random", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    feedback_id = f"{args.algorithm}_{args.environment}"
    if not args.random:
        feedback_path = (
            Path(__file__).parents[1].resolve()
            / args.save_folder
            / f"{feedback_id}_{args.seed}.pkl"
        )
    else:
        feedback_path = (
            Path(__file__).parents[1].resolve()
            / args.save_folder
            / f"random_{args.environment}.pkl"
        )
    checkpoints_path = "../main/gt_agents"

    # load "ensemble" of expert agents
    env_name = (
        args.environment
        if "ALE" not in args.environment
        else args.environment.replace("/", "-")
    )
    expert_model_paths = [
        os.path.join(checkpoints_path, args.algorithm, model)
        for model in os.listdir(os.path.join(checkpoints_path, args.algorithm))
        if env_name in model
    ]
    orig_len = len(expert_model_paths)
    # expert_model = (PPO if args.algorithm == "ppo" else SAC).load(
    #    os.path.join(checkpoints_path, args.algorithm, f"{args.environment.replace("/", "-")}_1", "best_model.zip")
    # )

    try:
        run_eval_scores = pd.read_csv(
            os.path.join(checkpoints_path, "collected_results.csv")
        )
        run_eval_scores = (
            run_eval_scores.loc[run_eval_scores["env"] == args.environment]
            .sort_values(by=["eval_score"], ascending=False)
            .head(args.top_n_models)["run"]
            .to_list()
        )
        expert_model_paths = [
            path
            for path in expert_model_paths
            if path.split(os.path.sep)[-1] in run_eval_scores
        ]
    except:
        print(
            "[WARN] No eval benchmark results are available. Check you eval benchmarks"
        )

    if "procgen" in args.environment:
        _, short_name, _ = args.environment.split("-")
        environment = Gym3ToGymnasium(ProcgenGym3Env(num=1, env_name=short_name))
        environment = SaveResetEnvWrapper(
            TransformObservation(
                environment, lambda obs: obs["rgb"], environment.observation_space
            )
        )
    elif "ALE/" in args.environment:
        environment = FrameStackObservation(WarpFrame(gym.make(args.environment)), 4)
        environment = SaveResetEnvWrapper(
            TransformObservation(
                environment, lambda obs: obs.squeeze(-1), environment.observation_space
            )
        )
    elif "MiniGrid" in args.environment:
        environment = SaveResetEnvWrapper(FlatObsWrapper(gym.make(args.environment)))
    elif "metaworld" in args.environment:
        environment_name = args.environment.replace("metaworld-", "")
        environment = SaveResetEnvWrapper(
            ppo_make_metaworld_env(environment_name, args.seed)
        )
    else:
        environment = SaveResetEnvWrapper(gym.make(args.environment))

    expert_models = []
    for expert_model_path in expert_model_paths:
        if os.path.isfile(
            os.path.join(expert_model_path, env_name, "vecnormalize.pkl")
        ):
            norm_env = VecNormalize.load(
                os.path.join(expert_model_path, env_name, "vecnormalize.pkl"),
                DummyVecEnv([lambda: environment]),
            )
        else:
            norm_env = None
        expert_models.append(
            (
                (PPO if args.algorithm == "ppo" else SAC).load(
                    os.path.join(expert_model_path, f"{env_name}.zip")
                ),
                norm_env,
            )
        )

    model_class = PPO if args.algorithm == "ppo" else SAC

    is_discrete_action = isinstance(environment.action_space, gym.spaces.Discrete)

    feedback = generate_feedback(
        model_class,
        expert_models,
        environment,
        environment_name=args.environment,
        total_steps_factor=args.n_steps_factor,
        n_feedback=args.n_feedback,
        segment_len=args.segment_len,
        checkpoints_path=checkpoints_path,
        algorithm=args.algorithm,
        device=device,
        random_sample=args.random,
        action_one_hot=is_discrete_action,
    )

    feedback_path.parent.mkdir(parents=True, exist_ok=True)
    with open(feedback_path, "wb") as feedback_file:
        print("FB path", feedback_path)
        pickle.dump(feedback, feedback_file, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
