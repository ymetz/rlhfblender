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
import cv2
import gymnasium as gym
import numpy as np
import pandas as pd
import procgen
import torch
from gymnasium.wrappers.stateful_observation import FrameStackObservation
from gymnasium.wrappers.transform_observation import TransformObservation
from minigrid.wrappers import FlatObsWrapper
from procgen import ProcgenGym3Env
from train_baselines.wrappers import Gym3ToGymnasium
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from torch import Tensor

from multi_type_feedback.datatypes import FeedbackDataset
from multi_type_feedback.save_reset_wrapper import SaveResetEnvWrapper


def predict_expert_value(
    expert_model: Union[PPO, SAC], observation: np.ndarray, actions: Tensor = None
) -> Tensor:
    """Return the value from the expert's value function for a given observation and actions."""

    observation = expert_model.policy.obs_to_tensor(observation)[0]
    with torch.no_grad():
        return torch.min(
            (
                torch.cat(
                    expert_model.policy.critic_target(observation, actions), dim=1
                )
                if isinstance(expert_model, SAC)
                else expert_model.policy.predict_values(observation)
            ),
            dim=1,
            keepdim=True,
        )[0]


def create_segments(arr, start_indices, done_indices, segment_length, min_segment_len):
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

        if len(segment) > min_segment_len:  # Only add non-empty segments
            segments.append(segment)

    return segments


def discounted_sum_numpy(rewards, discount_factor):
    rewards = np.array(rewards)
    n = len(rewards)
    discount_factors = discount_factor ** np.arange(n)
    return np.sum(rewards * discount_factors)


def generate_feedback(
    model_class: Type[Union[PPO, SAC]],
    expert_models: List[Union[PPO, SAC]],
    environment: gym.Env,
    environment_name: str = "HalfCheetah-v3",
    checkpoints_path: str = "rl_checkpoints",
    total_steps_factor: int = 50,
    n_feedback: int = 100,
    segment_len: int = 50,
    min_segment_len: int = 25,  # segments can be pruned at the beginning or end of episodes, remove if shorter than min_len
    algorithm: str = "sac",
    seed: int = 1337,
) -> FeedbackDataset:
    """Generate agent's observations and feedback in the training environment."""
    feedback_id = f"{algorithm}_{environment_name.replace('/', '-')}"
    checkpoints_dir = os.path.join(
        checkpoints_path, algorithm, f"{environment_name.replace('/', '-')}_1"
    )

    print(f"Generating feedback for: {feedback_id}")

    checkpoint_files = [
        sorted(
            [
                file
                for file in os.listdir(checkpoints_dir)
                if re.search(r"rl_model_.*\.zip", file)
            ]
            or [f"{environment_name}.zip"],
            key=lambda x: int(re.search(r"\d+", x).group()),
        )[1]
    ]

    total_steps = (
        n_feedback * total_steps_factor
    )  # how many steps we want to generate to sample from, a natural choice is the segment length
    num_checkpoints = (
        len(checkpoint_files) + 1
    )  # also sample steps from random actios as the 0th checkpoint
    steps_per_checkpoint = total_steps // num_checkpoints
    feedback_per_checkpoint = n_feedback // num_checkpoints
    gamma = expert_models[0][0].gamma

    # Sort the files based on the extracted numerical value, makes handling and debugging a bit easier, is shuffled later anyways if necessary
    checkpoint_files = ["random"] + checkpoint_files

    print(
        f"""
    Video Feedback Generation Debug Info:
      Feedback ID: {feedback_id}
      Checkpoints Directory: {checkpoints_dir}
      Number of Checkpoints: {num_checkpoints}
      Checkpoint Files: {checkpoint_files}
      Total Steps: {total_steps}
      Steps per Checkpoint: {steps_per_checkpoint}
      Total Feedback Instances: {n_feedback}
      Feedback per Checkpoint: {feedback_per_checkpoint}
      Env. Gamma: {gamma}
    """
    )

    segments = []
    state_copies = []
    for model_file in checkpoint_files:

        feedback = []
        # we already sample the indices for the number of generated feedback instances/segments (last segment_len steps should
        # not be sampled from)
        fb_indices = random.choices(
            range(0, steps_per_checkpoint + 1 - segment_len), k=feedback_per_checkpoint
        )
        final_segment_indices = list(set(fb_indices))

        if model_file != "random":
            model = model_class.load(
                os.path.join(checkpoints_dir, model_file),
                custom_objects={"learning_rate": 0.0, "lr_schedule": lambda _: 0.0},
            )
        else:
            model = None

        observation, _ = environment.reset()

        # now collect original data

        for step in range(steps_per_checkpoint):
            if step in final_segment_indices:
                state_copies.append(environment.save_state(observation=observation))

            render = environment.render()
            if model is not None:
                actions, _ = model.predict(observation, deterministic=True)
            else:
                actions = environment.action_space.sample()
            next_observation, reward, terminated, truncated, _ = environment.step(
                actions
            )
            done = terminated | truncated

            feedback.append(
                (np.expand_dims(observation, axis=0), actions, reward, done, render)
            )

            observation = next_observation if not done else environment.reset()[0]

        # generate feedback from collected examples, split at given indices and dones
        # | set(np.where([f[3] for f in feedback] == True)[0])
        segments.extend(
            create_segments(
                feedback,
                final_segment_indices,
                np.where(np.array([f[3] for f in feedback]) is True)[0],
                segment_len,
                min_segment_len,
            )
        )

        print(f"Generated segments: {len(segments)} of approx. {n_feedback}")

    # start by computing the evaluative fb. (for the comparative one, we just used samples segment pairs)
    opt_gaps = []
    single_initial_preds = []  # for debugging
    single_final_preds = []  # for debugging
    for seg in segments:
        # predict the initial value
        initial_vals = [
            predict_expert_value(expert_model[0], np.array(seg[0][0])).item()
            for expert_model in expert_models
        ]
        initial_val = np.mean(initial_vals)
        single_initial_preds.append(initial_vals)

        # sum the discounted rewards, don't add reward for last step because we use it to calculate final value
        discounted_rew_sum = discounted_sum_numpy([s[2] for s in seg[:-1]], gamma)

        # get the final value
        final_vals = [
            predict_expert_value(expert_model[0], np.array(seg[-1][0])).item()
            for expert_model in expert_models
        ]
        final_val = np.mean(final_vals)
        single_final_preds.append(final_vals)

        # opt gap is the expected returns - actual returns
        opt_gap = (initial_val - gamma ** len(seg) * final_val) - discounted_rew_sum
        opt_gaps.append(opt_gap)

    print("[INFO] Succesfully computed opt gaps")

    # instructive feedback, reset env and run expert model for demos for each segment
    corrections = []
    for i, state in enumerate(state_copies):
        expert_model = expert_models[
            np.random.randint(len(expert_models))
        ]  # sample random expert model, we just choose one for a segment
        _, _ = environment.reset()
        obs = environment.load_state(state)

        demo = []
        for _ in range(segment_len):
            # if we should normalize obs, do now
            if expert_model[1] is not None:  # if the normalize env instance is not none
                obs = expert_model[1].normalize_obs(obs)
            render = environment.render()
            action, _ = expert_model[0].predict(obs, deterministic=True)
            obs, rew, terminated, truncated, _ = environment.step(action)
            done = terminated | truncated
            demo.append((np.expand_dims(obs, axis=0), action, rew, done, render))

            if done:
                break
        corrections.append((segments[i], demo))

    # compute opt. gaps for the instructive feedback
    opt_gaps_instructive = []
    for i, (seg, demo) in enumerate(corrections):
        # predict the initial value
        initial_vals = [
            predict_expert_value(expert_model[0], np.array(demo[0][0])).item()
            for expert_model in expert_models
        ]
        initial_val = np.mean(initial_vals)

        # sum the discounted rewards, don't add reward for last step because we use it to calculate final value
        discounted_rew_sum = discounted_sum_numpy([s[2] for s in demo[:-1]], gamma)

        # get the final value
        final_vals = [
            predict_expert_value(expert_model[0], np.array(seg[-1][0])).item()
            for expert_model in expert_models
        ]
        final_val = np.mean(final_vals)

        # opt gap is the expected returns - actual returns
        opt_gap = (initial_val - gamma ** len(demo) * final_val) - discounted_rew_sum
        opt_gaps_instructive.append(opt_gap)

    # now save the generated feedback as videos, with the naming scheme: <segment/demo>_env_seed_segmentindex_optgap.mp4, round optgap to int
    for i, (seg, demo) in enumerate(corrections):

        # save the segment
        out = cv2.VideoWriter(
            f"feedback_videos/segment_{feedback_id}_{seed}_{i}_{int(opt_gaps[i])}.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            30,
            (render.shape[1], render.shape[0]),
        )
        for frame in [f[4] for f in seg]:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        out.release()

        # save the demo
        out = cv2.VideoWriter(
            f"feedback_videos/demo_{feedback_id}_{seed}_{i}_{int(opt_gaps_instructive[i])}.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            30,
            (render.shape[1], render.shape[0]),
        )
        for frame in [f[4] for f in demo]:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        out.release()

    print("[INFO] Succesfully generated videos for demonstrative feedback")


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
        default=int(30),
        help="How many feedback instances should be generated",
    )
    parser.add_argument(
        "--seed", type=int, default=1337, help="TODO: Seed for env and stuff"
    )
    parser.add_argument(
        "--segment-len",
        type=int,
        default=50,
        help="How long is the segment we generate feedback for",
    )
    parser.add_argument(
        "--save-folder", type=str, default="feedback", help="Where to save the feedback"
    )
    parser.add_argument("--top-n-models", type=int, default=1)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    os.makedirs("feedback_videos", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    feedback_id = f"{args.algorithm}_{args.environment}"
    feedback_path = (
        Path(__file__).parents[1].resolve()
        / args.save_folder
        / f"{feedback_id}_{args.seed}.pkl"
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
    print(expert_model_paths)
    orig_len = len(expert_model_paths)

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
        print(run_eval_scores)
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
        environment = FrameStackObservation(AtariWrapper(gym.make(args.environment)), 4)
        environment = SaveResetEnvWrapper(
            TransformObservation(
                environment, lambda obs: obs.squeeze(-1), environment.observation_space
            )
        )
    elif "MiniGrid" in args.environment:
        environment = SaveResetEnvWrapper(FlatObsWrapper(gym.make(args.environment)))
    else:
        environment = SaveResetEnvWrapper(
            gym.make(args.environment, render_mode="rgb_array")
        )

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
                    os.path.join(expert_model_path, "best_model.zip")
                ),
                norm_env,
            )
        )

    model_class = PPO if args.algorithm == "ppo" else SAC

    generate_feedback(
        model_class,
        expert_models,
        environment,
        environment_name=args.environment,
        total_steps_factor=args.n_steps_factor,
        n_feedback=args.n_feedback,
        segment_len=args.segment_len,
        checkpoints_path=checkpoints_path,
        algorithm=args.algorithm,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
