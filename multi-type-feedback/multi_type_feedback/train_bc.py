"""Module for training a behavioral cloning agent using demonstrations."""

import argparse
import os
import pickle
from os import path
from pathlib import Path

# register custom envs
import ale_py
import gymnasium as gym
import highway_env
import minigrid
import numpy as np
import torch
from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.types import Trajectory
from train_baselines.utils import ppo_make_metaworld_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed

import wandb
from multi_type_feedback.utils import TrainingUtils


def load_demonstrations(
    demo_path: str,
    noise_level: float = 0.0,
    discrete_action_space: bool = False,
    n_feedback: int = -1,
    seed: int = 42,
):
    """Load and process demonstration data from pickle file."""
    with open(demo_path, "rb") as f:
        feedback_data = pickle.load(f)

    observations = []
    actions = []
    terms = []

    for demo in feedback_data["demos"]:
        obs = np.vstack([p[0] for p in demo])
        acts = np.vstack([p[1] for p in demo])
        dones = np.vstack([p[-1] for p in demo])

        if noise_level > 0.0:
            # Add noise to demonstrations if specified
            obs_min, obs_max = np.min(obs, axis=0), np.max(obs, axis=0)
            obs_diff = obs_max - obs_min
            acts_min, acts_max = np.min(acts, axis=0), np.max(acts, axis=0)
            acts_diff = acts_max - acts_min

            # Helper function for truncated Gaussian noise
            def truncated_gaussian_vectorized(mean, width, low, upp):
                samples = np.random.normal(loc=mean, scale=width)
                return np.clip(samples, low, upp)

            obs = truncated_gaussian_vectorized(
                mean=obs,
                width=np.array(noise_level) * obs_diff,
                low=obs_min,
                upp=obs_max,
            )

            acts = truncated_gaussian_vectorized(
                mean=acts,
                width=np.array(noise_level) * acts_diff,
                low=acts_min,
                upp=acts_max,
            )

        if len(acts) <= 1:
            continue  # very short trajectory does not work
            print("SKIPPING VERY SHORT TRAJECTORY")

        observations.append(obs)

        # acts
        if discrete_action_space:
            acts = np.argmax(acts, axis=1)

        actions.append(acts[:-1])
        terms.append(dones)

    if n_feedback != -1 and n_feedback < len(observations):
        # is a bit inefficient as we first collected the entire dataset..but we just have to do it once
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(observations), size=n_feedback, replace=False)

        observations = [observations[i] for i in indices]
        actions = [actions[i] for i in indices]
        terms = [terms[i] for i in indices]

    return [
        Trajectory(
            obs=flat_obs,
            acts=flat_acts,
            terminal=terms[-1],
            infos=[{} for _ in range(len(flat_acts))],
        )
        for (flat_obs, flat_acts, terms) in zip(observations, actions, terms)
    ]


def main():
    parser = TrainingUtils.setup_base_parser()
    parser.add_argument(
        "--n-epochs", type=int, default=20, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training"
    )
    args = parser.parse_args()

    TrainingUtils.set_seeds(args.seed)
    feedback_id, model_id = TrainingUtils.get_model_ids(args)

    script_path = Path(__file__).parents[1].resolve()
    environment = TrainingUtils.setup_environment(
        args.environment, save_reset_wrapper=False
    )
    eval_env = TrainingUtils.setup_environment(args.environment)

    TrainingUtils.setup_wandb_logging(
        f"BC_{model_id}",
        args,
        {"model_type": "behavioral_cloning"},
        wandb_project_name=args.wandb_project_name,
    )

    demo_path = os.path.join(script_path, "feedback_regen", f"{feedback_id}.pkl")
    is_discrete_action = isinstance(environment.action_space, gym.spaces.Discrete)

    trajectories = load_demonstrations(
        demo_path,
        args.noise_level,
        discrete_action_space=is_discrete_action,
        n_feedback=args.n_feedback,
        seed=args.seed,
    )
    trajectories = rollout.flatten_trajectories(trajectories)

    bc_trainer = bc.BC(
        observation_space=environment.observation_space,
        action_space=environment.action_space,
        demonstrations=trajectories,
        batch_size=args.batch_size,
        rng=np.random.default_rng(args.seed),
        device=TrainingUtils.get_device(),
    )

    # Create BC trainer
    bc_trainer = bc.BC(
        observation_space=environment.observation_space,
        action_space=environment.action_space,
        demonstrations=trajectories,  # We'll manually pass transitions
        batch_size=args.batch_size,
        rng=rng,
        device="cuda:0",
        # learning_rate=args.learning_rate,
    )

    # Train the BC policy
    for epoch in range(args.n_epochs):
        stats = bc_trainer.train(n_epochs=1, progress_bar=False)

        # Evaluate policy
        mean_reward, std_reward = evaluate_policy(
            bc_trainer.policy, eval_env, n_eval_episodes=10
        )
        print(f"Epoch {epoch}: Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")
        wandb.log(
            {"epoch": epoch, "mean_reward": mean_reward, "std_reward": std_reward}
        )

    # Save the trained policy
    save_path = os.path.join("agents", f"BC_{MODEL_ID}")
    os.makedirs(save_path, exist_ok=True)
    bc_trainer.policy.save(os.path.join(save_path, "bc_policy"))

    # Final evaluation
    mean_reward, std_reward = evaluate_policy(
        bc_trainer.policy,
        eval_env,
        n_eval_episodes=100,
    )
    print(f"Final evaluation: Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")
    wandb.log(
        {
            "final_mean_reward": mean_reward,
            "final_std_reward": std_reward,
        }
    )


if __name__ == "__main__":
    main()
