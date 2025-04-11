from typing import Optional, Dict, Any, Tuple, List
import os
from pathlib import Path

import gymnasium as gym
import torch
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv

from train_baselines.train import ExperimentManager
from multi_type_feedback.feedback_oracle import FeedbackOracle
from multi_type_feedback.utils import TrainingUtils, get_project_root

class DynamicRLHFExperimentManager(ExperimentManager):
    """
    Experiment manager specific to DynamicRLHF that inherits from the base ExperimentManager.
    This allows reuse of hyperparameter loading and evaluation callback functionality.
    """
    def __init__(
        self,
        args,
        algo: str,
        env_id: str,
        log_folder: str,
        n_timesteps: int = int(1e5),
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        save_freq: int = -1,
        hyperparams: Optional[Dict[str, Any]] = None,
        env_kwargs: Optional[Dict[str, Any]] = None,
        trained_agent: str = "",
        seed: int = 0,
        device: str = "auto",
        total_iterations: int = 10,
    ):
        super().__init__(
            args=args,
            algo=algo,
            env_id=env_id,
            log_folder=log_folder,
            tensorboard_log=log_folder,
            n_timesteps=n_timesteps,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            save_freq=save_freq,
            hyperparams=hyperparams,
            env_kwargs=env_kwargs,
            trained_agent=trained_agent,
            seed=seed,
            device=device,
            use_wandb_callback=True,  # Enable W&B logging by default for RLHF
        )

    def setup_drlhf(self) -> Tuple[BaseAlgorithm, FeedbackOracle]:
        """
        Set up the DynamicRLHF experiment including the model and oracle.
        Reuses functionality from ExperimentManager.setup_experiment().
        """
        # Use parent class to setup basic experiment
        model, _ = self.setup_experiment()
        if model is None:
            raise ValueError("Failed to initialize model")

        # Setup oracle
        feedback_id, _ = TrainingUtils.get_model_ids(self.args)
        feedback_path = Path(self.args.reference_data_folder) / f"{feedback_id}.pkl"

        environment = TrainingUtils.setup_environment(self.args.environment, self.args.seed)
        expert_models = TrainingUtils.load_expert_models(
            env_name=self.args.environment,
            algorithm=self.args.algorithm,
            checkpoints_path=str(get_project_root() / self.args.expert_model_base_path),
            environment=environment,
            top_n_models=self.args.top_n_models,
        )

        oracle = FeedbackOracle(
            expert_models=expert_models,
            environment=environment,
            reference_data_path=feedback_path,
            noise_level=self.args.noise_level,
        )

        return model, oracle

    def create_drlhf(self, oracle: FeedbackOracle, env: VecEnv) -> 'DynamicRLHF':
        """
        Create a DynamicRLHF instance with the loaded hyperparameters.
        """
        from multi_type_feedback.dynamic_rlhf import DynamicRLHF
        
        return DynamicRLHF(
            oracle=oracle,
            env=env,
            env_name=self.env_name.gym_id,
            algorithm=self.algo,
            feedback_types=self.args.feedback_types,
            total_iterations=total_iterations,
            n_feedback_per_iteration=self.args.n_feedback_per_iteration,
            feedback_buffer_size=self.args.feedback_buffer_size,
            rl_steps_per_iteration=self.args.rl_steps_per_iteration,
            reward_training_epochs=self.args.reward_training_epochs,
            device=self.device,
            enable_wandb=True,
            wandb_project_name=self.args.wandb_project_name,
            num_ensemble_models=self.args.num_ensemble_models,
        )

def main():
    """Example usage of DynamicRLHFExperimentManager"""
    parser = TrainingUtils.setup_base_parser()
    
    parser.add_argument(
        "--feedback-types",
        nargs="+",
        type=str,
        default=["evaluative", "comparative", "demonstrative", "corrective"],
        help="Types of feedback to use",
    )
    parser.add_argument(
        "--sampling-strategy",
        type=str,
        default="random",
        choices=["random", "uncertainty"],
        help="Feedback sampling strategy",
    )
    parser.add_argument(
        "--save-folder",
        type=str,
        default="trained_agents_dynamic",
        help="Folder for finished feedback RL agents",
    )
    parser.add_argument(
        "--reference-data-folder",
        type=str,
        default="feedback",
        help="Folder containing pre-computed offline feedback for calibration",
    )
    parser.add_argument(
        "--n-timesteps",
        type=int,
        default=int(1e5),
        help="Total number of timesteps to train for",
    )
    parser.add_argument(
        "--rl-steps-per-iteration",
        type=int,
        default=5000,
        help="Number of environment steps per iteration",
    )
    parser.add_argument(
        "--n-feedback-per-iteration",
        type=int,
        default=20,
        help="Feedback instances collected per iteration",
    )
    parser.add_argument(
        "--reward-training-epochs",
        type=int,
        default=3,
        help="Number of epochs for reward model training",
    )
    parser.add_argument(
        "--top-n-models", 
        type=int, 
        default=3, 
        help="Top N models to use"
    )
    parser.add_argument(
        "--expert-model-base-path", 
        type=str, 
        default="train_baselines/gt_agents", 
        help="Expert model base path"
    )
    parser.add_argument(
        "--feedback-buffer-size",
        type=int,
        default=2000,
        help="Maximum size of the feedback buffer",
    )
    parser.add_argument(
        "--num-ensemble-models",
        type=int,
        default=4,
        help="Number of ensemble models for masksemble",
    )
    args = parser.parse_args()

    # Calculate total iterations based on desired timesteps
    total_iterations = max(1, args.n_timesteps // args.rl_steps_per_iteration)
    print(f"\nTraining for {args.n_timesteps} total timesteps")
    print(f"Using {args.rl_steps_per_iteration} steps per iteration")
    print(f"Will run for {total_iterations} iterations to approximate desired timesteps")
    args = parser.parse_args()
    
    exp_manager = DynamicRLHFExperimentManager(
        args=args,
        algo=args.algorithm,
        env_id=args.environment,
        log_folder="logs",
        n_timesteps=args.n_timesteps,
        total_iterations=total_iterations,
    )
    
    # Setup experiment
    model, oracle = exp_manager.setup_drlhf()
    
    # Create environment
    env = exp_manager.create_envs(n_envs=1)
    
    # Create DRLHF instance
    drlhf = exp_manager.create_drlhf(oracle, env)
    
    # Train
    drlhf.train(sampling_strategy=exp_manager.sampling_strategy)

if __name__ == "__main__":
    main()