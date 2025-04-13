from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
import torch
from pathlib import Path
import pickle
import pandas as pd
import time
import uuid
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from train_baselines.exp_manager import ExperimentManager
from torch.utils.data import DataLoader

import wandb
from multi_type_feedback.feedback_dataset import BufferDataset, load_flat_buffer_into_feedback_dataset
from multi_type_feedback.feedback_oracle import FeedbackOracle
from multi_type_feedback.networks import (
    SingleCnnNetwork,
    SingleNetwork,
    calculate_pairwise_loss,
    calculate_single_reward_loss,
)
from multi_type_feedback.utils import TrainingUtils, get_project_root, RewardVecEnvWrapper
from multi_type_feedback.dynamic_rlhf import DynamicRLHF


class DynamicRLHFDebugging(DynamicRLHF):
    def __init__(self, *args, **kwargs):
        # Optional log directory for saving statistics
        self.log_dir = kwargs.pop('log_dir', './debug_logs')
        
        # Call parent constructor
        super().__init__(*args, **kwargs)
        
        # Initialize debugging statistics
        self.init_debug_statistics()
    
    def init_debug_statistics(self):
        """Initialize data structures to track debugging statistics."""
        # Dictionary to store all debugging statistics
        self.debug_stats = {
            # Track feedback values per type
            'feedback_values': {ft: [] for ft in self.feedback_types},
            
            # Track uncertainties per type
            'uncertainties': {ft: [] for ft in self.feedback_types},
            
            # Track reward model losses
            'reward_model_losses': {ft: [] for ft in self.feedback_types},
            
            # Track ensemble rewards 
            'ensemble_rewards': [],
            
            # Track RL performance metrics
            'rl_performance': {
                'episode_returns': [],
                'episode_lengths': [],
                'timestamps': []
            },
            
            # Track feedback counts per iteration
            'feedback_counts': [],
            
            # Track training progress
            'iteration_metrics': [],
            
            # Track raw feedback data by type for detailed analysis
            'raw_feedback': {ft: [] for ft in self.feedback_types},
            
            # Track trajectory rewards to analyze oracle feedback alignment
            'trajectory_rewards': [],
            
            # Track model predictions vs. oracle ratings
            'evaluative_predictions': [],
            
            # Track comparative preferences and model predictions
            'comparative_predictions': [],
            
            # Environment name and algorithm for reference
            'env_name': self.env_name,
            'algorithm': self.algorithm,
            
            # Configuration details
            'config': {
                'feedback_types': self.feedback_types,
                'n_feedback_per_iteration': self.n_feedback_per_iteration,
                'feedback_buffer_size': self.feedback_buffer_size,
                'rl_steps_per_iteration': self.rl_steps_per_iteration,
                'num_ensemble_models': self.num_ensemble_models
            }
        }
    
    def train_iteration(self, sampling_strategy: str = "random"):
        """Override train_iteration to log statistics."""
        # Log current iteration
        iteration_metrics = {
            'timestamp': time.time(), 
            'iteration': len(self.debug_stats['iteration_metrics']),
            'strategy': sampling_strategy
        }
        
        # Collect trajectories
        trajectories, initial_states = self.collect_trajectories(
            self.n_feedback_per_iteration
        )
        
        # Track trajectory rewards for analysis
        trajectory_rewards = [sum(step[2] for step in traj) for traj in trajectories]
        self.debug_stats['trajectory_rewards'].extend(
            [(len(self.debug_stats['iteration_metrics']), i, r) 
             for i, r in enumerate(trajectory_rewards)]
        )
        iteration_metrics['mean_trajectory_reward'] = np.mean(trajectory_rewards)
        
        # Get feedback based on sampling strategy
        if sampling_strategy == "random":
            feedback, feedback_counts = self.sample_feedback_random(
                trajectories, initial_states
            )
        else:  # uncertainty
            feedback, feedback_counts = self.sample_feedback_uncertainty(
                trajectories, initial_states
            )
        
        # Log feedback counts
        self.debug_stats['feedback_counts'].append(feedback_counts)
        iteration_metrics['feedback_counts'] = dict(feedback_counts)
        
        # Track raw feedback for detailed analysis
        for i, feedback_item in enumerate(feedback):
            for feedback_type, feedback_data in feedback_item.items():
                if feedback_type == "selected_uncertainty":
                    continue
                
                # Store feedback with iteration and trajectory info
                feedback_entry = {
                    'iteration': len(self.debug_stats['iteration_metrics']),
                    'trajectory_idx': i,
                    'trajectory_reward': trajectory_rewards[i] if i < len(trajectory_rewards) else None,
                    'data': feedback_data
                }
                self.debug_stats['raw_feedback'][feedback_type].append(feedback_entry)
                
                # Extract and log specific values based on feedback type
                if feedback_type == "evaluative" and isinstance(feedback_data, tuple) and len(feedback_data) > 1:
                    rating = feedback_data[1]
                    self.debug_stats['feedback_values'][feedback_type].append((len(self.debug_stats['iteration_metrics']), i, rating))
                
                elif feedback_type in ["comparative", "descriptive_preference"] and isinstance(feedback_data, tuple) and len(feedback_data) > 1:
                    preference = feedback_data[1]
                    self.debug_stats['feedback_values'][feedback_type].append((len(self.debug_stats['iteration_metrics']), i, preference))
        
        # Update feedback buffers
        self.update_feedback_buffers(feedback)
        
        # Track buffer sizes
        buffer_sizes = {ft: len(self.feedback_buffers[ft]) for ft in self.feedback_types}
        iteration_metrics['buffer_sizes'] = buffer_sizes
        
        # Train reward models
        reward_metrics = self.train_reward_models()
        
        # Log reward model metrics
        for feedback_type, loss in reward_metrics.items():
            if feedback_type in self.debug_stats['reward_model_losses']:
                self.debug_stats['reward_model_losses'][feedback_type].append((len(self.debug_stats['iteration_metrics']), loss))
        
        iteration_metrics['reward_metrics'] = dict(reward_metrics)
        
        # Train RL agent with updated reward models
        self.train_rl_agent()
        
        # Evaluate RL agent performance
        eval_metrics = self.evaluate_rl_agent()
        iteration_metrics.update(eval_metrics)
        
        # Add iteration metrics
        self.debug_stats['iteration_metrics'].append(iteration_metrics)
        
        # Log metrics to wandb if available
        if self.wandb_logger is not None:
            metrics = {"feedback_counts": feedback_counts, **reward_metrics, **eval_metrics}
            wandb.log(metrics)
        
        # Save debugging statistics periodically
        if len(self.debug_stats['iteration_metrics']) % 5 == 0:
            self.save_debug_statistics()
        
        return feedback_counts, reward_metrics
    
    def evaluate_rl_agent(self, n_episodes=5):
        """Evaluate current RL agent and track performance metrics."""
        episode_returns = []
        episode_lengths = []
        
        # Create temporary eval environment
        eval_env = gym.make(self.env_name)
        
        for _ in range(n_episodes):
            obs, _ = eval_env.reset()
            done = False
            episode_return = 0
            episode_length = 0
            
            while not done:
                action, _ = self.rl_agent.predict(obs, deterministic=True)
                next_obs, reward, terminated, truncated, _ = eval_env.step(action)
                
                episode_return += reward
                episode_length += 1
                done = terminated or truncated
                obs = next_obs
            
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
        
        # Record timestamp for tracking evaluation over time
        current_time = time.time()
        
        # Track metrics
        self.debug_stats['rl_performance']['episode_returns'].append(np.mean(episode_returns))
        self.debug_stats['rl_performance']['episode_lengths'].append(np.mean(episode_lengths))
        self.debug_stats['rl_performance']['timestamps'].append(current_time)
        
        eval_metrics = {
            'mean_episode_return': np.mean(episode_returns),
            'std_episode_return': np.std(episode_returns),
            'mean_episode_length': np.mean(episode_lengths)
        }
        
        return eval_metrics
    
    def compute_model_uncertainty(self, trajectory, feedback_type):
        """Track uncertainties when computing them."""
        uncertainty = super().compute_model_uncertainty(trajectory, feedback_type)
        
        # Track uncertainty values
        self.debug_stats['uncertainties'][feedback_type].append((
            len(self.debug_stats['iteration_metrics']),
            uncertainty
        ))
        
        return uncertainty

    def compute_ensemble_reward(self, state, action):
        """Log ensemble rewards occasionally for analysis."""
        reward = super().compute_ensemble_reward(state, action)
        
        # Track some samples of rewards (not all, to avoid memory issues)
        if np.random.random() < 0.01:  # Sample 1% of rewards to avoid excessive memory usage
            self.debug_stats['ensemble_rewards'].append({
                'iteration': len(self.debug_stats['iteration_metrics']),
                'state': state[0].copy() if isinstance(state, np.ndarray) else state,
                'action': action[0].copy() if isinstance(action, np.ndarray) else action,
                'reward': reward[0] if isinstance(reward, np.ndarray) else reward
            })
        
        return reward
    
    def sample_feedback_uncertainty(self, trajectories, initial_states):
        """Track detailed uncertainty-based sampling information."""
        feedback, feedback_counts = super().sample_feedback_uncertainty(trajectories, initial_states)
        
        # Track trajectory-level uncertainties
        uncertainty_data = {ft: [] for ft in self.feedback_types}
        
        for i, trajectory in enumerate(trajectories):
            if i < len(feedback):
                feedback_item = feedback[i]
                for feedback_type in self.feedback_types:
                    if len(self.feedback_buffers[feedback_type]) > 0:
                        uncertainty = self.compute_model_uncertainty(trajectory, feedback_type)
                        uncertainty_data[feedback_type].append(uncertainty)
        
        # Add stats to the current iteration metrics
        if len(self.debug_stats['iteration_metrics']) > 0:
            current_metrics = self.debug_stats['iteration_metrics'][-1]
            for feedback_type, uncertainties in uncertainty_data.items():
                if uncertainties:
                    current_metrics[f'mean_uncertainty_{feedback_type}'] = np.mean(uncertainties)
                    current_metrics[f'max_uncertainty_{feedback_type}'] = np.max(uncertainties)
        
        return feedback, feedback_counts
    
    def save_debug_statistics(self):
        """Save debugging statistics to file."""
        # Create log directory if it doesn't exist
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = int(time.time())
        filename = f"{self.log_dir}/debug_stats_{self.env_name}_{self.algorithm}_{timestamp}.pkl"
        
        # Save statistics
        with open(filename, 'wb') as f:
            pickle.dump(self.debug_stats, f)
        
        print(f"Saved debugging statistics to {filename}")
        
        return filename
    
    def load_debug_statistics(self, filename):
        """Load debugging statistics from file."""
        with open(filename, 'rb') as f:
            self.debug_stats = pickle.load(f)
        
        print(f"Loaded debugging statistics from {filename}")
    
    def train(self, total_iterations: int, sampling_strategy: str = "random"):
        """Override train method to ensure final statistics are saved."""
        # Initialize callbacks if they exist
        if self.callbacks:
            for callback in self.callbacks:
                callback.init_callback(self.rl_agent)

        for iteration in range(total_iterations):
            print(f"\nIteration {iteration + 1}/{total_iterations}")

            feedback_counts, reward_metrics = self.train_iteration(sampling_strategy)

            # Print progress
            print("\nFeedback counts:")
            for feedback_type, count in feedback_counts.items():
                print(f"{feedback_type}: {count}")

            print("\nReward model losses:")
            for feedback_type, loss in reward_metrics.items():
                print(f"{feedback_type}: {loss:.4f}")

        if self.wandb_logger is not None:
            wandb.finish()

        # Cleanup callbacks if they exist
        if self.callbacks:
            for callback in self.callbacks:
                callback.on_training_end()
        
        # Save final statistics
        final_stats_path = self.save_debug_statistics()
        print(f"Training complete. Final statistics saved to {final_stats_path}")
        return final_stats_path

def main():
    parser = TrainingUtils.setup_base_parser()
    parser.add_argument(
        "--feedback-types",
        nargs="+",
        type=str,
        #default=["evaluative", "comparative", "demonstrative", "descriptive", "corrective", "descriptive_preference"],
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
        default="trained_agents",
        help="Folder for finished feedback RL agents",
    )
    parser.add_argument(
        "--reference-data-folder",
        type=str,
        default="feedback",
        help="Folder containing pre-computed offline feedback for calibration",
    )
    parser.add_argument(
        "--n-feedback-per-iteration",
        type=int,
        default=30,
        help="Feedback Instances collected per iteration",
    )
    parser.add_argument(
        "--rl-steps-per-iteration",
        type=int,
        default=5000,
        help="Number of training iterations",
    )
    parser.add_argument(
        "--n-timesteps",
        type=int,
        default=-1,
        help="Overwrite for RL training timesteps",
    )
    parser.add_argument(
        "--reward-training-epochs",
        type=int,
        default=5,
        help="Number of epochs",
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
        default=1000,
        help="Maximum size of the feedback buffer",
    )
    parser.add_argument(
        "--num-ensemble-models",
        type=int,
        default=4,
        help="Number of ensemble models for masksemble",
    )
    args = parser.parse_args()

    # Setup oracle
    feedback_id, _ = TrainingUtils.get_model_ids(args)
    device = TrainingUtils.get_device()
    feedback_path = Path(args.reference_data_folder) / f"{feedback_id}.pkl"

    gen_environment = TrainingUtils.setup_environment(args.environment, args.seed)
    expert_models = TrainingUtils.load_expert_models(
        env_name=args.environment,
        algorithm=args.algorithm,
        checkpoints_path=str(get_project_root() / args.expert_model_base_path),
        environment=gen_environment,
        top_n_models=args.top_n_models,
    )

    oracle = FeedbackOracle(
        expert_models=expert_models,
        environment=gen_environment,
        reference_data_path=feedback_path,
        noise_level=args.noise_level,
    )
    unique_id = str(uuid.uuid4())[:8]

    # Create expert manager
    exp_manager = ExperimentManager(
        args=args,
        algo=args.algorithm,
        env_id=args.environment,
        log_folder=args.save_folder,
        eval_freq=5000,
        n_eval_episodes=5,
        use_wandb_callback=False,
        tensorboard_log="tb_logs",
    )

    # Setup experiment and get hyperparameters
    hyperparams = exp_manager.get_hyperparam_config_for_algo()

    # Create environment
    rl_env = exp_manager.create_envs(n_envs=exp_manager.n_envs)

    # Create DynamicRLHF with ExperimentManager's hyperparameters and callbacks
    drlhf = DynamicRLHFDebugging(
        oracle=oracle,
        env=rl_env,
        gen_env=gen_environment, # normal gym env for creating trajectories
        env_name=args.environment,
        algorithm=args.algorithm,
        feedback_types=args.feedback_types,
        n_feedback_per_iteration=args.n_feedback_per_iteration,
        feedback_buffer_size=args.feedback_buffer_size,
        rl_steps_per_iteration=args.rl_steps_per_iteration,
        reward_training_epochs=args.reward_training_epochs,
        num_ensemble_models=args.num_ensemble_models,
        hyperparams=hyperparams,
        callbacks=exp_manager.callbacks,
        device=device,
        wandb_logger=None,
        tensorboard_log=exp_manager.tensorboard_log,
        seed=args.seed,
    )

    # Train

    n_timesteps = args.n_timesteps if args.n_timesteps > 0 else exp_manager.n_timesteps
    total_iterations = max(1, n_timesteps // drlhf.rl_steps_per_iteration)
    drlhf.train(total_iterations=total_iterations, sampling_strategy="random")

if __name__ == "__main__":
    main()