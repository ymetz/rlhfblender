import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import pytorch_lightning
import torch
import wandb
from pytorch_lightning import Trainer
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from torch.utils.data import DataLoader

from multi_type_feedback.continuous_wandb_sb3_logger import (
    create_continuous_wandb_logger,
)
from multi_type_feedback.dynamic_rlhf_callback import RewardModelUpdateCallback
from multi_type_feedback.feedback_dataset import (
    BufferDataset,
)
from multi_type_feedback.feedback_oracle import FeedbackOracle
from multi_type_feedback.multi_head_networks import (
    MultiHeadNetwork,
    # MultiHeadCnnNetwork,
)
from multi_type_feedback.networks import (
    SingleCnnNetwork,
    SingleNetwork,
    calculate_pairwise_loss,
    calculate_single_reward_loss,
)
from multi_type_feedback.unified_dataset import (
    create_dataloaders_by_type,
    create_unified_dataloaders,
)
from multi_type_feedback.unified_networks import (
    UnifiedCnnNetwork,
    UnifiedNetwork,
)
from multi_type_feedback.utils import (
    L2RegulationCallback,
    RewardFn,
    TrainingUtils,
    get_project_root,
)
from multi_type_feedback.wandb_logger import ContinuousWandbLogger
from train_baselines.exp_manager import ExperimentManager


def one_hot_vector(k, max_val):
    vec = np.zeros(max_val)
    np.put(vec, k, 1)
    return vec


def vectorized_one_hot_vector(k, max_val):
    vec = np.zeros((k.size, max_val))
    vec[np.arange(k.size), k] = 1
    return vec


def compute_grouped(tensor, k):
    """
    Compute standard deviation for groups of elements spaced k apart.

    Args:
        tensor: Input tensor of shape (N,) where N is divisible by k
        k: Number of predictions per input

    Returns:
        Tensor of shape (N//k,) containing standard deviations
    """
    # Reshape the tensor to group related predictions together
    n_inputs = tensor.shape[0] // k
    reshaped = tensor.reshape(k, n_inputs).t()  # Shape: (n_inputs, k)

    # Compute standard deviation along dimension 1 (across the k predictions)
    return torch.mean(reshaped, dim=1), torch.std(reshaped, dim=1)  # Shape: (n_inputs,)


class DynamicRLHFRewardFunction(RewardFn):
    """
    Custom reward function that wraps the ensemble reward computation from DynamicRLHF.
    This makes it compatible with ExperimentManager's reward_function approach.
    """

    def __init__(self, drlhf_agent):
        super().__init__()
        self.drlhf_agent = drlhf_agent

    def __call__(
        self,
        state: np.ndarray,
        actions: np.ndarray,
        next_state: np.ndarray,
        _done: np.ndarray,
    ) -> np.ndarray:
        """Return reward given the current state and action."""
        return self.drlhf_agent.compute_ensemble_reward(state, actions)


class DynamicRLHF:
    def __init__(
        self,
        oracle: FeedbackOracle,
        env_name: str = "Pendulum-v1",
        env_kwargs: Dict[str, Any] = {},  # Environment configuration
        algorithm: str = "ppo",
        feedback_types: List[str] = [
            "evaluative",
            "comparative",
            "demonstrative",
            "descriptive",
        ],
        n_feedback_per_iteration: int = 50,
        feedback_buffer_size: int = 2000,
        rl_steps_per_iteration: int = 5000,
        reward_training_epochs: int = 10,
        device: str = "cuda",
        num_ensemble_models: int = 4,
        initial_feedback_count: int = 500,
        apply_random_response_handling: bool = False,
        callbacks: List[BaseCallback] = None,
        hyperparams: Dict[str, Any] = None,  # Hyperparameters from ExperimentManager
        seed: int = None,
        wandb_logger: Any = None,
        custom_sb3_logger: Any = None,
        reward_model_type: str = "separate",  # Options: "separate", "multi-head", "unified"
        shared_layer_num: int = 5,
        head_layer_num: int = 1,
        feedback_embedding_dim: int = 32,
        exp_manager: ExperimentManager = None,  # Add ExperimentManager
    ):
        self.oracle = oracle
        self.env_name = env_name
        self.env_kwargs = env_kwargs  # Store environment kwargs
        self.algorithm = algorithm
        self.feedback_types = feedback_types
        self.n_feedback_per_iteration = n_feedback_per_iteration
        self.feedback_buffer_size = feedback_buffer_size
        self.rl_steps_per_iteration = rl_steps_per_iteration
        self.reward_training_epochs = reward_training_epochs
        self.device = device
        self.num_ensemble_models = num_ensemble_models
        self.initial_feedback_count = initial_feedback_count
        self.external_callbacks = callbacks or []
        self._hyperparams = hyperparams or {}
        self.seed = seed
        self.wandb_logger = wandb_logger
        self.wandb = wandb  # Store reference to wandb module
        self.exp_manager = exp_manager  # Store the experiment manager

        self.reward_model_type = reward_model_type
        self.shared_layer_num = shared_layer_num
        self.head_layer_num = head_layer_num
        self.feedback_embedding_dim = feedback_embedding_dim

        # Create a temporary environment to get action space info using proper setup
        temp_env = TrainingUtils.setup_environment(env_name, seed, env_kwargs=env_kwargs)
        self.action_one_hot = isinstance(temp_env.action_space, gym.spaces.Discrete)
        if self.action_one_hot:
            self.one_hot_dim = temp_env.action_space.n
        temp_env.close()

        # Initialize feedback buffers for each type
        self.feedback_buffers = {feedback_type: [] for feedback_type in feedback_types}

        # Initialize reward models
        self.reward_models = self._init_reward_models()

        # Initialize Welford's algorithm state for reward standardization
        self.reward_mean = None
        self.squared_distance_from_mean = None
        self.reward_counters = None

        if apply_random_response_handling:
            self._apply_random_response_handling()

        # Create reward function wrapper
        self.reward_function = DynamicRLHFRewardFunction(self)

        # Update the experiment manager with our reward function
        if self.exp_manager:
            self.exp_manager.reward_function = self.reward_function

        # Perform initial reward model training before initializing RL agent
        if self.initial_feedback_count > 0:
            self.rl_agent = None  # need for collect_trajectories
            self._initialize_reward_models_with_random_feedback()

        # Initialize RL agent using ExperimentManager
        self.rl_agent = self._init_rl_agent()

        # set custom logger
        if custom_sb3_logger:
            self.rl_agent.set_logger(custom_sb3_logger)

    def _init_rl_agent(self) -> Union[PPO, SAC]:
        """Initialize the RL agent using ExperimentManager."""
        if self.exp_manager:
            # Use ExperimentManager to create the model
            results = self.exp_manager.setup_experiment()
            if results is not None:
                model, saved_hyperparams = results
                return model
            else:
                raise ValueError("ExperimentManager failed to setup experiment")
        else:
            # Fallback to the original method if no ExperimentManager
            temp_env = gym.make(self.env_name)
            if self.algorithm == "ppo":
                return PPO(
                    env=temp_env,
                    verbose=1,
                    seed=self.seed,
                    device=self.device,
                    **self._hyperparams,
                )
            else:
                return SAC(
                    env=TrainingUtils.setup_environment(self.env_name, self.seed, env_kwargs=self.env_kwargs),
                    verbose=1,
                    seed=self.seed,
                    device=self.device,
                    **self._hyperparams,
                )

    def _init_reward_models(self):
        """
        Initialize reward models based on chosen architecture type.
        """
        # Create a temporary environment to get spaces using proper setup
        temp_env = TrainingUtils.setup_environment(self.env_name, self.seed, env_kwargs=self.env_kwargs)
        observation_space = temp_env.observation_space
        action_space = temp_env.action_space
        temp_env.close()

        if self.reward_model_type == "separate":
            # Original implementation: separate models for each feedback type
            return self._init_separate_reward_models(observation_space, action_space)
        elif self.reward_model_type == "multi-head":
            # Multi-head model with shared backbone
            return self._init_multi_head_reward_model(observation_space, action_space)
        elif self.reward_model_type == "unified":
            # Unified model with feedback type conditioning
            return self._init_unified_reward_model(observation_space, action_space)
        else:
            raise ValueError(f"Unknown reward model type: {self.reward_model_type}")

    def _init_separate_reward_models(self, observation_space, action_space):
        """Initialize separate reward models for each feedback type (original implementation)."""
        reward_models = {}

        for feedback_type in self.feedback_types:
            if "ALE/" in self.env_name or "procgen" in self.env_name:
                model = SingleCnnNetwork(
                    input_spaces=(observation_space, action_space),
                    hidden_dim=256,
                    action_hidden_dim=16,
                    layer_num=3,
                    cnn_channels=(16, 32, 32),
                    output_dim=1,
                    loss_function=(
                        calculate_single_reward_loss
                        if feedback_type in ["evaluative", "descriptive", "supervised"]
                        else calculate_pairwise_loss
                    ),
                    learning_rate=1e-5,
                    ensemble_count=self.num_ensemble_models,
                )
            else:
                model = SingleNetwork(
                    input_spaces=(observation_space, action_space),
                    hidden_dim=256,
                    action_hidden_dim=32,
                    layer_num=6,
                    output_dim=1,
                    loss_function=(
                        calculate_single_reward_loss
                        if feedback_type in ["evaluative", "descriptive", "supervised"]
                        else calculate_pairwise_loss
                    ),
                    learning_rate=1e-5,
                    ensemble_count=self.num_ensemble_models,
                )
            reward_models[feedback_type] = model

        return reward_models

    def _init_multi_head_reward_model(self, observation_space, action_space):
        """Initialize a multi-head model with shared backbone."""

        # Create appropriate model based on environment
        if "ALE/" in self.env_name or "procgen" in self.env_name:
            model = MultiHeadCnnNetwork(
                input_spaces=(observation_space, action_space),
                shared_layer_num=self.shared_layer_num,
                head_layer_num=self.head_layer_num,
                hidden_dim=256,
                action_hidden_dim=16,
                output_dim=1,
                feedback_types=self.feedback_types,
                learning_rate=1e-5,
                cnn_channels=(16, 32, 32),
                ensemble_count=self.num_ensemble_models,
            )
        else:
            model = MultiHeadNetwork(
                input_spaces=(observation_space, action_space),
                shared_layer_num=self.shared_layer_num,
                head_layer_num=self.head_layer_num,
                hidden_dim=256,
                action_hidden_dim=32,
                output_dim=1,
                feedback_types=self.feedback_types,
                learning_rate=1e-5,
                ensemble_count=self.num_ensemble_models,
            )

        # For multi-head, we return a dictionary with a single key
        # This is to maintain compatibility with the rest of the code
        return {"multi_head": model}

    def _init_unified_reward_model(self, observation_space, action_space):
        """Initialize a unified model with feedback type conditioning."""
        # Create appropriate model based on environment
        if "ALE/" in self.env_name or "procgen" in self.env_name:
            model = UnifiedCnnNetwork(
                input_spaces=(observation_space, action_space),
                layer_num=3,
                hidden_dim=256,
                action_hidden_dim=16,
                output_dim=1,
                feedback_types=self.feedback_types,
                learning_rate=1e-5,
                cnn_channels=(16, 32, 32),
                ensemble_count=self.num_ensemble_models,
                feedback_embedding_dim=self.feedback_embedding_dim,
            )
        else:
            model = UnifiedNetwork(
                input_spaces=(observation_space, action_space),
                layer_num=6,
                hidden_dim=256,
                action_hidden_dim=32,
                output_dim=1,
                feedback_types=self.feedback_types,
                learning_rate=1e-5,
                ensemble_count=self.num_ensemble_models,
                feedback_embedding_dim=self.feedback_embedding_dim,
            )

        # For unified, we return a dictionary with a single key
        # This is to maintain compatibility with the rest of the code
        return {"unified": model}

    def _initialize_reward_models_with_random_feedback(self):
        """Collect initial random feedback and train reward models before RL training begins."""
        print(f"\nInitializing reward models with {self.initial_feedback_count} random feedback samples...")

        # Create a temporary environment for trajectory collection using proper setup
        temp_env = TrainingUtils.setup_environment(self.env_name, self.seed, env_kwargs=self.env_kwargs)

        # Calculate how many batches of trajectories to collect
        batches_needed = (self.initial_feedback_count + self.n_feedback_per_iteration - 1) // self.n_feedback_per_iteration
        total_feedback_collected = 0
        feedback_counts = defaultdict(int)

        for batch in range(batches_needed):

            # Collect random trajectories
            trajectories, initial_states = self.collect_trajectories(self.n_feedback_per_iteration, temp_env)

            # Always use random sampling for initial feedback
            feedback, batch_counts = self.sample_feedback_random(trajectories, initial_states)

            # Update feedback counts
            for feedback_type, count in batch_counts.items():
                feedback_counts[feedback_type] += count
                total_feedback_collected += count

            # Update feedback buffers
            self.update_feedback_buffers(feedback)

            # Log progress if wandb is available
            if self.wandb_logger is not None and hasattr(self.wandb, "run") and self.wandb.run is not None:
                metrics_to_log = {}
                for feedback_type, count in feedback_counts.items():
                    metrics_to_log[f"initial_feedback/{feedback_type}_count"] = count
                metrics_to_log["initial_feedback/total_collected"] = total_feedback_collected
                metrics_to_log["initial_feedback/percent_complete"] = (
                    total_feedback_collected / self.initial_feedback_count
                ) * 100
                self.wandb.log(metrics_to_log)

            if total_feedback_collected >= self.initial_feedback_count:
                break

        temp_env.close()

        # Train the reward models with more epochs for initial training
        initial_training_epochs = self.reward_training_epochs * 2  # Train longer initially
        reward_metrics = self._train_reward_models_with_epochs(initial_training_epochs)

        print("\nInitial feedback counts:")
        for feedback_type, count in feedback_counts.items():
            print(f"{feedback_type}: {count}")

        print("\nInitial reward model losses:")
        for feedback_type, loss in reward_metrics.items():
            print(f"{feedback_type}: {loss:.4f}")

        # Log initial training metrics
        if self.wandb_logger is not None and hasattr(self.wandb, "run") and self.wandb.run is not None:
            metrics_to_log = {}
            for feedback_type, loss in reward_metrics.items():
                metrics_to_log[f"initial_reward_model/{feedback_type}_loss"] = loss
            self.wandb.log(metrics_to_log)

    def collect_trajectories(
        self, n_trajectories: int, env: gym.Env = None
    ) -> Tuple[List[List[Tuple[np.ndarray, np.ndarray, float, bool]]], List[Any]]:
        """Collect trajectories using current policy."""
        if env is None:
            env = TrainingUtils.setup_environment(self.env_name, self.seed, env_kwargs=self.env_kwargs)
            should_close = True
        else:
            should_close = False

        trajectories = []
        initial_states = []

        for _ in range(n_trajectories):
            trajectory = []
            obs, _ = env.reset()
            # Use the original approach for saving initial states
            initial_states.append(env.save_state(observation=obs))

            for _ in range(self.oracle.segment_len):
                if self.rl_agent is None:
                    # this is the case for initial generation, use random agent here
                    action = env.action_space.sample()
                else:
                    action, _ = self.rl_agent.predict(obs, deterministic=False)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                if self.action_one_hot:
                    action = one_hot_vector(action, self.one_hot_dim)
                done = terminated or truncated

                trajectory.append((np.expand_dims(obs, axis=0), action, reward, done))
                obs = next_obs

                if done:
                    break

            trajectories.append(trajectory)

        if should_close:
            env.close()

        return trajectories, initial_states

    def _train_reward_models_with_epochs(self, max_epochs=None):
        """
        Train reward models with specified number of epochs.
        Modified to handle different reward model architectures.
        """
        reward_metrics = {}

        # Use default epochs if not specified
        if max_epochs is None:
            max_epochs = self.reward_training_epochs

        if self.reward_model_type == "separate":
            # Original implementation: train separate models for each feedback type
            for feedback_type in self.feedback_types:
                buffer_data = self.feedback_buffers[feedback_type]
                if not buffer_data:
                    continue

                print(f"TRAINING REWARD MODEL FOR F.B. TYPE: {feedback_type}")

                # Create dataset from buffer
                full_dataset = BufferDataset(buffer_data)

                # Split dataset for validation
                val_size = int(len(full_dataset) * 0.368)
                train_size = len(full_dataset) - val_size

                if train_size <= 0 or val_size <= 0:
                    print(f"Skipping {feedback_type} training: insufficient data ({len(full_dataset)} samples)")
                    continue

                train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

                # Setup data loaders
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=self.num_ensemble_models,
                    shuffle=True,
                    pin_memory=True,
                    drop_last=True,
                )

                val_loader = DataLoader(
                    val_dataset,
                    batch_size=self.num_ensemble_models,
                    shuffle=False,
                    pin_memory=True,
                    drop_last=True,
                )

                # Configure callbacks and trainer
                callbacks = [
                    L2RegulationCallback(initial_l2=0.01),
                    pytorch_lightning.callbacks.EarlyStopping(monitor="val_loss", patience=3, mode="min"),
                ]

                trainer = Trainer(
                    max_epochs=max_epochs,
                    accelerator="auto",
                    devices="auto",
                    enable_progress_bar=False,
                    accumulate_grad_batches=32,
                    callbacks=callbacks,
                    logger=self.wandb_logger or False,
                    check_val_every_n_epoch=1,
                    enable_model_summary=False,
                    enable_checkpointing=False,
                )

                # Train the model
                trainer.fit(
                    self.reward_models[feedback_type],
                    train_dataloaders=train_loader,
                    val_dataloaders=val_loader,
                )

                # Extract final metrics
                final_metrics = trainer.callback_metrics
                train_loss = float(final_metrics.get("train_loss", -1.0))
                val_loss = float(final_metrics.get("val_loss", -1.0))

                reward_metrics[feedback_type] = val_loss

                print(f"{feedback_type} training complete: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        elif self.reward_model_type == "multi-head":
            # Multi-head implementation
            model = list(self.reward_models.values())[0]  # Only one model in dict

            # Check if we have any data to train on
            has_data = False
            for feedback_type in self.feedback_types:
                if self.feedback_buffers[feedback_type]:
                    has_data = True
                    break

            if not has_data:
                print("No data available for training")
                return {}

            # Create data loaders by feedback type
            dataloaders = create_dataloaders_by_type(
                self.feedback_buffers,
                batch_size=self.num_ensemble_models,
                val_split=0.368,
            )

            # Skip training if no dataloaders
            if not dataloaders:
                return {}

            # Configure callbacks
            callbacks = [
                L2RegulationCallback(initial_l2=0.01),
                pytorch_lightning.callbacks.EarlyStopping(monitor="val_loss", patience=3, mode="min"),
            ]

            # Train for each feedback type separately (but using the shared model)
            for feedback_type, (train_loader, val_loader) in dataloaders.items():
                print(f"Training multi-head model for {feedback_type}")

                # Configure trainer
                trainer = Trainer(
                    max_epochs=max_epochs,
                    accelerator="auto",
                    devices="auto",
                    enable_progress_bar=False,
                    accumulate_grad_batches=32,
                    callbacks=callbacks,
                    logger=self.wandb_logger or False,
                    check_val_every_n_epoch=1,
                    enable_model_summary=False,
                    enable_checkpointing=False,
                )

                # Train the model
                trainer.fit(
                    model,
                    train_dataloaders=train_loader,
                    val_dataloaders=val_loader,
                )

                # Extract final metrics
                final_metrics = trainer.callback_metrics
                val_loss = float(final_metrics.get(f"val_loss_{feedback_type}", -1.0))
                reward_metrics[feedback_type] = val_loss

                print(f"{feedback_type} training complete: val_loss={val_loss:.4f}")

        elif self.reward_model_type == "unified":
            # Unified implementation
            model = list(self.reward_models.values())[0]  # Only one model in dict

            # Check if we have any data to train on
            has_data = False
            for feedback_type in self.feedback_types:
                if self.feedback_buffers[feedback_type]:
                    has_data = True
                    break

            if not has_data:
                print("No data available for training")
                return {}

            # Create unified data module
            train_dataloader, val_dataloader = create_unified_dataloaders(
                self.feedback_buffers,
                batch_size=1,
                val_split=0.368,
            )

            # Configure callbacks
            callbacks = [
                L2RegulationCallback(initial_l2=0.01),
                pytorch_lightning.callbacks.EarlyStopping(monitor="val_loss", patience=3, mode="min"),
            ]

            trainer = Trainer(
                max_epochs=max_epochs,
                accelerator="auto",
                devices="auto",
                enable_progress_bar=False,
                accumulate_grad_batches=32,
                callbacks=callbacks,
                logger=self.wandb_logger or False,
                check_val_every_n_epoch=1,
                enable_model_summary=False,
                enable_checkpointing=False,
            )

            # Train the model
            trainer.fit(
                model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
            )

            # Extract final metrics for each feedback type
            final_metrics = trainer.callback_metrics

            for feedback_type in self.feedback_types:
                val_loss_key = f"val_loss_{feedback_type}"
                if val_loss_key in final_metrics:
                    reward_metrics[feedback_type] = float(final_metrics[val_loss_key])

            # Also add overall val_loss
            if "val_loss" in final_metrics:
                reward_metrics["overall"] = float(final_metrics["val_loss"])

            print(f"Unified model training complete")
            for fb_type, loss in reward_metrics.items():
                print(f"  {fb_type}: val_loss={loss:.4f}")

        return reward_metrics

    def _apply_random_response_handling(self):
        """Apply 10% random response handling to comparative loss functions."""
        # Store original loss functions
        original_loss_functions = {}

        for feedback_type in self.feedback_types:
            if feedback_type in [
                "comparative",
                "descriptive_preference",
                "demonstrative",
                "corrective",
            ]:
                # Save original function
                original_loss = self.reward_models[feedback_type].loss_function
                original_loss_functions[feedback_type] = original_loss

                # Create a new loss function that accounts for random responses
                def modified_loss_function(network, batch, orig_loss=original_loss):
                    # For pairwise comparisons
                    if hasattr(orig_loss, "__name__") and orig_loss.__name__ == "calculate_pairwise_loss":
                        (pair_obs, pair_actions, pair_masks), preferred_indices = batch

                        # Get observations/actions for both trajectories
                        obs1, obs2 = pair_obs[0], pair_obs[1]
                        actions1, actions2 = pair_actions[0], pair_actions[1]

                        # Get rewards from network
                        outputs1 = network(obs1, actions1)
                        outputs2 = network(obs2, actions2)

                        # Sum rewards over trajectory
                        rewards1 = outputs1.sum(dim=1).squeeze(-1)
                        rewards2 = outputs2.sum(dim=1).squeeze(-1)

                        # Calculate reward differences
                        reward_diff = rewards1 - rewards2

                        # Apply 10% random response probability
                        # P(choose 1) = 0.9 * sigmoid(r1-r2) + 0.05
                        probs = 0.9 * torch.sigmoid(reward_diff) + 0.05

                        # Get probability of the chosen trajectory
                        chosen_probs = torch.where(preferred_indices == 0, probs, 1 - probs)

                        # Negative log likelihood loss
                        loss = -torch.mean(torch.log(chosen_probs + 1e-8))
                        return loss
                    else:
                        # For other loss types, use original
                        return orig_loss(network, batch)

                # Assign modified loss function
                self.reward_models[feedback_type].loss_function = modified_loss_function

        return original_loss_functions

    def compute_model_uncertainty(
        self,
        trajectory: List[Tuple[np.ndarray, np.ndarray, float, bool]],
        feedback_type: str,
    ) -> float:
        """Compute uncertainty for a trajectory using the ensemble variance of the specific reward model."""
        reward_model = self.reward_models[feedback_type]

        # Stack observations and actions from trajectory
        states = torch.vstack([torch.as_tensor(step[0]).float() for step in trajectory]).to(self.device)
        actions = torch.vstack([torch.as_tensor(step[1]).float() for step in trajectory]).to(self.device)

        # Get predictions from all ensemble members
        with torch.no_grad():
            if reward_model.ensemble_count > 1:
                states_expanded = states.unsqueeze(0).expand(reward_model.ensemble_count, *states.shape)
                actions_expanded = actions.unsqueeze(0).expand(reward_model.ensemble_count, *actions.shape)
                predictions = reward_model(states_expanded, actions_expanded)  # Shape: [ensemble_size, traj_len, 1]

                # Compute trajectory-level uncertainty as mean of step-wise uncertainties
                step_uncertainties = predictions.std(dim=0)  # Standard deviation across ensemble members
                trajectory_uncertainty = step_uncertainties.mean().item()  # Mean uncertainty across trajectory
            else:
                trajectory_uncertainty = 0.0

        return trajectory_uncertainty

    def compute_trajectory_overall_uncertainty(
        self, trajectory: List[Tuple[np.ndarray, np.ndarray, float, bool]], strategy: str = "average"
    ) -> float:
        """
        Compute overall uncertainty for a trajectory across all feedback types.

        Args:
            trajectory: Single trajectory to compute uncertainty for
            strategy: How to combine uncertainties across feedback types ("average", "min", "max")

        Returns:
            Overall uncertainty score for the trajectory
        """
        uncertainties = []

        for feedback_type in self.feedback_types:
            if len(self.feedback_buffers[feedback_type]) > 0:  # Only if model has been trained
                uncertainty = self.compute_model_uncertainty(trajectory, feedback_type)
                # Basic normalization could be added here if needed in the future
                uncertainties.append(uncertainty)
            else:
                # If no feedback yet, set high uncertainty to encourage exploration
                uncertainties.append(float("inf"))

        if not uncertainties:
            return 0.0

        # Handle infinite uncertainties (untrained models)
        if any(u == float("inf") for u in uncertainties):
            return float("inf")

        # Combine uncertainties based on strategy
        if strategy == "average":
            return np.mean(uncertainties)
        elif strategy == "min":
            return np.min(uncertainties)
        elif strategy == "max":
            return np.max(uncertainties)
        else:
            raise ValueError(f"Unknown uncertainty combination strategy: {strategy}")

    def select_queries_by_uncertainty(
        self, trajectories: List[List], initial_states: List[np.ndarray], n_queries: int, strategy: str = "average"
    ) -> tuple[List[List], List[np.ndarray]]:
        """
        Select top N queries based on model uncertainty.

        Args:
            trajectories: List of trajectories to select from
            initial_states: Corresponding initial states
            n_queries: Number of queries to select
            strategy: How to combine uncertainties across feedback types

        Returns:
            Selected trajectories and their initial states
        """
        if len(trajectories) <= n_queries:
            return trajectories, initial_states

        # Compute overall uncertainty for each trajectory
        trajectory_uncertainties = []
        for trajectory in trajectories:
            uncertainty = self.compute_trajectory_overall_uncertainty(trajectory, strategy)
            trajectory_uncertainties.append(uncertainty)

        # Handle case where some trajectories have infinite uncertainty
        finite_uncertainties = [u for u in trajectory_uncertainties if u != float("inf")]
        if len(finite_uncertainties) < len(trajectory_uncertainties):
            # Prioritize trajectories with infinite uncertainty (untrained models)
            inf_indices = [i for i, u in enumerate(trajectory_uncertainties) if u == float("inf")]
            finite_indices = [i for i, u in enumerate(trajectory_uncertainties) if u != float("inf")]

            # Take all infinite uncertainty trajectories first, then top finite ones
            selected_indices = inf_indices[:n_queries]
            if len(selected_indices) < n_queries:
                remaining_needed = n_queries - len(selected_indices)
                finite_uncertainties_with_idx = [
                    (finite_indices[i], trajectory_uncertainties[finite_indices[i]]) for i in range(len(finite_indices))
                ]
                finite_uncertainties_with_idx.sort(key=lambda x: x[1], reverse=True)
                selected_indices.extend([idx for idx, _ in finite_uncertainties_with_idx[:remaining_needed]])
        else:
            # All uncertainties are finite, select top N
            uncertainty_with_idx = [(i, u) for i, u in enumerate(trajectory_uncertainties)]
            uncertainty_with_idx.sort(key=lambda x: x[1], reverse=True)
            selected_indices = [idx for idx, _ in uncertainty_with_idx[:n_queries]]

        # Return selected trajectories and initial states
        selected_trajectories = [trajectories[i] for i in selected_indices]
        selected_initial_states = [initial_states[i] for i in selected_indices]

        return selected_trajectories, selected_initial_states

    def sample_feedback_uncertainty(
        self, trajectories: List[List], initial_states: List[np.ndarray]
    ) -> tuple[List[Dict], Dict[str, int]]:
        """Sample feedback types based on ensemble variance for each reward model."""
        # Calculate uncertainties for each trajectory and feedback type
        trajectory_uncertainties = []

        for trajectory in trajectories:
            uncertainties = {}
            for feedback_type in self.feedback_types:
                if len(self.feedback_buffers[feedback_type]) > 0:  # Only if model has been trained
                    uncertainty = self.compute_model_uncertainty(trajectory, feedback_type)
                else:
                    # If no feedback yet, set high uncertainty to encourage exploration
                    uncertainty = float("inf")
                uncertainties[feedback_type] = uncertainty
            trajectory_uncertainties.append(uncertainties)

        # Sample feedback types based on uncertainties
        feedback_counts = defaultdict(int)
        all_feedback = []

        # For each trajectory, sample feedback type with probability proportional to uncertainty
        for trajectory, initial_state, uncertainties in zip(trajectories, initial_states, trajectory_uncertainties):
            # Normalize uncertainties to probabilities
            total_uncertainty = sum(uncertainties.values())
            if total_uncertainty == float("inf"):
                # If no feedback yet for some types, sample uniformly from those
                untrained_types = [ft for ft in self.feedback_types if len(self.feedback_buffers[ft]) == 0]
                feedback_type = np.random.choice(untrained_types)
            else:
                probs = [uncertainties[ft] / total_uncertainty for ft in self.feedback_types]
                feedback_type = np.random.choice(self.feedback_types, p=probs)

            # Handle different feedback types
            feedback_dict = {}
            if feedback_type in ["comparative", "descriptive_preference"]:
                # Need a second trajectory for comparison
                trajectory2, _ = self.collect_trajectories(1)
                feedback = self.oracle.get_feedback((trajectory, trajectory2[0]), initial_state, feedback_type)
            else:
                feedback = self.oracle.get_feedback(trajectory, initial_state, feedback_type)

            feedback_dict[feedback_type] = feedback
            feedback_counts[feedback_type] += 1
            all_feedback.append(feedback_dict)

        return all_feedback, feedback_counts

    def sample_feedback_random(
        self, trajectories: List[List], initial_states: List[np.ndarray]
    ) -> tuple[List[Dict], Dict[str, int]]:
        """Randomly sample feedback types."""
        feedback_distribution = np.ones(len(self.feedback_types)) / len(self.feedback_types)
        selected_types = np.random.choice(
            self.feedback_types,
            size=len(trajectories),
            p=feedback_distribution,
        )

        feedback_counts = defaultdict(int)
        all_feedback = []

        for trajectory, initial_state, feedback_type in zip(trajectories, initial_states, selected_types):
            feedback_dict = {}

            # Handle different feedback types
            if feedback_type in ["comparative", "descriptive_preference"]:
                # Need a second trajectory for comparison
                trajectory2, _ = self.collect_trajectories(1)
                feedback = self.oracle.get_feedback((trajectory, trajectory2[0]), initial_state, feedback_type)
            else:
                feedback = self.oracle.get_feedback(trajectory, initial_state, feedback_type)

            feedback_dict[feedback_type] = feedback
            feedback_counts[feedback_type] += 1
            all_feedback.append(feedback_dict)

        return all_feedback, feedback_counts

    def update_feedback_buffers(self, new_feedback: List[Dict]):
        """Update feedback buffers with new feedback while maintaining size limit."""
        for feedback_dict in new_feedback:
            for feedback_type, feedback in feedback_dict.items():
                if feedback_type != "uncertainty":  # Skip uncertainty metadata
                    if feedback_type == "supervised":
                        # for supervised feedback, we get a list of states and associated rewards
                        # so extend instead of append
                        if len(self.feedback_buffers[feedback_type]) >= self.feedback_buffer_size:
                            # Remove oldest feedback
                            self.feedback_buffers[feedback_type] = self.feedback_buffers[feedback_type][len(feedback) :]
                        self.feedback_buffers[feedback_type].extend(feedback)
                    else:
                        if len(self.feedback_buffers[feedback_type]) >= self.feedback_buffer_size:
                            # Remove oldest feedback
                            self.feedback_buffers[feedback_type].pop(0)
                        self.feedback_buffers[feedback_type].append(feedback)

    def train_reward_models(self):
        """Train reward models with default number of epochs."""
        return self._train_reward_models_with_epochs(self.reward_training_epochs)

    def standardize_rewards(self, rewards: torch.Tensor):
        """
        Standardizes the input using the rolling mean and standard deviation of the rewards.
        Uses Welford's algorithm for numerically stable online computation.

        Input should be a tensor of shape (batch_size, model_count).
        """
        model_count = rewards.shape[1]

        if self.reward_mean is None:
            self.reward_mean = torch.zeros(model_count).to(self.device)

        if self.squared_distance_from_mean is None:
            self.squared_distance_from_mean = torch.zeros(model_count).to(self.device)

        if self.reward_counters is None:
            self.reward_counters = torch.zeros(model_count).to(self.device)

        standard_deviation = torch.ones(model_count).to(self.device)

        for batch_idx in range(rewards.shape[0]):
            for reward_index in range(model_count):
                reward = rewards[batch_idx, reward_index]

                # Welford's algorithm for calculating running mean and variance
                self.reward_counters[reward_index] += 1

                difference = reward - self.reward_mean[reward_index]
                self.reward_mean[reward_index] += difference / self.reward_counters[reward_index]
                new_difference = reward - self.reward_mean[reward_index]
                self.squared_distance_from_mean[reward_index] += difference * new_difference

                if self.reward_counters[reward_index] > 1:
                    variance = self.squared_distance_from_mean[reward_index] / (self.reward_counters[reward_index] - 1)
                    standard_deviation[reward_index] = torch.sqrt(variance)

                rewards[batch_idx, reward_index] = (reward - self.reward_mean[reward_index]) / standard_deviation[reward_index]

        return rewards

    def compute_ensemble_reward(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Compute ensemble reward prediction based on model architecture.
        Modified to handle different reward model architectures.
        """
        device = self.device

        # Handle one-hot encoding for discrete actions
        if self.action_one_hot:
            action = vectorized_one_hot_vector(np.array(action), self.one_hot_dim)

        # Add batch dimension to actions if not present
        if len(action.shape) < 2:
            action = np.expand_dims(action, axis=0)

        # Convert to torch tensors of shape [batch_size, ...]
        state_tensor = torch.as_tensor(state, device=device, dtype=torch.float32).unsqueeze(1)
        action_tensor = torch.as_tensor(action, device=device, dtype=torch.float32).unsqueeze(1)

        # Lists to accumulate each model's reward and uncertainty
        model_rewards = []
        model_uncertainties = []

        with torch.no_grad():
            if self.reward_model_type == "separate":
                # Original implementation: separate models for each feedback type
                for feedback_type, reward_model in self.reward_models.items():
                    # Only use models which have some feedback
                    if len(self.feedback_buffers[feedback_type]) == 0:
                        continue

                    if reward_model.ensemble_count > 1:
                        # Expand along ensemble dimension
                        st_expanded = state_tensor.repeat(
                            reward_model.ensemble_count,
                            *[1] * (len(state_tensor.shape) - 1),
                        )
                        act_expanded = action_tensor.repeat(
                            reward_model.ensemble_count,
                            *[1] * (len(action_tensor.shape) - 1),
                        )

                        # Get predictions
                        predictions = reward_model(st_expanded, act_expanded)

                        # Make sure we reduce the final dimension if necessary
                        if predictions.dim() == 3 and predictions.shape[-1] == 1:
                            predictions = predictions.squeeze(-1)

                        mean_reward, uncertainty = compute_grouped(predictions, reward_model.ensemble_count)
                    else:
                        # Single model in the ensemble
                        predictions = reward_model(state_tensor, action_tensor)
                        if predictions.dim() == 2 and predictions.shape[1] == 1:
                            predictions = predictions.squeeze(-1)
                        mean_reward = predictions
                        uncertainty = torch.zeros_like(mean_reward)

                    # Collect
                    model_rewards.append(mean_reward)  # shape [batch_size,]
                    model_uncertainties.append(uncertainty)  # shape [batch_size,]

            elif self.reward_model_type == "multi-head":
                # Multi-head model: get predictions from each head
                multi_head_model = list(self.reward_models.values())[0]  # Only one model

                # Get predictions for all heads at once
                st_expanded = state_tensor.repeat(
                    multi_head_model.ensemble_count,
                    *[1] * (len(state_tensor.shape) - 1),
                )
                act_expanded = action_tensor.repeat(
                    multi_head_model.ensemble_count,
                    *[1] * (len(action_tensor.shape) - 1),
                )

                # Forward pass with no specific feedback type to get all heads
                all_outputs = multi_head_model(st_expanded, act_expanded)

                # Process each feedback type's output
                for feedback_type, outputs in all_outputs.items():
                    # Only use heads which have some feedback
                    if len(self.feedback_buffers[feedback_type]) == 0:
                        continue

                    # Make sure we reduce the final dimension if necessary
                    if outputs.dim() == 3 and outputs.shape[-1] == 1:
                        outputs = outputs.squeeze(-1)

                    mean_reward, uncertainty = compute_grouped(outputs, multi_head_model.ensemble_count)

                    # Collect
                    model_rewards.append(mean_reward)  # shape [batch_size,]
                    model_uncertainties.append(uncertainty)  # shape [batch_size,]

            elif self.reward_model_type == "unified":
                # Unified model: get predictions for each feedback type
                unified_model = list(self.reward_models.values())[0]  # Only one model

                # For each feedback type that has data, get predictions
                for feedback_type in self.feedback_types:
                    # Only use feedback types which have some feedback
                    if len(self.feedback_buffers[feedback_type]) == 0:
                        continue

                    # Expand along ensemble dimension
                    st_expanded = state_tensor.repeat(
                        unified_model.ensemble_count,
                        *[1] * (len(state_tensor.shape) - 1),
                    )
                    act_expanded = action_tensor.repeat(
                        unified_model.ensemble_count,
                        *[1] * (len(action_tensor.shape) - 1),
                    )

                    # Forward pass with specific feedback type
                    predictions = unified_model(st_expanded, act_expanded, feedback_type)

                    mean_reward, uncertainty = compute_grouped(predictions, unified_model.ensemble_count)

                    # Collect
                    model_rewards.append(mean_reward)  # shape [batch_size,]
                    model_uncertainties.append(uncertainty)  # shape [batch_size,]

        # If no models have feedback, return zeros for the entire batch
        if not model_rewards:
            return np.zeros(state.shape[0], dtype=np.float32)

        # Stack across models => shape (#models, batch_size)
        stacked_rewards = torch.stack(model_rewards, dim=0)
        stacked_uncerts = torch.stack(model_uncertainties, dim=0)

        # Apply standardization using Welford's algorithm
        # Transpose to (batch_size, #models) for standardization, then transpose back
        rewards_for_standardization = stacked_rewards.transpose(0, 1)  # shape (batch_size, #models)
        standardized_rewards = self.standardize_rewards(rewards_for_standardization)
        stacked_rewards = standardized_rewards.transpose(0, 1)  # back to (#models, batch_size)

        # Calculate final rewards => shape [batch_size,]
        batch_size = state.shape[0]
        final_rewards = torch.zeros(batch_size, device=device, dtype=torch.float32)

        # Loop over each environment in the batch
        for i in range(batch_size):
            # For the i-th environment, gather all model rewards/uncertainties
            r_i = stacked_rewards[:, i]  # shape (#models,)
            u_i = stacked_uncerts[:, i]  # shape (#models,)

            if torch.any(u_i > 0):
                # If any model has a positive uncertainty, weight by 1 / uncertainty
                w_i = torch.where(u_i > 0, 1.0 / u_i, torch.ones_like(u_i))
                # Normalize weights
                w_i /= w_i.sum()
                final_rewards[i] = (r_i * w_i).sum()
            else:
                # Otherwise, just average over the models
                final_rewards[i] = r_i.mean()

        return final_rewards.cpu().numpy()  # shape: [batch_size,]

    def train(
        self,
        total_timesteps: int,
        sampling_strategy: str = "random",
        query_sampling_strategy: str = "none",
        query_sampling_multiplier: float = 2.0,
    ):
        """
        Run full training loop with a single call to learn() and using callbacks
        for reward model updates.
        """

        # Create reward model callback
        reward_model_callback = RewardModelUpdateCallback(
            drlhf_agent=self,
            update_freq=self.rl_steps_per_iteration,
            sampling_strategy=sampling_strategy,
            query_sampling_strategy=query_sampling_strategy,
            query_sampling_multiplier=query_sampling_multiplier,
            verbose=1,
        )

        # Combine with other callbacks
        if self.external_callbacks:
            if isinstance(self.external_callbacks, list):
                all_callbacks = [reward_model_callback] + self.external_callbacks
                callback = CallbackList(all_callbacks)
            else:
                # If it's a single callback, create a list
                callback = CallbackList([reward_model_callback, self.external_callbacks])
        else:
            callback = reward_model_callback

        # Train with a single call to learn()
        if self.exp_manager:
            # Use ExperimentManager's learn method
            self.exp_manager.learn(self.rl_agent)
        else:
            # Fallback to direct learning
            self.rl_agent.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                reset_num_timesteps=True,
            )

        # Clean up wandb if needed
        if self.wandb_logger is not None and hasattr(self.wandb_logger, "experiment"):
            # Only finish if we own the wandb run
            if self.wandb_logger.experiment is self.wandb.run:
                self.wandb.finish()


def main():
    parser = TrainingUtils.setup_base_parser()
    parser.add_argument(
        "--feedback-types",
        nargs="+",
        type=str,
        default=[
            "evaluative",
            "comparative",
            "demonstrative",
            "corrective",
            "descriptive",
            "descriptive_preference",
        ],
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
        "--query-sampling-strategy",
        type=str,
        default="none",
        choices=["none", "average", "min", "max"],
        help="Query selection strategy based on uncertainty",
    )
    parser.add_argument(
        "--query-sampling-multiplier",
        type=float,
        default=2.0,
        help="Multiplier for number of queries to sample before filtering",
    )
    parser.add_argument(
        "--reward-model-type",
        type=str,
        default="separate",
        choices=["separate", "multi-head", "unified"],
        help="Reward Model mode",
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
        default=50,
        help="Feedback Instances collected per iteration",
    )
    parser.add_argument(
        "--rl-steps-per-iteration",
        type=int,
        default=10000,
        help="Number of steps between reward model updates",
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
        default=20,
        help="Number of epochs",
    )
    parser.add_argument(
        "--initial-feedback-count",
        type=int,
        default=500,
        help="Number of feedback samples to collect before starting RL training",
    )
    parser.add_argument(
        "--feedback-buffer-size",
        type=int,
        default=5000,
        help="Maximum size of the feedback buffer",
    )
    parser.add_argument("--top-n-models", type=int, default=3, help="Top N models to use")
    parser.add_argument(
        "--random-response-handling",
        action="store_true",
        default=False,
        help="Disable the 10% random response handling from Christiano et al.",
    )
    parser.add_argument(
        "--expert-model-base-path",
        type=str,
        default="gt_agents",
        help="Expert model base path",
    )
    parser.add_argument(
        "--num-ensemble-models",
        type=int,
        default=4,
        help="Number of ensemble models for masksemble",
    )
    parser.add_argument(
        "--shared-layer-number",
        type=int,
        default=5,
        help="Number of shared layers for multi-head policy",
    )
    parser.add_argument(
        "--head-layer-num",
        type=int,
        default=1,
        help="Number of layers for prediction head in multi-head policy",
    )
    args = parser.parse_args()

    # Setup oracle
    feedback_id, _ = TrainingUtils.get_model_ids(args)
    device = TrainingUtils.get_device()
    feedback_path = Path(args.reference_data_folder) / f"{feedback_id}.pkl"

    gen_environment = TrainingUtils.setup_environment(
        args.environment, args.seed, env_kwargs=TrainingUtils.parse_env_kwargs(args.environment_kwargs)
    )
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
    # unique_id = str(uuid.uuid4())[:8]

    reward_model_type = args.reward_model_type if len(args.feedback_types) > 1 else f"single_{''.join(args.feedback_types)}"

    # Initialize wandb
    wandb.init(
        name=f"DYNAMIC_RL_{args.algorithm}_{args.environment}_{reward_model_type}_{args.seed}",
        project=args.wandb_project_name,
        config={
            "algorithm": args.algorithm,
            "feedback_types": args.feedback_types,
            "n_feedback_per_iteration": args.n_feedback_per_iteration,
            "rl_steps_per_iteration": args.rl_steps_per_iteration,
            "reward_training_epochs": args.reward_training_epochs,
            "feedback_buffer_size": args.feedback_buffer_size,
            "reward_model_type": args.reward_model_type,
            "query_sampling_strategy": args.query_sampling_strategy,
        },
    )

    continuous_lightning_logger = ContinuousWandbLogger()

    # Create experiment manager
    exp_manager = ExperimentManager(
        args=args,
        algo=args.algorithm,
        env_id=args.environment,
        log_folder=args.save_folder,
        eval_freq=5000,
        n_eval_episodes=5,
        use_wandb_callback=True,
        wandb_callback_continuous=True,
        # Note: reward_function will be set by DynamicRLHF
        reward_function=None,
    )

    # Setup experiment and get hyperparameters
    hyperparams = exp_manager.get_hyperparam_config_for_algo()

    custom_sb3_logger = create_continuous_wandb_logger(
        global_step_offset=0,
        run_id=wandb.run.id,  # Reuse the same W&B run
        additional_formats=["stdout"],
        folder="logs",
    )

    # Create DynamicRLHF with ExperimentManager
    drlhf = DynamicRLHF(
        oracle=oracle,
        env_name=args.environment,
        algorithm=args.algorithm,
        feedback_types=args.feedback_types,
        n_feedback_per_iteration=args.n_feedback_per_iteration,
        feedback_buffer_size=args.feedback_buffer_size,
        rl_steps_per_iteration=args.rl_steps_per_iteration,
        reward_training_epochs=args.reward_training_epochs,
        num_ensemble_models=args.num_ensemble_models,
        apply_random_response_handling=args.random_response_handling,
        initial_feedback_count=args.initial_feedback_count,
        hyperparams=hyperparams,
        callbacks=exp_manager.callbacks,
        device=device,
        wandb_logger=continuous_lightning_logger,
        custom_sb3_logger=custom_sb3_logger,
        seed=args.seed,
        reward_model_type=args.reward_model_type,
        shared_layer_num=args.shared_layer_number,
        head_layer_num=args.head_layer_num,
        exp_manager=exp_manager,
    )

    n_timesteps = args.n_timesteps if args.n_timesteps > 0 else exp_manager.n_timesteps
    drlhf.train(
        total_timesteps=n_timesteps,
        sampling_strategy=args.sampling_strategy,
        query_sampling_strategy=args.query_sampling_strategy,
        query_sampling_multiplier=args.query_sampling_multiplier,
    )


if __name__ == "__main__":
    main()
