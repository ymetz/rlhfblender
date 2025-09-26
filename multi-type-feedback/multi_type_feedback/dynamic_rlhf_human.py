import os
import uuid
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Union

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
    LoadFeedbackDataset,
)
from multi_type_feedback.feedback_oracle import FeedbackOracle
from multi_type_feedback.multi_head_networks import (
    MultiHeadNetwork,
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


class EnsembleValidationDataset:
    """
    Wrapper dataset that expands validation examples for ensemble models.
    The entire dataset is repeated num_ensemble_models times to ensure proper batch structure
    for masksemble where batch is split into chunks: [A,B,C,A,B,C,A,B,C,A,B,C]
    """
    
    def __init__(self, base_dataset, num_ensemble_models):
        self.base_dataset = base_dataset
        self.num_ensemble_models = num_ensemble_models
        
    def __len__(self):
        # The entire dataset is repeated num_ensemble_models times
        return len(self.base_dataset) * self.num_ensemble_models
        
    def __getitem__(self, idx):
        # Map back to base dataset: examples cycle through the base dataset
        base_idx = idx % len(self.base_dataset)
        return self.base_dataset[base_idx]


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
        nr_of_iterations: int = 20,
        feedback_budget: int = 1500,
        feedback_buffer_size: int = 750,
        n_feedback_per_iteration: Optional[int] = None,   # now optional, computed if None
        rl_steps_per_iteration: Optional[int] = None,     # now optional, computed after init_rl
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
        self.nr_of_iterations = nr_of_iterations
        self.feedback_budget = feedback_budget
        self.initial_feedback_count = initial_feedback_count
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
        self.segment_len = 50  # Length of each trajectory segment

        # Create a temporary environment to get action space info using proper setup
        temp_env = TrainingUtils.setup_environment(env_name, seed, env_kwargs=env_kwargs)
        self.action_space = temp_env.action_space  # Store action space
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

        # (1) Compute n_feedback_per_iteration immediately (based on budget only)
        if n_feedback_per_iteration is None:
            remaining_budget = self.feedback_budget - self.initial_feedback_count
            if remaining_budget <= 0:
                raise ValueError(
                    f"Initial feedback count ({self.initial_feedback_count}) "
                    f"exceeds or equals total budget ({self.feedback_budget})"
                )
            if remaining_budget % self.nr_of_iterations != 0:
                self.n_feedback_per_iteration = remaining_budget // self.nr_of_iterations
                actual_budget = (
                    self.initial_feedback_count
                    + self.n_feedback_per_iteration * self.nr_of_iterations
                )
                print(
                    f"Warning: Budget {self.feedback_budget} cannot be evenly "
                    f"distributed over {self.nr_of_iterations} iterations."
                )
                print(
                    f"Using {self.n_feedback_per_iteration} feedback per iteration, "
                    f"actual total budget: {actual_budget}"
                )
            else:
                self.n_feedback_per_iteration = remaining_budget // self.nr_of_iterations
                print(f"Computed n_feedback_per_iteration: {self.n_feedback_per_iteration}")
        else:
            self.n_feedback_per_iteration = n_feedback_per_iteration

        # (2) Defer RL-steps-per-iteration computation; set placeholders
        self.total_timesteps: Optional[int] = None
        self.rl_steps_per_iteration = rl_steps_per_iteration  # may be None until init
        

        # Perform initial reward model training before initializing RL agent
        # Only if oracle is available (for automatic feedback generation)
        if self.initial_feedback_count > 0 and self.oracle is not None:
            self.rl_agent = None  # need for collect_trajectories
            self._initialize_reward_models_with_random_feedback()

        # Initialize RL agent using ExperimentManager
        self.rl_agent = self._init_rl_agent()

        self._compute_rl_steps_after_init()

        # set custom logger
        if custom_sb3_logger:
            self.rl_agent.set_logger(custom_sb3_logger)

    def _init_rl_agent(self) -> PPO | SAC:
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
            # Get hyperparameters with defaults
            hyperparams = dict(self._hyperparams) if self._hyperparams else {}

            # Set default policy if not specified
            if "policy" not in hyperparams:
                hyperparams["policy"] = "MlpPolicy"

            temp_env = TrainingUtils.setup_environment(self.env_name, self.seed, env_kwargs=self.env_kwargs)

            if self.algorithm == "ppo":
                return PPO(
                    env=temp_env,
                    verbose=1,
                    seed=self.seed,
                    device=self.device,
                    **hyperparams,
                )
            else:
                return SAC(
                    env=temp_env,
                    verbose=1,
                    seed=self.seed,
                    device=self.device,
                    **hyperparams,
                )
            
    def _compute_rl_steps_after_init(self):
        """
        Compute total_timesteps and rl_steps_per_iteration once exp_manager has been initialized.
        """
        if self.total_timesteps is None:
            # Prefer exp_manager.n_timesteps if available
            if self.exp_manager and hasattr(self.exp_manager, "n_timesteps"):
                self.total_timesteps = int(self.exp_manager.n_timesteps)
            else:
                raise ValueError(
                    "total_timesteps is not available. Ensure ExperimentManager sets n_timesteps."
                )

        if self.rl_steps_per_iteration is None:
            if self.total_timesteps % self.nr_of_iterations != 0:
                print(
                    f"Warning: Total timesteps ({self.total_timesteps}) not evenly "
                    f"divisible by nr_of_iterations ({self.nr_of_iterations})"
                )
                self.rl_steps_per_iteration = self.total_timesteps // self.nr_of_iterations
                actual_timesteps = self.rl_steps_per_iteration * self.nr_of_iterations
                print(
                    f"Using {self.rl_steps_per_iteration} RL steps per iteration, "
                    f"actual total timesteps: {actual_timesteps}"
                )
            else:
                self.rl_steps_per_iteration = self.total_timesteps // self.nr_of_iterations
                print(f"Computed rl_steps_per_iteration: {self.rl_steps_per_iteration}")

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
            # Use regular MultiHeadNetwork with CNN channels for CNN environments
            model = MultiHeadNetwork(
                input_spaces=(observation_space, action_space),
                shared_layer_num=self.shared_layer_num,
                head_layer_num=self.head_layer_num,
                hidden_dim=256,
                action_hidden_dim=16,
                output_dim=1,
                feedback_types=self.feedback_types,
                learning_rate=1e-5,
                cnn_channels=[16, 32, 32],
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
        """
        Initialize reward models. In the new asynchronous workflow, we skip initial feedback collection
        and start with untrained reward models that will be trained once human feedback is collected.
        """
        if self.oracle is None:
            print("\nSkipping initial feedback collection - using asynchronous workflow.")
            print("Reward models will be trained once human feedback is loaded.")
            return

        # Legacy oracle-based initialization (for backward compatibility)
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

            # Use oracle for initial feedback (legacy mode)
            feedback, batch_counts = self._sample_feedback_with_oracle(trajectories, initial_states)

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

    def _sample_feedback_with_oracle(
        self, trajectories: list[list], initial_states: list[np.ndarray]
    ) -> tuple[list[dict], dict[str, int]]:
        """
        Legacy method to sample feedback using oracle (for backward compatibility).
        Only used when oracle is provided.
        """
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

    def collect_trajectories(
        self, n_trajectories: int, env: gym.Env | None = None, render: bool = False
    ) -> Tuple[List[List[Tuple[np.ndarray, np.ndarray, float, bool, float]]], List[Any]]:
        """Collect trajectories using current policy, including reward model uncertainties."""
        if env is None:
            # add render_mode="rgb_array" if rendering is enabled
            env_kwargs = self.env_kwargs.copy()
            if render:
                env_kwargs["render_mode"] = "rgb_array"  #
            env = TrainingUtils.setup_environment(self.env_name, self.seed, env_kwargs=env_kwargs)
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

            if render:
                render_img = env.render()

            for _ in range(self.segment_len):
                if self.rl_agent is None:
                    # this is the case for initial generation, use random agent here
                    action = env.action_space.sample()
                else:
                    action, _ = self.rl_agent.predict(obs, deterministic=False)

                # Compute uncertainty for current state-action pair
                uncertainty = self._compute_step_uncertainty(obs, action)

                next_obs, reward, terminated, truncated, _ = env.step(action)
                if self.action_one_hot:
                    action = one_hot_vector(action, self.one_hot_dim)
                if render:
                    next_render = env.render()
                done = terminated or truncated

                # Very special case for bug in metaworld...comment out for other envs
                render_img = np.rot90(render_img, k=2)
                trajectory.append((np.expand_dims(obs, axis=0), action, reward, done, uncertainty, render_img))
                obs = next_obs
                render_img = next_render if render else None

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
                val_size = int(len(full_dataset) * 0.3)
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

                # Wrap validation dataset to ensure proper ensemble batching
                ensemble_val_dataset = EnsembleValidationDataset(val_dataset, self.num_ensemble_models)
                # Batch size must be a multiple of num_ensemble_models for masksemble
                # Use all validation data in one batch for small datasets, but cap it for memory
                max_val_batch_size = min(64, len(ensemble_val_dataset))  # Cap at 64 for memory
                val_batch_size = (max_val_batch_size // self.num_ensemble_models) * self.num_ensemble_models
                val_batch_size = max(val_batch_size, self.num_ensemble_models)  # Ensure at least one set
                
                val_loader = DataLoader(
                    ensemble_val_dataset,
                    batch_size=val_batch_size,
                    shuffle=False,
                    pin_memory=True,
                    drop_last=False,  # Don't drop last for validation
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
                    logger=self.wandb_logger or None,
                    check_val_every_n_epoch=1,
                    enable_model_summary=False,
                    enable_checkpointing=False,
                    log_every_n_steps=1,  # Log every step for small datasets
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

            # Wrap validation DataLoaders with ensemble dataset for proper masksemble batching
            for feedback_type, (train_loader, val_loader) in dataloaders.items():
                val_dataset = val_loader.dataset
                ensemble_val_dataset = EnsembleValidationDataset(val_dataset, self.num_ensemble_models)
                
                # Calculate appropriate batch size for validation
                max_val_batch_size = min(64, len(ensemble_val_dataset))
                val_batch_size = (max_val_batch_size // self.num_ensemble_models) * self.num_ensemble_models
                val_batch_size = max(val_batch_size, self.num_ensemble_models)
                
                wrapped_val_loader = DataLoader(
                    ensemble_val_dataset,
                    batch_size=val_batch_size,
                    shuffle=False,
                    pin_memory=True,
                    drop_last=False,
                )
                # Replace the validation loader
                dataloaders[feedback_type] = (train_loader, wrapped_val_loader)

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
                    log_every_n_steps=1,  # Log every step for small datasets
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

            # Wrap validation DataLoader with ensemble dataset for proper masksemble batching
            val_dataset = val_dataloader.dataset
            ensemble_val_dataset = EnsembleValidationDataset(val_dataset, self.num_ensemble_models)
            
            # Calculate appropriate batch size for validation
            max_val_batch_size = min(64, len(ensemble_val_dataset))
            val_batch_size = (max_val_batch_size // self.num_ensemble_models) * self.num_ensemble_models
            val_batch_size = max(val_batch_size, self.num_ensemble_models)
            
            val_dataloader = DataLoader(
                ensemble_val_dataset,
                batch_size=val_batch_size,
                shuffle=False,
                pin_memory=True,
                drop_last=False,
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
                log_every_n_steps=1,  # Log every step for small datasets
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
        """Compute uncertainty for a trajectory using the ensemble variance of the reward model."""
        device = self.device
    
        # Stack observations and actions from trajectory
        states = torch.vstack([torch.as_tensor(step[0]).float() for step in trajectory]).to(device)
        actions = torch.vstack([torch.as_tensor(step[1]).float() for step in trajectory]).to(device)
    
        with torch.no_grad():
            if self.reward_model_type == "separate":
                reward_model = self.reward_models[feedback_type]
                if reward_model.ensemble_count > 1:
                    states_expanded = states.unsqueeze(0).expand(reward_model.ensemble_count, *states.shape)
                    actions_expanded = actions.unsqueeze(0).expand(reward_model.ensemble_count, *actions.shape)
                    preds = reward_model(states_expanded, actions_expanded)  # [E, T, 1] or [E, T]
                    if preds.dim() == 3 and preds.shape[-1] == 1:
                        preds = preds.squeeze(-1)
                    step_unc = preds.std(dim=0)                # [T]
                    traj_unc = step_unc.mean().item()
                else:
                    traj_unc = 0.0
    
            elif self.reward_model_type == "multi-head":
                model = list(self.reward_models.values())[0]   # {"multi_head": model}
                if model.ensemble_count > 1:
                    states_expanded = states.unsqueeze(0).expand(model.ensemble_count, *states.shape)
                    actions_expanded = actions.unsqueeze(0).expand(model.ensemble_count, *actions.shape)
                    preds = model(states_expanded, actions_expanded, feedback_type)  # [E, T, 1] or [E, T]
                    if preds.dim() == 3 and preds.shape[-1] == 1:
                        preds = preds.squeeze(-1)
                    step_unc = preds.std(dim=0)                # [T]
                    traj_unc = step_unc.mean().item()
                else:
                    traj_unc = 0.0
    
            elif self.reward_model_type == "unified":
                model = list(self.reward_models.values())[0]   # {"unified": model}
                if model.ensemble_count > 1:
                    states_expanded = states.unsqueeze(0).expand(model.ensemble_count, *states.shape)
                    actions_expanded = actions.unsqueeze(0).expand(model.ensemble_count, *actions.shape)
                    preds = model(states_expanded, actions_expanded, feedback_type)  # [E, T, 1] or [E, T]
                    if preds.dim() == 3 and preds.shape[-1] == 1:
                        preds = preds.squeeze(-1)
                    step_unc = preds.std(dim=0)                # [T]
                    traj_unc = step_unc.mean().item()
                else:
                    traj_unc = 0.0
            else:
                raise ValueError(f"Unknown reward_model_type: {self.reward_model_type}")
    
        return traj_unc

    def _compute_step_uncertainty(self, state: np.ndarray, action: np.ndarray) -> float:
        """Compute uncertainty for a single state-action pair across all trained reward models."""
        # If no reward models have been trained yet, return high uncertainty
        trained_models = [fb_type for fb_type in self.feedback_types if len(self.feedback_buffers[fb_type]) > 0]

        if not trained_models:
            return 1.0  # High uncertainty when no models trained

        # Create a single-step trajectory for uncertainty computation
        single_step_trajectory = [(np.expand_dims(state, axis=0), action, 0.0, False)]

        # Compute uncertainty across all trained feedback types
        uncertainties = []
        for feedback_type in trained_models:
            uncertainty = self.compute_model_uncertainty(single_step_trajectory, feedback_type)
            uncertainties.append(uncertainty)

        # Return average uncertainty across all trained models
        return np.mean(uncertainties) if uncertainties else 1.0

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
        """
        DEPRECATED: This method relied on oracle feedback and is not used in the asynchronous workflow.
        In the new workflow, human feedback is collected via the frontend.
        """
        raise NotImplementedError(
            "sample_feedback_uncertainty is deprecated. "
            "Use the asynchronous workflow with frontend human feedback collection instead."
        )

    def sample_feedback_random(
        self, trajectories: List[List], initial_states: List[np.ndarray]
    ) -> tuple[List[Dict], Dict[str, int]]:
        """
        DEPRECATED: This method relied on oracle feedback and is not used in the asynchronous workflow.
        In the new workflow, human feedback is collected via the frontend.
        """
        raise NotImplementedError(
            "sample_feedback_random is deprecated. "
            "Use the asynchronous workflow with frontend human feedback collection instead."
        )

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

    def save_reward_models_checkpoint(self, checkpoint_step: int, exp_id: str) -> dict[str, str]:
        """
        Save reward models for a specific checkpoint for use with projection system.

        Args:
            checkpoint_step: The checkpoint step number
            exp_id: Experiment ID

        Returns:
            Dictionary mapping feedback types to saved model paths
        """
        from pathlib import Path
        import pytorch_lightning as pl

        # Create checkpoint directory
        checkpoint_dir = Path(f"multi-type-feedback/reward_models/checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        saved_models = {}

        if self.reward_model_type == "separate":
            # Save each feedback type's model separately
            for feedback_type, model in self.reward_models.items():
                # Create a descriptive filename
                model_filename = f"{self.algorithm}_{self.env_name.lower().replace('-', '_')}_{exp_id}_{feedback_type}_{checkpoint_step}.ckpt"
                model_path = checkpoint_dir / model_filename

                # Create proper PyTorch Lightning checkpoint
                checkpoint_dict = {
                    "state_dict": model.state_dict(),
                    "lr_schedulers": [],
                    "epoch": checkpoint_step,
                    "global_step": checkpoint_step,
                    "pytorch-lightning_version": pl.__version__,
                    "hyper_parameters": model.hparams,
                    "optimizer_states": [],
                    "callbacks": {},
                }

                torch.save(checkpoint_dict, model_path)
                saved_models[feedback_type] = str(model_path)
                print(f"Saved {feedback_type} reward model to: {model_path}")

        elif self.reward_model_type in ["multi-head", "unified"]:
            # Save the single model that handles all feedback types
            model = list(self.reward_models.values())[0]
            model_filename = f"{self.algorithm}_{self.env_name.lower().replace('-', '_')}_{exp_id}_{self.reward_model_type}_{checkpoint_step}.ckpt"
            model_path = checkpoint_dir / model_filename

            # Create proper PyTorch Lightning checkpoint
            checkpoint_dict = {
                "state_dict": model.state_dict(),
                "lr_schedulers": [],
                "epoch": checkpoint_step,
                "global_step": checkpoint_step,
                "pytorch-lightning_version": pl.__version__,
                "hyper_parameters": model.hparams,
                "optimizer_states": [],
                "callbacks": {},
            }

            torch.save(checkpoint_dict, model_path)
            saved_models["unified"] = str(model_path)
            print(f"Saved {self.reward_model_type} reward model to: {model_path}")

        return saved_models

    def save(self, save_path: str, checkpoint_step: int = None, exp_id: str = None) -> None:
        """
        Save the DynamicRLHF model including RL agent, reward models, and training state.
        Uses the same paths as save_reward_models_checkpoint for compatibility with projections.

        Args:
            save_path: Base path to save the model (without extension)
            checkpoint_step: Optional checkpoint step for projection compatibility
            exp_id: Optional experiment ID for projection compatibility
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # If checkpoint_step and exp_id are provided, use projection-compatible paths
        if checkpoint_step is not None and exp_id is not None:
            # Use save_reward_models_checkpoint for reward models (projection compatibility)
            saved_model_paths = self.save_reward_models_checkpoint(checkpoint_step, exp_id)

            # Save RL agent to projection-compatible path
            agent_dir = Path("multi-type-feedback/train_baselines/dynamic_rlhf_agents")
            agent_dir.mkdir(parents=True, exist_ok=True)
            agent_filename = f"{self.algorithm}_{self.env_name.lower().replace('-', '_')}_{exp_id}_{checkpoint_step}.zip"
            agent_path = agent_dir / agent_filename
            self.rl_agent.save(agent_path)  # Remove .zip as save() adds it

            # Store paths in state data for loading reference
            state_data_model_paths = saved_model_paths
            state_data_agent_path = str(agent_path)
        else:
            # Use local paths (original behavior for backward compatibility)
            # Save the RL agent (StableBaselines3)
            rl_agent_path = save_path / "rl_agent"
            self.rl_agent.save(rl_agent_path)

            # Save reward models locally
            reward_models_path = save_path / "reward_models"
            reward_models_path.mkdir(exist_ok=True)

            state_data_model_paths = {}
            for feedback_type, model in self.reward_models.items():
                model_path = reward_models_path / f"{feedback_type}.ckpt"
                torch.save(model.state_dict(), model_path)
                state_data_model_paths[feedback_type] = str(model_path)

            state_data_agent_path = str(rl_agent_path) + ".zip"

        # Save training state and configuration
        state_data = {
            "env_name": self.env_name,
            "env_kwargs": self.env_kwargs,
            "algorithm": self.algorithm,
            "feedback_types": self.feedback_types,
            "n_feedback_per_iteration": self.n_feedback_per_iteration,
            "feedback_buffer_size": self.feedback_buffer_size,
            "rl_steps_per_iteration": self.rl_steps_per_iteration,
            "reward_training_epochs": self.reward_training_epochs,
            "device": self.device,
            "num_ensemble_models": self.num_ensemble_models,
            "initial_feedback_count": self.initial_feedback_count,
            "reward_model_type": self.reward_model_type,
            "shared_layer_num": self.shared_layer_num,
            "head_layer_num": self.head_layer_num,
            "feedback_embedding_dim": self.feedback_embedding_dim,
            "action_one_hot": self.action_one_hot,
            "one_hot_dim": getattr(self, "one_hot_dim", None),
            "feedback_buffers": self.feedback_buffers,
            # Save Welford's algorithm state for reward standardization
            "reward_mean": self.reward_mean.cpu().numpy() if self.reward_mean is not None else None,
            "squared_distance_from_mean": (
                self.squared_distance_from_mean.cpu().numpy() if self.squared_distance_from_mean is not None else None
            ),
            "reward_counters": self.reward_counters.cpu().numpy() if self.reward_counters is not None else None,
            # Store model and agent paths for loading
            "saved_model_paths": state_data_model_paths,
            "saved_agent_path": state_data_agent_path,
            "checkpoint_step": checkpoint_step,
            "exp_id": exp_id,
        }

        state_path = str(save_path) + "state.pkl"
        with open(state_path, "wb") as f:
            pickle.dump(state_data, f)

        print(f"DynamicRLHF model saved to {save_path}")
        if checkpoint_step is not None and exp_id is not None:
            print(f"Models saved to projection-compatible paths for checkpoint {checkpoint_step}")

    @classmethod
    def load(cls, load_path: str, oracle: FeedbackOracle = None, exp_manager: ExperimentManager = None) -> "DynamicRLHF":
        """
        Load a DynamicRLHF model from disk.

        Args:
            load_path: Base path to load the model from
            oracle: FeedbackOracle instance (required for full functionality)
            exp_manager: ExperimentManager instance (optional)

        Returns:
            Loaded DynamicRLHF instance
        """
        load_path = Path(load_path)

        # Load training state and configuration
        state_path = load_path / "state.pkl"
        with open(state_path, "rb") as f:
            state_data = pickle.load(f)

        # Create DynamicRLHF instance with saved configuration
        drlhf = cls(
            oracle=oracle,
            env_name=state_data["env_name"],
            env_kwargs=state_data["env_kwargs"],
            algorithm=state_data["algorithm"],
            feedback_types=state_data["feedback_types"],
            n_feedback_per_iteration=state_data["n_feedback_per_iteration"],
            feedback_buffer_size=state_data["feedback_buffer_size"],
            rl_steps_per_iteration=state_data["rl_steps_per_iteration"],
            reward_training_epochs=state_data["reward_training_epochs"],
            device=state_data["device"],
            num_ensemble_models=state_data["num_ensemble_models"],
            initial_feedback_count=state_data["initial_feedback_count"],
            reward_model_type=state_data["reward_model_type"],
            shared_layer_num=state_data["shared_layer_num"],
            head_layer_num=state_data["head_layer_num"],
            feedback_embedding_dim=state_data["feedback_embedding_dim"],
            exp_manager=exp_manager,
        )

        # Load the RL agent - check if using projection-compatible paths
        if "saved_agent_path" in state_data:
            # Load from projection-compatible path
            agent_path = state_data["saved_agent_path"]
            print(f"Loading RL agent from projection path: {agent_path}")
        else:
            # Load from local path (backward compatibility)
            agent_path = str(Path(load_path) / "rl_agent.zip")
            print(f"Loading RL agent from local path: {agent_path}")

        if drlhf.algorithm == "ppo":
            drlhf.rl_agent = PPO.load(agent_path)
        else:
            drlhf.rl_agent = SAC.load(agent_path)

        # Load reward models - check if using projection-compatible paths
        if "saved_model_paths" in state_data:
            # Load from projection-compatible paths
            saved_model_paths = state_data["saved_model_paths"]
            print(f"Loading reward models from projection paths")

            for feedback_type, model in drlhf.reward_models.items():
                if feedback_type in saved_model_paths:
                    model_path = saved_model_paths[feedback_type]
                    print(f"Loading {feedback_type} model from: {model_path}")
                    if Path(model_path).exists():
                        # Use PyTorch Lightning's load_from_checkpoint for proper loading
                        model_class = type(model)
                        loaded_model = model_class.load_from_checkpoint(model_path)
                        drlhf.reward_models[feedback_type] = loaded_model
                        loaded_model.to(drlhf.device)
                        loaded_model.eval()
                    else:
                        print(f"Warning: Model file not found: {model_path}")
        else:
            # Load from local paths (backward compatibility)
            reward_models_path = Path(load_path) / "reward_models"
            for feedback_type, model in drlhf.reward_models.items():
                model_path = reward_models_path / f"{feedback_type}.ckpt"
                if model_path.exists():
                    model.load_state_dict(torch.load(model_path, map_location=drlhf.device))
                    model.to(drlhf.device)

        # Restore training state
        drlhf.action_one_hot = state_data["action_one_hot"]
        if state_data.get("one_hot_dim") is not None:
            drlhf.one_hot_dim = state_data["one_hot_dim"]
        drlhf.feedback_buffers = state_data["feedback_buffers"]

        # Restore Welford's algorithm state
        if state_data["reward_mean"] is not None:
            drlhf.reward_mean = torch.tensor(state_data["reward_mean"]).to(drlhf.device)
        if state_data["squared_distance_from_mean"] is not None:
            drlhf.squared_distance_from_mean = torch.tensor(state_data["squared_distance_from_mean"]).to(drlhf.device)
        if state_data["reward_counters"] is not None:
            drlhf.reward_counters = torch.tensor(state_data["reward_counters"]).to(drlhf.device)

        print(f"DynamicRLHF model loaded from {load_path}")
        return drlhf

    def load_feedback_dataset(self, feedback_dataset_path_prefix: str) -> dict:
        """
        Load processed feedback dataset and integrate it into DynamicRLHF feedback buffers.
        Uses LoadFeedbackDataset for each feedback type individually.

        Args:
            feedback_dataset_path_prefix: Base path to the processed feedback dataset files (file prefix),
            e.g., "path/to/feedback_dataset/dynamic_rlhf_feedback" + "_{feedback_type}.pkl"

        Returns:
            Dictionary with feedback loading statistics
        """

        try:
            # Statistics tracking
            stats = {"loaded_counts": defaultdict(int), "total_loaded": 0}

            # Mapping from internal feedback type names to LoadFeedbackDataset expected names
            feedback_type_mapping = {
                "evaluative": "evaluative",
                "comparative": "comparative",
                "demonstrative": "demonstrative",
                "corrective": "corrective",
                "descriptive": "descriptive",
            }

            # Process each feedback type that we support
            for internal_type in self.feedback_types:
                if internal_type in feedback_type_mapping:
                    feedback_type = feedback_type_mapping[internal_type]

                    # Construct the path for this feedback type
                    type_file_path = f"{feedback_dataset_path_prefix}{feedback_type}.pkl"

                    if os.path.exists(type_file_path):
                        print(f"Loading {feedback_type} feedback from {type_file_path}")

                        try:
                            # Create environment for demonstrative feedback if needed
                            env = None
                            if feedback_type == "demonstrative":
                                from multi_type_feedback.utils import TrainingUtils

                                env = TrainingUtils.setup_environment(self.env_name, self.seed, env_kwargs=self.env_kwargs)

                            # Load dataset using LoadFeedbackDataset
                            dataset = LoadFeedbackDataset(
                                dataset_path=type_file_path,
                                feedback_type=feedback_type,
                                n_feedback=-1,  # Load all available feedback
                                noise_level=0.0,  # No noise for real human feedback
                                env=env,
                                env_name=self.env_name,
                                seed=self.seed,
                            )

                            # Extract data from the dataset and populate buffers
                            for i in range(len(dataset)):
                                feedback_item = dataset[i]

                                # Add to feedback buffers with size management
                                if len(self.feedback_buffers[internal_type]) >= self.feedback_buffer_size:
                                    # Remove oldest feedback
                                    self.feedback_buffers[internal_type].pop(0)

                                self.feedback_buffers[internal_type].append(feedback_item)
                                stats["loaded_counts"][internal_type] += 1
                                stats["total_loaded"] += 1

                            print(f"Loaded {len(dataset)} {feedback_type} feedback items")

                            # Clean up environment if created
                            if env is not None:
                                env.close()

                        except Exception as e:
                            print(f"Error loading {feedback_type} feedback: {e}")
                            continue
                    else:
                        print(f"Feedback file not found: {type_file_path}")

            print(f"Total feedback loaded: {stats['total_loaded']}")
            print(f"Feedback counts by type: {dict(stats['loaded_counts'])}")

            return stats

        except Exception as e:
            print(f"Error loading feedback dataset: {e}")
            return {"error": str(e)}

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
            existing = list(getattr(self.exp_manager, "callbacks", []) or [])
            self.exp_manager.callbacks = [reward_model_callback] + existing
            self.exp_manager.learn(self.rl_agent)
        else:
            self.rl_agent.learn(total_timesteps=total_timesteps,
                                callback=reward_model_callback,
                                reset_num_timesteps=True)

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
        "--feedback-budget",
        type=int,
        default=1500,
        help="Total feedback budget for the entire training run",
    )
    parser.add_argument(
        "--nr-of-iterations",
        type=int,
        default=20,
        help="Number of reward model update iterations (computes rl-steps-per-iteration from total timesteps)",
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
        default=100,
        help="Number of feedback samples to collect before starting RL training",
    )
    parser.add_argument(
        "--feedback-buffer-size",
        type=int,
        default=500,
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

    # Calculate remaining budget after initial feedback
    remaining_budget = args.feedback_budget - args.initial_feedback_count
    if remaining_budget <= 0:
        raise ValueError(f"Initial feedback count ({args.initial_feedback_count}) exceeds or equals total budget ({args.feedback_budget})")

    if remaining_budget % args.nr_of_iterations != 0:
        # Round down to ensure we don't exceed budget
        n_feedback_per_iteration = remaining_budget // args.nr_of_iterations
        actual_budget = args.initial_feedback_count + (n_feedback_per_iteration * args.nr_of_iterations)
        print(f"Warning: Budget {args.feedback_budget} cannot be evenly distributed over {args.nr_of_iterations} iterations.")
        print(f"Using {n_feedback_per_iteration} feedback per iteration, actual total budget: {actual_budget}")
    else:
        n_feedback_per_iteration = remaining_budget // args.nr_of_iterations
        print(f"Computed n_feedback_per_iteration: {n_feedback_per_iteration}")

    uuid_str = f"_{uuid.uuid4()}"
    exp_manager = ExperimentManager(
        args=args,
        algo=args.algorithm,
        env_id=args.environment,
        log_folder=args.save_folder,
        eval_freq=5000,
        n_eval_episodes=5,
        use_wandb_callback=True,
        wandb_callback_continuous=True,
        reward_function=None,
        uuid_str=uuid_str,
    )

    # Setup oracle
    feedback_id, _ = TrainingUtils.get_model_ids(args)
    device = TrainingUtils.get_device()
    feedback_path = Path(args.reference_data_folder) / f"{feedback_id}.pkl"
    gen_environment = TrainingUtils.setup_environment(args.environment, args.seed)
    expert_models = TrainingUtils.load_expert_models(
        env_name=args.environment,
        algorithm=args.expert_algorithm if args.expert_algorithm else args.algorithm,
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

    reward_model_type = args.reward_model_type if len(args.feedback_types) > 1 else f"single_{''.join(args.feedback_types)}"

    # Initialize wandb
    wandb.init(
        name=f"DYNAMIC_RL_{args.algorithm}_{args.environment}_{reward_model_type}_{args.seed}",
        project=args.wandb_project_name,
        config={
            "algorithm": args.algorithm,
            "feedback_types": args.feedback_types,
            "nr_of_iterations": args.nr_of_iterations,
            "feedback_budget": args.feedback_budget,
            "reward_training_epochs": args.reward_training_epochs,
            "feedback_buffer_size": args.feedback_buffer_size,
            "reward_model_type": args.reward_model_type,
            "sampling_strategy": args.sampling_strategy,
            "query_sampling_strategy": args.query_sampling_strategy,
            "initial_feedback_count": args.initial_feedback_count,
        },
    )

    continuous_lightning_logger = ContinuousWandbLogger()
    custom_sb3_logger = create_continuous_wandb_logger(
        global_step_offset=0,
        run_id=wandb.run.id,
        additional_formats=["stdout"],
        folder="logs",
    )

    print("DYNAMIC RLHF FEEDBACK TYPES:", args.feedback_types)

    drlhf = DynamicRLHF(
        oracle=oracle,
        env_name=args.environment,
        algorithm=args.algorithm,
        feedback_types=args.feedback_types,
        nr_of_iterations=args.nr_of_iterations,
        feedback_budget=args.feedback_budget,
        feedback_buffer_size=args.feedback_buffer_size,
        reward_training_epochs=args.reward_training_epochs,
        num_ensemble_models=args.num_ensemble_models,
        apply_random_response_handling=args.random_response_handling,
        initial_feedback_count=args.initial_feedback_count,
        hyperparams=exp_manager.get_hyperparam_config_for_algo(),
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

    wandb.config.update({
        "n_feedback_per_iteration": drlhf.n_feedback_per_iteration,
        "rl_steps_per_iteration": drlhf.rl_steps_per_iteration,
        "total_timesteps": drlhf.total_timesteps,
    }, allow_val_change=True)

    drlhf.train(
        sampling_strategy=args.sampling_strategy,
        query_sampling_strategy=args.query_sampling_strategy,
        query_sampling_multiplier=args.query_sampling_multiplier
    )

if __name__ == "__main__":
    main()