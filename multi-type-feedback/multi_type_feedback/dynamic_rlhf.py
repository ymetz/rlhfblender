import pickle
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
import uuid

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
    FiLMUnifiedNetwork,
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


class _EpochLossLogger(pytorch_lightning.Callback):
    """Records train_loss and val_loss at every epoch for post-training inspection."""

    def __init__(self):
        self.history: list[tuple[int, float, float]] = []  # (epoch, train, val)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        m = trainer.callback_metrics
        train = float(m.get("train_loss", float("nan")))
        val = float(m.get("val_loss", float("nan")))
        self.history.append((trainer.current_epoch, train, val))

    def print_history(self, label: str = "") -> None:
        if not self.history:
            return
        prefix = f"  [{label}] " if label else "  "
        print(f"{prefix}epoch  train_loss   val_loss")
        for epoch, tl, vl in self.history:
            gap = "" if (vl != vl or tl != tl) else f"  gap={vl - tl:+.4f}"
            print(f"{prefix}{epoch:>5}  {tl:>10.4f}  {vl:>10.4f}{gap}")


class QuantileNormalizer:
    """
    Running quantile normalizer that maps each feedback type's reward to [0, 1]
    using a sorted buffer of recent predictions.

    Unlike mean/std standardization, this is invariant to scale and shift —
    it only preserves ordinal information.  This makes it safe to aggregate
    rewards from feedback types that live on incompatible scales (e.g.
    comparative NLL utility vs. evaluative MSE predictions).
    """

    def __init__(self, buffer_size: int = 5000):
        self.buffer_size = buffer_size
        self.buffers: Dict[str, np.ndarray] = {}     # fb_type → sorted recent rewards
        self._buf_counts: Dict[str, int] = {}

    def update_and_normalize(
        self, rewards: torch.Tensor, feedback_type: str
    ) -> torch.Tensor:
        """
        Update the running buffer for *feedback_type* and return quantile-
        normalized rewards in [0, 1].

        Args:
            rewards: 1-D tensor of raw reward predictions (batch_size,)
            feedback_type: which type produced these rewards

        Returns:
            Tensor of same shape, values in [0, 1]
        """
        r_np = rewards.detach().cpu().numpy().ravel()

        if feedback_type not in self.buffers:
            self.buffers[feedback_type] = r_np.copy()
            self._buf_counts[feedback_type] = len(r_np)
        else:
            buf = self.buffers[feedback_type]
            combined = np.concatenate([buf, r_np])
            if len(combined) > self.buffer_size:
                combined = combined[-self.buffer_size:]
            self.buffers[feedback_type] = combined
            self._buf_counts[feedback_type] = len(combined)

        # Compute quantile rank: fraction of buffer values <= each reward
        buf = self.buffers[feedback_type]
        sorted_buf = np.sort(buf)
        # searchsorted gives index where r_np would be inserted to keep sorted
        indices = np.searchsorted(sorted_buf, r_np, side="right")
        quantiles = indices.astype(np.float32) / max(len(sorted_buf), 1)

        return torch.as_tensor(quantiles, device=rewards.device, dtype=rewards.dtype)


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

    :param uncertainty_penalty: If > 0, subtract ``uncertainty_penalty * ensemble_std``
        from the mean reward.  This discourages the RL agent from exploiting regions
        of the state-action space where the reward model is uncertain (reward hacking).
    """

    def __init__(self, drlhf_agent, uncertainty_penalty: float = 0.0):
        super().__init__()
        self.drlhf_agent = drlhf_agent
        self.uncertainty_penalty = uncertainty_penalty

    def __call__(
        self,
        state: np.ndarray,
        actions: np.ndarray,
        next_state: np.ndarray,
        _done: np.ndarray,
    ) -> np.ndarray:
        """Return reward given the current state and action."""
        
        if self.uncertainty_penalty > 0.0:
            reward, uncertainty = self.drlhf_agent.compute_ensemble_reward_with_uncertainty(state, actions)
            return reward - self.uncertainty_penalty * uncertainty
        return self.drlhf_agent.compute_ensemble_reward(state, actions)


class DynamicRLHF:
    def __init__(
        self,
        oracle: FeedbackOracle,
        env_name: str = "Pendulum-v1",
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
        reward_model_type: str = "separate",  # Options: "separate", "multi-head", "unified", "film-unified"
        shared_layer_num: int = 5,
        head_layer_num: int = 1,
        feedback_embedding_dim: int = 32,
        exp_manager: ExperimentManager = None,  # Add ExperimentManager
        env_kwargs: Optional[Dict[str, Any]] = None,
        uncertainty_penalty: float = 0.0,
        reward_normalization: str = "welford",  # "welford" (mean/std) or "quantile"
        responserank_weight: float = 0.5,
        reward_batch_size: int = 0,  # 0 = auto (8 for film-unified, 1 otherwise)
        reward_model_hidden_dim: int = 256,
        reward_model_layer_num: int = 6,
        use_gt_reward: bool = False,
        oversample_rewards: bool = False,
    ):
        self.oracle = oracle
        self.env_name = env_name
        self.algorithm = algorithm
        self.feedback_types = feedback_types
        self.n_feedback_per_iteration = n_feedback_per_iteration
        self.feedback_buffer_size = feedback_buffer_size
        self.rl_steps_per_iteration = rl_steps_per_iteration
        self.nr_of_iterations = nr_of_iterations
        self.feedback_budget = feedback_budget
        self.initial_feedback_count = initial_feedback_count
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
        self.env_kwargs = env_kwargs or {}
        self.uncertainty_penalty = uncertainty_penalty
        self.reward_normalization = reward_normalization
        self.responserank_weight = responserank_weight
        self.reward_batch_size = reward_batch_size
        self.reward_model_hidden_dim = reward_model_hidden_dim
        self.reward_model_layer_num = reward_model_layer_num
        self.use_gt_reward = use_gt_reward
        self.oversample_rewards = oversample_rewards

        self.reward_model_type = reward_model_type
        self.shared_layer_num = shared_layer_num
        self.head_layer_num = head_layer_num
        self.feedback_embedding_dim = feedback_embedding_dim

        # Create a temporary environment to get action space info using proper setup
        temp_env = TrainingUtils.setup_environment(env_name, seed, env_kwargs=self.env_kwargs or None)
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

        # Quantile normalizer (alternative to Welford)
        self.quantile_normalizer = QuantileNormalizer(buffer_size=5000)

        if apply_random_response_handling:
            self._apply_random_response_handling()

        # Create reward function wrapper (skipped in gt-reward debug mode)
        if not self.use_gt_reward:
            self.reward_function = DynamicRLHFRewardFunction(self, uncertainty_penalty=self.uncertainty_penalty)
            if self.exp_manager:
                self.exp_manager.reward_function = self.reward_function
        else:
            self.reward_function = None
            print("  [GT-REWARD MODE] Reward model disabled — using environment ground-truth reward.")

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
        if self.initial_feedback_count > 0:
            self.rl_agent = None  # need for collect_trajectories
            self._initialize_reward_models_with_random_feedback()
        
        # Initialize RL agent using ExperimentManager
        self.rl_agent = self._init_rl_agent()

        self._compute_rl_steps_after_init()

        # set custom logger
        if custom_sb3_logger:
            self.rl_agent.set_logger(custom_sb3_logger)

        # Set up fixed evaluation/holdout sets for consistent tracking
        self._init_eval_sets()

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
            if hasattr(self.exp_manager, "n_timesteps"):
                self.total_timesteps = int(self.exp_manager.n_timesteps)
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
                    env=TrainingUtils.setup_environment(self.env_name, self.seed, env_kwargs=self.env_kwargs or None),
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
        temp_env = TrainingUtils.setup_environment(self.env_name, self.seed, env_kwargs=self.env_kwargs or None)
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
        elif self.reward_model_type == "film-unified":
            # FiLM-conditioned unified model
            return self._init_film_unified_reward_model(observation_space, action_space)
        else:
            raise ValueError(f"Unknown reward model type: {self.reward_model_type}")

    def _init_eval_sets(self, holdout_segments: int = 64, pairwise_pairs: int = 64, ood_segments: int = 64):
        """
        Prepare small, fixed evaluation sets used across iterations to compute metrics.
        - Supervised/evaluative/descriptive holdout segments with ground-truth totals
        - Pairwise unordered holdout pairs with labels
        - OOD holdout using random policy segments
        """
        try:
            env = TrainingUtils.setup_environment(self.env_name, (self.seed or 0) + 123, env_kwargs=self.env_kwargs or None)
            self.eval_holdout = []  # list of ((obs, act, mask), gt_total)
            self.eval_holdout_sa = []  # list of (state, action, gt_step_reward)

            # Collect segments using a random policy for a fixed distribution
            for _ in range(max(holdout_segments, 8)):
                trajectory = []
                obs, _ = env.reset()
                for _t in range(self.oracle.segment_len):
                    action = env.action_space.sample()
                    nobs, reward, terminated, truncated, _info = env.step(action)
                    done = terminated or truncated
                    a = one_hot_vector(action, env.action_space.n) if isinstance(env.action_space, gym.spaces.Discrete) else action
                    trajectory.append((np.expand_dims(obs, axis=0), a, reward, done))
                    # also stash per-step
                    self.eval_holdout_sa.append((np.expand_dims(obs, axis=0), a, reward))
                    obs = nobs
                    if done:
                        break
                # Convert trajectory to supervised format and store
                sup = self.oracle.get_supervised_feedback(trajectory)
                # sup is list over steps; package as one segment to match training interface
                obs_t = torch.vstack([s[0][0] for s in sup])
                act_t = torch.vstack([s[0][1] for s in sup])
                mask_t = torch.ones(obs_t.shape[0]).unsqueeze(-1)
                # pad to segment_len for consistency
                if obs_t.shape[0] < self.oracle.segment_len:
                    pad = self.oracle.segment_len - obs_t.shape[0]
                    obs_t = torch.cat([obs_t, torch.zeros(pad, *obs_t.shape[1:])], dim=0)
                    act_t = torch.cat([act_t, torch.zeros(pad, *act_t.shape[1:])], dim=0)
                    mask_t = torch.cat([mask_t, torch.zeros(pad, 1)], dim=0)
                gt_total = sum([s[1].item() for s in sup])
                self.eval_holdout.append(((obs_t, act_t, mask_t), float(gt_total)))

            # Create pairwise unordered holdout pairs
            self.eval_pairs = []  # list of (((o1,a1,m1),(o2,a2,m2)), label 0/1 where 1 means traj2 better)
            for _ in range(max(pairwise_pairs, 8)):
                traj1 = self.oracle.get_random_trajectory()
                traj2 = self.oracle.get_random_trajectory()
                # compute discounted returns for label
                r1 = self.oracle._compute_discounted_return(traj1)  # noqa: SLF001
                r2 = self.oracle._compute_discounted_return(traj2)
                # tensors
                obs1 = torch.vstack([torch.as_tensor(p[0]).float() for p in traj1])
                act1 = torch.vstack([torch.as_tensor(p[1]).float() for p in traj1])
                mask1 = torch.ones(len(traj1)).unsqueeze(-1)
                if len(traj1) < self.oracle.segment_len:
                    pad = self.oracle.segment_len - len(traj1)
                    obs1 = torch.cat([obs1, torch.zeros(pad, *obs1.shape[1:])], dim=0)
                    act1 = torch.cat([act1, torch.zeros(pad, *act1.shape[1:])], dim=0)
                    mask1 = torch.cat([mask1, torch.zeros(pad, 1)], dim=0)
                obs2 = torch.vstack([torch.as_tensor(p[0]).float() for p in traj2])
                act2 = torch.vstack([torch.as_tensor(p[1]).float() for p in traj2])
                mask2 = torch.ones(len(traj2)).unsqueeze(-1)
                if len(traj2) < self.oracle.segment_len:
                    pad = self.oracle.segment_len - len(traj2)
                    obs2 = torch.cat([obs2, torch.zeros(pad, *obs2.shape[1:])], dim=0)
                    act2 = torch.cat([act2, torch.zeros(pad, *act2.shape[1:])], dim=0)
                    mask2 = torch.cat([mask2, torch.zeros(pad, 1)], dim=0)
                label = 1 if r2 > r1 else 0
                self.eval_pairs.append(((obs1, act1, mask1), (obs2, act2, mask2), label))

            # OOD segments: collect expert-guided or random-with-noise
            self.ood_holdout = []
            for _ in range(max(ood_segments, 8)):
                # try a high-return demo if available, else random
                try:
                    init_obs, _ = env.reset()
                    init_state = env.save_state(observation=init_obs)
                    demo = self.oracle._get_best_demonstration(init_state)
                    src = demo if demo is not None else self.oracle.get_random_trajectory()
                except Exception:
                    src = self.oracle.get_random_trajectory()
                sup = self.oracle.get_supervised_feedback(src)
                obs_t = torch.vstack([s[0][0] for s in sup])
                act_t = torch.vstack([s[0][1] for s in sup])
                mask_t = torch.ones(obs_t.shape[0]).unsqueeze(-1)
                if obs_t.shape[0] < self.oracle.segment_len:
                    pad = self.oracle.segment_len - obs_t.shape[0]
                    obs_t = torch.cat([obs_t, torch.zeros(pad, *obs_t.shape[1:])], dim=0)
                    act_t = torch.cat([act_t, torch.zeros(pad, *act_t.shape[1:])], dim=0)
                    mask_t = torch.cat([mask_t, torch.zeros(pad, 1)], dim=0)
                gt_total = sum([s[1].item() for s in sup])
                self.ood_holdout.append(((obs_t, act_t, mask_t), float(gt_total)))
            env.close()
        except Exception:
            # Fallback: empty sets if env init fails
            self.eval_holdout, self.eval_pairs, self.ood_holdout, self.eval_holdout_sa = [], [], [], []

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
                    hidden_dim=self.reward_model_hidden_dim,
                    action_hidden_dim=32,
                    layer_num=self.reward_model_layer_num,
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
                hidden_dim=self.reward_model_hidden_dim,
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
                layer_num=self.reward_model_layer_num,
                hidden_dim=self.reward_model_hidden_dim,
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

    def _init_film_unified_reward_model(self, observation_space, action_space):
        """Initialize a FiLM-conditioned unified reward model."""
        model = FiLMUnifiedNetwork(
            input_spaces=(observation_space, action_space),
            layer_num=self.reward_model_layer_num,
            hidden_dim=self.reward_model_hidden_dim,
            action_hidden_dim=32,
            output_dim=1,
            feedback_types=self.feedback_types,
            learning_rate=1e-5,
            ensemble_count=self.num_ensemble_models,
            feedback_embedding_dim=self.feedback_embedding_dim,
            responserank_weight=self.responserank_weight,
        )
        return {"film_unified": model}

    def _initialize_reward_models_with_random_feedback(self):
        """Collect initial random feedback and train reward models before RL training begins."""
        print(
            f"\nInitializing reward models with {self.initial_feedback_count} random feedback samples..."
        )

        # Create a temporary environment for trajectory collection using proper setup
        temp_env = TrainingUtils.setup_environment(self.env_name, self.seed, env_kwargs=self.env_kwargs or None)

        # Calculate how many batches of trajectories to collect
        batches_needed = (
            self.initial_feedback_count + self.n_feedback_per_iteration - 1
        ) // self.n_feedback_per_iteration
        total_feedback_collected = 0
        feedback_counts = defaultdict(int)

        for batch in range(batches_needed):

            # Collect random trajectories
            trajectories, initial_states = self.collect_trajectories(
                self.n_feedback_per_iteration, temp_env
            )

            # Always use random sampling for initial feedback
            feedback, batch_counts = self.sample_feedback_random(
                trajectories, initial_states
            )

            # Update feedback counts
            for feedback_type, count in batch_counts.items():
                feedback_counts[feedback_type] += count
                total_feedback_collected += count

            # Update feedback buffers
            self.update_feedback_buffers(feedback)

            # Log progress if wandb is available
            if (
                self.wandb_logger is not None
                and hasattr(self.wandb, "run")
                and self.wandb.run is not None
            ):
                metrics_to_log = {}
                for feedback_type, count in feedback_counts.items():
                    metrics_to_log[f"initial_feedback/{feedback_type}_count"] = count
                metrics_to_log["initial_feedback/total_collected"] = (
                    total_feedback_collected
                )
                metrics_to_log["initial_feedback/percent_complete"] = (
                    total_feedback_collected / self.initial_feedback_count
                ) * 100
                self.wandb.log(metrics_to_log)

            if total_feedback_collected >= self.initial_feedback_count:
                break

        temp_env.close()

        # Train the reward models with more epochs for initial training
        initial_training_epochs = (
            self.reward_training_epochs * 2
        )  # Train longer initially
        reward_metrics = self._train_reward_models_with_epochs(initial_training_epochs)

        print("\nInitial feedback counts:")
        for feedback_type, count in feedback_counts.items():
            print(f"{feedback_type}: {count}")

        print("\nInitial reward model losses:")
        for feedback_type, loss in reward_metrics.items():
            print(f"{feedback_type}: {loss:.4f}")
        self.print_reward_model_diagnostics()

        # Log initial training metrics
        if (
            self.wandb_logger is not None
            and hasattr(self.wandb, "run")
            and self.wandb.run is not None
        ):
            metrics_to_log = {}
            for feedback_type, loss in reward_metrics.items():
                metrics_to_log[f"initial_reward_model/{feedback_type}_loss"] = loss
            self.wandb.log(metrics_to_log)

    def collect_trajectories(
        self, n_trajectories: int, env: gym.Env = None
    ) -> Tuple[List[List[Tuple[np.ndarray, np.ndarray, float, bool]]], List[Any]]:
        """Collect trajectories using current policy."""
        if env is None:
            env = TrainingUtils.setup_environment(self.env_name, self.seed, env_kwargs=self.env_kwargs or None)
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
                    # Normalize obs if the training env uses VecNormalize, so the policy
                    # sees the same input distribution it was trained on.
                    vec_normalize = self.rl_agent.get_vec_normalize_env()
                    obs_for_policy = vec_normalize.normalize_obs(obs[np.newaxis])[0] if vec_normalize is not None else obs
                    action, _ = self.rl_agent.predict(obs_for_policy, deterministic=False)
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

    def _pl_accelerator(self) -> tuple[str, int]:
        """Map self.device to a (accelerator, devices) pair for PyTorch Lightning.

        Prevents PL from auto-selecting MPS on Apple Silicon when self.device is
        'cpu', which would cause float64 → MPS conversion errors.
        """
        d = str(self.device).lower()
        if d == "mps":
            return "mps", 1
        if d.startswith("cuda"):
            return "gpu", 1
        return "cpu", 1

    def _train_reward_models_with_epochs(self, max_epochs=None):
        """
        Train reward models with specified number of epochs.
        Modified to handle different reward model architectures.
        """
        reward_metrics = {}

        # Use default epochs if not specified
        if max_epochs is None:
            max_epochs = self.reward_training_epochs

        # Compute accumulate_grad_batches dynamically so that gradient steps
        # are not eliminated when the buffer is small.  Target ~4 effective
        # gradient steps per epoch regardless of buffer size, capped at 32.
        total_buffer = sum(len(v) for v in self.feedback_buffers.values())
        train_size_est = max(1, int(total_buffer * 0.632))
        batches_per_epoch = max(1, train_size_est // max(1, self.num_ensemble_models))
        # We want at least 4 grad steps / epoch → accumulate at most batches/4
        accum_grad = max(1, min(32, batches_per_epoch // 4))

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
                    print(
                        f"Skipping {feedback_type} training: insufficient data ({len(full_dataset)} samples)"
                    )
                    continue

                train_dataset, val_dataset = torch.utils.data.random_split(
                    full_dataset, [train_size, val_size]
                )

                # Setup data loaders
                # pin_memory=False: MPS (Apple Silicon) routes pinned memory through
                # Metal APIs, which reject float64 tensors produced by default_collate.
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=self.num_ensemble_models,
                    shuffle=True,
                    pin_memory=False,
                    drop_last=True,
                )

                val_loader = DataLoader(
                    val_dataset,
                    batch_size=self.num_ensemble_models,
                    shuffle=False,
                    pin_memory=False,
                    drop_last=False,
                )

                # Configure callbacks and trainer
                loss_logger = _EpochLossLogger()
                callbacks = [
                    L2RegulationCallback(initial_l2=0.01),
                    pytorch_lightning.callbacks.EarlyStopping(
                        monitor="val_loss", patience=3, mode="min"
                    ),
                    loss_logger,
                ]

                _acc, _devs = self._pl_accelerator()
                trainer = Trainer(
                    max_epochs=max_epochs,
                    accelerator=_acc,
                    devices=_devs,
                    precision="32-true",
                    enable_progress_bar=False,
                    accumulate_grad_batches=accum_grad,
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
                reward_metrics[f"{feedback_type}_train"] = train_loss

                print(f"{feedback_type} training complete: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
                loss_logger.print_history(label=feedback_type)

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

            # Train for each feedback type separately (but using the shared model)
            for feedback_type, (train_loader, val_loader) in dataloaders.items():
                print(f"Training multi-head model for {feedback_type}")

                loss_logger = _EpochLossLogger()
                callbacks = [
                    L2RegulationCallback(initial_l2=0.01),
                    pytorch_lightning.callbacks.EarlyStopping(
                        monitor="val_loss", patience=3, mode="min"
                    ),
                    loss_logger,
                ]

                # Configure trainer
                _acc, _devs = self._pl_accelerator()
                trainer = Trainer(
                    max_epochs=max_epochs,
                    accelerator=_acc,
                    devices=_devs,
                    precision="32-true",
                    enable_progress_bar=False,
                    accumulate_grad_batches=accum_grad,
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
                train_loss = float(final_metrics.get("train_loss", -1.0))
                val_loss = float(final_metrics.get(f"val_loss_{feedback_type}", -1.0))
                reward_metrics[feedback_type] = val_loss
                reward_metrics[f"{feedback_type}_train"] = train_loss

                print(f"{feedback_type} training complete: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
                loss_logger.print_history(label=feedback_type)

        elif self.reward_model_type in ("unified", "film-unified"):
            # Unified / FiLM-unified implementation
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
            if self.reward_batch_size > 0:
                effective_batch_size = self.reward_batch_size
            else:
                # Auto: Masksemble requires batch_size > n_masks to avoid
                # mask collapse (near-zero train_loss, premature early stopping).
                # Use at least 4× the ensemble count; film-unified uses the same
                # heuristic (previously was 8, which was also too small for 4 masks).
                effective_batch_size = max(self.num_ensemble_models * 4, 16)
            # Recompute accum_grad using the actual unified batch size.
            # The value computed above used num_ensemble_models as the batch-size
            # proxy (correct for separate/multi-head), but unified now uses a
            # larger effective_batch_size, so there are far fewer batches per
            # epoch.  Without this correction accum_grad was overestimated by
            # ~4× resulting in only 1 real gradient step per epoch.
            _batches_unified = max(1, train_size_est // max(1, effective_batch_size))
            accum_grad = max(1, min(32, _batches_unified // 4))
            train_dataloader, val_dataloader = create_unified_dataloaders(
                self.feedback_buffers,
                batch_size=effective_batch_size,
                val_split=0.368,
                partition_size=4,
                oversample_rewards=self.oversample_rewards,
            )

            loss_logger = _EpochLossLogger()
            callbacks = [
                L2RegulationCallback(initial_l2=0.01),
                pytorch_lightning.callbacks.EarlyStopping(
                    monitor="val_loss", patience=3, mode="min"
                ),
                loss_logger,
            ]

            _acc, _devs = self._pl_accelerator()
            trainer = Trainer(
                max_epochs=max_epochs,
                accelerator=_acc,
                devices=_devs,
                precision="32-true",
                enable_progress_bar=False,
                accumulate_grad_batches=accum_grad,
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
            final_train_loss = float(final_metrics.get("train_loss", float("nan")))

            for feedback_type in self.feedback_types:
                val_loss_key = f"val_loss_{feedback_type}"
                if val_loss_key in final_metrics:
                    reward_metrics[feedback_type] = float(final_metrics[val_loss_key])

            # Overall val and train loss
            if "val_loss" in final_metrics:
                reward_metrics["overall"] = float(final_metrics["val_loss"])
            reward_metrics["overall_train"] = final_train_loss

            print(f"Unified model training complete (train_loss={final_train_loss:.4f})")
            for fb_type, loss in reward_metrics.items():
                tag = "train" if fb_type.endswith("_train") else "val"
                print(f"  {fb_type}: {tag}_loss={loss:.4f}")
            loss_logger.print_history(label="unified")

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
                    if (
                        hasattr(orig_loss, "__name__")
                        and orig_loss.__name__ == "calculate_pairwise_loss"
                    ):
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
                        chosen_probs = torch.where(
                            preferred_indices == 0, probs, 1 - probs
                        )

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
    
            elif self.reward_model_type in ("unified", "film-unified"):
                model = list(self.reward_models.values())[0]
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


    def compute_trajectory_overall_uncertainty(
        self, trajectory: List[Tuple[np.ndarray, np.ndarray, float, bool]], 
        strategy: str = "average"
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
        self, 
        trajectories: List[List], 
        initial_states: List[np.ndarray],
        n_queries: int,
        strategy: str = "average"
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
                finite_uncertainties_with_idx = [(finite_indices[i], trajectory_uncertainties[finite_indices[i]]) 
                                                for i in range(len(finite_indices))]
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
                if (
                    len(self.feedback_buffers[feedback_type]) > 0
                ):  # Only if model has been trained
                    uncertainty = self.compute_model_uncertainty(
                        trajectory, feedback_type
                    )
                else:
                    # If no feedback yet, set high uncertainty to encourage exploration
                    uncertainty = float("inf")
                uncertainties[feedback_type] = uncertainty
            trajectory_uncertainties.append(uncertainties)

        # Sample feedback types based on uncertainties
        feedback_counts = defaultdict(int)
        all_feedback = []

        # For each trajectory, sample feedback type with probability proportional to uncertainty
        for trajectory, initial_state, uncertainties in zip(
            trajectories, initial_states, trajectory_uncertainties
        ):
            # Normalize uncertainties to probabilities
            total_uncertainty = sum(uncertainties.values())
            if total_uncertainty == float("inf"):
                # If no feedback yet for some types, sample uniformly from those
                untrained_types = [
                    ft
                    for ft in self.feedback_types
                    if len(self.feedback_buffers[ft]) == 0
                ]
                feedback_type = np.random.choice(untrained_types)
            else:
                probs = [
                    uncertainties[ft] / total_uncertainty for ft in self.feedback_types
                ]
                feedback_type = np.random.choice(self.feedback_types, p=probs)

            # Handle different feedback types
            feedback_dict = {}
            if feedback_type in ["comparative", "descriptive_preference"]:
                # Need a second trajectory for comparison
                trajectory2, _ = self.collect_trajectories(1)
                feedback = self.oracle.get_feedback(
                    (trajectory, trajectory2[0]), initial_state, feedback_type
                )
            else:
                feedback = self.oracle.get_feedback(
                    trajectory, initial_state, feedback_type
                )

            feedback_dict[feedback_type] = feedback
            feedback_counts[feedback_type] += 1
            all_feedback.append(feedback_dict)

        return all_feedback, feedback_counts

    def sample_feedback_random(
        self, trajectories: List[List], initial_states: List[np.ndarray]
    ) -> tuple[List[Dict], Dict[str, int]]:
        """Randomly sample feedback types."""
        feedback_distribution = np.ones(len(self.feedback_types)) / len(
            self.feedback_types
        )
        selected_types = np.random.choice(
            self.feedback_types,
            size=len(trajectories),
            p=feedback_distribution,
        )

        feedback_counts = defaultdict(int)
        all_feedback = []

        for trajectory, initial_state, feedback_type in zip(
            trajectories, initial_states, selected_types
        ):
            feedback_dict = {}

            # Handle different feedback types
            if feedback_type in ["comparative", "descriptive_preference"]:
                # Need a second trajectory for comparison
                trajectory2, _ = self.collect_trajectories(1)
                feedback = self.oracle.get_feedback(
                    (trajectory, trajectory2[0]), initial_state, feedback_type
                )
            else:
                feedback = self.oracle.get_feedback(
                    trajectory, initial_state, feedback_type
                )

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
                        # Supervised feedback stores per-step items (~segment_len per trajectory)
                        # via extend, so scale the buffer cap accordingly to match other types
                        # which store one item per trajectory via append.
                        effective_cap = self.feedback_buffer_size * max(1, self.oracle.segment_len)
                        if len(self.feedback_buffers[feedback_type]) >= effective_cap:
                            # Remove oldest feedback
                            self.feedback_buffers[feedback_type] = (
                                self.feedback_buffers[feedback_type][len(feedback) :]
                            )
                        self.feedback_buffers[feedback_type].extend(feedback)
                    else:
                        if (
                            len(self.feedback_buffers[feedback_type])
                            >= self.feedback_buffer_size
                        ):
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

    def compute_ensemble_reward(
        self, state: np.ndarray, action: np.ndarray
    ) -> np.ndarray:
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
        state_tensor = torch.as_tensor(
            state, device=device, dtype=torch.float32
        ).unsqueeze(1)
        action_tensor = torch.as_tensor(
            action, device=device, dtype=torch.float32
        ).unsqueeze(1)

        # Lists to accumulate each model's reward and uncertainty
        model_rewards = []
        model_uncertainties = []
        reward_fb_types = []  # track which feedback type produced each reward

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

                        mean_reward, uncertainty = compute_grouped(
                            predictions, reward_model.ensemble_count
                        )
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
                    reward_fb_types.append(feedback_type)

            elif self.reward_model_type == "multi-head":
                # Multi-head model: get predictions from each head
                multi_head_model = list(self.reward_models.values())[
                    0
                ]  # Only one model

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

                    mean_reward, uncertainty = compute_grouped(
                        outputs, multi_head_model.ensemble_count
                    )

                    # Collect
                    model_rewards.append(mean_reward)  # shape [batch_size,]
                    model_uncertainties.append(uncertainty)  # shape [batch_size,]
                    reward_fb_types.append(feedback_type)

            elif self.reward_model_type in ("unified", "film-unified"):
                # Unified / FiLM-unified: same forward API (obs, act, feedback_type)
                unified_model = list(self.reward_models.values())[0]

                for feedback_type in self.feedback_types:
                    if len(self.feedback_buffers[feedback_type]) == 0:
                        continue

                    st_expanded = state_tensor.repeat(
                        unified_model.ensemble_count,
                        *[1] * (len(state_tensor.shape) - 1),
                    )
                    act_expanded = action_tensor.repeat(
                        unified_model.ensemble_count,
                        *[1] * (len(action_tensor.shape) - 1),
                    )

                    predictions = unified_model(
                        st_expanded, act_expanded, feedback_type
                    )

                    mean_reward, uncertainty = compute_grouped(
                        predictions, unified_model.ensemble_count
                    )

                    model_rewards.append(mean_reward)
                    model_uncertainties.append(uncertainty)
                    reward_fb_types.append(feedback_type)

        # If no models have feedback, return zeros for the entire batch
        if not model_rewards:
            return np.zeros(state.shape[0], dtype=np.float32)

        # Stack across models => shape (#models, batch_size)
        stacked_rewards = torch.stack(model_rewards, dim=0)
        stacked_uncerts = torch.stack(model_uncertainties, dim=0)

        # Normalize rewards across feedback types before aggregation
        # Skip normalization entirely when there is only one active feedback type —
        # there is nothing to normalize across, and the running statistics introduce
        # non-stationarity that destabilises PPO.
        n_active = stacked_rewards.shape[0]
        if n_active > 1:
            if self.reward_normalization == "quantile" and reward_fb_types:
                for m_idx in range(n_active):
                    fb_type = reward_fb_types[m_idx] if m_idx < len(reward_fb_types) else f"model_{m_idx}"
                    stacked_rewards[m_idx] = self.quantile_normalizer.update_and_normalize(
                        stacked_rewards[m_idx], fb_type
                    )
            else:
                rewards_for_standardization = stacked_rewards.transpose(0, 1)
                standardized_rewards = self.standardize_rewards(rewards_for_standardization)
                stacked_rewards = standardized_rewards.transpose(0, 1)

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

    def compute_ensemble_reward_with_uncertainty(self, state: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Like compute_ensemble_reward, but also returns average model uncertainty per input."""
        device = self.device
        # Handle one-hot encoding
        if self.action_one_hot:
            action = vectorized_one_hot_vector(np.array(action), self.one_hot_dim)
        if len(action.shape) < 2:
            action = np.expand_dims(action, axis=0)
        state_tensor = torch.as_tensor(state, device=device, dtype=torch.float32).unsqueeze(1)
        action_tensor = torch.as_tensor(action, device=device, dtype=torch.float32).unsqueeze(1)
        model_rewards, model_uncertainties = [], []
        reward_fb_types = []
        with torch.no_grad():
            if self.reward_model_type == "separate":
                for feedback_type, reward_model in self.reward_models.items():
                    if len(self.feedback_buffers[feedback_type]) == 0:
                        continue
                    if reward_model.ensemble_count > 1:
                        st_exp = state_tensor.repeat(reward_model.ensemble_count, *[1] * (len(state_tensor.shape) - 1))
                        ac_exp = action_tensor.repeat(reward_model.ensemble_count, *[1] * (len(action_tensor.shape) - 1))
                        preds = reward_model(st_exp, ac_exp)
                        if preds.dim() == 3 and preds.shape[-1] == 1:
                            preds = preds.squeeze(-1)
                        mean_r, unc = compute_grouped(preds, reward_model.ensemble_count)
                    else:
                        preds = reward_model(state_tensor, action_tensor)
                        if preds.dim() == 2 and preds.shape[1] == 1:
                            preds = preds.squeeze(-1)
                        mean_r = preds
                        unc = torch.zeros_like(mean_r)
                    model_rewards.append(mean_r)
                    model_uncertainties.append(unc)
                    reward_fb_types.append(feedback_type)
            elif self.reward_model_type == "multi-head":
                multi = list(self.reward_models.values())[0]
                st_exp = state_tensor.repeat(multi.ensemble_count, *[1] * (len(state_tensor.shape) - 1))
                ac_exp = action_tensor.repeat(multi.ensemble_count, *[1] * (len(action_tensor.shape) - 1))
                all_outputs = multi(st_exp, ac_exp)
                for feedback_type, outputs in all_outputs.items():
                    if len(self.feedback_buffers[feedback_type]) == 0:
                        continue
                    if outputs.dim() == 3 and outputs.shape[-1] == 1:
                        outputs = outputs.squeeze(-1)
                    mean_r, unc = compute_grouped(outputs, multi.ensemble_count)
                    model_rewards.append(mean_r)
                    model_uncertainties.append(unc)
                    reward_fb_types.append(feedback_type)
            elif self.reward_model_type in ("unified", "film-unified"):
                uni = list(self.reward_models.values())[0]
                for feedback_type in self.feedback_types:
                    if len(self.feedback_buffers[feedback_type]) == 0:
                        continue
                    st_exp = state_tensor.repeat(uni.ensemble_count, *[1] * (len(state_tensor.shape) - 1))
                    ac_exp = action_tensor.repeat(uni.ensemble_count, *[1] * (len(action_tensor.shape) - 1))
                    preds = uni(st_exp, ac_exp, feedback_type)
                    if preds.dim() == 3 and preds.shape[-1] == 1:
                        preds = preds.squeeze(-1)
                    mean_r, unc = compute_grouped(preds, uni.ensemble_count)
                    model_rewards.append(mean_r)
                    model_uncertainties.append(unc)
                    reward_fb_types.append(feedback_type)
        if not model_rewards:
            zeros = np.zeros(state.shape[0], dtype=np.float32)
            return zeros, zeros
        stacked_rewards = torch.stack(model_rewards, dim=0)
        stacked_uncerts = torch.stack(model_uncertainties, dim=0)
        # Normalize rewards across feedback types before aggregation
        # Skip when only one active type (see compute_ensemble_reward)
        n_active = stacked_rewards.shape[0]
        if n_active > 1:
            if self.reward_normalization == "quantile" and reward_fb_types:
                for m_idx in range(n_active):
                    fb_type = reward_fb_types[m_idx] if m_idx < len(reward_fb_types) else f"model_{m_idx}"
                    stacked_rewards[m_idx] = self.quantile_normalizer.update_and_normalize(
                        stacked_rewards[m_idx], fb_type
                    )
            else:
                rewards_for_standardization = stacked_rewards.transpose(0, 1)
                standardized_rewards = self.standardize_rewards(rewards_for_standardization)
                stacked_rewards = standardized_rewards.transpose(0, 1)
        batch_size = state.shape[0]
        final_rewards = torch.zeros(batch_size, device=device, dtype=torch.float32)
        avg_uncert = torch.zeros(batch_size, device=device, dtype=torch.float32)
        for i in range(batch_size):
            r_i = stacked_rewards[:, i]
            u_i = stacked_uncerts[:, i]
            if torch.any(u_i > 0):
                w_i = torch.where(u_i > 0, 1.0 / u_i, torch.ones_like(u_i))
                w_i /= w_i.sum()
                final_rewards[i] = (r_i * w_i).sum()
            else:
                final_rewards[i] = r_i.mean()
            avg_uncert[i] = u_i.mean()
        return final_rewards.cpu().numpy(), avg_uncert.cpu().numpy()

    def _predict_segment_totals(self, feedback_type: str, batch_segments: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
        """Predict total rewards and uncertainties for a batch of segments for a specific feedback type."""
        device = self.device
        if len(batch_segments) == 0:
            return np.array([]), np.array([])
        # Stack batch → (B, T, D)
        obs = torch.stack([s[0] for s in batch_segments]).to(device).float()
        actions = torch.stack([s[1] for s in batch_segments]).to(device).float()
        masks = torch.stack([s[2] for s in batch_segments]).to(device).float()
        # Ensure 3D: (B, T, D) — add sequence dim only if input is 2D (single-step)
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
            actions = actions.unsqueeze(1)
            masks = masks.unsqueeze(1)
        with torch.no_grad():
            if self.reward_model_type == "separate":
                model = self.reward_models[feedback_type]
                if model.ensemble_count > 1:
                    rep = [model.ensemble_count] + [1] * (len(obs.shape) - 1)
                    obs_r = obs.repeat(*rep)
                    act_r = actions.repeat(*rep)
                    out = model(obs_r, act_r)
                    # squeeze output dim (1) then sum over time → (ensemble*B,)
                    totals = (out * masks.repeat(*rep)).squeeze(-1).sum(dim=1)
                    mean_r, unc = compute_grouped(totals, model.ensemble_count)
                else:
                    out = model(obs, actions)
                    totals = (out * masks).squeeze(-1).sum(dim=1)
                    mean_r, unc = totals, torch.zeros_like(totals)
            elif self.reward_model_type == "multi-head":
                model = list(self.reward_models.values())[0]
                rep = [model.ensemble_count] + [1] * (len(obs.shape) - 1)
                obs_r = obs.repeat(*rep)
                act_r = actions.repeat(*rep)
                out = model(obs_r, act_r, feedback_type)
                totals = (out * masks.repeat(*rep)).squeeze(-1).sum(dim=1)
                mean_r, unc = compute_grouped(totals, model.ensemble_count)
            else:  # unified
                model = list(self.reward_models.values())[0]
                rep = [model.ensemble_count] + [1] * (len(obs.shape) - 1)
                obs_r = obs.repeat(*rep)
                act_r = actions.repeat(*rep)
                out = model(obs_r, act_r, feedback_type)
                totals = (out * masks.repeat(*rep)).squeeze(-1).sum(dim=1)
                mean_r, unc = compute_grouped(totals, model.ensemble_count)
        return mean_r.cpu().numpy(), unc.cpu().numpy()

    @staticmethod
    def _pearson_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        if len(x) == 0 or len(y) == 0:
            return float("nan"), float("nan")
        x = np.asarray(x)
        y = np.asarray(y)
        x_mu, y_mu = x.mean(), y.mean()
        vx, vy = x - x_mu, y - y_mu
        denom = np.sqrt((vx**2).sum()) * np.sqrt((vy**2).sum()) + 1e-8
        pearson = float(np.clip((vx * vy).sum() / denom, -1.0, 1.0))
        # Spearman via ranking
        rx = np.argsort(np.argsort(x))
        ry = np.argsort(np.argsort(y))
        rx_mu, ry_mu = rx.mean(), ry.mean()
        vrx, vry = rx - rx_mu, ry - ry_mu
        denom_s = np.sqrt((vrx**2).sum()) * np.sqrt((vry**2).sum()) + 1e-8
        spearman = float(np.clip((vrx * vry).sum() / denom_s, -1.0, 1.0))
        return pearson, spearman

    @staticmethod
    def _binary_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
        """Compute ROC-AUC without sklearn. y_true in {0,1}."""
        y = np.asarray(y_true).astype(np.int32)
        s = np.asarray(scores).astype(np.float64)
        # Rank scores, handle ties by average rank
        order = np.argsort(s)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(s) + 1)
        # average ranks for ties
        _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
        sum_ranks = np.bincount(inv, ranks)
        avg_ranks = sum_ranks / counts
        ranks = avg_ranks[inv]
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        sum_ranks_pos = (ranks * y).sum()
        auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        return float(auc)

    def _rollout(self, n_episodes: int = 5) -> tuple[list[list[tuple]], list[float], list[float]]:
        env = TrainingUtils.setup_environment(self.env_name, (self.seed or 0) + 321, env_kwargs=self.env_kwargs or None)
        trajectories, returns, successes = [], [], []
        for _ in range(n_episodes):
            traj = []
            obs, _ = env.reset()
            ep_return = 0.0
            ep_success = 0.0
            for _t in range(self.oracle.segment_len):
                if self.rl_agent is None:
                    action = env.action_space.sample()
                else:
                    action, _ = self.rl_agent.predict(obs, deterministic=True)
                nobs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                a = one_hot_vector(action, env.action_space.n) if isinstance(env.action_space, gym.spaces.Discrete) else action
                traj.append((np.expand_dims(obs, axis=0), a, reward, done))
                ep_return += reward
                # Meta-World success proxy if provided
                if isinstance(info, dict):
                    if "success" in info:
                        ep_success = max(ep_success, float(info["success"]))
                    elif "is_success" in info:
                        ep_success = max(ep_success, float(info["is_success"]))
                obs = nobs
                if done:
                    break
            returns.append(ep_return)
            successes.append(ep_success)
            trajectories.append(traj)
        env.close()
        return trajectories, returns, successes

    def _log_iteration_metrics(self, step: int):
        """Compute and log evaluation metrics at the end of an iteration/update."""
        if not (hasattr(self, "wandb") and self.wandb.run is not None):
            return
        metrics = {}

        # 1) Proxy Gap: compare downstream GT returns vs learned reward returns
        try:
            trajs, gt_returns, successes = self._rollout(n_episodes=5)
            # predicted totals via learned reward
            pred_returns = []
            for traj in trajs:
                states = np.squeeze(np.array([p[0] for p in traj]), axis=1)
                actions = np.array([p[1] for p in traj])
                preds = self.compute_ensemble_reward(states, actions)
                pred_returns.append(float(preds.sum()))
            pr, sr = self._pearson_spearman(np.array(gt_returns), np.array(pred_returns))
            metrics.update({
                "proxy_gap/gt_return_mean": float(np.mean(gt_returns)),
                "proxy_gap/pred_return_mean": float(np.mean(pred_returns)),
                "proxy_gap/pearson": pr,
                "proxy_gap/spearman": sr,
                "proxy_gap/abs_gap": float(np.mean(pred_returns) - np.mean(gt_returns)),
            })
            if len(successes) > 0:
                metrics["external/win_rate"] = float(np.mean(successes))
        except Exception:
            pass

        # 2) Reward model metrics per feedback type (correlation on holdout)
        if hasattr(self, "eval_holdout") and self.eval_holdout:
            for fb in self.feedback_types:
                # Skip if we don't have a model for that type yet
                if self.reward_model_type == "separate" and fb not in self.reward_models:
                    continue
                batch_segments = [h[0] for h in self.eval_holdout]
                gt_totals = np.array([h[1] for h in self.eval_holdout])
                try:
                    preds, uncs = self._predict_segment_totals(fb, batch_segments)
                except Exception:
                    preds, uncs = np.array([]), np.array([])
                if preds.size > 0:
                    pr, sr = self._pearson_spearman(preds, gt_totals)
                    metrics.update({
                        f"reward_model/{fb}_pearson": pr,
                        f"reward_model/{fb}_spearman": sr,
                        f"reward_model/{fb}_mse": float(np.mean((preds - gt_totals) ** 2)),
                        f"reward_model/{fb}_pred_mean": float(np.mean(preds)),
                        f"reward_model/{fb}_pred_std": float(np.std(preds)),
                        f"reward_model/{fb}_uncertainty_mean": float(np.mean(uncs)) if uncs.size > 0 else float("nan"),
                    })

        # Pairwise metrics: accuracy and ROC-AUC
        if hasattr(self, "eval_pairs") and self.eval_pairs:
            for fb in self.feedback_types:
                if fb not in ["comparative", "demonstrative", "corrective", "descriptive_preference"]:
                    continue
                try:
                    pair_segments = []
                    labels = []
                    for (o1, a1, m1), (o2, a2, m2), lbl in self.eval_pairs:
                        pair_segments.append(((o1, a1, m1), (o2, a2, m2)))
                        labels.append(lbl)
                    left = [p[0] for p in pair_segments]
                    right = [p[1] for p in pair_segments]
                    l_pred, _ = self._predict_segment_totals(fb, left)
                    r_pred, _ = self._predict_segment_totals(fb, right)
                    if l_pred.size > 0 and r_pred.size > 0:
                        scores = r_pred - l_pred
                        preds = (scores > 0).astype(np.int32)
                        acc = float(np.mean(preds == np.array(labels)))
                        auc = self._binary_roc_auc(np.array(labels), scores)
                        metrics.update({
                            f"reward_model/{fb}_pair_acc": acc,
                            f"reward_model/{fb}_pair_auc": auc,
                        })
                except Exception:
                    continue

        # 3) Predicted reward distribution and avg uncertainty on holdout SA
        if hasattr(self, "eval_holdout_sa") and self.eval_holdout_sa:
            try:
                states = np.squeeze(np.array([s[0] for s in self.eval_holdout_sa]), axis=1)
                actions = np.array([s[1] for s in self.eval_holdout_sa])
                gt_step = np.array([s[2] for s in self.eval_holdout_sa])
                pred, unc = self.compute_ensemble_reward_with_uncertainty(states, actions)
                pr_s, sr_s = self._pearson_spearman(pred, gt_step)
                # Reward hacking proxy: top 10% predicted but bottom 25% GT
                top_p = np.percentile(pred, 90)
                low_q = np.percentile(gt_step, 25)
                hack_frac = float(np.mean((pred >= top_p) & (gt_step <= low_q)))
                metrics.update({
                    "pred_dist/pred_mean": float(np.mean(pred)),
                    "pred_dist/pred_std": float(np.std(pred)),
                    "pred_dist/pred_min": float(np.min(pred)),
                    "pred_dist/pred_max": float(np.max(pred)),
                    "uncertainty/avg": float(np.mean(unc)),
                    "scale_offset/holdout_pred_mean": float(np.mean(pred)),
                    "scale_offset/holdout_pred_std": float(np.std(pred)),
                    "scale_offset/holdout_gt_mean": float(np.mean(gt_step)),
                    "scale_offset/holdout_gt_std": float(np.std(gt_step)),
                    "correlation/step_pearson": pr_s,
                    "correlation/step_spearman": sr_s,
                    "reward_hacking/frac_high_pred_low_gt": hack_frac,
                })
            except Exception:
                pass

        # 4) OOD vs ID performance (using ood_holdout totals)
        try:
            # ID correlation based on step-level above if available
            id_pearson = metrics.get("correlation/step_pearson", float("nan"))
            if hasattr(self, "ood_holdout") and self.ood_holdout:
                segs = [h[0] for h in self.ood_holdout]
                gt = np.array([h[1] for h in self.ood_holdout])
                fb = "evaluative" if "evaluative" in self.feedback_types else self.feedback_types[0]
                ood_pred, _ = self._predict_segment_totals(fb, segs)
                ood_pr, ood_sr = self._pearson_spearman(ood_pred, gt)
                metrics.update({
                    "ood/pearson": ood_pr,
                    "ood/spearman": ood_sr,
                    "ood/id_minus_ood_pearson": (id_pearson - ood_pr) if not np.isnan(ood_pr) and not np.isnan(id_pearson) else float("nan"),
                })
        except Exception:
            pass

        # 5) EPIC alternative: 1 - Spearman on holdout totals as proxy distance
        try:
            if hasattr(self, "eval_holdout") and self.eval_holdout:
                fb = "evaluative" if "evaluative" in self.feedback_types else self.feedback_types[0]
                segs = [h[0] for h in self.eval_holdout]
                gt = np.array([h[1] for h in self.eval_holdout])
                preds, _ = self._predict_segment_totals(fb, segs)
                _, sp = self._pearson_spearman(preds, gt)
                metrics["epic_proxy/1_minus_spearman"] = float(1.0 - (sp if not np.isnan(sp) else 0.0))
        except Exception:
            pass

        if metrics:
            self.wandb.log(metrics, step=step)

    def print_reward_model_diagnostics(self):
        """Print reward model quality metrics to stdout (no wandb needed)."""
        lines = []

        # Per-type holdout correlation (segment-level)
        if hasattr(self, "eval_holdout") and self.eval_holdout:
            batch_segments = [h[0] for h in self.eval_holdout]
            gt_totals = np.array([h[1] for h in self.eval_holdout])
            for fb in self.feedback_types:
                try:
                    preds, uncs = self._predict_segment_totals(fb, batch_segments)
                    if preds.size > 0:
                        pr, sr = self._pearson_spearman(preds, gt_totals)
                        mse = float(np.mean((preds - gt_totals) ** 2))
                        lines.append(
                            f"    {fb:20s}  pearson={pr:+.3f}  spearman={sr:+.3f}  "
                            f"MSE={mse:.4f}  pred_mean={np.mean(preds):.3f}  gt_mean={np.mean(gt_totals):.3f}"
                        )
                except Exception:
                    lines.append(f"    {fb:20s}  (prediction failed)")

        # Step-level correlation (ensemble output vs GT per-step reward)
        if hasattr(self, "eval_holdout_sa") and self.eval_holdout_sa:
            try:
                states = np.squeeze(np.array([s[0] for s in self.eval_holdout_sa]), axis=1)
                actions = np.array([s[1] for s in self.eval_holdout_sa])
                gt_step = np.array([s[2] for s in self.eval_holdout_sa])
                pred, unc = self.compute_ensemble_reward_with_uncertainty(states, actions)
                pr_s, sr_s = self._pearson_spearman(pred, gt_step)
                lines.append(
                    f"    {'ensemble (step)':20s}  pearson={pr_s:+.3f}  spearman={sr_s:+.3f}  "
                    f"pred_std={np.std(pred):.4f}  gt_std={np.std(gt_step):.4f}"
                )
            except Exception as e:
                lines.append(f"    ensemble (step)       (failed: {e})")

        # Pairwise accuracy
        if hasattr(self, "eval_pairs") and self.eval_pairs:
            for fb in self.feedback_types:
                if fb not in ["comparative", "demonstrative", "corrective", "descriptive_preference"]:
                    continue
                try:
                    left = [p[0] for p in [(e[0:2]) for e in [(((o1, a1, m1), (o2, a2, m2)), lbl) for (o1, a1, m1), (o2, a2, m2), lbl in self.eval_pairs]]]
                    right = [p[1] for p in [(e[0:2]) for e in [(((o1, a1, m1), (o2, a2, m2)), lbl) for (o1, a1, m1), (o2, a2, m2), lbl in self.eval_pairs]]]
                    labels = [lbl for _, _, lbl in self.eval_pairs]
                    l_pred, _ = self._predict_segment_totals(fb, [l for l in left])
                    r_pred, _ = self._predict_segment_totals(fb, [r for r in right])
                    if l_pred.size > 0 and r_pred.size > 0:
                        scores = r_pred - l_pred
                        acc = float(np.mean((scores > 0).astype(int) == np.array(labels)))
                        lines.append(f"    {fb + ' (pair)':20s}  accuracy={acc:.3f}")
                except Exception:
                    pass

        if lines:
            print("  Reward model diagnostics:")
            for line in lines:
                print(line)
        else:
            print("  Reward model diagnostics: no holdout data available")

    def train(self, total_timesteps: Optional[int] = None, sampling_strategy: str = "random", query_sampling_strategy: str = "none", query_sampling_multiplier: float = 2.0):
        """
        Run full training loop with a single call to learn() and using callbacks
        for reward model updates.
        """

        if total_timesteps is not None:
            self.total_timesteps = int(total_timesteps)
            # If the user overrides total_timesteps post-init, recompute RL steps
            self._compute_rl_steps_after_init()

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
                callback = CallbackList(
                    [reward_model_callback, self.external_callbacks]
                )
        else:
            callback = reward_model_callback


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

    def save_reward_models_checkpoint(self, checkpoint_step: int, exp_id: str) -> dict:
        """Save reward models to projection-compatible checkpoint paths."""
        import pytorch_lightning as pl

        checkpoint_dir = Path("multi-type-feedback/reward_models/checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        saved_models = {}

        if self.reward_model_type == "separate":
            for feedback_type, model in self.reward_models.items():
                model_filename = f"{self.algorithm}_{self.env_name.lower().replace('-', '_')}_{exp_id}_{feedback_type}_{checkpoint_step}.ckpt"
                model_path = checkpoint_dir / model_filename
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
        else:
            model = list(self.reward_models.values())[0]
            model_filename = f"{self.algorithm}_{self.env_name.lower().replace('-', '_')}_{exp_id}_{self.reward_model_type}_{checkpoint_step}.ckpt"
            model_path = checkpoint_dir / model_filename
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
        Mirrors the human version's save() for cross-compatibility with the projection system.
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        if checkpoint_step is not None and exp_id is not None:
            saved_model_paths = self.save_reward_models_checkpoint(checkpoint_step, exp_id)
            agent_dir = Path("multi-type-feedback/train_baselines/dynamic_rlhf_agents")
            agent_dir.mkdir(parents=True, exist_ok=True)
            agent_filename = f"{self.algorithm}_{self.env_name.lower().replace('-', '_')}_{exp_id}_{checkpoint_step}.zip"
            agent_path = agent_dir / agent_filename
            self.rl_agent.save(agent_path)
            state_data_model_paths = saved_model_paths
            state_data_agent_path = str(agent_path)
        else:
            rl_agent_path = save_path / "rl_agent"
            self.rl_agent.save(rl_agent_path)
            reward_models_path = save_path / "reward_models"
            reward_models_path.mkdir(exist_ok=True)
            state_data_model_paths = {}
            for feedback_type, model in self.reward_models.items():
                model_path = reward_models_path / f"{feedback_type}.ckpt"
                torch.save(model.state_dict(), model_path)
                state_data_model_paths[feedback_type] = str(model_path)
            state_data_agent_path = str(rl_agent_path) + ".zip"

        state_data = {
            "env_name": self.env_name,
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
            "reward_model_hidden_dim": self.reward_model_hidden_dim,
            "reward_model_layer_num": self.reward_model_layer_num,
            "action_one_hot": self.action_one_hot,
            "one_hot_dim": getattr(self, "one_hot_dim", None),
            "feedback_buffers": self.feedback_buffers,
            "reward_mean": self.reward_mean.cpu().numpy() if self.reward_mean is not None else None,
            "squared_distance_from_mean": (
                self.squared_distance_from_mean.cpu().numpy()
                if self.squared_distance_from_mean is not None
                else None
            ),
            "reward_counters": self.reward_counters.cpu().numpy() if self.reward_counters is not None else None,
            "saved_model_paths": state_data_model_paths,
            "saved_agent_path": state_data_agent_path,
            "checkpoint_step": checkpoint_step,
            "exp_id": exp_id,
        }

        state_path = str(save_path) + "_state.pkl"
        with open(state_path, "wb") as f:
            pickle.dump(state_data, f)

        print(f"DynamicRLHF model saved to {save_path}")

    @classmethod
    def load(cls, load_path: str, oracle: "FeedbackOracle" = None, exp_manager: "ExperimentManager" = None) -> "DynamicRLHF":
        """
        Load a DynamicRLHF (simulated) instance from a checkpoint written by save().

        Args:
            load_path: Base directory that was passed to save() (without _state.pkl suffix)
            oracle: FeedbackOracle required for further training; can be None for eval-only use
            exp_manager: Optional ExperimentManager for RL training
        """
        load_path = Path(load_path)
        state_path = str(load_path) + "_state.pkl"
        with open(state_path, "rb") as f:
            state_data = pickle.load(f)

        drlhf = cls(
            oracle=oracle,
            env_name=state_data["env_name"],
            algorithm=state_data["algorithm"],
            feedback_types=state_data["feedback_types"],
            n_feedback_per_iteration=state_data["n_feedback_per_iteration"],
            feedback_buffer_size=state_data["feedback_buffer_size"],
            rl_steps_per_iteration=state_data["rl_steps_per_iteration"],
            reward_training_epochs=state_data["reward_training_epochs"],
            device=state_data["device"],
            num_ensemble_models=state_data["num_ensemble_models"],
            initial_feedback_count=0,  # skip re-initialization; restore buffers below
            reward_model_type=state_data["reward_model_type"],
            shared_layer_num=state_data["shared_layer_num"],
            head_layer_num=state_data["head_layer_num"],
            feedback_embedding_dim=state_data["feedback_embedding_dim"],
            reward_model_hidden_dim=state_data.get("reward_model_hidden_dim", 256),
            reward_model_layer_num=state_data.get("reward_model_layer_num", 6),
            exp_manager=exp_manager,
        )

        agent_path = state_data.get("saved_agent_path", str(load_path / "rl_agent.zip"))
        if drlhf.algorithm.lower() == "ppo":
            drlhf.rl_agent = PPO.load(agent_path)
        else:
            drlhf.rl_agent = SAC.load(agent_path)

        saved_model_paths = state_data.get("saved_model_paths")
        if saved_model_paths:
            for feedback_type, model in drlhf.reward_models.items():
                if feedback_type in saved_model_paths:
                    model_path = saved_model_paths[feedback_type]
                    if Path(model_path).exists():
                        model_class = type(model)
                        loaded_model = model_class.load_from_checkpoint(model_path)
                        drlhf.reward_models[feedback_type] = loaded_model
                        loaded_model.to(drlhf.device)
                        loaded_model.eval()
        else:
            reward_models_path = load_path / "reward_models"
            for feedback_type, model in drlhf.reward_models.items():
                model_path = reward_models_path / f"{feedback_type}.ckpt"
                if model_path.exists():
                    model.load_state_dict(torch.load(model_path, map_location=drlhf.device))
                    model.to(drlhf.device)

        drlhf.action_one_hot = state_data["action_one_hot"]
        if state_data.get("one_hot_dim") is not None:
            drlhf.one_hot_dim = state_data["one_hot_dim"]
        drlhf.feedback_buffers = state_data["feedback_buffers"]

        if state_data["reward_mean"] is not None:
            drlhf.reward_mean = torch.tensor(state_data["reward_mean"]).to(drlhf.device)
        if state_data["squared_distance_from_mean"] is not None:
            drlhf.squared_distance_from_mean = torch.tensor(state_data["squared_distance_from_mean"]).to(drlhf.device)
        if state_data["reward_counters"] is not None:
            drlhf.reward_counters = torch.tensor(state_data["reward_counters"]).to(drlhf.device)

        print(f"DynamicRLHF model loaded from {load_path}")
        return drlhf


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
        "--expert-algorithm",
        type=str,
        default=None,
        help="Optional: We can load the expert policy with a separate training algorithm",
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
        default=250,
        help="Number of feedback samples to collect before starting RL training",
    )
    parser.add_argument(
        "--feedback-buffer-size",
        type=int,
        default=750,
        help="Maximum size of the feedback buffer",
    )
    parser.add_argument(
        "--top-n-models", type=int, default=1, help="Top N models to use"
    )
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