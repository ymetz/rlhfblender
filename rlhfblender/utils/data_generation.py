import os
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import cv2
import gymnasium as gym
import numpy as np
from databases import Database
from stable_baselines3.common.vec_env import VecEnv

import rlhfblender.data_collection.environment_handler as environment_handler
import rlhfblender.data_collection.framework_selector as framework_selector
import rlhfblender.data_handling.database_handler as db_handler
from rlhfblender.data_collection.episode_recorder import EpisodeRecorder
from rlhfblender.data_models.global_models import Environment, Experiment, Project
from rlhfblender.utils import process_env_name
from rlhfblender.utils.read_sb3_configs import read_sb3_configs

# Initialize database
database = Database(os.environ.get("RLHFBLENDER_DB_HOST", "sqlite:///rlhfblender.db"))

DATA_ROOT_DIR = "data"
BENCHMARK_DIR = "saved_benchmarks"


def _resolve_norm_env_path(model_path: str, env_id: str) -> str:
    """
    Resolve directory that should contain SB3 stats/config artifacts.
    Supports both:
      - <run_dir>/<processed_env>/vecnormalize.pkl
      - <processed_env_dir>/vecnormalize.pkl
    """
    env_component = process_env_name(env_id)
    candidates = [
        os.path.join(model_path, env_component),
        model_path,
    ]

    for candidate in candidates:
        if os.path.isfile(os.path.join(candidate, "vecnormalize.pkl")) or os.path.isfile(os.path.join(candidate, "config.yml")):
            return candidate

    return candidates[0]


def _read_training_config(model_path: str, env_id: str) -> dict[str, Any]:
    """Best-effort read of SB3 zoo `config.yml` for env wrapper settings."""
    env_component = process_env_name(env_id)
    config_candidates = [
        os.path.join(model_path, env_component, "config.yml"),
        os.path.join(model_path, "config.yml"),
    ]

    for config_path in config_candidates:
        if not os.path.isfile(config_path):
            continue
        try:
            parsed = read_sb3_configs(config_path)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue

    return {}


def _parse_normalize_kwargs(normalize_value: Any) -> tuple[bool, dict[str, Any]]:
    """Normalize SB3 `normalize` config into (enabled, kwargs)."""
    if isinstance(normalize_value, dict):
        return True, dict(normalize_value)

    if isinstance(normalize_value, str):
        stripped = normalize_value.strip()
        if stripped.lower() in {"", "false", "0", "none"}:
            return False, {}
        if stripped.startswith("dict("):
            try:
                parsed = eval(stripped, {"__builtins__": {}, "dict": dict}, {})
                if isinstance(parsed, dict):
                    return True, dict(parsed)
            except Exception:
                pass
        return True, {}

    return bool(normalize_value), {}


def _resolve_benchmark_environment_config(
    exp: Experiment,
    model_path: str,
    norm_env_path: str,
) -> dict[str, Any]:
    """
    Compose environment config for benchmark rollouts.
    Ensures normalization settings are available when training used VecNormalize.
    """
    env_config = deepcopy(exp.environment_config) if isinstance(exp.environment_config, dict) else {}
    env_config.setdefault("env_kwargs", {})

    training_config = _read_training_config(model_path=model_path, env_id=exp.env_id)

    if "normalize" not in env_config:
        normalize_enabled = False
        normalize_kwargs: dict[str, Any] = {}

        if "normalize" in training_config:
            normalize_enabled, normalize_kwargs = _parse_normalize_kwargs(training_config.get("normalize"))
        elif os.path.isfile(os.path.join(norm_env_path, "vecnormalize.pkl")):
            # Fallback: if stats exist, assume training used VecNormalize.
            normalize_enabled = True

        if normalize_enabled:
            env_config["normalize"] = True
            env_config["normalize_kwargs"] = normalize_kwargs

    if env_config.get("normalize", False):
        env_config.setdefault("normalize_kwargs", {})

        # Align with SB3 behavior: include gamma in normalize kwargs when available.
        if "gamma" in training_config and "gamma" not in env_config["normalize_kwargs"]:
            env_config["normalize_kwargs"]["gamma"] = training_config["gamma"]

    if "frame_stack" not in env_config and "frame_stack" in training_config:
        env_config["frame_stack"] = training_config["frame_stack"]

    return env_config


def _sanitize_for_filename(value: str) -> str:
    """Return a filesystem-friendly name."""
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)


def _normalize_observation_step(obs_step: Any, expected_obs_dim: int) -> np.ndarray | None:
    """Flatten and validate one observation step against the expected dimension."""
    flat_obs = np.asarray(obs_step, dtype=np.float32).reshape(-1)
    if expected_obs_dim > 0 and flat_obs.size != expected_obs_dim:
        return None
    return flat_obs


def _normalize_action_step(
    action_step: Any,
    is_discrete_action: bool,
    action_dim: int,
) -> tuple[np.ndarray | None, str | None]:
    """Normalize one action step to fixed dimensional representation."""
    flat_action = np.asarray(action_step)

    if is_discrete_action:
        flat_action = flat_action.reshape(-1)
        if flat_action.size == 0:
            return None, "invalid_action"
        if flat_action.size == action_dim and np.isfinite(flat_action).all():
            action_index = int(np.argmax(flat_action))
        else:
            action_index = int(flat_action[0])
        if action_index < 0 or action_index >= action_dim:
            return None, "invalid_action"
        one_hot = np.zeros(action_dim, dtype=np.float32)
        one_hot[action_index] = 1.0
        return one_hot, None

    flat_action = np.asarray(action_step, dtype=np.float32).reshape(-1)
    if action_dim > 0 and flat_action.size != action_dim:
        return None, "dim_mismatch"
    return flat_action, None


def _filter_consistent_supervised_samples(
    samples: list[tuple[tuple[Any, Any, Any], Any]],
    checkpoint_step: str,
) -> list[tuple[tuple[Any, Any, Any], Any]]:
    """Keep only samples matching the most common tensor-shape signature."""
    if not samples:
        return samples

    shape_counter = Counter(
        (
            tuple(sample[0][0].shape),  # obs tensor shape
            tuple(sample[0][1].shape),  # action tensor shape
            tuple(sample[0][2].shape),  # mask tensor shape
        )
        for sample in samples
    )
    target_signature, _ = shape_counter.most_common(1)[0]
    filtered_samples = []
    for sample in samples:
        signature = (
            tuple(sample[0][0].shape),
            tuple(sample[0][1].shape),
            tuple(sample[0][2].shape),
        )
        if signature == target_signature:
            filtered_samples.append(sample)

    if len(filtered_samples) < len(samples):
        print(
            "[INFO] Dropped inconsistent supervised samples before training: "
            f"kept={len(filtered_samples)}, dropped={len(samples) - len(filtered_samples)}, "
            f"target_signature={target_signature}, checkpoint={checkpoint_step}"
        )

    return filtered_samples


def _create_supervised_reward_model(
    env_id: str,
    observation_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
    learning_rate: float,
    ensemble_count: int,
):
    """Create reward model architecture compatible with environment observation/action spaces."""
    from multi_type_feedback.networks import (
        SingleCnnNetwork,
        SingleNetwork,
        calculate_single_reward_loss,
    )

    is_visual_env = "ALE/" in env_id or "procgen" in env_id
    model_class = SingleCnnNetwork if is_visual_env else SingleNetwork
    return model_class(
        input_spaces=(observation_space, action_space),
        hidden_dim=256,
        action_hidden_dim=(16 if is_visual_env else 32),
        layer_num=(3 if is_visual_env else 6),
        cnn_channels=((16, 32, 32) if is_visual_env else None),
        output_dim=1,
        loss_function=calculate_single_reward_loss,
        learning_rate=learning_rate,
        ensemble_count=ensemble_count,
    )


def _maybe_warm_start_reward_model(reward_model, warm_start_checkpoint: str | None) -> None:
    """Load model weights from a previous checkpoint when available."""
    import torch

    if not warm_start_checkpoint or not os.path.isfile(warm_start_checkpoint):
        return

    try:
        warm_state = torch.load(warm_start_checkpoint, map_location="cpu", weights_only=False)
        state_dict = (
            warm_state["state_dict"]
            if isinstance(warm_state, dict) and "state_dict" in warm_state
            else warm_state
        )
        reward_model.load_state_dict(state_dict, strict=False)
        print(f"[INFO] Warm-started reward model from {warm_start_checkpoint}")
    except Exception as warm_error:
        print(f"[WARN] Failed to warm-start from {warm_start_checkpoint}: {warm_error}")


def _extract_step_reward_samples(
    episode_data: dict[str, np.ndarray],
    max_trajectories: int,
    observation_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
) -> list[tuple[tuple[Any, Any, Any], Any]]:
    """
    Convert episode-wise benchmark recordings into supervised single-step samples:
    ((obs_t, act_t, mask_t), reward_t).
    """
    import torch

    if max_trajectories <= 0:
        return []

    is_discrete_action = isinstance(action_space, gym.spaces.Discrete)
    action_dim = action_space.n if is_discrete_action else int(np.prod(getattr(action_space, "shape", ()) or (1,)))
    expected_obs_dim = int(np.prod(getattr(observation_space, "shape", ()) or (1,)))

    samples: list[tuple[tuple[Any, Any, Any], Any]] = []
    used_trajectories = 0
    skipped_counts = {"dim_mismatch": 0, "invalid_action": 0, "non_finite_reward": 0}

    n_episodes = min(
        len(episode_data.get("obs", [])),
        len(episode_data.get("actions", [])),
        len(episode_data.get("rewards", [])),
        len(episode_data.get("dones", [])),
    )

    for episode_idx in range(n_episodes):
        if used_trajectories >= max_trajectories:
            break

        obs_episode = np.asarray(episode_data["obs"][episode_idx])
        action_episode = np.asarray(episode_data["actions"][episode_idx])
        reward_episode = np.asarray(episode_data["rewards"][episode_idx])
        done_episode = np.asarray(episode_data["dones"][episode_idx])

        if obs_episode.size == 0 or action_episode.size == 0 or reward_episode.size == 0:
            continue

        flat_done = done_episode.reshape(-1) if done_episode.shape else np.array([done_episode])
        if flat_done.size == 0 or not bool(flat_done[-1]):
            # Ignore incomplete trajectories.
            continue

        used_trajectories += 1
        n_steps = min(obs_episode.shape[0], action_episode.shape[0], reward_episode.shape[0])

        for step_idx in range(n_steps):
            obs_step = _normalize_observation_step(obs_episode[step_idx], expected_obs_dim)
            if obs_step is None:
                skipped_counts["dim_mismatch"] += 1
                continue

            normalized_action, error_type = _normalize_action_step(
                action_episode[step_idx], is_discrete_action=is_discrete_action, action_dim=action_dim
            )
            if normalized_action is None:
                skipped_counts[error_type or "dim_mismatch"] += 1
                continue

            reward_value = float(np.asarray(reward_episode[step_idx]).reshape(-1)[0])
            if not np.isfinite(reward_value):
                skipped_counts["non_finite_reward"] += 1
                continue

            obs_tensor = torch.as_tensor(obs_step, dtype=torch.float32).unsqueeze(0)
            action_tensor = torch.as_tensor(normalized_action, dtype=torch.float32).unsqueeze(0)
            mask_tensor = torch.ones((1, 1), dtype=torch.float32)
            reward_tensor = torch.as_tensor(reward_value, dtype=torch.float32)

            samples.append(((obs_tensor, action_tensor, mask_tensor), reward_tensor))

    if any(value > 0 for value in skipped_counts.values()):
        print(
            "[INFO] Supervised sample filtering summary: "
            f"dim_mismatch={skipped_counts['dim_mismatch']}, "
            f"invalid_action={skipped_counts['invalid_action']}, "
            f"non_finite_reward={skipped_counts['non_finite_reward']}, "
            f"kept={len(samples)}"
        )

    return samples


def _build_and_train_supervised_reward_model(  # noqa: C901
    samples: list[tuple[tuple[Any, Any, Any], Any]],
    observation_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
    env_id: str,
    exp_name: str,
    checkpoint_step: str,
    save_dir: str,
    warm_start_checkpoint: str | None = None,
    max_epochs: int = 100,
    patience: int = 8,
    validation_split: float = 0.2,
    batch_size: int = 64,
    learning_rate: float = 1e-5,
    ensemble_count: int = 4,
    seed: int = 42,
) -> str | None:
    """Train a supervised reward model from step-level samples and save as a PL-compatible checkpoint."""
    if len(samples) < 4:
        print(
            f"[WARN] Not enough supervised samples for checkpoint {checkpoint_step}. "
            f"Need >=4, got {len(samples)}. Skipping reward model training."
        )
        return None

    samples = _filter_consistent_supervised_samples(samples=samples, checkpoint_step=checkpoint_step)

    if len(samples) < 4:
        print(
            f"[WARN] Not enough consistent supervised samples for checkpoint {checkpoint_step} after filtering. "
            f"Need >=4, got {len(samples)}. Skipping reward model training."
        )
        return None

    import pytorch_lightning as pl
    import torch
    from pytorch_lightning import Callback, Trainer
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    from torch.utils.data import DataLoader, Dataset, random_split

    class _StepRewardDataset(Dataset):
        def __init__(self, data: list[tuple[tuple[Any, Any, Any], Any]]):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            return self.data[index]

    class _BestStateCallback(Callback):
        def __init__(self):
            super().__init__()
            self.best_val_loss = float("inf")
            self.best_state_dict: dict[str, Any] | None = None

        def on_validation_epoch_end(self, trainer, pl_module):
            if trainer.sanity_checking:
                return
            metric = trainer.callback_metrics.get("val_loss")
            if metric is None:
                return
            val_loss = float(metric.detach().cpu().item())
            if np.isfinite(val_loss) and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in pl_module.state_dict().items()
                }

    pl.seed_everything(seed, workers=True)

    full_dataset = _StepRewardDataset(samples)
    raw_val_size = int(len(full_dataset) * validation_split)
    val_size = min(max(1, raw_val_size), len(full_dataset) - 1)
    train_size = len(full_dataset) - val_size

    if train_size < 1 or val_size < 1:
        print(
            f"[WARN] Invalid train/validation split for checkpoint {checkpoint_step} "
            f"(train={train_size}, val={val_size}). Skipping."
        )
        return None

    split_generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_dataset, lengths=[train_size, val_size], generator=split_generator)

    effective_ensemble = ensemble_count if train_size >= ensemble_count else 1

    # Masksemble-based models require enough samples and ensemble-compatible batching.
    # Fall back to a single model if either split is too small.
    if effective_ensemble > 1 and (train_size < effective_ensemble or val_size < effective_ensemble):
        print(
            f"[INFO] Falling back to ensemble_count=1 for checkpoint {checkpoint_step} "
            f"(train_size={train_size}, val_size={val_size}, requested_ensemble={effective_ensemble})."
        )
        effective_ensemble = 1

    effective_batch_size = max(1, min(batch_size, train_size))
    if effective_ensemble > 1:
        # Ensure train batch size is a positive multiple of ensemble size.
        effective_batch_size = (effective_batch_size // effective_ensemble) * effective_ensemble
        if effective_batch_size < effective_ensemble:
            effective_batch_size = effective_ensemble

    val_batch_size = max(1, min(effective_batch_size, val_size))
    if effective_ensemble > 1:
        val_batch_size = (val_batch_size // effective_ensemble) * effective_ensemble
        if val_batch_size < effective_ensemble:
            val_batch_size = effective_ensemble

    train_loader = DataLoader(
        train_set,
        batch_size=effective_batch_size,
        shuffle=True,
        pin_memory=False,
        num_workers=0,
        drop_last=effective_ensemble > 1,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=val_batch_size,
        shuffle=False,
        pin_memory=False,
        num_workers=0,
        drop_last=effective_ensemble > 1,
    )

    reward_model = _create_supervised_reward_model(
        env_id=env_id,
        observation_space=observation_space,
        action_space=action_space,
        learning_rate=learning_rate,
        ensemble_count=effective_ensemble,
    )
    _maybe_warm_start_reward_model(reward_model=reward_model, warm_start_checkpoint=warm_start_checkpoint)

    best_state_callback = _BestStateCallback()
    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=max(1, patience),
        min_delta=1e-6,
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(
        max_epochs=max(1, max_epochs),
        accelerator=accelerator,
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[early_stopping, best_state_callback],
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        enable_model_summary=False,
    )

    trainer.fit(reward_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    if best_state_callback.best_state_dict is not None:
        reward_model.load_state_dict(best_state_callback.best_state_dict)

    sanitized_exp_name = _sanitize_for_filename(exp_name if exp_name else process_env_name(env_id))
    model_filename = f"reward_model_{sanitized_exp_name}_cp{checkpoint_step}.ckpt"
    model_path = str(Path(save_dir) / model_filename)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    checkpoint_dict = {
        "state_dict": deepcopy(reward_model.state_dict()),
        "lr_schedulers": [],
        "epoch": int(getattr(trainer, "current_epoch", 0)),
        "global_step": int(getattr(trainer, "global_step", 0)),
        "pytorch-lightning_version": pl.__version__,
        "hyper_parameters": reward_model.hparams,
        "optimizer_states": [],
        "callbacks": {},
    }
    torch.save(checkpoint_dict, model_path)

    val_metric = trainer.callback_metrics.get("val_loss")
    if val_metric is not None:
        print(
            f"[INFO] Trained reward model for checkpoint {checkpoint_step}. "
            f"Best val_loss={float(val_metric.detach().cpu().item()):.6f}. Saved to: {model_path}"
        )
    else:
        print(f"[INFO] Trained reward model for checkpoint {checkpoint_step}. Saved to: {model_path}")

    return model_path


async def init_db():
    # Make sure all database tables exist
    await db_handler.create_table_from_model(database, Project)
    await db_handler.create_table_from_model(database, Experiment)
    await db_handler.create_table_from_model(database, Environment)


def get_custom_thumbnail_creator(env_id: str):
    try:
        if "BabyAI" in env_id:
            from rlhfblender.utils.babyai_utils import trajectory_plotter as tp

            return tp.generate_thumbnail
    except Exception:
        return None

    return None


async def add_to_project(project: str = "RLHF-Blender", env: str | None = None, exp: str | None = None):
    """Add an environment or experiment to a project."""
    # Check if project exists
    if not await db_handler.check_if_exists(database, Project, key=project, key_column="project_name"):
        # Register new project
        await db_handler.add_entry(
            database,
            Project,
            Project(project_name=project, created_timestamp=int(time.time())).model_dump(),
        )
        existing_envs = []
        existing_exps = []
    else:
        # Get the project and existing envs and exps
        project_obj: Project = await db_handler.get_single_entry(database, Project, key=project, key_column="project_name")
        existing_envs = project_obj.project_environments
        existing_exps = project_obj.project_experiments

    # Now add env or exp to project
    if env is not None and env not in existing_envs:
        await db_handler.update_entry(
            database,
            Project,
            key=project,
            key_column="project_name",
            data={"project_environments": [*existing_envs, env]},
        )
    if exp is not None and exp not in existing_exps:
        await db_handler.update_entry(
            database,
            Project,
            key=project,
            key_column="project_name",
            data={"project_experiments": [*existing_exps, exp]},
        )


async def register_env(
    env_id: str,
    entry_point: str | None = "",
    display_name: str = "",
    additional_gym_packages: list[str] | None = None,
    env_kwargs: dict | None = None,
    action_names: list[str] | None = None,
    env_description: str = "",
    project: str = "RLHF-Blender",
):
    """Register an environment in the database."""
    env_name = display_name if display_name != "" else env_id
    env_kwargs = env_kwargs if env_kwargs is not None else {}
    additional_gym_packages = additional_gym_packages if additional_gym_packages is not None else []

    # Check if environment is already registered
    if not await db_handler.check_if_exists(database, Environment, key=env_id, key_column="registration_id"):
        # Register the environment
        env: Environment = environment_handler.initial_registration(
            env_id=env_id,
            entry_point=entry_point,
            additional_gym_packages=additional_gym_packages,
            gym_env_kwargs=env_kwargs,
            action_names=action_names,
        )

        env.env_name = env_name
        env.description = env_description

        await db_handler.add_entry(database, Environment, env.model_dump())
        await add_to_project(project=project, env=env_id)
        print(f"Registered environment {env_name} in project {project}")
    else:
        print(f"Environment with id {env_id} already exists. Skipping registration.")


async def register_experiment(
    exp_name: str,
    env_id: str,
    algorithm: str | None = None,
    env_kwargs: dict | None = None,
    path: str | None = "",
    framework: str = "StableBaselines3",
    exp_kwargs: dict | None = None,
    project: str | None = "RLHF-Blender",
):
    """Register an experiment in the database."""
    env_kwargs = env_kwargs if env_kwargs is not None else {}
    exp_kwargs = exp_kwargs if exp_kwargs is not None else {}

    # Check if experiment is already registered
    if not await db_handler.check_if_exists(database, Experiment, key=exp_name, key_column="exp_name"):
        exp = Experiment(
            exp_name=exp_name,
            env_id=env_id,
            path=path,
            algorithm=algorithm.lower() if algorithm else "",
            environment_config={"env_kwargs": env_kwargs},
            framework=framework,
            **exp_kwargs,
        )
        await db_handler.add_entry(database, Experiment, exp.model_dump())
        await add_to_project(project=project, exp=exp_name)
        print(f"Registered experiment {exp_name} in project {project}")
    else:
        print(f"Experiment with name {exp_name} already exists. Skipping registration.")


async def run_benchmark(requests: list[dict]) -> list[str]:
    """Run benchmarks and generate data."""
    benchmarked_experiments = []
    latest_reward_models_by_exp: dict[str, str] = {}
    for benchmark_run in requests:
        # Check if experiment exists
        if benchmark_run["exp"] != "" and await db_handler.check_if_exists(
            database, Experiment, key=benchmark_run["exp"], key_column="exp_name"
        ):
            exp: Experiment = await db_handler.get_single_entry(
                database, Experiment, key=benchmark_run["exp"], key_column="exp_name"
            )
            database_env = await db_handler.get_single_entry(
                database, Environment, key=exp.env_id, key_column="registration_id"
            )
            benchmark_run["env"] = exp.env_id if "env" not in benchmark_run else benchmark_run["env"]
        else:
            # Register experiment and environment if necessary
            if not await db_handler.check_if_exists(
                database, Environment, key=benchmark_run["env"], key_column="registration_id"
            ):
                # We lazily register the environment if it is not registered yet, this is only done once
                await register_env(env_id=benchmark_run["env"])
            database_env = await db_handler.get_single_entry(
                database, Environment, key=benchmark_run["env"], key_column="registration_id"
            )
            # Create and register a "dummy" experiment
            exp_name = f"{benchmark_run['env']}_{benchmark_run['benchmark_type']}_experiment"
            await register_experiment(
                exp_name=exp_name,
                env_id=benchmark_run["env"],
                path=benchmark_run["path"],
                algorithm=benchmark_run.get("algorithm", None),
                framework=benchmark_run.get("framework", "random"),
                env_kwargs=benchmark_run.get("env_kwargs", {}),
            )
            exp: Experiment = await db_handler.get_single_entry(database, Experiment, key=exp_name, key_column="exp_name")

        # Add the current checkpoint to the experiment
        existing_checkpoints = exp.checkpoint_list if exp.checkpoint_list else []
        if benchmark_run["checkpoint_step"] not in existing_checkpoints:
            existing_checkpoints.append(benchmark_run["checkpoint_step"])
            # sort the checkpoints
            existing_checkpoints.sort(key=lambda x: int(x))
            exp.checkpoint_list = existing_checkpoints
            await db_handler.update_entry(
                database,
                Experiment,
                key=exp.id,
                data={"checkpoint_list": existing_checkpoints},
            )

        norm_env_path = _resolve_norm_env_path(model_path=benchmark_run["path"], env_id=exp.env_id)
        benchmark_env_config = _resolve_benchmark_environment_config(
            exp=exp,
            model_path=benchmark_run["path"],
            norm_env_path=norm_env_path,
        )

        benchmark_env = (
            environment_handler.get_environment(
                exp.env_id,
                environment_config=benchmark_env_config,
                n_envs=1,
                norm_env_path=norm_env_path,
                checkpoint_step=benchmark_run["checkpoint_step"],
                additional_packages=database_env.additional_gym_packages,
                gym_entry_point=database_env.gym_entry_point,
            )
            if "BabyAI" not in exp.env_id
            else gym.make(exp.env_id, render_mode="rgb_array")
        )

        framework = exp.framework
        if benchmark_run["benchmark_type"] == "random":
            framework = "Random"

        agent = framework_selector.get_agent(framework=framework)(
            observation_space=benchmark_env.observation_space,
            action_space=benchmark_env.action_space,
            exp=exp,
            env=benchmark_env,
            device="auto",
            checkpoint_step=benchmark_run["checkpoint_step"],
        )

        save_file_name = os.path.join(
            process_env_name(exp.env_id), f"{process_env_name(exp.env_id)}_{exp.id}_{benchmark_run['checkpoint_step']}"
        )

        # Create persistent initial state path if consistent start state is enabled
        persistent_state_path = None
        if benchmark_run.get("consistent_start_state", False):
            persistent_state_path = os.path.join(
                "data", "initial_states", f"{process_env_name(exp.env_id)}_{exp.id}_initial_state.pkl"
            )

        # Create an instance of EpisodeRecorder with the required parameters
        recorder = EpisodeRecorder(
            agent=agent,
            env=benchmark_env,
            n_eval_episodes=benchmark_run["n_episodes"],
            max_steps=benchmark_run.get("max_steps", int(2e4)),
            save_path=os.path.join("data", "saved_benchmarks", save_file_name),
            overwrite=True,
            render=True,
            deterministic=False,
            reset_to_initial_state=True,
            persistent_initial_state_path=persistent_state_path,
        )

        # Call the record_episodes method to start recording
        recorder.record_episodes()

        should_train_reward_model = (
            benchmark_run.get("train_reward_model", False)
            and benchmark_run["benchmark_type"] != "random"
            and int(benchmark_run.get("reward_model_trajectories", 0)) > 0
        )

        if should_train_reward_model:
            reward_exp_key = str(exp.exp_name)
            warm_start_path = latest_reward_models_by_exp.get(reward_exp_key)
            reward_sample_name = os.path.join(
                process_env_name(exp.env_id),
                f"{process_env_name(exp.env_id)}_{exp.id}_{benchmark_run['checkpoint_step']}_reward_samples_tmp",
            )
            reward_sample_path = os.path.join("data", "saved_benchmarks", reward_sample_name)

            reward_recorder = EpisodeRecorder(
                agent=agent,
                env=benchmark_env,
                n_eval_episodes=int(benchmark_run.get("reward_model_trajectories", 100)),
                max_steps=benchmark_run.get("max_steps", int(2e4)),
                save_path=reward_sample_path,
                overwrite=True,
                render=False,
                deterministic=False,
                reset_to_initial_state=False,
            )

            trained_model_path = None
            reward_npz_path = f"{reward_sample_path}.npz"
            reward_npz = None
            try:
                print(
                    f"[INFO] Sampling {benchmark_run.get('reward_model_trajectories', 100)} trajectories "
                    f"for supervised reward training at checkpoint {benchmark_run['checkpoint_step']}."
                )
                reward_recorder.record_episodes()
                reward_npz = np.load(reward_npz_path, allow_pickle=True)
                reward_episode_data = split_data(reward_npz)
                samples = _extract_step_reward_samples(
                    reward_episode_data,
                    max_trajectories=int(benchmark_run.get("reward_model_trajectories", 100)),
                    observation_space=benchmark_env.observation_space,
                    action_space=benchmark_env.action_space,
                )

                trained_model_path = _build_and_train_supervised_reward_model(
                    samples=samples,
                    observation_space=benchmark_env.observation_space,
                    action_space=benchmark_env.action_space,
                    env_id=exp.env_id,
                    exp_name=benchmark_run.get("exp", ""),
                    checkpoint_step=str(benchmark_run["checkpoint_step"]),
                    save_dir=str(benchmark_run.get("reward_model_save_dir", "multi-type-feedback/reward_models/checkpoints")),
                    warm_start_checkpoint=warm_start_path,
                    max_epochs=int(benchmark_run.get("reward_model_max_epochs", 100)),
                    patience=int(benchmark_run.get("reward_model_patience", 8)),
                    validation_split=float(benchmark_run.get("reward_model_val_split", 0.2)),
                    batch_size=int(benchmark_run.get("reward_model_batch_size", 64)),
                    learning_rate=float(benchmark_run.get("reward_model_learning_rate", 1e-5)),
                    ensemble_count=int(benchmark_run.get("reward_model_ensemble_count", 4)),
                    seed=int(benchmark_run.get("reward_model_seed", 42)),
                )
            except Exception as reward_model_error:
                print(
                    f"[WARN] Reward model training failed for checkpoint "
                    f"{benchmark_run['checkpoint_step']}: {reward_model_error}"
                )
            finally:
                if reward_npz is not None:
                    reward_npz.close()
                if os.path.isfile(reward_npz_path):
                    os.remove(reward_npz_path)

            if trained_model_path:
                latest_reward_models_by_exp[reward_exp_key] = trained_model_path

        # Register benchmarked experiment
        benchmarked_experiments.append(exp.id)

        if isinstance(benchmark_env, VecEnv):
            benchmark_env.close()
        elif hasattr(benchmark_env, "close"):
            benchmark_env.close()

    return benchmarked_experiments


def split_data(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Splits the data into episodes."""
    episode_ends = np.argwhere(data["dones"])
    episodes = {}
    for name, data_item in data.items():
        if data_item.shape:
            episodes[name] = np.split(data_item, episode_ends.flatten() + 1)
        else:
            episodes[name] = data_item
    return episodes


def encode_video(renders: np.ndarray, path: str) -> None:
    """Encodes renders into a .mp4 video and saves it at path."""
    # Create video in H264 format
    try:
        out = cv2.VideoWriter(
            f"{path}.mp4",
            cv2.VideoWriter_fourcc(*"avc1"),
            24,
            (renders.shape[2], renders.shape[1]),
        )
    except Exception:
        print("AVC1 codec not available, using MP4V codec instead.")
        try:
            out = cv2.VideoWriter(
                f"{path}.mp4",
                cv2.VideoWriter_fourcc(*"mp4v"),
                24,
                (renders.shape[2], renders.shape[1]),
            )
        except Exception as e:
            print(f"Error creating video writer: {e}")
    for render in renders:
        # Convert to BGR
        render = cv2.cvtColor(render, cv2.COLOR_RGB2BGR)
        out.write(render)
    out.release()


async def generate_data(benchmark_dicts: list[dict]):
    """Main async method to generate data."""
    requests = []
    skipped = 0
    for item in benchmark_dicts:
        benchmark_run = item
        save_file_name = os.path.join(
            f"{benchmark_run['env']}_{benchmark_run['benchmark_type']}_{benchmark_run['exp']}_{benchmark_run['checkpoint_step']}"
        )

        # Skip processing if data already exists
        episode_dir = Path(DATA_ROOT_DIR) / "episodes" / os.path.splitext(save_file_name)[0]
        env_states_dir = Path(DATA_ROOT_DIR) / "env_states" / os.path.splitext(save_file_name)[0]
        has_episode_dir = episode_dir.is_dir()
        has_env_states = env_states_dir.is_dir() and any(env_states_dir.glob("env_states_*.npy"))

        if has_episode_dir and has_env_states:
            skipped += 1
            continue
        if has_episode_dir and not has_env_states:
            print(
                "[INFO] Existing episode data found without env_states. "
                f"Regenerating benchmark for {save_file_name}."
            )

        # Otherwise, run the benchmark
        requests.append(benchmark_run)

    print(
        f"Skipped pre-processing for {skipped} benchmarks because data already exists. Remove data to trigger re-processing."
    )
    if len(requests) > 0:
        print(f"Running processing for {len(requests)} benchmarks.")

    benchmarked_experiments = await run_benchmark(requests)

    if len(benchmarked_experiments) > 0:
        print(f"Processing benchmark data for {len(benchmarked_experiments)} experiments.")
        await _process_benchmark_data(requests, benchmarked_experiments)


async def _process_benchmark_data(requests: list[dict], benchmarked_experiments: list[str]):
    """Process the benchmark data and create videos, thumbnails, and rewards."""

    # Now create the video/thumbnail/reward data etc.
    for benchmark_run, exp_id in zip(requests, benchmarked_experiments, strict=False):
        # Path to benchmark file
        save_file_name = os.path.join(
            process_env_name(benchmark_run["env"]),
            f"{process_env_name(benchmark_run['env'])}_{exp_id}_{benchmark_run['checkpoint_step']}.npz",
        )
        data = np.load(f"{DATA_ROOT_DIR}/{BENCHMARK_DIR}/{save_file_name}", allow_pickle=True)
        episode_data = split_data(data)

        for episode_idx, _ in enumerate(episode_data["dones"]):
            dir_name = f"data/episodes/{os.path.splitext(save_file_name)[0]}"
            save_episode = {}
            for name, _ in episode_data.items():
                if name == "additional_metrics" or name == "renders" or name == "env_states":
                    continue
                save_episode[name] = episode_data[name][episode_idx]
            os.makedirs(dir_name, exist_ok=True)
            np.savez(f"{dir_name}/benchmark_{episode_idx}.npz", **save_episode)
            os.makedirs(
                f"data/rewards/{os.path.splitext(save_file_name)[0]}",
                exist_ok=True,
            )
            np.save(
                f"data/rewards/{os.path.splitext(save_file_name)[0]}/rewards_{episode_idx}.npy",
                np.array([rew for rew in episode_data["rewards"][episode_idx]]),
            )

            # Save env_states if they exist
            if "env_states" in episode_data and episode_data["env_states"] is not None:
                os.makedirs(
                    f"data/env_states/{os.path.splitext(save_file_name)[0]}",
                    exist_ok=True,
                )
                np.save(
                    f"data/env_states/{os.path.splitext(save_file_name)[0]}/env_states_{episode_idx}.npy",
                    episode_data["env_states"][episode_idx],
                )

            if "uncertainty" in episode_data:
                os.makedirs(
                    f"data/uncertainty/{os.path.splitext(save_file_name)[0]}",
                    exist_ok=True,
                )
                np.save(
                    f"data/uncertainty/{os.path.splitext(save_file_name)[0]}/uncertainty_{episode_idx}.npy",
                    np.array([unc for unc in episode_data["uncertainty"][episode_idx]]),
                )
            elif "infos" in episode_data:
                os.makedirs(
                    f"data/uncertainty/{os.path.splitext(save_file_name)[0]}",
                    exist_ok=True,
                )
                np.save(
                    f"data/uncertainty/{os.path.splitext(save_file_name)[0]}/uncertainty_{episode_idx}.npy",
                    np.array([info.item()["entropy"] for info in episode_data["infos"][episode_idx]]),
                )

        # Create video
        for episode_idx, renders in enumerate(episode_data["renders"]):
            dir_name = f"data/renders/{os.path.splitext(save_file_name)[0]}"
            if not os.path.isdir(dir_name):
                os.makedirs(dir_name)
            encode_video(renders, f"{dir_name}/{episode_idx}")

            dir_name = f"data/thumbnails/{os.path.splitext(save_file_name)[0]}"
            if not os.path.isdir(dir_name):
                os.makedirs(dir_name)

            # Check if custom thumbnail creator exists
            custom_thumbnail_creator = get_custom_thumbnail_creator(benchmark_run["env"])
            if custom_thumbnail_creator is not None:
                # Create custom thumbnail
                save_image = custom_thumbnail_creator(
                    benchmark_run["env"],
                    episode_data["infos"][episode_idx][0].get("seed", None),
                    episode_data["actions"][episode_idx],
                )
            else:
                if renders is not None and len(renders.shape) == 4 and renders.shape[0] > 1:
                    save_image = renders[-2]  # Use the second-to-last frame of the episode
                else:
                    save_image = np.zeros((128, 128, 3), dtype=np.uint8)  # Placeholder image
                # Save first frame of the episode
            save_image = cv2.cvtColor(save_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{dir_name}/{episode_idx}.jpg", save_image)

        # Delete original save file
        os.remove(f"data/{BENCHMARK_DIR}/{save_file_name}")

        # Delete the last episode if incomplete
        episode_idx = len(episode_data["dones"]) - 1
        episode_dir = f"data/episodes/{os.path.splitext(save_file_name)[0]}"
        rewards_dir = f"data/rewards/{os.path.splitext(save_file_name)[0]}"
        uncertainty_dir = f"data/uncertainty/{os.path.splitext(save_file_name)[0]}"
        renders_dir = f"data/renders/{os.path.splitext(save_file_name)[0]}"
        thumbnails_dir = f"data/thumbnails/{os.path.splitext(save_file_name)[0]}"
        env_states_dir = f"data/env_states/{os.path.splitext(save_file_name)[0]}"

        episode_file = f"{episode_dir}/benchmark_{episode_idx}.npz"
        rewards_file = f"{rewards_dir}/rewards_{episode_idx}.npy"
        uncertainty_file = f"{uncertainty_dir}/uncertainty_{episode_idx}.npy"
        renders_file = f"{renders_dir}/{episode_idx}.mp4"
        thumbnails_file = f"{thumbnails_dir}/{episode_idx}.jpg"
        env_states_file = f"{env_states_dir}/env_states_{episode_idx}.npy"

        print(f"Deleting last episode {episode_idx} if incomplete")
        os.remove(episode_file)
        os.remove(rewards_file)
        if "infos" in episode_data:
            os.remove(uncertainty_file)
        os.remove(renders_file)
        os.remove(thumbnails_file)
        if os.path.isfile(env_states_file):
            os.remove(env_states_file)
