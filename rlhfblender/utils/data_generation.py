import os
import time
from typing import Dict, List, Optional

import cv2
import gymnasium as gym
import numpy as np
from databases import Database

import rlhfblender.data_collection.environment_handler as environment_handler
import rlhfblender.data_collection.framework_selector as framework_selector
import rlhfblender.data_handling.database_handler as db_handler
from rlhfblender.data_collection.episode_recorder import EpisodeRecorder
from rlhfblender.data_models.global_models import Environment, Experiment, Project
from rlhfblender.utils import process_env_name

# Initialize database
database = Database(os.environ.get("RLHFBLENDER_DB_HOST", "sqlite:///rlhfblender.db"))


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


async def add_to_project(project: str = "RLHF-Blender", env: Optional[str] = None, exp: Optional[str] = None):
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
    entry_point: Optional[str] = "",
    display_name: str = "",
    additional_gym_packages: Optional[List[str]] = None,
    env_kwargs: Optional[Dict] = None,
    env_description: str = "",
    project: str = "RLHF-Blender",
):
    """Register an environment in the database."""
    env_name = display_name if display_name != "" else env_id
    env_kwargs = env_kwargs if env_kwargs is not None else {}
    print(env_kwargs)
    additional_gym_packages = additional_gym_packages if additional_gym_packages is not None else []

    # Check if environment is already registered
    if not await db_handler.check_if_exists(database, Environment, key=env_id, key_column="registration_id"):
        # Register the environment
        env: Environment = environment_handler.initial_registration(
            env_id=env_id,
            entry_point=entry_point,
            additional_gym_packages=additional_gym_packages,
            gym_env_kwargs=env_kwargs,
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
    env_kwargs: Optional[Dict] = None,
    path: Optional[str] = "",
    framework: str = "StableBaselines3",
    exp_kwargs: Optional[Dict] = None,
    project: Optional[str] = "RLHF-Blender",
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
            environment_config=env_kwargs,
            framework=framework,
            **exp_kwargs,
        )
        await db_handler.add_entry(database, Experiment, exp.model_dump())
        await add_to_project(project=project, exp=exp_name)
        print(f"Registered experiment {exp_name} in project {project}")
    else:
        print(f"Experiment with name {exp_name} already exists. Skipping registration.")


async def run_benchmark(requests: List[Dict]) -> List[str]:
    """Run benchmarks and generate data."""
    benchmarked_experiments = []
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
                framework=benchmark_run.get("framework", "random"),
                env_kwargs=benchmark_run.get("env_kwargs", {}),
            )
            exp: Experiment = await db_handler.get_single_entry(database, Experiment, key=exp_name, key_column="exp_name")

        # Add the current checkpoint to the experiment
        existing_checkpoints = exp.checkpoint_list if exp.checkpoint_list else []
        if benchmark_run["checkpoint_step"] not in existing_checkpoints:
            existing_checkpoints.append(benchmark_run["checkpoint_step"])
            exp.checkpoint_list = existing_checkpoints
            await db_handler.update_entry(
                database,
                Experiment,
                key=exp.id,
                data={"checkpoint_list": existing_checkpoints},
            )

        benchmark_env = (
            environment_handler.get_environment(
                exp.env_id,
                environment_config=exp.environment_config,
                n_envs=1,
                norm_env_path=os.path.join(benchmark_run["path"], process_env_name(exp.env_id)),
                additional_packages=database_env.additional_gym_packages,
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

        # Create an instance of EpisodeRecorder with the required parameters
        recorder = EpisodeRecorder(
            agent=agent,
            env=benchmark_env,
            n_eval_episodes=benchmark_run["n_episodes"],
            max_steps=int(2e4),
            save_path=os.path.join("data", "saved_benchmarks", save_file_name),
            overwrite=True,
            render=True,
            deterministic=False,
            reset_to_initial_state=False,
        )

        # Call the record_episodes method to start recording
        recorder.record_episodes()

        # Register benchmarked experiment
        benchmarked_experiments.append(exp.id)

    return benchmarked_experiments


def split_data(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
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
    except Exception as e:
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


async def generate_data(benchmark_dicts: List[Dict]):
    """Main async method to generate data."""
    DATA_ROOT_DIR = "data"
    BENCHMARK_DIR = "saved_benchmarks"

    requests = []
    skipped = 0
    for item in benchmark_dicts:
        benchmark_run = item
        save_file_name = os.path.join(
            f"{benchmark_run['env']}_{benchmark_run['benchmark_type']}_{benchmark_run['exp']}_{benchmark_run['checkpoint_step']}"
        )

        # Skip processing if data already exists
        if os.path.isdir(f"{DATA_ROOT_DIR}/episodes/{os.path.splitext(save_file_name)[0]}"):
            skipped += 1
            continue

        # Otherwise, run the benchmark
        requests.append(benchmark_run)

    print(
        f"Skipped pre-processing for {skipped} benchmarks because data already exists. Remove data to trigger re-processing."
    )
    if len(requests) > 0:
        print(f"Running processing for {len(requests)} benchmarks.")

    benchmarked_experiments = await run_benchmark(requests)

    # Now create the video/thumbnail/reward data etc.
    for benchmark_run, exp_id in zip(requests, benchmarked_experiments):
        # Path to benchmark file
        save_file_name = os.path.join(
            process_env_name(benchmark_run["env"]),
            f"{process_env_name(benchmark_run['env'])}_{exp_id}_{benchmark_run['checkpoint_step']}.npz",
        )
        data = np.load(f"data/{BENCHMARK_DIR}/{save_file_name}", allow_pickle=True)
        episode_data = split_data(data)

        for episode_idx, _ in enumerate(episode_data["dones"]):
            dir_name = f"data/episodes/{os.path.splitext(save_file_name)[0]}"
            save_episode = {}
            for name, _ in episode_data.items():
                if name == "additional_metrics" or name == "renders":
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
                np.cumsum(episode_data["rewards"][episode_idx]),
            )

            # Save uncertainty data if available
            if "infos" in episode_data:
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
                if renders is not None and len(renders.shape) == 4 and renders.shape[0] > 0:
                    save_image = renders[0]
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

        episode_file = f"{episode_dir}/benchmark_{episode_idx}.npz"
        rewards_file = f"{rewards_dir}/rewards_{episode_idx}.npy"
        uncertainty_file = f"{uncertainty_dir}/uncertainty_{episode_idx}.npy"
        renders_file = f"{renders_dir}/{episode_idx}.mp4"
        thumbnails_file = f"{thumbnails_dir}/{episode_idx}.jpg"

        print(f"Deleting last episode {episode_idx} if incomplete")
        os.remove(episode_file)
        os.remove(rewards_file)
        if "infos" in episode_data:
            os.remove(uncertainty_file)
        os.remove(renders_file)
        os.remove(thumbnails_file)
