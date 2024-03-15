"""
Make sure that the data for the demo is available, run at application startup.
Expects pre-trained models in the experimentation directory. First runs benchmarks with the provided
models, then creates video/thumbnail/reward data etc.
"""

import os
import time
from types import SimpleNamespace as sn
from typing import Dict, List

import cv2
import gymnasium as gym
import numpy as np
from databases import Database
from pydantic import BaseModel

from rlhfblender.data_collection import framework_selector as framework_selector
from rlhfblender.data_collection.environment_handler import get_environment, initial_registration
from rlhfblender.data_collection.episode_recorder import EpisodeRecorder
from rlhfblender.data_handling import database_handler as db_handler
from rlhfblender.data_models import Environment, Experiment

DATA_ROOT_DIR = "data"
BENCHMARK_DIR = "saved_benchmarks"

database = Database(os.environ.get("RLHFBLENDER_DB_HOST", "sqlite:///rlhfblender.db"))


def get_custom_thumbnail_creator(env_id: str):
    try:
        if "BabyAI" in env_id:
            from rlhfblender.utils.babyai_utils import trajectory_plotter as tp

            return tp.generate_thumbnail
    except Exception:
        return None

    return None


class BenchmarkRequestModel(BaseModel):
    """
    A request model for a single benchmark run
    """

    env_id: str = ""
    path: str = ""
    benchmark_type: str = "random"
    benchmark_id: str = ""
    checkpoint_step: int = -1
    n_episodes: int = 1
    force_overwrite: bool = False
    render: bool = True
    deterministic: bool = False
    reset_state: bool = False
    split_by_episode: bool = False


# Run benchmarks with the provided models
async def run_benchmark(request: List[BenchmarkRequestModel]) -> list[Experiment]:
    """
    Run an agent in the provided environment with the given parameters. The experiment id only has to provided
    if benchmark_type trained is used.
    :param request (
    :return:
    """
    benchmarked_experiments = []
    for benchmark_run in request:
        print(request)
        if benchmark_run.benchmark_id != "":
            exp: Experiment = await db_handler.get_single_entry(
                database, Experiment, key=benchmark_run.benchmark_id, key_column="exp_name"
            )
        else:
            # for the experiments, we need to register the environment first (e.g. for annotations, naming of the action space)
            if not db_handler.check_if_exists(database, Environment, value=benchmark_run.env_id, column="registration_id"):
                # We lazily register the environment if it is not registered yet, this is only done once
                database_env = initial_registration(
                    benchmark_run.env_id,
                    additional_gym_packages=(
                        benchmark_run.additional_packages if "additional_packages" in benchmark_run else []
                    ),
                )
                await db_handler.add_entry(database, Environment, database_env.model_dump())

            # create and register a "dummy" experiment
            exp: Experiment = Experiment(
                exp_name=f"{benchmark_run.env_id}_{benchmark_run.framwork}_{benchmark_run.benchmark_type}_Experiment",
                env_id=benchmark_run.env_id,
                framework=benchmark_run.framework,
                created_timestamp=int(time.time()),
            )
            await db_handler.add_entry(database, Experiment, exp)

        # add the current checkpoint to the experiment
        existing_checkpoints = exp.checkpoint_list if "checkpoint_list" in exp else []
        if benchmark_run.checkpoint_step not in existing_checkpoints:
            existing_checkpoints.append(benchmark_run.checkpoint_step)
            exp.checkpoint_list = existing_checkpoints
            await db_handler.update_entry(
                database,
                Experiment,
                key=exp.id,
                data={"checkpoint_list": existing_checkpoints},
            )

        benchmark_env = (
            get_environment(
                benchmark_run.env_id,
                environment_config=exp.environment_config,
                n_envs=1,
                norm_env_path=os.path.join(benchmark_run.path, benchmark_run.env_id),
                # this is how SB-Zoo does it, so we stick to it for easy cross-compatabily
                additional_packages=benchmark_run.additional_packages if "additional_packages" in benchmark_run else [],
            )
            if "BabyAI" not in benchmark_run.env_id
            else gym.make(benchmark_run.env_id, render_mode="rgb_array")
        )
        framework = exp.framework
        print(exp, framework)
        if benchmark_run.benchmark_type == "random":
            framework = "Random"
        agent = framework_selector.get_agent(framework=framework)(
            observation_space=benchmark_env.observation_space,
            action_space=benchmark_env.action_space,
            exp=sn(
                **{
                    "algorithm": "ppo",
                    "env_id": benchmark_run.env_id,
                    "path": benchmark_run.path,
                }
            ),
            env=benchmark_env,
            device="auto",
            checkpoint_step=benchmark_run.checkpoint_step,
        )

        save_file_name = f"{benchmark_run.env_id}_{exp.id}_{benchmark_run.checkpoint_step}"

        EpisodeRecorder.record_episodes(
            agent,
            benchmark_env,
            benchmark_run.n_episodes,
            max_steps=int(2e4),
            save_path=os.path.join("data", "saved_benchmarks", save_file_name),
            overwrite=True,
            render=benchmark_run.render,
            deterministic=benchmark_run.deterministic,
            reset_to_initial_state=benchmark_run.reset_state,
        )
        benchmarked_experiments.append(exp.id)

    return benchmarked_experiments


# Now, create the video/thumbnail/reward data etc.
def split_data(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Splits the data into episodes."""
    episode_ends = np.argwhere(data["dones"]).squeeze()
    episodes = {}
    for name, data_item in data.items():
        if data_item.shape:
            episodes[name] = np.split(data_item, episode_ends)
        else:
            episodes[name] = data_item

    return episodes


def encode_video(renders: np.ndarray, path: str) -> None:
    """
    Encodes renders of shape [n_frames, height, width, 3] into a .mp4 video and
    saves it at path.
    """
    # Create video in H264 format
    out = cv2.VideoWriter(
        f"{path}.mp4",
        cv2.VideoWriter_fourcc(*"avc1"),
        24,
        (renders.shape[2], renders.shape[1]),
    )
    for render in renders:
        # Convert to BGR
        render = cv2.cvtColor(render, cv2.COLOR_RGB2BGR)
        out.write(render)
    out.release()


async def generate_data(benchmark_dicts: List[Dict]):
    """
    Main async method, do all your calls that need to be awaited here.
    :param benchmark_dicts:
    :return:
    """
    requests = []

    skipped = 0
    for item in benchmark_dicts:
        benchmark_run = BenchmarkRequestModel(**item)
        save_file_name = os.path.join(
            f"{benchmark_run.env_id}_{benchmark_run.benchmark_type}_{benchmark_run.benchmark_id}_{benchmark_run.checkpoint_step}"
        )

        # Skip processing
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
        save_file_name = os.path.join(f"{benchmark_run.env_id}_{exp_id}_{benchmark_run.checkpoint_step}.npz")
        data = np.load(f"{DATA_ROOT_DIR}/{BENCHMARK_DIR}/{save_file_name}", allow_pickle=True)
        episode_data = split_data(data)

        for episode_idx, _ in enumerate(episode_data["dones"]):
            dir_name = f"{DATA_ROOT_DIR}/episodes/{os.path.splitext(save_file_name)[0]}"
            save_episode = {}
            for name, _ in episode_data.items():
                if name == "additional_metrics" or name == "renders":
                    continue
                save_episode[name] = episode_data[name][episode_idx]
            os.makedirs(dir_name, exist_ok=True)
            np.savez(f"{dir_name}/benchmark_{episode_idx}.npz", **save_episode)
            os.makedirs(
                f"{DATA_ROOT_DIR}/rewards/{os.path.splitext(save_file_name)[0]}",
                exist_ok=True,
            )
            np.save(
                f"{DATA_ROOT_DIR}/rewards/{os.path.splitext(save_file_name)[0]}/rewards_{episode_idx}.npy",
                np.cumsum(episode_data["rewards"][episode_idx]),
            )

            # Save uncertainty data if available (for now, just use entropy from info dict)
            if "infos" in episode_data:
                os.makedirs(
                    f"{DATA_ROOT_DIR}/uncertainty/{os.path.splitext(save_file_name)[0]}",
                    exist_ok=True,
                )
                np.save(
                    f"{DATA_ROOT_DIR}/uncertainty/{os.path.splitext(save_file_name)[0]}/uncertainty_{episode_idx}.npy",
                    np.array([info["entropy"] for info in episode_data["infos"][episode_idx]]),
                )

        # Create video
        for episode_idx, renders in enumerate(episode_data["renders"]):
            dir_name = f"{DATA_ROOT_DIR}/renders/{os.path.splitext(save_file_name)[0]}"
            if not os.path.isdir(dir_name):
                os.makedirs(dir_name)
            encode_video(renders, f"{dir_name}/{episode_idx}")

            dir_name = f"{DATA_ROOT_DIR}/thumbnails/{os.path.splitext(save_file_name)[0]}"
            if not os.path.isdir(dir_name):
                os.makedirs(dir_name)

            # Check if custom thumbnail creator exists
            custom_thumbnail_creator = get_custom_thumbnail_creator(benchmark_run.env_id)
            if custom_thumbnail_creator is not None:
                # Create custom thumbnail with env id and seed (info the info dict)
                save_image = custom_thumbnail_creator(
                    benchmark_run.env_id,
                    episode_data["infos"][episode_idx][0].get("seed", None),
                    episode_data["actions"][episode_idx],
                )
            else:
                save_image = renders[0]
                # Save first frame of the episode, first convert to BGR
            save_image = cv2.cvtColor(save_image, cv2.COLOR_RGB2BGR)

            cv2.imwrite(f"{dir_name}/{episode_idx}.jpg", save_image)

        # delete original save_file, not needed anymore
        os.remove(f"{DATA_ROOT_DIR}/{BENCHMARK_DIR}/{save_file_name}")

        # Delete the last episode, as it is not complete (TODO: Fix this)
        episode_idx = len(episode_data["dones"]) - 1
        episode_dir = f"{DATA_ROOT_DIR}/episodes/{os.path.splitext(save_file_name)[0]}"
        rewards_dir = f"{DATA_ROOT_DIR}/rewards/{os.path.splitext(save_file_name)[0]}"
        uncertainty_dir = f"{DATA_ROOT_DIR}/uncertainty/{os.path.splitext(save_file_name)[0]}"
        renders_dir = f"{DATA_ROOT_DIR}/renders/{os.path.splitext(save_file_name)[0]}"
        thumbnails_dir = f"{DATA_ROOT_DIR}/thumbnails/{os.path.splitext(save_file_name)[0]}"

        episode_file = f"{episode_dir}/benchmark_{episode_idx}.npz"
        rewards_file = f"{rewards_dir}/rewards_{episode_idx}.npy"
        uncertainty_file = f"{uncertainty_dir}/uncertainty_{episode_idx}.npy"
        renders_file = f"{renders_dir}/{episode_idx}.mp4"
        thumbnails_file = f"{thumbnails_dir}/{episode_idx}.jpg"

        print(f"Deleting last episode of {episode_idx}")
        os.remove(episode_file)
        os.remove(rewards_file)
        if "infos" in episode_data:
            os.remove(uncertainty_file)
        os.remove(renders_file)
        os.remove(thumbnails_file)
