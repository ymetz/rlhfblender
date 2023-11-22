import os
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List

import cv2
import numpy as np
from databases import Database
from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from rlhfblender.config import DB_HOST
from rlhfblender.data_collection import framework_selector
from rlhfblender.data_collection.demo_session import (
    check_socket_connection,
    close_demo_session,
    create_new_session,
    demo_perform_step,
)
from rlhfblender.data_collection.environment_handler import (
    get_environment,
    initial_registration,
)
from rlhfblender.data_collection.episode_recorder import (
    BenchmarkSummary,
    EpisodeRecorder,
    convert_infos,
)
from rlhfblender.data_handling import database_handler as db_handler
from rlhfblender.data_models.agent import RandomAgent
from rlhfblender.data_models.feedback_models import UnprocessedFeedback
from rlhfblender.data_models.global_models import (
    AggregatedRecordedEpisodes,
    Dataset,
    Environment,
    EpisodeID,
    Experiment,
    RecordedEpisodes,
)

database = Database(DB_HOST)

router = APIRouter(prefix="/data")


@router.get("/get_available_frameworks", response_model=List[str])
def get_available_frameworks():
    """
    Return a list of all available frameworks
    :return:
    """
    return framework_selector.get_framework_list()


@router.get("/get_algorithms", response_model=List[str])
def get_algorithms(selected_framework: str):
    """
    Return a list of all available algorithms for the given framework
    :param selected_framework:
    :return:
    """
    return framework_selector.get_connector(selected_framework).get_algorithms()


class BenchmarkModel(BaseModel):
    """
    A request model for a single benchmark run
    """

    env_id: int = 0
    benchmark_type: str = "random"
    benchmark_id: int = -1
    checkpoint_step: int = -1


class BenchmarkRequestModel(BenchmarkModel):
    n_episodes: int = 1
    force_overwrite: bool = False
    render: bool = True
    deterministic: bool = True
    reset_state: bool = False
    gym_registration_id: str = ""
    split_by_episode: bool = False
    record_episode_videos: bool = False


class VideoRequestModel(BenchmarkModel):
    episode_id: int = -1


class BenchmarkResponseModel(BaseModel):
    n_steps: int
    models: List[BenchmarkSummary]
    env_space_info: dict


# Feedback Type Enum
class FeedbackType(str, Enum):
    rating = "rating"
    ranking = "ranking"
    demonstration = "demonstration"
    correction = "correction"
    text = "text"


class FeedbackModel(BenchmarkModel):
    episode_id: int = -1
    feedback: dict = {}
    feedback_type: FeedbackType = FeedbackType.rating
    feedback_time: float = -1.0


@router.post("/run_benchmark", response_model=dict, tags=["DATA"])
async def run_benchmark(request: List[BenchmarkRequestModel]) -> BenchmarkResponseModel:
    """
    Run an agent in the provided environment with the given parameters. The experiment id only has to provided
    if benchmark_type trained is used.
    :param request (
    :return:
    """
    env_ids = []
    for env_id, gym_registration_id in set([(r.env_id, r.gym_registration_id) for r in request]):
        if gym_registration_id != "" and int(env_id) < 0:
            # We lazily register the environment if it is not registered yet, this is only done once
            env_id = await initial_registration(database, gym_registration_id)
            env_ids.append(env_id)
        else:
            env_ids.append(-1)  # The case for datasets

    benchmarked_models = []
    for i, benchmark_run in enumerate(request):
        if benchmark_run.benchmark_type == "dataset":
            # Dummy env id for datasets
            db_env = Environment(id=-1, name="dataset", gym_registration_id="dataset")
            db_dataset = await db_handler.get_single_entry(database, Dataset, id=benchmark_run.benchmark_id)
            benchmarked_models.append(EpisodeRecorder.get_aggregated_data(os.path.join("datasets", db_dataset.dataset_path)))
            continue
        db_env: Environment = await db_handler.get_single_entry(
            database,
            Environment,
            id=int(benchmark_run.env_id) if int(benchmark_run.env_id) >= 0 else env_ids[0],
        )
        registration_id = db_env.registration_id

        save_file_name = (
            f"{registration_id}_{benchmark_run.benchmark_type}_{benchmark_run.benchmark_id}_{benchmark_run.checkpoint_step}"
        )

        # If there is a cached benchmark, use it instead. Force_overwrites leads to the existing saved data to be
        # rewritten
        if benchmark_run.force_overwrite or not os.path.isfile(
            os.path.join("data", "saved_benchmarks", save_file_name + ".npz")
        ):
            if benchmark_run.benchmark_type == "trained":
                exp: Experiment = await db_handler.get_single_entry(database, Experiment, id=benchmark_run.benchmark_id)
                benchmark_env = get_environment(
                    registration_id,
                    environment_config=exp.environment_config,
                    n_envs=1,
                    norm_env_path=os.path.join(exp.path, db_env.registration_id),
                    # this is how SB-Zoo does it, so we stick to it for easy cross-compatibility
                    additional_packages=db_env.additional_gym_packages,
                )
                agent = framework_selector.get_agent(framework=exp.framework)(
                    observation_space=benchmark_env.observation_space,
                    action_space=benchmark_env.action_space,
                    exp=exp,
                    device="auto",
                    checkpoint_step=benchmark_run.checkpoint_step,
                )
            else:
                benchmark_env = get_environment(
                    registration_id,
                    n_envs=1,
                    additional_packages=db_env.additional_gym_packages,
                )
                agent = RandomAgent(benchmark_env.observation_space, benchmark_env.action_space)

            benchmarked_models.append(
                EpisodeRecorder.record_episodes(
                    agent,
                    benchmark_env,
                    benchmark_run.n_episodes,
                    max_steps=int(2e4),
                    # Record a maximum of 20k steps
                    save_path=os.path.join("data", "saved_benchmarks", save_file_name),
                    overwrite=True,
                    render=benchmark_run.render,
                    deterministic=benchmark_run.deterministic,
                    reset_to_initial_state=benchmark_run.reset_state,
                )
            )

        else:
            benchmarked_models.append(
                EpisodeRecorder.get_aggregated_data(os.path.join("data", "saved_benchmarks", save_file_name))
            )

    # Only return the last checkpoint (is the only one e.g. for a single benchmark run)
    # return data mainly for space infos, the frontend should query other results by itself via get_step_benchmark_data
    return BenchmarkResponseModel(
        n_steps=sum([bm.benchmark_steps for bm in benchmarked_models]),
        models=benchmarked_models,
        env_space_info={
            "obs_space": db_env.observation_space_info,
            "action_space": db_env.action_space_info,
        },
    )


@router.get("/get_rewards", response_model=List, tags=["DATA"])
async def get_rewards(
    env_name: str,
    benchmark_type: str,
    benchmark_id: int,
    checkpoint_step: int,
    episode_num: int,
):
    """Return step rewards a list for the selected episode"""
    # Replace with your rewards file path
    rewards = np.load(
        os.path.join(
            "data",
            "rewards",
            f"{env_name}_{benchmark_type}_{benchmark_id}_{checkpoint_step}",
            f"rewards_{episode_num}.npy",
        ),
    )

    return rewards.tolist()


@router.get("/get_uncertainty", response_model=List, tags=["DATA"])
async def get_uncertainty(
    env_name: str,
    benchmark_type: str,
    benchmark_id: int,
    checkpoint_step: int,
    episode_num: int,
):
    """Return step rewards a list for the selected episode"""
    # Replace with your rewards file path
    uncertainty = np.load(
        os.path.join(
            "data",
            "uncertainty",
            f"{env_name}_{benchmark_type}_{benchmark_id}_{checkpoint_step}",
            f"uncertainty_{episode_num}.npy",
        ),
    )

    return uncertainty.tolist()


@router.get("/get_video", response_class=FileResponse)
async def get_video(
    env_name: str,
    benchmark_type: str,
    benchmark_id: int,
    checkpoint_step: int,
    episode_num: int,
):
    # Replace with your video file path
    return FileResponse(
        os.path.join(
            "data",
            "renders",
            f"{env_name}_{benchmark_type}_{benchmark_id}_{checkpoint_step}",
            f"{episode_num}.mp4",
        ),
        media_type="video/mp4",
    )


@router.get("/get_thumbnail", response_class=FileResponse)
async def get_thumbnail(
    env_name: str,
    benchmark_type: str,
    benchmark_id: int,
    checkpoint_step: int,
    episode_num: int,
):
    """
    Should return a telling thumbnail at some point, right now just fetches
    the first frame of an episode.
    """
    return FileResponse(
        os.path.join(
            "data",
            "thumbnails",
            f"{env_name}_{benchmark_type}_{benchmark_id}_{checkpoint_step}",
            f"{episode_num}.jpg",
        ),
        media_type="image/jpg",
    )


class DetailRequest(BaseModel):
    env_name: str
    benchmark_type: str
    benchmark_id: int
    checkpoint_step: int
    episode_num: int


class SingleStepDetailRequest(DetailRequest):
    step: int


@router.post("/get_single_step_details", response_model=Dict[str, Any], tags=["DATA"])
async def get_single_step_details(request: SingleStepDetailRequest):
    """
    Returns a dictionary containing all the data for a single step: the action distribution, the action, the reward,
    the info dict, and action space.
    """
    action_space = {}
    db_env = next(
        filter(
            lambda env: env.env_name == request.env_name,
            await db_handler.get_all(database, Environment),
        )
    )
    if db_env is not None:
        action_space = db_env.action_space_info

    # Get the action distribution
    episode_benchmark_data = np.load(
        os.path.join(
            "data",
            "episodes",
            f"{request.env_name}_{request.benchmark_type}_{request.benchmark_id}_{request.checkpoint_step}",
            f"benchmark_{request.episode_num}.npz",
        ),
        allow_pickle=True,
    )

    action_distribution = episode_benchmark_data["probs"][request.step]
    action = episode_benchmark_data["actions"][request.step]
    reward = episode_benchmark_data["rewards"][request.step]
    info = episode_benchmark_data["infos"][request.step]

    return {
        "action_distribution": action_distribution.tolist(),
        "action": action.item(),
        "reward": reward.item(),
        "info": info,
        "action_space": action_space,
    }


@router.post("/get_actions_for_episode", response_model=List[int], tags=["DATA"])
async def get_actions_for_episode(request: DetailRequest):
    """
    Returns a list of all actions for a given episode
    """
    episode_benchmark_data = np.load(
        os.path.join(
            "data",
            "episodes",
            f"{request.env_name}_{request.benchmark_type}_{request.benchmark_id}_{request.checkpoint_step}",
            f"benchmark_{request.episode_num}.npz",
        ),
        allow_pickle=True,
    )

    return episode_benchmark_data["actions"].tolist()


@router.post("/save_feature_feedback")
async def save_feature_feedback(image: UploadFile = File(...)):
    import base64
    import io

    from PIL import Image

    print("Saving image...")
    contents = await image.read()

    contents_str = contents.decode("utf-8")

    # Remove the first part of the string "data:image/png;base64,"
    contents_str = contents_str.replace("data:image/png;base64,", "")

    img = Image.open(io.BytesIO(base64.b64decode(contents_str)))

    # Save image
    os.makedirs(os.path.join("data", "feature_feedback"), exist_ok=True)
    # get current time formatted as string
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    img.save(os.path.join("data", "feature_feedback", "feature_feedback_" + current_time + ".png"))

    return {"message": "Image saved successfully"}


@router.get("/episode_ids_chronologically", response_model=List[EpisodeID])
async def get_episode_ids_chronologically():
    """
    TODO: Find a better way to do this/reflect on why this is really necessary.
    This function defines the chronological order in which episodes are
    ordered. It defines the form of the identifier and the order.
    """

    # Right now, all data is generated with gen_data.py and then postprocessed
    # using deconstruct_data.ipynb. The IDs are thus built from this very
    # specific process, which will likely change in the future. Right now we
    # also assume that data/renders is full and contains all relevant episodes.
    episode_ids = []
    dir_names = os.listdir(os.path.join("data", "renders"))
    for dir_name in dir_names:
        if "_trained" not in dir_name and "_random" not in dir_name:
            continue
        split_file_name = dir_name.split("_")
        env_name = split_file_name[0]
        benchmark_type = split_file_name[1]
        benchmark_id = split_file_name[2]
        checkpoint_step = split_file_name[3]
        episode_files = os.listdir(os.path.join("data", "renders", dir_name))
        for episode_file in episode_files:
            num = episode_file.split(".")[0]
            episode_id = EpisodeID(
                env_name=env_name,
                benchmark_type=benchmark_type,
                benchmark_id=int(benchmark_id),
                checkpoint_step=int(checkpoint_step),
                episode_num=int(num),
            )
            episode_ids.append(episode_id)
    episode_ids = sorted(episode_ids, key=lambda x: (x.checkpoint_step, x.episode_num))
    return episode_ids


class ActionLabelRequest(BaseModel):
    envId: int


@router.post("/get_action_label_urls", response_model=List[str])
async def get_action_label_urls(request: ActionLabelRequest):
    """
    Returns a list of urls for the action labels of the given environment
    """
    db_env = await db_handler.get_single_entry(database, Environment, id=request.envId)
    if db_env is None:
        return []
    db_env_name = db_env.env_name

    # Check in data/action_labels/<env_name> for all files
    action_label_dir = os.path.join("data", "action_labels", db_env_name)
    if not os.path.isdir(action_label_dir):
        return []
    action_label_files = [file for file in os.listdir(action_label_dir) if file.endswith(".png") or file.endswith(".svg")]

    # Return the urls
    return [f"/action_labels/{db_env_name}/{file}" for file in action_label_files]


@router.post("/reset_sampler")
async def reset_sampler(request: Request):
    """
    Resets the sampler for the given experiment
    """
    experiment_id = request.query_params.get("experiment_id", None)
    sampling_strategy = request.query_params.get("sampling_strategy", None)
    if experiment_id is None:
        return "No experiment id given"
    experiment: Experiment = await db_handler.get_single_entry(database, Experiment, id=experiment_id)
    environment = await db_handler.get_single_entry(database, Environment, id=experiment.env_id)

    request.app.state.sampler.set_sampler(experiment, environment, sampling_strategy)

    return {
        "session_id": request.app.state.feedback_translator.set_translator(experiment, environment),
        "environment_id": experiment.env_id,
    }


@router.get("/get_all_episodes", response_model=List[EpisodeID])
async def get_all_episodes(request: Request):
    """
    Returns all episodes for the current configuration of the sampler
    :param request:
    :return:
    """
    return request.app.state.sampler.get_full_episode_list()


@router.get("/sample_episodes", response_model=List[EpisodeID])
async def sample_episodes(request: Request):
    """
    Samples episodes from the database
    """
    num_episodes = request.query_params.get("num_episodes", 1)
    num_episodes = int(num_episodes)
    return request.app.state.sampler.sample(batch_size=num_episodes)


@router.post("/give_feedback")
async def give_feedback(request: Request):
    """
    Provides feedback for a given episode
    """
    feedback = UnprocessedFeedback(**await request.json())

    print("UNPROCESSED FEEDBACK: ", feedback)

    request.app.state.feedback_translator.give_feedback(feedback.session_id, feedback)


@router.post("/submit_current_feedback")
async def submit_current_feedback(request: Request):
    """
    Submits the current feedback to the database
    """
    session_id = request.query_params.get("session_id", None)
    if session_id is None:
        return "No session id given"
    request.app.state.feedback_translator.submit(session_id)
    return "Feedback submitted"


@router.post("/initialize_demo_session")
async def initialize_demo_session(request: Request):
    """
    To generate demos, we initialize a gym environment in a separate process, then communicate with it via
    a socket. This function initializes the environment via a gym id, optional seed and returns the port
    :param request:
    :return:
    """
    request = await request.json()
    env_id = request["env_id"]
    seed = request["seed"]
    session_id = request["session_id"]

    action_space = {}
    db_env = await db_handler.get_single_entry(database, Environment, id=env_id)
    if db_env is not None:
        action_space = db_env.action_space_info

    try:
        pid, demo_number = await create_new_session(session_id, db_env.registration_id, int(seed))

        first_step = demo_perform_step(session_id, [])
        success = True
    except Exception as e:
        pid = -1
        first_step = {"reward": 0, "done": False, "infos": {}}
        success = False

    return {
        "pid": pid,
        "demo_number": demo_number,
        "action_space": action_space,
        "step": first_step,
        "success": success,
    }


@router.post("/demo_step")
async def demo_step(request: Request):
    """
    Performs a step in the demo environment
    :param request:
    :return:
    """
    request = await request.json()
    session_id = request["session_id"]
    action = request["action"]

    try:
        return_data = demo_perform_step(session_id, action)
        success = True
    except Exception as e:
        return_data = {"reward": 0, "done": False, "infos": {}}
        success = False

    return {"step": return_data, "success": success}


@router.post("/end_demo_session")
async def end_demo_session(request: Request):
    """
    Closes the demo session
    :param request:
    :return:
    """
    request = await request.json()
    session_id = request["session_id"]
    pid = request["pid"]

    return close_demo_session(session_id, pid)


@router.get("/get_demo_image", response_class=FileResponse)
async def get_demo_image(session_id: str):
    """
    Returns the demo image
    :param session_id:
    :return:
    """
    return FileResponse(
        os.path.join("data", "current_demos", f"{session_id}.jpg"),
        media_type="image/jpg",
    )


@router.get("/check_demo_connection")
async def check_demo_connection(session_id: str):
    """
    Checks whether the demo session is still alive
    :param session_id:
    :return:
    """
    return check_socket_connection(session_id)


def _load_data(
    load_file_name: str,
    env_id: int = -1,
    return_raw: bool = False,
    env_space_info: dict = {},
    split_by_episode: bool = False,
    record_videos: bool = False,
) -> List[RecordedEpisodes]:
    load_episodes = EpisodeRecorder.load_episodes(os.path.join("data", "saved_benchmarks", load_file_name))

    if load_episodes.obs.size == 0:
        return RecordedEpisodes()

    if not split_by_episode:
        split_indices = [0, len(load_episodes.dones)]
    else:
        split_indices = np.where(load_episodes.dones)[0]
        # add 0 and the last index to the split indices
        split_indices = np.concatenate(([0], split_indices, [len(load_episodes.dones)]))

    multi_dones_per_episode = len(split_indices) / load_episodes.episode_lengths.shape[0] > 1

    recorded_episodes = [
        RecordedEpisodes(
            env_id=env_id,
            obs=load_episodes.obs[split_indices[i] : split_indices[i + 1]].tolist(),
            actions=load_episodes.actions[split_indices[i] : split_indices[i + 1]].tolist(),
            rewards=load_episodes.rewards[split_indices[i] : split_indices[i + 1]].tolist(),
            dones=load_episodes.dones[split_indices[i] : split_indices[i + 1]].tolist(),
            infos=[info for info in load_episodes.infos[split_indices[i] : split_indices[i + 1]]],
            renders=load_episodes.renders[split_indices[i] : split_indices[i + 1]].tolist(),
            features=load_episodes.features[split_indices[i] : split_indices[i + 1]].tolist(),
            probs=load_episodes.probs[split_indices[i] : split_indices[i + 1]].tolist(),
            episode_rewards=load_episodes.episode_rewards[i].tolist()
            if len(load_episodes.episode_rewards) > 0 and not multi_dones_per_episode
            else [],
            episode_lengths=load_episodes.episode_lengths[i].tolist()
            if len(load_episodes.episode_lengths) > 0 and not multi_dones_per_episode
            else [],
            additional_metrics=load_episodes.additional_metrics.item(),
        )
        for i in range(len(split_indices) - 1)
    ]

    # generate a video for each episode, consisting of the renders (redundant unpacking)
    if record_videos:
        for i, episode in enumerate(recorded_episodes):
            renders = np.array(episode.renders)
            out = cv2.VideoWriter(
                os.path.join("data", "saved_renders", f"{load_file_name}_{i}.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                24,
                (renders.shape[2], renders.shape[1]),
            )
            for render in renders:
                render = cv2.convertScaleAbs(render)
                render = cv2.cvtColor(render, cv2.COLOR_RGB2BGR)
                out.write(render)
            out.release()

    # Zero out the renders to save memory
    if (
        not return_raw and len(load_episodes.obs.shape) >= 3
    ):  # If the observation is an image, overwrite it because it is too large
        for episode in recorded_episodes:
            episode.obs = [[0]]
            episode.renders = [[0]]
            episode.features = [[0]]
            episode.probs = [[0]]

    return recorded_episodes


def _load_dataset(db_dataset: Dataset, return_raw=False, split_by_episode: bool = False) -> List[RecordedEpisodes]:
    print("load_dataset:", db_dataset.dataset_name)

    load_episodes = EpisodeRecorder.load_episodes(os.path.join("datasets", db_dataset.dataset_path))

    if load_episodes.obs.size == 0:
        return RecordedEpisodes()

    # For datasets, make sure the last step is also marked as done
    load_episodes.dones[-1] = True

    if not return_raw and np.prod(load_episodes.obs.shape[1:]) > 100:
        load_episodes.obs = np.zeros(1)
    if not return_raw and np.prod(load_episodes.renders.shape[1:]) > 100:
        load_episodes.renders = np.zeros(1)
    if not return_raw and np.prod(load_episodes.features.shape[1:]) > 100:
        load_episodes.features = np.zeros(1)

    if not split_by_episode:
        split_indices = [0, len(load_episodes.dones)]
    else:
        split_indices = np.where(load_episodes.dones)[0]
        # add 0 and the last index to the split indices
        split_indices = np.concatenate(([0], split_indices, [len(load_episodes.dones)]))

    multi_dones_per_episode = len(split_indices) / load_episodes.episode_lengths.shape[0] > 1

    return [
        RecordedEpisodes(
            env_id=-1,
            obs=load_episodes.obs[split_indices[i] : split_indices[i + 1]].tolist(),
            actions=load_episodes.actions[split_indices[i] : split_indices[i + 1]].tolist(),
            rewards=load_episodes.rewards[split_indices[i] : split_indices[i + 1]].tolist(),
            dones=load_episodes.dones[split_indices[i] : split_indices[i + 1]].tolist(),
            infos=convert_infos(load_episodes.infos[split_indices[i] : split_indices[i + 1]]),
            renders=load_episodes.renders[split_indices[i] : split_indices[i + 1]].tolist(),
            features=load_episodes.features[split_indices[i] : split_indices[i + 1]].tolist(),
            probs=load_episodes.probs[split_indices[i] : split_indices[i + 1]].tolist(),
            episode_rewards=load_episodes.episode_rewards[i].tolist()
            if len(load_episodes.episode_rewards) > 0 and not multi_dones_per_episode
            else [],
            episode_lengths=load_episodes.episode_lengths[i].tolist()
            if len(load_episodes.episode_lengths) > 0 and not multi_dones_per_episode
            else [],
            additional_metrics=load_episodes.additional_metrics.item(),
        )
        for i in range(len(split_indices) - 1)
    ]
