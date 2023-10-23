from typing import List

import numpy as np
from pydantic import BaseModel, validator

"""
    Main Data Models
    All data models should be defined here
    Functions/Data Tables are defined for these data models

"""


class Project(BaseModel):
    id: int = -1
    project_name: str = ""
    created_timestamp: int = -1
    project_path: str = ""
    project_description: str = ""
    project_tags: list = []
    project_environments: list = []
    project_datasets: list = []
    project_experiments: list = []
    wandb_entity: str = ""

    @validator(
        "project_tags",
        "project_environments",
        "project_datasets",
        "project_experiments",
        pre=True,
    )
    def process_(cls, in_list):
        if type(in_list) == list:
            return in_list
        return eval(in_list)


class Experiment(BaseModel):
    id: int = -1
    exp_name: str = ""
    created_timestamp: int = -1
    run_timestamp: int = -1
    last_modified: int
    pid: int = -1
    status: List[str] = []
    env_id: int = -1
    environment_config: dict = {}
    framework: str = ""
    path: str = ""
    algorithm: str = ""
    hyperparams: dict = {}
    num_timesteps: int = -1
    checkpoint_frequency: int = -1
    checkpoint_list: list = []
    episodes_per_eval: int = -1
    parallel_envs: int = -1
    observation_space_info: dict = {}
    action_space_info: dict = {}
    exp_tags: list = []
    exp_comment: str = ""
    wandb_tracking: bool = False
    device: str = "auto"
    trained_agent_path: str = ""
    seed: int = -1

    @validator(
        "status",
        "checkpoint_list",
        "exp_tags",
        "environment_config",
        "hyperparams",
        "observation_space_info",
        "action_space_info",
        pre=True,
    )
    def process_(cls, in_value):
        if type(in_value) == list or type(in_value) == dict:
            return in_value
        return eval(in_value)


class Environment(BaseModel):
    id: int = -1  # Is autofilled in database
    env_name: str = ""
    registered: int = 0
    registration_id: str = ""
    type: str = ""
    observation_space_info: dict = {}
    action_space_info: dict = {}
    has_state_loading: int = 0
    description: str = ""
    tags: list = []
    env_path: str = ""
    additional_gym_packages: List[str] = []

    @validator(
        "tags",
        "additional_gym_packages",
        "observation_space_info",
        "action_space_info",
        pre=True,
    )
    def process_(cls, in_value):
        if type(in_value) == list or type(in_value) == dict:
            return in_value
        return eval(in_value)


class Dataset(BaseModel):
    id: int = -1
    dataset_name: str = ""
    created_timestamp: int = -1
    dataset_path: str = ""
    dataset_description: str = ""
    dataset_tags: list = []
    dataset_environment: str = ""

    @validator("dataset_tags", pre=True)
    def process_(cls, in_list):
        if type(in_list) == list:
            return in_list
        return eval(in_list)


class TrackingItem(BaseModel):
    id: int = -1
    tracking_id: str = -1
    exp_id: int = -1
    exp_name: str = ""
    env_id: int = -1
    env_name: str = ""
    step_value: int = -1
    obs: str = ""
    is_image: int = 0
    has_state: int = 0
    state: str = ""
    dataset_sample_index: int = -1
    interpret_obs_as_state: int = 0


class EvaluationConfig(BaseModel):
    id: int = -1
    eval_name: str = ""
    exp_id: int = -1
    eval_config: dict = {}
    eval_tags: list = []
    eval_comment: str = ""
    eval_timestamp: int = -1

    @validator("eval_tags", pre=True)
    def process_(cls, in_list):
        if type(in_list) == list:
            return in_list
        return eval(in_list)


class RecordedEpisodes(BaseModel):
    env_id: int = -1
    obs: list = []
    rewards: list = []
    dones: list = []
    infos: list = []
    actions: list = []
    renders: list = []
    features: list = []
    probs: list = []
    episode_rewards: list = []
    episode_lengths: list = []
    env_space_info: dict = {}
    additional_metrics: dict = {}


class AggregatedRecordedEpisodes(BaseModel):
    episode_rewards: list = []
    episode_lengths: list = []
    additional_metrics: list = []


class EpisodeID(BaseModel):
    """
    Right now we generate a few rollouts with a specific configuration and save
    the associated data in data/<data_name>/<configuration>/<episode_num>.
    """
    env_name: str = ""  # e.g.: BreakoutNoFrameskip-v4
    benchmark_type: str = ""  # e.g.: trained
    benchmark_id: int = -1  # e.g.: 1
    checkpoint_step: int = -1  # e.g.: 1000000
    episode_num: int = -1