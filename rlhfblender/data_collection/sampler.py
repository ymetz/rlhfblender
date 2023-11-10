"""
A sampler for episodes for HITL experiments
"""
import os
from enum import Enum
from typing import List, Optional
import numpy as np

from rlhfblender.data_collection.feedback_model import FeedbackModel
from rlhfblender.data_models.global_models import Environment, EpisodeID, Experiment


class SamplerType(Enum):
    sequential = 1
    random = 2
    model_based = 3
    manual = 4


class Sampler:
    def __init__(
        self,
        experiment: Experiment,
        env: Environment,
        saved_episode_dir: str,
        max_episode_count: int = 1000,
        sampler_type: SamplerType = SamplerType.sequential,
        sample_model: Optional[FeedbackModel] = None,
    ):

        self.experiment = experiment
        self.env = env
        self.saved_episode_dir = saved_episode_dir
        self.max_episode_count = max_episode_count
        self.sampler_type = sampler_type
        self.sample_model = sample_model

        self.episode_count = None
        self.episode_buffer: List[EpisodeID] = []
        self.episode_pointer = 0

        if experiment is not None and env is not None:
            self.set_sampler(experiment, env)

    def set_sampler(
        self,
        experiment: Experiment,
        env: Environment,
        sampling_strategy: str = "sequential",
    ):

        self.experiment = experiment
        self.env = env

        env_name = (
            env.env_name if "ALE" not in env.env_name else env.env_name.split("/")[-1]
        )  # Kinda special case but okay
        self.episode_buffer = []
        for checkpoint in self.experiment.checkpoint_list:
            cp_path = os.path.join(
                self.saved_episode_dir,
                env_name
                + "_trained_"
                + str(self.experiment.id)
                + "_"
                + str(checkpoint),
            )
            if not os.path.exists(cp_path):
                continue
            for file in os.listdir(cp_path):
                if file.endswith(".webm"):
                    episode_info = EpisodeID(
                        benchmark_id=self.experiment.id,
                        benchmark_type="trained",
                        env_name=env_name,
                        checkpoint_step=checkpoint,
                        episode_num=file.split(".")[0],
                    )

                    self.episode_buffer.append(episode_info)

        self.episode_buffer = sorted(
            self.episode_buffer, key=lambda x: (x.checkpoint_step, x.episode_num)
        )

        self.episode_count = len(self.episode_buffer)
        if self.episode_count > self.max_episode_count:
            pass

        self.episode_pointer = 0

        self.sampler_type = SamplerType[sampling_strategy]

        print("Sampler initialized with {} episodes".format(self.episode_count))

        self.reset()

    def get_full_episode_list(self) -> List[EpisodeID]:
        return self.episode_buffer

    def set_config(self, **kwargs):

        for key, value in kwargs.items():
            if key == "max_episode_count":
                self.max_episode_count = value
            elif key == "sampler_type":
                self.sampler_type = value
            elif key == "sample_model":
                self.sample_model = value
            else:
                raise Exception("Invalid key {} for sampler config".format(key))

    def reset(self):
        if self.sampler_type == SamplerType.sequential:
            self.episode_pointer = 0

        if self.sample_model is not None:
            self.sample_model.reset()

    def sample(self, batch_size: int = 1) -> List[EpisodeID]:
        """
        Return a list of episodes
        :param batch_size:
        :return:
        """
        if self.sampler_type == SamplerType.sequential:
            sampled_batch = self.episode_buffer[
                self.episode_pointer: self.episode_pointer + batch_size
            ]
            self.episode_pointer += batch_size
        elif self.sampler_type == SamplerType.random:
            sampled_batch = np.random.choice(self.episode_buffer, batch_size)
        elif self.sampler_type == SamplerType.model_based:
            sampled_batch = self.sample_model.sample(self.episode_buffer, batch_size)
        else:
            raise Exception("Invalid sampler type")
        return sampled_batch

    def configure_sampler(self, **kwargs):

        self.sampler_type = kwargs.get("sampler_type", SamplerType.sequential)
        self.sample_model = kwargs.get("sample_model", None)
        self.max_episode_count = kwargs.get("max_episode_count", 1000)
        self.episode_pointer = kwargs.get("episode_pointer", 0)
