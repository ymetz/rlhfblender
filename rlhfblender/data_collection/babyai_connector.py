import os
from typing import Dict, Optional

import gymnasium as gym
import numpy as np
import torch as th
from config import DB_HOST
from databases import Database
from scipy.stats import entropy

from rlhfblender.data_models.agent import TrainedAgent
from rlhfblender.data_models.global_models import Experiment
from rlhfblender.utils import babyai_utils as utils

DATABASE = Database(DB_HOST)


class BabyAIAgent(TrainedAgent):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        exp: Experiment,
        env: gym.Env,
        device="auto",
        **kwargs,
    ):
        super().__init__(observation_space, action_space, exp.path, device=device)

        # If checkpoint step is provided, load the model from the checkpoint instead of the fully trained model
        if "checkpoint_step" in kwargs:
            path = os.path.join(
                exp.path, "rl_model_{}_steps".format(kwargs["checkpoint_step"])
            )
        else:
            path = os.path.join(exp.path, "{}".format(exp.env_id))

        torch_model = th.load(os.path.join(path, "model.pt"), map_location="cpu")

        obss_preprocessor = utils.ObssPreprocessor(
            "TrainedModel", env.observation_space, load_vocab_from=path
        )
        self.model = utils.ModelAgent(torch_model, obss_preprocessor, argmax=True)
        self.agent_state = None
        if "deterministic" in kwargs:
            self.deterministic = kwargs["deterministic"]
        else:
            self.deterministic = False

        self.current_prediction = None

    def act(self, observation) -> np.ndarray:
        act = self.model.act(observation)
        self.current_prediction = act
        return act["action"]

    def reset(self):
        pass

    def additional_outputs(
        self, observation, action, output_list=None
    ) -> Optional[Dict]:
        """
        If the model has additional outputs, they can be accessed here.
        :param observation:
        :param action:
        :param output_list:
        :return:
        """
        if output_list is None:
            output_list = []

        out_dict = {}
        if "log_probs" in output_list:
            out_dict["log_probs"] = (
                np.array([self.current_prediction["dist"].probs.squeeze().numpy()])
                if self.current_prediction is not None
                else np.array([0.0])
            )
        if any(v in ["value", "entropy"] for v in output_list):
            out_dict["value"] = (
                np.array([self.current_prediction["value"].squeeze().numpy()])
                if self.current_prediction is not None
                else np.array([0.0])
            )
            out_dict["entropy"] = (
                np.array(
                    [entropy(self.current_prediction["dist"].probs.squeeze().numpy())]
                )
                if self.current_prediction is not None
                else np.array([0.0])
            )
        return out_dict

    def extract_features(self, observation):
        return np.zeros(0)
