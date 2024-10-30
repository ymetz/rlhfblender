import os
from typing import Dict, Optional

import gymnasium
import numpy as np
import torch as th
from scipy.stats import entropy

from rlhfblender.data_models.agent import TrainedAgent
from rlhfblender.data_models.global_models import Experiment
from rlhfblender.utils import babyai_utils as utils


class BabyAIAgent(TrainedAgent):
    """
    BabyAI agents have a separate network architecture and preprocessing pipeline.
    To be compatible with pre-trained models, we instantiate BabyAI agents as a separate class.

    : param observation_space: The observation space of the environment
    : param action_space: The action space of the environment
    : param exp: The experiment object
    : param env: The environment object
    : param device: The device on which to run the model
    : param kwargs: Additional keyword arguments (e.g. the checkpoint step [checkpoint_step=<int>, deterministic=<bool>])
    """

    def __init__(
        self,
        observation_space: gymnasium.spaces.Space,
        action_space: gymnasium.spaces.Space,
        exp: Experiment,
        env: gymnasium.Env,
        device="auto",
        **kwargs,
    ):
        super().__init__(observation_space, action_space, env, exp.path, device=device)

        # If checkpoint step is provided, load the model from the checkpoint instead of the fully trained model
        if "checkpoint_step" in kwargs:
            path = os.path.join(exp.path, "rl_model_{}_steps".format(kwargs["checkpoint_step"]))
        else:
            path = os.path.join(exp.path, f"{exp.env_id}")

        torch_model = th.load(os.path.join(path, "model.pt"), map_location="cpu")

        obss_preprocessor = utils.ObssPreprocessor("TrainedModel", env.observation_space, load_vocab_from=path)
        self.model = utils.ModelAgent(torch_model, obss_preprocessor, argmax=True)
        self.agent_state = None
        if "deterministic" in kwargs:
            self.deterministic = kwargs["deterministic"]
        else:
            self.deterministic = False

        self.current_prediction = None

    def act(self, observation) -> np.ndarray:
        """
        Return the action to take in the environment
        """
        act = self.model.act(observation)
        self.current_prediction = act
        return act["action"][0]

    def reset(self):
        """
        Reset the agent.
        No need to reset the model, as it is stateless/ can handle environement resets internally.
        """
        pass

    def additional_outputs(self, observation, action, output_list=None) -> Optional[Dict]:
        """
        If the model has additional outputs, they can be accessed here. Containts the current outputs for the previous act().
        :param observation: The observation from the environment
        :param action: The action taken in the environment
        :param output_list: A list of outputs to return
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
                np.array([entropy(self.current_prediction["dist"].probs.squeeze().numpy())])
                if self.current_prediction is not None
                else np.array([0.0])
            )
        return out_dict

    def extract_features(self, observation):
        """
        Extract the latent features from the observation
        Currently not implemented for BabyAI
        :param observation: The observation from the environment
        """
        return np.zeros(0)
