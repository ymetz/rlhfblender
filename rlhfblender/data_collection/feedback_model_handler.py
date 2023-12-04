import numpy as np
from imitation.rewards.reward_nets import (
    BasicRewardNet,
    BasicShapedRewardNet,
    CnnRewardNet,
    RewardEnsemble,
    RewardNet,
    RewardNetWithVariance,
)

from rlhfblender.data_collection import feedback_model
from rlhfblender.data_models.feedback_models import (
    FeedbackType,
    StandardizedFeedback,
    UnprocessedFeedback,
)


class FeedbackModelHandler:
    """
    This class handles the feedback model (trainable with collected feedback).
    It contains a reference to a FeedbackDataset and a FeedbackNet.

    Collected feedback can be submitted to the model handler, and training/re-training can be initiated.

    : param session_id: The session ID
    : param feedback_model_cls: The feedback model class
    : param observation_space: The observation space of the environment
    : param action_space: The action space of the environment
    : param feedback_dataset: The feedback dataset
    : param feedback_net: The feedback net
    """

    def __init__(
        self,
        session_id: str,
        feedback_model_cls: RewardNet,
        observation_space,
        action_space,
    ):
        self.session_id = session_id
        self.feedback_model_cls = feedback_model_cls
        self.observation_space = observation_space
        self.action_space = action_space
        self.feedback_dataset = feedback_model.FeedbackDataset(observation_space, action_space)
        self.feedback_net = feedback_model_cls(observation_space, action_space)
