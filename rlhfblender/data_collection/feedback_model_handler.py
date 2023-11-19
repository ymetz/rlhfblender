"""
This file contains the class that handles the feedback model.
It contains a reference to a FeedbackDataset and a FeedbackNet.
Collected feedback can be submitted to the model handler, and training/re-training can be initiated.
"""
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
