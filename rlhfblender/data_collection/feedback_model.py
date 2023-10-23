from typing import List

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from data_models.feedback_models import StandardizedFeedback
from data_models.global_models import Experiment
from imitation.rewards.reward_nets import RewardNet
from torch.utils.data import DataLoader, Dataset


class FeedbackDataset(Dataset):
    def __init__(self, obs_buffer, action_buffer, reward_buffer, user_feedback_buffer):
        self.obs_buffer = obs_buffer
        self.action_buffer = action_buffer
        self.reward_buffer = reward_buffer
        self.user_feedback_buffer = user_feedback_buffer

    def __len__(self):
        return len(self.obs_buffer)

    def __getitem__(self, idx):
        return (
            self.obs_buffer[idx],
            self.action_buffer[idx],
            self.reward_buffer[idx],
            self.user_feedback_buffer[idx],
        )


class FeedbackNet(RewardNet):
    def __init__(self, observation_space, action_space):
        """
        Initialize the feedback model
        :param observation_space:
        :param action_space:
        """

        super().__init__(observation_space, action_space)

        self.observation_space = observation_space
        self.action_space = action_space

        self.fc1 = nn.Linear(observation_space.shape[0] + action_space.shape[0], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(
        self,
        obs: th.Tensor,
        action: th.Tensor,
        next_obs: th.Tensor = None,
        done: th.Tensor = None,
        **kwargs
    ):
        x = th.cat([obs, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def predict(
        self,
        obs: th.Tensor,
        action: th.Tensor,
        next_obs: th.Tensor = None,
        done: th.Tensor = None,
        **kwargs
    ):
        return self.forward(obs, action, next_obs, done)


class FeedbackModel:
    def __init__(self, experiment: Experiment, env: gym.Env):
        """
        Initialize the feedback model
        :param experiment:
        :param env:
        """
        self.experiment = experiment
        self.env = env
        self.model = None

        self.selected_episodes: List[str] = []
        self.selected_episodes_feedback: List[StandardizedFeedback] = []

    def train(self, epochs: int = 10):
        """
        Train the feedback model on the selected episodes
        :param epochs: Number of epochs to train the model
        :return: None
        """

        obs_buffer = []
        action_buffer = []
        reward_buffer = []

        for episode in self.selected_episodes:
            data = np.load(episode)
            obs_buffer.append(data["obs"])
            action_buffer.append(data["actions"])
            reward_buffer.append(data["rewards"])

        # GT reward is in reward buffer, additionaly we have a feedback buffer
        # with the feedback from the user
        user_feedback_buffer = []
        for episode_feedback in self.selected_episodes_feedback:
            user_feedback_buffer.append(episode_feedback.feedback)

        obs_buffer = np.concatenate(obs_buffer, axis=0)
        action_buffer = np.concatenate(action_buffer, axis=0)
        reward_buffer = np.concatenate(reward_buffer, axis=0)
        user_feedback_buffer = np.concatenate(user_feedback_buffer, axis=0)

        dataset = FeedbackDataset(
            obs_buffer, action_buffer, reward_buffer, user_feedback_buffer
        )
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        self.model = FeedbackNet(self.env.observation_space, self.env.action_space)
        self.model.train()
        optimizer = th.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            for obs, action, reward, user_feedback in dataloader:
                optimizer.zero_grad()
                pred = self.model(obs, action)
                loss = criterion(pred, reward)
                loss.backward()
                optimizer.step()

        self.model.eval()

    def predict(self, obs, action):
        """
        Predict the reward for a given observation and action
        :param obs:
        :param action:
        :return:
        """
        if self.model is None:
            raise Exception("Model is not trained yet")
        return self.model(obs, action).detach().numpy()
