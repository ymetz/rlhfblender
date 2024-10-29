import os

import gymnasium as gym
import numpy as np
import pytest

# Assuming your code is in a module named 'rlhfblender'
from rlhfblender.data_collection.episode_recorder import EpisodeRecorder, convert_infos
from rlhfblender.data_models.agent import RandomAgent


class TestEpisodeRecorder:

    @pytest.fixture(scope="class")
    def env(self):
        # Create the Pendulum-v1 environment
        env = gym.make("Pendulum-v1")
        yield env
        env.close()

    @pytest.fixture(scope="class")
    def agent(self, env):
        # Instantiate the RandomAgent with the environment's spaces
        return RandomAgent(observation_space=env.observation_space, action_space=env.action_space)

    @pytest.fixture(scope="class")
    def save_path(self):
        # Define the path where episodes will be saved
        return os.path.join("test_data", "episode_recording")

    @pytest.fixture(scope="class", autouse=True)
    def cleanup(self, save_path):
        # Cleanup code to remove the test file after all tests in the class are done
        yield
        if os.path.exists(save_path + ".npz"):
            os.remove(save_path + ".npz")

    @pytest.mark.dependency()
    def test_record_episodes(self, env, agent, save_path):
        """
        Test recording episodes with the RandomAgent.
        """
        # Remove existing file if it exists
        if os.path.exists(save_path + ".npz"):
            os.remove(save_path + ".npz")

        # Instantiate the EpisodeRecorder
        recorder = EpisodeRecorder(
            agent=agent,
            env=env,
            n_eval_episodes=1,
            max_steps=500,
            save_path="test_data/episode_recording",
            overwrite=True,
            render=False,
            deterministic=False,
            reset_to_initial_state=False,
        )

        # Record episodes
        recorder.record_episodes()

        # Check if the data was saved
        assert os.path.exists("test_data/episode_recording.npz")

    @pytest.mark.dependency(depends=["test_record_episodes"])
    def test_load_episodes(self, save_path):
        """
        Test loading episodes that were recorded.
        """
        # Load the episodes
        episodes = EpisodeRecorder.load_episodes(save_path)
        # Check that the episodes have data
        assert episodes.obs.size > 0
        assert episodes.actions.size > 0
        assert episodes.rewards.size > 0
        assert episodes.dones.size > 0

    @pytest.mark.dependency(depends=["test_record_episodes"])
    def test_episode_data(self, save_path):
        """
        Test the integrity of the recorded episode data.
        """
        # Load the episodes
        episodes = EpisodeRecorder.load_episodes(save_path)
        # Check shapes of the recorded data
        assert episodes.obs.shape[0] == episodes.actions.shape[0]
        assert episodes.obs.shape[0] == episodes.rewards.shape[0]
        assert episodes.obs.shape[0] == episodes.dones.shape[0]
        # Ensure that the rewards are floats
        assert episodes.rewards.dtype == np.float32 or episodes.rewards.dtype == np.float64

    @pytest.mark.dependency(depends=["test_record_episodes"])
    def test_convert_infos(self, save_path):
        """
        Test the convert_infos function on the loaded infos.
        """
        # Load the episodes
        episodes = EpisodeRecorder.load_episodes(save_path)
        # Convert infos
        infos = convert_infos(episodes.infos)
        # Check that infos have been converted properly
        assert isinstance(infos, list)
        assert isinstance(infos[0], dict)
        assert "id" in infos[0]
        assert "episode step" in infos[0]
