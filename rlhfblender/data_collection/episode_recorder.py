import os
import random
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Union

import gymnasium as gym
import numpy as np
from pydantic import BaseModel
from stable_baselines3.common.vec_env import VecEnv, VecMonitor, is_vecenv_wrapped

from rlhfblender.data_collection import RecordedEpisodesContainer
from rlhfblender.data_collection.metrics_processor import process_metrics
from rlhfblender.data_models.agent import BaseAgent


class BenchmarkSummary(BaseModel):
    benchmark_steps: int = 0
    episode_lengths: List[int] = []
    episode_rewards: List[int] = []
    additional_metrics: dict


class EpisodeRecorder(object):
    def __init__(self):
        pass

    @staticmethod
    def record_episodes(
        agent: BaseAgent,
        env: Union[gym.Env, VecEnv],
        n_eval_episodes: int = 2,
        max_steps: int = int(1e4),
        save_path: str = "",
        overwrite: bool = False,
        deterministic: bool = False,
        render: bool = True,
        reset_to_initial_state: bool = True,
        additional_out_attributes: Optional[Dict[str, Any]] = None,
        callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    ):
        """
        Runs policy for ``n_eval_episodes`` episodes and returns average reward.
        If a vector env is passed in, this divides the episodes to evaluate onto the
        different elements of the vector env. This static division of work is done to
        remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
        details and discussion.

        .. note::
            If environment has not been wrapped with ``Monitor`` wrapper, reward and
            episode lengths are counted as it appears with ``env.step`` calls. If
            the environment contains wrappers that modify rewards or episode lengths
            (e.g. reward scaling, early episode reset), these will affect the evaluation
            results as well. You can avoid this by wrapping environment with ``Monitor``
            wrapper before anything else.

        :param agent: The agent to select actions
        :param env: The gym environment or ``VecEnv`` environment.
        :param n_eval_episodes: Number of episode to evaluate the agent
        :param deterministic: Whether to use deterministic or stochastic actions
        :param overwrite: Whether to overwrite the existing results
        :param save_path: Path to save the evaluation results
        :param max_steps: If greater than 0, will only run the environment for the given number of steps (default: -1)
        :param render: Whether to render the environment or not
        :param callback: callback function to do additional checks,
            called after each step. Gets locals() and globals() passed as parameters.
        :param reset_to_initial_state: If set, the environment is set to an initial state for each episode
        :param additional_out_attributes: Additional attributes to save in the output file, might depend on the library used
        :return: Mean reward per episode, std of reward per episode.
        """
        is_monitor_wrapped = False
        # Avoid circular import
        from stable_baselines3.common.monitor import Monitor

        # if not isinstance(env, VecEnv):
        #    env = DummyVecEnv([lambda: env])

        is_monitor_wrapped = (
            (is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]) if isinstance(env, VecEnv) else False
        )

        n_envs = env.num_envs if isinstance(env, VecEnv) else 1
        episode_rewards = []
        episode_lengths = []

        episode_counts = np.zeros(n_envs, dtype="int")
        # Divides episodes among different sub environments in the vector as evenly as possible
        episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

        current_rewards = np.zeros(n_envs)
        current_lengths = np.zeros(n_envs, dtype="int")

        obs_buffer = []
        feature_extractor_buffer = []
        actions_buffer = []
        dones_buffer = []
        rew_buffer = []
        render_buffer = []
        infos_buffer = []
        probs_buffer = []

        seed = random.randint(0, 2000)
        if not isinstance(env, VecEnv):
            observations, _ = env.reset(seed=seed)
            rewards = 0
            dones = False
            infos = {}
            # A bit of a special case for babyai, but in the future, we might use gymnasium with reset infos anyways
            if isinstance(env.observation_space, gym.spaces.Dict) and "mission" in observations.keys():
                infos["mission"] = observations["mission"]
                infos["seed"] = seed
        else:
            env.seed(seed=seed)
            observations = env.reset()
            rewards = np.zeros(n_envs)
            dones = np.zeros(n_envs, dtype=bool)
            infos = [{} for _ in range(n_envs)]
            if isinstance(env.observation_space, gym.spaces.Dict) and "mission" in observations[0].keys():
                for i in range(n_envs):
                    infos[i]["mission"] = observations[i]["mission"]
                    infos[i]["seed"] = seed
        if reset_to_initial_state:
            initial_states = tuple(deepcopy(e) for e in env.envs)
        else:
            initial_states = None
        reset_obs = observations.copy()

        states = None
        values = None
        probs = None
        total_steps = 0

        while (episode_counts < episode_count_targets).any() and total_steps <= max_steps:
            actions = agent.act(observations)
            # If policy is not part of the model, we have directly loaded a policy
            additional_out_attributes = agent.additional_outputs(
                observations,
                actions,
                output_list=[
                    "log_probs",
                    "feature_extractor_output",
                    "entropy",
                    "value",
                ],
            )

            # Expand dims, if env was not wrapped with DummyVecEnv
            if not isinstance(env, VecEnv):
                rewards = np.expand_dims(rewards, axis=0)
                dones = np.expand_dims(dones, axis=0)
                infos = [infos]
                # If not dummy vec env, we need to reset ourselves
                if dones:
                    seed = random.randint(0, 1000000)
                    if isinstance(env, VecEnv):
                        env.seed(seed=seed)
                        observations = env.reset()
                    else:
                        observation, _ = env.reset(seed=seed)
                    if isinstance(env.observation_space, gym.spaces.Dict) and "mission" in observation.keys():
                        infos[0]["mission"] = observation["mission"]
                        infos[0]["seed"] = seed
            obs_buffer.append(np.squeeze(observations))
            actions_buffer.append(np.squeeze(actions))
            rew_buffer.append(np.squeeze(rewards))
            dones_buffer.append(np.squeeze(dones))
            current_rewards += rewards
            current_lengths += 1
            tmp_info_buffer = []
            for i in range(n_envs):
                if episode_counts[i] < episode_count_targets[i]:
                    # unpack values so that the callback can access the local variables

                    if "feature_extractor_output" in additional_out_attributes:
                        feature_extractor_buffer.append(np.squeeze(additional_out_attributes["feature_extractor_output"][i]))
                    if "log_probs" in additional_out_attributes:
                        probs_buffer.append(additional_out_attributes["log_probs"][i])

                    if n_envs == 1:
                        info = {
                            **infos[i],
                            **{
                                k: v.item()
                                for k, v in additional_out_attributes.items()
                                if k not in ["log_probs", "feature_extractor_output"]
                            },
                        }
                    else:
                        info = {
                            **infos[i],
                            **{
                                k: np.squeeze(v[i]).item()
                                for k, v in additional_out_attributes.items()
                                if k not in ["log_probs", "feature_extractor_output"]
                            },
                        }

                    if callback is not None:
                        callback(locals(), globals())

                    total_steps += 1

                    if dones[i]:
                        if is_monitor_wrapped:
                            # Atari wrapper can send a "done" signal when
                            # the agent loses a life, but it does not correspond
                            # to the true end of episode
                            if "episode" in info.keys():
                                # Do not trust "done" with episode endings.
                                # Monitor wrapper includes "episode" key in info if environment
                                # has been wrapped with it. Use those rewards instead.
                                episode_rewards.append(info["episode"]["r"])
                                episode_lengths.append(info["episode"]["l"])
                                info["label"] = "Real Game End"
                                # Only increment at the real end of an episode
                                episode_counts[i] += 1
                                # Reset the done environment to the initial state
                                if reset_to_initial_state:
                                    env.envs[i] = deepcopy(initial_states[i])
                                    observations[i] = reset_obs[i]

                        else:
                            episode_rewards.append(current_rewards[i])
                            episode_lengths.append(current_lengths[i])
                            episode_counts[i] += 1
                            if reset_to_initial_state:
                                env.envs[i] = deepcopy(initial_states[i])
                                observations[i] = reset_obs[i]
                        # Remove terminal_observation as we don't need it to pass to the next episode
                        if "terminal_observation" in info.keys():
                            info.pop("terminal_observation")
                        current_rewards[i] = 0
                        current_lengths[i] = 0
                        if states is not None:
                            states[i] *= 0

                    tmp_info_buffer.append(info)

            infos_buffer.append(np.squeeze(tmp_info_buffer) if len(tmp_info_buffer) > 0 else np.squeeze(infos))

            if render:
                render_frame = env.render()
                render_buffer.append(np.squeeze(render_frame))
            total_steps += 1

            if not isinstance(env, VecEnv):
                observation, reward, terminated, truncated, info = env.step(actions)
                done = terminated or truncated
                observations = np.expand_dims(observation, axis=0)
                rewards = np.expand_dims(reward, axis=0)
                dones = np.expand_dims(done, axis=0)
                infos = [info]
            else:
                observation, rewards, dones, info = env.step(actions)

        # Add last render frame to buffer
        if render:
            render_frame = env.render()
            render_buffer.append(np.squeeze(render_frame))

        if n_envs == 1:
            if len(infos_buffer) > 0 and len(infos_buffer[0].shape) > 0:
                infos_buffer = [info[0].item() for info in infos_buffer]
            else:
                infos_buffer = [info.item() for info in infos_buffer]
            current_step_in_episode = 0
            for i in range(len(infos_buffer)):
                infos_buffer[i]["id"] = i
                if dones_buffer[i]:
                    current_step_in_episode = 0
                infos_buffer[i]["episode step"] = current_step_in_episode
                current_step_in_episode += 1
            if "value" in infos_buffer[0].keys():
                infos_buffer[np.argmax([i["value"] for i in infos_buffer])]["label"] = "Max. Value"
                infos_buffer[np.argmin([i["value"] for i in infos_buffer])]["label"] = "Min. Value"
                infos_buffer[np.argmax(rew_buffer)]["label"] = "Max. Step Reward"
                infos_buffer[np.argmin(rew_buffer)]["label"] = "Min. Step Reward"

        if render:
            render_buffer = np.array(render_buffer)
        else:
            render_buffer = np.zeros(1)

        if len(obs_buffer[0].shape) == 2:
            obs_buffer = np.expand_dims(obs_buffer, axis=-1)
        else:
            obs_buffer = np.array(obs_buffer)
        actions_buffer = np.array(actions_buffer)
        dones_buffer = np.array(dones_buffer)
        rew_buffer = np.array(rew_buffer)
        episode_rewards = np.array(episode_rewards)
        episode_lengths = np.array(episode_lengths)
        feature_extractor_buffer = np.array(feature_extractor_buffer)
        infos_buffer = np.array(infos_buffer)
        probs_buffer = np.array(probs_buffer)

        # If overwrite is not set, we will append to the existing buffer
        if not overwrite and os.path.isfile(save_path + ".npz"):
            previous_bm = np.load(save_path + ".npz", allow_pickle=True)
            obs_buffer = np.concatenate((previous_bm["obs"], obs_buffer), axis=0)
            actions_buffer = np.concatenate((previous_bm["actions"], actions_buffer), axis=0)
            dones_buffer = np.concatenate((previous_bm["dones"], dones_buffer), axis=0)
            rew_buffer = np.concatenate((previous_bm["rewards"], rew_buffer), axis=0)
            episode_rewards = np.concatenate((previous_bm["episode_rewards"], episode_rewards), axis=0)
            episode_lengths = np.concatenate((previous_bm["episode_lengths"], episode_lengths), axis=0)
            feature_extractor_buffer = np.concatenate((previous_bm["features"], feature_extractor_buffer), axis=0)
            infos_buffer = np.concatenate((previous_bm["infos"], infos_buffer), axis=0)
            probs_buffer = np.concatenate((previous_bm["probs"], probs_buffer), axis=0)
            render_buffer = np.concatenate((previous_bm["renders"], render_buffer), axis=0)

        # Recompute metrics (e.g. mean, std, etc), in case the buffers have changed
        additional_metrics = process_metrics(
            RecordedEpisodesContainer(
                obs=obs_buffer,
                actions=actions_buffer,
                dones=dones_buffer,
                rewards=rew_buffer,
                episode_rewards=episode_rewards,
                episode_lengths=episode_lengths,
                features=feature_extractor_buffer,
                infos=infos_buffer,
                probs=probs_buffer,
                renders=render_buffer,
                additional_metrics={},
            )
        )
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(os.path.join(save_path + ".npz"), "wb") as f:
            np.savez(
                f,
                obs=obs_buffer,
                rewards=rew_buffer,
                dones=dones_buffer,
                actions=actions_buffer,
                renders=render_buffer,
                infos=infos_buffer,
                feature_extractor_buffer=feature_extractor_buffer,
                probs=probs_buffer,
                episode_rewards=episode_rewards,
                episode_lengths=episode_lengths,
                additional_metrics=additional_metrics,
            )

        return BenchmarkSummary(
            benchmark_steps=rew_buffer.shape[0],
            episode_lengths=episode_lengths.tolist(),
            episode_rewards=episode_lengths.tolist(),
            additional_metrics=additional_metrics,
        )

    @staticmethod
    def load_episodes(save_path) -> RecordedEpisodesContainer:
        """
        Load episodes from a file.
        """
        print("[INFO] Loading episodes from {}".format(save_path))
        bm = np.load(save_path + ".npz", allow_pickle=True)

        print("[INFO] Successfully loaded NPZ episodes from {}".format(save_path))

        return RecordedEpisodesContainer(
            obs=bm["obs"],
            rewards=bm["rewards"],
            dones=bm["dones"],
            infos=bm["infos"],
            actions=bm["actions"],
            renders=bm["renders"],
            features=bm["feature_extractor_buffer"],
            probs=bm["probs"],
            episode_rewards=bm["episode_rewards"],
            episode_lengths=np.array([1]),  # bm["episode_lengths"],
            additional_metrics=bm["additional_metrics"],
        )

    @staticmethod
    def get_aggregated_data(save_path) -> BenchmarkSummary:
        """
        Load episodes from a file.
        """
        bm = np.load(save_path + ".npz", allow_pickle=True)

        return BenchmarkSummary(
            benchmark_steps=bm["rewards"].shape[0],
            episode_lengths=np.array([1]).tolist(),  # bm["episode_lengths"].tolist(),
            episode_rewards=bm["episode_rewards"].tolist(),
            additional_metrics=bm["additional_metrics"].item(),
        )


def convert_infos(infos: np.ndarray):
    """
    Convert a numpy array of dict objects to a list of dict objects.
    Add id, episode step, and label to each info dict.
    """
    infos = infos.tolist()
    for i, info in enumerate(infos):
        for key, value in info.items():
            if isinstance(value, np.generic):
                info[key] = value.item()
        info["id"] = i
        info["episode step"] = i
    return infos
