import os
import tempfile
import time
from copy import deepcopy
from functools import wraps
from threading import Thread
from typing import Any, Dict, Optional

import gymnasium as gym
import torch as th
from sb3_contrib import TQC
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video


class SaveVecNormalizeCallback(BaseCallback):
    """
    Callback for saving a VecNormalize wrapper every ``save_freq`` steps

    :param save_freq: (int)
    :param save_path: (str) Path to the folder where ``VecNormalize`` will be saved, as ``vecnormalize.pkl``
    :param name_prefix: (str) Common prefix to the saved ``VecNormalize``, if None (default)
        only one file will be kept.
    """

    def __init__(
        self,
        save_freq: int,
        save_path: str,
        name_prefix: Optional[str] = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            if self.name_prefix is not None:
                path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.pkl")
            else:
                path = os.path.join(self.save_path, "vecnormalize.pkl")
            if self.model.get_vec_normalize_env() is not None:
                self.model.get_vec_normalize_env().save(path)
                if self.verbose > 1:
                    print(f"Saving VecNormalize to {path}")
        return True


class ParallelTrainCallback(BaseCallback):
    """
    Callback to explore (collect experience) and train (do gradient steps)
    at the same time using two separate threads.
    Normally used with off-policy algorithms and `train_freq=(1, "episode")`.

    TODO:
    - blocking mode: wait for the model to finish updating the policy before collecting new experience
        at the end of a rollout
    - force sync mode: stop training to update to the latest policy for collecting
        new experience

    :param gradient_steps: Number of gradient steps to do before
        sending the new policy
    :param verbose: Verbosity level
    :param sleep_time: Limit the fps in the thread collecting experience.
    """

    def __init__(self, gradient_steps: int = 100, verbose: int = 0, sleep_time: float = 0.0):
        super().__init__(verbose)
        self.batch_size = 0
        self._model_ready = True
        self._model = None
        self.gradient_steps = gradient_steps
        self.process = None
        self.model_class = None
        self.sleep_time = sleep_time

    def _init_callback(self) -> None:
        temp_file = tempfile.TemporaryFile()

        # Windows TemporaryFile is not a io Buffer
        # we save the model in the logs/ folder
        if os.name == "nt":
            temp_file = os.path.join("logs", "model_tmp.zip")

        self.model.save(temp_file)

        # TODO: add support for other algorithms
        for model_class in [SAC, TQC]:
            if isinstance(self.model, model_class):
                self.model_class = model_class
                break

        assert self.model_class is not None, f"{self.model} is not supported for parallel training"
        self._model = self.model_class.load(temp_file)

        self.batch_size = self._model.batch_size

        # Disable train method
        def patch_train(function):
            @wraps(function)
            def wrapper(*args, **kwargs):
                return

            return wrapper

        # Add logger for parallel training
        self._model.set_logger(self.model.logger)
        self.model.train = patch_train(self.model.train)

        # Hack: Re-add correct values at save time
        def patch_save(function):
            @wraps(function)
            def wrapper(*args, **kwargs):
                return self._model.save(*args, **kwargs)

            return wrapper

        self.model.save = patch_save(self.model.save)

    def train(self) -> None:
        self._model_ready = False

        self.process = Thread(target=self._train_thread, daemon=True)
        self.process.start()

    def _train_thread(self) -> None:
        self._model.train(gradient_steps=self.gradient_steps, batch_size=self.batch_size)
        self._model_ready = True

    def _on_step(self) -> bool:
        if self.sleep_time > 0:
            time.sleep(self.sleep_time)
        return True

    def _on_rollout_end(self) -> None:
        if self._model_ready:
            self._model.replay_buffer = deepcopy(self.model.replay_buffer)
            self.model.set_parameters(deepcopy(self._model.get_parameters()))
            self.model.actor = self.model.policy.actor
            if self.num_timesteps >= self._model.learning_starts:
                self.train()
            # Do not wait for the training loop to finish
            # self.process.join()

    def _on_training_end(self) -> None:
        # Wait for the thread to terminate
        if self.process is not None:
            if self.verbose > 0:
                print("Waiting for training thread to terminate")
            self.process.join()


class VideoRecorderCallback(BaseCallback):
    def __init__(
        self,
        eval_env: gym.Env,
        render_freq: int,
        n_eval_episodes: int = 1,
        deterministic: bool = True,
    ):
        """
        Records a video of an agent's trajectory traversing ``eval_env`` and logs it to TensorBoard

        :param eval_env: A gym environment from which the trajectory is recorded
        :param render_freq: Render the agent's trajectory every eval_freq call of the callback.
        :param n_eval_episodes: Number of episodes to render
        :param deterministic: Whether to use deterministic or stochastic policy
        """
        super().__init__()
        self._eval_env = eval_env
        self._render_freq = render_freq
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic

    def _on_step(self) -> bool:
        if self.n_calls % self._render_freq == 0:
            screens = []

            def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
                """
                Renders the environment in its current state, recording the screen in the captured `screens` list

                :param _locals: A dictionary containing all local variables of the callback's scope
                :param _globals: A dictionary containing all global variables of the callback's scope
                """
                screen = self._eval_env.render()
                # PyTorch uses CxHxW vs HxWxC gym (and tensorflow) image convention
                screens.append(screen.transpose(2, 0, 1))

            evaluate_policy(
                self.model,
                self._eval_env,
                callback=grab_screens,
                n_eval_episodes=self._n_eval_episodes,
                deterministic=self._deterministic,
            )
            self.logger.record(
                "trajectory/video",
                Video(th.ByteTensor([screens]), fps=40),
                exclude=("stdout", "log", "json", "csv"),
            )
        return True
