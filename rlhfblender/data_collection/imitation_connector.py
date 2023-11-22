import difflib
import importlib
import os
import time
import uuid
from typing import List, Optional

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3.common.utils import set_random_seed

from rlhfblender.data_handling.database_handler import get_single_entry
from rlhfblender.data_models import connector
from rlhfblender.data_models.global_models import (
    Environment,
    EvaluationConfig,
    Experiment,
    Project,
)
from rlhfblender.utils.exp_manager import ExperimentManager as exp_manager

from .sb_zoo_connector import StableBaselines3Agent

# Register custom envs


class ImitationConnector(connector.Connector):
    def __init__(self):
        super().__init__()

    def start_training(self, experiment: Experiment, project: Project):
        self._run_training(experiment, project)

    def continue_training(self, experiment: Experiment, project: Project):
        self._run_training(experiment, project, continue_training=True)

    def _run_training(
        self,
        experiment: Experiment,
        project: Project,
        continue_training: bool = False,
        sweep_config: Optional[dict] = None,
    ):
        """

        :param experiment:
        :param project:
        :return:
        """
        try:
            env = get_single_entry(self.database, Environment, experiment.env_id)
        except Exception as e:
            print(e)
            return

        # Going through custom gym packages to let them register in the global registory
        for env_module in env.additional_gym_packages:
            importlib.import_module(env_module)

        registration_env_id = env.registration_id
        registered_envs = set(gym.envs.registry.env_specs.keys())  # pytype: disable=module-attr

        # If the environment is not found, suggest the closest match
        if registration_env_id not in registered_envs:
            try:
                closest_match = difflib.get_close_matches(registration_env_id, registered_envs, n=1)[0]
            except IndexError:
                closest_match = "'no close match found...'"
            raise ValueError(f"{registration_env_id} not found in gym registry, you maybe meant {closest_match}?")

        # Unique id to ensure there is no race condition for the folder creation
        f"_{uuid.uuid4()}" if experiment.pid else ""
        if experiment.seed < 0:
            # Seed but with a random one
            experiment.seed = np.random.randint(2**32 - 1, dtype="int64").item()

        set_random_seed(experiment.seed)

        # Setting num threads to 1 makes things run faster on cpu
        if experiment.num_threads > 0:
            th.set_num_threads(experiment.num_threads)

        if continue_training and experiment.trained_agent_path != "":
            assert experiment.trained_agent_path.endswith(".zip") and os.path.isfile(
                experiment.trained_agent_path
            ), "The trained_agent must be a valid path to a .zip file"

        print("=" * 10, registration_env_id, "=" * 10)
        print(f"Seed: {experiment.seed}")

        # Set random seed
        set_random_seed(experiment.seed)
        # Create the experiment directory
        experiment_dir = os.path.join(self.experiment_dir, experiment.exp_name + str(experiment.id))

        if experiment.wandb_tracking:
            try:
                import wandb
            except ImportError:
                raise ImportError(
                    "if you want to use Weights & Biases to track experiment, please install W&B via `pip install wandb`"
                )

            run_name = f"{experiment.env_id}__{experiment.algorithm}__{experiment.seed}__{int(time.time())}"
            run = wandb.init(
                name=run_name,
                project=project.project_name,
                entity=project.wandb_entity,
                config=experiment.dict(),
                sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                monitor_gym=True,  # auto-upload the videos of agents playing the game
                save_code=True,  # optional
            )
            f"runs/{run_name}"
        else:
            pass

        # Prepare experiment and launch hyperparameter optimization if needed
        results = exp_manager.setup_experiment()
        if results is not None:
            model, saved_hyperparams = results
            if experiment.wandb_tracking:
                # we need to save the loaded hyperparameters
                experiment.saved_hyperparams = saved_hyperparams
                run.config.setdefaults(experiment.dict())

            # Normal training
            if model is not None:
                exp_manager.learn(model)
                exp_manager.save_trained_model(model)

    def _combine_experiments(self, experiments: List[Experiment]):
        """
        Combines multiple experiment configurations into one.
        Keep shared settings, and create parameter configs
        """
        # Get the shared settings
        shared_settings = experiments[0].dict()
        for key in shared_settings:
            if key not in ["algorithm", "env_id", "environment_config"]:
                del shared_settings[key]

        # Create the parameter configs
        parameter_configs = []
        for experiment in experiments:
            experiment_settings = experiment.dict()
            for key in shared_settings:
                if key not in experiment_settings:
                    experiment_settings[key] = shared_settings[key]

            parameter_configs.append(experiment_settings)

        return shared_settings, parameter_configs

    def start_evaluation(
        self,
        experiment: Experiment,
        project: Project,
        evaluation_config: EvaluationConfig,
    ):
        """
        Starts evaluation of the experiment.
        :param experiment: Experiment object
        :param project: Project object
        :param evaluation_config: EvaluationConfig object
        :return:
        """

    def start_evaluation_sweep(
        self,
        experiments: List[Experiment],
        project: Project,
        evaluation_configs: List[EvaluationConfig],
    ):
        """
        Starts evaluation of multiple experiments.
        :param experiments:
        :param project:
        :param evaluation_configs:
        :return:
        """

    @staticmethod
    def get_algorithms() -> List[str]:
        """
        Returns all available algorithms.
        :return:
        """
        return [
            "BC",
            "GAIL",
            "AIRL",
            "DAGGER",
            "DENSITY",
            "MCE_IRL",
            "PREFERENCE_COMPARISON",
        ]

    def get_algorithm_default_config(self, algorithm_name: str) -> dict:
        """
        Returns the parameter settings values of a selected algorithm.
        :param algorithm_name:
        :return:
        """
        from imitation.algorithms import (
            bc,
            dagger,
            density,
            mce_irl,
            preference_comparison,
        )
        from imitation.algorithms.adversarial import airl, gail

        selected_algorithm = None
        if algorithm_name == "BC":
            selected_algorithm = bc.BC
        elif algorithm_name == "GAIL":
            selected_algorithm = gail.GAIL
        elif algorithm_name == "AIRL":
            selected_algorithm = airl.AIRL
        elif algorithm_name == "DAGGER":
            selected_algorithm = dagger.DAggerTrainer
        elif algorithm_name == "DENSITY":
            selected_algorithm = density.Density
        elif algorithm_name == "MCE_IRL":
            selected_algorithm = mce_irl.MCEIRL
        elif algorithm_name == "PREFERENCE_COMPARISON":
            selected_algorithm = preference_comparison.PreferenceComparison

        if selected_algorithm is None:
            raise ValueError(f"Algorithm {algorithm_name} is not supported.")

    def get_evaluation_agent(self, env: gym.Env, path: str) -> StableBaselines3Agent:
        """
        Returns the model that can be used to run in an environment.
        :return:
        """
        agent = StableBaselines3Agent(env.observation_space, env.action_space)
        agent.load(path)
        return agent
