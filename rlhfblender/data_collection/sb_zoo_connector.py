import difflib
import os
import time
import uuid

import gymnasium as gym
import numpy as np
import stable_baselines3.common.policies
import torch as th
from training_baselines.exp_manager import ExperimentManager
from training_baselines.utils import ALGOS
from stable_baselines3.common.utils import set_random_seed

import rlhfblender.data_models.connector as connector

# Register custom envs
from rlhfblender.data_handling.database_handler import get_single_entry
from rlhfblender.data_models.agent import TrainedAgent
from rlhfblender.data_models.global_models import (
    Environment,
    EvaluationConfig,
    Experiment,
    Project,
)
from rlhfblender.utils import process_env_name
from rlhfblender.utils.read_sb3_configs import read_sb3_configs


class StableBaselines3Agent(TrainedAgent):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        env: gym.Env,
        exp: Experiment,
        device="auto",
        **kwargs,
    ):
        super().__init__(observation_space, action_space, env, exp.path, device=device)

        # If checkpoint step is provided, load the model from the checkpoint instead of the fully trained model
        if "checkpoint_step" in kwargs:
            path = os.path.join(exp.path, "rl_model_{}_steps.zip".format(kwargs["checkpoint_step"]))
        else:
            path = os.path.join(exp.path, f"{exp.env_id}.zip")

        # try to infer algorithm from saved model
        try:
            config = read_sb3_configs(os.path.join(path, process_env_name(exp.env_id), "config.yml"))
            print("Found config file")
            algo = config["algo"]
        except Exception:
            print("Could not read an algorithm from the config file, defaulting to PPO")
            algo = exp.algorithm.lower() if exp.algorithm else "ppo"

        # for some models, we use schedules for training, but we don't need them here for inference, we set it to 0
        self.model = ALGOS[algo].load(path, device=device, custom_objects={"learning_rate": 0.0, "clip_range": 0.0})
        self.agent_state = None
        if "deterministic" in kwargs:
            self.deterministic = kwargs["deterministic"]
        else:
            self.deterministic = False

    def act(self, observation) -> np.ndarray:
        act, state = self.model.predict(observation, state=self.agent_state, deterministic=self.deterministic)
        # Do the state handling internally if necessary
        if state is not None:
            self.agent_state = state
        return act

    def reset(self):
        pass

    def additional_outputs(self, observation, action, output_list=None) -> dict | None:
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
        obs_to_tensor = self.model.policy.obs_to_tensor(observation)[0]
        """if "log_probs" in output_list:
            if isinstance(self.model.policy, stable_baselines3.common.policies.ActorCriticPolicy) and isinstance(
                self.model.policy.action_dist,
                stable_baselines3.common.distributions.CategoricalDistribution,
            ):
                out_dict["log_probs"] = (
                    self.model.policy.get_distribution(obs_to_tensor).distribution.probs.detach().cpu().numpy()
                )
            else:
                out_dict["log_prob"] = np.array([0.0])
        """
        #if "feature_extractor_output" in output_list:
        #    out_dict["feature_extractor_output"] = self.model.policy.extract_features(obs_to_tensor).detach().cpu().numpy()
        if any(v in ["value", "entropy"] for v in output_list):
            if isinstance(self.model.policy, stable_baselines3.common.policies.ActorCriticPolicy):
                value, _, entropy = self.model.policy.evaluate_actions(
                    obs_to_tensor, th.from_numpy(action).to(self.model.policy.device)
                )
                out_dict["value"] = value.detach().cpu().numpy()
                out_dict["entropy"] = entropy.detach().cpu().numpy()
            else:
                out_dict["value"] = np.array([0.0])
                out_dict["entropy"] = np.array([0.0])

        return out_dict

    def extract_features(self, observation):
        if isinstance(self.model.policy, stable_baselines3.common.base_class.BasePolicy):
            return self.model.policy.extract_features(self.model.policy.obs_to_tensor(observation)[0]).detach().cpu().numpy()
        else:
            return np.zeros(0)


class StableBaselines3ZooConnector(connector.Connector):
    def __init__(self):
        super().__init__()

    def start_training(self, experiment: Experiment, project: Project):
        self._run_training(experiment, project)

    def continue_training(self, experiment: Experiment, project: Project):
        self._run_training(experiment, project, continue_training=True)

    def start_training_sweep(self, experiments: list[Experiment], project: Project):
        # Combine experiments into one, create hyperparameter sweep
        # and run training
        sweep_experiment = self._combine_experiments(experiments)
        sweep_config = self._create_sweep_config(sweep_experiment)
        self._run_training(sweep_experiment, project, sweep_config=sweep_config)

    def _run_training(
        self,
        experiment: Experiment,
        project: Project,
        continue_training: bool = False,
        sweep_config: dict | None = None,
    ):
        """

        :param experiment:
        :param project:
        :return:
        """
        try:
            env = get_single_entry(self.database, Environment, key=experiment.env_id, key_column="registration_id")
        except Exception as e:
            print(e)
            return

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
            except ImportError as err:
                raise ImportError(
                    "if you want to use Weights & Biases to track experiment, please install W&B via `pip install wandb`"
                ) from err

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
            tensorboard_log = f"runs/{run_name}"
        else:
            tensorboard_log = experiment_dir

        # Create the experiment manager
        exp_manager = ExperimentManager(
            None,
            experiment.algorithm,
            registration_env_id,
            experiment_dir,
            tensorboard_log=tensorboard_log,
            n_timesteps=experiment.num_timesteps,
            eval_freq=experiment.callback_frequency,
            n_eval_episodes=experiment.episodes_per_eval // 5,
            save_freq=experiment.callback_frequency,
            hyperparams=experiment.hyperparams,
            env_kwargs=experiment.environment_config,
            trained_agent=experiment.trained_agent_path if continue_training else None,
            optimize_hyperparameters=False,
            seed=experiment.seed,
            save_replay_buffer=False,
            verbose=1,
            device=experiment.device,
        )

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

    def _combine_experiments(self, experiments: list[Experiment]):
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
        experiments: list[Experiment],
        project: Project,
        evaluation_configs: list[EvaluationConfig],
    ):
        """
        Starts evaluation of multiple experiments.
        :param experiments:
        :param project:
        :param evaluation_configs:
        :return:
        """

    @staticmethod
    def get_algorithms() -> list[str]:
        """
        Returns all available algorithms.
        :return:
        """
        return list(ALGOS.keys())

    def get_algorithm_default_config(self, algorithm_name: str) -> dict:
        """
        Returns the parameter settings values of a selected algorithm.
        :param algorithm_name:
        :return:
        """

    def get_evaluation_agent(self, env: gym.Env, path: str) -> StableBaselines3Agent:
        """
        Returns the model that can be used to run in an environment.
        :return:
        """
        agent = StableBaselines3Agent(env.observation_space, env.action_space)
        agent.load(path)
        return agent
