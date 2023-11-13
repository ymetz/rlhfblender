from abc import abstractmethod
from typing import List

from databases import Database

from rlhfblender.data_models.agent import BaseAgent
from rlhfblender.data_models.global_models import EvaluationConfig, Experiment, Project


class Connector(object):
    def __init__(
        self, database: Database, experiment_dir: str, gym_compatible: bool = True
    ):
        """
        Common Connector Super class for all data connectors.
        :param database: Database object<databases.Database>
        :param experiment_dir: Directory where the experiments are stored
        :param gym_compatible: Boolean, if the data connector is compatible with gym
        """
        self.database = database
        self.experiment_dir = experiment_dir
        self.gym_compatible = gym_compatible

    @abstractmethod
    def start_training(self, experiment: Experiment, project: Project):
        """
        Starts training of the experiment.
        :param experiment: Experiment object
        :param project: Project object
        :return:
        """

    @abstractmethod
    def start_training_sweep(self, experiments: List[Experiment], project: Project):
        """
        Starts training of the experiment.
        :param experiments: List of Experiment object
        :param project: Experiment object
        :return:
        """

    @abstractmethod
    def continue_training(self, experiment: Experiment, project: Project):
        """
        Continues training of the experiment.
        :param experiment: Experiment object
        :param project: Project object
        :return:
        """

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
    def get_algorithms(self) -> List[str]:
        """
        Returns all available algorithms.
        :return:
        """

    @abstractmethod
    def get_algorithm_default_config(self, algorithm_name: str) -> dict:
        """
        Returns the parameter settings values of a selected algorithm.
        :param algorithm_name:
        :return:
        """

    @abstractmethod
    def get_evaluation_agent(self) -> BaseAgent:
        """
        Returns the model that can be used to run in an environment.
        :return:
        """
