from typing import Type

from rlhfblender.data_collection.babyai_connector import BabyAIAgent
from rlhfblender.data_collection.sb_zoo_connector import (StableBaselines3Agent,
                                              StableBaselines3ZooConnector)
from rlhfblender.data_models.agent import TrainedAgent
from rlhfblender.data_models.connector import Connector

SUPPORTED_FRAMEWORK_LIST = ["StableBaselines3", "BabyAI"]


def get_connector(framework: str) -> Type[Connector]:
    """
    Get the connector for the given framework.
    :param framework:
    :return:
    """
    assert framework in SUPPORTED_FRAMEWORK_LIST, "Framework not supported"
    if framework == "StableBaselines3":
        return StableBaselines3ZooConnector
    else:
        raise ValueError("Framework not supported.")


def get_agent(framework: str) -> Type[TrainedAgent]:
    """
    Get the agent for the given framework.
    :param framework:
    :return:
    """
    assert framework in SUPPORTED_FRAMEWORK_LIST, "Framework not supported"
    if framework == "StableBaselines3":
        return StableBaselines3Agent
    elif framework == "BabyAI":
        return BabyAIAgent
    else:
        raise ValueError("Framework not supported.")


def get_framework_list() -> list:
    """
    Get the list of supported frameworks.
    :return:
    """
    return SUPPORTED_FRAMEWORK_LIST
