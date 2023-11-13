from typing import Type

from data_models.feedback_models import StandardizedFeedback, UnprocessedFeedback
from data_models.global_models import (
    Dataset,
    Environment,
    EvaluationConfig,
    Experiment,
    Project,
    RecordedEpisodes,
    TrackingItem,
)
from pydantic import BaseModel


def get_model_by_name(name) -> Type[BaseModel]:
    if name == "project":
        return Project
    elif name == "experiment":
        return Experiment
    elif name == "environment":
        return Environment
    elif name == "dataset":
        return Dataset
    elif name == "trackingItem":
        return TrackingItem
    elif name == "evaluationConfig":
        return EvaluationConfig
    elif name == "recordedEpisodes":
        return RecordedEpisodes
    elif name == "episodeFeedback":
        return UnprocessedFeedback
    elif name == "standardizedFeedback":
        return StandardizedFeedback
    else:
        return None
