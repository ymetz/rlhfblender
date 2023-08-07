import enum
from typing import List, Union

import gym
from data_models.global_models import EpisodeID
from pydantic import BaseModel


class FeedbackType(enum.Enum):
    """
    This is the type of feedback we get from the user when evaluating an episode.
    """

    rating = "rating"
    comparison = "comparison"
    ranking = "ranking"
    demonstration = "demonstration"
    correction = "correction"
    auxiliary_reward = "auxiliary_reward"
    goal = "goal"
    comment = "comment"
    other = "other"

    def __str__(self):
        # just return the enum value
        return self.name


class FeedbackDimension(enum.Enum):
    def __str__(self):
        # just return the enum value
        return self.name

    def __repr__(self):
        # just return the enum value
        return self.name


class EpisodeFeedback(BaseModel):
    """
    This is the feedback we get from the user when evaluating an episode.
    """

    episode_id: EpisodeID
    timestamp: int = -1
    feedback_type: FeedbackType = FeedbackType.rating
    text_feedback: str = ""  # e.g.: "The agent is doing well in the beginning, but then it fails to collect the key."
    text_assignment: List[
        float
    ] = []  # If we already have a step assignment for the text feedback, we can provide it here.
    numeric_feedback: float = 0.0  # e.g.: 0.5
    numeric_assignment: List[
        float
    ] = []  # If we already have a step assignment for the numeric feedback, we can provide it here.
    array_feedback: List[
        Union[float, int]
    ] = []  # e.g.: [0.1, 0.2, 0.3, 0.4, 0.5], array feedback is already assigned to steps.
    ranking_array_feedback: List[
        Union[int]
    ] = []  # e.g.: [0.1, 0.2, 0.3, 0.4, 0.5], array feedback is already assigned to steps.


class Intention(FeedbackDimension):
    evaluate = 1
    instruct = 2
    describe = 3
    none = 4


class Expression(FeedbackDimension):
    explicit = 1
    implicit = 2


class Actuality(FeedbackDimension):
    observed = 1
    hypothetical = 2


class Relation(FeedbackDimension):
    absolute = 1
    relative = 2


class Content(FeedbackDimension):
    instance = 1
    feature = 2


class Granularity(FeedbackDimension):
    state = 1
    segment = 2
    episode = 3
    entire = 4


class Target(BaseModel):
    id: int = -1
    origin: str = "replay"
    timestamp: int = -1


class Episode(Target):
    reference: EpisodeID = None


class State(Target):
    reference: EpisodeID = None
    step: int = -1


class Segment(Target):
    reference: EpisodeID = None
    start: int = -1
    end: int = -1


class StandardizedFeedbackType(BaseModel):
    intention: Intention = Intention.evaluate
    actuality: Actuality = Actuality.observed
    relation: Relation = Relation.relative
    content: Content = Content.instance
    granularity: Granularity = Granularity.episode


class Evaluation(BaseModel):
    rating: Union[float, int, List[float], List[int]] = None
    comparison: Union[float, int, List[float], List[int]] = None


class Instruction(BaseModel):
    demonstration: Union[Episode, State, Segment] = None
    correction: Union[Episode, State, Segment] = None
    goal: Union[Episode, State, Segment] = None
    optimalilty: Union[float, List[float]] = None


class Description(BaseModel):
    feature_selection: List[dict] = None
    feature_importance: Union[float, List[float]] = None
    feature_ranking: Union[float, List[float]] = None


class StandardizedFeedback(BaseModel):
    feedback_id: int = -1
    feedback_timestamp: int = -1
    feedback_type: StandardizedFeedbackType = StandardizedFeedbackType()
    content: Union[Evaluation, Instruction, Description] = None


class AbsoluteFeedback(StandardizedFeedback):
    episode_id: EpisodeID
    target: Union[Episode, State, Segment] = None


class RelativeFeedback(StandardizedFeedback):
    episode_ids: List[EpisodeID] = []
    target: List[Union[Episode, State, Segment]] = []
