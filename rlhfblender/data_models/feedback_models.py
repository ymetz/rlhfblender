import enum
from typing import List, Union

from pydantic import BaseModel

from rlhfblender.data_models.global_models import EpisodeID


class FeedbackType(enum.Enum):
    """
    This is the type of feedback we get from the user when evaluating an episode.
    """

    rating = "evaluative"
    comparison = "comparative"
    ranking = "comparative"
    demonstration = "demonstrative"
    correction = "corrective"
    goal = "goal"
    featureSelection = "featureSelection"
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


class UnprocessedFeedback(BaseModel):
    """
    This is the feedback we get from the user when evaluating an episode.
    It's a superset of the feedback types we support, potentially containing different feedback content.
    Needs to be translated into a standardized format.
    """

    session_id: str = ""
    feedback_type: FeedbackType = FeedbackType.rating

    targets: List[dict] = []
    granularity: str = "episode"
    timestamp: int = -1
    text_feedback: str = ""  # e.g.: "The agent is doing well in the beginning, but then it fails to collect the key."

    # Evaluative feedback content
    score: float | None = 0.0  # e.g.: 0.5
    preferences: List[int] | None = []  # e.g.: [1, 1, 2, 3, 4] for a partial ordering

    # Instructional feedback content
    action: int | List[float] | None = None
    state: dict | None = None
    action_preferences: List[int] | List[List[float]] | None = None
    state_preferences: List[dict] | None = None

    # Demo feedback is handled separately
    is_demo: bool = False
    demo_preferences: Union[List[int], None] = None

    # Descriptive feedback content
    feature_selection: List[dict] | None | str = None
    feature_importance: float | List[float] | None | str = None
    feature_selections_preferences: List[List[dict]] | None = None
    feature_importance_preferences: List[float | List[float]] | None = None

    # Meta information
    user_id: int = -1


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


class Origin(enum.Enum):
    # This is the target origin. Offline and online targets both are observed, generated targets materialize by human
    # input. Therefore, online/offline correspond to observed and generated to hypothetical feedback.
    offline = 1
    online = 2
    generated = 3

    def __str__(self):
        # just return the enum value
        return self.name

    def __repr__(self):
        # just return the enum value
        return self.name


class Target(BaseModel):
    target_id: str = ""
    origin: Origin = Origin.offline
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


class Entire(Target):
    # No target-specific reference necessary (except potentially the model/agent)
    pass


class StandardizedFeedbackType(BaseModel):
    intention: Intention = Intention.evaluate
    actuality: Actuality = Actuality.observed
    relation: Relation = Relation.relative
    content: Content = Content.instance
    granularity: Granularity = Granularity.episode

    # hash function
    def __hash__(self):
        return hash(
            (
                self.intention,
                self.actuality,
                self.relation,
                self.content,
                self.granularity,
            )
        )


class Evaluation(BaseModel):
    # A scalar value representing the evaluative value for a given target
    score: float = None


class RelativeEvaluation(BaseModel):
    # An array of preference values (gives as a ranking), needs to match the list of targets,
    # we assume a consistent partial ordering
    preferences: List[float] = None


class Instruction(BaseModel):
    # An instruction might either be an action or a goal
    action: Union[int, List[float]] = None
    goal: dict = None


class RelativeInstruction(Instruction):
    # A relative instruction is a preference over actions or goals
    action_preferences: List[int] = None
    goal_preferences: List[dict] = None


class Description(BaseModel):
    feature_selection: List[dict] | str = None  # A list of feature selections or a file path as a string
    feature_importance: Union[float, List[float]] | str = None  # A list of feature importances or a file path as a string


class RelativeDescription(Description):
    # A relative description is a preference over feature selections, importances, or rankings
    feature_selections_preferences: List[List[dict]] = None
    feature_importance_preferences: List[Union[float, List[float]]] = None


class StandardizedFeedback(BaseModel):
    feedback_id: int = -1
    feedback_timestamp: int = -1
    feedback_type: StandardizedFeedbackType = StandardizedFeedbackType()


class AbsoluteFeedback(StandardizedFeedback):
    target: Target = None
    content: Union[Evaluation, Instruction, Description] = None


class RelativeFeedback(StandardizedFeedback):
    target: List[Target] = None
    content: Union[RelativeEvaluation, RelativeInstruction, RelativeDescription] = None


def get_target(target: dict, granularity: str) -> Target | None:
    if granularity == "episode":
        return Episode(
            target_id=target["target_id"],
            reference=target["reference"],
            origin=get_origin(target["origin"]),
            timestamp=target["timestamp"],
        )
    elif granularity == "state":
        return State(
            target_id=target["target_id"],
            reference=target["reference"],
            origin=get_origin(target["origin"]),
            timestamp=target["timestamp"],
            step=target["step"],
        )
    elif granularity == "segment":
        return Segment(
            target_id=target["target_id"],
            reference=target["reference"],
            origin=get_origin(target["origin"]),
            timestamp=target["timestamp"],
            start=target["start"],
            end=target["end"],
        )
    elif granularity == "entire":
        return Entire(
            target_id=target["target_id"],
            origin=get_origin(target["origin"]),
            timestamp=target["timestamp"],
        )
    return None


def get_granularity(granularity: str) -> Granularity:
    if granularity == "episode":
        return Granularity.episode
    elif granularity == "state":
        return Granularity.state
    elif granularity == "segment":
        return Granularity.segment
    elif granularity == "entire":
        return Granularity.entire
    return Granularity.episode


def get_origin(origin: str) -> Origin:
    if origin == "offline":
        return Origin.offline
    elif origin == "online":
        return Origin.online
    elif origin == "generated":
        return Origin.generated
    return Origin.offline
