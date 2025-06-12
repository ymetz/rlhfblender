# feedback_models.py
from typing import Union, Literal, Optional, List
from pydantic import BaseModel, Field
from enum import Enum

# Keep your existing Origin enum
class Origin(Enum):
    offline = 1
    online = 2
    generated = 3

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
    
class FeedbackType(Enum):
    rating = "evaluative"
    comparison = "comparative"
    ranking = "comparative"
    demonstration = "demonstrative"
    correction = "corrective"
    goal = "goal"
    featureSelection = "featureSelection"
    descriptivePreferences = "descriptivePreferences"
    text = "text"
    meta = "meta"
    other = "other"

    def __str__(self):
        return self.name
    
class UnprocessedFeedback(BaseModel):
    """
    This is the feedback we get from the user when evaluating an episode.
    It's a superset of the feedback types we support, potentially containing different feedback content.
    Needs to be translated into a standardized format.
    """

    session_id: str = ""
    feedback_type: FeedbackType = FeedbackType.rating

    targets: list[dict] = []
    granularity: str = "episode"
    timestamp: int = -1

    # Evaluative feedback content
    score: float | None = 0.0  # e.g.: 0.5
    preferences: list[int] | None = []  # e.g.: [1, 1, 2, 3, 4] for a partial ordering

    # Instructional feedback content
    action: int | list[float] | None = None
    state: dict | None = None
    action_preferences: list[int] | list[list[float]] | None = None
    state_preferences: list[dict] | None = None

    # Demo feedback is handled separately
    is_demo: bool = False
    demo_preferences: list[int] | None = None

    # Descriptive feedback content
    feature_selection: list[dict] | None | str = None
    feature_importance: float | list[float] | None | str = None
    feature_selections_preferences: list[list[dict]] | None = None
    feature_importance_preferences: list[float | list[float]] | None = None

    # Text feedback content
    text_feedback: str = ""  # e.g.: "The agent is doing well in the beginning, but then it fails to collect the key."

    # Meta information
    user_id: int = -1
    meta_action: str = ""  # e.g.: "skip", "submit", "next", "back", "start", "end"


# Base Target class
class Target(BaseModel):
    target_id: str = ""
    origin: Origin = Origin.offline
    timestamp: int = -1
    # Store the actual trajectory data
    data: Optional[List[tuple]] = None  # List of (obs, action, reward) tuples
    
    # Episode reference information (replaces string reference)
    env_name: str = ""
    benchmark_type: str = ""
    benchmark_id: int = -1
    checkpoint_step: int = -1
    episode_num: int = -1

class Episode(Target):
    # All episode info is now in the base Target class
    pass
    
class State(Target):
    step: int = -1  # Which step within the episode

class Segment(Target):
    start: int = -1  # Start step
    end: int = -1    # End step

class Entire(Target):
    # No target-specific reference necessary
    pass

# Content types for different feedback
class FeedbackContent(BaseModel):
    """Base class for all feedback content"""
    pass

class Rating(FeedbackContent):
    score: float
    
class Ranking(FeedbackContent):
    preferences: List[float]
    
class Correction(FeedbackContent):
    action_preferences: List[int]
    
class Demonstration(FeedbackContent):
    actions: List[Union[int, List[float]]] = Field(default_factory=list)
    
class FeatureSelection(FeedbackContent):
    features: Union[List[dict], str]
    importance: Optional[Union[float, List[float], str]] = None
    
class TextFeedback(FeedbackContent):
    text: str

class MetaFeedback(FeedbackContent):
    action: str  # "skip", "submit", etc.

# Simplified Feedback Type
class SimplifiedFeedbackType(str, Enum):
    rating = "rating"
    ranking = "ranking"
    comparison = "comparison"
    correction = "correction"
    demonstration = "demonstration"
    feature_selection = "feature_selection"
    text = "text"
    meta = "meta"

# Unified Feedback Model
class StandardizedFeedback(BaseModel):
    feedback_id: int
    timestamp: int
    session_id: str
    
    # Simplified type system
    feedback_type: SimplifiedFeedbackType
    
    # Single or multiple targets based on feedback type
    targets: List[Target]
    
    # Content is a discriminated union
    content: Union[
        Rating, Ranking, Correction, 
        Demonstration, FeatureSelection, 
        TextFeedback, MetaFeedback
    ]
    
    # Optional metadata that can be inferred or specified
    granularity: Literal["state", "segment", "episode", "entire"] = "episode"
    
    @property
    def is_relative(self) -> bool:
        """Infer if feedback is relative based on type"""
        return self.feedback_type in [
            SimplifiedFeedbackType.ranking,
            SimplifiedFeedbackType.comparison,
            SimplifiedFeedbackType.correction
        ]
    
    @property
    def is_absolute(self) -> bool:
        return not self.is_relative
    
    @property
    def is_hypothetical(self) -> bool:
        """Infer if feedback is hypothetical"""
        return self.feedback_type == SimplifiedFeedbackType.demonstration