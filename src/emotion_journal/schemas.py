from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from .config import MAX_TEXT_LENGTH

FeedbackValue = Literal["helpful", "not_helpful", "unsure"]
ConfidenceBand = Literal["high", "medium", "low"]
ResourceAction = Literal["opened", "helpful", "dismissed"]
ResourceType = Literal["video", "website", "game", "support"]
CopingStyle = Literal["watch", "read", "play", "move"]


class CoachSummary(BaseModel):
    turn_count: int = 0
    final_step: Optional[str] = None
    framing_emotion: Optional[str] = None
    selected_coping_style: Optional[CopingStyle] = None
    resource_ids: List[str] = Field(default_factory=list)
    used_llm: bool = False
    safety_mode: bool = False


class JournalInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH)
    location: Optional[str] = None
    activity: Optional[str] = None

    @field_validator("text")
    @classmethod
    def clean_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("text must not be empty")
        return stripped

    @field_validator("location", "activity")
    @classmethod
    def clean_optional(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


class JournalEntryCreate(JournalInput):
    feedback: Optional[FeedbackValue] = None
    coach_summary: Optional[CoachSummary] = None


class FeedbackUpdate(BaseModel):
    feedback: FeedbackValue


class ResourceCard(BaseModel):
    id: str
    title: str
    url: str
    resource_type: ResourceType
    coping_style: CopingStyle
    provider: str
    embed_kind: str
    duration_minutes: Optional[int] = None
    summary: str
    emotion_tags: List[str] = Field(default_factory=list)
    tone_tags: List[str] = Field(default_factory=list)
    is_browser_safe: bool = True
    is_crisis_safe: bool = False


class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    emotion: str
    confidence: float
    recommendation: str
    disclaimer: str
    is_crisis: bool
    scores: Dict[str, float]
    support_message: Optional[str] = None
    model_name: Optional[str] = None
    confidence_band: Optional[ConfidenceBand] = None
    reflection_summary: Optional[str] = None
    interpretation: Optional[str] = None
    follow_up_prompts: List[str] = Field(default_factory=list)
    explanation_phrases: List[str] = Field(default_factory=list)
    resources: List[ResourceCard] = Field(default_factory=list)
    suggested_resource_ids: List[str] = Field(default_factory=list)
    coach_opening: Optional[str] = None
    coach_state: Dict[str, object] = Field(default_factory=dict)
    suggested_replies: List[str] = Field(default_factory=list)
    coach_available: bool = False


class JournalEntryResponse(PredictionResponse):
    id: int
    created_at: str
    text: str
    location: Optional[str] = None
    activity: Optional[str] = None
    feedback: Optional[FeedbackValue] = None
    coach_state_summary: Optional[str] = None
    coach_summary: Optional[CoachSummary] = None


class EntriesResponse(BaseModel):
    entries: List[JournalEntryResponse]


class AnalyticsResponse(BaseModel):
    total_entries: int
    counts_by_emotion: Dict[str, int]
    trend_buckets: List[Dict[str, object]]
    feedback_counts: Dict[str, int]
    feedback_usefulness_rate: Optional[float] = None
    confidence_band_counts: Dict[str, int] = Field(default_factory=dict)
    top_explanation_phrases_by_emotion: Dict[str, List[Dict[str, object]]] = Field(default_factory=dict)
    resource_action_counts: Dict[str, int] = Field(default_factory=dict)
    top_helpful_resources: List[Dict[str, object]] = Field(default_factory=list)
    preferred_coping_styles: Dict[str, int] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    status: str
    model_ready: bool
    llm_mode_available: bool
    db_path: str


class ResourcesResponse(BaseModel):
    resources: List[ResourceCard]


class ResourceSummaryResponse(BaseModel):
    total_resources: int
    counts_by_emotion: Dict[str, int]
    counts_by_coping_style: Dict[str, int]
    counts_by_type: Dict[str, int]
    crisis_safe_count: int
    coverage_gaps: List[Dict[str, str]] = Field(default_factory=list)
    validation_errors: List[str] = Field(default_factory=list)


class ResourceInteractionCreate(BaseModel):
    resource_id: str
    action: ResourceAction
    emotion: str
    entry_id: Optional[int] = None


class ResourceInteractionResponse(BaseModel):
    id: int
    created_at: str
    resource_id: str
    action: ResourceAction
    emotion: str
    entry_id: Optional[int] = None


class CoachTurnRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH)
    emotion: str
    confidence_band: Optional[ConfidenceBand] = None
    user_message: str = Field(..., min_length=1)
    coach_state: Dict[str, object] = Field(default_factory=dict)
    is_crisis: bool = False
    use_llm: bool = False


class CoachTurnResponse(BaseModel):
    assistant_message: str
    coach_state: Dict[str, object]
    suggested_replies: List[str] = Field(default_factory=list)
    resource_ids: List[str] = Field(default_factory=list)
    used_llm: bool = False
