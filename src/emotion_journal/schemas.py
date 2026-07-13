from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from .config import MAX_TEXT_LENGTH

FeedbackValue = Literal["helpful", "not_helpful", "unsure"]
ConfidenceBand = Literal["high", "medium", "low"]
ResourceAction = Literal["opened", "helpful", "dismissed"]
ResourceType = Literal["video", "website", "game", "support"]
CopingStyle = Literal["watch", "read", "play", "move"]
SourceTier = Literal["official", "nonprofit", "educational", "activity", "crisis_support"]
ResourceIntent = Literal[
    "ground",
    "plan",
    "reframe",
    "connect",
    "watch",
    "read",
    "play",
    "move",
    "resource",
    "tips",
    "none",
]
CoachMode = Literal["deterministic", "rewrite", "structured"]
AgentMode = Literal["deterministic", "structured", "fallback"]
ResourceSource = Literal["catalog", "ai_suggested"]
GeneratedResourceType = Literal["video", "website", "game"]
CoreEmotion = Literal["sadness", "joy", "love", "anger", "fear", "surprise"]
EmotionTag = Literal[
    "anxiety",
    "overwhelm",
    "grief",
    "loneliness",
    "shame",
    "guilt",
    "burnout",
    "frustration",
    "boundary",
    "disappointment",
    "relief",
    "gratitude",
    "pride",
    "confusion",
    "hope",
    "connection",
    "work_stress",
    "self_advocacy",
]


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
    use_llm: bool = False

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
    goal_tags: List[str] = Field(default_factory=list)
    source_tier: Optional[SourceTier] = None
    reviewed_at: Optional[str] = None
    rationale: Optional[str] = None
    source: ResourceSource = "catalog"


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
    secondary_emotions: List[str] = Field(default_factory=list)
    emotion_tags: List[str] = Field(default_factory=list)
    top_margin: Optional[float] = None
    is_mixed: bool = False
    uncertainty_reason: Optional[str] = None
    calibration_notes: List[str] = Field(default_factory=list)
    classifier_mode: str = "calibrated"
    classifier_source: str = "artifact"
    classifier_fallback_reason: Optional[str] = None
    resources: List[ResourceCard] = Field(default_factory=list)
    suggested_resource_ids: List[str] = Field(default_factory=list)
    used_llm_recommender: bool = False
    generated_resource_count: int = 0
    recommender_fallback_reason: Optional[str] = None
    recommender_model: Optional[str] = None
    coach_opening: Optional[str] = None
    coach_state: Dict[str, object] = Field(default_factory=dict)
    suggested_replies: List[str] = Field(default_factory=list)
    tips: List[str] = Field(default_factory=list)
    practical_steps: List[str] = Field(default_factory=list)
    reflection_question: Optional[str] = None
    communication_draft: Optional[str] = None
    confidence_note: Optional[str] = None
    used_llm: bool = False
    coach_mode: CoachMode = "deterministic"
    agent_mode: AgentMode = "deterministic"
    agent_model: Optional[str] = None
    agent_fallback_reason: Optional[str] = None
    fallback_reason: Optional[str] = None
    resource_intent: Optional[ResourceIntent] = None
    resource_rationales: Dict[str, str] = Field(default_factory=dict)
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


class ReadinessResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    status: str
    app_environment: str
    deployment_mode: str
    model_ready: bool
    database_ready: bool
    resources_ready: bool
    model_name: Optional[str] = None
    model_artifact_source: str = "local"
    resource_count: int = 0
    db_path: str
    checks: Dict[str, str] = Field(default_factory=dict)


class ResourcesResponse(BaseModel):
    resources: List[ResourceCard]


class ResourceSummaryResponse(BaseModel):
    total_resources: int
    counts_by_emotion: Dict[str, int]
    counts_by_coping_style: Dict[str, int]
    counts_by_type: Dict[str, int]
    counts_by_source_tier: Dict[str, int] = Field(default_factory=dict)
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
    tips: List[str] = Field(default_factory=list)
    practical_steps: List[str] = Field(default_factory=list)
    reflection_question: Optional[str] = None
    communication_draft: Optional[str] = None
    confidence_note: Optional[str] = None
    used_llm: bool = False
    coach_mode: CoachMode = "deterministic"
    agent_mode: AgentMode = "deterministic"
    agent_model: Optional[str] = None
    agent_fallback_reason: Optional[str] = None
    resource_intent: Optional[ResourceIntent] = None
    resource_rationales: Dict[str, str] = Field(default_factory=dict)
    fallback_reason: Optional[str] = None


class StructuredCoachDraft(BaseModel):
    assistant_message: str = Field(..., min_length=1, max_length=520)
    tips: List[str] = Field(default_factory=list, max_length=3)
    practical_steps: List[str] = Field(default_factory=list, max_length=4)
    suggested_replies: List[str] = Field(default_factory=list, max_length=4)
    resource_intent: ResourceIntent = "none"
    resource_ids: List[str] = Field(default_factory=list, max_length=6)
    reflection_question: Optional[str] = Field(default=None, max_length=240)
    communication_draft: Optional[str] = Field(default=None, max_length=520)
    confidence_note: Optional[str] = Field(default=None, max_length=240)
    refusal_reason: Optional[str] = Field(default=None, max_length=220)

    @field_validator("assistant_message")
    @classmethod
    def clean_message(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("assistant_message must not be empty")
        banned_terms = ("diagnose", "diagnosis", "treatment plan", "therapy session")
        lowered = stripped.lower()
        if any(term in lowered for term in banned_terms):
            raise ValueError("assistant_message must stay non-clinical")
        return stripped

    @field_validator("reflection_question", "communication_draft", "confidence_note", "refusal_reason")
    @classmethod
    def clean_optional_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            return None
        banned_terms = ("diagnose", "diagnosis", "treatment plan", "therapy session")
        lowered = stripped.lower()
        if any(term in lowered for term in banned_terms):
            raise ValueError("agent text must stay non-clinical")
        return stripped

    @field_validator("tips", "practical_steps", "suggested_replies", "resource_ids")
    @classmethod
    def clean_string_lists(cls, values: List[str]) -> List[str]:
        cleaned = []
        for value in values:
            stripped = str(value).strip()
            lowered = stripped.lower()
            banned_terms = ("diagnose", "diagnosis", "treatment plan", "therapy session")
            if any(term in lowered for term in banned_terms):
                raise ValueError("agent list text must stay non-clinical")
            if stripped and stripped not in cleaned:
                cleaned.append(stripped)
        return cleaned


class StructuredEmotionDraft(BaseModel):
    primary_emotion: CoreEmotion
    secondary_emotions: List[CoreEmotion] = Field(default_factory=list, max_length=3)
    emotion_tags: List[EmotionTag] = Field(default_factory=list, max_length=6)
    intensity: float = Field(default=0.5, ge=0, le=1)
    confidence: ConfidenceBand = "medium"
    is_mixed: bool = False
    themes: List[str] = Field(default_factory=list, max_length=5)
    rationale: str = Field(..., min_length=1, max_length=380)
    refusal_reason: Optional[str] = Field(default=None, max_length=220)

    @field_validator("secondary_emotions")
    @classmethod
    def dedupe_secondary(cls, values: List[str]) -> List[str]:
        cleaned = []
        for value in values:
            if value not in cleaned:
                cleaned.append(value)
        return cleaned[:3]

    @field_validator("emotion_tags", "themes")
    @classmethod
    def clean_short_lists(cls, values: List[str]) -> List[str]:
        cleaned = []
        for value in values:
            stripped = str(value).strip()
            if stripped and stripped not in cleaned:
                cleaned.append(stripped)
        return cleaned


_CLINICAL_TERMS = ("diagnose", "diagnosis", "treatment plan", "therapy session", "medication")


class StructuredResourceSuggestion(BaseModel):
    """A single LLM-generated resource idea, validated before it reaches users."""

    title: str = Field(..., min_length=1, max_length=140)
    url: str = Field(..., min_length=4, max_length=400)
    resource_type: GeneratedResourceType = "website"
    coping_style: CopingStyle = "read"
    provider: str = Field(default="", max_length=120)
    summary: str = Field(..., min_length=1, max_length=320)
    why: str = Field(..., min_length=1, max_length=280)
    goal_tags: List[str] = Field(default_factory=list, max_length=4)

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped.startswith(("http://", "https://")):
            raise ValueError("resource url must be an http(s) link")
        return stripped

    @field_validator("title", "summary", "why", "provider")
    @classmethod
    def clean_text(cls, value: str) -> str:
        stripped = (value or "").strip()
        lowered = stripped.lower()
        if any(term in lowered for term in _CLINICAL_TERMS):
            raise ValueError("suggestion text must stay non-clinical")
        return stripped

    @field_validator("goal_tags")
    @classmethod
    def clean_goal_tags(cls, values: List[str]) -> List[str]:
        cleaned = []
        for value in values:
            stripped = str(value).strip().lower()
            if stripped and stripped not in cleaned:
                cleaned.append(stripped)
        return cleaned


class StructuredResourceRecommendation(BaseModel):
    """Full LLM recommendation payload: a re-ranking of the catalog plus fresh ideas."""

    ranked_catalog_ids: List[str] = Field(default_factory=list, max_length=12)
    generated: List[StructuredResourceSuggestion] = Field(default_factory=list, max_length=5)
    refusal_reason: Optional[str] = Field(default=None, max_length=220)

    @field_validator("ranked_catalog_ids")
    @classmethod
    def dedupe_ids(cls, values: List[str]) -> List[str]:
        cleaned = []
        for value in values:
            stripped = str(value).strip()
            if stripped and stripped not in cleaned:
                cleaned.append(stripped)
        return cleaned
