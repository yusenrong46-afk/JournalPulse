from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .activity_resources import activity_resource_matches_constraints
from .config import CHAT_PROVIDER_BUDGET_SECONDS, STREAM_READ_GRACE_SECONDS, Settings
from .domain import FEELINGS, AffectiveState, ModelRun, ReflectionCopy
from .guided_action import (
    GUIDED_ACTION_JSON_SCHEMA,
    GUIDED_ACTION_PROMPT_VERSION,
    GUIDED_ACTION_SYSTEM_PROMPT,
    ActivityDirective,
    GuidedActionContext,
    load_guided_action_skill,
)
from .http_clients import managed_http_client

logger = logging.getLogger(__name__)

_DIAGNOSTIC_FIELDS = {
    "reply", "offer_action", "resource_intent", "card_reason", "summary", "feelings",
    "model", "provider", "latency_ms", "prompt_tokens", "completion_tokens", "schema_valid",
    "used_fallback", "fallback_reason", "prompt_version",
    "activity", "move", "goal", "selected_resource_id", "constraints", "search_topic",
    "skill_version", "skill_hash",
}
_DIAGNOSTIC_ERROR_TYPES = {
    "json_invalid", "missing", "extra_forbidden", "literal_error", "string_type",
    "string_too_short", "string_too_long", "list_type", "too_long", "bool_type",
    "int_parsing", "int_type", "greater_than_equal", "value_error", "model_type", "dict_type",
    "invalid_activity_operation",
}
_DIAGNOSTIC_STAGES = {
    "provider_json", "envelope", "content_format", "output_schema", "output_semantics",
    "usage_metadata", "completion_truncated", "provider_transport", "upstream_error", "upstream_http",
    "provider_refusal",
}


@dataclass(frozen=True)
class CompletionDiagnostic:
    """Bounded error categories that may leave the server; never rejected values."""

    stage: str
    errors: tuple[tuple[str, str], ...] = ()
    content_kind: str | None = None
    finish_reason: str | None = None
    provider: str | None = None

    @property
    def headers(self) -> dict[str, str]:
        # Live log access can fail. Authenticated callers can still distinguish
        # routing failures from schema failures without seeing provider text.
        headers = {"X-JournalPulse-Error-Stage": (
            self.stage if self.stage in _DIAGNOSTIC_STAGES else "provider_response"
        )}
        if self.errors:
            categories = []
            for field, error_type in self.errors[:5]:
                safe_field = field if field in _DIAGNOSTIC_FIELDS or field == "root" else "unrecognized_field"
                safe_type = error_type if (
                    error_type in _DIAGNOSTIC_ERROR_TYPES or error_type == "nonblank_reason_required"
                ) else "validation_error"
                categories.append(f"{safe_field}.{safe_type}")
            headers["X-JournalPulse-Error-Fields"] = ",".join(categories)
        for name, value, allowed in [
            ("Content", self.content_kind, {"empty", "markdown_fenced", "json_like", "plain_text"}),
            ("Finish", self.finish_reason, {"stop", "length", "content_filter", "tool_calls", "error"}),
            ("Provider", self.provider, {"Azure", "OpenAI", "Amazon Bedrock", "openrouter"}),
        ]:
            if value is not None:
                headers[f"X-JournalPulse-Error-{name}"] = value if value in allowed else "other"
        return headers


def _log_completion_rejection(
    stage: str, exc: Exception, *, content_kind: str | None = None,
    finish_reason: str | None = None, provider: str | None = None,
) -> CompletionDiagnostic:
    """Log fixed validation metadata, never provider text, input, context or traceback.

    Pydantic's normal error strings contain rejected values. Even an unknown
    field name can be private model output, so only known field names survive.
    """
    errors = []
    if isinstance(exc, ValidationError):
        for error in exc.errors(include_url=False, include_context=False, include_input=False)[:5]:
            location = error.get("loc", ())
            field = location[0] if location else "root"
            if field != "root" and field not in _DIAGNOSTIC_FIELDS:
                field = "unrecognized_field"
            error_type = error.get("type")
            errors.append({
                "field": field,
                "type": error_type if error_type in _DIAGNOSTIC_ERROR_TYPES else "validation_error",
            })
    elif isinstance(exc, ActivitySemanticError):
        errors.append({"field": "activity", "type": "invalid_activity_operation"})
    elif stage == "output_semantics":
        errors.append({"field": "card_reason", "type": "nonblank_reason_required"})
    diagnostic = {
        "stage": stage, "exception_type": type(exc).__name__,
        "prompt_version": CONVERSATION_PROMPT_VERSION, "errors": errors,
    }
    logger.warning("luna_completion_rejected %s", json.dumps(diagnostic, sort_keys=True))
    return CompletionDiagnostic(
        stage, tuple((error["field"], error["type"]) for error in errors),
        content_kind, finish_reason, provider,
    )


def _content_kind(text: str) -> str:
    """Classify syntax shape without recording words, length or a private-text digest."""
    stripped = text.lstrip()
    if not stripped:
        return "empty"
    if stripped.startswith("```"):
        return "markdown_fenced"
    return "json_like" if stripped.startswith(("{", "[")) else "plain_text"


class UnsupportedProviderResponse(Exception):
    """Provider message.content was not a string or a text-part array."""


class ConversationProviderError(Exception):
    """The conversation model failed, or its completion cannot be shown."""

    def __init__(
        self, message: str, *, status_code: int = 502, diagnostic: CompletionDiagnostic | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.diagnostic = diagnostic

    @property
    def diagnostic_headers(self) -> dict[str, str]:
        return self.diagnostic.headers if self.diagnostic else {}


MAX_MODEL_RESPONSE_BYTES = 256_000
# A retry shorter than this cannot plausibly return a full Luna reply, so it is not started.
MIN_PROVIDER_ATTEMPT_SECONDS = 5.0


def _bounded_provider_post(
    client: httpx.Client, url: str, *, timeout: float, **kwargs: Any
) -> httpx.Response:
    """Bound decoded response bytes and elapsed streaming time for fixed provider calls.

    A per-read timeout alone can be renewed by a slowly trickled response. Check
    elapsed time between chunks as well; an in-flight read can add at most 5s.
    Redirects are disabled so provider credentials cannot follow another host.
    """
    deadline = time.monotonic() + timeout
    # Connecting and sending happen before the first chunk check, so cap them separately;
    # otherwise each phase could spend the whole attempt deadline on its own.
    setup = min(timeout, 10)
    with client.stream(
        "POST", url,
        timeout=httpx.Timeout(timeout, connect=setup, write=setup, pool=setup, read=min(timeout, 5)),
        follow_redirects=False, **kwargs,
    ) as response:
        content = bytearray()
        for chunk in response.iter_bytes():
            if time.monotonic() > deadline:
                raise httpx.ReadTimeout("Provider response deadline exceeded")
            if len(content) + len(chunk) > MAX_MODEL_RESPONSE_BYTES:
                raise httpx.DecodingError("Provider response size limit exceeded")
            content.extend(chunk)
        # iter_bytes() is decoded already; omitting encoding headers avoids a
        # second decompression when creating the bounded in-memory response.
        return httpx.Response(
            response.status_code, content=bytes(content), request=response.request,
        )


class StructuredReflection(BaseModel):
    valence: float = Field(ge=-1.0, le=1.0)
    arousal: float = Field(ge=0.0, le=1.0)
    agency: float = Field(ge=0.0, le=1.0)
    emotion_tags: list[str] = Field(max_length=6)
    confidence: float = Field(ge=0.0, le=1.0)
    uncertainty: str | None = Field(default=None, max_length=240)
    summary: str = Field(min_length=1, max_length=420)
    interpretation: str = Field(min_length=1, max_length=420)
    reflection_question: str = Field(min_length=1, max_length=240)
    resource_intent: str = Field(min_length=1, max_length=40)


REFLECTION_JSON_SCHEMA: dict[str, Any] = {
    "name": "journalpulse_reflection",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "valence",
            "arousal",
            "agency",
            "emotion_tags",
            "confidence",
            "uncertainty",
            "summary",
            "interpretation",
            "reflection_question",
            "resource_intent",
        ],
        "properties": {
            "valence": {"type": "number", "minimum": -1, "maximum": 1},
            "arousal": {"type": "number", "minimum": 0, "maximum": 1},
            "agency": {"type": "number", "minimum": 0, "maximum": 1},
            "emotion_tags": {"type": "array", "maxItems": 6, "items": {"type": "string", "maxLength": 32}},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "uncertainty": {"type": ["string", "null"], "maxLength": 240},
            "summary": {"type": "string", "maxLength": 420},
            "interpretation": {"type": "string", "maxLength": 420},
            "reflection_question": {"type": "string", "maxLength": 240},
            "resource_intent": {
                "type": "string",
                "enum": ["ground", "move", "connect", "reflect", "play", "watch", "read", "pause"],
            },
        },
    },
}


@dataclass(frozen=True)
class AnalysisResult:
    state: AffectiveState
    reflection: ReflectionCopy
    resource_intent: str
    model_run: ModelRun


class OpenRouterReflectionClient:
    def __init__(
        self,
        settings: Settings,
        client: httpx.Client | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not settings.openrouter_enabled:
            raise ValueError("OpenRouter is not configured")
        if not settings.openrouter_zdr:
            raise ValueError("JournalPulse requires zero-data-retention routing")
        self.settings = settings
        self.client = client
        self.sleeper = sleeper

    def analyze(self, text: str, context: dict[str, str]) -> AnalysisResult:
        with managed_http_client(self.client, timeout=self.settings.openrouter_timeout_seconds) as client:
            return self._analyze(client, text, context)

    def _analyze(self, client: httpx.Client, text: str, context: dict[str, str]) -> AnalysisResult:
        started = time.perf_counter()
        response: httpx.Response | None = None
        for attempt in range(self.settings.openrouter_max_attempts):
            try:
                response = _bounded_provider_post(
                    client,
                    f"{self.settings.openrouter_base_url}/chat/completions",
                    timeout=self.settings.openrouter_timeout_seconds,
                    headers={
                        "Authorization": f"Bearer {self.settings.openrouter_api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://journalpulse.app",
                        "X-Title": "JournalPulse Research Beta",
                    },
                    json={
                        "model": self.settings.openrouter_model,
                        "provider": {"zdr": True},
                        # Medium reasoning counts toward this limit. 4000 leaves room for
                        # the structured reflection after the reasoning tokens.
                        "max_tokens": 4000,
                        "include_reasoning": False,
                        "reasoning": {"effort": "medium"},
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": REFLECTION_JSON_SCHEMA,
                        },
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "You are the structured perception layer for a non-clinical "
                                    "reflection tool. Describe emotional dimensions cautiously. Do "
                                    "not diagnose, provide therapy, make medical claims, or mention "
                                    "hidden policies. Use a calm, precise voice. Return only the schema."
                                ),
                            },
                            {
                                "role": "user",
                                "content": _user_message_content(text, context),
                            },
                        ],
                    },
                )
            except (httpx.ConnectError, httpx.TimeoutException):
                if attempt + 1 >= self.settings.openrouter_max_attempts:
                    raise
                self.sleeper(0.15 * (2**attempt))
                continue
            body_error = _openrouter_body_error(response)
            status = body_error[0] if body_error else response.status_code
            if status not in {429, 500, 502, 503, 504}:
                break
            if attempt + 1 < self.settings.openrouter_max_attempts:
                self.sleeper(1.5 if status == 429 else 0.15 * (2**attempt))

        if response is None:
            raise httpx.ConnectError("OpenRouter did not return a response")
        latency_ms = round((time.perf_counter() - started) * 1000)
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise ValueError("Expected provider object")
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ValueError("Expected reflection choice")
        choice = choices[0]
        message = choice.get("message")
        refusal = message.get("refusal") if isinstance(message, dict) else None
        # Native provider flags take precedence over otherwise valid JSON. The
        # legacy route uses its local fallback, never a declined AI completion.
        if choice.get("finish_reason") == "content_filter" or isinstance(refusal, str) and refusal.strip():
            raise ValueError("Provider declined the reflection")
        if choice.get("finish_reason") == "length":
            raise ValueError("Provider truncated the reflection")
        if not isinstance(message, dict) or "content" not in message:
            raise ValueError("Expected reflection message")
        content = message["content"]
        structured = structured_reflection_from_content(content)
        usage = body.get("usage", {})
        if not isinstance(usage, dict):
            raise ValueError("Expected usage object")
        run = ModelRun(
            model=body.get("model", self.settings.openrouter_model),
            provider=body.get("provider", "openrouter"),
            latency_ms=latency_ms,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            schema_valid=True,
        )
        return AnalysisResult(
            state=AffectiveState(
                valence=structured.valence,
                arousal=structured.arousal,
                agency=structured.agency,
                emotion_tags=structured.emotion_tags,
                confidence=structured.confidence,
                uncertainty=structured.uncertainty,
            ),
            reflection=ReflectionCopy(
                summary=structured.summary,
                interpretation=structured.interpretation,
                reflection_question=structured.reflection_question,
            ),
            resource_intent=structured.resource_intent,
            model_run=run,
        )


def _user_message_content(text: str, context: dict[str, str]) -> str:
    if not context:
        return text
    details = "\n".join(f"{key}: {value}" for key, value in context.items())
    return f"{text}\n\nContext:\n{details}"


def _text_from_content_parts(content: list[Any]) -> str:
    if not content:
        raise UnsupportedProviderResponse("Provider content array was empty")
    chunks: list[str] = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "text" or not isinstance(part.get("text"), str):
            raise UnsupportedProviderResponse("Provider content included a non-text part")
        chunks.append(part["text"])
    return "".join(chunks)


def _openrouter_body_error(response: httpx.Response) -> tuple[int, str] | None:
    """OpenRouter sometimes returns HTTP 200 with an error object and no choices."""
    if response.status_code >= 400:
        return None
    try:
        payload = response.json()
    except (ValueError, RecursionError):
        return None
    if not isinstance(payload, dict) or "choices" in payload:
        return None
    error = payload.get("error")
    message = "OpenRouter did not return a completion."
    code = 502
    if isinstance(error, dict):
        if error.get("message"):
            message = str(error["message"])
        if isinstance(error.get("code"), int):
            code = error["code"]
    return code, message[:500]


def _json_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return _text_from_content_parts(content)
    raise UnsupportedProviderResponse(
        f"Unsupported provider content type: {type(content).__name__}"
    )


def structured_reflection_from_content(content: Any) -> StructuredReflection:
    return StructuredReflection.model_validate_json(_json_text_from_content(content))


def deterministic_reflection(text: str, state: AffectiveState | None = None) -> AnalysisResult:
    state = state or AffectiveState(
        valence=0.0,
        arousal=0.5,
        agency=0.5,
        emotion_tags=["unclassified"],
        confidence=0.0,
        uncertainty="The language model was not used; adjust this state before continuing.",
    )
    return AnalysisResult(
        state=state,
        reflection=ReflectionCopy(
            summary="You captured a moment that still deserves a little attention.",
            interpretation=(
                "The automated interpretation is unavailable, so your own state check is the source of truth."
            ),
            reflection_question="What would feel meaningfully different after one small action?",
        ),
        resource_intent="reflect",
        model_run=ModelRun(
            model="deterministic-fallback",
            latency_ms=0,
            schema_valid=True,
            used_fallback=True,
            fallback_reason="llm_not_used",
        ),
    )


def safe_analyze(
    settings: Settings,
    *,
    text: str,
    context: dict[str, str],
    consent: bool,
    self_report: AffectiveState | None,
    client: OpenRouterReflectionClient | None = None,
) -> AnalysisResult:
    if not consent or not settings.openrouter_enabled:
        return deterministic_reflection(text, self_report)
    try:
        return (client or OpenRouterReflectionClient(settings)).analyze(text, context)
    except UnsupportedProviderResponse:
        raise
    except (httpx.HTTPError, KeyError, ValueError, ValidationError, RecursionError) as exc:
        fallback = deterministic_reflection(text, self_report)
        return AnalysisResult(
            state=fallback.state,
            reflection=fallback.reflection,
            resource_intent=fallback.resource_intent,
            model_run=fallback.model_run.model_copy(
                update={"fallback_reason": f"openrouter_{exc.__class__.__name__}"}
            ),
        )


CONVERSATION_PROMPT_VERSION = GUIDED_ACTION_PROMPT_VERSION
CONVERSATION_SYSTEM_PROMPT = GUIDED_ACTION_SYSTEM_PROMPT

CONVERSATION_JSON_SCHEMA: dict[str, Any] = {
    "name": "journalpulse_conversation_turn",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "reply",
            "offer_action",
            "resource_intent",
            "card_reason",
            "summary",
            "feelings",
        ],
        "properties": {
            "reply": {"type": "string", "minLength": 1, "maxLength": 1200},
            "offer_action": {"type": "boolean"},
            "resource_intent": {
                "type": "string",
                "enum": ["ground", "move", "connect", "reflect", "play", "watch", "read", "pause"],
            },
            "card_reason": {"type": "string", "maxLength": 240},
            "summary": {"type": "string", "minLength": 1, "maxLength": 420},
            "feelings": {
                "type": "array",
                "maxItems": 3,
                "items": {"type": "string", "enum": list(FEELINGS)},
            },
        },
    },
}


class ConversationTurnOutput(BaseModel):
    # Provider-side structured output is a request, not a local trust boundary.
    # Validate it again without coercing readiness flags or accepting invented fields.
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    reply: str = Field(min_length=1, max_length=1200)
    offer_action: bool
    resource_intent: Literal["ground", "move", "connect", "reflect", "play", "watch", "read", "pause"]
    card_reason: str = Field(max_length=240)
    summary: str = Field(min_length=1, max_length=420)
    feelings: list[str] = Field(max_length=3)

    @field_validator("feelings")
    @classmethod
    def known_feelings(cls, value: list[str]) -> list[str]:
        unknown = [item for item in value if item not in FEELINGS]
        if unknown:
            raise ValueError(f"unknown feelings: {unknown}")
        return list(dict.fromkeys(value))

    def require_card_reason(self) -> ConversationTurnOutput:
        if self.offer_action and not self.card_reason.strip():
            raise ValueError("card_reason is required when an action is offered")
        return self


@dataclass(frozen=True)
class ConversationCompletion:
    reply: str
    offer_action: bool
    resource_intent: str
    card_reason: str
    summary: str
    model_run: ModelRun
    feelings: tuple[str, ...] = ()
    # Defaults preserve the old six-field journal contract and simple test clients.
    activity: ActivityDirective | None = None


class GuidedActionTurnOutput(ConversationTurnOutput):
    activity: ActivityDirective

    def require_activity_semantics(self, context: GuidedActionContext) -> GuidedActionTurnOutput:
        choosing = self.activity.selected_resource_id is not None or self.activity.search_topic is not None
        if self.offer_action != choosing:
            raise ActivitySemanticError("Readiness must describe the validated activity operation")
        if choosing and (
            not context.action_allowed
            or context.preference == "listen"
            or context.activity_state in {"active", "paused", "awaiting_report"}
        ):
            raise ActivitySemanticError("A new activity is unavailable in this conversation state")
        if self.activity.selected_resource_id is not None:
            candidates = {item["id"]: item for item in context.candidates}
            selected = candidates.get(self.activity.selected_resource_id)
            if selected is None:
                raise ActivitySemanticError("The selected resource was not offered by the server")
            if not activity_resource_matches_constraints(selected, self.activity.constraints):
                raise ActivitySemanticError("The selected resource does not fit the activity constraints")
        return self


class ActivitySemanticError(ValueError):
    """A fixed diagnostic category, without exposing rejected IDs or model text."""


def build_guided_request(
    settings: Settings, messages: list[dict[str, str]], context: GuidedActionContext,
) -> dict[str, Any]:
    """The application and benchmark share the exact guided provider request.

    Context is escaped user data. Snippets and participant notes never become a
    dynamically assembled system instruction, even when they contain role labels.
    """
    return {
        "model": settings.chat_model,
        "provider": {"zdr": True, "require_parameters": True},
        "max_tokens": 4000,
        "include_reasoning": False,
        "reasoning": {"effort": "medium"},
        "response_format": {"type": "json_schema", "json_schema": GUIDED_ACTION_JSON_SCHEMA},
        "messages": [
            {"role": "system", "content": CONVERSATION_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps({"activity_context": context.prompt_data()},
                                                     ensure_ascii=False)},
            *messages,
        ],
    }


class OpenRouterConversationClient:
    """One Luna turn. There is no canned reply when the provider fails."""

    def __init__(
        self,
        settings: Settings,
        client: httpx.Client | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not settings.openrouter_api_key or not settings.chat_model:
            raise ValueError("OpenRouter is not configured")
        if not settings.openrouter_zdr:
            raise ValueError("JournalPulse requires zero-data-retention routing")
        self.settings = settings
        self.client = client
        self.sleeper = sleeper
        self.clock = clock

    def complete(self, messages: list[dict[str, str]]) -> ConversationCompletion:
        # Parameter set is limited to what OpenRouter lists for openai/gpt-6-luna
        # (models list and endpoints page, 2026-10-04): max_tokens, response_format,
        # structured_outputs, reasoning, include_reasoning. Temperature is not supported.
        # The request asks for medium effort, the model's default, and excludes
        # reasoning text from the reply. Requiring parameter support prevents
        # routing to endpoints that silently ignore the requested output format.
        body = {
            "model": self.settings.chat_model,
            "provider": {"zdr": True, "require_parameters": True},
            "max_tokens": 4000,
            "include_reasoning": False,
            "reasoning": {"effort": "medium"},
            "response_format": {
                "type": "json_schema",
                "json_schema": CONVERSATION_JSON_SCHEMA,
            },
            "messages": [
                {"role": "system", "content": CONVERSATION_SYSTEM_PROMPT},
                *messages,
            ],
        }
        return self._complete_request(body)

    def complete_guided(
        self, messages: list[dict[str, str]], context: GuidedActionContext,
    ) -> ConversationCompletion:
        return self._complete_request(build_guided_request(self.settings, messages, context), context=context)

    def _complete_request(
        self, body: dict[str, Any], *, context: GuidedActionContext | None = None,
    ) -> ConversationCompletion:
        started = time.perf_counter()
        response = self._post(body)
        latency_ms = round((time.perf_counter() - started) * 1000)
        stage = "provider_json"
        content_kind = finish_reason = diagnostic_provider = None
        try:
            payload = response.json()
            stage = "envelope"
            if not isinstance(payload, dict):
                raise ValueError("expected provider object")
            choices = payload.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                raise ValueError("expected completion choice")
            choice = choices[0]
            raw_finish = choice.get("finish_reason")
            finish_reason = raw_finish if isinstance(raw_finish, str) else "other"
            raw_provider = payload.get("provider", "openrouter")
            diagnostic_provider = raw_provider if isinstance(raw_provider, str) else "other"
            if choice.get("finish_reason") == "length":
                diagnostic = _log_completion_rejection("completion_truncated", ValueError())
                raise ConversationProviderError(
                    "The model reply was cut off. Nothing was saved.", diagnostic=diagnostic,
                )
            message = choice["message"]
            if not isinstance(message, dict):
                if finish_reason != "content_filter":
                    raise ValueError("expected completion message")
                message = {}
            refusal = message.get("refusal")
            if finish_reason == "content_filter" or isinstance(refusal, str) and refusal.strip():
                diagnostic = _log_completion_rejection(
                    "provider_refusal", ValueError(),
                    finish_reason=finish_reason, provider=diagnostic_provider,
                )
                raise ConversationProviderError(
                    "Luna's AI provider declined this reply. Your message was not saved.",
                    status_code=422, diagnostic=diagnostic,
                )
            content = message["content"]
            stage = "content_format"
            text = _json_text_from_content(content)
            content_kind = _content_kind(text)
            stage = "output_schema"
            structured = (
                GuidedActionTurnOutput.model_validate_json(text)
                if context is not None else ConversationTurnOutput.model_validate_json(text)
            )
            stage = "output_semantics"
            structured.require_card_reason()
            if isinstance(structured, GuidedActionTurnOutput) and context is not None:
                structured.require_activity_semantics(context)
            stage = "usage_metadata"
            usage = payload.get("usage", {})
            if not isinstance(usage, dict):
                raise ValueError("expected usage object")
            raw_provider = payload.get("provider", "openrouter")
            provider = raw_provider if isinstance(raw_provider, str) else "openrouter"
            model_run = ModelRun(
                model=payload.get("model", self.settings.chat_model),
                provider=provider,
                latency_ms=latency_ms,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                schema_valid=True,
                prompt_version=CONVERSATION_PROMPT_VERSION,
                skill_version=load_guided_action_skill().version,
                skill_hash=load_guided_action_skill().sha256,
            )
        except UnsupportedProviderResponse as exc:
            _log_completion_rejection(stage, exc)
            raise
        except (KeyError, TypeError, ValueError, ValidationError, RecursionError) as exc:
            diagnostic = _log_completion_rejection(
                stage, exc, content_kind=content_kind,
                finish_reason=finish_reason, provider=diagnostic_provider,
            )
            raise ConversationProviderError(
                "The model reply did not match the conversation schema. Nothing was saved.",
                diagnostic=diagnostic,
            ) from exc
        return ConversationCompletion(
            reply=structured.reply,
            offer_action=structured.offer_action,
            resource_intent=structured.resource_intent,
            card_reason=structured.card_reason,
            summary=structured.summary,
            feelings=tuple(structured.feelings),
            model_run=model_run,
            activity=structured.activity if isinstance(structured, GuidedActionTurnOutput) else None,
        )

    def _post(self, body: dict[str, Any]) -> httpx.Response:
        with managed_http_client(self.client, timeout=self.settings.chat_timeout_seconds) as client:
            return self._post_with_client(client, body)

    def _attempt_seconds(self, started: float) -> float:
        left = CHAT_PROVIDER_BUDGET_SECONDS - (self.clock() - started) - STREAM_READ_GRACE_SECONDS
        return min(self.settings.chat_timeout_seconds, left)

    def _may_retry(self, attempt: int, started: float, delay: float) -> bool:
        """Retry only if another attempt still fits the shared per-turn budget."""
        if attempt + 1 >= self.settings.openrouter_max_attempts:
            return False
        return self._attempt_seconds(started) - delay >= MIN_PROVIDER_ATTEMPT_SECONDS

    def _post_with_client(self, client: httpx.Client, body: dict[str, Any]) -> httpx.Response:
        response: httpx.Response | None = None
        started = self.clock()
        for attempt in range(self.settings.openrouter_max_attempts):
            try:
                response = _bounded_provider_post(
                    client,
                    f"{self.settings.openrouter_base_url}/chat/completions",
                    timeout=self._attempt_seconds(started),
                    headers={
                        "Authorization": f"Bearer {self.settings.openrouter_api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://journalpulse.app",
                        "X-Title": "JournalPulse Research Beta",
                    },
                    json=body,
                )
            except (httpx.HTTPError, OSError) as exc:
                response = None
                delay = 0.15 * (2**attempt)
                if not self._may_retry(attempt, started, delay):
                    raise ConversationProviderError(
                        "Luna did not respond in time. Nothing was saved.",
                        status_code=503,
                        diagnostic=CompletionDiagnostic("provider_transport"),
                    ) from exc
                self.sleeper(delay)
                continue
            body_error = _openrouter_body_error(response)
            status = body_error[0] if body_error else response.status_code
            if status not in {429, 500, 502, 503, 504}:
                break
            delay = 1.5 if status == 429 else 0.15 * (2**attempt)
            if self._may_retry(attempt, started, delay):
                self.sleeper(delay)
                continue
            if body_error is not None:
                raise ConversationProviderError(
                    "Luna is temporarily unavailable. Nothing was saved. Please try again.",
                    status_code=429 if body_error[0] == 429 else 502,
                    diagnostic=CompletionDiagnostic("upstream_error"),
                )
            break
        if response is None:
            raise ConversationProviderError(
                "Luna did not respond in time. Nothing was saved.",
                status_code=503,
                diagnostic=CompletionDiagnostic("provider_transport"),
            )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ConversationProviderError(
                "Luna is temporarily unavailable. Nothing was saved. Please try again.",
                status_code=503 if exc.response.status_code in {429, 500, 502, 503, 504} else 502,
                diagnostic=CompletionDiagnostic("upstream_http"),
            ) from exc
        return response
