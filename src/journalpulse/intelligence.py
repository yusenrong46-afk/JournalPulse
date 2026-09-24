from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import BaseModel, Field, ValidationError

from .config import Settings
from .domain import AffectiveState, ModelRun, ReflectionCopy


class UnsupportedProviderResponse(Exception):
    """Provider message.content was not a string or a text-part array."""


class ConversationProviderError(Exception):
    """The conversation model failed, or its completion cannot be shown."""

    def __init__(self, message: str, *, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


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
        self.client = client or httpx.Client(timeout=settings.openrouter_timeout_seconds)
        self.sleeper = sleeper

    def analyze(self, text: str, context: dict[str, str]) -> AnalysisResult:
        started = time.perf_counter()
        response: httpx.Response | None = None
        for attempt in range(self.settings.openrouter_max_attempts):
            try:
                response = self.client.post(
                    f"{self.settings.openrouter_base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.settings.openrouter_api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://journalpulse.app",
                        "X-Title": "JournalPulse Research Beta",
                    },
                    json={
                        "model": self.settings.openrouter_model,
                        "provider": {"zdr": True},
                        "max_tokens": 700,
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
            if response.status_code not in {429, 500, 502, 503, 504}:
                break
            if attempt + 1 < self.settings.openrouter_max_attempts:
                self.sleeper(0.15 * (2**attempt))

        if response is None:
            raise httpx.ConnectError("OpenRouter did not return a response")
        latency_ms = round((time.perf_counter() - started) * 1000)
        response.raise_for_status()
        body = response.json()
        content = body["choices"][0]["message"]["content"]
        structured = structured_reflection_from_content(content)
        usage = body.get("usage", {})
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
    except (httpx.HTTPError, KeyError, ValueError, ValidationError) as exc:
        fallback = deterministic_reflection(text, self_report)
        return AnalysisResult(
            state=fallback.state,
            reflection=fallback.reflection,
            resource_intent=fallback.resource_intent,
            model_run=fallback.model_run.model_copy(
                update={"fallback_reason": f"openrouter_{exc.__class__.__name__}"}
            ),
        )


CONVERSATION_PROMPT_VERSION = "2026-09-24.1"

CONVERSATION_SYSTEM_PROMPT = (
    "You are a non-clinical journaling companion. Write in plain text, about 120 words at most, "
    "and ask one question at a time. Reflect the person's own words. Do not diagnose, give medical "
    "or crisis advice, claim memory of other conversations, or output URLs, phone numbers, or "
    "resource names. Set offer_action only when the person asks what to do next or sounds ready "
    "to try one small thing. Return only the schema."
)

CONVERSATION_JSON_SCHEMA: dict[str, Any] = {
    "name": "journalpulse_conversation_turn",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["reply", "offer_action", "resource_intent", "card_reason", "summary"],
        "properties": {
            "reply": {"type": "string", "minLength": 1, "maxLength": 1200},
            "offer_action": {"type": "boolean"},
            "resource_intent": {
                "type": "string",
                "enum": ["ground", "move", "connect", "reflect", "play", "watch", "read", "pause"],
            },
            "card_reason": {"type": "string", "maxLength": 240},
            "summary": {"type": "string", "minLength": 1, "maxLength": 420},
        },
    },
}


class ConversationTurnOutput(BaseModel):
    reply: str = Field(min_length=1, max_length=1200)
    offer_action: bool
    resource_intent: str = Field(min_length=1, max_length=40)
    card_reason: str = Field(default="", max_length=240)
    summary: str = Field(min_length=1, max_length=420)

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


class OpenRouterConversationClient:
    """One Luna turn. There is no canned reply when the provider fails."""

    def __init__(
        self,
        settings: Settings,
        client: httpx.Client | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not settings.openrouter_api_key or not settings.chat_model:
            raise ValueError("OpenRouter is not configured")
        if not settings.openrouter_zdr:
            raise ValueError("JournalPulse requires zero-data-retention routing")
        self.settings = settings
        self.client = client or httpx.Client(timeout=settings.chat_timeout_seconds)
        self.sleeper = sleeper

    def complete(self, messages: list[dict[str, str]]) -> ConversationCompletion:
        # Parameter set is limited to what OpenRouter lists for openai/gpt-6-luna
        # (models list and endpoints page, 2026-09-24): max_tokens, response_format,
        # structured_outputs, reasoning, include_reasoning. Temperature is not supported.
        # Reasoning defaults to medium, so the request asks for low effort and excludes
        # reasoning text from the reply. Bedrock does not list response_format.
        body = {
            "model": self.settings.chat_model,
            "provider": {"zdr": True},
            "max_tokens": 4000,
            "include_reasoning": False,
            "reasoning": {"effort": "low"},
            "response_format": {
                "type": "json_schema",
                "json_schema": CONVERSATION_JSON_SCHEMA,
            },
            "messages": [
                {"role": "system", "content": CONVERSATION_SYSTEM_PROMPT},
                *messages,
            ],
        }
        started = time.perf_counter()
        response = self._post(body)
        latency_ms = round((time.perf_counter() - started) * 1000)
        payload = response.json()
        choice = payload["choices"][0]
        if choice.get("finish_reason") == "length":
            raise ConversationProviderError("The model reply was cut off. Nothing was saved.")
        try:
            content = choice["message"]["content"]
            structured = ConversationTurnOutput.model_validate_json(
                _json_text_from_content(content)
            ).require_card_reason()
        except UnsupportedProviderResponse:
            raise
        except (KeyError, ValueError, ValidationError) as exc:
            raise ConversationProviderError(
                "The model reply did not match the conversation schema. Nothing was saved."
            ) from exc
        usage = payload.get("usage", {})
        raw_provider = payload.get("provider", "openrouter")
        provider = raw_provider if isinstance(raw_provider, str) else "openrouter"
        return ConversationCompletion(
            reply=structured.reply,
            offer_action=structured.offer_action,
            resource_intent=structured.resource_intent,
            card_reason=structured.card_reason,
            summary=structured.summary,
            model_run=ModelRun(
                model=payload.get("model", self.settings.chat_model),
                provider=provider,
                latency_ms=latency_ms,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                schema_valid=True,
                prompt_version=CONVERSATION_PROMPT_VERSION,
            ),
        )

    def _post(self, body: dict[str, Any]) -> httpx.Response:
        response: httpx.Response | None = None
        for attempt in range(self.settings.openrouter_max_attempts):
            try:
                response = self.client.post(
                    f"{self.settings.openrouter_base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.settings.openrouter_api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://journalpulse.app",
                        "X-Title": "JournalPulse Research Beta",
                    },
                    json=body,
                )
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                if attempt + 1 >= self.settings.openrouter_max_attempts:
                    raise ConversationProviderError(
                        "Luna did not respond in time. Nothing was saved.",
                        status_code=503,
                    ) from exc
                self.sleeper(0.15 * (2**attempt))
                continue
            if response.status_code not in {429, 500, 502, 503, 504}:
                break
            if attempt + 1 < self.settings.openrouter_max_attempts:
                self.sleeper(0.15 * (2**attempt))
        if response is None:
            raise ConversationProviderError(
                "Luna did not respond in time. Nothing was saved.",
                status_code=503,
            )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:500]
            raise ConversationProviderError(
                f"The model request failed. Nothing was saved. {detail}".strip()
            ) from exc
        return response
