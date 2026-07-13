import json
import os
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import httpx

from .config import (
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODE,
    LLM_API_KEY_ENV,
    LLM_APP_TITLE_ENV,
    LLM_APP_URL_ENV,
    LLM_BASE_URL_ENV,
    LLM_MODEL_ENV,
    LLM_MODE_ENV,
    LLM_TIMEOUT_SECONDS,
    RESOURCE_DOMAIN_SAFELIST,
)
from .schemas import (
    StructuredCoachDraft,
    StructuredEmotionDraft,
    StructuredResourceRecommendation,
)

SUPPORTED_LLM_MODES = {"off", "rewrite", "structured"}
ALLOWED_RESOURCE_INTENTS = {
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
}


class OpenAICompatibleCoachAdapter:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout_seconds: float = LLM_TIMEOUT_SECONDS,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_env(cls) -> Optional["OpenAICompatibleCoachAdapter"]:
        api_key = os.getenv(LLM_API_KEY_ENV)
        model = os.getenv(LLM_MODEL_ENV)
        if not api_key or not model:
            return None
        base_url = os.getenv(LLM_BASE_URL_ENV, DEFAULT_LLM_BASE_URL)
        return cls(api_key=api_key, base_url=base_url, model=model)

    def headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        app_url = os.getenv(LLM_APP_URL_ENV, "").strip()
        app_title = os.getenv(LLM_APP_TITLE_ENV, "").strip()
        if app_url:
            headers["HTTP-Referer"] = app_url
        if app_title:
            headers["X-Title"] = app_title
            headers["X-OpenRouter-Title"] = app_title
        return headers

    def rewrite(self, *, draft_message: str, suggested_replies: Iterable[str], context: dict) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Rewrite the assistant draft in a calm, concise, non-clinical tone. "
                        "Do not add diagnosis, therapy claims, or new safety advice. "
                        "Keep the same intent and keep it under 70 words."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "draft_message": draft_message,
                            "suggested_replies": list(suggested_replies),
                            "context": context,
                        }
                    ),
                },
            ],
            "temperature": 0.5,
            "max_tokens": 120,
        }
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers(),
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
        return body["choices"][0]["message"]["content"].strip()

    def structured(
        self,
        *,
        deterministic_payload: dict,
        allowed_resources: Iterable[dict],
        context: dict,
    ) -> dict:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return only JSON for a non-clinical journaling coach. "
                        "Stay brief, practical, and supportive. Do not diagnose, provide therapy, "
                        "make medical claims, or invent resources. Use only allowed resource_ids "
                        "and allowed resource_intent values."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "required_json_shape": {
                                "assistant_message": "string under 520 characters",
                                "tips": "array of up to 3 short strings",
                                "practical_steps": "array of up to 4 concrete strings",
                                "suggested_replies": "array of up to 4 short strings",
                                "resource_intent": sorted(ALLOWED_RESOURCE_INTENTS),
                                "resource_ids": "array using only allowed resource ids",
                                "reflection_question": "optional short question",
                                "communication_draft": "optional short message the user could adapt",
                                "confidence_note": "optional short uncertainty/confidence note",
                                "refusal_reason": "optional string",
                            },
                            "deterministic_fallback": deterministic_payload,
                            "allowed_resources": list(allowed_resources),
                            "context": context,
                        }
                    ),
                },
            ],
            "temperature": 0.2,
            "max_tokens": 520,
            "response_format": {"type": "json_object"},
        }
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers(),
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
        content = body["choices"][0]["message"]["content"]
        return json.loads(content)

    def classify_emotion(self, *, text: str, calibrated_context: dict) -> dict:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return only JSON for a non-clinical journaling emotion classifier. "
                        "Do not diagnose, infer mental disorders, give therapy, or provide crisis counseling. "
                        "Classify the emotional signal for reflection support using only allowed emotions and tags."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "required_json_shape": {
                                "primary_emotion": ["sadness", "joy", "love", "anger", "fear", "surprise"],
                                "secondary_emotions": "array of up to 3 core emotions",
                                "emotion_tags": [
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
                                ],
                                "intensity": "number between 0 and 1",
                                "confidence": ["low", "medium", "high"],
                                "is_mixed": "boolean",
                                "themes": "array of up to 5 short strings",
                                "rationale": "short non-clinical rationale under 380 characters",
                                "refusal_reason": "optional string",
                            },
                            "journal_text": text,
                            "calibrated_fallback": calibrated_context,
                        }
                    ),
                },
            ],
            "temperature": 0.0,
            "max_tokens": 260,
            "response_format": {"type": "json_object"},
        }
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers(),
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
        content = body["choices"][0]["message"]["content"]
        return json.loads(content)

    def recommend(
        self,
        *,
        text: str,
        analysis_context: dict,
        catalog_options: Iterable[dict],
        allowed_domains: Iterable[str],
        max_generated: int = 3,
    ) -> dict:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You curate browser-safe, non-clinical wellbeing resources for a journaling "
                        "app. Return only JSON. Do not diagnose, give therapy, or make medical claims. "
                        "First re-rank the most relevant catalog ids for this entry. Then suggest fresh, "
                        "specific resources the catalog lacks. Every generated url MUST point to one of "
                        "the allowed_domains (a homepage, channel, or canonical page is safer than a deep "
                        "link). Prefer a varied mix of watch/read/play/move coping styles. Never suggest "
                        "crisis hotlines here."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "required_json_shape": {
                                "ranked_catalog_ids": "array of the most relevant allowed catalog ids, best first",
                                "generated": [
                                    {
                                        "title": "string",
                                        "url": "https url on an allowed domain",
                                        "resource_type": ["video", "website", "game"],
                                        "coping_style": ["watch", "read", "play", "move"],
                                        "provider": "short source name",
                                        "summary": "one sentence on what it is",
                                        "why": "one sentence on why it fits this entry",
                                        "goal_tags": ["ground", "planning", "reframing", "connection", "movement"],
                                    }
                                ],
                                "refusal_reason": "optional string",
                            },
                            "max_generated": max_generated,
                            "journal_text": text,
                            "analysis": analysis_context,
                            "allowed_domains": list(allowed_domains),
                            "catalog_options": list(catalog_options),
                        }
                    ),
                },
            ],
            "temperature": 0.4,
            "max_tokens": 700,
            "response_format": {"type": "json_object"},
        }
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers(),
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
        content = body["choices"][0]["message"]["content"]
        return json.loads(content)


def configured_llm_mode() -> str:
    mode = os.getenv(LLM_MODE_ENV, DEFAULT_LLM_MODE).strip().lower()
    return mode if mode in SUPPORTED_LLM_MODES else DEFAULT_LLM_MODE


def llm_adapter_available() -> bool:
    return configured_llm_mode() != "off" and OpenAICompatibleCoachAdapter.from_env() is not None


def validate_structured_emotion_payload(payload: dict) -> dict:
    draft = StructuredEmotionDraft.model_validate(payload)
    secondary = [emotion for emotion in draft.secondary_emotions if emotion != draft.primary_emotion]
    return {
        "primary_emotion": draft.primary_emotion,
        "secondary_emotions": secondary[:3],
        "emotion_tags": draft.emotion_tags[:6],
        "intensity": round(float(draft.intensity), 4),
        "confidence": draft.confidence,
        "is_mixed": draft.is_mixed or bool(secondary),
        "themes": draft.themes[:5],
        "rationale": draft.rationale.strip(),
        "refusal_reason": draft.refusal_reason,
    }


def validate_structured_coach_payload(
    payload: dict,
    *,
    allowed_resource_ids: Iterable[str],
    fallback_replies: Iterable[str],
) -> dict:
    draft = StructuredCoachDraft.model_validate(payload)
    allowed_ids = set(allowed_resource_ids)
    unknown_ids = [resource_id for resource_id in draft.resource_ids if resource_id not in allowed_ids]
    if unknown_ids:
        raise ValueError(f"LLM returned unknown resource_ids: {', '.join(unknown_ids)}")
    if draft.resource_intent not in ALLOWED_RESOURCE_INTENTS:
        raise ValueError(f"LLM returned unsupported resource_intent: {draft.resource_intent}")

    suggested_replies = draft.suggested_replies or list(fallback_replies)
    practical_steps = draft.practical_steps or draft.tips
    return {
        "assistant_message": draft.assistant_message,
        "tips": draft.tips[:3],
        "practical_steps": practical_steps[:4],
        "suggested_replies": suggested_replies[:4],
        "resource_intent": draft.resource_intent,
        "resource_ids": draft.resource_ids[:6],
        "reflection_question": draft.reflection_question,
        "communication_draft": draft.communication_draft,
        "confidence_note": draft.confidence_note,
        "refusal_reason": draft.refusal_reason,
    }


def maybe_rewrite_coach_message(
    draft_message: str,
    suggested_replies: List[str],
    *,
    context: dict,
    use_llm: bool,
) -> tuple:
    if not use_llm or configured_llm_mode() != "rewrite":
        return draft_message, False

    adapter = OpenAICompatibleCoachAdapter.from_env()
    if adapter is None:
        return draft_message, False

    try:
        rewritten = adapter.rewrite(
            draft_message=draft_message,
            suggested_replies=suggested_replies,
            context=context,
        )
    except Exception:
        return draft_message, False
    return rewritten, True


def maybe_generate_structured_coach_response(
    deterministic_payload: dict,
    *,
    allowed_resources: Iterable[dict],
    context: dict,
    use_llm: bool,
) -> Tuple[Optional[dict], bool, Optional[str]]:
    if not use_llm or configured_llm_mode() != "structured":
        return None, False, None

    adapter = OpenAICompatibleCoachAdapter.from_env()
    if adapter is None:
        return None, False, "llm_adapter_unavailable"

    resource_options = [
        {
            "id": resource["id"],
            "title": resource["title"],
            "coping_style": resource.get("coping_style"),
            "goal_tags": resource.get("goal_tags", []),
            "source_tier": resource.get("source_tier"),
        }
        for resource in allowed_resources
    ]
    allowed_resource_ids = [resource["id"] for resource in resource_options]

    try:
        raw_payload = adapter.structured(
            deterministic_payload=deterministic_payload,
            allowed_resources=resource_options,
            context=context,
        )
        structured_payload = validate_structured_coach_payload(
            raw_payload,
            allowed_resource_ids=allowed_resource_ids,
            fallback_replies=deterministic_payload.get("suggested_replies", []),
        )
    except Exception as exc:
        return None, False, f"structured_llm_invalid:{exc.__class__.__name__}"

    structured_payload["agent_model"] = getattr(adapter, "model", None)
    return structured_payload, True, None


def url_domain_allowed(url: str, allowed_domains: Iterable[str]) -> bool:
    """Return True only if the url's host equals or is a subdomain of an allowed domain."""

    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    if not host:
        return False
    host = host[4:] if host.startswith("www.") else host
    for domain in allowed_domains:
        domain = domain.lower().strip()
        if host == domain or host.endswith("." + domain):
            return True
    return False


def validate_resource_recommendation_payload(
    payload: dict,
    *,
    allowed_catalog_ids: Iterable[str],
    allowed_domains: Iterable[str],
    max_generated: int,
) -> dict:
    draft = StructuredResourceRecommendation.model_validate(payload)
    allowed_ids = set(allowed_catalog_ids)
    ranked_ids = [resource_id for resource_id in draft.ranked_catalog_ids if resource_id in allowed_ids]

    generated: List[dict] = []
    seen_urls = set()
    for item in draft.generated:
        if len(generated) >= max_generated:
            break
        if not url_domain_allowed(item.url, allowed_domains):
            continue
        normalized_url = item.url.rstrip("/")
        if normalized_url in seen_urls:
            continue
        seen_urls.add(normalized_url)
        generated.append(
            {
                "title": item.title,
                "url": item.url,
                "resource_type": item.resource_type,
                "coping_style": item.coping_style,
                "provider": item.provider or "AI-suggested",
                "summary": item.summary,
                "why": item.why,
                "goal_tags": item.goal_tags,
            }
        )

    return {
        "ranked_catalog_ids": ranked_ids,
        "generated": generated,
        "refusal_reason": draft.refusal_reason,
    }


def maybe_generate_resource_suggestions(
    text: str,
    *,
    analysis_context: dict,
    catalog_options: Iterable[dict],
    use_llm: bool,
    max_generated: int = 3,
) -> Tuple[Optional[dict], bool, Optional[str]]:
    """Ask the LLM to re-rank the catalog and propose fresh, safelisted resources.

    Returns (payload, used, fallback_reason). Always safe to ignore: any failure or
    missing configuration returns (None, False, reason) so the catalog-only path stands.
    """

    if not use_llm or configured_llm_mode() == "off":
        return None, False, None

    adapter = OpenAICompatibleCoachAdapter.from_env()
    if adapter is None:
        return None, False, "llm_adapter_unavailable"

    options = list(catalog_options)
    allowed_catalog_ids = [option["id"] for option in options]
    try:
        raw_payload = adapter.recommend(
            text=text,
            analysis_context=analysis_context,
            catalog_options=options,
            allowed_domains=RESOURCE_DOMAIN_SAFELIST,
            max_generated=max_generated,
        )
        validated = validate_resource_recommendation_payload(
            raw_payload,
            allowed_catalog_ids=allowed_catalog_ids,
            allowed_domains=RESOURCE_DOMAIN_SAFELIST,
            max_generated=max_generated,
        )
    except Exception as exc:
        return None, False, f"resource_llm_invalid:{exc.__class__.__name__}"

    validated["agent_model"] = getattr(adapter, "model", None)
    return validated, True, None
