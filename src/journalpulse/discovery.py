"""Optional Brave retrieval and Luna editorial selection, without journal access.

Candidate pages are never fetched: this avoids arbitrary-host SSRF and DNS-rebinding
risks until a separately reviewed, IP-pinned page reader is available. Recommendations
therefore expose their snippet-only evidence, rather than implying full-page review.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from html import unescape
from typing import Any, Protocol

import httpx
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .auth import AuthContext
from .config import Settings
from .discovery_models import (
    MAX_CANDIDATES,
    MAX_RESULTS,
    DiscoveryCandidate,
    DiscoveryProvenance,
    DiscoveryRequest,
    DiscoveryResponse,
    checked_source_url,
    source_url_identity,
)
from .discovery_prompts import DISCOVERY_PROMPT_VERSION, DISCOVERY_SYSTEM_PROMPT
from .domain import GenerationErrorResponse, ModelRun, SafetyMode
from .intelligence import UnsupportedProviderResponse, _json_text_from_content
from .persistence import Repository
from .safety import SUPPORT_FALLBACK_MESSAGE, assess_safety

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
MAX_PROVIDER_BYTES = 256_000
LIMITATIONS = [
    "Suggestions use search-provider snippets; full pages were not read.",
    "Links were returned by Brave and structurally checked. Their availability and claims are not verified.",
]


class DiscoveryProviderError(Exception):
    """A failed provider call or invalid response; no invented results replace it."""

    def __init__(self, message: str, *, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


class DiscoveryClient(Protocol):
    def search(self, payload: DiscoveryRequest) -> DiscoveryResponse:
        """Retrieve and select resources using only the explicit request fields."""
        ...


class RefinementOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    additional_terms: str = Field(max_length=160)


class SelectedCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    candidate_id: int = Field(ge=0, lt=MAX_CANDIDATES, strict=True)
    why_selected: str = Field(min_length=1, max_length=420)


class SelectionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    selections: list[SelectedCandidate] = Field(max_length=MAX_RESULTS)


REFINEMENT_SCHEMA: dict[str, Any] = {
    "name": "journalpulse_discovery_refinement",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["additional_terms"],
        "properties": {"additional_terms": {"type": "string", "maxLength": 160}},
    },
}
SELECTION_SCHEMA: dict[str, Any] = {
    "name": "journalpulse_discovery_selection",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["selections"],
        "properties": {
            "selections": {
                "type": "array",
                "maxItems": MAX_RESULTS,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["candidate_id", "why_selected"],
                    "properties": {
                        "candidate_id": {"type": "integer", "minimum": 0, "maximum": MAX_CANDIDATES - 1},
                        "why_selected": {"type": "string", "minLength": 1, "maxLength": 420},
                    },
                },
            },
        },
    },
}


@dataclass(frozen=True)
class SearchSnippet:
    title: str
    url: str
    description: str


def _provider_json(
    client: httpx.Client, method: str, url: str, *, timeout: float, **kwargs: Any
) -> dict[str, Any]:
    """Read a bounded JSON response; redirects and automatic retries are disabled."""
    deadline = time.monotonic() + timeout
    try:
        # Small read timeouts plus the stream deadline stop a slowly trickled reply
        # from renewing its timeout indefinitely. Fixed provider hosts are the only destinations.
        provider_timeout = httpx.Timeout(timeout, read=min(timeout, 5))
        with client.stream(
            method, url, timeout=provider_timeout, follow_redirects=False, **kwargs
        ) as response:
            if response.status_code != 200:
                status = 503 if response.status_code in {401, 403, 429, 500, 502, 503, 504} else 502
                raise DiscoveryProviderError(
                    "Resource discovery is temporarily unavailable.", status_code=status
                )
            content = bytearray()
            for chunk in response.iter_bytes():
                if time.monotonic() > deadline:
                    raise DiscoveryProviderError("Resource discovery timed out.", status_code=503)
                if len(content) + len(chunk) > MAX_PROVIDER_BYTES:
                    raise DiscoveryProviderError("The discovery provider response exceeded its size limit.")
                content.extend(chunk)
        payload = json.loads(content)
        if not isinstance(payload, dict) or "error" in payload:
            raise DiscoveryProviderError("Resource discovery could not complete the provider request.")
        return payload
    except (httpx.HTTPError, OSError) as exc:
        raise DiscoveryProviderError(
            "Resource discovery did not respond in time. Please try again.", status_code=503
        ) from exc
    except (ValueError, UnicodeError, RecursionError) as exc:
        # A small body can still exceed the JSON decoder's nesting limit.
        raise DiscoveryProviderError("The discovery provider returned an invalid response.") from exc


class OpenWebDiscoveryClient:
    """At most one search and two model calls, returning up to three source-linked suggestions.

    Refinement keeps the original approved goal verbatim in the search query. No
    journal repository, conversation identifier, or raw-entry field is accepted.
    """

    def __init__(
        self, settings: Settings, client: httpx.Client | None = None,
        query_validator: Callable[[str], str] | None = None,
    ) -> None:
        if not settings.discovery_enabled:
            raise ValueError("Brave search and zero-data-retention Luna must be enabled")
        self.settings = settings
        self.client = client
        self.query_validator = query_validator

    def search(self, payload: DiscoveryRequest) -> DiscoveryResponse:
        if not payload.llm_consent:
            raise ValueError("Resource discovery requires explicit AI consent")
        context = nullcontext(self.client) if self.client else httpx.Client(follow_redirects=False)
        with context as client:
            assert client is not None
            query = payload.original_query
            model_runs: list[ModelRun] = []
            if payload.feedback:
                refined, run = self._model(
                    client,
                    REFINEMENT_SCHEMA,
                    {
                        "task": "refine",
                        "original_goal": payload.original_query,
                        "previous_query": payload.previous_query,
                        "feedback": payload.feedback,
                    },
                )
                try:
                    focus = RefinementOutput.model_validate(refined).additional_terms
                except ValidationError as exc:
                    raise DiscoveryProviderError("The query refinement did not match its schema.") from exc
                # The model can add a focus, but cannot replace the user's original goal.
                query = f"{payload.original_query} {focus}".strip()
                model_runs.append(run)
            if self.query_validator is not None:
                try:
                    query = self.query_validator(query)
                except ValueError as exc:
                    raise DiscoveryProviderError(
                        "Search refinement exceeded the general activity topic. Please edit it.",
                        status_code=422,
                    ) from exc
            candidates = self._retrieve(client, query, set(payload.excluded_urls))
            selected: list[DiscoveryCandidate] = []
            if candidates:
                output, run = self._model(
                    client,
                    SELECTION_SCHEMA,
                    {
                        "task": "select",
                        "original_goal": payload.original_query,
                        "search_query": query,
                        "feedback": payload.feedback,
                        "maximum_results": MAX_RESULTS,
                        "evidence_kind": "search_snippet",
                        "candidates": [
                            {
                                "candidate_id": index,
                                "title": item.title,
                                "url": item.url,
                                "snippet": item.description,
                            }
                            for index, item in enumerate(candidates)
                        ],
                    },
                )
                selected = self._resolve_selections(output, candidates)
                model_runs.append(run)
            return DiscoveryResponse(
                original_query=payload.original_query,
                updated_query=query,
                candidates=selected,
                provenance=DiscoveryProvenance(
                    prompt_version=DISCOVERY_PROMPT_VERSION,
                    retrieved_at=datetime.now(UTC).isoformat(),
                    candidate_count=len(candidates),
                    model_runs=model_runs,
                ),
                limitations=LIMITATIONS,
            )

    def _retrieve(self, client: httpx.Client, query: str, excluded: set[str]) -> list[SearchSnippet]:
        response = _provider_json(
            client,
            "GET",
            BRAVE_SEARCH_URL,
            timeout=self.settings.search_timeout_seconds,
            headers={
                "X-Subscription-Token": self.settings.search_api_key or "",
                "Accept": "application/json",
            },
            params={"q": query, "count": MAX_CANDIDATES, "safesearch": "strict", "text_decorations": "false"},
        )
        web = response.get("web", {})
        results = web.get("results") if isinstance(web, dict) else None
        # Brave omits the web section when it has no web results.
        if results is None and ("web" not in response or web == {}):
            return []
        if not isinstance(results, list):
            raise DiscoveryProviderError("The search provider returned an invalid result list.")
        candidates: list[SearchSnippet] = []
        seen = set(excluded)
        for item in results[:MAX_CANDIDATES]:
            if not isinstance(item, dict):
                continue
            title, raw_url, description = (item.get(key) for key in ("title", "url", "description"))
            if not (
                isinstance(title, str)
                and title.strip()
                and isinstance(raw_url, str)
                and raw_url.strip()
                and isinstance(description, str)
                and description.strip()
            ):
                continue
            # Brave can encode punctuation even with text_decorations=false.
            # Decode text once; React still renders strings safely, and the
            # provider's URL stays unchanged so we do not alter its destination.
            title = unescape(title).strip()
            description = unescape(description).strip()
            if not title or not description:
                continue
            try:
                url = checked_source_url(raw_url)
            except (ValueError, UnicodeError):
                continue
            identity = source_url_identity(url)
            if identity in seen:
                continue
            seen.add(identity)
            candidates.append(
                SearchSnippet(title=title[:200], url=url, description=description[:800])
            )
        return candidates

    def _model(
        self, client: httpx.Client, schema: dict[str, Any], data: dict[str, Any]
    ) -> tuple[dict[str, Any], ModelRun]:
        started = time.perf_counter()
        response = _provider_json(
            client,
            "POST",
            f"{self.settings.openrouter_base_url}/chat/completions",
            timeout=min(self.settings.chat_timeout_seconds, 25),
            headers={
                "Authorization": f"Bearer {self.settings.openrouter_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.settings.chat_model,
                "provider": {"zdr": True, "require_parameters": True},
                "max_tokens": 4000,
                "include_reasoning": False,
                "reasoning": {"effort": "medium"},
                "response_format": {"type": "json_schema", "json_schema": schema},
                "messages": [
                    {"role": "system", "content": DISCOVERY_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(data)},
                ],
            },
        )
        try:
            choice = response["choices"][0]
            if not isinstance(choice, dict):
                raise ValueError("invalid choice")
            message = choice.get("message")
            if choice.get("finish_reason") == "content_filter" or (
                isinstance(message, dict)
                and isinstance(message.get("refusal"), str)
                and message["refusal"].strip()
            ):
                # Native provider declines outrank any accompanying JSON. Stop
                # refinement here, before another model call or a web search.
                raise DiscoveryProviderError(
                    "Luna's AI provider declined this discovery request.", status_code=422
                )
            if choice.get("finish_reason") == "length":
                raise ValueError("truncated")
            if not isinstance(message, dict):
                raise ValueError("invalid message")
            output = json.loads(_json_text_from_content(message["content"]))
            if not isinstance(output, dict):
                raise ValueError("expected object")
            usage = response.get("usage", {})
            if not isinstance(usage, dict):
                raise ValueError("invalid usage")
            raw_provider = response.get("provider")
            run = ModelRun(
                model=response.get("model", self.settings.chat_model),
                provider=raw_provider if isinstance(raw_provider, str) else "openrouter",
                latency_ms=round((time.perf_counter() - started) * 1000),
                schema_valid=True,
                prompt_version=DISCOVERY_PROMPT_VERSION,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
            )
        except (
            KeyError, IndexError, TypeError, ValueError, RecursionError, UnsupportedProviderResponse
        ) as exc:
            raise DiscoveryProviderError("Luna's discovery response did not match its schema.") from exc
        return output, run

    @staticmethod
    def _resolve_selections(
        output: dict[str, Any], candidates: list[SearchSnippet]
    ) -> list[DiscoveryCandidate]:
        try:
            selections = SelectionOutput.model_validate(output).selections
            ids = [item.candidate_id for item in selections]
            if len(ids) != len(set(ids)) or any(index >= len(candidates) for index in ids):
                raise ValueError("candidate IDs must be unique retrieved candidates")
            return [
                DiscoveryCandidate(
                    title=candidates[item.candidate_id].title,
                    url=candidates[item.candidate_id].url,
                    description=candidates[item.candidate_id].description,
                    why_selected=item.why_selected,
                )
                for item in selections
            ]
        except (ValueError, ValidationError) as exc:
            raise DiscoveryProviderError(
                "Luna selected an invalid or repeated source. Please try again."
            ) from exc


def register_discovery_routes(
    app: FastAPI,
    *,
    settings: Settings,
    repositories: Callable[[AuthContext], Repository],
    auth_dependency: Callable[..., AuthContext],
    enforce_generation_limit: Callable[[AuthContext, Repository], None],
    client: DiscoveryClient | None = None,
) -> None:
    """Register opt-in, consent-gated, rate-limited stateless discovery."""

    @app.post(
        "/v1/discovery/search",
        response_model=DiscoveryResponse,
        responses={422: {
            "model": GenerationErrorResponse,
            "description": "Invalid input, support routing, or provider decline",
        }},
    )
    def search_resources(
        payload: DiscoveryRequest, auth: AuthContext = Depends(auth_dependency)
    ) -> DiscoveryResponse:
        # A sentence boundary prevents negation in the topic from hiding risk in
        # later feedback. Reuse the app's support router before any external call.
        safety = assess_safety(f"{payload.original_query}. {payload.feedback or ''}", payload.locale)
        if safety.mode == SafetyMode.SUPPORT:
            raise HTTPException(
                status_code=422,
                detail=(safety.support_message or SUPPORT_FALLBACK_MESSAGE)
                + " Web discovery is paused. Talk with Luna for support options.",
            )
        if not payload.llm_consent:
            raise HTTPException(status_code=403, detail="Approve AI use before searching for resources.")
        if not settings.discovery_enabled:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Web discovery is unavailable. "
                    "Brave search and Luna with zero data retention must be configured."
                ),
            )
        enforce_generation_limit(auth, repositories(auth))
        try:
            return (client or OpenWebDiscoveryClient(settings)).search(payload)
        except DiscoveryProviderError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
