"""Stateless discovery contracts: only an approved topic and explicit search feedback."""

from __future__ import annotations

import ipaddress
import re
from typing import Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .domain import ModelRun

MAX_CANDIDATES = 6
MAX_RESULTS = 3
MAX_EXCLUDED_URLS = 30


def checked_source_url(value: str) -> str:
    """Check HTTPS syntax while preserving the source's path, query, and fragment.

    This is a structural check, not a network fetch, fact check, or guarantee of
    availability. No candidate host is contacted by this slice.
    """
    if len(value) > 2048 or re.search(r"[\s\x00-\x1f\x7f\\]", value):
        raise ValueError("Use a public HTTPS URL without whitespace or backslashes")
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Use a public HTTPS URL without credentials")
    if parsed.port not in (None, 443):
        raise ValueError("Only standard HTTPS links are allowed")
    host = parsed.hostname.lower().rstrip(".")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        host = host.encode("idna").decode("ascii")
        labels = host.split(".")
        if (
            len(labels) < 2
            or len(host) > 253
            or any(not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label) for label in labels)
            or host.endswith(
                (
                    ".localhost",
                    ".local",
                    ".localdomain",
                    ".internal",
                    ".test",
                    ".invalid",
                    ".lan",
                    ".home.arpa",
                )
            )
        ):
            raise ValueError("Use a public HTTPS hostname") from None
        # Legacy numeric host forms can be interpreted as loopback by browsers.
        if all(re.fullmatch(r"(?:[0-9]+|0x[0-9a-f]+)", label) for label in labels):
            raise ValueError("Numeric shorthand hostnames are not allowed") from None
    else:
        if not address.is_global or address.is_multicast or address.is_reserved:
            raise ValueError("Private and reserved address links are not allowed")
        if address.version == 6:
            if address.scope_id:
                raise ValueError("Scoped address links are not allowed")
            host = f"[{address.compressed}]"
    # Path, query, and anchors may affect what opens. Repeat normalization belongs
    # to comparison keys, never the source link that the person follows.
    return urlunsplit(("https", host, parsed.path or "/", parsed.query, parsed.fragment))


def source_url_identity(value: str) -> str:
    """Normalize a comparison key for exclusions, without rewriting source links."""
    parsed = urlsplit(checked_source_url(value))
    query = urlencode(
        [
            (key, item)
            for key, item in parse_qsl(parsed.query, keep_blank_values=True)
            if not key.lower().startswith("utm_") and key.lower() not in {"fbclid", "gclid"}
        ]
    )
    return urlunsplit(("https", parsed.netloc, parsed.path.rstrip("/") or "/", query, ""))


class DiscoveryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    original_query: str = Field(min_length=3, max_length=160)
    previous_query: str | None = Field(default=None, min_length=3, max_length=400)
    feedback: str | None = Field(default=None, min_length=1, max_length=600)
    excluded_urls: list[str] = Field(default_factory=list, max_length=MAX_EXCLUDED_URLS)
    llm_consent: bool = False
    locale: str = Field(default="CA", min_length=2, max_length=8)

    @field_validator("excluded_urls")
    @classmethod
    def validate_exclusions(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(source_url_identity(value) for value in values))

    @model_validator(mode="after")
    def refinement_has_previous_query(self) -> DiscoveryRequest:
        if self.feedback and not self.previous_query:
            raise ValueError("Refinement needs the previous query and the original goal")
        return self


class DiscoveryCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str = Field(min_length=1, max_length=200)
    url: str = Field(min_length=1, max_length=2048)
    description: str = Field(min_length=1, max_length=800)
    why_selected: str = Field(min_length=1, max_length=420)
    evidence_kind: Literal["search_snippet"] = "search_snippet"

    @field_validator("url")
    @classmethod
    def validate_source_url(cls, value: str) -> str:
        return checked_source_url(value)


class DiscoveryModelRun(ModelRun):
    generation_id: str | None = Field(default=None, max_length=200)
    cost_usd: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class DiscoveryProvenance(BaseModel):
    search_provider: Literal["brave"] = "brave"
    prompt_version: str
    retrieved_at: str
    candidate_count: int = Field(ge=0, le=MAX_CANDIDATES)
    search_calls: Literal[1] = 1
    model_runs: list[DiscoveryModelRun] = Field(max_length=2)
    page_fetches: Literal[0] = 0

    @field_validator("model_runs", mode="before")
    @classmethod
    def legacy_model_runs(cls, values: list) -> list:
        return [value.model_dump() if isinstance(value, ModelRun) else value for value in values]


class DiscoveryResponse(BaseModel):
    original_query: str
    updated_query: str
    candidates: list[DiscoveryCandidate] = Field(max_length=MAX_RESULTS)
    provenance: DiscoveryProvenance
    limitations: list[str]
