"""Inline discovery disclosure and lifecycle boundaries with local test doubles."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi.testclient import TestClient

from journalpulse.activity_resources import verify_resource_token
from journalpulse.api import create_app
from journalpulse.config import Settings
from journalpulse.discovery import BRAVE_SEARCH_URL, OpenWebDiscoveryClient
from journalpulse.discovery_models import (
    DiscoveryCandidate,
    DiscoveryProvenance,
    DiscoveryRequest,
    DiscoveryResponse,
)
from journalpulse.domain import ConversationMessage, MessageRole, SafetyMode
from journalpulse.inline_discovery import public_search_query
from journalpulse.persistence import SQLiteRepository

OWNER = UUID("30000000-0000-4000-8000-000000000033")
OTHER = UUID("40000000-0000-4000-8000-000000000044")
HEADERS = {"X-JournalPulse-User": str(OWNER)}
NOW = datetime(2026, 10, 5, 12, tzinfo=UTC)


def configured(tmp_path: Path, **overrides) -> Settings:
    return replace(
        Settings(
            environment="test",
            database_path=tmp_path / "inline-search.db",
            resource_catalog_path=Path(__file__).resolve().parents[1] / "assets/resources/catalog.json",
            openrouter_api_key="test-only-model-key",
            openrouter_model="local-test-double",
            openrouter_base_url="https://openrouter.ai/api/v1",
            openrouter_zdr=True,
            openrouter_timeout_seconds=1,
            supabase_url=None,
            supabase_anon_key=None,
            raw_text_retention_default=False,
            search_feature_enabled=True,
            search_api_key="test-only-brave-key",
            analysis_rate_limit_per_minute=100,
            write_signing_key="test-only-inline-signing-key-0123456789",
        ),
        **overrides,
    )


def result(payload: DiscoveryRequest) -> DiscoveryResponse:
    return DiscoveryResponse(
        original_query=payload.original_query,
        updated_query=payload.original_query,
        candidates=[
            DiscoveryCandidate(
                title="Fictional quiet resource",
                url="https://resources.example.org/quiet",
                description="A synthetic snippet; no duration or accessibility was established.",
                why_selected="Its snippet discusses a quiet pause.",
            )
        ],
        provenance=DiscoveryProvenance(
            prompt_version="local-discovery-double",
            retrieved_at=NOW.isoformat(),
            candidate_count=1,
            model_runs=[],
        ),
        limitations=["Local fixture; no real search or page review."],
    )


class DiscoveryDouble:
    def __init__(self, *, block: bool = False) -> None:
        self.requests: list[DiscoveryRequest] = []
        self.block = block
        self.started, self.release = threading.Event(), threading.Event()

    def search(self, payload: DiscoveryRequest) -> DiscoveryResponse:
        self.requests.append(payload)
        if self.block:
            self.started.set()
            assert self.release.wait(10), "the test did not release the local discovery double"
        return result(payload)


def start(client: TestClient, *, consent: bool = True, source: str | None = None) -> dict:
    response = client.post(
        "/v1/conversations",
        headers=HEADERS,
        json={"llm_consent": consent, "retain_text": False, "source_entry_id": source},
    )
    assert response.status_code == 201, response.text
    return response.json()


def request_body(**overrides) -> dict:
    return {"expected_revision": 0, "llm_consent": True, "original_query": "quiet meditation", **overrides}


@pytest.mark.parametrize("private_field", ["original_query", "previous_query", "feedback"])
def test_private_user_text_is_rejected_before_any_search(tmp_path: Path, private_field: str) -> None:
    discovery = DiscoveryDouble()
    with TestClient(create_app(settings=configured(tmp_path), discovery_client=discovery)) as client:
        conversation = start(client)
        body = request_body(**{private_field: "My friend Alice cancelled at 12 Private Lane"})
        if private_field == "feedback":
            body["previous_query"] = "quiet meditation"
        response = client.post(f"/v1/conversations/{conversation['id']}/discover", headers=HEADERS, json=body)
        assert response.status_code == 422
        assert discovery.requests == []


@pytest.mark.parametrize("missing", ["conversation_consent", "request_consent", "signing_key", "owner"])
def test_inline_prerequisites_fail_before_search(tmp_path: Path, missing: str) -> None:
    discovery = DiscoveryDouble()
    settings = (
        configured(tmp_path, write_signing_key=None) if missing == "signing_key" else configured(tmp_path)
    )
    with TestClient(create_app(settings=settings, discovery_client=discovery)) as client:
        conversation = start(client, consent=missing != "conversation_consent")
        body = request_body(llm_consent=missing != "request_consent")
        headers = {"X-JournalPulse-User": str(OTHER)} if missing == "owner" else HEADERS
        response = client.post(f"/v1/conversations/{conversation['id']}/discover", headers=headers, json=body)
        assert response.status_code == {"signing_key": 503, "owner": 404}.get(missing, 409)
        assert discovery.requests == []


def test_model_refinement_cannot_add_private_terms_before_brave(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert str(request.url) == "https://openrouter.ai/api/v1/chat/completions", (
            "No Brave request may contain the model's private refinement"
        )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps({"additional_terms": "Alice Private Lane"}),
                        },
                    }
                ],
            },
        )

    settings = configured(tmp_path)
    provider = OpenWebDiscoveryClient(
        settings,
        httpx.Client(transport=httpx.MockTransport(handle)),
        query_validator=public_search_query,
    )
    with TestClient(create_app(settings=settings, discovery_client=provider)) as client:
        conversation = start(client)
        response = client.post(
            f"/v1/conversations/{conversation['id']}/discover",
            headers=HEADERS,
            json=request_body(previous_query="quiet meditation", feedback="shorter text"),
        )
        assert response.status_code == 422, response.text
    assert len(requests) == 1
    assert all(not str(request.url).startswith(BRAVE_SEARCH_URL) for request in requests)


@pytest.mark.parametrize("change", ["listen", "delete_source"])
def test_slow_results_are_not_offered_after_context_is_revoked(tmp_path: Path, change: str) -> None:
    discovery = DiscoveryDouble(block=True)
    app = create_app(settings=configured(tmp_path), discovery_client=discovery)
    with TestClient(app) as client, TestClient(app) as other:
        saved = client.post("/v1/journal/entries", headers=HEADERS, json={"text": "Fictional saved writing."})
        assert saved.status_code == 201
        source = saved.json()
        conversation = start(client, source=source["id"])
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(
                client.post,
                f"/v1/conversations/{conversation['id']}/discover",
                headers=HEADERS,
                json=request_body(),
            )
            try:
                assert discovery.started.wait(10), "the search did not reach the local double"
                if change == "listen":
                    changed = other.post(
                        f"/v1/conversations/{conversation['id']}/preference",
                        headers=HEADERS,
                        json={
                            "client_request_id": str(uuid4()),
                            "expected_revision": 0,
                            "preference": "listen",
                        },
                    )
                    assert changed.status_code == 200
                else:
                    assert (
                        other.delete(f"/v1/journal/entries/{source['id']}", headers=HEADERS).status_code
                        == 204
                    )
            finally:
                discovery.release.set()
            rejected = pending.result(timeout=10)
        assert rejected.status_code == (409 if change == "listen" else 404)
        assert "resource_token" not in rejected.text


def test_signed_search_offer_has_no_journal_data_and_cannot_grant_another_owner_access(
    tmp_path: Path,
) -> None:
    settings, discovery = configured(tmp_path), DiscoveryDouble()
    with TestClient(create_app(settings=settings, discovery_client=discovery, clock=lambda: NOW)) as client:
        saved = client.post(
            "/v1/journal/entries",
            headers=HEADERS,
            json={"text": "FICTIONAL_PRIVATE_ENTRY: a disagreement at work."},
        )
        assert saved.status_code == 201
        conversation = start(client, source=saved.json()["id"])
        response = client.post(
            f"/v1/conversations/{conversation['id']}/discover", headers=HEADERS, json=request_body()
        )
        assert response.status_code == 200, response.text
        sent = discovery.requests[0].model_dump_json()
        assert "FICTIONAL_PRIVATE_ENTRY" not in sent
        assert conversation["id"] not in sent
        offer = response.json()["offers"][0]
        verified = verify_resource_token(
            settings,
            offer["resource_token"],
            user_id=OWNER,
            conversation_id=UUID(conversation["id"]),
            conversation_incarnation_id=UUID(conversation["incarnation_id"]),
            conversation_revision=0,
            now=NOW,
        )
        assert verified["source"] == "search_snippet"
        assert verified["timer_enabled"] is False
        with pytest.raises(ValueError):
            verify_resource_token(
                settings,
                offer["resource_token"],
                user_id=OTHER,
                conversation_id=UUID(conversation["id"]),
                conversation_incarnation_id=UUID(conversation["incarnation_id"]),
                conversation_revision=0,
                now=NOW,
            )
        created = client.post(
            f"/v1/conversations/{conversation['id']}/activity-sessions",
            headers=HEADERS,
            json={
                "client_request_id": str(uuid4()),
                "expected_conversation_revision": 0,
                "resource_id": verified["id"],
                "resource_token": offer["resource_token"],
            },
        )
        assert created.status_code == 201, created.text
        assert created.json()["selection"]["eligible_for_ope"] is False
        assert created.json()["selection"]["propensity"] is None
        assert created.json()["status"] == "offered"
        assert created.json()["report"] is None


def test_unknown_snippet_metadata_cannot_satisfy_hard_activity_constraints(tmp_path: Path) -> None:
    discovery = DiscoveryDouble()
    with TestClient(create_app(settings=configured(tmp_path), discovery_client=discovery)) as client:
        conversation = start(client)
        response = client.post(
            f"/v1/conversations/{conversation['id']}/discover",
            headers=HEADERS,
            json=request_body(constraints={"time_minutes": 2, "no_audio": True, "seated": True}),
        )
        assert response.status_code == 200, response.text
        assert len(discovery.requests) == 1
        assert response.json()["offers"] == []


def test_a_capped_chat_cannot_spend_another_inline_search(tmp_path: Path) -> None:
    settings, discovery = configured(tmp_path), DiscoveryDouble()
    repository = SQLiteRepository(settings.database_path)
    with TestClient(create_app(settings=settings, discovery_client=discovery, clock=lambda: NOW)) as client:
        started = start(client)
        conversation = repository.get_conversation(OWNER, UUID(started["id"]))
        assert conversation is not None
        for index in range(20):
            moment = NOW + timedelta(microseconds=index + 1)
            user = ConversationMessage(
                conversation_id=conversation.id,
                role=MessageRole.USER,
                content=f"Fictional turn {index + 1}.",
                created_at=moment,
                safety_mode=SafetyMode.NORMAL,
                client_message_id=uuid4(),
            )
            assistant = ConversationMessage(
                conversation_id=conversation.id,
                role=MessageRole.ASSISTANT,
                content="Local fixture reply.",
                created_at=moment,
                safety_mode=SafetyMode.NORMAL,
            )
            conversation, _, _ = repository.commit_turn(
                conversation,
                user,
                assistant,
                expected_revision=conversation.revision,
            )
        response = client.post(
            f"/v1/conversations/{conversation.id}/discover",
            headers=HEADERS,
            json=request_body(expected_revision=conversation.revision),
        )
        assert response.status_code == 409
        assert discovery.requests == []


class TwoResultDiscovery(DiscoveryDouble):
    def search(self, payload: DiscoveryRequest) -> DiscoveryResponse:
        first = result(payload)
        second = first.candidates[0].model_copy(update={
            "title": "Fictional second resource", "url": "https://resources.example.org/second",
        })
        return first.model_copy(update={
            "candidates": [first.candidates[0], second],
            "provenance": first.provenance.model_copy(update={"candidate_count": 2}),
        })


def test_saving_another_offer_is_not_recorded_as_a_rejection(tmp_path: Path) -> None:
    settings = configured(tmp_path)
    with TestClient(create_app(
        settings=settings, discovery_client=TwoResultDiscovery(), clock=lambda: NOW,
    )) as client:
        conversation = start(client)
        response = client.post(
            f"/v1/conversations/{conversation['id']}/discover", headers=HEADERS, json=request_body()
        )
        assert response.status_code == 200, response.text
        offers = response.json()["offers"]
        assert len(offers) == 2
        saved = []
        for offer in offers:
            verified = verify_resource_token(
                settings, offer["resource_token"], user_id=OWNER,
                conversation_id=UUID(conversation["id"]), conversation_revision=0, now=NOW,
                conversation_incarnation_id=UUID(conversation["incarnation_id"]),
            )
            created = client.post(
                f"/v1/conversations/{conversation['id']}/activity-sessions",
                headers=HEADERS,
                json={
                    "client_request_id": str(uuid4()), "expected_conversation_revision": 0,
                    "resource_id": verified["id"], "resource_token": offer["resource_token"],
                },
            )
            assert created.status_code == 201, created.text
            saved.append(created.json()["id"])
        first = client.get(f"/v1/activity-sessions/{saved[0]}", headers=HEADERS).json()
        assert first["status"] == "stopped", "choosing another option is not a rejection"
        assert first["started_at"] is None and first["report"] is None
        assert first["expires_at"] is None and not first["check_in_issued"]
        assert first["revision"] == 1
        current = client.get(f"/v1/activity-sessions/{saved[1]}", headers=HEADERS).json()
        assert current["status"] == "offered"
