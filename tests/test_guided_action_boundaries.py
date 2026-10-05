"""Independent product boundaries for the in-chat activity loop.

The recommendation fixture is trusted app data, not a model prediction. These
tests establish ownership, participation semantics and lifecycle behavior; they
do not grade Luna's language or claim an activity changes someone's feelings.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from journalpulse.activity_resources import builtin_activities
from journalpulse.api import create_app
from journalpulse.config import Settings
from journalpulse.domain import (
    ActionCard,
    Conversation,
    ConversationMode,
    Goal,
    ModelRun,
    PolicyDecision,
    SafetyMode,
    SafetyResult,
)
from journalpulse.intelligence import ConversationCompletion
from journalpulse.persistence import SQLiteRepository

OWNER = UUID("10000000-0000-4000-8000-000000000011")
OTHER = UUID("20000000-0000-4000-8000-000000000022")
OWNER_HEADERS = {"X-JournalPulse-User": str(OWNER)}
OTHER_HEADERS = {"X-JournalPulse-User": str(OTHER)}
RESOURCE_ID = "guided_meditation_2m"


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 5, 12, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


class ChatDouble:
    def complete(self, messages: list[dict[str, str]]) -> ConversationCompletion:
        del messages
        return ConversationCompletion(
            reply="Test-only reflection.",
            offer_action=False,
            resource_intent="reflect",
            card_reason="",
            summary="Fictional boundary test.",
            model_run=ModelRun(model="local-test-double", latency_ms=0, schema_valid=True),
        )


class FollowUpDouble:
    def __init__(self) -> None:
        self.calls = 0
        self.block = False
        self.started = threading.Event()
        self.release = threading.Event()

    def __call__(self, conversation, messages, session):
        del conversation, messages, session
        self.calls += 1
        if self.block:
            self.started.set()
            assert self.release.wait(10), "the test did not release its follow-up"
        return (
            "Your report is saved. We can leave it here.",
            ModelRun(model="local-follow-up-double", latency_ms=0, schema_valid=True),
        )


def settings(tmp_path: Path) -> Settings:
    return Settings(
        environment="test",
        database_path=tmp_path / "activity-boundaries.db",
        resource_catalog_path=Path(__file__).resolve().parents[1] / "assets/resources/catalog.json",
        openrouter_api_key="test-only-not-a-provider-key",
        openrouter_model="test-double",
        openrouter_base_url="https://provider.example",
        openrouter_zdr=True,
        openrouter_timeout_seconds=1,
        supabase_url=None,
        supabase_anon_key=None,
        raw_text_retention_default=False,
        analysis_rate_limit_per_minute=200,
        write_signing_key="test-only-activity-signing-key-0123456789",
    )


def seed_conversation(
    configured: Settings,
    clock: Clock,
    *,
    source: dict | None = None,
) -> Conversation:
    """Install a trusted offer while keeping recommendation behavior out of scope."""
    resource = next(item for item in builtin_activities() if item["id"] == RESOURCE_ID)
    conversation = Conversation(
        user_id=OWNER,
        created_at=clock(),
        updated_at=clock(),
        llm_consent=True,
        retain_text=False,
        locale="CA",
        prompt_version="independent-test-fixture",
        mode=ConversationMode.AI,
        source_entry_id=UUID(source["id"]) if source else None,
        source_entry_created_at=datetime.fromisoformat(source["created_at"]) if source else None,
        safety=SafetyResult(mode=SafetyMode.NORMAL, locale="CA", exploration_allowed=True),
        ready_for_action=True,
        card=ActionCard(
            resource_intent="ground",
            card_reason="A short, silent option.",
            goal=Goal.SETTLE,
            actions=[resource],
            # This legacy preview field does not describe a randomized choice.
            # The new session contract must expose null propensity and OPE=false.
            decision_preview=PolicyDecision(
                action_id=RESOURCE_ID,
                propensity=1,
                policy_name="test-fixture",
                policy_version="v1",
                safe_action_ids=[RESOURCE_ID],
                context_snapshot={},
                explanation="Trusted fixture only.",
                eligible_for_ope=False,
            ),
        ),
    )
    return SQLiteRepository(configured.database_path).create_conversation(conversation)


def offered(client: TestClient, conversation: Conversation) -> dict:
    response = client.post(
        f"/v1/conversations/{conversation.id}/activity-sessions",
        headers=OWNER_HEADERS,
        json={
            "client_request_id": str(uuid4()),
            "expected_conversation_revision": conversation.revision,
            "resource_id": RESOURCE_ID,
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def command(client: TestClient, session: dict, value: str, **overrides) -> dict:
    response = client.post(
        f"/v1/activity-sessions/{session['id']}/commands",
        headers=OWNER_HEADERS,
        json={
            "client_request_id": str(uuid4()),
            "expected_revision": session["revision"],
            "expected_conversation_revision": session["conversation_revision"],
            "command": value,
            **overrides,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def reported(client: TestClient, session: dict, **overrides) -> dict:
    response = client.post(
        f"/v1/activity-sessions/{session['id']}/report",
        headers=OWNER_HEADERS,
        json={
            "client_request_id": str(uuid4()),
            "expected_revision": session["revision"],
            "expected_conversation_revision": session["conversation_revision"],
            "participation": "not_tried",
            "state_change": "unsure",
            **overrides,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def follow_up_body(session: dict) -> dict:
    return {
        "client_request_id": str(uuid4()),
        "expected_revision": session["revision"],
        "expected_conversation_revision": session["conversation_revision"],
    }


@pytest.mark.parametrize("operation", ["read", "command", "report", "follow_up"])
def test_other_account_cannot_read_or_mutate_an_activity(tmp_path: Path, operation: str) -> None:
    configured, clock, model = settings(tmp_path), Clock(), FollowUpDouble()
    app = create_app(
        settings=configured,
        clock=clock,
        conversation_client=ChatDouble(),
        activity_follow_up_generator=model,
    )
    with TestClient(app) as client:
        session = offered(client, seed_conversation(configured, clock))
        endpoint = f"/v1/activity-sessions/{session['id']}"
        if operation == "read":
            response = client.get(endpoint, headers=OTHER_HEADERS)
        else:
            body = follow_up_body(session)
            suffix = {"command": "commands", "report": "report", "follow_up": "follow-up"}[operation]
            if operation == "command":
                body["command"] = "start"
            if operation == "report":
                body["participation"] = "completed"
            response = client.post(f"{endpoint}/{suffix}", headers=OTHER_HEADERS, json=body)
        assert response.status_code == 404
        assert model.calls == 0
        assert client.get(endpoint, headers=OWNER_HEADERS).json()["status"] == "offered"


def test_expiry_requests_one_check_in_without_claiming_participation(tmp_path: Path) -> None:
    configured, clock, model = settings(tmp_path), Clock(), FollowUpDouble()
    app = create_app(settings=configured, clock=clock, activity_follow_up_generator=model)
    with TestClient(app) as client:
        conversation = seed_conversation(configured, clock)
        session = command(client, offered(client, conversation), "start")
        assert (
            client.get(f"/v1/conversations/{conversation.id}", headers=OWNER_HEADERS).json()["conversation"][
                "status"
            ]
            == "open"
        )
        clock.now += timedelta(seconds=121)
        # Read/sync is the recovery path when a browser was closed at the deadline.
        synced = client.get(f"/v1/activity-sessions/{session['id']}", headers=OWNER_HEADERS)
        assert synced.status_code == 200, synced.text
        expired = synced.json()
        assert expired["status"] == "awaiting_report"
        assert expired["check_in_issued"] is True
        assert expired["report"] is None
        assert expired["follow_up_status"] == "none"
        assert model.calls == 0
        again = client.get(f"/v1/activity-sessions/{session['id']}", headers=OWNER_HEADERS).json()
        assert again["revision"] == expired["revision"]
        assert again["report"] is None
        assert client.get("/v1/export", headers=OWNER_HEADERS).json()["outcomes"] == []


def test_report_replay_is_exact_and_follow_up_replay_does_not_generate_twice(tmp_path: Path) -> None:
    configured, clock, model = settings(tmp_path), Clock(), FollowUpDouble()
    app = create_app(settings=configured, clock=clock, activity_follow_up_generator=model)
    with TestClient(app) as client:
        session = command(client, offered(client, seed_conversation(configured, clock)), "start")
        session = command(client, session, "finish_early")
        body = {
            **follow_up_body(session),
            "participation": "not_tried",
            "state_change": "unsure",
        }
        endpoint = f"/v1/activity-sessions/{session['id']}/report"
        first = client.post(endpoint, headers=OWNER_HEADERS, json=body)
        replay = client.post(endpoint, headers=OWNER_HEADERS, json=body)
        assert first.status_code == replay.status_code == 200
        assert replay.json()["report"]["participation"] == "not_tried"
        assert replay.json()["revision"] == first.json()["revision"]
        changed = client.post(endpoint, headers=OWNER_HEADERS, json={**body, "participation": "completed"})
        assert changed.status_code == 409
        follow_body = follow_up_body(first.json())
        follow_endpoint = f"/v1/activity-sessions/{session['id']}/follow-up"
        follow = client.post(follow_endpoint, headers=OWNER_HEADERS, json=follow_body)
        duplicate = client.post(follow_endpoint, headers=OWNER_HEADERS, json=follow_body)
        assert follow.status_code == duplicate.status_code == 200
        assert duplicate.json()["follow_up_message_id"] == follow.json()["follow_up_message_id"]
        assert model.calls == 1
        altered_follow_up = client.post(
            follow_endpoint,
            headers=OWNER_HEADERS,
            json={**follow_body, "expected_revision": follow_body["expected_revision"] + 1},
        )
        assert altered_follow_up.status_code == 409
        assert model.calls == 1
        exported = client.get("/v1/export", headers=OWNER_HEADERS).json()
        assert exported["outcomes"] == []
        activity = exported["activity_sessions"][0]
        assert activity["report"]["participation"] == "not_tried"
        assert activity["selection"]["eligible_for_ope"] is False
        assert activity["selection"]["propensity"] is None


def test_source_deletion_during_follow_up_discards_late_private_reply(tmp_path: Path) -> None:
    configured, clock, model = settings(tmp_path), Clock(), FollowUpDouble()
    model.block = True
    app = create_app(settings=configured, clock=clock, activity_follow_up_generator=model)
    with TestClient(app) as client, TestClient(app) as other:
        source_response = client.post(
            "/v1/journal/entries",
            headers=OWNER_HEADERS,
            json={"text": "FICTIONAL_PRIVATE_SOURCE: a cancelled plan disappointed me."},
        )
        assert source_response.status_code == 201
        source = source_response.json()
        session = command(
            client, offered(client, seed_conversation(configured, clock, source=source)), "start"
        )
        session = reported(client, command(client, session, "finish_early"), note="FICTIONAL_PRIVATE_REPORT")
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(
                client.post,
                f"/v1/activity-sessions/{session['id']}/follow-up",
                headers=OWNER_HEADERS,
                json=follow_up_body(session),
            )
            try:
                assert model.started.wait(10), "follow-up did not reach the local test double"
                deleted = other.delete(f"/v1/journal/entries/{source['id']}", headers=OWNER_HEADERS)
                assert deleted.status_code == 204
            finally:
                model.release.set()
            assert pending.result(timeout=10).status_code == 404
        assert client.get(f"/v1/activity-sessions/{session['id']}", headers=OWNER_HEADERS).status_code == 404
        exported = client.get("/v1/export", headers=OWNER_HEADERS)
        assert "FICTIONAL_PRIVATE_SOURCE" not in exported.text
        assert "FICTIONAL_PRIVATE_REPORT" not in exported.text
        assert exported.json()["activity_sessions"] == []


def test_listening_choice_invalidates_an_unstarted_offer(tmp_path: Path) -> None:
    configured, clock = settings(tmp_path), Clock()
    with TestClient(create_app(settings=configured, clock=clock)) as client:
        conversation = seed_conversation(configured, clock)
        session = offered(client, conversation)
        changed = client.post(
            f"/v1/conversations/{conversation.id}/preference",
            headers=OWNER_HEADERS,
            json={"client_request_id": str(uuid4()), "expected_revision": 0, "preference": "listen"},
        )
        assert changed.status_code == 200, changed.text
        for revision in (0, changed.json()["revision"]):
            rejected = client.post(
                f"/v1/activity-sessions/{session['id']}/commands",
                headers=OWNER_HEADERS,
                json={
                    **follow_up_body(session),
                    "expected_conversation_revision": revision,
                    "command": "start",
                },
            )
            assert rejected.status_code == 409
        assert (
            client.get(f"/v1/activity-sessions/{session['id']}", headers=OWNER_HEADERS).json()["status"]
            != "active"
        )


def test_message_cap_keeps_reports_available_but_allows_one_final_follow_up(tmp_path: Path) -> None:
    configured, clock, model = settings(tmp_path), Clock(), FollowUpDouble()
    app = create_app(
        settings=configured,
        clock=clock,
        conversation_client=ChatDouble(),
        activity_follow_up_generator=model,
    )
    with TestClient(app) as client:
        conversation = seed_conversation(configured, clock)
        session = command(client, offered(client, conversation), "start")
        for turn in range(20):
            response = client.post(
                f"/v1/conversations/{conversation.id}/messages",
                headers=OWNER_HEADERS,
                json={"client_message_id": str(uuid4()), "text": f"Fictional turn {turn + 1}."},
            )
            assert response.status_code == 200, response.text
        blocked = client.post(
            f"/v1/conversations/{conversation.id}/messages",
            headers=OWNER_HEADERS,
            json={"client_message_id": str(uuid4()), "text": "Another ordinary turn."},
        )
        assert blocked.status_code == 409
        # Get the canonical revisions after ordinary conversation turns.
        session = client.get(f"/v1/activity-sessions/{session['id']}", headers=OWNER_HEADERS).json()
        session = reported(client, command(client, session, "finish_early"), participation="partial")
        body = follow_up_body(session)
        endpoint = f"/v1/activity-sessions/{session['id']}/follow-up"
        first = client.post(endpoint, headers=OWNER_HEADERS, json=body)
        duplicate = client.post(endpoint, headers=OWNER_HEADERS, json=body)
        assert first.status_code == duplicate.status_code == 200
        assert first.json()["final_follow_up"] is True
        assert model.calls == 1
        new_offer = client.post(
            f"/v1/conversations/{conversation.id}/activity-sessions",
            headers=OWNER_HEADERS,
            json={
                "client_request_id": str(uuid4()),
                "resource_id": RESOURCE_ID,
                "expected_conversation_revision": first.json()["conversation_revision"],
            },
        )
        assert new_offer.status_code == 409
        assert model.calls == 1


def test_current_risk_in_report_routes_to_support_without_a_model_call(tmp_path: Path) -> None:
    configured, clock, model = settings(tmp_path), Clock(), FollowUpDouble()
    app = create_app(settings=configured, clock=clock, activity_follow_up_generator=model)
    with TestClient(app) as client:
        conversation = seed_conversation(configured, clock)
        session = command(client, offered(client, conversation), "start")
        session = reported(
            client,
            command(client, session, "finish_early"),
            participation="stopped",
            note="I have a suicide plan tonight.",
        )
        detail = client.get(f"/v1/conversations/{conversation.id}", headers=OWNER_HEADERS).json()
        assert detail["conversation"]["safety_mode"] == "support"
        follow = client.post(
            f"/v1/activity-sessions/{session['id']}/follow-up",
            headers=OWNER_HEADERS,
            json=follow_up_body(session),
        )
        assert follow.status_code == 200, follow.text
        assert model.calls == 0
        assert follow.json()["report"]["participation"] == "stopped"


def test_message_cap_does_not_allow_an_unstarted_offer_to_start(tmp_path: Path) -> None:
    configured, clock = settings(tmp_path), Clock()
    app = create_app(settings=configured, clock=clock, conversation_client=ChatDouble())
    with TestClient(app) as client:
        conversation = seed_conversation(configured, clock)
        session = offered(client, conversation)
        for turn in range(20):
            response = client.post(
                f"/v1/conversations/{conversation.id}/messages",
                headers=OWNER_HEADERS,
                json={"client_message_id": str(uuid4()), "text": f"Fictional turn {turn + 1}."},
            )
            assert response.status_code == 200, response.text
        current = client.get(f"/v1/activity-sessions/{session['id']}", headers=OWNER_HEADERS).json()
        rejected = client.post(
            f"/v1/activity-sessions/{session['id']}/commands",
            headers=OWNER_HEADERS,
            json={**follow_up_body(current), "command": "start"},
        )
        assert rejected.status_code == 409
        unchanged = client.get(f"/v1/activity-sessions/{session['id']}", headers=OWNER_HEADERS).json()
        assert unchanged["status"] == "offered"
        assert unchanged["started_at"] is None
        assert unchanged["report"] is None


def test_idle_close_clears_derived_private_text_from_activity_export(tmp_path: Path) -> None:
    configured, clock, model = settings(tmp_path), Clock(), FollowUpDouble()
    app = create_app(settings=configured, clock=clock, activity_follow_up_generator=model)
    with TestClient(app) as client:
        conversation = seed_conversation(configured, clock)
        session = command(client, offered(client, conversation), "start")
        session = reported(client, command(client, session, "finish_early"), note="FICTIONAL_PRIVATE_REPORT")
        follow = client.post(
            f"/v1/activity-sessions/{session['id']}/follow-up",
            headers=OWNER_HEADERS,
            json=follow_up_body(session),
        )
        assert follow.status_code == 200, follow.text
        clock.now += timedelta(hours=25)
        detail = client.get(f"/v1/conversations/{conversation.id}", headers=OWNER_HEADERS).json()
        assert detail["conversation"]["status"] == "closed"
        exported = client.get("/v1/export", headers=OWNER_HEADERS)
        assert "FICTIONAL_PRIVATE_REPORT" not in exported.text
        activity = exported.json()["activity_sessions"][0]
        assert activity["report"]["note"] is None
        assert activity.get("follow_up_reply") is None
        assert activity["recommendation_reason"] is None
