"""New recommendations cannot leave an obsolete unstarted activity actionable."""

import json
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient

from journalpulse.api import create_app
from journalpulse.intelligence import OpenRouterConversationClient
from test_guided_action_boundaries import (
    OWNER_HEADERS,
    Clock,
    command,
    follow_up_body,
    offered,
    reported,
    seed_conversation,
    settings,
)


def provider_reply(minutes: int, *, choosing: bool = True, stop: bool = False) -> httpx.Response:
    content = {
        "reply": f"A {minutes}-minute option is available." if choosing else "You can finish when you want.",
        "offer_action": choosing,
        "resource_intent": "ground",
        "card_reason": "This fits your available time." if choosing else "",
        "summary": "A fictional quiet pause.",
        "feelings": [],
        "activity": {
            "move": "pause" if stop else "negotiate" if choosing else "reflect",
            "goal": "settle",
            "selected_resource_id": f"guided_meditation_{minutes}m" if choosing else None,
            "search_topic": None,
            "constraints": {
                "time_minutes": minutes,
                "no_audio": True,
                "no_video": True,
                "seated": True,
                "avoid_breath_focus": False,
            },
        },
    }
    return httpx.Response(200, json={
        "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(content)}}],
    })


@pytest.mark.parametrize("state", ["offered", "active", "paused", "awaiting_report"])
def test_negotiation_withdraws_only_unstarted_offers_and_rejects_old_start(
    tmp_path: Path, state: str
) -> None:
    configured, clock = settings(tmp_path), Clock()
    replies = iter((provider_reply(2), provider_reply(1, choosing=state == "offered")))
    model = OpenRouterConversationClient(
        configured, client=httpx.Client(transport=httpx.MockTransport(lambda _: next(replies)))
    )
    with TestClient(create_app(settings=configured, clock=clock, conversation_client=model)) as client:
        created = client.post("/v1/conversations", headers=OWNER_HEADERS, json={"llm_consent": True})
        assert created.status_code == 201, created.text
        endpoint = f"/v1/conversations/{created.json()['id']}"
        first = client.post(endpoint + "/messages", headers=OWNER_HEADERS, json={
            "client_message_id": str(uuid4()), "text": "I have two minutes for a quiet pause.",
        })
        assert first.status_code == 200, first.text
        saved = client.post(endpoint + "/activity-sessions", headers=OWNER_HEADERS, json={
            "client_request_id": str(uuid4()), "expected_conversation_revision": 1,
            "resource_id": "guided_meditation_2m",
        })
        assert saved.status_code == 201, saved.text
        session = saved.json()
        if state != "offered":
            session = command(client, session, "start")
        if state == "paused":
            session = command(client, session, "pause")
        elif state == "awaiting_report":
            session = command(client, session, "finish_early")
        stale_start = {
            "client_request_id": str(uuid4()), "expected_revision": session["revision"],
            "expected_conversation_revision": 1, "command": "start",
        }
        clock.now += timedelta(seconds=1)
        corrected = client.post(endpoint + "/messages", headers=OWNER_HEADERS, json={
            "client_message_id": str(uuid4()), "text": "Actually I only have one minute. Make it shorter.",
        })
        assert corrected.status_code == 200, corrected.text
        assert corrected.json()["conversation"]["activity_constraints"]["time_minutes"] == 1
        current = client.get(endpoint + "/activity-sessions", headers=OWNER_HEADERS).json()
        if state != "offered":
            assert current["status"] == state
            assert current["revision"] == session["revision"]
            assert current["expires_at"] == session["expires_at"]
            return
        assert current["status"] == "stopped"
        assert current["started_at"] is None and not current["check_in_issued"]
        assert current["revision"] == session["revision"] + 1
        assert current["conversation_revision"] == 2
        old_endpoint = f"/v1/activity-sessions/{current['id']}/commands"
        # A request prepared before the new signed turn cannot win after that commit.
        assert client.post(old_endpoint, headers=OWNER_HEADERS, json=stale_start).status_code == 409
        refreshed_start = {
            **stale_start, "client_request_id": str(uuid4()), "expected_revision": current["revision"],
            "expected_conversation_revision": 2,
        }
        assert client.post(old_endpoint, headers=OWNER_HEADERS, json=refreshed_start).status_code == 409
        replacement = client.post(endpoint + "/activity-sessions", headers=OWNER_HEADERS, json={
            "client_request_id": str(uuid4()), "expected_conversation_revision": 2,
            "resource_id": "guided_meditation_1m",
        })
        assert replacement.status_code == 201, replacement.text
        started = command(client, replacement.json(), "start")
        assert started["duration_seconds"] == 60
        assert started["resource"]["id"] == "guided_meditation_1m"


def test_explicit_pause_in_pending_follow_up_stops_another_active_session(tmp_path: Path) -> None:
    configured, clock = settings(tmp_path), Clock()
    model = OpenRouterConversationClient(
        configured,
        client=httpx.Client(transport=httpx.MockTransport(
            lambda _: provider_reply(2, choosing=False, stop=True)
        )),
    )
    chat = seed_conversation(configured, clock)
    with TestClient(create_app(settings=configured, clock=clock, conversation_client=model)) as client:
        first = reported(
            client, command(client, command(client, offered(client, chat), "start"), "stop"),
            note="Please stop the activities, I am done.",
        )
        second = command(client, offered(client, chat), "start")
        reply = client.post(
            f"/v1/activity-sessions/{first['id']}/follow-up",
            headers=OWNER_HEADERS, json=follow_up_body(first),
        )
        assert reply.status_code == 200, reply.text
        assert reply.json()["follow_up_status"] == "ready"
        assert reply.json()["report"] == first["report"]
        current = client.get(f"/v1/activity-sessions/{second['id']}", headers=OWNER_HEADERS).json()
        assert current["status"] == "stopped"
        assert current["expires_at"] is None
        assert not current["check_in_issued"]
