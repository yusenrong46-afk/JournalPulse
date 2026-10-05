"""Reported chat activities reach the garden through a minimal, owner-only history."""

from pathlib import Path

from fastapi.testclient import TestClient

from journalpulse.api import create_app
from test_guided_action_boundaries import (
    OTHER_HEADERS,
    OWNER_HEADERS,
    ChatDouble,
    Clock,
    FollowUpDouble,
    command,
    offered,
    reported,
    seed_conversation,
    settings,
)

PRIVATE_NOTE = "FICTIONAL_PRIVATE_NOTE about the meeting"


def app_for(tmp_path: Path):
    configured, clock = settings(tmp_path), Clock()
    app = create_app(
        settings=configured, clock=clock, conversation_client=ChatDouble(),
        activity_follow_up_generator=FollowUpDouble(),
    )
    return configured, clock, app


def test_history_lists_only_reported_activities_without_private_text(tmp_path: Path) -> None:
    configured, clock, app = app_for(tmp_path)
    with TestClient(app) as client:
        session = command(client, offered(client, seed_conversation(configured, clock)), "start")
        session = command(client, session, "finish_early")
        reported(client, session, participation="partial", fit="good", note=PRIVATE_NOTE)
        # An unstarted offer in another chat is not a garden entry.
        offered(client, seed_conversation(configured, clock))

        response = client.get("/v1/activity-history", headers=OWNER_HEADERS)
        assert response.status_code == 200, response.text
        items = response.json()["items"]
        assert len(items) == 1
        item = items[0]
        assert item["id"] == session["id"]
        assert item["participation"] == "partial"
        assert item["fit"] == "good"
        assert item["state_change"] == "unsure"
        assert item["title"] == session["resource"]["title"]
        assert set(item) == {
            "id", "conversation_id", "title", "kind", "goal", "participation", "fit",
            "state_change", "helpfulness", "reported_at",
        }
        assert PRIVATE_NOTE not in response.text
        assert "follow_up" not in response.text


def test_history_is_owner_only(tmp_path: Path) -> None:
    configured, clock, app = app_for(tmp_path)
    with TestClient(app) as client:
        session = command(client, offered(client, seed_conversation(configured, clock)), "start")
        reported(client, command(client, session, "finish_early"), participation="completed")
        assert client.get("/v1/activity-history", headers=OTHER_HEADERS).json() == {"items": []}
        assert len(client.get("/v1/activity-history", headers=OWNER_HEADERS).json()["items"]) == 1


def test_history_is_newest_first_and_bounded(tmp_path: Path) -> None:
    configured, clock, app = app_for(tmp_path)
    with TestClient(app) as client:
        ids = []
        for minutes in range(3):
            clock.now = clock.now.replace(minute=minutes * 10)
            session = command(client, offered(client, seed_conversation(configured, clock)), "start")
            session = command(client, session, "finish_early")
            ids.append(reported(client, session, participation="not_tried")["id"])
        items = client.get("/v1/activity-history?limit=2", headers=OWNER_HEADERS).json()["items"]
        assert [item["id"] for item in items] == [ids[2], ids[1]]
        assert client.get("/v1/activity-history?limit=0", headers=OWNER_HEADERS).status_code == 422
