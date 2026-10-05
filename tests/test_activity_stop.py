"""Stopping in ordinary chat also withdraws timers and automatic questions."""

from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from journalpulse.api import create_app
from journalpulse.domain import ModelRun
from journalpulse.guided_action import ActivityDirective, GuidedActionContext
from journalpulse.intelligence import ConversationCompletion
from test_guided_action_boundaries import (
    OWNER_HEADERS,
    Clock,
    command,
    offered,
    reported,
    seed_conversation,
    settings,
)


class StopAndResumeChat:
    def complete(self, messages: list[dict[str, str]]) -> ConversationCompletion:
        raise AssertionError("This test must exercise the guided conversation directive")

    def complete_guided(
        self,
        messages: list[dict[str, str]],
        context: GuidedActionContext,
    ) -> ConversationCompletion:
        latest = messages[-1]["content"]
        stopping = "Stop" in latest
        return ConversationCompletion(
            reply="Of course. We can leave it here." if stopping else "Would a short quiet pause fit?",
            offer_action=not stopping,
            resource_intent="pause" if stopping else "ground",
            card_reason="" if stopping else "A short quiet pause fits your request.",
            summary="Fictional activity control test.",
            model_run=ModelRun(model="local-stop-directive-test-double", latency_ms=0, schema_valid=True),
            activity=ActivityDirective(
                move="pause" if stopping else "propose",
                goal="settle",
                selected_resource_id=None if stopping else "guided_meditation_2m",
                constraints=context.constraints,
                search_topic=None,
            ),
        )


@pytest.mark.parametrize("state", ["active", "awaiting_report", "completed"])
def test_text_stop_withdraws_automatic_checkin_and_preserves_saved_reports(tmp_path: Path, state: str):
    configured, clock = settings(tmp_path), Clock()
    chat = seed_conversation(configured, clock)
    with TestClient(
        create_app(settings=configured, clock=clock, conversation_client=StopAndResumeChat())
    ) as client:
        session = command(client, offered(client, chat), "start")
        if state != "active":
            session = command(client, session, "finish_early")
            assert session["check_in_issued"]
        if state == "completed":
            session = reported(client, session, note="Fictional unchanged outcome.")
        saved_report = session["report"]
        response = client.post(
            f"/v1/conversations/{chat.id}/messages",
            headers=OWNER_HEADERS,
            json={"client_message_id": str(uuid4()), "text": "Stop. Please do not ask any more questions."},
        )
        assert response.status_code == 200, response.text
        assert response.json()["conversation"]["activity_move"] == "pause"
        clock.now += timedelta(minutes=10)
        reread = client.get(f"/v1/activity-sessions/{session['id']}", headers=OWNER_HEADERS)
        assert reread.status_code == 200
        stopped = reread.json()
        assert stopped["status"] == ("completed" if state == "completed" else "stopped")
        assert stopped["expires_at"] is None and not stopped["check_in_issued"]
        assert stopped["report"] == saved_report
        restarted = client.post(
            f"/v1/activity-sessions/{session['id']}/commands",
            headers=OWNER_HEADERS,
            json={
                "client_request_id": str(uuid4()),
                "command": "start",
                "expected_revision": stopped["revision"],
                "expected_conversation_revision": stopped["conversation_revision"],
            },
        )
        assert restarted.status_code == 409
        assert "paused activities" in restarted.json()["detail"]
        # A subsequent explicit request changes the move and permits a fresh offer;
        # the previous timer is never resumed or made into a completion claim.
        again = client.post(
            f"/v1/conversations/{chat.id}/messages",
            headers=OWNER_HEADERS,
            json={"client_message_id": str(uuid4()), "text": "I would like a quiet pause again now."},
        )
        assert again.status_code == 200, again.text
        current = again.json()["conversation"]
        fresh = client.post(
            f"/v1/conversations/{chat.id}/activity-sessions",
            headers=OWNER_HEADERS,
            json={
                "client_request_id": str(uuid4()),
                "resource_id": "guided_meditation_2m",
                "expected_conversation_revision": current["revision"],
            },
        )
        assert fresh.status_code == 201, fresh.text
        new_started = command(client, fresh.json(), "start")
        assert new_started["status"] == "active" and new_started["report"] is None


def test_explicit_stop_control_still_offers_optional_participation_report(tmp_path: Path):
    configured, clock = settings(tmp_path), Clock()
    chat = seed_conversation(configured, clock)
    with TestClient(
        create_app(settings=configured, clock=clock, conversation_client=StopAndResumeChat())
    ) as client:
        active = command(client, offered(client, chat), "start")
        stopped = command(client, active, "stop")
        assert stopped["status"] == "stopped" and stopped["check_in_issued"]
        assert stopped["report"] is None
