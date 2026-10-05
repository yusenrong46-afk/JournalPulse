"""API regression: a past offer must not override a person's latest direction."""

import json
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from journalpulse.activity_resources import ActivityConstraints
from journalpulse.api import create_app
from journalpulse.intelligence import OpenRouterConversationClient
from test_conversations_api import ScriptedClient, chat_settings, say, start


def test_latest_non_action_turn_withdraws_previous_readiness(tmp_path: Path):
    model = ScriptedClient(offers=[True, False, True])
    with TestClient(create_app(settings=chat_settings(tmp_path), conversation_client=model)) as client:
        chat = start(client)
        first = say(client, chat["id"], "I'd like to find one small thing.")
        assert first.json()["conversation"]["ready_for_action"] is True
        stopped = say(client, chat["id"], "Actually, please just help me understand this.")
        assert stopped.status_code == 200
        assert stopped.json()["conversation"]["ready_for_action"] is False
        restored = client.get(f"/v1/conversations/{chat['id']}").json()
        assert restored["conversation"]["ready_for_action"] is False
        fresh = say(client, chat["id"], "Now I'm ready for one small thing.")
        assert fresh.json()["conversation"]["ready_for_action"] is True


def test_empty_current_model_feelings_clear_earlier_inferred_labels(tmp_path: Path):
    calls = []

    def provider(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        output = {
            "reply": "We can use your current account.", "summary": "The person corrected an earlier label.",
            "offer_action": False, "resource_intent": "reflect", "card_reason": "",
            "feelings": ["anxious"] if len(calls) == 1 else [],
            "activity": {
                "move": "reflect", "goal": None, "selected_resource_id": None, "search_topic": None,
                "constraints": ActivityConstraints().model_dump(mode="json"),
            },
        }
        return httpx.Response(200, json={
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(output)}}],
        })

    settings = chat_settings(tmp_path)
    with httpx.Client(transport=httpx.MockTransport(provider)) as transport:
        model = OpenRouterConversationClient(settings, client=transport)
        with TestClient(create_app(settings=settings, conversation_client=model)) as client:
            chat = start(client)
            first = say(client, chat["id"], "I am anxious about a fictional presentation.")
            assert first.status_code == 200 and first.json()["conversation"]["feelings"] == ["anxious"]
            corrected = say(client, chat["id"], "That feeling passed. Please clear the earlier label.")
            assert corrected.status_code == 200
            assert corrected.json()["conversation"]["feelings"] == []
            restored = client.get(f"/v1/conversations/{chat['id']}").json()
            assert restored["conversation"]["feelings"] == []
            assert len(calls) == 2
