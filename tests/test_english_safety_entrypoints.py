"""New English risk phrases must reach support before any provider across all entry points."""
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from journalpulse.api import create_app
from test_conversations_api import ScriptedClient, chat_settings, say, start
from test_discovery import FakeDiscoveryClient, configured, request
from test_guided_action_boundaries import (
    OWNER_HEADERS,
    Clock,
    FollowUpDouble,
    command,
    follow_up_body,
    offered,
    reported,
    seed_conversation,
)
from test_guided_action_boundaries import settings as activity_settings
from test_journals_api import HEADERS, RecordingClient, save
from test_journals_api import settings as journal_settings
from test_research_beta_api import USER_A, reflection_payload
from test_research_beta_api import settings as reflection_settings

RISK = ["I intend to end my life tonight.", "I'm considering suicide.", "I'm gonna unalive myself tonight.",
        "I don't want to die until tonight.", "I'm not planning to kill myself until tomorrow.",
        "I won't hurt myself until tonight.", "I'm not suicidal yet.",
        "I won't kill myself until tonight.", "I will not harm myself before I say goodbye"]


class ForbiddenProvider:
    def __getattr__(self, name):
        raise AssertionError("Safety routing must not access a model provider")


@pytest.mark.parametrize("text", RISK)
def test_chat_support_bypasses_generation(tmp_path, text):
    model = ScriptedClient(offers=[True])
    with TestClient(create_app(settings=chat_settings(tmp_path), conversation_client=model)) as client:
        result = say(client, start(client)["id"], text)
        assert result.status_code == 200
        assert result.json()["conversation"]["safety_mode"] == "support"
        assert result.json()["conversation"]["ready_for_action"] is False
        assert model.calls == []


@pytest.mark.parametrize("text", RISK)
def test_journal_reflection_support_bypasses_generation(tmp_path, text):
    model = RecordingClient()
    with TestClient(create_app(settings=journal_settings(tmp_path), conversation_client=model)) as client:
        entry = save(client, text)
        result = client.post(f"/v1/journal/entries/{entry['id']}/reflect", headers=HEADERS,
                             json={"llm_consent": True})
        assert result.status_code == 200
        assert result.json()["safety"]["mode"] == "support"
        assert model.calls == []


@pytest.mark.parametrize("text", RISK)
@pytest.mark.parametrize("endpoint", ["/v1/reflections", "/v1/reflections/analyze"])
def test_standalone_reflection_support_bypasses_generation(tmp_path, text, endpoint):
    settings = replace(reflection_settings(tmp_path), openrouter_api_key="fictional-unused-key")
    with TestClient(create_app(settings=settings, intelligence_client=ForbiddenProvider())) as client:
        payload = reflection_payload(text=text, llm_consent=True)
        if endpoint.endswith("/analyze"):
            payload = {"text": text, "llm_consent": True, "locale": "CA"}
        result = client.post(endpoint, headers={"X-JournalPulse-User": USER_A}, json=payload)
        assert result.status_code in {200, 201}, result.text
        assert result.json()["safety"]["mode"] == "support"
        assert result.json()["model_run"]["fallback_reason"] == "support_mode_llm_bypassed"


@pytest.mark.parametrize("text", RISK)
def test_discovery_support_bypasses_search(tmp_path, text):
    provider = FakeDiscoveryClient()
    with TestClient(create_app(settings=configured(tmp_path), discovery_client=provider)) as client:
        result = client.post("/v1/discovery/search", json=request(original_query=text).model_dump())
        assert result.status_code == 422
        assert "Web discovery is paused" in result.json()["detail"]
        assert provider.calls == []


@pytest.mark.parametrize("text", RISK)
def test_activity_feedback_support_bypasses_followup(tmp_path, text):
    settings, clock, model = activity_settings(tmp_path), Clock(), FollowUpDouble()
    with TestClient(create_app(settings=settings, clock=clock, activity_follow_up_generator=model)) as client:
        chat = seed_conversation(settings, clock)
        session = command(client, offered(client, chat), "start")
        session = reported(client, command(client, session, "finish_early"),
                           participation="stopped", note=text)
        detail = client.get(f"/v1/conversations/{chat.id}", headers=OWNER_HEADERS).json()
        assert detail["conversation"]["safety_mode"] == "support"
        followup = client.post(f"/v1/activity-sessions/{session['id']}/follow-up", headers=OWNER_HEADERS,
                               json=follow_up_body(session))
        assert followup.status_code == 200
        assert model.calls == 0


@pytest.mark.parametrize("text", ["I dont want to die.", "I wont hurt myself.",
                                  "I'm not currently suicidal.", "I have no thoughts of suicide.",
                                  "I don't intend to end my life.", "I'm not planning to kill myself."])
def test_clear_denials_do_not_lock_the_chat(tmp_path, text):
    model = ScriptedClient(offers=[False])
    with TestClient(create_app(settings=chat_settings(tmp_path), conversation_client=model)) as client:
        result = say(client, start(client)["id"], text)
        assert result.status_code == 200
        assert result.json()["conversation"]["safety_mode"] == "normal"
        assert len(model.calls) == 1
