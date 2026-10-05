"""Malformed upstream replies produce a controlled API failure without saved turns."""

from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from journalpulse.api import create_app
from journalpulse.intelligence import OpenRouterConversationClient
from test_conversations_api import USER_A, chat_settings, say, start
from test_journals_api import save


def test_malformed_chat_envelope_saves_neither_side_of_a_turn(tmp_path: Path):
    settings = chat_settings(tmp_path)
    model = OpenRouterConversationClient(settings, client=httpx.Client(transport=httpx.MockTransport(
        lambda _: httpx.Response(200, json={"choices": []}),
    )))
    with TestClient(create_app(settings=settings, conversation_client=model)) as client:
        chat = start(client)
        failed = say(client, chat["id"], "Please help me reflect.")
        assert failed.status_code == 502
        restored = client.get(
            f"/v1/conversations/{chat['id']}", headers={"X-JournalPulse-User": USER_A},
        ).json()
        assert restored["messages"] == []
        assert restored["conversation"]["revision"] == chat["revision"]


def test_malformed_journal_envelope_preserves_saved_writing_only(tmp_path: Path):
    settings = chat_settings(tmp_path)
    model = OpenRouterConversationClient(settings, client=httpx.Client(transport=httpx.MockTransport(
        lambda _: httpx.Response(200, json={"choices": [None]}),
    )))
    with TestClient(create_app(settings=settings, conversation_client=model)) as client:
        entry = save(client, "A fictional quiet walk mattered to me.")
        failed = client.post(
            f"/v1/journal/entries/{entry['id']}/reflect",
            headers={"X-JournalPulse-User": USER_A}, json={"llm_consent": True},
        )
        assert failed.status_code == 502
        assert "entry is still saved" in failed.json()["detail"]
        assert "Nothing was saved" not in failed.json()["detail"]
        exported = client.get("/v1/export", headers={"X-JournalPulse-User": USER_A}).json()
        assert exported["journal_entries"] == [entry]
        assert exported["conversation_messages"] == []
        assert exported["reflections"] == []


def test_filtered_chat_is_a_controlled_decline_without_saved_turn(tmp_path: Path):
    settings = chat_settings(tmp_path)
    model = OpenRouterConversationClient(settings, client=httpx.Client(transport=httpx.MockTransport(
        lambda _: httpx.Response(200, json={"choices": [{
            "finish_reason": "content_filter", "message": None,
        }]}),
    )))
    with TestClient(create_app(settings=settings, conversation_client=model)) as client:
        chat = start(client)
        declined = say(client, chat["id"], "Fictional journal discussion.")
        assert declined.status_code == 422
        assert "declined" in declined.json()["detail"]
        assert declined.headers["X-JournalPulse-Error-Stage"] == "provider_refusal"
        restored = client.get(
            f"/v1/conversations/{chat['id']}", headers={"X-JournalPulse-User": USER_A},
        ).json()
        assert restored["messages"] == []
        assert restored["conversation"]["revision"] == chat["revision"]
