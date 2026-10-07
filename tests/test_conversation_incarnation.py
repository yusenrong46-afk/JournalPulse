"""A delayed turn must never cross deletion into a replacement chat identity."""

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from journalpulse.api import create_app
from journalpulse.persistence import SQLiteRepository
from test_conversations_api import USER_A, ScriptedClient, chat_settings, say, start


@pytest.mark.parametrize("retain_original", [False, True])
def test_late_turn_cannot_restore_deleted_chat_even_with_same_timestamp(tmp_path, retain_original):
    fixed = datetime(2026, 10, 6, 12, tzinfo=UTC)
    configured = chat_settings(tmp_path)

    class ReplacingClient(ScriptedClient):
        replace = None

        def complete(self, messages):
            self.replace()
            return super().complete(messages)

    model = ReplacingClient(offers=[False])
    with TestClient(
        create_app(settings=configured, conversation_client=model, clock=lambda: fixed)
    ) as client:
        cid = str(uuid4())
        original = start(client, client_request_id=cid, retain_text=retain_original)
        replacement = {}
        new_cid = str(uuid4())

        def replace():
            assert (
                client.delete(f"/v1/conversations/{cid}", headers={"X-JournalPulse-User": USER_A}).status_code
                == 204
            )
            assert client.post("/v1/conversations", headers={"X-JournalPulse-User": USER_A}, json={
                "client_request_id": cid, "llm_consent": False,
            }).status_code == 409
            replacement.update(start(client, client_request_id=new_cid, llm_consent=False, retain_text=False))

        model.replace = replace
        response = say(client, cid, "Fictional words from the deleted chat.")
        assert original["created_at"] == replacement["created_at"]
        assert response.status_code == 404
        current = client.get(f"/v1/conversations/{new_cid}", headers={"X-JournalPulse-User": USER_A}).json()
        assert current["conversation"]["mode"] == "guided"
        assert current["conversation"]["llm_consent"] is False
        assert current["conversation"]["retain_text"] is False
        assert current["messages"] == []
        repository = SQLiteRepository(configured.database_path)
        repository.close_conversation(UUID(USER_A), UUID(new_cid))
        assert repository.list_messages(UUID(USER_A), UUID(new_cid)) == []


def test_stale_browser_nonce_is_rejected_before_any_model_call(tmp_path):
    model = ScriptedClient(offers=[False])
    with TestClient(create_app(settings=chat_settings(tmp_path), conversation_client=model)) as client:
        cid = str(uuid4())
        old = start(client, client_request_id=cid)
        # A nonce mismatch still refuses a request before generation even when the
        # object exists. Deletion tombstones add another independent boundary.
        old["incarnation_id"] = str(uuid4())
        response = client.post(
            f"/v1/conversations/{cid}/messages",
            headers={"X-JournalPulse-User": USER_A},
            json={
                "client_message_id": str(uuid4()),
                "text": "Fictional stale browser draft.",
                "expected_incarnation_id": old["incarnation_id"],
            },
        )
        assert response.status_code == 409
        assert model.calls == []
