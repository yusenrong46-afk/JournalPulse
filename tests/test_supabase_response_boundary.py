"""Malformed storage responses must not become false empty histories or HTTP 500s."""

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from journalpulse.api import create_app
from journalpulse.journal_models import JournalEntry
from journalpulse.persistence import StorageUnavailable
from test_conversations_api import chat_settings
from test_supabase_repository import USER_ID, conversation, record, repository, turn


def _read(repo, operation: str):
    return {
        "journal_list": lambda: repo.list_journal_entries(USER_ID),
        "journal_get": lambda: repo.get_journal_entry(USER_ID, USER_ID),
        "reflection_list": lambda: repo.list_reflections(USER_ID),
        "reflection_all": lambda: repo.list_all_reflections(USER_ID),
        "reflection_get": lambda: repo.get_reflection(USER_ID, USER_ID),
        "outcomes": lambda: repo.list_outcomes(USER_ID),
        "conversation_get": lambda: repo.get_conversation(USER_ID, USER_ID),
        "messages": lambda: repo.list_messages(USER_ID, USER_ID),
        "export": lambda: repo.export_user_data(USER_ID),
    }[operation]()


@pytest.mark.parametrize("body", [{}, None, [None], [{}], [{"record": None}], [{"record": {}}]])
def test_invalid_journal_page_is_a_controlled_storage_failure(tmp_path: Path, body):
    repo = repository(tmp_path, lambda _: httpx.Response(200, content=json.dumps(body)))
    with repo.client, pytest.raises(StorageUnavailable, match="invalid response"):
        repo.list_journal_entries(USER_ID)


@pytest.mark.parametrize("operation", [
    "journal_get", "reflection_list", "reflection_all", "reflection_get",
    "outcomes", "conversation_get", "messages", "export",
])
def test_invalid_stored_record_is_a_controlled_storage_failure(tmp_path: Path, operation: str):
    repo = repository(tmp_path, lambda _: httpx.Response(200, json=[{"record": None}]))
    with repo.client, pytest.raises(StorageUnavailable, match="invalid response"):
        _read(repo, operation)


@pytest.mark.parametrize("operation", ["journal_get", "reflection_get", "conversation_get"])
def test_empty_owned_result_preserves_missing_record_semantics(tmp_path: Path, operation: str):
    repo = repository(tmp_path, lambda _: httpx.Response(200, json=[]))
    with repo.client:
        assert _read(repo, operation) is None


@pytest.mark.parametrize("operation", [
    "journal_list", "reflection_list", "reflection_all", "outcomes", "messages",
])
def test_empty_owned_page_remains_a_valid_result(tmp_path: Path, operation: str):
    repo = repository(tmp_path, lambda _: httpx.Response(200, json=[]))
    with repo.client:
        assert _read(repo, operation) == []


@pytest.mark.parametrize("operation", [
    "save_journal", "save_reflection", "create_conversation", "commit_turn",
])
def test_malformed_write_receipt_is_an_uncertain_storage_failure(tmp_path: Path, operation: str):
    # A malformed receipt cannot establish whether a write committed; retry IDs
    # must remain available rather than turning this into a user-input conflict.
    repo = repository(tmp_path, lambda _: httpx.Response(200, json={}))
    chat = conversation()
    user, assistant = turn(chat)
    entry = JournalEntry(user_id=USER_ID, text="Fictional writing.")
    with repo.client, pytest.raises(StorageUnavailable, match="invalid response"):
        {
            "save_journal": lambda: repo.save_journal_entry(entry),
            "save_reflection": lambda: repo.save_reflection(record()),
            "create_conversation": lambda: repo.create_conversation(chat),
            "commit_turn": lambda: repo.commit_turn(chat, user, assistant, expected_revision=0),
        }[operation]()


def test_storage_schema_failure_returns_503_without_private_record_values(tmp_path: Path):
    private_fixture = "Fictional private writing must not enter the response."
    repo = repository(tmp_path, lambda _: httpx.Response(200, json=[{"record": {"text": private_fixture}}]))
    app = create_app(settings=chat_settings(tmp_path), repository_factory=lambda _: repo)
    with repo.client, TestClient(app) as client:
        failed = client.get("/v1/journal/entries")
    assert failed.status_code == 503
    assert "could not confirm" in failed.json()["detail"].lower()
    assert private_fixture not in failed.text


@pytest.mark.parametrize("operation", ["delete_journal", "delete_conversation"])
@pytest.mark.parametrize("body", ["false", {}, [], None, 1])
def test_delete_receipts_require_actual_booleans(tmp_path: Path, operation: str, body):
    repo = repository(tmp_path, lambda _: httpx.Response(200, content=json.dumps(body)))
    with repo.client, pytest.raises(StorageUnavailable, match="invalid response"):
        if operation == "delete_journal":
            repo.delete_journal_entry(USER_ID, USER_ID)
        else:
            repo.delete_conversation(USER_ID, USER_ID)


@pytest.mark.parametrize("body", ["7", True, -1, {}, None])
def test_account_delete_receipts_require_nonnegative_integer_counts(tmp_path: Path, body):
    repo = repository(tmp_path, lambda _: httpx.Response(200, content=json.dumps(body)))
    with repo.client, pytest.raises(StorageUnavailable, match="invalid response"):
        repo.delete_user_data(USER_ID)


@pytest.mark.parametrize("body", [{}, {"closed": "3"}, {"closed": -1}, {"closed": True}, [], None])
def test_stale_close_receipts_require_nonnegative_integer_counts(tmp_path: Path, body):
    repo = repository(tmp_path, lambda _: httpx.Response(200, content=json.dumps(body)))
    with repo.client, pytest.raises(StorageUnavailable, match="invalid response"):
        repo.close_stale_conversations(USER_ID, now=datetime.now(UTC))


@pytest.mark.parametrize("body", [
    {"allowed": "false", "retry_after": 0},
    {"allowed": False, "retry_after": "9"},
    {"allowed": False, "retry_after": -1},
    {"allowed": False, "retry_after": True},
    {"allowed": False, "retry_after": 0},
    {},
])
def test_invalid_quota_receipts_fail_closed(tmp_path: Path, body):
    repo = repository(tmp_path, lambda _: httpx.Response(200, json=body))
    with repo.client, pytest.raises(StorageUnavailable, match="invalid response"):
        repo.consume_rate_limit(USER_ID, "generation", limit=1, window_seconds=60, now=datetime.now(UTC))


@pytest.mark.parametrize("status", [302, 307, 204])
def test_unexpected_read_status_cannot_claim_an_empty_history(tmp_path: Path, status: int):
    repo = repository(tmp_path, lambda _: httpx.Response(status))
    with repo.client, pytest.raises(StorageUnavailable):
        repo.list_journal_entries(USER_ID)


def test_rest_delete_receipt_requires_a_row_array(tmp_path: Path):
    repo = repository(tmp_path, lambda _: httpx.Response(200, json={"unexpected": "receipt"}))
    with repo.client, pytest.raises(StorageUnavailable, match="invalid response"):
        repo.delete_reflection(USER_ID, USER_ID)
