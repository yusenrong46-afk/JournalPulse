import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import httpx
import pytest

from journalpulse.config import Settings
from journalpulse.domain import (
    AffectiveState,
    Conversation,
    ConversationMessage,
    InteractionPreference,
    MessageRole,
    OutcomeRecord,
    PolicyDecision,
    ReflectionCopy,
    ReflectionRecord,
    SafetyMode,
    SafetyResult,
    TargetState,
)
from journalpulse.journal_models import JournalEntry
from journalpulse.persistence import (
    ConversationNotFound,
    JournalEntryNotFound,
    SQLiteRepository,
    SupabaseRepository,
)
from journalpulse.signing import sign_text

OWNER = UUID("00000000-0000-4000-8000-000000000001")
OTHER = UUID("00000000-0000-4000-8000-000000000002")
KEY = "journal-test-signing-key-0123456789abcdef"


def entry(**updates: object) -> JournalEntry:
    data = {"id": uuid4(), "user_id": OWNER, "created_at": datetime.now(UTC), "text": "A quiet moment."}
    return JournalEntry.model_validate({**data, **updates})


def chat(source: JournalEntry) -> Conversation:
    return Conversation(
        id=uuid4(),
        user_id=OWNER,
        llm_consent=True,
        locale="CA",
        source_entry_id=source.id,
        source_entry_created_at=source.created_at,
        prompt_version="journal-test",
        created_at=source.created_at,
        updated_at=source.created_at,
    )


def test_exact_retry_returns_original_entry_and_changed_reuse_conflicts(tmp_path: Path):
    repo = SQLiteRepository(tmp_path / "journal.db")
    original = entry(text="  Exact writing.\n")
    assert repo.save_journal_entry(original) == original
    retry = original.model_copy(update={"created_at": original.created_at + timedelta(minutes=1)})
    assert repo.save_journal_entry(retry) == original
    for changed in (
        original.model_copy(update={"text": "Exact writing."}),
        original.model_copy(update={"user_id": OTHER}),
    ):
        with pytest.raises(ValueError):
            repo.save_journal_entry(changed)
    assert repo.get_journal_entry(OTHER, original.id) is None
    assert repo.delete_journal_entry(OTHER, original.id) is False
    assert repo.get_journal_entry(OWNER, original.id) == original


def test_journal_pagination_is_owned_deterministic_and_export_has_every_entry(tmp_path: Path):
    repo = SQLiteRepository(tmp_path / "journal.db")
    moment = datetime.now(UTC)
    owned = [entry(created_at=moment + timedelta(seconds=index)) for index in range(53)]
    for saved in [*owned, entry(user_id=OTHER)]:
        repo.save_journal_entry(saved)
    assert repo.list_journal_entries(OWNER, limit=2, offset=1) == [owned[-2], owned[-3]]
    assert len(repo.export_user_data(OWNER)["journal_entries"]) == 53
    assert repo.delete_user_data(OWNER) == 53
    assert repo.list_journal_entries(OWNER) == []
    assert len(repo.list_journal_entries(OTHER)) == 1


def test_source_deletion_removes_chat_messages_receipts_and_rejects_delayed_turn(tmp_path: Path):
    repo = SQLiteRepository(tmp_path / "journal.db")
    source = repo.save_journal_entry(entry())
    conversation = repo.create_conversation(chat(source))
    conversation = repo.change_preference(
        OWNER,
        conversation.id,
        request_id=uuid4(),
        preference=InteractionPreference.LISTEN,
        expected_revision=0,
        now=datetime.now(UTC),
    )
    user = ConversationMessage(
        conversation_id=conversation.id,
        client_message_id=uuid4(),
        role=MessageRole.USER,
        content="Hello",
        safety_mode=SafetyMode.NORMAL,
    )
    assistant = ConversationMessage(
        conversation_id=conversation.id,
        role=MessageRole.ASSISTANT,
        content="I hear you.",
        safety_mode=SafetyMode.NORMAL,
    )
    conversation, _, _ = repo.commit_turn(conversation, user, assistant, expected_revision=1)
    delayed_user = user.model_copy(update={"id": uuid4(), "client_message_id": uuid4()})
    delayed_reply = assistant.model_copy(update={"id": uuid4()})
    assert repo.delete_journal_entry(OWNER, source.id) is True
    assert repo.export_user_data(OWNER)["conversations"] == []
    assert repo.export_user_data(OWNER)["conversation_messages"] == []
    assert repo.export_user_data(OWNER)["conversation_preference_requests"] == []
    with pytest.raises(ConversationNotFound):
        repo.commit_turn(conversation, delayed_user, delayed_reply, expected_revision=2)
    with pytest.raises(JournalEntryNotFound):
        repo.create_conversation(chat(source))


def test_conversation_id_cannot_switch_to_another_journal_source(tmp_path: Path):
    repo = SQLiteRepository(tmp_path / "journal.db")
    first, second = repo.save_journal_entry(entry()), repo.save_journal_entry(entry())
    conversation = repo.create_conversation(chat(first))
    assert repo.create_conversation(conversation) == conversation
    with pytest.raises(ValueError):
        repo.create_conversation(
            conversation.model_copy(
                update={
                    "source_entry_id": second.id,
                    "source_entry_created_at": second.created_at,
                }
            )
        )


def test_source_deletion_removes_accepted_reflection_and_its_outcome(tmp_path: Path):
    repo = SQLiteRepository(tmp_path / "journal.db")
    source = repo.save_journal_entry(entry())
    conversation = repo.create_conversation(chat(source))
    reflection = ReflectionRecord(
        user_id=OWNER,
        state=AffectiveState(valence=0, arousal=0.5, agency=0.5),
        target=TargetState(goal="reflect"),
        reflection=ReflectionCopy(
            summary="A source-derived summary.",
            interpretation="A possible reading.",
            reflection_question="What stood out?",
        ),
        safety=SafetyResult(mode=SafetyMode.NORMAL, locale="CA", exploration_allowed=True),
        decision=PolicyDecision(
            action_id="pause",
            propensity=1,
            policy_name="test",
            policy_version="1",
            safe_action_ids=["pause"],
            context_snapshot={},
            explanation="A reviewed pause.",
        ),
    )
    accepted = repo.accept_conversation(OWNER, conversation.id, reflection, expected_revision=0)
    repo.save_outcome(OutcomeRecord(user_id=OWNER, decision_id=accepted.decision.decision_id, completed=True))
    assert len(repo.export_user_data(OWNER)["reflections"]) == 1
    assert repo.delete_journal_entry(OWNER, source.id)
    exported = repo.export_user_data(OWNER)
    for table in ("journal_entries", "reflections", "outcomes", "conversations"):
        assert exported[table] == []


def test_delayed_start_cannot_attach_to_a_recreated_entry_uuid(tmp_path: Path):
    repo = SQLiteRepository(tmp_path / "journal.db")
    source = repo.save_journal_entry(entry())
    delayed_start = chat(source)
    assert repo.delete_journal_entry(OWNER, source.id)
    recreated = source.model_copy(update={"created_at": source.created_at + timedelta(seconds=1)})
    repo.save_journal_entry(recreated)
    with pytest.raises(JournalEntryNotFound):
        repo.create_conversation(delayed_start)
    assert repo.get_conversation(OWNER, delayed_start.id) is None


def test_supabase_journal_mutations_use_signed_owner_bound_functions(tmp_path: Path):
    saved = entry()
    calls: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer user-session"
        arguments = json.loads(request.content)
        assert arguments["signature"] == sign_text(arguments["payload"], KEY)
        envelope = json.loads(arguments["payload"])
        assert envelope["user_id"] == str(OWNER)
        calls.append((request.url.path, envelope))
        return httpx.Response(200, json=saved.model_dump(mode="json") if "save" in request.url.path else True)

    settings = Settings(
        environment="test",
        database_path=tmp_path / "unused.db",
        resource_catalog_path=Path("unused.json"),
        openrouter_api_key=None,
        openrouter_model="unused",
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_zdr=True,
        openrouter_timeout_seconds=2,
        raw_text_retention_default=False,
        supabase_url="https://example.supabase.co",
        supabase_anon_key="public",
        write_signing_key=KEY,
    )
    repo = SupabaseRepository(
        settings, "user-session", client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    assert repo.save_journal_entry(saved) == saved
    assert repo.delete_journal_entry(OWNER, saved.id) is True
    assert calls[0][0].endswith("rpc/jp_save_journal_entry")
    assert calls[0][1]["purpose"] == "save_journal_entry"
    assert calls[1][0].endswith("rpc/jp_delete_journal_entry")
    assert calls[1][1]["purpose"] == "delete_journal_entry"
