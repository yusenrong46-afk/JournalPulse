"""Behavioral boundaries for the additive in-chat lifecycle, with injected time."""

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from journalpulse.activity_lifecycle import ActivityConflict, ActivityNotFound
from journalpulse.activity_models import (
    ActivityCommandRequest,
    ActivityFollowUpRequest,
    ActivityReportRequest,
    ActivityResource,
    ActivitySelectionProvenance,
    ActivitySession,
    ActivityStatus,
)
from journalpulse.domain import (
    ActionCard,
    ActivityFollowUpDirective,
    Conversation,
    Goal,
    InteractionPreference,
    ModelRun,
    PolicyDecision,
    SafetyMode,
)
from journalpulse.journal_models import JournalEntry
from journalpulse.persistence import ConversationClosed, ConversationStale, SQLiteRepository
from journalpulse.safety import assess_safety

OWNER = UUID("00000000-0000-4000-8000-000000000001")
OTHER = UUID("00000000-0000-4000-8000-000000000002")
NOW = datetime(2026, 10, 5, 12, tzinfo=UTC)


def setup(tmp_path: Path, *, source: bool = False, retain: bool = False):
    repo = SQLiteRepository(tmp_path / "activity.db")
    entry = (
        repo.save_journal_entry(JournalEntry(user_id=OWNER, text="Fictional source", created_at=NOW))
        if source
        else None
    )
    chat = repo.create_conversation(
        Conversation(
            user_id=OWNER,
            llm_consent=True,
            locale="CA",
            prompt_version="test",
            created_at=NOW,
            updated_at=NOW,
            retain_text=retain,
            source_entry_id=entry.id if entry else None,
            source_entry_created_at=entry.created_at if entry else None,
        )
    )
    session = ActivitySession(
        user_id=OWNER,
        conversation_id=chat.id,
        source_entry_id=chat.source_entry_id,
        resource=ActivityResource(
            id="guided_meditation_2m",
            title="Quiet pause",
            format="timer",
            kind="meditation",
            duration_seconds=120,
            provenance="builtin",
        ),
        selection=ActivitySelectionProvenance(
            selection_source="llm",
            recommended_resource_id="guided_meditation_2m",
            selected_resource_id="guided_meditation_2m",
        ),
        duration_seconds=120,
        remaining_seconds=120,
        recommendation_reason="Private situation fit.",
    )
    offered = repo.offer_activity_session(
        session, request_id=session.id, expected_conversation_revision=0, now=NOW
    )
    return repo, chat, offered, entry


def command(repo, session, name, moment=NOW, request_id=None, chat_revision=0):
    request = ActivityCommandRequest(
        client_request_id=request_id or uuid4(),
        expected_revision=session.revision,
        expected_conversation_revision=chat_revision,
        command=name,
    )
    return repo.command_activity_session(OWNER, session.id, request, now=moment), request


def awaiting(repo, session):
    active, _ = command(repo, session, "start")
    return command(repo, active, "expire", NOW + timedelta(seconds=120))[0]


def report(repo, session, *, note="Private outcome.", participation="partial", safety=None, chat_revision=0):
    request = ActivityReportRequest(
        client_request_id=uuid4(),
        expected_revision=session.revision,
        expected_conversation_revision=chat_revision,
        participation=participation,
        state_change="same",
        fit="mixed",
        note=note,
    )
    saved = repo.report_activity_session(
        OWNER, session.id, request, now=NOW + timedelta(seconds=121), safety=safety
    )
    return saved, request


def test_pause_resume_uses_authoritative_deadline_and_keeps_chat_open(tmp_path):
    repo, chat, offered, _ = setup(tmp_path)
    active, start_request = command(repo, offered, "start")
    assert active.expires_at == NOW + timedelta(seconds=120)
    assert (
        repo.command_activity_session(OWNER, active.id, start_request, now=NOW + timedelta(seconds=12))
        == active
    )
    paused, _ = command(repo, active, "pause", NOW + timedelta(seconds=30.2))
    assert paused.remaining_seconds == 90
    resumed, _ = command(repo, paused, "resume", NOW + timedelta(seconds=80))
    assert resumed.expires_at == NOW + timedelta(seconds=170)
    with pytest.raises(ActivityConflict, match="not finished"):
        command(repo, resumed, "expire", NOW + timedelta(seconds=169))
    expired, expiry = command(repo, resumed, "expire", NOW + timedelta(seconds=171))
    assert expired.status == ActivityStatus.AWAITING_REPORT
    assert expired.check_in_issued and expired.report is None
    assert (
        repo.command_activity_session(OWNER, expired.id, expiry, now=NOW + timedelta(seconds=190)) == expired
    )
    assert repo.get_conversation(OWNER, chat.id).status == "open"
    assert repo.get_conversation(OWNER, chat.id).revision == 0
    assert expired.selection.propensity is None and not expired.selection.eligible_for_ope


def test_two_instances_accept_one_pause_and_request_id_conflict_changes_nothing(tmp_path):
    repo, _, offered, _ = setup(tmp_path)
    active, _ = command(repo, offered, "start")
    second = SQLiteRepository(repo.path)
    request = ActivityCommandRequest(
        client_request_id=uuid4(),
        expected_revision=active.revision,
        expected_conversation_revision=0,
        command="pause",
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        receipts = list(
            pool.map(
                lambda adapter: adapter.command_activity_session(
                    OWNER, active.id, request, now=NOW + timedelta(seconds=20)
                ),
                [repo, second],
            )
        )
    assert receipts[0] == receipts[1]
    assert receipts[0].revision == 2
    with pytest.raises(ActivityConflict, match="already in use"):
        repo.command_activity_session(
            OWNER, active.id, request.model_copy(update={"command": "stop"}), now=NOW
        )
    assert repo.get_activity_session(OTHER, active.id) is None
    with pytest.raises(ActivityNotFound):
        repo.command_activity_session(OTHER, active.id, request, now=NOW)


def test_report_is_saved_once_and_failed_model_retry_does_not_duplicate_outcome(tmp_path):
    repo, chat, offered, _ = setup(tmp_path)
    saved, report_request = report(repo, awaiting(repo, offered))
    assert saved.report.participation == "partial"
    assert repo.list_outcomes(OWNER) == [], "partial must not become a legacy completed=true outcome"
    assert repo.report_activity_session(OWNER, saved.id, report_request, now=NOW) == saved
    request = ActivityFollowUpRequest(
        client_request_id=uuid4(), expected_revision=saved.revision, expected_conversation_revision=0
    )
    claimed, owns = repo.claim_activity_follow_up(OWNER, saved.id, request, now=NOW + timedelta(seconds=122))
    assert owns
    duplicate, owns = repo.claim_activity_follow_up(
        OWNER, saved.id, request, now=NOW + timedelta(seconds=123)
    )
    assert not owns and duplicate == claimed
    failed = repo.finish_activity_follow_up(
        OWNER,
        saved.id,
        request_id=request.client_request_id,
        expected_revision=claimed.revision,
        expected_conversation_revision=0,
        expected_session_created_at=claimed.created_at,
        reply=None,
        model_run=None,
        now=NOW + timedelta(seconds=124),
    )
    assert failed.report == saved.report
    retried, owns = repo.claim_activity_follow_up(OWNER, saved.id, request, now=NOW + timedelta(seconds=125))
    assert owns and retried.follow_up_attempts == 2
    ready = repo.finish_activity_follow_up(
        OWNER,
        saved.id,
        request_id=request.client_request_id,
        expected_revision=retried.revision,
        expected_conversation_revision=0,
        expected_session_created_at=retried.created_at,
        reply="It stayed the same; we can leave it here.",
        model_run=ModelRun(model="fixture", latency_ms=1, schema_valid=True),
        now=NOW + timedelta(seconds=126),
    )
    assert ready.follow_up_status == "ready"
    assert len(repo.list_messages(OWNER, chat.id)) == 1
    assert repo.get_conversation(OWNER, chat.id).revision == 1
    assert (
        repo.finish_activity_follow_up(
            OWNER,
            saved.id,
            request_id=request.client_request_id,
            expected_revision=retried.revision,
            expected_conversation_revision=0,
            expected_session_created_at=retried.created_at,
            reply="A duplicate reply",
            model_run=None,
            now=NOW,
        )
        == ready
    )
    assert len(repo.list_messages(OWNER, chat.id)) == 1


def test_listen_invalidates_generation_and_close_clears_unretained_outcome_text(tmp_path):
    repo, chat, offered, _ = setup(tmp_path)
    saved, _ = report(repo, awaiting(repo, offered))
    request = ActivityFollowUpRequest(
        client_request_id=uuid4(), expected_revision=saved.revision, expected_conversation_revision=0
    )
    claimed, _ = repo.claim_activity_follow_up(OWNER, saved.id, request, now=NOW)
    repo.change_preference(
        OWNER,
        chat.id,
        request_id=uuid4(),
        preference=InteractionPreference.LISTEN,
        expected_revision=0,
        now=NOW,
    )
    with pytest.raises(ConversationStale):
        repo.finish_activity_follow_up(
            OWNER,
            saved.id,
            request_id=request.client_request_id,
            expected_revision=claimed.revision,
            expected_conversation_revision=0,
            expected_session_created_at=claimed.created_at,
            reply="stale",
            model_run=None,
            now=NOW,
        )
    repo.close_conversation(OWNER, chat.id)
    exported = repo.export_user_data(OWNER)
    session = exported["activity_sessions"][0]
    assert session["report"]["note"] is None
    assert session["recommendation_reason"] is None and session.get("follow_up_reply") is None
    assert exported["activity_receipts"] == []


def test_deleted_source_cascades_and_pending_reply_cannot_recreate_it(tmp_path):
    repo, chat, offered, entry = setup(tmp_path, source=True)
    saved, _ = report(repo, awaiting(repo, offered))
    request = ActivityFollowUpRequest(
        client_request_id=uuid4(), expected_revision=saved.revision, expected_conversation_revision=0
    )
    claimed, _ = repo.claim_activity_follow_up(OWNER, saved.id, request, now=NOW)
    assert repo.delete_journal_entry(OWNER, entry.id)
    assert repo.list_activity_sessions(OWNER) == []
    assert repo.export_user_data(OWNER)["activity_receipts"] == []
    with pytest.raises(ActivityNotFound):
        repo.finish_activity_follow_up(
            OWNER,
            saved.id,
            request_id=request.client_request_id,
            expected_revision=claimed.revision,
            expected_conversation_revision=0,
            expected_session_created_at=claimed.created_at,
            reply="deleted text",
            model_run=None,
            now=NOW,
        )


def test_current_risk_in_report_is_sticky_before_any_followup_and_cannot_clear(tmp_path):
    repo, chat, offered, _ = setup(tmp_path)
    note = "I plan to kill myself tonight."
    saved, _ = report(repo, awaiting(repo, offered), note=note, safety=assess_safety(note))
    assert saved.report.note == note
    assert repo.get_conversation(OWNER, chat.id).safety_mode == SafetyMode.SUPPORT
    assert repo.get_conversation(OWNER, chat.id).revision == 1


def test_new_offer_supersedes_only_unstarted_offer(tmp_path):
    repo, chat, first, _ = setup(tmp_path)
    newer = first.model_copy(update={"id": uuid4()})
    second = repo.offer_activity_session(
        newer, request_id=newer.id, expected_conversation_revision=0, now=NOW
    )
    assert repo.get_activity_session(OWNER, first.id).status == ActivityStatus.DECLINED
    active, _ = command(repo, second, "start")
    with pytest.raises(ActivityConflict, match="Finish or stop"):
        repo.offer_activity_session(
            newer.model_copy(update={"id": uuid4()}),
            request_id=uuid4(),
            expected_conversation_revision=0,
            now=NOW,
        )
    stopped, _ = command(repo, active, "stop")
    saved, _ = report(repo, stopped, participation="stopped")
    assert saved.status == ActivityStatus.STOPPED


def test_started_activity_cannot_resurrect_an_idle_chat(tmp_path):
    repo, chat, offered, _ = setup(tmp_path)
    active, _ = command(repo, offered, "start")
    with pytest.raises(ConversationClosed):
        command(repo, active, "expire", NOW + timedelta(hours=24, seconds=1))
    assert repo.get_activity_session(OWNER, active.id).report is None
    assert repo.close_stale_conversations(OWNER, now=NOW + timedelta(hours=25)) == 1
    assert repo.get_activity_session(OWNER, active.id).status == ActivityStatus.STOPPED


def test_followup_attempts_are_bounded_without_losing_the_report(tmp_path):
    repo, chat, offered, _ = setup(tmp_path)
    saved, _ = report(repo, awaiting(repo, offered))
    request = ActivityFollowUpRequest(
        client_request_id=uuid4(), expected_revision=saved.revision, expected_conversation_revision=0
    )
    for attempt in range(3):
        claimed, owns = repo.claim_activity_follow_up(OWNER, saved.id, request, now=NOW)
        assert owns and claimed.follow_up_attempts == attempt + 1
        repo.finish_activity_follow_up(
            OWNER,
            saved.id,
            request_id=request.client_request_id,
            expected_revision=claimed.revision,
            expected_conversation_revision=0,
            expected_session_created_at=claimed.created_at,
            reply=None,
            model_run=None,
            now=NOW,
        )
    with pytest.raises(ActivityConflict, match="no available"):
        repo.claim_activity_follow_up(OWNER, saved.id, request, now=NOW)
    assert repo.get_activity_session(OWNER, saved.id).report == saved.report


def test_delayed_followup_cannot_commit_into_recreated_ids(tmp_path):
    repo, chat, offered, source = setup(tmp_path, source=True)
    saved, _ = report(repo, awaiting(repo, offered))
    request = ActivityFollowUpRequest(
        client_request_id=uuid4(), expected_revision=saved.revision, expected_conversation_revision=0
    )
    original, _ = repo.claim_activity_follow_up(OWNER, saved.id, request, now=NOW)
    repo.delete_journal_entry(OWNER, source.id)
    later = NOW + timedelta(hours=1)
    restored = repo.save_journal_entry(
        source.model_copy(update={"text": "New fictional writing.", "created_at": later})
    )
    recreated = repo.create_conversation(
        chat.model_copy(
            update={"created_at": later, "updated_at": later, "source_entry_created_at": restored.created_at}
        )
    )
    fresh = repo.offer_activity_session(
        offered, request_id=offered.id, expected_conversation_revision=0, now=later
    )
    active, _ = command(repo, fresh, "start", later)
    ended, _ = command(repo, active, "expire", later + timedelta(seconds=120))
    new_report = ActivityReportRequest(
        client_request_id=uuid4(),
        expected_revision=ended.revision,
        expected_conversation_revision=0,
        participation="completed",
    )
    saved_new = repo.report_activity_session(OWNER, fresh.id, new_report, now=later + timedelta(seconds=121))
    current, _ = repo.claim_activity_follow_up(OWNER, fresh.id, request, now=later + timedelta(seconds=122))
    assert current.revision == original.revision
    with pytest.raises(ActivityConflict, match="Activity changed"):
        repo.finish_activity_follow_up(
            OWNER,
            fresh.id,
            request_id=request.client_request_id,
            expected_revision=original.revision,
            expected_conversation_revision=0,
            expected_session_created_at=original.created_at,
            reply="Obsolete private source",
            model_run=None,
            now=later + timedelta(seconds=123),
        )
    assert repo.get_activity_session(OWNER, fresh.id).report == saved_new.report
    assert repo.list_messages(OWNER, recreated.id) == []


def test_outcome_negotiation_card_uses_actual_followup_message_identity(tmp_path):
    repo, chat, offered, _ = setup(tmp_path)
    saved, _ = report(repo, awaiting(repo, offered))
    request = ActivityFollowUpRequest(
        client_request_id=uuid4(), expected_revision=saved.revision, expected_conversation_revision=0
    )
    claimed, _ = repo.claim_activity_follow_up(OWNER, saved.id, request, now=NOW)
    incorrect_message = uuid4()
    card = ActionCard(
        resource_intent="ground",
        card_reason="Another quiet option.",
        actions=[{"id": "guided_meditation_2m", "title": "Quiet meditation"}],
        offered_message_id=incorrect_message,
        goal=Goal.SETTLE,
        decision_preview=PolicyDecision(
            action_id="guided_meditation_2m",
            recommended_action_id="guided_meditation_2m",
            propensity=None,
            eligible_for_ope=False,
            policy_name="luna-guided-action",
            policy_version="test",
            safe_action_ids=["guided_meditation_2m"],
            context_snapshot={},
            explanation="A user-welcomed alternative.",
        ),
    )
    finished = repo.finish_activity_follow_up(
        OWNER,
        saved.id,
        request_id=request.client_request_id,
        expected_revision=claimed.revision,
        expected_conversation_revision=0,
        expected_session_created_at=claimed.created_at,
        reply="Would you like a different quiet pause?",
        model_run=None,
        now=NOW,
        directive=ActivityFollowUpDirective(card=card, goal=Goal.SETTLE),
    )
    current = repo.get_conversation(OWNER, chat.id)
    assert current.card is None
    assert current.activity_card.offered_message_id == finished.follow_up_message_id
    assert current.activity_card.offered_message_id != incorrect_message
    assert current.activity_card.offered_message_id == repo.list_messages(OWNER, chat.id)[0].id
    assert len(repo.list_activity_sessions(OWNER, chat.id)) == 1, (
        "proposal never automatically starts another session"
    )
