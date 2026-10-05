from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from math import ceil
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID, uuid5

import httpx
from pydantic import BaseModel

from .activity_lifecycle import (
    ActivityConflict,
    ActivityNotFound,
    apply_activity_command,
    invalidated_activity,
)
from .activity_models import (
    FOLLOW_UP_LEASE_SECONDS,
    MAX_ACTIVITY_RECEIPTS,
    MAX_FOLLOW_UP_ATTEMPTS,
    ActivityCommandRequest,
    ActivityFollowUpRequest,
    ActivityReportRequest,
    ActivitySession,
    ActivityStatus,
)
from .config import Settings
from .domain import (
    ActivityFollowUpDirective,
    Conversation,
    ConversationMessage,
    ConversationRequestInputs,
    ConversationStatus,
    InteractionPreference,
    MessageRole,
    ModelRun,
    OutcomeRecord,
    ReflectionRecord,
    SafetyMode,
    SafetyResult,
)
from .http_clients import managed_http_client
from .journal_models import JournalEntry
from .signing import readiness_probe, signed_payload

EXPORT_PAGE_SIZE = 500
STALE_AFTER = timedelta(hours=24)


class DuplicateOutcomeError(ValueError):
    pass


class ConversationNotFound(LookupError):
    pass


class ConversationClosed(RuntimeError):
    pass


class ConversationStale(RuntimeError):
    """The conversation changed after this request read it; nothing was written."""


class ConversationAlreadyAccepted(RuntimeError):
    pass


class StorageUnavailable(RuntimeError):
    pass


class JournalEntryNotFound(LookupError):
    """An owned source entry disappeared before the conversation could be created."""


Turn = tuple[Conversation, ConversationMessage, ConversationMessage]


def require_same_turn_request(
    stored: ConversationMessage,
    *,
    text: str | None,
    inputs: ConversationRequestInputs | None,
) -> None:
    """Replay verified requests only; deleting text also removes our ability to verify it.

    Keep no text hash that could reveal short sensitive messages by guessing. Legacy
    messages have no reported-input snapshot, so only their retained text is checked.
    """
    if stored.content is None:
        raise ValueError("This message's text is no longer retained; its retry cannot be verified.")
    if stored.content != text or (stored.request_inputs is not None and stored.request_inputs != inputs):
        raise ValueError("Message request ID is already in use")


class Repository(Protocol):
    def save_journal_entry(self, entry: JournalEntry) -> JournalEntry: ...
    def get_journal_entry(self, user_id: UUID, entry_id: UUID) -> JournalEntry | None: ...
    def list_journal_entries(
        self,
        user_id: UUID,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[JournalEntry]: ...
    def delete_journal_entry(self, user_id: UUID, entry_id: UUID) -> bool: ...
    def save_reflection(self, record: ReflectionRecord) -> ReflectionRecord: ...
    def save_outcome(self, record: OutcomeRecord) -> OutcomeRecord: ...
    def list_reflections(self, user_id: UUID, *, limit: int, offset: int) -> list[ReflectionRecord]: ...
    def list_all_reflections(self, user_id: UUID) -> list[ReflectionRecord]: ...
    def get_reflection(self, user_id: UUID, reflection_id: UUID) -> ReflectionRecord | None: ...
    def list_outcomes(self, user_id: UUID) -> list[OutcomeRecord]: ...
    def delete_reflection(self, user_id: UUID, reflection_id: UUID) -> bool: ...
    def export_user_data(self, user_id: UUID) -> dict: ...
    def delete_user_data(self, user_id: UUID) -> int: ...
    def create_conversation(self, conversation: Conversation) -> Conversation: ...
    def get_conversation(self, user_id: UUID, conversation_id: UUID) -> Conversation | None: ...
    def list_messages(self, user_id: UUID, conversation_id: UUID) -> list[ConversationMessage]: ...
    def commit_turn(
        self,
        conversation: Conversation,
        user_message: ConversationMessage,
        assistant_message: ConversationMessage,
        *,
        expected_revision: int,
    ) -> Turn: ...
    def accept_conversation(
        self,
        user_id: UUID,
        conversation_id: UUID,
        record: ReflectionRecord,
        *,
        expected_revision: int,
    ) -> ReflectionRecord: ...
    def change_preference(
        self,
        user_id: UUID,
        conversation_id: UUID,
        *,
        request_id: UUID,
        preference: InteractionPreference,
        expected_revision: int,
        now: datetime,
    ) -> Conversation: ...
    def close_conversation(self, user_id: UUID, conversation_id: UUID) -> Conversation | None: ...
    def delete_conversation(self, user_id: UUID, conversation_id: UUID) -> bool: ...
    def close_stale_conversations(self, user_id: UUID, *, now: datetime) -> int: ...
    def get_activity_session(self, user_id: UUID, session_id: UUID) -> ActivitySession | None: ...
    def list_activity_sessions(
        self, user_id: UUID, conversation_id: UUID | None = None
    ) -> list[ActivitySession]: ...
    def offer_activity_session(
        self,
        session: ActivitySession,
        *,
        request_id: UUID,
        expected_conversation_revision: int,
        now: datetime,
    ) -> ActivitySession: ...
    def command_activity_session(
        self, user_id: UUID, session_id: UUID, request: ActivityCommandRequest, *, now: datetime
    ) -> ActivitySession: ...
    def report_activity_session(
        self,
        user_id: UUID,
        session_id: UUID,
        request: ActivityReportRequest,
        *,
        now: datetime,
        safety: SafetyResult | None = None,
    ) -> ActivitySession: ...
    def claim_activity_follow_up(
        self, user_id: UUID, session_id: UUID, request: ActivityFollowUpRequest, *, now: datetime
    ) -> tuple[ActivitySession, bool]: ...
    def finish_activity_follow_up(
        self,
        user_id: UUID,
        session_id: UUID,
        *,
        request_id: UUID,
        expected_revision: int,
        expected_conversation_revision: int,
        expected_session_created_at: datetime,
        reply: str | None,
        model_run: ModelRun | None,
        now: datetime,
        directive: ActivityFollowUpDirective | None = None,
    ) -> ActivitySession: ...
    def consume_rate_limit(
        self, user_id: UUID, bucket: str, *, limit: int, window_seconds: int, now: datetime
    ) -> tuple[bool, int]: ...


class SQLiteRepository:
    """Local and test adapter. Enforces the same lifecycle rules as the database functions."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=15)
        connection.row_factory = sqlite3.Row
        try:
            # SQLite's own context commits/rolls back, but does not close its fd.
            with connection:
                yield connection
        finally:
            connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """One serialized write transaction, the SQLite analogue of a locked row."""
        connection = sqlite3.connect(self.path, timeout=15, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS activity_sessions (
                    id TEXT PRIMARY KEY, user_id TEXT NOT NULL, conversation_id TEXT NOT NULL,
                    source_entry_id TEXT, status TEXT NOT NULL, revision INTEGER NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL, payload_json TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_activity_one_nonterminal
                    ON activity_sessions(user_id,conversation_id)
                    WHERE status IN ('offered','active','paused','awaiting_report');
                CREATE INDEX IF NOT EXISTS idx_activity_owner_created
                    ON activity_sessions(user_id,created_at DESC,id DESC);
                CREATE TABLE IF NOT EXISTS activity_receipts (
                    id TEXT NOT NULL, user_id TEXT NOT NULL, session_id TEXT NOT NULL,
                    operation TEXT NOT NULL, request_hash TEXT NOT NULL, created_at TEXT NOT NULL,
                    PRIMARY KEY(user_id,id)
                );
                CREATE INDEX IF NOT EXISTS idx_activity_receipts_session ON activity_receipts(session_id);
                CREATE TABLE IF NOT EXISTS journal_entries (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_journal_entries_user_created
                    ON journal_entries(user_id, created_at DESC, id DESC);
                -- Match the chronological expression used by pagination; a raw
                -- ISO-string index cannot order differing offsets correctly.
                CREATE INDEX IF NOT EXISTS idx_journal_entries_user_chronological
                    ON journal_entries(user_id, julianday(created_at) DESC, id DESC);
                CREATE TABLE IF NOT EXISTS reflections (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    text TEXT,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_reflections_user_created
                    ON reflections(user_id, created_at DESC);
                CREATE TABLE IF NOT EXISTS outcomes (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    decision_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_outcomes_user_decision_unique
                    ON outcomes(user_id, decision_id);
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_conversations_user_updated
                    ON conversations(user_id, status, updated_at);
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    client_message_id TEXT,
                    role TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_conversation_created
                    ON conversation_messages(conversation_id, created_at);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_client_id
                    ON conversation_messages(conversation_id, client_message_id)
                    WHERE client_message_id IS NOT NULL;
                CREATE TABLE IF NOT EXISTS rate_limit_events (
                    user_id TEXT NOT NULL,
                    bucket TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_rate_limit_user_bucket
                    ON rate_limit_events(user_id, bucket, created_at);
                CREATE TABLE IF NOT EXISTS conversation_preference_requests (
                    id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    preference TEXT NOT NULL,
                    expected_revision INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (conversation_id, id),
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                """
            )
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(conversations)")}
            if "revision" not in columns:
                connection.execute("ALTER TABLE conversations ADD COLUMN revision INTEGER NOT NULL DEFAULT 0")
            if "source_entry_id" not in columns:
                connection.execute("ALTER TABLE conversations ADD COLUMN source_entry_id TEXT")
            if "source_entry_created_at" not in columns:
                connection.execute("ALTER TABLE conversations ADD COLUMN source_entry_created_at TEXT")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversations_source_entry ON conversations(source_entry_id)"
            )
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(reflections)")}
            if "conversation_id" not in columns:
                connection.execute("ALTER TABLE reflections ADD COLUMN conversation_id TEXT")
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_reflections_conversation
                    ON reflections(conversation_id) WHERE conversation_id IS NOT NULL
                """
            )

    # Inline activity sessions -------------------------------------------------------

    def _activity_chat_locked(
        self,
        connection: sqlite3.Connection,
        user_id: UUID,
        conversation_id: UUID,
    ) -> Conversation:
        row = connection.execute(
            "SELECT * FROM conversations WHERE id = ? AND user_id = ?",
            (str(conversation_id), str(user_id)),
        ).fetchone()
        if row is None:
            raise ConversationNotFound(str(conversation_id))
        current = self._conversation(row)
        if current.source_entry_id is not None:
            if current.source_entry_created_at is None:
                raise JournalEntryNotFound(str(current.source_entry_id))
            source = connection.execute(
                "SELECT id FROM journal_entries WHERE id = ? AND user_id = ? AND created_at = ?",
                (str(current.source_entry_id), str(user_id), current.source_entry_created_at.isoformat()),
            ).fetchone()
            if source is None:
                raise JournalEntryNotFound(str(current.source_entry_id))
        return current

    @staticmethod
    def _activity_row(row: sqlite3.Row) -> ActivitySession:
        return ActivitySession.model_validate_json(row["payload_json"])

    def _activity_locked(
        self,
        connection: sqlite3.Connection,
        user_id: UUID,
        session_id: UUID,
    ) -> tuple[Conversation, ActivitySession]:
        row = connection.execute(
            "SELECT * FROM activity_sessions WHERE id = ? AND user_id = ?",
            (str(session_id), str(user_id)),
        ).fetchone()
        if row is None:
            raise ActivityNotFound("Activity not found")
        session = self._activity_row(row)
        return self._activity_chat_locked(connection, user_id, session.conversation_id), session

    @staticmethod
    def _activity_guard(
        conversation: Conversation,
        session: ActivitySession | None,
        expected_conversation_revision: int,
        expected_revision: int | None = None,
        *,
        allow_support: bool = False,
        allow_listen: bool = True,
        now: datetime | None = None,
    ) -> None:
        if conversation.status != ConversationStatus.OPEN or (
            now is not None and conversation.updated_at < now - STALE_AFTER
        ):
            raise ConversationClosed(str(conversation.id))
        if conversation.revision != expected_conversation_revision:
            raise ConversationStale(str(conversation.id))
        if session is not None and session.revision != expected_revision:
            raise ActivityConflict("Activity changed")
        if not allow_support and conversation.safety_mode == SafetyMode.SUPPORT:
            raise ActivityConflict("Support mode pauses ordinary activities")
        if not allow_listen and conversation.interaction_preference == InteractionPreference.LISTEN:
            raise ActivityConflict("This chat is set to Just talk")
        if not allow_listen and conversation.activity_move == "pause":
            raise ActivityConflict("This chat has paused activities")

    @staticmethod
    def _activity_input_hash(operation: str, body: dict[str, Any]) -> str:
        canonical = json.dumps({"operation": operation, **body}, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    @staticmethod
    def _activity_receipt(
        connection: sqlite3.Connection,
        user_id: UUID,
        session_id: UUID,
        request_id: UUID,
        operation: str,
        request_hash: str,
    ) -> bool:
        receipt = connection.execute(
            "SELECT * FROM activity_receipts WHERE user_id = ? AND id = ?",
            (str(user_id), str(request_id)),
        ).fetchone()
        if receipt is not None:
            if (receipt["session_id"], receipt["operation"], receipt["request_hash"]) != (
                str(session_id),
                operation,
                request_hash,
            ):
                raise ActivityConflict("Activity request ID is already in use")
            return True
        count = connection.execute(
            "SELECT COUNT(*) FROM activity_receipts WHERE session_id = ?",
            (str(session_id),),
        ).fetchone()[0]
        # Leave room for stop/report/follow-up even after repeated pause controls.
        limit = 48 if operation in {"start", "pause", "resume"} else MAX_ACTIVITY_RECEIPTS
        if count >= limit:
            raise ActivityConflict("Activity control limit reached; finish the activity to check in")
        return False

    @staticmethod
    def _store_activity_receipt(
        connection: sqlite3.Connection,
        user_id: UUID,
        session_id: UUID,
        request_id: UUID,
        operation: str,
        request_hash: str,
        now: datetime,
    ) -> None:
        connection.execute(
            "INSERT INTO activity_receipts (id,user_id,session_id,operation,request_hash,created_at) "
            "VALUES (?,?,?,?,?,?)",
            (str(request_id), str(user_id), str(session_id), operation, request_hash, now.isoformat()),
        )

    @staticmethod
    def _write_activity(connection: sqlite3.Connection, session: ActivitySession) -> None:
        connection.execute(
            "UPDATE activity_sessions SET status=?,revision=?,updated_at=?,payload_json=? WHERE id=?",
            (
                session.status.value,
                session.revision,
                session.updated_at.isoformat(),
                json.dumps(session.storage_payload()),
                str(session.id),
            ),
        )

    def get_activity_session(self, user_id: UUID, session_id: UUID) -> ActivitySession | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM activity_sessions WHERE user_id=? AND id=?",
                (str(user_id), str(session_id)),
            ).fetchone()
        return self._activity_row(row) if row else None

    def list_activity_sessions(
        self, user_id: UUID, conversation_id: UUID | None = None
    ) -> list[ActivitySession]:
        query = "SELECT * FROM activity_sessions WHERE user_id=?"
        parameters = [str(user_id)]
        if conversation_id is not None:
            query += " AND conversation_id=?"
            parameters.append(str(conversation_id))
        query += " ORDER BY julianday(created_at) DESC,id DESC"
        with self.connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._activity_row(row) for row in rows]

    def offer_activity_session(
        self,
        session: ActivitySession,
        *,
        request_id: UUID,
        expected_conversation_revision: int,
        now: datetime,
    ) -> ActivitySession:
        body = {
            "expected_conversation_revision": expected_conversation_revision,
            "resource_id": session.resource.id,
            "duration_seconds": session.duration_seconds,
        }
        request_hash = self._activity_input_hash("offer", body)
        with self.transaction() as connection:
            conversation = self._activity_chat_locked(connection, session.user_id, session.conversation_id)
            existing = connection.execute(
                "SELECT * FROM activity_sessions WHERE id=?",
                (str(session.id),),
            ).fetchone()
            if existing is not None:
                saved = self._activity_row(existing)
                if (saved.user_id, saved.conversation_id) != (session.user_id, session.conversation_id):
                    raise ActivityConflict("Activity request ID is already in use")
                if self._activity_receipt(
                    connection, session.user_id, session.id, request_id, "offer", request_hash
                ):
                    return saved
            self._activity_guard(
                conversation, None, expected_conversation_revision, allow_listen=False, now=now
            )
            if (
                connection.execute(
                    "SELECT COUNT(*) FROM conversation_messages "
                    "WHERE user_id=? AND conversation_id=? AND role='user'",
                    (str(session.user_id), str(session.conversation_id)),
                ).fetchone()[0]
                >= 20
                or connection.execute(
                    "SELECT COUNT(*) FROM activity_sessions WHERE user_id=? AND conversation_id=?",
                    (str(session.user_id), str(session.conversation_id)),
                ).fetchone()[0]
                >= 20
            ):
                raise ActivityConflict("This chat has reached its activity limit")
            if session.source_entry_id != conversation.source_entry_id:
                raise ActivityConflict("Activity source changed")
            active = connection.execute(
                "SELECT * FROM activity_sessions WHERE user_id=? AND conversation_id=? "
                "AND status IN ('offered','active','paused','awaiting_report')",
                (str(session.user_id), str(session.conversation_id)),
            ).fetchone()
            if active is not None:
                current = self._activity_row(active)
                if current.status != ActivityStatus.OFFERED:
                    raise ActivityConflict("Finish or stop the current activity before starting another")
                # Negotiation supersedes an unstarted offer; an active activity is
                # never replaced silently by a newer recommendation.
                declined = current.model_copy(
                    update={
                        "status": ActivityStatus.DECLINED,
                        "revision": current.revision + 1,
                        "updated_at": now,
                    }
                )
                self._write_activity(connection, declined)
            fresh = session.model_copy(
                update={
                    "revision": 0,
                    "status": ActivityStatus.OFFERED,
                    "created_at": now,
                    "updated_at": now,
                }
            )
            connection.execute(
                "INSERT INTO activity_sessions (id,user_id,conversation_id,source_entry_id,status,revision,"
                "created_at,updated_at,payload_json) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    str(fresh.id),
                    str(fresh.user_id),
                    str(fresh.conversation_id),
                    str(fresh.source_entry_id) if fresh.source_entry_id else None,
                    fresh.status.value,
                    fresh.revision,
                    now.isoformat(),
                    now.isoformat(),
                    json.dumps(fresh.storage_payload()),
                ),
            )
            self._store_activity_receipt(
                connection, fresh.user_id, fresh.id, request_id, "offer", request_hash, now
            )
            return fresh

    def command_activity_session(
        self,
        user_id: UUID,
        session_id: UUID,
        request: ActivityCommandRequest,
        *,
        now: datetime,
    ) -> ActivitySession:
        request_hash = self._activity_input_hash(request.command, request.model_dump(mode="json"))
        with self.transaction() as connection:
            conversation, session = self._activity_locked(connection, user_id, session_id)
            if self._activity_receipt(
                connection, user_id, session_id, request.client_request_id, request.command, request_hash
            ):
                return session
            self._activity_guard(
                conversation,
                session,
                request.expected_conversation_revision,
                request.expected_revision,
                allow_listen=request.command in {"stop", "decline"},
                now=now,
            )
            if (
                request.command == "start"
                and connection.execute(
                    "SELECT COUNT(*) FROM conversation_messages "
                    "WHERE user_id=? AND conversation_id=? AND role='user'",
                    (str(user_id), str(conversation.id)),
                ).fetchone()[0]
                >= 20
            ):
                raise ActivityConflict("This chat has reached its activity limit")
            changed = apply_activity_command(session, request.command, now)
            self._write_activity(connection, changed)
            self._store_activity_receipt(
                connection, user_id, session_id, request.client_request_id, request.command, request_hash, now
            )
            return changed

    def report_activity_session(
        self,
        user_id: UUID,
        session_id: UUID,
        request: ActivityReportRequest,
        *,
        now: datetime,
        safety: SafetyResult | None = None,
    ) -> ActivitySession:
        request_hash = self._activity_input_hash("report", request.model_dump(mode="json"))
        with self.transaction() as connection:
            conversation, session = self._activity_locked(connection, user_id, session_id)
            if self._activity_receipt(
                connection, user_id, session_id, request.client_request_id, "report", request_hash
            ):
                return session
            self._activity_guard(
                conversation,
                session,
                request.expected_conversation_revision,
                request.expected_revision,
                allow_support=True,
                now=now,
            )
            if (
                session.report is not None
                or session.status
                not in {
                    ActivityStatus.AWAITING_REPORT,
                    ActivityStatus.STOPPED,
                }
                or session.started_at is None
            ):
                raise ActivityConflict("This activity is not awaiting a report")
            changed = session.model_copy(
                update={
                    "report": request.participant_report(),
                    "reported_at": now,
                    "status": ActivityStatus.STOPPED
                    if request.participation == "stopped"
                    else ActivityStatus.COMPLETED,
                    "revision": session.revision + 1,
                    "updated_at": now,
                    "follow_up_status": "pending",
                    "expires_at": None,
                }
            )
            self._write_activity(connection, changed)
            self._store_activity_receipt(
                connection, user_id, session_id, request.client_request_id, "report", request_hash, now
            )
            if (
                safety is not None
                and safety.mode == SafetyMode.SUPPORT
                and conversation.safety_mode != SafetyMode.SUPPORT
            ):
                supported = conversation.model_copy(
                    update={
                        "safety_mode": SafetyMode.SUPPORT,
                        "safety": safety,
                        "card": None,
                        "activity_card": None,
                        "ready_for_action": False,
                        "revision": conversation.revision + 1,
                        "updated_at": now,
                    }
                )
                connection.execute(
                    "UPDATE conversations SET payload_json=?,revision=?,updated_at=? WHERE id=?",
                    (supported.model_dump_json(), supported.revision, now.isoformat(), str(conversation.id)),
                )
                self._invalidate_activity_sessions(connection, supported, now=now, clear_text=False)
                refreshed = connection.execute(
                    "SELECT * FROM activity_sessions WHERE id=?", (str(session_id),)
                ).fetchone()
                return self._activity_row(refreshed)
            return changed

    def claim_activity_follow_up(
        self,
        user_id: UUID,
        session_id: UUID,
        request: ActivityFollowUpRequest,
        *,
        now: datetime,
    ) -> tuple[ActivitySession, bool]:
        request_hash = self._activity_input_hash("follow_up", request.model_dump(mode="json"))
        with self.transaction() as connection:
            conversation, session = self._activity_locked(connection, user_id, session_id)
            replay = self._activity_receipt(
                connection, user_id, session_id, request.client_request_id, "follow_up", request_hash
            )
            if session.follow_up_status == "ready":
                return session, False
            if (
                session.follow_up_status == "generating"
                and session.follow_up_lease_until is not None
                and session.follow_up_lease_until > now
            ):
                return session, False
            self._activity_guard(
                conversation,
                session,
                request.expected_conversation_revision,
                session.revision if replay else request.expected_revision,
                allow_support=True,
                now=now,
            )
            if session.report is None or session.follow_up_attempts >= MAX_FOLLOW_UP_ATTEMPTS:
                raise ActivityConflict("The saved check-in has no available follow-up attempt")
            count = connection.execute(
                "SELECT COUNT(*) FROM conversation_messages "
                "WHERE user_id=? AND conversation_id=? AND role='user'",
                (str(user_id), str(conversation.id)),
            ).fetchone()[0]
            final = count >= 20
            if final:
                reserved = connection.execute(
                    "SELECT payload_json FROM activity_sessions "
                    "WHERE user_id=? AND conversation_id=? AND id<>?",
                    (str(user_id), str(conversation.id), str(session.id)),
                ).fetchall()
                if any(
                    ActivitySession.model_validate_json(row["payload_json"]).final_follow_up
                    for row in reserved
                ):
                    raise ActivityConflict("This chat has reached its final check-in response")
            claimed = session.model_copy(
                update={
                    "follow_up_status": "generating",
                    "follow_up_request_id": request.client_request_id,
                    "follow_up_lease_until": now + timedelta(seconds=FOLLOW_UP_LEASE_SECONDS),
                    "follow_up_attempts": session.follow_up_attempts + 1,
                    "final_follow_up": session.final_follow_up or final,
                    "revision": session.revision + 1,
                    "updated_at": now,
                }
            )
            self._write_activity(connection, claimed)
            if not replay:
                self._store_activity_receipt(
                    connection, user_id, session_id, request.client_request_id, "follow_up", request_hash, now
                )
            return claimed, True

    def finish_activity_follow_up(
        self,
        user_id: UUID,
        session_id: UUID,
        *,
        request_id: UUID,
        expected_revision: int,
        expected_conversation_revision: int,
        expected_session_created_at: datetime,
        reply: str | None,
        model_run: ModelRun | None,
        now: datetime,
        directive: ActivityFollowUpDirective | None = None,
    ) -> ActivitySession:
        with self.transaction() as connection:
            conversation, session = self._activity_locked(connection, user_id, session_id)
            if session.created_at != expected_session_created_at:
                raise ActivityConflict("Activity changed")
            if session.follow_up_status == "ready" and session.follow_up_request_id == request_id:
                return session
            self._activity_guard(
                conversation,
                session,
                expected_conversation_revision,
                expected_revision,
                allow_support=True,
                now=now,
            )
            if session.follow_up_status != "generating" or session.follow_up_request_id != request_id:
                raise ActivityConflict("This follow-up changed")
            if reply is None:
                failed = session.model_copy(
                    update={
                        "follow_up_status": "failed",
                        "follow_up_lease_until": None,
                        "revision": session.revision + 1,
                        "updated_at": now,
                    }
                )
                self._write_activity(connection, failed)
                return failed
            if not reply.strip() or len(reply) > 1200:
                raise ValueError("Invalid activity follow-up")
            message_id = uuid5(request_id, "activity-follow-up-message")
            assistant = ConversationMessage(
                id=message_id,
                conversation_id=conversation.id,
                role=MessageRole.ASSISTANT,
                content=reply,
                created_at=now,
                safety_mode=conversation.safety_mode,
                model_run=model_run,
            )
            if directive is not None and (directive.card is not None or directive.search_topic is not None):
                count = connection.execute(
                    "SELECT COUNT(*) FROM conversation_messages "
                    "WHERE user_id=? AND conversation_id=? AND role='user'",
                    (str(user_id), str(conversation.id)),
                ).fetchone()[0]
                if (
                    session.final_follow_up
                    or count >= 20
                    or conversation.safety_mode == SafetyMode.SUPPORT
                    or conversation.interaction_preference == InteractionPreference.LISTEN
                ):
                    raise ActivityConflict("This check-in cannot offer another activity")
            self._insert_message(connection, user_id, assistant)
            directive_update = (
                {}
                if directive is None
                else {
                    "card": None,
                    "activity_card": directive.card.model_copy(update={"offered_message_id": message_id})
                    if directive.card is not None
                    else None,
                    "activity_constraints": directive.constraints or conversation.activity_constraints,
                    "activity_goal": directive.goal or conversation.activity_goal,
                    "activity_search_topic": directive.search_topic,
                    "activity_move": directive.move,
                    "ready_for_action": directive.card is not None,
                }
            )
            committed = conversation.model_copy(
                update={"updated_at": now, "revision": conversation.revision + 1, **directive_update}
            )
            connection.execute(
                "UPDATE conversations SET payload_json=?,revision=?,updated_at=? WHERE id=?",
                (committed.model_dump_json(), committed.revision, now.isoformat(), str(committed.id)),
            )
            if committed.activity_move == "pause" and conversation.activity_move != "pause":
                self._invalidate_activity_sessions(connection, committed, now=now, clear_text=False)
            else:
                self._supersede_activity_offers(connection, conversation, committed, now=now)
            finished = session.model_copy(
                update={
                    "follow_up_status": "ready",
                    "follow_up_reply": None,
                    "follow_up_model_run": model_run,
                    "follow_up_message_id": message_id,
                    "follow_up_lease_until": None,
                    "revision": session.revision + 1,
                    "updated_at": now,
                }
            )
            self._write_activity(connection, finished)
            return finished

    def _invalidate_activity_sessions(
        self,
        connection: sqlite3.Connection,
        conversation: Conversation,
        *,
        now: datetime,
        clear_text: bool,
    ) -> None:
        rows = connection.execute(
            "SELECT * FROM activity_sessions WHERE user_id=? AND conversation_id=?",
            (str(conversation.user_id), str(conversation.id)),
        ).fetchall()
        for row in rows:
            session = self._activity_row(row)
            self._write_activity(connection, invalidated_activity(session, now, clear_text=clear_text))
        if clear_text:
            connection.execute(
                "DELETE FROM activity_receipts WHERE session_id IN "
                "(SELECT id FROM activity_sessions WHERE conversation_id=?)",
                (str(conversation.id),),
            )

    def _activity_receipts(self, user_id: UUID) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM activity_receipts WHERE user_id=? ORDER BY created_at,id",
                (str(user_id),),
            ).fetchall()
        return [dict(row) for row in rows]

    def _supersede_activity_offers(
        self,
        connection: sqlite3.Connection,
        previous: Conversation,
        current: Conversation,
        *,
        now: datetime,
    ) -> None:
        recommendation_fields = (
            "card", "activity_card", "activity_constraints", "activity_goal", "activity_search_topic"
        )
        if all(getattr(previous, field) == getattr(current, field) for field in recommendation_fields):
            return
        rows = connection.execute(
            "SELECT * FROM activity_sessions WHERE user_id=? AND conversation_id=? AND status='offered'",
            (str(current.user_id), str(current.id)),
        ).fetchall()
        for row in rows:
            session = self._activity_row(row)
            if session.started_at is not None:
                continue
            # Negotiation withdraws the old unstarted offer; it is not a user's
            # rejection and must not exclude that resource from future choices.
            self._write_activity(connection, session.model_copy(update={
                "status": ActivityStatus.STOPPED,
                "revision": session.revision + 1,
                "updated_at": now,
                "expires_at": None,
                "check_in_issued": False,
            }))

    # Standalone writing -------------------------------------------------------------

    def save_journal_entry(self, entry: JournalEntry) -> JournalEntry:
        """Save immutable writing; replaying its UUID requires the same owner and exact text.

        The first server-assigned timestamp wins, so a network retry does not change
        the saved entry or require the client to predict a server clock.
        """
        entry = JournalEntry.model_validate(entry.model_dump())
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT payload_json FROM journal_entries WHERE id = ?", (str(entry.id),)
            ).fetchone()
            if row is not None:
                saved = JournalEntry.model_validate_json(row["payload_json"])
                if (saved.user_id, saved.text) != (entry.user_id, entry.text):
                    raise ValueError("Journal request ID is already in use")
                return saved
            connection.execute(
                "INSERT INTO journal_entries (id, user_id, created_at, payload_json) VALUES (?, ?, ?, ?)",
                (str(entry.id), str(entry.user_id), entry.created_at.isoformat(), entry.model_dump_json()),
            )
        return entry

    def get_journal_entry(self, user_id: UUID, entry_id: UUID) -> JournalEntry | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM journal_entries WHERE id = ? AND user_id = ?",
                (str(entry_id), str(user_id)),
            ).fetchone()
        return JournalEntry.model_validate_json(row["payload_json"]) if row else None

    def list_journal_entries(
        self,
        user_id: UUID,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[JournalEntry]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT payload_json FROM journal_entries WHERE user_id = ?
                ORDER BY julianday(created_at) DESC, id DESC LIMIT ? OFFSET ?
                """,
                (str(user_id), limit, offset),
            ).fetchall()
        return [JournalEntry.model_validate_json(row["payload_json"]) for row in rows]

    def delete_journal_entry(self, user_id: UUID, entry_id: UUID) -> bool:
        """Delete writing and every linked chat in one serialized transaction.

        A model call holds no database transaction. Removing its conversation here
        makes a delayed commit fail instead of recreating derived journal content.
        """
        with self.transaction() as connection:
            owned = connection.execute(
                "SELECT id FROM journal_entries WHERE id = ? AND user_id = ?",
                (str(entry_id), str(user_id)),
            ).fetchone()
            if owned is None:
                return False
            rows = connection.execute(
                "SELECT id FROM conversations WHERE user_id = ? AND source_entry_id = ?",
                (str(user_id), str(entry_id)),
            ).fetchall()
            for row in rows:
                self._delete_conversation(connection, user_id, UUID(row["id"]))
            connection.execute(
                "DELETE FROM journal_entries WHERE id = ? AND user_id = ?",
                (str(entry_id), str(user_id)),
            )
        return True

    # Reflections and outcomes -------------------------------------------------------

    def _insert_reflection(
        self, connection: sqlite3.Connection, record: ReflectionRecord, conversation_id: UUID | None
    ) -> ReflectionRecord:
        existing = connection.execute(
            "SELECT user_id, payload_json, conversation_id FROM reflections WHERE id = ?", (str(record.id),)
        ).fetchone()
        if existing is not None:
            if existing["user_id"] != str(record.user_id):
                raise ValueError("Reflection request ID belongs to another user")
            saved = ReflectionRecord.model_validate_json(existing["payload_json"])
            if existing["conversation_id"] != (str(conversation_id) if conversation_id else None) or (
                conversation_id is not None
                and (saved.decision.action_id, saved.state, saved.target, saved.self_report_input)
                != (record.decision.action_id, record.state, record.target, record.self_report_input)
            ):
                raise ValueError("Reflection request ID is already in use")
            return saved
        try:
            connection.execute(
                """
                INSERT INTO reflections (id, user_id, created_at, text, payload_json, conversation_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(record.id),
                    str(record.user_id),
                    record.created_at.isoformat(),
                    record.text,
                    record.model_dump_json(),
                    str(conversation_id) if conversation_id else None,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError("Reflection request ID is already in use") from exc
        return record

    def save_reflection(self, record: ReflectionRecord) -> ReflectionRecord:
        with self.transaction() as connection:
            return self._insert_reflection(connection, record, None)

    def save_outcome(self, record: OutcomeRecord) -> OutcomeRecord:
        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT user_id, decision_id, payload_json FROM outcomes WHERE id = ?",
                (str(record.id),),
            ).fetchone()
            if existing is not None:
                if existing["user_id"] != str(record.user_id) or existing["decision_id"] != str(
                    record.decision_id
                ):
                    raise ValueError("Outcome request ID is already in use")
                return OutcomeRecord.model_validate_json(existing["payload_json"])
            rows = connection.execute(
                "SELECT payload_json FROM reflections WHERE user_id = ?", (str(record.user_id),)
            ).fetchall()
            if not any(
                ReflectionRecord.model_validate_json(item["payload_json"]).decision.decision_id
                == record.decision_id
                for item in rows
            ):
                raise ValueError("Policy decision does not belong to this user")
            try:
                connection.execute(
                    """
                    INSERT INTO outcomes (id, user_id, decision_id, created_at, payload_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        str(record.id),
                        str(record.user_id),
                        str(record.decision_id),
                        record.created_at.isoformat(),
                        record.model_dump_json(),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise DuplicateOutcomeError("An outcome already exists for this decision") from exc
        return record

    def list_reflections(self, user_id: UUID, *, limit: int = 50, offset: int = 0) -> list[ReflectionRecord]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT payload_json FROM reflections
                WHERE user_id = ? ORDER BY datetime(created_at) DESC, id DESC LIMIT ? OFFSET ?
                """,
                (str(user_id), limit, offset),
            ).fetchall()
        return [ReflectionRecord.model_validate_json(row["payload_json"]) for row in rows]

    def list_all_reflections(self, user_id: UUID) -> list[ReflectionRecord]:
        return self.list_reflections(user_id, limit=-1, offset=0)

    def get_reflection(self, user_id: UUID, reflection_id: UUID) -> ReflectionRecord | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM reflections WHERE id = ? AND user_id = ?",
                (str(reflection_id), str(user_id)),
            ).fetchone()
        return ReflectionRecord.model_validate_json(row["payload_json"]) if row else None

    def list_outcomes(self, user_id: UUID) -> list[OutcomeRecord]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM outcomes WHERE user_id = ? ORDER BY datetime(created_at) DESC",
                (str(user_id),),
            ).fetchall()
        return [OutcomeRecord.model_validate_json(row["payload_json"]) for row in rows]

    def _delete_reflection(self, connection: sqlite3.Connection, user_id: UUID, reflection_id: UUID) -> bool:
        row = connection.execute(
            "SELECT payload_json FROM reflections WHERE id = ? AND user_id = ?",
            (str(reflection_id), str(user_id)),
        ).fetchone()
        if row is None:
            return False
        record = ReflectionRecord.model_validate_json(row["payload_json"])
        connection.execute(
            "DELETE FROM outcomes WHERE user_id = ? AND decision_id = ?",
            (str(user_id), str(record.decision.decision_id)),
        )
        connection.execute(
            "DELETE FROM reflections WHERE id = ? AND user_id = ?", (str(reflection_id), str(user_id))
        )
        return True

    def delete_reflection(self, user_id: UUID, reflection_id: UUID) -> bool:
        with self.transaction() as connection:
            return self._delete_reflection(connection, user_id, reflection_id)

    def export_user_data(self, user_id: UUID) -> dict:
        return {
            "activity_sessions": [item.storage_payload() for item in self.list_activity_sessions(user_id)],
            "activity_receipts": self._activity_receipts(user_id),
            "journal_entries": [
                item.model_dump(mode="json") for item in self.list_journal_entries(user_id, limit=-1)
            ],
            "reflections": [item.model_dump(mode="json") for item in self.list_all_reflections(user_id)],
            "outcomes": [item.model_dump(mode="json") for item in self.list_outcomes(user_id)],
            "conversations": [item.model_dump(mode="json") for item in self._list_conversations(user_id)],
            "conversation_messages": [
                item.model_dump(mode="json") for item in self._list_all_messages(user_id)
            ],
            "conversation_preference_requests": self._preference_requests(user_id),
        }

    def delete_user_data(self, user_id: UUID) -> int:
        """Delete every journal row; the returned count covers journal data only."""
        deleted = 0
        with self.transaction() as connection:
            for table in (
                "activity_receipts",
                "activity_sessions",
                "outcomes",
                "conversation_messages",
                "reflections",
                "conversation_preference_requests",
                "conversations",
                "journal_entries",
            ):
                cursor = connection.execute(f"DELETE FROM {table} WHERE user_id = ?", (str(user_id),))
                deleted += cursor.rowcount
        # Usage counters hold no journal content and are expired by the retention job;
        # keeping them here stops a journal deletion from resetting the generation limit.
        return deleted

    # Conversations --------------------------------------------------------------------

    def _preference_requests(self, user_id: UUID) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM conversation_preference_requests WHERE user_id = ? ORDER BY created_at, id",
                (str(user_id),),
            ).fetchall()
        return [dict(row) for row in rows]

    def change_preference(
        self,
        user_id: UUID,
        conversation_id: UUID,
        *,
        request_id: UUID,
        preference: InteractionPreference,
        expected_revision: int,
        now: datetime,
    ) -> Conversation:
        """Apply a user command atomically; replay returns current state, not old state."""
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM conversations WHERE id = ? AND user_id = ?",
                (str(conversation_id), str(user_id)),
            ).fetchone()
            if row is None:
                raise ConversationNotFound(str(conversation_id))
            current = self._conversation(row)
            receipt = connection.execute(
                "SELECT * FROM conversation_preference_requests WHERE conversation_id = ? AND id = ?",
                (str(conversation_id), str(request_id)),
            ).fetchone()
            if receipt:
                if (receipt["preference"], receipt["expected_revision"]) != (preference, expected_revision):
                    raise ValueError("Preference request ID is already in use")
                return current
            if current.status != ConversationStatus.OPEN:
                raise ConversationClosed(str(conversation_id))
            if current.revision != expected_revision:
                raise ConversationStale(str(conversation_id))
            if current.safety_mode.value == "support":
                raise ValueError("Support mode cannot be changed")
            changed = current.model_copy(
                update={
                    "interaction_preference": preference,
                    "card": None,
                    "activity_card": None,
                    "ready_for_action": preference == InteractionPreference.ACT,
                    "revision": current.revision + 1,
                    "updated_at": now,
                }
            )
            connection.execute(
                "UPDATE conversations SET payload_json = ?, revision = ?, updated_at = ? WHERE id = ?",
                (changed.model_dump_json(), changed.revision, now.isoformat(), str(conversation_id)),
            )
            if preference == InteractionPreference.LISTEN:
                self._invalidate_activity_sessions(connection, changed, now=now, clear_text=False)
            else:
                self._supersede_activity_offers(connection, current, changed, now=now)
            connection.execute(
                "INSERT INTO conversation_preference_requests VALUES (?, ?, ?, ?, ?, ?)",
                (
                    str(request_id),
                    str(conversation_id),
                    str(user_id),
                    preference.value,
                    expected_revision,
                    now.isoformat(),
                ),
            )
        return changed

    @staticmethod
    def _conversation(row: sqlite3.Row) -> Conversation:
        data = json.loads(row["payload_json"])
        data["revision"] = row["revision"]
        data["status"] = row["status"]
        return Conversation.model_validate(data)

    def create_conversation(self, conversation: Conversation) -> Conversation:
        with self.transaction() as connection:
            source_entry_id = conversation.source_entry_id
            if source_entry_id is not None:
                source_row = connection.execute(
                    "SELECT payload_json FROM journal_entries WHERE id = ? AND user_id = ?",
                    (str(source_entry_id), str(conversation.user_id)),
                ).fetchone()
                if (
                    source_row is None
                    or JournalEntry.model_validate_json(source_row["payload_json"]).created_at
                    != conversation.source_entry_created_at
                ):
                    # UUIDs can be reused after deletion. Timestamp identity prevents
                    # a start computed from a deleted incarnation attaching to its replacement.
                    raise JournalEntryNotFound(str(source_entry_id))
            existing = connection.execute(
                "SELECT * FROM conversations WHERE id = ?", (str(conversation.id),)
            ).fetchone()
            if existing is not None:
                if existing["user_id"] != str(conversation.user_id):
                    raise ValueError("Conversation request ID belongs to another user")
                saved = self._conversation(existing)
                if (
                    saved.source_entry_id,
                    saved.source_entry_created_at,
                    saved.llm_consent,
                    saved.retain_text,
                    saved.locale,
                ) != (
                    source_entry_id,
                    conversation.source_entry_created_at,
                    conversation.llm_consent,
                    conversation.retain_text,
                    conversation.locale,
                ):
                    raise ValueError("Conversation request ID is already in use")
                return saved
            fresh = conversation.model_copy(
                update={"status": ConversationStatus.OPEN, "revision": 0, "reflection_id": None}
            )
            connection.execute(
                """
                INSERT INTO conversations (
                    id, user_id, created_at, updated_at, status, payload_json, revision,
                    source_entry_id, source_entry_created_at
                ) VALUES (?, ?, ?, ?, 'open', ?, 0, ?, ?)
                """,
                (
                    str(fresh.id),
                    str(fresh.user_id),
                    fresh.created_at.isoformat(),
                    fresh.updated_at.isoformat(),
                    fresh.model_dump_json(),
                    str(source_entry_id) if source_entry_id else None,
                    conversation.source_entry_created_at.isoformat()
                    if conversation.source_entry_created_at
                    else None,
                ),
            )
        return fresh

    def get_conversation(self, user_id: UUID, conversation_id: UUID) -> Conversation | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM conversations WHERE id = ? AND user_id = ?",
                (str(conversation_id), str(user_id)),
            ).fetchone()
        return self._conversation(row) if row else None

    def list_messages(self, user_id: UUID, conversation_id: UUID) -> list[ConversationMessage]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT payload_json FROM conversation_messages
                WHERE user_id = ? AND conversation_id = ?
                ORDER BY created_at ASC, role DESC
                """,
                (str(user_id), str(conversation_id)),
            ).fetchall()
        return [ConversationMessage.model_validate_json(row["payload_json"]) for row in rows]

    def commit_turn(
        self,
        conversation: Conversation,
        user_message: ConversationMessage,
        assistant_message: ConversationMessage,
        *,
        expected_revision: int,
    ) -> Turn:
        if user_message.client_message_id is None:
            raise ValueError("A turn needs a client message ID")
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM conversations WHERE id = ? AND user_id = ?",
                (str(conversation.id), str(conversation.user_id)),
            ).fetchone()
            if row is None:
                raise ConversationNotFound(str(conversation.id))
            stored = self._stored_turn(connection, conversation.id, user_message.client_message_id)
            if stored is not None:
                require_same_turn_request(
                    stored[1],
                    text=user_message.content,
                    inputs=user_message.request_inputs,
                )
                return stored
            if row["status"] != ConversationStatus.OPEN.value:
                raise ConversationClosed(str(conversation.id))
            if row["revision"] != expected_revision:
                raise ConversationStale(str(conversation.id))
            current = self._conversation(row)
            committed = conversation.model_copy(
                update={
                    "status": ConversationStatus.OPEN,
                    "revision": expected_revision + 1,
                    "reflection_id": None,
                    "interaction_preference": current.interaction_preference,
                    "source_entry_id": current.source_entry_id,
                    "source_entry_created_at": current.source_entry_created_at,
                }
            )
            # Match PostgreSQL: an older writer cannot restore ordinary offers
            # while the canonical choice is Listen. Support still takes precedence.
            if (
                current.interaction_preference == InteractionPreference.LISTEN
                and committed.safety_mode != SafetyMode.SUPPORT
            ):
                committed = committed.model_copy(
                    update={"card": None, "activity_card": None, "ready_for_action": False}
                )
            if committed.safety_mode == SafetyMode.SUPPORT or committed.activity_move == "pause":
                committed = committed.model_copy(update={"activity_card": None})
            self._insert_message(connection, conversation.user_id, user_message)
            self._insert_message(connection, conversation.user_id, assistant_message)
            connection.execute(
                """
                UPDATE conversations SET updated_at = ?, payload_json = ?, revision = ?
                WHERE id = ?
                """,
                (
                    committed.updated_at.isoformat(),
                    committed.model_dump_json(),
                    committed.revision,
                    str(committed.id),
                ),
            )
            if (
                committed.safety_mode == SafetyMode.SUPPORT and current.safety_mode != SafetyMode.SUPPORT
            ) or (committed.activity_move == "pause" and current.activity_move != "pause"):
                self._invalidate_activity_sessions(
                    connection, committed, now=committed.updated_at, clear_text=False
                )
            else:
                self._supersede_activity_offers(connection, current, committed, now=committed.updated_at)
        return committed, user_message, assistant_message

    def accept_conversation(
        self,
        user_id: UUID,
        conversation_id: UUID,
        record: ReflectionRecord,
        *,
        expected_revision: int,
    ) -> ReflectionRecord:
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM conversations WHERE id = ? AND user_id = ?",
                (str(conversation_id), str(user_id)),
            ).fetchone()
            if row is None:
                raise ConversationNotFound(str(conversation_id))
            current = self._conversation(row)
            if current.reflection_id is not None:
                if current.reflection_id == record.id:
                    saved = connection.execute(
                        "SELECT payload_json, conversation_id FROM reflections WHERE id = ?",
                        (str(record.id),),
                    ).fetchone()
                    saved_record = ReflectionRecord.model_validate_json(saved["payload_json"])
                    if saved["conversation_id"] != str(conversation_id) or (
                        saved_record.decision.action_id,
                        saved_record.state,
                        saved_record.target,
                    ) != (record.decision.action_id, record.state, record.target):
                        raise ValueError("Reflection request ID is already in use")
                    return saved_record
                raise ConversationAlreadyAccepted(str(conversation_id))
            if current.status != ConversationStatus.OPEN:
                raise ConversationClosed(str(conversation_id))
            if current.revision != expected_revision:
                raise ConversationStale(str(conversation_id))
            if (
                current.interaction_preference == InteractionPreference.LISTEN
                and current.safety_mode != SafetyMode.SUPPORT
            ):
                raise ConversationStale(str(conversation_id))
            saved_record = self._insert_reflection(connection, record, conversation_id)
            self._close_locked(connection, current, reflection_id=record.id, now=datetime.now(UTC))
        return saved_record

    def close_conversation(self, user_id: UUID, conversation_id: UUID) -> Conversation | None:
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM conversations WHERE id = ? AND user_id = ?",
                (str(conversation_id), str(user_id)),
            ).fetchone()
            if row is None:
                return None
            current = self._conversation(row)
            if current.status == ConversationStatus.CLOSED:
                return current
            return self._close_locked(connection, current, reflection_id=None, now=datetime.now(UTC))

    def delete_conversation(self, user_id: UUID, conversation_id: UUID) -> bool:
        with self.transaction() as connection:
            return self._delete_conversation(connection, user_id, conversation_id)

    def _delete_conversation(
        self,
        connection: sqlite3.Connection,
        user_id: UUID,
        conversation_id: UUID,
    ) -> bool:
        row = connection.execute(
            "SELECT * FROM conversations WHERE id = ? AND user_id = ?",
            (str(conversation_id), str(user_id)),
        ).fetchone()
        if row is None:
            return False
        current = self._conversation(row)
        if current.reflection_id is not None:
            self._delete_reflection(connection, user_id, current.reflection_id)
        connection.execute(
            "DELETE FROM activity_receipts WHERE user_id=? AND session_id IN "
            "(SELECT id FROM activity_sessions WHERE conversation_id=?)",
            (str(user_id), str(conversation_id)),
        )
        connection.execute(
            "DELETE FROM activity_sessions WHERE user_id=? AND conversation_id=?",
            (str(user_id), str(conversation_id)),
        )
        connection.execute(
            "DELETE FROM conversation_messages WHERE user_id = ? AND conversation_id = ?",
            (str(user_id), str(conversation_id)),
        )
        connection.execute(
            "DELETE FROM conversation_preference_requests WHERE user_id = ? AND conversation_id = ?",
            (str(user_id), str(conversation_id)),
        )
        connection.execute(
            "DELETE FROM conversations WHERE user_id = ? AND id = ?",
            (str(user_id), str(conversation_id)),
        )
        return True

    def close_stale_conversations(self, user_id: UUID, *, now: datetime) -> int:
        return self._purge(now=now, only_user=user_id)["closed"]

    def purge_expired_conversations(self, *, now: datetime) -> dict[str, int]:
        """The scheduled retention job, for local runs and tests."""
        return self._purge(now=now, only_user=None)

    def _purge(self, *, now: datetime, only_user: UUID | None) -> dict[str, int]:
        cutoff = (now - STALE_AFTER).isoformat()
        closed = 0
        purged = 0
        with self.transaction() as connection:
            query = "SELECT * FROM conversations WHERE status = 'open' AND updated_at < ?"
            params: list[Any] = [cutoff]
            if only_user is not None:
                query += " AND user_id = ?"
                params.append(str(only_user))
            for row in connection.execute(query, params).fetchall():
                idle = self._conversation(row)
                if not idle.retain_text:
                    purged += self._purge_messages(connection, idle.id)
                self._close_locked(connection, idle, reflection_id=None, now=now)
                closed += 1
            query = "SELECT * FROM conversations WHERE status = 'closed'"
            params = []
            if only_user is not None:
                query += " AND user_id = ?"
                params.append(str(only_user))
            for row in connection.execute(query, params).fetchall():
                conversation = self._conversation(row)
                if not conversation.retain_text:
                    purged += self._purge_messages(connection, conversation.id)
            expired = (now - timedelta(days=1)).isoformat()
            if only_user is None:
                connection.execute("DELETE FROM rate_limit_events WHERE created_at < ?", (expired,))
            else:
                connection.execute(
                    "DELETE FROM rate_limit_events WHERE created_at < ? AND user_id = ?",
                    (expired, str(only_user)),
                )
        return {"closed": closed, "purged_messages": purged}

    def _close_locked(
        self,
        connection: sqlite3.Connection,
        current: Conversation,
        *,
        reflection_id: UUID | None,
        now: datetime,
    ) -> Conversation:
        closed = current.model_copy(
            update={
                "status": ConversationStatus.CLOSED,
                "activity_card": None,
                "updated_at": now,
                "revision": current.revision + 1,
                "reflection_id": reflection_id or current.reflection_id,
            }
        )
        connection.execute(
            """
            UPDATE conversations SET updated_at = ?, status = 'closed', payload_json = ?, revision = ?
            WHERE id = ?
            """,
            (closed.updated_at.isoformat(), closed.model_dump_json(), closed.revision, str(closed.id)),
        )
        self._invalidate_activity_sessions(connection, closed, now=now, clear_text=not closed.retain_text)
        if not closed.retain_text:
            self._purge_messages(connection, closed.id)
        return closed

    # Rate limiting --------------------------------------------------------------------

    def consume_rate_limit(
        self, user_id: UUID, bucket: str, *, limit: int, window_seconds: int, now: datetime
    ) -> tuple[bool, int]:
        cutoff = now - timedelta(seconds=window_seconds)
        with self.transaction() as connection:
            connection.execute(
                "DELETE FROM rate_limit_events WHERE user_id = ? AND bucket = ? AND created_at <= ?",
                (str(user_id), bucket, cutoff.isoformat()),
            )
            rows = connection.execute(
                """
                SELECT created_at FROM rate_limit_events WHERE user_id = ? AND bucket = ?
                ORDER BY created_at ASC
                """,
                (str(user_id), bucket),
            ).fetchall()
            if len(rows) >= limit:
                oldest = datetime.fromisoformat(rows[0]["created_at"])
                retry_after = (oldest + timedelta(seconds=window_seconds) - now).total_seconds()
                return False, max(1, ceil(retry_after))
            connection.execute(
                "INSERT INTO rate_limit_events (user_id, bucket, created_at) VALUES (?, ?, ?)",
                (str(user_id), bucket, now.isoformat()),
            )
        return True, 0

    # Helpers ----------------------------------------------------------------------------

    def _list_conversations(self, user_id: UUID) -> list[Conversation]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM conversations WHERE user_id = ? ORDER BY created_at ASC", (str(user_id),)
            ).fetchall()
        return [self._conversation(row) for row in rows]

    def _list_all_messages(self, user_id: UUID) -> list[ConversationMessage]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM conversation_messages WHERE user_id = ? ORDER BY created_at ASC",
                (str(user_id),),
            ).fetchall()
        return [ConversationMessage.model_validate_json(row["payload_json"]) for row in rows]

    def _insert_message(
        self, connection: sqlite3.Connection, user_id: UUID, message: ConversationMessage
    ) -> None:
        connection.execute(
            """
            INSERT INTO conversation_messages (
                id, conversation_id, user_id, client_message_id, role, created_at, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(message.id),
                str(message.conversation_id),
                str(user_id),
                str(message.client_message_id) if message.client_message_id else None,
                message.role.value,
                message.created_at.isoformat(),
                message.model_dump_json(),
            ),
        )

    def _stored_turn(
        self, connection: sqlite3.Connection, conversation_id: UUID, client_message_id: UUID
    ) -> Turn | None:
        row = connection.execute(
            """
            SELECT payload_json FROM conversation_messages
            WHERE conversation_id = ? AND client_message_id = ?
            """,
            (str(conversation_id), str(client_message_id)),
        ).fetchone()
        if row is None:
            return None
        stored_user = ConversationMessage.model_validate_json(row["payload_json"])
        assistant_row = connection.execute(
            """
            SELECT payload_json FROM conversation_messages
            WHERE conversation_id = ? AND role = 'assistant' AND created_at >= ?
            ORDER BY created_at ASC LIMIT 1
            """,
            (str(conversation_id), stored_user.created_at.isoformat()),
        ).fetchone()
        conversation_row = connection.execute(
            "SELECT * FROM conversations WHERE id = ?", (str(conversation_id),)
        ).fetchone()
        if assistant_row is None or conversation_row is None:
            return None
        return (
            self._conversation(conversation_row),
            stored_user,
            ConversationMessage.model_validate_json(assistant_row["payload_json"]),
        )

    def _purge_messages(self, connection: sqlite3.Connection, conversation_id: UUID) -> int:
        rows = connection.execute(
            "SELECT id, payload_json FROM conversation_messages WHERE conversation_id = ?",
            (str(conversation_id),),
        ).fetchall()
        cleared_count = 0
        for row in rows:
            message = ConversationMessage.model_validate_json(row["payload_json"])
            if message.content is None:
                continue
            cleared = message.model_copy(update={"content": None})
            connection.execute(
                "UPDATE conversation_messages SET payload_json = ? WHERE id = ?",
                (cleared.model_dump_json(), row["id"]),
            )
            cleared_count += 1
        return cleared_count


def _error_message(response: httpx.Response) -> str:
    try:
        body = response.json()
    except (ValueError, RecursionError):
        return response.text
    return str(body.get("message", "")) if isinstance(body, dict) else str(body)


class SupabaseRepository:
    """PostgREST adapter. Reads use the person's JWT under RLS; writes go through the
    database functions, and provenance writes are signed with the server key."""

    def __init__(self, settings: Settings, access_token: str, client: httpx.Client | None = None) -> None:
        if not settings.supabase_enabled:
            raise ValueError("Supabase is not configured")
        assert settings.supabase_url is not None
        self.base_url = f"{settings.supabase_url.rstrip('/')}/rest/v1"
        self.signing_key = settings.write_signing_key
        self.page_size = EXPORT_PAGE_SIZE
        self.headers = {
            "apikey": settings.supabase_anon_key or "",
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }
        self.client = client

    def _request(self, method: str, table: str, **kwargs: Any) -> Any:
        try:
            with managed_http_client(self.client, timeout=15) as client:
                response = client.request(
                    method,
                    f"{self.base_url}/{table}",
                    headers=self.headers,
                    timeout=15,
                    follow_redirects=False,
                    **kwargs,
                )
        except httpx.HTTPError as exc:
            raise StorageUnavailable("The database could not be reached") from exc
        if response.status_code >= 400:
            if table == "rpc/jp_consume_rate_limit_v2" and response.status_code == 404:
                # Missing migration / schema-cache refresh must not fall back to an
                # unsigned or local-only paid-call limit.
                raise StorageUnavailable("The signed usage counter is unavailable")
            self._raise_for(response)
        if not 200 <= response.status_code < 300 or method == "GET" and response.status_code != 200:
            raise StorageUnavailable("The database returned an unexpected response status")
        try:
            decoded = response.json() if response.content else []
        except (ValueError, RecursionError) as exc:
            raise StorageUnavailable("The database returned an invalid response") from exc
        # PostgREST reads return arrays. An empty object is not an empty history,
        # and malformed rows must not become an unhandled application exception.
        return self._rows(decoded) if method == "GET" else decoded

    @staticmethod
    def _object(value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise StorageUnavailable("The database returned an invalid response")
        return value

    @staticmethod
    def _rows(value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
            raise StorageUnavailable("The database returned an invalid response")
        return value

    @staticmethod
    def _boolean(value: Any) -> bool:
        # Coercion would make the string "false" grant a paid generation request.
        if not isinstance(value, bool):
            raise StorageUnavailable("The database returned an invalid response")
        return value

    @staticmethod
    def _nonnegative_integer(value: Any) -> int:
        # Python treats booleans as integers; database count receipts must not.
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise StorageUnavailable("The database returned an invalid response")
        return value

    @classmethod
    def _model[T: BaseModel](cls, model: type[T], value: Any) -> T:
        try:
            return model.model_validate(cls._object(value))
        except (ValueError, TypeError, RecursionError) as exc:
            # Storage schema failures are upstream failures, not bad user input.
            # Do not expose persisted private values in the public error message.
            raise StorageUnavailable("The database returned an invalid response") from exc

    @classmethod
    def _row_model[T: BaseModel](cls, model: type[T], row: dict[str, Any]) -> T:
        return cls._model(model, row.get("record"))

    @staticmethod
    def _raise_for(response: httpx.Response) -> None:
        message = _error_message(response)
        status = response.status_code
        if status == 404 and message == "Conversation not found":
            raise ConversationNotFound(message)
        if status == 404 and message == "Activity not found":
            raise ActivityNotFound(message)
        if status == 404 and message == "Journal entry not found":
            raise JournalEntryNotFound(message)
        if status == 409:
            if message.startswith(
                (
                    "Activity",
                    "This activity",
                    "The timer",
                    "Finish or stop",
                    "This chat has reached",
                    "This chat has paused",
                    "This chat is set",
                    "Support mode pauses",
                    "The saved check-in",
                    "This follow-up",
                )
            ):
                raise ActivityConflict(message)
            if message == "Conversation is closed":
                raise ConversationClosed(message)
            if message == "Conversation changed":
                raise ConversationStale(message)
            if message == "Conversation already accepted":
                raise ConversationAlreadyAccepted(message)
            if message == "An outcome already exists for this decision":
                raise DuplicateOutcomeError(message)
            raise ValueError(message)
        if status == 404 and message.startswith("Policy decision"):
            raise ValueError(message)
        if status >= 500:
            raise StorageUnavailable(message or f"Database error {status}")
        response.raise_for_status()

    def _signed(self, purpose: str, user_id: UUID, body: dict[str, Any]) -> dict[str, str]:
        if not self.signing_key:
            raise StorageUnavailable("The server signing key is not configured")
        return signed_payload(purpose, user_id, body, self.signing_key)

    # Versioned signed activity RPCs; direct client writes remain disabled by RLS.

    def get_activity_session(self, user_id: UUID, session_id: UUID) -> ActivitySession | None:
        rows = self._request(
            "GET",
            "activity_sessions",
            params={
                "select": "record",
                "user_id": f"eq.{user_id}",
                "id": f"eq.{session_id}",
                "limit": 1,
            },
        )
        return self._row_model(ActivitySession, rows[0]) if rows else None

    def list_activity_sessions(
        self, user_id: UUID, conversation_id: UUID | None = None
    ) -> list[ActivitySession]:
        params = {"select": "record", "user_id": f"eq.{user_id}", "order": "created_at.desc,id.desc"}
        if conversation_id is not None:
            params["conversation_id"] = f"eq.{conversation_id}"
        return [
            self._row_model(ActivitySession, row) for row in self._select_all("activity_sessions", params)
        ]

    def offer_activity_session(
        self,
        session: ActivitySession,
        *,
        request_id: UUID,
        expected_conversation_revision: int,
        now: datetime,
    ) -> ActivitySession:
        del now  # PostgreSQL owns start/expiry timestamps even if the API clock differs.
        fingerprint = {
            "expected_conversation_revision": expected_conversation_revision,
            "resource_id": session.resource.id,
            "duration_seconds": session.duration_seconds,
        }
        saved = self._request(
            "POST",
            "rpc/jp_offer_activity_v1",
            json=self._signed(
                "offer_activity",
                session.user_id,
                {
                    "session": session.storage_payload(),
                    "request_id": str(request_id),
                    "expected_conversation_revision": expected_conversation_revision,
                    "request_hash": SQLiteRepository._activity_input_hash("offer", fingerprint),
                },
            ),
        )
        return self._model(ActivitySession, saved)

    def command_activity_session(
        self,
        user_id: UUID,
        session_id: UUID,
        request: ActivityCommandRequest,
        *,
        now: datetime,
    ) -> ActivitySession:
        del now
        saved = self._request(
            "POST",
            "rpc/jp_activity_command_v1",
            json=self._signed(
                "activity_command",
                user_id,
                {
                    "session_id": str(session_id),
                    "request": request.model_dump(mode="json"),
                    "request_hash": SQLiteRepository._activity_input_hash(
                        request.command, request.model_dump(mode="json")
                    ),
                },
            ),
        )
        return self._model(ActivitySession, saved)

    def report_activity_session(
        self,
        user_id: UUID,
        session_id: UUID,
        request: ActivityReportRequest,
        *,
        now: datetime,
        safety: SafetyResult | None = None,
    ) -> ActivitySession:
        del now
        saved = self._request(
            "POST",
            "rpc/jp_report_activity_v1",
            json=self._signed(
                "report_activity",
                user_id,
                {
                    "session_id": str(session_id),
                    "request": request.model_dump(mode="json"),
                    "request_hash": SQLiteRepository._activity_input_hash(
                        "report", request.model_dump(mode="json")
                    ),
                    "safety": safety.model_dump(mode="json")
                    if safety is not None and safety.mode == SafetyMode.SUPPORT
                    else None,
                },
            ),
        )
        return self._model(ActivitySession, saved)

    def claim_activity_follow_up(
        self,
        user_id: UUID,
        session_id: UUID,
        request: ActivityFollowUpRequest,
        *,
        now: datetime,
    ) -> tuple[ActivitySession, bool]:
        del now
        saved = self._object(
            self._request(
                "POST",
                "rpc/jp_claim_activity_followup_v1",
                json=self._signed(
                    "claim_activity_follow_up",
                    user_id,
                    {
                        "session_id": str(session_id),
                        "request": request.model_dump(mode="json"),
                        "request_hash": SQLiteRepository._activity_input_hash(
                            "follow_up", request.model_dump(mode="json")
                        ),
                    },
                ),
            )
        )
        return self._model(ActivitySession, saved.get("session")), self._boolean(saved.get("claimed"))

    def finish_activity_follow_up(
        self,
        user_id: UUID,
        session_id: UUID,
        *,
        request_id: UUID,
        expected_revision: int,
        expected_conversation_revision: int,
        expected_session_created_at: datetime,
        reply: str | None,
        model_run: ModelRun | None,
        now: datetime,
        directive: ActivityFollowUpDirective | None = None,
    ) -> ActivitySession:
        # The API creates a deterministic reply identity; the RPC commits it with
        # the session and chat revisions, or keeps only the independently saved report.
        session = self.get_activity_session(user_id, session_id)
        if session is None:
            raise ActivityNotFound("Activity not found")
        conversation = self.get_conversation(user_id, session.conversation_id)
        if conversation is None:
            raise ConversationNotFound(str(session.conversation_id))
        assistant = (
            None
            if reply is None
            else ConversationMessage(
                id=uuid5(request_id, "activity-follow-up-message"),
                conversation_id=session.conversation_id,
                role=MessageRole.ASSISTANT,
                content=reply,
                created_at=now,
                safety_mode=conversation.safety_mode,
                model_run=model_run,
            ).model_dump(mode="json")
        )
        saved = self._request(
            "POST",
            "rpc/jp_finish_activity_followup_v1",
            json=self._signed(
                "finish_activity_follow_up",
                user_id,
                {
                    "session_id": str(session_id),
                    "request_id": str(request_id),
                    "expected_revision": expected_revision,
                    "expected_conversation_revision": expected_conversation_revision,
                    "expected_session_created_at": expected_session_created_at.isoformat(),
                    "assistant_message": assistant,
                    "directive": directive.model_dump(mode="json") if directive is not None else None,
                },
            ),
        )
        return self._model(ActivitySession, saved)

    def change_preference(
        self,
        user_id: UUID,
        conversation_id: UUID,
        *,
        request_id: UUID,
        preference: InteractionPreference,
        expected_revision: int,
        now: datetime,
    ) -> Conversation:
        saved = self._request(
            "POST",
            "rpc/jp_change_preference",
            json=self._signed(
                "change_preference",
                user_id,
                {
                    "conversation_id": str(conversation_id),
                    "client_request_id": str(request_id),
                    "preference": preference.value,
                    "expected_revision": expected_revision,
                    "updated_at": now.isoformat(),
                },
            ),
        )
        return self._model(Conversation, saved)

    def _select_all(self, table: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Read every matching row in fixed-size pages; PostgREST caps each response."""
        rows: list[dict[str, Any]] = []
        offset = 0
        while True:
            page = self._request("GET", table, params={**params, "limit": self.page_size, "offset": offset})
            rows.extend(page)
            if len(page) < self.page_size:
                return rows
            offset += self.page_size

    @classmethod
    def _conversation(cls, row: dict[str, Any]) -> Conversation:
        record = cls._object(row.get("record"))
        return cls._model(Conversation, {**record, "revision": row.get("revision", 0)})

    def save_journal_entry(self, entry: JournalEntry) -> JournalEntry:
        saved = self._request(
            "POST",
            "rpc/jp_save_journal_entry",
            json=self._signed("save_journal_entry", entry.user_id, {"entry": entry.model_dump(mode="json")}),
        )
        return self._model(JournalEntry, saved)

    def get_journal_entry(self, user_id: UUID, entry_id: UUID) -> JournalEntry | None:
        rows = self._request(
            "GET",
            "journal_entries",
            params={"select": "record", "id": f"eq.{entry_id}", "user_id": f"eq.{user_id}", "limit": 1},
        )
        return self._row_model(JournalEntry, rows[0]) if rows else None

    def list_journal_entries(
        self,
        user_id: UUID,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[JournalEntry]:
        rows = self._request(
            "GET",
            "journal_entries",
            params={
                "select": "record",
                "user_id": f"eq.{user_id}",
                "order": "created_at.desc,id.desc",
                "limit": limit,
                "offset": offset,
            },
        )
        return [self._row_model(JournalEntry, row) for row in rows]

    def delete_journal_entry(self, user_id: UUID, entry_id: UUID) -> bool:
        deleted = self._request(
            "POST",
            "rpc/jp_delete_journal_entry",
            json=self._signed("delete_journal_entry", user_id, {"entry_id": str(entry_id)}),
        )
        return self._boolean(deleted)

    def save_reflection(self, record: ReflectionRecord) -> ReflectionRecord:
        saved = self._request(
            "POST",
            "rpc/jp_save_reflection",
            json=self._signed(
                "save_reflection", record.user_id, {"reflection": record.model_dump(mode="json")}
            ),
        )
        return self._model(ReflectionRecord, saved)

    def save_outcome(self, record: OutcomeRecord) -> OutcomeRecord:
        saved = self._request(
            "POST", "rpc/save_outcome_record", json={"payload": record.model_dump(mode="json")}
        )
        return self._model(OutcomeRecord, saved)

    def list_reflections(self, user_id: UUID, *, limit: int = 50, offset: int = 0) -> list[ReflectionRecord]:
        rows = self._request(
            "GET",
            "reflections",
            params={
                "select": "record",
                "user_id": f"eq.{user_id}",
                "order": "created_at.desc,id.desc",
                "limit": limit,
                "offset": offset,
            },
        )
        return [self._row_model(ReflectionRecord, row) for row in rows]

    def list_all_reflections(self, user_id: UUID) -> list[ReflectionRecord]:
        rows = self._select_all(
            "reflections",
            {"select": "record", "user_id": f"eq.{user_id}", "order": "created_at.desc,id.desc"},
        )
        return [self._row_model(ReflectionRecord, row) for row in rows]

    def get_reflection(self, user_id: UUID, reflection_id: UUID) -> ReflectionRecord | None:
        rows = self._request(
            "GET",
            "reflections",
            params={"select": "record", "id": f"eq.{reflection_id}", "user_id": f"eq.{user_id}", "limit": 1},
        )
        return self._row_model(ReflectionRecord, rows[0]) if rows else None

    def list_outcomes(self, user_id: UUID) -> list[OutcomeRecord]:
        rows = self._select_all(
            "outcomes", {"select": "record", "user_id": f"eq.{user_id}", "order": "created_at.desc,id.desc"}
        )
        return [self._row_model(OutcomeRecord, row) for row in rows]

    def delete_reflection(self, user_id: UUID, reflection_id: UUID) -> bool:
        rows = self._request(
            "DELETE", "reflections", params={"id": f"eq.{reflection_id}", "user_id": f"eq.{user_id}"}
        )
        return bool(self._rows(rows))

    def export_user_data(self, user_id: UUID) -> dict:
        owned = {"user_id": f"eq.{user_id}"}
        conversations = self._select_all(
            "conversations", {**owned, "select": "record,revision", "order": "created_at.asc,id.asc"}
        )
        messages = self._select_all(
            "conversation_messages", {**owned, "select": "record", "order": "created_at.asc,id.asc"}
        )
        audit: dict[str, list[dict[str, Any]]] = {}
        for table in ("policy_decisions", "model_runs", "safety_events", "affective_observations"):
            order_column = "observed_at" if table == "affective_observations" else "created_at"
            audit[table] = self._select_all(
                table, {**owned, "select": "*", "order": f"{order_column}.asc,id.asc"}
            )
        journal_rows = self._select_all(
            "journal_entries",
            {**owned, "select": "record", "order": "created_at.desc,id.desc"},
        )
        return {
            "journal_entries": [
                self._row_model(JournalEntry, row).model_dump(mode="json") for row in journal_rows
            ],
            "reflections": [item.model_dump(mode="json") for item in self.list_all_reflections(user_id)],
            "outcomes": [item.model_dump(mode="json") for item in self.list_outcomes(user_id)],
            "conversations": [self._conversation(row).model_dump(mode="json") for row in conversations],
            "conversation_messages": [
                self._row_model(ConversationMessage, row).model_dump(mode="json") for row in messages
            ],
            "conversation_preference_requests": self._select_all(
                "conversation_preference_requests",
                {**owned, "select": "*", "order": "created_at.asc,id.asc"},
            ),
            "activity_sessions": [item.storage_payload() for item in self.list_activity_sessions(user_id)],
            "activity_receipts": self._select_all(
                "activity_receipts",
                {**owned, "select": "*", "order": "created_at.asc,id.asc"},
            ),
            **audit,
        }

    def delete_user_data(self, user_id: UUID) -> int:
        del user_id
        rows = self._request("POST", "rpc/delete_my_journalpulse_data", json={})
        return self._nonnegative_integer(rows)

    def create_conversation(self, conversation: Conversation) -> Conversation:
        saved = self._request(
            "POST",
            "rpc/jp_create_conversation",
            json=self._signed(
                "create_conversation",
                conversation.user_id,
                {"conversation": conversation.model_dump(mode="json")},
            ),
        )
        return self._model(Conversation, saved)

    def get_conversation(self, user_id: UUID, conversation_id: UUID) -> Conversation | None:
        rows = self._request(
            "GET",
            "conversations",
            params={
                "select": "record,revision",
                "id": f"eq.{conversation_id}",
                "user_id": f"eq.{user_id}",
                "limit": 1,
            },
        )
        return self._conversation(rows[0]) if rows else None

    def list_messages(self, user_id: UUID, conversation_id: UUID) -> list[ConversationMessage]:
        rows = self._select_all(
            "conversation_messages",
            {
                "select": "record",
                "user_id": f"eq.{user_id}",
                "conversation_id": f"eq.{conversation_id}",
                "order": "created_at.asc,id.asc",
            },
        )
        return [self._row_model(ConversationMessage, row) for row in rows]

    def commit_turn(
        self,
        conversation: Conversation,
        user_message: ConversationMessage,
        assistant_message: ConversationMessage,
        *,
        expected_revision: int,
    ) -> Turn:
        saved = self._request(
            "POST",
            "rpc/jp_commit_turn",
            json=self._signed(
                "commit_turn",
                conversation.user_id,
                {
                    "expected_revision": expected_revision,
                    "conversation": conversation.model_dump(mode="json"),
                    "user_message": user_message.model_dump(mode="json"),
                    "assistant_message": assistant_message.model_dump(mode="json"),
                },
            ),
        )
        receipt = self._object(saved)
        return (
            self._model(Conversation, receipt.get("conversation")),
            self._model(ConversationMessage, receipt.get("user_message")),
            self._model(ConversationMessage, receipt.get("assistant_message")),
        )

    def accept_conversation(
        self,
        user_id: UUID,
        conversation_id: UUID,
        record: ReflectionRecord,
        *,
        expected_revision: int,
    ) -> ReflectionRecord:
        saved = self._request(
            "POST",
            "rpc/jp_accept_conversation",
            json=self._signed(
                "accept_conversation",
                user_id,
                {
                    "conversation_id": str(conversation_id),
                    "expected_revision": expected_revision,
                    "client_card_revision": expected_revision,
                    "reflection": record.model_dump(mode="json"),
                },
            ),
        )
        return self._model(ReflectionRecord, saved)

    def close_conversation(self, user_id: UUID, conversation_id: UUID) -> Conversation | None:
        del user_id
        try:
            saved = self._request(
                "POST", "rpc/jp_close_conversation", json={"p_conversation_id": str(conversation_id)}
            )
        except ConversationNotFound:
            return None
        return self._model(Conversation, saved)

    def delete_conversation(self, user_id: UUID, conversation_id: UUID) -> bool:
        del user_id
        deleted = self._request(
            "POST", "rpc/jp_delete_conversation", json={"p_conversation_id": str(conversation_id)}
        )
        return self._boolean(deleted)

    def close_stale_conversations(self, user_id: UUID, *, now: datetime) -> int:
        del user_id, now
        result = self._request("POST", "rpc/jp_close_my_stale_conversations", json={})
        return self._nonnegative_integer(self._object(result).get("closed"))

    def consume_rate_limit(
        self, user_id: UUID, bucket: str, *, limit: int, window_seconds: int, now: datetime
    ) -> tuple[bool, int]:
        del now
        result = self._request(
            "POST",
            "rpc/jp_consume_rate_limit_v2",
            json=self._signed(
                "consume_rate_limit",
                user_id,
                {
                    "bucket": bucket,
                    "max_events": limit,
                    "window_seconds": window_seconds,
                },
            ),
        )
        receipt = self._object(result)
        allowed = self._boolean(receipt.get("allowed"))
        retry_after = self._nonnegative_integer(receipt.get("retry_after"))
        if not allowed and retry_after < 1:
            raise StorageUnavailable("The database returned an invalid response")
        return allowed, retry_after


def supabase_readiness(settings: Settings, client: httpx.Client | None = None) -> dict[str, str]:
    """Probe the database with the public key: reachable, schema current, signing key shared."""
    if not settings.supabase_enabled:
        return {"database": "not_configured"}
    assert settings.supabase_url is not None
    probe = readiness_probe(settings.write_signing_key or "missing-key-for-probe-only-0000000")
    try:
        with managed_http_client(client, timeout=5) as transport:
            response = transport.post(
                # Keep the live reliability API's RPC unchanged during a preview rollout.
                f"{settings.supabase_url.rstrip('/')}/rest/v1/rpc/jp_readiness_v3",
                headers={
                    "apikey": settings.supabase_anon_key or "",
                    "Authorization": f"Bearer {settings.supabase_anon_key or ''}",
                    "Content-Type": "application/json",
                },
                json=probe,
                timeout=5,
                follow_redirects=False,
            )
    except httpx.HTTPError:
        return {"database": "unreachable"}
    if response.status_code == 404:
        return {"database": "reachable", "schema": "schema_missing"}
    if response.status_code >= 400:
        return {"database": f"error:{response.status_code}"}
    try:
        body = response.json()
    except (ValueError, RecursionError):
        return {"database": "invalid_response"}
    if not isinstance(body, dict):
        return {"database": "invalid_response"}
    signing = str(body.get("signing", "unknown"))
    if not settings.write_signing_key:
        signing = "not_configured"
    return {
        "database": "reachable",
        "schema": "schema_ready"
        if body.get("schema") == "guided-action-1" and body.get("activities") == "ready"
        else f"schema_outdated:{body.get('schema')}",
        "signing": signing,
        "activities": str(body.get("activities", "missing")),
        "retention_job": str(body.get("retention_job", "unknown")),
        "retention_state": str(body.get("retention_state", "unknown")),
        "retention_last_success": str(body.get("retention_last_success", "unknown")),
        "retention_last_run_status": str(body.get("retention_last_run_status", "unknown")),
        "retention_overdue": str(body.get("retention_overdue", "unknown")),
    }
