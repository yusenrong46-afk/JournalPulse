from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from math import ceil
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID

import httpx

from .config import Settings
from .domain import (
    Conversation,
    ConversationMessage,
    ConversationStatus,
    OutcomeRecord,
    ReflectionRecord,
)
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


Turn = tuple[Conversation, ConversationMessage, ConversationMessage]


class Repository(Protocol):
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
    def close_conversation(self, user_id: UUID, conversation_id: UUID) -> Conversation | None: ...
    def delete_conversation(self, user_id: UUID, conversation_id: UUID) -> bool: ...
    def close_stale_conversations(self, user_id: UUID, *, now: datetime) -> int: ...
    def consume_rate_limit(
        self, user_id: UUID, bucket: str, *, limit: int, window_seconds: int, now: datetime
    ) -> tuple[bool, int]: ...


class SQLiteRepository:
    """Local and test adapter. Enforces the same lifecycle rules as the database functions."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=15)
        connection.row_factory = sqlite3.Row
        return connection

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
                """
            )
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(conversations)")}
            if "revision" not in columns:
                connection.execute("ALTER TABLE conversations ADD COLUMN revision INTEGER NOT NULL DEFAULT 0")
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(reflections)")}
            if "conversation_id" not in columns:
                connection.execute("ALTER TABLE reflections ADD COLUMN conversation_id TEXT")
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_reflections_conversation
                    ON reflections(conversation_id) WHERE conversation_id IS NOT NULL
                """
            )

    # Reflections and outcomes -------------------------------------------------------

    def _insert_reflection(
        self, connection: sqlite3.Connection, record: ReflectionRecord, conversation_id: UUID | None
    ) -> ReflectionRecord:
        existing = connection.execute(
            "SELECT user_id, payload_json FROM reflections WHERE id = ?", (str(record.id),)
        ).fetchone()
        if existing is not None:
            if existing["user_id"] != str(record.user_id):
                raise ValueError("Reflection request ID belongs to another user")
            return ReflectionRecord.model_validate_json(existing["payload_json"])
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
            "reflections": [item.model_dump(mode="json") for item in self.list_all_reflections(user_id)],
            "outcomes": [item.model_dump(mode="json") for item in self.list_outcomes(user_id)],
            "conversations": [item.model_dump(mode="json") for item in self._list_conversations(user_id)],
            "conversation_messages": [
                item.model_dump(mode="json") for item in self._list_all_messages(user_id)
            ],
        }

    def delete_user_data(self, user_id: UUID) -> int:
        """Delete every journal row; the returned count covers journal data only."""
        deleted = 0
        with self.transaction() as connection:
            for table in ("outcomes", "conversation_messages", "reflections", "conversations"):
                cursor = connection.execute(f"DELETE FROM {table} WHERE user_id = ?", (str(user_id),))
                deleted += cursor.rowcount
        # Usage counters hold no journal content and are expired by the retention job;
        # keeping them here stops a journal deletion from resetting the generation limit.
        return deleted

    # Conversations --------------------------------------------------------------------

    @staticmethod
    def _conversation(row: sqlite3.Row) -> Conversation:
        data = json.loads(row["payload_json"])
        data["revision"] = row["revision"]
        data["status"] = row["status"]
        return Conversation.model_validate(data)

    def create_conversation(self, conversation: Conversation) -> Conversation:
        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM conversations WHERE id = ?", (str(conversation.id),)
            ).fetchone()
            if existing is not None:
                if existing["user_id"] != str(conversation.user_id):
                    raise ValueError("Conversation request ID belongs to another user")
                return self._conversation(existing)
            fresh = conversation.model_copy(
                update={"status": ConversationStatus.OPEN, "revision": 0, "reflection_id": None}
            )
            connection.execute(
                """
                INSERT INTO conversations (
                    id, user_id, created_at, updated_at, status, payload_json, revision
                ) VALUES (?, ?, ?, ?, 'open', ?, 0)
                """,
                (
                    str(fresh.id),
                    str(fresh.user_id),
                    fresh.created_at.isoformat(),
                    fresh.updated_at.isoformat(),
                    fresh.model_dump_json(),
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
                return stored
            if row["status"] != ConversationStatus.OPEN.value:
                raise ConversationClosed(str(conversation.id))
            if row["revision"] != expected_revision:
                raise ConversationStale(str(conversation.id))
            committed = conversation.model_copy(
                update={
                    "status": ConversationStatus.OPEN,
                    "revision": expected_revision + 1,
                    "reflection_id": None,
                }
            )
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
                        "SELECT payload_json FROM reflections WHERE id = ?", (str(record.id),)
                    ).fetchone()
                    return ReflectionRecord.model_validate_json(saved["payload_json"])
                raise ConversationAlreadyAccepted(str(conversation_id))
            if current.status != ConversationStatus.OPEN:
                raise ConversationClosed(str(conversation_id))
            if current.revision != expected_revision:
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
                "DELETE FROM conversation_messages WHERE user_id = ? AND conversation_id = ?",
                (str(user_id), str(conversation_id)),
            )
            connection.execute(
                "DELETE FROM conversations WHERE user_id = ? AND id = ?", (str(user_id), str(conversation_id))
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
    except ValueError:
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
        self.client = client or httpx.Client(timeout=15)

    def _request(self, method: str, table: str, **kwargs: Any) -> Any:
        try:
            response = self.client.request(method, f"{self.base_url}/{table}", headers=self.headers, **kwargs)
        except httpx.HTTPError as exc:
            raise StorageUnavailable("The database could not be reached") from exc
        if response.status_code >= 400:
            self._raise_for(response)
        return response.json() if response.content else []

    @staticmethod
    def _raise_for(response: httpx.Response) -> None:
        message = _error_message(response)
        status = response.status_code
        if status == 404 and message == "Conversation not found":
            raise ConversationNotFound(message)
        if status == 409:
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

    @staticmethod
    def _conversation(row: dict[str, Any]) -> Conversation:
        return Conversation.model_validate({**row["record"], "revision": row.get("revision", 0)})

    def save_reflection(self, record: ReflectionRecord) -> ReflectionRecord:
        saved = self._request(
            "POST",
            "rpc/jp_save_reflection",
            json=self._signed(
                "save_reflection", record.user_id, {"reflection": record.model_dump(mode="json")}
            ),
        )
        return ReflectionRecord.model_validate(saved)

    def save_outcome(self, record: OutcomeRecord) -> OutcomeRecord:
        saved = self._request(
            "POST", "rpc/save_outcome_record", json={"payload": record.model_dump(mode="json")}
        )
        return OutcomeRecord.model_validate(saved)

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
        return [ReflectionRecord.model_validate(row["record"]) for row in rows]

    def list_all_reflections(self, user_id: UUID) -> list[ReflectionRecord]:
        rows = self._select_all(
            "reflections",
            {"select": "record", "user_id": f"eq.{user_id}", "order": "created_at.desc,id.desc"},
        )
        return [ReflectionRecord.model_validate(row["record"]) for row in rows]

    def get_reflection(self, user_id: UUID, reflection_id: UUID) -> ReflectionRecord | None:
        rows = self._request(
            "GET",
            "reflections",
            params={"select": "record", "id": f"eq.{reflection_id}", "user_id": f"eq.{user_id}", "limit": 1},
        )
        return ReflectionRecord.model_validate(rows[0]["record"]) if rows else None

    def list_outcomes(self, user_id: UUID) -> list[OutcomeRecord]:
        rows = self._select_all(
            "outcomes", {"select": "record", "user_id": f"eq.{user_id}", "order": "created_at.desc,id.desc"}
        )
        return [OutcomeRecord.model_validate(row["record"]) for row in rows]

    def delete_reflection(self, user_id: UUID, reflection_id: UUID) -> bool:
        rows = self._request(
            "DELETE", "reflections", params={"id": f"eq.{reflection_id}", "user_id": f"eq.{user_id}"}
        )
        return bool(rows)

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
        return {
            "reflections": [item.model_dump(mode="json") for item in self.list_all_reflections(user_id)],
            "outcomes": [item.model_dump(mode="json") for item in self.list_outcomes(user_id)],
            "conversations": [self._conversation(row).model_dump(mode="json") for row in conversations],
            "conversation_messages": [row["record"] for row in messages],
            **audit,
        }

    def delete_user_data(self, user_id: UUID) -> int:
        del user_id
        rows = self._request("POST", "rpc/delete_my_journalpulse_data", json={})
        return rows if isinstance(rows, int) else int(rows or 0)

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
        return Conversation.model_validate(saved)

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
        return [ConversationMessage.model_validate(row["record"]) for row in rows]

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
        return (
            Conversation.model_validate(saved["conversation"]),
            ConversationMessage.model_validate(saved["user_message"]),
            ConversationMessage.model_validate(saved["assistant_message"]),
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
                    "reflection": record.model_dump(mode="json"),
                },
            ),
        )
        return ReflectionRecord.model_validate(saved)

    def close_conversation(self, user_id: UUID, conversation_id: UUID) -> Conversation | None:
        del user_id
        try:
            saved = self._request(
                "POST", "rpc/jp_close_conversation", json={"p_conversation_id": str(conversation_id)}
            )
        except ConversationNotFound:
            return None
        return Conversation.model_validate(saved)

    def delete_conversation(self, user_id: UUID, conversation_id: UUID) -> bool:
        del user_id
        deleted = self._request(
            "POST", "rpc/jp_delete_conversation", json={"p_conversation_id": str(conversation_id)}
        )
        return bool(deleted)

    def close_stale_conversations(self, user_id: UUID, *, now: datetime) -> int:
        del user_id, now
        result = self._request("POST", "rpc/jp_close_my_stale_conversations", json={})
        return int(result.get("closed", 0)) if isinstance(result, dict) else 0

    def consume_rate_limit(
        self, user_id: UUID, bucket: str, *, limit: int, window_seconds: int, now: datetime
    ) -> tuple[bool, int]:
        del user_id, now
        result = self._request(
            "POST",
            "rpc/jp_consume_rate_limit",
            json={"p_bucket": bucket, "p_max_events": limit, "p_window_seconds": window_seconds},
        )
        return bool(result["allowed"]), int(result["retry_after"])


def supabase_readiness(settings: Settings, client: httpx.Client | None = None) -> dict[str, str]:
    """Probe the database with the public key: reachable, schema current, signing key shared."""
    if not settings.supabase_enabled:
        return {"database": "not_configured"}
    assert settings.supabase_url is not None
    probe = readiness_probe(settings.write_signing_key or "missing-key-for-probe-only-0000000")
    try:
        response = (client or httpx.Client(timeout=5)).post(
            f"{settings.supabase_url.rstrip('/')}/rest/v1/rpc/jp_readiness",
            headers={
                "apikey": settings.supabase_anon_key or "",
                "Authorization": f"Bearer {settings.supabase_anon_key or ''}",
                "Content-Type": "application/json",
            },
            json=probe,
        )
    except httpx.HTTPError:
        return {"database": "unreachable"}
    if response.status_code == 404:
        return {"database": "reachable", "schema": "schema_missing"}
    if response.status_code >= 400:
        return {"database": f"error:{response.status_code}"}
    body = response.json()
    signing = str(body.get("signing", "unknown"))
    if not settings.write_signing_key:
        signing = "not_configured"
    return {
        "database": "reachable",
        "schema": "schema_ready"
        if body.get("schema") == "phase-a-1"
        else f"schema_outdated:{body.get('schema')}",
        "signing": signing,
        "retention_job": str(body.get("retention_job", "unknown")),
    }
