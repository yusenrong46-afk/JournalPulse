from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Protocol
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


class DuplicateOutcomeError(ValueError):
    pass


class Repository(Protocol):
    def save_reflection(self, record: ReflectionRecord) -> ReflectionRecord: ...
    def save_outcome(self, record: OutcomeRecord) -> OutcomeRecord: ...
    def list_reflections(self, user_id: UUID, *, limit: int, offset: int) -> list[ReflectionRecord]: ...
    def list_outcomes(self, user_id: UUID) -> list[OutcomeRecord]: ...
    def delete_reflection(self, user_id: UUID, reflection_id: UUID) -> bool: ...
    def export_user_data(self, user_id: UUID) -> dict: ...
    def delete_user_data(self, user_id: UUID) -> int: ...
    def save_conversation(self, conversation: Conversation) -> Conversation: ...
    def get_conversation(self, user_id: UUID, conversation_id: UUID) -> Conversation | None: ...
    def list_messages(self, user_id: UUID, conversation_id: UUID) -> list[ConversationMessage]: ...
    def save_turn(
        self,
        conversation: Conversation,
        user_message: ConversationMessage,
        assistant_message: ConversationMessage,
    ) -> tuple[Conversation, ConversationMessage, ConversationMessage]: ...
    def close_conversation(
        self,
        user_id: UUID,
        conversation_id: UUID,
        *,
        purge: bool,
        reflection_id: UUID | None = None,
        now: datetime,
    ) -> Conversation | None: ...
    def delete_conversation(self, user_id: UUID, conversation_id: UUID) -> bool: ...
    def close_stale_conversations(
        self, user_id: UUID, *, older_than: datetime, now: datetime
    ) -> int: ...


class SQLiteRepository:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

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
                CREATE INDEX IF NOT EXISTS idx_outcomes_user_decision
                    ON outcomes(user_id, decision_id);
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
                """
            )

    def save_reflection(self, record: ReflectionRecord) -> ReflectionRecord:
        payload = record.model_dump(mode="json")
        with self.connect() as connection:
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
                    INSERT INTO reflections (id, user_id, created_at, text, payload_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        str(record.id),
                        str(record.user_id),
                        record.created_at.isoformat(),
                        record.text,
                        json.dumps(payload),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("Reflection request ID is already in use") from exc
        return record

    def save_outcome(self, record: OutcomeRecord) -> OutcomeRecord:
        payload = record.model_dump(mode="json")
        with self.connect() as connection:
            existing = connection.execute(
                "SELECT user_id, decision_id, payload_json FROM outcomes WHERE id = ?",
                (str(record.id),),
            ).fetchone()
            if existing is not None:
                if (
                    existing["user_id"] != str(record.user_id)
                    or existing["decision_id"] != str(record.decision_id)
                ):
                    raise ValueError("Outcome request ID is already in use")
                return OutcomeRecord.model_validate_json(existing["payload_json"])
            reflections = connection.execute(
                "SELECT payload_json FROM reflections WHERE user_id = ?", (str(record.user_id),)
            ).fetchall()
            if not any(
                ReflectionRecord.model_validate_json(item["payload_json"]).decision.decision_id
                == record.decision_id
                for item in reflections
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
                        json.dumps(payload),
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
                WHERE user_id = ? ORDER BY datetime(created_at) DESC LIMIT ? OFFSET ?
                """,
                (str(user_id), limit, offset),
            ).fetchall()
        return [ReflectionRecord.model_validate_json(row["payload_json"]) for row in rows]

    def list_outcomes(self, user_id: UUID) -> list[OutcomeRecord]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM outcomes WHERE user_id = ? ORDER BY datetime(created_at) DESC",
                (str(user_id),),
            ).fetchall()
        return [OutcomeRecord.model_validate_json(row["payload_json"]) for row in rows]

    def delete_reflection(self, user_id: UUID, reflection_id: UUID) -> bool:
        with self.connect() as connection:
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
                "DELETE FROM reflections WHERE id = ? AND user_id = ?",
                (str(reflection_id), str(user_id)),
            )
        return True

    def export_user_data(self, user_id: UUID) -> dict:
        return {
            "reflections": [
                item.model_dump(mode="json") for item in self.list_reflections(user_id, limit=10000)
            ],
            "outcomes": [item.model_dump(mode="json") for item in self.list_outcomes(user_id)],
            "conversations": [
                item.model_dump(mode="json") for item in self._list_conversations(user_id)
            ],
            "conversation_messages": [
                item.model_dump(mode="json") for item in self._list_all_messages(user_id)
            ],
        }

    def delete_user_data(self, user_id: UUID) -> int:
        with self.connect() as connection:
            reflection_count = connection.execute(
                "SELECT COUNT(*) FROM reflections WHERE user_id = ?", (str(user_id),)
            ).fetchone()[0]
            outcome_count = connection.execute(
                "SELECT COUNT(*) FROM outcomes WHERE user_id = ?", (str(user_id),)
            ).fetchone()[0]
            conversation_count = connection.execute(
                "SELECT COUNT(*) FROM conversations WHERE user_id = ?", (str(user_id),)
            ).fetchone()[0]
            message_count = connection.execute(
                "SELECT COUNT(*) FROM conversation_messages WHERE user_id = ?", (str(user_id),)
            ).fetchone()[0]
            connection.execute("DELETE FROM outcomes WHERE user_id = ?", (str(user_id),))
            connection.execute("DELETE FROM conversation_messages WHERE user_id = ?", (str(user_id),))
            connection.execute("DELETE FROM conversations WHERE user_id = ?", (str(user_id),))
            connection.execute("DELETE FROM reflections WHERE user_id = ?", (str(user_id),))
        return int(reflection_count + outcome_count + conversation_count + message_count)

    def save_conversation(self, conversation: Conversation) -> Conversation:
        payload = json.dumps(conversation.model_dump(mode="json"))
        with self.connect() as connection:
            existing = connection.execute(
                "SELECT user_id, payload_json FROM conversations WHERE id = ?",
                (str(conversation.id),),
            ).fetchone()
            if existing is not None:
                if existing["user_id"] != str(conversation.user_id):
                    raise ValueError("Conversation request ID belongs to another user")
                return Conversation.model_validate_json(existing["payload_json"])
            connection.execute(
                """
                INSERT INTO conversations (id, user_id, created_at, updated_at, status, payload_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(conversation.id),
                    str(conversation.user_id),
                    conversation.created_at.isoformat(),
                    conversation.updated_at.isoformat(),
                    conversation.status.value,
                    payload,
                ),
            )
        return conversation

    def get_conversation(self, user_id: UUID, conversation_id: UUID) -> Conversation | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM conversations WHERE id = ? AND user_id = ?",
                (str(conversation_id), str(user_id)),
            ).fetchone()
        if row is None:
            return None
        return Conversation.model_validate_json(row["payload_json"])

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

    def save_turn(
        self,
        conversation: Conversation,
        user_message: ConversationMessage,
        assistant_message: ConversationMessage,
    ) -> tuple[Conversation, ConversationMessage, ConversationMessage]:
        with self.connect() as connection:
            existing = self._existing_turn(connection, user_message)
            if existing is not None:
                return existing
            try:
                self._insert_message(connection, conversation.user_id, user_message)
                self._insert_message(connection, conversation.user_id, assistant_message)
            except sqlite3.IntegrityError:
                existing = self._existing_turn(connection, user_message)
                if existing is None:
                    raise
                return existing
            connection.execute(
                """
                UPDATE conversations
                SET updated_at = ?, status = ?, payload_json = ?
                WHERE id = ? AND user_id = ?
                """,
                (
                    conversation.updated_at.isoformat(),
                    conversation.status.value,
                    json.dumps(conversation.model_dump(mode="json")),
                    str(conversation.id),
                    str(conversation.user_id),
                ),
            )
        return conversation, user_message, assistant_message

    def close_conversation(
        self,
        user_id: UUID,
        conversation_id: UUID,
        *,
        purge: bool,
        reflection_id: UUID | None = None,
        now: datetime,
    ) -> Conversation | None:
        conversation = self.get_conversation(user_id, conversation_id)
        if conversation is None:
            return None
        closed = conversation.model_copy(
            update={
                "status": ConversationStatus.CLOSED,
                "updated_at": now,
                "reflection_id": reflection_id or conversation.reflection_id,
            }
        )
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE conversations
                SET updated_at = ?, status = ?, payload_json = ?
                WHERE id = ? AND user_id = ?
                """,
                (
                    closed.updated_at.isoformat(),
                    closed.status.value,
                    json.dumps(closed.model_dump(mode="json")),
                    str(conversation_id),
                    str(user_id),
                ),
            )
            if purge:
                self._purge_messages(connection, user_id, conversation_id)
        return closed

    def delete_conversation(self, user_id: UUID, conversation_id: UUID) -> bool:
        conversation = self.get_conversation(user_id, conversation_id)
        if conversation is None:
            return False
        if conversation.reflection_id is not None:
            self.delete_reflection(user_id, conversation.reflection_id)
        with self.connect() as connection:
            connection.execute(
                "DELETE FROM conversation_messages WHERE user_id = ? AND conversation_id = ?",
                (str(user_id), str(conversation_id)),
            )
            connection.execute(
                "DELETE FROM conversations WHERE user_id = ? AND id = ?",
                (str(user_id), str(conversation_id)),
            )
        return True

    def close_stale_conversations(self, user_id: UUID, *, older_than: datetime, now: datetime) -> int:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT payload_json FROM conversations
                WHERE user_id = ? AND status = 'open' AND updated_at < ?
                """,
                (str(user_id), older_than.isoformat()),
            ).fetchall()
        closed = 0
        for row in rows:
            conversation = Conversation.model_validate_json(row["payload_json"])
            self.close_conversation(
                user_id,
                conversation.id,
                purge=not conversation.retain_text,
                now=now,
            )
            closed += 1
        return closed

    def _list_conversations(self, user_id: UUID) -> list[Conversation]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM conversations WHERE user_id = ? ORDER BY created_at ASC",
                (str(user_id),),
            ).fetchall()
        return [Conversation.model_validate_json(row["payload_json"]) for row in rows]

    def _list_all_messages(self, user_id: UUID) -> list[ConversationMessage]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT payload_json FROM conversation_messages
                WHERE user_id = ? ORDER BY created_at ASC
                """,
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
                json.dumps(message.model_dump(mode="json")),
            ),
        )

    def _existing_turn(
        self, connection: sqlite3.Connection, user_message: ConversationMessage
    ) -> tuple[Conversation, ConversationMessage, ConversationMessage] | None:
        if user_message.client_message_id is None:
            return None
        row = connection.execute(
            """
            SELECT payload_json FROM conversation_messages
            WHERE conversation_id = ? AND client_message_id = ?
            """,
            (str(user_message.conversation_id), str(user_message.client_message_id)),
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
            (str(user_message.conversation_id), stored_user.created_at.isoformat()),
        ).fetchone()
        conversation_row = connection.execute(
            "SELECT payload_json FROM conversations WHERE id = ?",
            (str(user_message.conversation_id),),
        ).fetchone()
        if assistant_row is None or conversation_row is None:
            return None
        return (
            Conversation.model_validate_json(conversation_row["payload_json"]),
            stored_user,
            ConversationMessage.model_validate_json(assistant_row["payload_json"]),
        )

    def _purge_messages(
        self, connection: sqlite3.Connection, user_id: UUID, conversation_id: UUID
    ) -> None:
        rows = connection.execute(
            """
            SELECT id, payload_json FROM conversation_messages
            WHERE user_id = ? AND conversation_id = ?
            """,
            (str(user_id), str(conversation_id)),
        ).fetchall()
        for row in rows:
            message = ConversationMessage.model_validate_json(row["payload_json"])
            if message.content is None:
                continue
            cleared = message.model_copy(update={"content": None})
            connection.execute(
                "UPDATE conversation_messages SET payload_json = ? WHERE id = ?",
                (json.dumps(cleared.model_dump(mode="json")), row["id"]),
            )


class SupabaseRepository:
    """PostgREST adapter that preserves Supabase RLS by forwarding the user's JWT."""

    def __init__(self, settings: Settings, access_token: str, client: httpx.Client | None = None) -> None:
        if not settings.supabase_enabled:
            raise ValueError("Supabase is not configured")
        assert settings.supabase_url is not None
        self.base_url = f"{settings.supabase_url.rstrip('/')}/rest/v1"
        self.headers = {
            "apikey": settings.supabase_anon_key or "",
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }
        self.client = client or httpx.Client(timeout=15)

    def _request(self, method: str, table: str, **kwargs):
        response = self.client.request(method, f"{self.base_url}/{table}", headers=self.headers, **kwargs)
        response.raise_for_status()
        return response.json() if response.content else []

    def save_reflection(self, record: ReflectionRecord) -> ReflectionRecord:
        payload = record.model_dump(mode="json")
        saved = self._request("POST", "rpc/save_reflection_bundle", json={"payload": payload})
        return ReflectionRecord.model_validate(saved)

    def save_outcome(self, record: OutcomeRecord) -> OutcomeRecord:
        try:
            saved = self._request(
                "POST",
                "rpc/save_outcome_record",
                json={"payload": record.model_dump(mode="json")},
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 409:
                raise DuplicateOutcomeError("An outcome already exists for this decision") from exc
            if exc.response.status_code in {400, 404}:
                raise ValueError("Policy decision does not belong to this user") from exc
            raise
        return OutcomeRecord.model_validate(saved)

    def list_reflections(self, user_id: UUID, *, limit: int = 50, offset: int = 0) -> list[ReflectionRecord]:
        rows = self._request(
            "GET",
            "reflections",
            params={
                "select": "record",
                "user_id": f"eq.{user_id}",
                "order": "created_at.desc",
                "limit": limit,
                "offset": offset,
            },
        )
        return [ReflectionRecord.model_validate(row["record"]) for row in rows]

    def list_outcomes(self, user_id: UUID) -> list[OutcomeRecord]:
        rows = self._request(
            "GET",
            "outcomes",
            params={"select": "record", "user_id": f"eq.{user_id}", "order": "created_at.desc"},
        )
        return [OutcomeRecord.model_validate(row["record"]) for row in rows]

    def delete_reflection(self, user_id: UUID, reflection_id: UUID) -> bool:
        rows = self._request(
            "DELETE",
            "reflections",
            params={"id": f"eq.{reflection_id}", "user_id": f"eq.{user_id}"},
        )
        return bool(rows)

    def export_user_data(self, user_id: UUID) -> dict:
        conversations = self._request(
            "GET",
            "conversations",
            params={"select": "record", "user_id": f"eq.{user_id}", "order": "created_at.asc"},
        )
        messages = self._request(
            "GET",
            "conversation_messages",
            params={"select": "record", "user_id": f"eq.{user_id}", "order": "created_at.asc"},
        )
        return {
            "reflections": [
                item.model_dump(mode="json") for item in self.list_reflections(user_id, limit=10000)
            ],
            "outcomes": [item.model_dump(mode="json") for item in self.list_outcomes(user_id)],
            "conversations": [row["record"] for row in conversations],
            "conversation_messages": [row["record"] for row in messages],
        }

    def delete_user_data(self, user_id: UUID) -> int:
        rows = self._request("POST", "rpc/delete_my_journalpulse_data", json={})
        if isinstance(rows, int):
            return rows
        return int(rows or 0)

    def save_conversation(self, conversation: Conversation) -> Conversation:
        existing = self.get_conversation(conversation.user_id, conversation.id)
        if existing is not None:
            return existing
        saved = self._request(
            "POST",
            "conversations",
            json={
                "id": str(conversation.id),
                "user_id": str(conversation.user_id),
                "created_at": conversation.created_at.isoformat(),
                "updated_at": conversation.updated_at.isoformat(),
                "status": conversation.status.value,
                "reflection_id": (
                    str(conversation.reflection_id) if conversation.reflection_id else None
                ),
                "record": conversation.model_dump(mode="json"),
            },
        )
        row = saved[0] if isinstance(saved, list) else saved
        return Conversation.model_validate(row["record"])

    def get_conversation(self, user_id: UUID, conversation_id: UUID) -> Conversation | None:
        rows = self._request(
            "GET",
            "conversations",
            params={
                "select": "record",
                "id": f"eq.{conversation_id}",
                "user_id": f"eq.{user_id}",
                "limit": 1,
            },
        )
        if not rows:
            return None
        return Conversation.model_validate(rows[0]["record"])

    def list_messages(self, user_id: UUID, conversation_id: UUID) -> list[ConversationMessage]:
        rows = self._request(
            "GET",
            "conversation_messages",
            params={
                "select": "record",
                "user_id": f"eq.{user_id}",
                "conversation_id": f"eq.{conversation_id}",
                "order": "created_at.asc",
            },
        )
        return [ConversationMessage.model_validate(row["record"]) for row in rows]

    def save_turn(
        self,
        conversation: Conversation,
        user_message: ConversationMessage,
        assistant_message: ConversationMessage,
    ) -> tuple[Conversation, ConversationMessage, ConversationMessage]:
        saved = self._request(
            "POST",
            "rpc/save_conversation_turn",
            json={
                "payload": {
                    "conversation": conversation.model_dump(mode="json"),
                    "user_message": user_message.model_dump(mode="json"),
                    "assistant_message": assistant_message.model_dump(mode="json"),
                }
            },
        )
        return (
            Conversation.model_validate(saved["conversation"]),
            ConversationMessage.model_validate(saved["user_message"]),
            ConversationMessage.model_validate(saved["assistant_message"]),
        )

    def close_conversation(
        self,
        user_id: UUID,
        conversation_id: UUID,
        *,
        purge: bool,
        reflection_id: UUID | None = None,
        now: datetime,
    ) -> Conversation | None:
        del now
        current = self.get_conversation(user_id, conversation_id)
        if current is None:
            return None
        if reflection_id is not None:
            linked = current.model_copy(update={"reflection_id": reflection_id})
            self._request(
                "PATCH",
                "conversations",
                params={"id": f"eq.{conversation_id}", "user_id": f"eq.{user_id}"},
                json={
                    "reflection_id": str(reflection_id),
                    "record": linked.model_dump(mode="json"),
                },
            )
        saved = self._request(
            "POST",
            "rpc/close_conversation",
            json={"conversation_id": str(conversation_id), "purge": purge},
        )
        return Conversation.model_validate(saved)

    def delete_conversation(self, user_id: UUID, conversation_id: UUID) -> bool:
        conversation = self.get_conversation(user_id, conversation_id)
        if conversation is None:
            return False
        if conversation.reflection_id is not None:
            self.delete_reflection(user_id, conversation.reflection_id)
        self._request(
            "DELETE",
            "conversation_messages",
            params={"conversation_id": f"eq.{conversation_id}", "user_id": f"eq.{user_id}"},
        )
        self._request(
            "DELETE",
            "conversations",
            params={"id": f"eq.{conversation_id}", "user_id": f"eq.{user_id}"},
        )
        return True

    def close_stale_conversations(self, user_id: UUID, *, older_than: datetime, now: datetime) -> int:
        rows = self._request(
            "GET",
            "conversations",
            params={
                "select": "record",
                "user_id": f"eq.{user_id}",
                "status": "eq.open",
                "updated_at": f"lt.{older_than.isoformat()}",
            },
        )
        closed = 0
        for row in rows:
            conversation = Conversation.model_validate(row["record"])
            self.close_conversation(
                user_id,
                conversation.id,
                purge=not conversation.retain_text,
                now=now,
            )
            closed += 1
        return closed
