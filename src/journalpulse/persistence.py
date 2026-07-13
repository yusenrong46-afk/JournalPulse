from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Protocol
from uuid import UUID

import httpx

from .config import Settings
from .domain import OutcomeRecord, ReflectionRecord


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
        }

    def delete_user_data(self, user_id: UUID) -> int:
        with self.connect() as connection:
            reflection_count = connection.execute(
                "SELECT COUNT(*) FROM reflections WHERE user_id = ?", (str(user_id),)
            ).fetchone()[0]
            outcome_count = connection.execute(
                "SELECT COUNT(*) FROM outcomes WHERE user_id = ?", (str(user_id),)
            ).fetchone()[0]
            connection.execute("DELETE FROM outcomes WHERE user_id = ?", (str(user_id),))
            connection.execute("DELETE FROM reflections WHERE user_id = ?", (str(user_id),))
        return int(reflection_count + outcome_count)


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
        return {
            "reflections": [
                item.model_dump(mode="json") for item in self.list_reflections(user_id, limit=10000)
            ],
            "outcomes": [item.model_dump(mode="json") for item in self.list_outcomes(user_id)],
        }

    def delete_user_data(self, user_id: UUID) -> int:
        rows = self._request("POST", "rpc/delete_my_journalpulse_data", json={})
        if isinstance(rows, int):
            return rows
        return int(rows or 0)
