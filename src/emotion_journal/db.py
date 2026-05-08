import json
import sqlite3
from pathlib import Path
from typing import Optional

from .analytics import build_analytics
from .config import DEFAULT_DB_PATH


ENTRY_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS journal_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    text TEXT NOT NULL,
    emotion TEXT NOT NULL,
    confidence REAL NOT NULL,
    recommendation TEXT NOT NULL,
    location TEXT,
    activity TEXT,
    feedback TEXT
);
CREATE INDEX IF NOT EXISTS idx_journal_entries_emotion
    ON journal_entries(emotion);
CREATE INDEX IF NOT EXISTS idx_journal_entries_created_at
    ON journal_entries(created_at);
"""

RESOURCE_INTERACTIONS_SQL = """
CREATE TABLE IF NOT EXISTS resource_interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    resource_id TEXT NOT NULL,
    action TEXT NOT NULL,
    emotion TEXT NOT NULL,
    entry_id INTEGER
);
CREATE INDEX IF NOT EXISTS idx_resource_interactions_resource_id
    ON resource_interactions(resource_id);
CREATE INDEX IF NOT EXISTS idx_resource_interactions_action
    ON resource_interactions(action);
CREATE INDEX IF NOT EXISTS idx_resource_interactions_created_at
    ON resource_interactions(created_at);
"""

ENTRY_ADDITIONAL_COLUMNS = {
    "reflection_summary": "TEXT",
    "interpretation": "TEXT",
    "confidence_band": "TEXT",
    "model_name": "TEXT",
    "support_message": "TEXT",
    "follow_up_prompts_json": "TEXT",
    "explanation_phrases_json": "TEXT",
    "coach_state_summary": "TEXT",
    "coach_summary_json": "TEXT",
    "suggested_resource_ids_json": "TEXT",
}


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def _existing_columns(connection: sqlite3.Connection, table_name: str) -> set:
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row[1] for row in rows}


def _serialize_list(values) -> Optional[str]:
    if not values:
        return None
    return json.dumps(values)


def _serialize_object(value) -> Optional[str]:
    if not value:
        return None
    return json.dumps(value)


def _legacy_coach_summary(entry: dict) -> Optional[dict]:
    summary = entry.get("coach_state_summary")
    if not summary:
        return None

    parts = {}
    for part in str(summary).split("|"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        parts[key] = value

    selected_style = parts.get("style")
    if selected_style == "none":
        selected_style = None
    return {
        "turn_count": 0,
        "final_step": parts.get("step"),
        "framing_emotion": parts.get("emotion"),
        "selected_coping_style": selected_style,
        "resource_ids": entry.get("suggested_resource_ids", []),
        "used_llm": False,
        "safety_mode": bool(entry.get("support_message")),
    }


def _deserialize_entry(row: sqlite3.Row) -> dict:
    entry = dict(row)
    entry["follow_up_prompts"] = (
        json.loads(entry.pop("follow_up_prompts_json"))
        if entry.get("follow_up_prompts_json")
        else []
    )
    entry["explanation_phrases"] = (
        json.loads(entry.pop("explanation_phrases_json"))
        if entry.get("explanation_phrases_json")
        else []
    )
    entry["suggested_resource_ids"] = (
        json.loads(entry.pop("suggested_resource_ids_json"))
        if entry.get("suggested_resource_ids_json")
        else []
    )
    entry["coach_summary"] = (
        json.loads(entry.pop("coach_summary_json"))
        if entry.get("coach_summary_json")
        else None
    )
    if entry["coach_summary"] is None:
        entry["coach_summary"] = _legacy_coach_summary(entry)
    return entry


def initialize_database(db_path: Path = DEFAULT_DB_PATH) -> None:
    with _connect(db_path) as connection:
        connection.executescript(ENTRY_SCHEMA_SQL)
        connection.executescript(RESOURCE_INTERACTIONS_SQL)
        existing = _existing_columns(connection, "journal_entries")
        for column_name, column_type in ENTRY_ADDITIONAL_COLUMNS.items():
            if column_name not in existing:
                connection.execute(
                    f"ALTER TABLE journal_entries ADD COLUMN {column_name} {column_type}"
                )
        connection.commit()


def insert_entry(
    *,
    text: str,
    emotion: str,
    confidence: float,
    recommendation: str,
    location: Optional[str] = None,
    activity: Optional[str] = None,
    feedback: Optional[str] = None,
    reflection_summary: Optional[str] = None,
    interpretation: Optional[str] = None,
    confidence_band: Optional[str] = None,
    model_name: Optional[str] = None,
    support_message: Optional[str] = None,
    follow_up_prompts=None,
    explanation_phrases=None,
    coach_state_summary: Optional[str] = None,
    coach_summary=None,
    suggested_resource_ids=None,
    db_path: Path = DEFAULT_DB_PATH,
) -> dict:
    initialize_database(db_path)
    with _connect(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO journal_entries (
                text, emotion, confidence, recommendation, location, activity, feedback,
                reflection_summary, interpretation, confidence_band, model_name, support_message,
                follow_up_prompts_json, explanation_phrases_json, coach_state_summary, coach_summary_json,
                suggested_resource_ids_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                text,
                emotion,
                confidence,
                recommendation,
                location,
                activity,
                feedback,
                reflection_summary,
                interpretation,
                confidence_band,
                model_name,
                support_message,
                _serialize_list(follow_up_prompts),
                _serialize_list(explanation_phrases),
                coach_state_summary,
                _serialize_object(coach_summary),
                _serialize_list(suggested_resource_ids),
            ),
        )
        entry_id = cursor.lastrowid
        connection.commit()
    return get_entry(entry_id, db_path=db_path)


def get_entry(entry_id: int, db_path: Path = DEFAULT_DB_PATH) -> dict:
    initialize_database(db_path)
    with _connect(db_path) as connection:
        row = connection.execute(
            "SELECT * FROM journal_entries WHERE id = ?",
            (entry_id,),
        ).fetchone()
    if row is None:
        raise KeyError(f"Entry {entry_id} does not exist")
    return _deserialize_entry(row)


def update_feedback(entry_id: int, feedback: str, db_path: Path = DEFAULT_DB_PATH) -> dict:
    initialize_database(db_path)
    with _connect(db_path) as connection:
        cursor = connection.execute(
            "UPDATE journal_entries SET feedback = ? WHERE id = ?",
            (feedback, entry_id),
        )
        if cursor.rowcount == 0:
            raise KeyError(f"Entry {entry_id} does not exist")
        connection.commit()
    return get_entry(entry_id, db_path=db_path)


def list_entries(
    *,
    emotion: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> list:
    initialize_database(db_path)
    clauses = []
    params = []
    if emotion:
        clauses.append("emotion = ?")
        params.append(emotion)
    if start_date:
        clauses.append("date(created_at) >= date(?)")
        params.append(start_date)
    if end_date:
        clauses.append("date(created_at) <= date(?)")
        params.append(end_date)

    sql = "SELECT * FROM journal_entries"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY datetime(created_at) DESC, id DESC"

    with _connect(db_path) as connection:
        rows = connection.execute(sql, tuple(params)).fetchall()
    return [_deserialize_entry(row) for row in rows]


def record_resource_interaction(
    *,
    resource_id: str,
    action: str,
    emotion: str,
    entry_id: Optional[int] = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> dict:
    initialize_database(db_path)
    with _connect(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO resource_interactions (resource_id, action, emotion, entry_id)
            VALUES (?, ?, ?, ?)
            """,
            (resource_id, action, emotion, entry_id),
        )
        interaction_id = cursor.lastrowid
        connection.commit()
        row = connection.execute(
            "SELECT * FROM resource_interactions WHERE id = ?",
            (interaction_id,),
        ).fetchone()
    return dict(row)


def list_resource_interactions(
    *,
    emotion: Optional[str] = None,
    action: Optional[str] = None,
    resource_id: Optional[str] = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> list:
    initialize_database(db_path)
    clauses = []
    params = []
    if emotion:
        clauses.append("emotion = ?")
        params.append(emotion)
    if action:
        clauses.append("action = ?")
        params.append(action)
    if resource_id:
        clauses.append("resource_id = ?")
        params.append(resource_id)

    sql = "SELECT * FROM resource_interactions"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY datetime(created_at) DESC, id DESC"

    with _connect(db_path) as connection:
        rows = connection.execute(sql, tuple(params)).fetchall()
    return [dict(row) for row in rows]


def get_analytics(
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> dict:
    from .resources import load_resource_catalog

    entries = list_entries(start_date=start_date, end_date=end_date, db_path=db_path)
    interactions = list_resource_interactions(db_path=db_path)
    return build_analytics(entries, resource_interactions=interactions, resource_catalog=load_resource_catalog())
