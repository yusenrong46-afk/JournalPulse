"""Chronological journal paging stays indexed, including mixed UTC offsets."""

from datetime import datetime
from pathlib import Path
from uuid import uuid4

from journalpulse.journal_models import JournalEntry
from journalpulse.persistence import SQLiteRepository


def test_journal_paging_matches_instants_and_avoids_temporary_sort(tmp_path: Path):
    repo = SQLiteRepository(tmp_path / "journal.db")
    owner = uuid4()
    # The second timestamp sorts first as text, but represents the older instant.
    moments = ["2026-10-05T08:30:00+00:00", "2026-10-05T09:00:00+01:00"]
    entries = [JournalEntry(user_id=owner, text="Fictional writing.", created_at=datetime.fromisoformat(x))
               for x in moments]
    for entry in entries:
        repo.save_journal_entry(entry)
    assert repo.list_journal_entries(owner, limit=1) == [entries[0]]
    assert repo.list_journal_entries(owner, limit=1, offset=1) == [entries[1]]
    with repo.connect() as connection:
        plan = connection.execute(
            "EXPLAIN QUERY PLAN SELECT payload_json FROM journal_entries WHERE user_id = ? "
            "ORDER BY julianday(created_at) DESC, id DESC LIMIT ? OFFSET ?",
            (str(owner), 1, 1),
        ).fetchall()
    assert not any("TEMP B-TREE" in row["detail"] for row in plan)
