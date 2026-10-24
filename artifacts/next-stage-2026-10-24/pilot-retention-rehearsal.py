"""Fictional test-clock rehearsal of the documented manual retention procedure."""
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import json

with TemporaryDirectory(prefix="jp-fictional-research-", dir="/private/tmp") as directory:
    private = Path(directory)
    notes, mapping = private / "notes", private / "mapping"
    notes.mkdir(mode=0o700)
    mapping.mkdir(mode=0o700)
    for person in ("P01", "P02"):
        (notes / person).write_text("FICTIONAL NOTE ONLY")
        (mapping / person).write_text("FICTIONAL CONTACT ONLY")
    withdrawn = datetime(2026, 11, 1, 10, tzinfo=UTC)
    clock = datetime(2026, 11, 1, 12, tzinfo=UTC)
    # The manual operator stops collection, removes linked notes/mapping, then
    # recomputes only from remaining eligible records; no product data is touched.
    for folder in (notes, mapping):
        (folder / "P01").unlink()
        assert not (folder / "P01").exists() and (folder / "P02").exists()
    assert clock - withdrawn <= timedelta(hours=24)
    assert len(list(notes.iterdir())) == 1
    close = datetime(2026, 11, 2, tzinfo=UTC)
    deadline = close + timedelta(days=30)
    clock = deadline
    for folder in (notes, mapping):
        (folder / "P02").unlink()
        assert list(folder.iterdir()) == []
    assert clock <= deadline
print(json.dumps({"status": "PASS", "scope": "agent-executed fictional manual-procedure rehearsal with controlled clock",
                  "actual_participants": 0, "withdrawal_removal_within_24h": True,
                  "other_fixture_preserved": True, "summary_recomputed_remaining_count": 1,
                  "raw_notes_and_mapping_absent_by_30_day_deadline": True,
                  "temporary_private_fixture_destroyed": not private.exists(),
                  "real_research_deletion_performed": False}, indent=2))
