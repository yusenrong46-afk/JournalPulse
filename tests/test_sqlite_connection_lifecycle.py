import sqlite3
from pathlib import Path

import pytest

from journalpulse.persistence import SQLiteRepository


def test_read_connections_close_after_commit_and_rollback(tmp_path: Path):
    repository = SQLiteRepository(tmp_path / "test.db")
    with repository.connect() as committed:
        committed.execute("CREATE TABLE lifecycle_check (value TEXT)")
        committed.execute("INSERT INTO lifecycle_check VALUES ('committed')")
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        committed.execute("SELECT 1")

    with pytest.raises(RuntimeError, match="abort"):
        with repository.connect() as rolled_back:
            rolled_back.execute("INSERT INTO lifecycle_check VALUES ('rolled back')")
            raise RuntimeError("abort")
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        rolled_back.execute("SELECT 1")
    with repository.connect() as checking:
        assert [row[0] for row in checking.execute("SELECT value FROM lifecycle_check")] == ["committed"]
