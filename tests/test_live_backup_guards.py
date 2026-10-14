import hashlib
import importlib.util
import io
import tarfile
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "live_backup", Path(__file__).parents[1] / "scripts/live_backup.py",
)
assert spec and spec.loader
backup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backup)


def test_only_approved_project_and_session_connections_are_accepted():
    params = backup.source_parameters(
        "postgresql://postgres.cggixttnyprrbaotmjxj:fictional@aws-0-ca-central-1.pooler.supabase.com:5432/postgres"
    )
    assert params["sslmode"] == "verify-full"
    assert params["password"] == "fictional"
    for uri in [
        "postgresql://postgres:secret@localhost/postgres",
        "postgresql://postgres.other:secret@aws-0-ca-central-1.pooler.supabase.com/postgres",
        "postgresql://postgres.cggixttnyprrbaotmjxj:secret@evil.example/postgres",
        "postgresql://postgres:secret@db.cggixttnyprrbaotmjxj.supabase.co/postgres?host=evil.example",
        "postgresql://postgres.cggixttnyprrbaotmjxj:secret@aws-0-ca-central-1.pooler.supabase.com:6543/postgres",
        "postgresql://[broken",
    ]:
        with pytest.raises(backup.BackupError) as error:
            backup.source_parameters(uri)
        assert "secret" not in str(error.value) and "evil.example" not in str(error.value)


def test_archive_integrity_and_no_extraction_paths():
    dump = b"fictional custom dump"
    manifest = {"dump_sha256": hashlib.sha256(dump).hexdigest()}
    assert backup.unpack(backup.pack(dump, manifest)) == (dump, manifest)
    with pytest.raises(backup.BackupError):
        backup.unpack(backup.pack(dump, {"dump_sha256": "0" * 64}))
    memory = io.BytesIO()
    with tarfile.open(fileobj=memory, mode="w") as archive:
        item = tarfile.TarInfo("../../outside")
        archive.addfile(item, io.BytesIO())
    with pytest.raises(backup.BackupError):
        backup.unpack(memory.getvalue())


def test_private_directory_refuses_symlink(tmp_path):
    destination = tmp_path / "real"
    destination.mkdir()
    link = tmp_path / "link"
    link.symlink_to(destination)
    with pytest.raises(backup.BackupError):
        backup.private_directory(link)
    backup.private_directory(destination)
    assert destination.stat().st_mode & 0o777 == 0o700
