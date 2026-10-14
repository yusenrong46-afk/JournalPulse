"""Verify fictional dump/restore and signing overlap in a network-isolated Docker container.

This cannot validate managed backups, PITR, hosted auth or post-backup deletions.
It requires an in-memory database without networks or persistent mounts and refuses existing test databases.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import verify_postgres_schema as base
from journalpulse.journal_models import JournalEntry
from journalpulse.signing import sign_text

SOURCE = "jp_recovery_source"
RESTORED = "jp_recovery_restored"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not args.container.startswith("jp-trust-hardening-"):
        raise SystemExit("Use a task-owned jp-trust-hardening- container.")
    info = json.loads(subprocess.check_output(["docker", "inspect", args.container]))[0]
    if (info["HostConfig"]["NetworkMode"] != "none"
            or "/var/lib/postgresql/data" not in (info["HostConfig"].get("Tmpfs") or {})
            or any(mount["Type"] != "tmpfs" for mount in info["Mounts"])):
        raise SystemExit("Use a network-isolated container with tmpfs data and no persistent mounts.")

    def run(command: list[str], data: bytes | None = None) -> bytes:
        result = subprocess.run(["docker", "exec", "-i", args.container, *command], input=data,
                                capture_output=True, check=False)
        if result.returncode:
            raise RuntimeError(result.stderr.decode())
        return result.stdout

    def sql(database: str, query: str) -> str:
        return run(["psql", "-U", "postgres", "-d", database, "-X", "-At", "-v", "ON_ERROR_STOP=1"],
                   query.encode()).decode()

    existing = sql("postgres", f"select datname from pg_database where datname in ('{SOURCE}','{RESTORED}')")
    if existing.strip():
        raise SystemExit("Refusing to replace existing recovery test databases. Use a fresh container.")
    sql("postgres", f"create database {SOURCE}; create database {RESTORED};")
    sql(SOURCE, (base.ROOT / "scripts/pg_harness.sql").read_text())
    for migration in sorted((base.ROOT / "supabase/migrations").glob("*.sql")):
        sql(SOURCE, migration.read_text())

    old_key = "previous-fictional-test-key-0123456789abcdef"
    future = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
    expired = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    probe = "readiness:fictional-recovery-check"
    current_signature = sign_text(probe, base.SIGNING_KEY)
    previous_signature = sign_text(probe, old_key)
    sql(SOURCE, f"""
      insert into auth.users (id,email) values
        ('{base.USER_A}','recovery-a@example.test'), ('{base.USER_B}','recovery-b@example.test');
      insert into private.server_secrets (name,value) values
        ('write_signing_key',{base.literal(base.SIGNING_KEY)}),
        ('write_signing_key_previous',{base.literal(old_key)});
      update private.server_secrets set expires_at={base.literal(future)}::timestamptz
        where name='write_signing_key_previous';
    """)
    assertions = []

    def expect(database: str, expression: str, label: str) -> None:
        result = sql(database, f"select ({expression})::text;")
        if result.strip() != "true":
            raise AssertionError(label)
        assertions.append(label)

    def valid(signature: str) -> str:
        return f"private.signature_valid({base.literal(probe)},{base.literal(signature)})"

    expect(SOURCE, valid(current_signature), "current key accepted")
    expect(SOURCE, valid(previous_signature), "previous key accepted during overlap")
    expect(SOURCE, f"not {valid('0' * 64)}", "forged signature rejected")
    sql(SOURCE, f"update private.server_secrets set expires_at={base.literal(expired)}::timestamptz "
                "where name='write_signing_key_previous'")
    expect(SOURCE, f"not {valid(previous_signature)}", "previous key rejected after expiry")
    sql(SOURCE, "update private.server_secrets set expires_at=null "
                "where name='write_signing_key_previous'")
    expect(SOURCE, f"not {valid(previous_signature)}", "missing overlap deadline fails closed")
    expect(SOURCE, valid(current_signature), "missing previous deadline does not break current key")
    sql(SOURCE, "update private.server_secrets set expires_at='infinity' "
                "where name='write_signing_key_previous'")
    expect(SOURCE, f"not {valid(previous_signature)}", "infinite overlap fails closed")
    sql(SOURCE, f"update private.server_secrets set expires_at={base.literal(future)}::timestamptz "
                "where name='write_signing_key_previous'")

    kept = JournalEntry(user_id=base.USER_A, text="Fictional recovery entry.")
    other = JournalEntry(user_id=base.USER_B, text="Other account fictional entry.")
    erased = JournalEntry(user_id=base.USER_A, text="Fictional entry deleted before the backup.")
    for entry in (kept, other, erased):
        save = base.captured(lambda repo, entry=entry: repo.save_journal_entry(entry))
        sql(SOURCE, f"{base.as_user(entry.user_id)} select {save};")
    deletion = base.captured(lambda repo: repo.delete_journal_entry(base.USER_A, erased.id))
    sql(SOURCE, f"{base.as_user(base.USER_A)} select {deletion};")

    archive = run(["pg_dump", "-U", "postgres", "-Fc", "-d", SOURCE])
    run(["pg_restore", "-U", "postgres", "--exit-on-error", "-d", RESTORED], archive)
    expect(RESTORED, "(select count(*) from public.journal_entries)=2", "both retained records restored")
    expect(RESTORED, f"not exists(select 1 from public.journal_entries where id='{erased.id}')",
           "pre-backup erased content stays absent")
    expect(RESTORED, f"exists(select 1 from private.deleted_object_ids where object_id='{erased.id}')",
           "deletion marker restored")
    visible = sql(RESTORED, f"{base.as_user(base.USER_A)} select id from public.journal_entries;")
    if str(kept.id) not in visible or str(other.id) in visible:
        raise AssertionError("owner isolation after restore")
    assertions.append("owner isolation after restore")
    replay = base.captured(lambda repo: repo.save_journal_entry(erased))
    try:
        sql(RESTORED, f"{base.as_user(base.USER_A)} select {replay};")
    except RuntimeError as error:
        if "Creation request ID was deleted" not in str(error):
            raise
    else:
        raise AssertionError("deleted ID replay accepted after restore")
    assertions.append("deleted ID replay rejected after restore")
    new = JournalEntry(id=uuid4(), user_id=base.USER_A, text="Fictional post-restore write.")
    save_new = base.captured(lambda repo: repo.save_journal_entry(new))
    sql(RESTORED, f"{base.as_user(base.USER_A)} select {save_new};")
    expect(RESTORED, f"exists(select 1 from public.journal_entries where id='{new.id}')",
           "new signed writes work after restore")
    expect(RESTORED, valid(current_signature), "current signing key restored")
    expect(RESTORED, valid(previous_signature), "bounded verification overlap restored")
    report = {"status": "PASS", "scope": "local fictional PostgreSQL only", "assertions": assertions,
              "archive_sha256": hashlib.sha256(archive).hexdigest(), "archive_bytes": len(archive),
              "hosted_backup_verified": False, "hosted_pitr_verified": False,
              "post_backup_deletion_reconciliation_verified": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
