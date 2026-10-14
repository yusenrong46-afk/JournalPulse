# /// script
# requires-python = ">=3.12"
# dependencies = ["psycopg[binary]==3.3.6"]
# ///
"""Encrypted logical backup of the approved JournalPulse database; no secret argv.

Run with `uv run --script scripts/live_backup.py create` in your own terminal.
Database credentials and encryption passphrase are entered without echo. Only
ciphertext and non-content verification status are written outside Git.
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote, urlsplit
from uuid import uuid4

PROJECT = "cggixttnyprrbaotmjxj"
IMAGE = "supabase/postgres@sha256:c282d393ae56fd165b7ada2ddbac55e582a601b86f3840aa2bbc112e43d3a570"
BACKUPS = Path.home() / ".local/share/journalpulse-backups"
SCHEMAS = ("public", "private", "auth", "storage", "extensions")
MAX_BYTES = 128 * 1024 * 1024


class BackupError(Exception):
    """Messages are fixed categories and never carry source data or credentials."""


def source_parameters(uri: str) -> dict[str, str]:
    try:
        parts = urlsplit(uri)
        user, password = unquote(parts.username or ""), unquote(parts.password or "")
        host = parts.hostname or ""
        direct = host == f"db.{PROJECT}.supabase.co" and user == "postgres"
        pooler = host.endswith(".pooler.supabase.com") and user == f"postgres.{PROJECT}"
        if (parts.scheme not in {"postgres", "postgresql"} or not (direct or pooler)
                or parts.port not in {None, 5432} or parts.path != "/postgres"
                or parts.query or parts.fragment or not password):
            raise ValueError
    except ValueError:
        raise BackupError(
            "Use this project's direct or SESSION-pooler URI, port 5432, without query options."
        ) from None
    return {"host": host, "port": "5432", "dbname": "postgres", "user": user,
            "password": password, "sslmode": "verify-full", "sslrootcert": "system",
            "connect_timeout": "15", "application_name": "journalpulse-private-backup"}


def private_directory(path: Path) -> None:
    if path.is_symlink():
        raise BackupError("Refusing a symlinked private directory.")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)


def run_private(command: list[str], *, data: bytes | None = None,
                env: dict[str, str] | None = None, timeout: int = 300) -> bytes:
    try:
        result = subprocess.run(command, input=data, capture_output=True,
                                env=env, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        raise BackupError("Private operation could not finish; diagnostics were suppressed.") from None
    if result.returncode and b"Creation request ID was deleted" in result.stderr:
        raise BackupError("Retired creation ID rejected.")
    if result.returncode or len(result.stdout) > MAX_BYTES:
        raise BackupError("A private operation failed or exceeded the size limit; contents were not logged.")
    return result.stdout


def crypt(data: bytes, passphrase: str, home: Path, *, decrypt: bool = False) -> bytes:
    if not 16 <= len(passphrase) <= 1024 or "\n" in passphrase or "\x00" in passphrase:
        raise BackupError("Passphrase must be 16-1024 characters without newline or NUL.")
    private_directory(home)
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, passphrase.encode() + b"\n")
        os.close(write_fd)
        write_fd = -1
        command = ["gpg", "--homedir", str(home), "--batch", "--no-symkey-cache",
                   "--pinentry-mode", "loopback", "--passphrase-fd", str(read_fd), "--output", "-"]
        command += ["--decrypt"] if decrypt else ["--symmetric", "--cipher-algo", "AES256"]
        result = subprocess.run(command, input=data, capture_output=True, pass_fds=(read_fd,), check=False)
        if result.returncode or len(result.stdout) > MAX_BYTES:
            raise BackupError("Encryption/decryption failed; check the passphrase and GPG setup privately.")
        return result.stdout
    finally:
        os.close(read_fd)
        if write_fd >= 0:
            os.close(write_fd)


def pack(dump: bytes, manifest: dict) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as archive:
        for name, data in (("database.dump", dump), ("manifest.json", json.dumps(manifest).encode())):
            item = tarfile.TarInfo(name)
            item.size, item.mode = len(data), 0o600
            archive.addfile(item, io.BytesIO(data))
    if buf.tell() > MAX_BYTES:
        raise BackupError("Backup exceeds this tool's reviewed memory limit.")
    return buf.getvalue()


def unpack(data: bytes) -> tuple[bytes, dict]:
    if len(data) > MAX_BYTES:
        raise BackupError("Archive exceeds the size limit.")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
        members = archive.getmembers()
        if ({m.name for m in members} != {"database.dump", "manifest.json"} or len(members) != 2
                or any(not m.isfile() or m.size > MAX_BYTES for m in members)):
            raise BackupError("Unexpected archive members.")
        dump_file, manifest_file = archive.extractfile("database.dump"), archive.extractfile("manifest.json")
        assert dump_file is not None and manifest_file is not None
        dump, manifest = dump_file.read(), json.loads(manifest_file.read())
    if hashlib.sha256(dump).hexdigest() != manifest.get("dump_sha256"):
        raise BackupError("Backup payload checksum mismatch.")
    return dump, manifest


def fingerprints(connection) -> dict:
    from psycopg import sql

    tables = connection.execute("select schemaname,tablename from pg_tables "
                                "where schemaname=any(%s) order by 1,2", (list(SCHEMAS),)).fetchall()
    results = {}
    for schema, table in tables:
        # Only per-row digests cross this query boundary, never journal contents.
        digest, count = hashlib.sha256(), 0
        query = sql.SQL("select md5(row_to_json(t)::text) from {} t order by 1").format(
            sql.Identifier(schema, table))
        with connection.cursor(name="jp_fingerprint_" + uuid4().hex) as cursor:
            cursor.execute(query)
            for (row_hash,) in cursor:
                count += 1
                digest.update(row_hash.encode())
        results[f"{schema}.{table}"] = {"count": count, "sha256": digest.hexdigest()}
    return results


def create_backup(uri: str, passphrase: str) -> Path:
    import psycopg

    # libpq accepts hostaddr/service/options overrides from the environment.
    # Refuse them rather than letting a hidden setting redirect a private export.
    if any(key.startswith("PG") for key in os.environ):
        raise BackupError("Unset PostgreSQL environment overrides before starting the private backup.")
    params = source_parameters(uri)
    private_directory(BACKUPS)
    with psycopg.connect(**params) as connection:
        connection.execute("set transaction isolation level repeatable read read only")
        connection.execute("set local timezone='UTC'")
        version = int(connection.execute("show server_version_num").fetchone()[0])
        if version // 10000 != 17:
            raise BackupError("Source PostgreSQL major changed; review the pinned client before export.")
        size = connection.execute("select pg_database_size(current_database())").fetchone()[0]
        if size > MAX_BYTES:
            raise BackupError("Database exceeds the reviewed export size; stop for a larger-backup plan.")
        for table in ("storage.objects", "vault.secrets"):
            if connection.execute(f"select count(*) from {table}").fetchone()[0]:
                raise BackupError("Storage objects or Vault secrets need an additional recovery procedure.")
        extensions = connection.execute(
            "select e.extname,n.nspname,e.extversion from pg_extension e "
            "join pg_namespace n on n.oid=e.extnamespace where n.nspname=any(%s)",
            (list(SCHEMAS),),
        ).fetchall()
        schema_owners = connection.execute(
            "select nspname,pg_get_userbyid(nspowner) from pg_namespace where nspname=any(%s)",
            (list(SCHEMAS),),
        ).fetchall()
        snapshot = connection.execute("select pg_export_snapshot()").fetchone()[0]
        if not re.fullmatch(r"[0-9A-Fa-f-]+", snapshot):
            raise BackupError("Unexpected snapshot identifier.")
        hashes = fingerprints(connection)
        roles = connection.execute("select rolname,rolbypassrls from pg_roles "
                                   "where rolname !~ '^pg_' and rolname<>'postgres'").fetchall()
        env = {k: v for k, v in os.environ.items() if not k.startswith("PG")}
        for key, value in params.items():
            name = {"dbname": "PGDATABASE", "user": "PGUSER", "password": "PGPASSWORD",
                    "connect_timeout": "PGCONNECT_TIMEOUT", "application_name": "PGAPPNAME"}.get(
                        key, "PG" + key.upper())
            env[name] = str(value)
        command = ["docker", "run", "--rm", "--entrypoint", "pg_dump"]
        for key in sorted(k for k in env if k.startswith("PG")):
            command += ["--env", key]
        command += [IMAGE, "--format=custom", f"--snapshot={snapshot}"]
        command += [f"--schema={schema}" for schema in SCHEMAS]
        dump = run_private(command, env=env)
        manifest = {"format": 2, "project": PROJECT, "postgres_version": version,
                    "created_at": datetime.now(UTC).isoformat(), "schemas": SCHEMAS,
                    "roles": roles, "extensions": extensions, "schema_owners": schema_owners,
                    "tables": hashes, "dump_sha256": hashlib.sha256(dump).hexdigest(),
                    "excluded": ["Storage file bytes", "Vault secrets", "cron jobs", "replication state"]}
        # Keep the source snapshot alive through the entire export.
        payload = pack(dump, manifest)
    encrypted = crypt(payload, passphrase, BACKUPS / ".gnupg")
    path = BACKUPS / (datetime.now(UTC).strftime("backup-%Y%m%dT%H%M%SZ-") + uuid4().hex[:8] + ".gpg")
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as handle:
        handle.write(encrypted)
    # Verify encryption and archive integrity before retaining this as a backup.
    unpack(crypt(path.read_bytes(), passphrase, BACKUPS / ".gnupg", decrypt=True))
    return path


def ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def verify_restore(dump: bytes, manifest: dict) -> dict:
    """All restored production records stay in tmpfs with no external network."""
    if manifest.get("project") != PROJECT or manifest.get("postgres_version", 0) // 10000 != 17:
        raise BackupError("Archive project or database version does not match this recovery tool.")
    name = "jp-private-recovery-" + uuid4().hex[:12]
    run_private(["docker", "run", "--detach", "--name", name, "--network", "none",
                 "--tmpfs", "/var/lib/postgresql/data:rw,nosuid,size=512m",
                 "--env", "PGDATA=/var/lib/postgresql/data", "--env", "POSTGRES_HOST_AUTH_METHOD=trust",
                 "--env", "POSTGRES_PASSWORD=fictional-local-only", IMAGE,
                 "postgres", "-D", "/var/lib/postgresql/data", "-c", "shared_preload_libraries=",
                 "-c", "listen_addresses=", "-c", "log_statement=none",
                 "-c", "log_min_error_statement=panic"])
    try:
        info = json.loads(run_private(["docker", "inspect", name]))[0]
        if (info["HostConfig"]["NetworkMode"] != "none" or info["HostConfig"].get("PortBindings")
                or any(mount["Type"] != "tmpfs" for mount in info["Mounts"])):
            raise BackupError("Recovery isolation verification failed.")
        for _ in range(30):
            ready = subprocess.run(["docker", "exec", name, "pg_isready", "-U", "supabase_admin"],
                                   capture_output=True, check=False)
            final_server = subprocess.run(["docker", "exec", name, "cat", "/proc/1/comm"],
                                          capture_output=True, check=False)
            if ready.returncode == 0 and final_server.stdout.strip() in {b"postgres", b".postgres-wrapp"}:
                break
            time.sleep(1)
        else:
            raise BackupError("Isolated recovery database did not start.")

        def sql(query: str, database: str = "jp_recovered") -> str:
            return run_private(["docker", "exec", "-i", name, "psql", "-U", "supabase_admin", "-d", database,
                                "-X", "-qAt", "-v", "ON_ERROR_STOP=1"],
                               data=query.encode()).decode().strip()

        def new_database(database: str, archive: bytes) -> None:
            run_private(["docker", "exec", name, "createdb", "-U", "supabase_admin",
                         "-T", "template0", database])
            # Fresh template0 has an empty public schema; the archive creates its own.
            sql("drop schema public;", database)
            for schema, owner in manifest["schema_owners"]:
                if schema not in SCHEMAS:
                    raise BackupError("Unexpected recovery schema.")
                sql("create schema " + ident(schema) + " authorization " + ident(owner) + ";", database)
            for extension, schema, version in manifest["extensions"]:
                if schema not in SCHEMAS:
                    raise BackupError("Unexpected extension schema.")
                sql("create extension if not exists " + ident(extension) + " with schema "
                    + ident(schema) + " version " + literal(version) + ";", database)
            toc = run_private(["docker", "exec", "-i", name, "pg_restore", "--list"], data=archive)
            filtered = b"\n".join(line for line in toc.splitlines() if b" SCHEMA - " not in line)
            run_private(["docker", "exec", "-i", name, "sh", "-c",
                         "cat > /var/lib/postgresql/data/jp-restore.list"], data=filtered)
            run_private(["docker", "exec", "-i", name, "pg_restore", "-U", "supabase_admin", "-d", database,
                         "--exit-on-error", "--use-list", "/var/lib/postgresql/data/jp-restore.list"],
                        data=archive)

        for role, bypass in manifest["roles"]:
            # Preserve named ACL targets without importing any password or enabling logins.
            sql("do $role$ begin if not exists(select 1 from pg_roles where rolname=" + literal(role)
                + ") then create role " + ident(role) + " nologin "
                + ("bypassrls" if bypass else "nobypassrls") + "; end if; end $role$;", "postgres")
        new_database("jp_recovered", dump)
        for table, expected in manifest["tables"].items():
            schema, name_part = table.split(".", 1)
            if schema not in SCHEMAS:
                raise BackupError("Unexpected recovery schema.")
            rows = sql("set timezone='UTC'; select md5(row_to_json(t)::text) from "
                       + ident(schema) + "." + ident(name_part) + " t order by 1;").splitlines()
            digest = hashlib.sha256("".join(rows).encode()).hexdigest()
            if len(rows) != expected["count"] or digest != expected["sha256"]:
                raise BackupError("Restored table verification failed; data remains isolated.")
        for role in ("anon", "authenticated"):
            if sql(f"select has_table_privilege('{role}','private.server_secrets','SELECT')") != "f":
                raise BackupError("Restored private-secret permissions are unsafe.")

        # These accounts and records are generated only inside the isolated copy.
        owner_a, owner_b, entry_id = str(uuid4()), str(uuid4()), str(uuid4())
        sql(f"insert into auth.users(id,email) values ('{owner_a}','recovery-a@example.test'),"
            f"('{owner_b}','recovery-b@example.test');")

        def signed_call(purpose: str, fields: dict, database: str = "jp_recovered") -> str:
            envelope = {"purpose": purpose, "user_id": owner_a,
                        "issued_at": datetime.now(UTC).isoformat(), "expected_erasure_revision": 0, **fields}
            payload = json.dumps(envelope, separators=(",", ":"))
            # The production signing key never leaves the restored container.
            return sql("begin; set local search_path=public,private,extensions; "
                       "do $test$ declare p text := " + literal(payload)
                       + "; k text; begin select value into k from private.server_secrets "
                       "where name='write_signing_key'; perform set_config('jp.test_payload',p,true); "
                       "perform set_config('jp.test_signature',encode(hmac(convert_to(p,'UTF8'),"
                       "convert_to(k,'UTF8'),'sha256'),'hex'),true); end $test$;"
                       f"set local role authenticated; set local request.jwt.claim.sub='{owner_a}';"
                       f"select public.jp_{purpose}(current_setting('jp.test_payload'),"
                       "current_setting('jp.test_signature')); commit;", database)

        entry = {"id": entry_id, "user_id": owner_a, "created_at": datetime.now(UTC).isoformat(),
                 "text": "Fictional recovery verification record."}
        signed_call("save_journal_entry", {"entry": entry})
        if sql(f"begin; set local role authenticated; set local request.jwt.claim.sub='{owner_b}';"
               f"select count(*) from public.journal_entries where id='{entry_id}'; commit;") != "0":
            raise BackupError("Cross-account read isolation failed in the recovered database.")
        # Snapshot the local fixture before deletion; no plaintext archive touches disk.
        before_delete = run_private(["docker", "exec", name, "pg_dump", "-U", "supabase_admin", "-Fc",
                                     "-d", "jp_recovered"])
        signed_call("delete_journal_entry", {"entry_id": entry_id})
        if sql(f"select count(*) from private.deleted_object_ids where object_id='{entry_id}'") != "1":
            raise BackupError("The post-backup deletion marker was not retained.")
        new_database("jp_reconciled", before_delete)
        # Replay this acknowledged deletion before allowing any access to an old restore.
        signed_call("delete_journal_entry", {"entry_id": entry_id}, "jp_reconciled")
        if sql(f"select count(*) from public.journal_entries where id='{entry_id}'", "jp_reconciled") != "0":
            raise BackupError("Post-backup deletion reconciliation failed.")
        try:
            signed_call("save_journal_entry", {"entry": entry}, "jp_reconciled")
        except BackupError as error:
            if str(error) != "Retired creation ID rejected.":
                raise
            # Validate this is specifically a retired-identity barrier, not an arbitrary failure.
            if sql(f"select count(*) from private.deleted_object_ids where object_id='{entry_id}'",
                   "jp_reconciled") != "1":
                raise
        else:
            raise BackupError("A deleted creation ID was accepted after recovery.")
        return {"status": "PASS", "scope": "logical export restored only locally",
                "table_checksums_match": True, "owner_isolation": True, "signed_write": True,
                "fixture_deletion_reconciliation": True, "container_removed": True,
                "hosted_auth_service_tested": False, "managed_backups_verified": False}
    finally:
        run_private(["docker", "rm", "--force", "--volumes", name], timeout=30)


def verify_backup(path: Path, passphrase: str) -> dict:
    if path.is_symlink() or path.parent.resolve() != BACKUPS.resolve() or path.suffix != ".gpg":
        raise BackupError("Only encrypted backups in the dedicated private folder can be restored.")
    dump, manifest = unpack(crypt(path.read_bytes(), passphrase, BACKUPS / ".gnupg", decrypt=True))
    report = verify_restore(dump, manifest)
    report_path = path.with_suffix(".verified.json")
    if report_path.is_symlink():
        raise BackupError("Refusing a symlinked verification report.")
    temporary = report_path.with_name(report_path.name + "." + uuid4().hex + ".tmp")
    with os.fdopen(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as handle:
        json.dump(report, handle, indent=2)
    temporary.replace(report_path)
    # Rotation follows successful verification. Never remove the sole known-good copy.
    verified = sorted(BACKUPS.glob("backup-*.verified.json"), reverse=True)
    for expired in verified[3:]:
        encrypted = expired.with_name(expired.name.removesuffix(".verified.json") + ".gpg")
        if not encrypted.is_symlink() and encrypted.is_file():
            encrypted.unlink()
            expired.unlink()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["create", "verify"])
    parser.add_argument("backup", type=Path, nargs="?", help="Existing encrypted backup for verify")
    args = parser.parse_args()
    try:
        if not sys.stdin.isatty():
            raise BackupError("Run this command in your own interactive terminal for hidden input.")
        if args.action == "create":
            if args.backup is not None:
                raise BackupError("The create action does not accept an existing backup path.")
            uri = getpass.getpass(
                "Supabase SESSION-pooler connection URI (hidden; do not reset the password): "
            )
            phrase = getpass.getpass("New backup encryption passphrase (hidden; keep it somewhere safe): ")
            if not 16 <= len(phrase) <= 1024 or phrase != getpass.getpass("Repeat encryption passphrase: "):
                raise BackupError("Passphrase must match and contain 16-1024 characters.")
            path = create_backup(uri, phrase)
            print(f"Encrypted backup created and encryption verified: {path}")
        else:
            if args.backup is None:
                raise BackupError("The verify action requires an encrypted backup path.")
            path = args.backup.expanduser().absolute()
            phrase = getpass.getpass("Existing backup encryption passphrase (hidden): ")
        report = verify_backup(path, phrase)
        print(json.dumps(report))
    except Exception as error:
        # Driver exception strings may contain hostnames, credentials or rejected SQL values.
        message = str(error) if isinstance(error, BackupError) else "Backup failed; diagnostics suppressed."
        print(message)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
