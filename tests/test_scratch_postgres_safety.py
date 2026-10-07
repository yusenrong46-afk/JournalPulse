"""Destructive verification helpers must reject hosted routing before running psql."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(monkeypatch: pytest.MonkeyPatch, name: str):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    for variable in ("PGHOST", "PGHOSTADDR", "PGSERVICE"):
        monkeypatch.delenv(variable, raising=False)
    return importlib.import_module(name)


@pytest.mark.parametrize("module_name,reset_name", [
    ("verify_postgres_schema", "reset_database"),
    ("integration_stack", "prepare_database"),
])
def test_hosted_dsn_is_rejected_before_any_subprocess(monkeypatch, module_name, reset_name):
    module = load_script(monkeypatch, module_name)
    monkeypatch.setenv("JOURNALPULSE_PG_DSN", "postgresql://user:private-value@hosted.example/postgres")
    calls = []
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(SystemExit, match="restricted to local") as error:
        getattr(module, reset_name)()
    assert "private-value" not in str(error.value)
    assert calls == []


@pytest.mark.parametrize("configured", [
    None,
    "postgresql://postgres@127.0.0.1:55432/postgres",
    "postgresql://postgres@localhost:5432/postgres",
    "postgresql://postgres@[::1]:5432/postgres",
    "postgresql:///postgres?host=/var/run/postgresql",
    "postgresql://%2Fvar%2Frun%2Fpostgresql/postgres",
])
def test_local_peer_loopback_and_unix_connections_remain_available(monkeypatch, configured):
    module = load_script(monkeypatch, "scratch_postgres")
    module.require_local_postgres_dsn(configured)


@pytest.mark.parametrize("configured,env", [
    ("postgresql://localhost/postgres?host=hosted.example", {}),
    ("postgresql://localhost/postgres?hostaddr=192.0.2.1", {}),
    ("postgresql://localhost/postgres?host=localhost,hosted.example", {}),
    ("postgresql://localhost/postgres?service=hosted", {}),
    (None, {"PGHOST": "hosted.example"}),
    ("postgresql://localhost/postgres", {"PGHOSTADDR": "192.0.2.1"}),
    (None, {"PGSERVICE": "hosted"}),
])
def test_libpq_host_overrides_cannot_bypass_the_guard(monkeypatch, configured, env):
    module = load_script(monkeypatch, "scratch_postgres")
    for variable, value in env.items():
        monkeypatch.setenv(variable, value)
    with pytest.raises(SystemExit, match="restricted to local"):
        module.require_local_postgres_dsn(configured)


def test_local_reset_still_targets_only_the_named_scratch_database(monkeypatch):
    module = load_script(monkeypatch, "verify_postgres_schema")
    monkeypatch.setenv("JOURNALPULSE_PG_DSN", "postgresql://postgres@127.0.0.1:55432/postgres")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs["input"]))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(module.subprocess, "run", run)
    module.reset_database()
    assert len(calls) == 1
    assert "drop database if exists jp_verify" in calls[0][1]
    assert calls[0][0][1] == "postgresql://postgres@127.0.0.1:55432/postgres"


@pytest.mark.parametrize('module_name', ['verify_postgres_schema', 'integration_stack'])
def test_database_query_override_cannot_change_scratch_target(monkeypatch, module_name):
    module = load_script(monkeypatch, module_name)
    monkeypatch.setenv('JOURNALPULSE_PG_DSN', 'postgresql://localhost/postgres?dbname=unrelated_local')
    builder = module.dsn_for if module_name == 'verify_postgres_schema' else module.dsn
    with pytest.raises(SystemExit, match='database'):
        builder('jp_verify' if module_name == 'verify_postgres_schema' else 'jp_integration')


@pytest.mark.parametrize('module_name', ['verify_postgres_schema', 'integration_stack'])
def test_rewritten_unix_socket_uri_keeps_libpq_scheme_delimiter(monkeypatch, module_name):
    module = load_script(monkeypatch, module_name)
    monkeypatch.setenv('JOURNALPULSE_PG_DSN', 'postgresql:///postgres?host=/var/run/postgresql')
    builder = module.dsn_for if module_name == 'verify_postgres_schema' else module.dsn
    uri = builder('jp_verify')
    assert uri == 'postgresql:///jp_verify?host=/var/run/postgresql'


def test_integration_never_changes_the_shared_authenticator(monkeypatch):
    module = load_script(monkeypatch, 'integration_stack')
    monkeypatch.setenv('JOURNALPULSE_PG_DSN', 'postgresql://localhost/postgres')
    statements = []
    monkeypatch.setattr(module, 'psql', lambda database, sql: statements.append((database, sql)) or '')
    module.prepare_database()
    setup_sql = '\n'.join(sql for _, sql in statements)
    assert 'alter role authenticator ' not in setup_sql.lower()
    assert 'create role authenticator ' not in setup_sql.lower()
    assert module.AUTHENTICATOR_ROLE.startswith('jp_test_auth_')
    assert module.AUTHENTICATOR_PASSWORD != 'integration-authenticator'
    assert module.AUTHENTICATOR_ROLE in setup_sql


def test_postgrest_connection_uses_only_its_own_credentials(monkeypatch):
    module = load_script(monkeypatch, 'integration_stack')
    monkeypatch.setenv('JOURNALPULSE_PG_DSN', 'postgresql://admin@localhost/postgres?user=admin&password=admin-fixture')
    uri = module.dsn('jp_integration', user='jp_test_auth_fixture', password='synthetic-isolated')
    from urllib.parse import parse_qs, urlsplit
    parsed = urlsplit(uri)
    assert parsed.username == 'jp_test_auth_fixture'
    assert 'user' not in parse_qs(parsed.query)
    assert 'password' not in parse_qs(parsed.query)
