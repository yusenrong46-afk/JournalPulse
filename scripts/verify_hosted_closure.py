"""Hosted closure smoke: create only new fictional users, never accept existing IDs.

Run only after real recovery, migration, and a verified staged deployment. Keys are
hidden terminal inputs. All Admin requests are pinned to this Supabase project.
No model, search, invitation or email-link endpoint is called.
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import re
import secrets
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit
from uuid import UUID, uuid4

import httpx

SUPABASE = "https://cggixttnyprrbaotmjxj.supabase.co"
PROJECT = "prj_EOSYVlo1KjkdnOAT8Smm72LdgFrV"


class SmokeStopped(Exception):
    pass


def app_origin(value: str) -> str:
    parts = urlsplit(value)
    if (parts.scheme != "https" or parts.username or parts.password or parts.port
            or parts.query or parts.fragment or parts.path not in {"", "/"}
            or not re.fullmatch(r"journalpulse(?:-[a-z0-9-]+)?\.vercel\.app", parts.hostname or "")):
        raise SmokeStopped("Use the verified JournalPulse deployment origin without a path or credentials.")
    return value.rstrip("/")


def verified_recovery() -> bool:
    folder = Path.home() / ".local/share/journalpulse-backups"
    for path in folder.glob("backup-*.verified.json"):
        if path.is_symlink():
            continue
        try:
            report = json.loads(path.read_text())
            encrypted = path.with_name(path.name.removesuffix(".verified.json") + ".gpg")
            if (report.get("status") == "PASS" and report.get("table_checksums_match") is True
                    and report.get("fixture_deletion_reconciliation") is True
                    and report.get("container_removed") is True and encrypted.is_file()
                    and not encrypted.is_symlink() and encrypted.stat().st_mode & 0o077 == 0):
                return True
        except (OSError, ValueError):
            continue
    return False


@dataclass
class Fixture:
    id: str
    email: str
    run_id: str
    password: str = field(repr=False)
    token: str = field(default="", repr=False)
    refresh_token: str = field(default="", repr=False)


def fixture_identity(data: dict, fixture: Fixture) -> bool:
    """Never clean up an identity selected only by a name, email, or supplied ID."""
    return (data.get("id") == fixture.id and data.get("email") == fixture.email
            and data.get("app_metadata", {}).get("journalpulse_closure_test") == fixture.run_id)


class HostedSmoke:
    def __init__(self, client: httpx.Client, app: str, admin_key: str, anon_key: str):
        self.client, self.app = client, app_origin(app)
        self.admin = {"apikey": admin_key, "Authorization": "Bearer " + admin_key}
        self.public = {"apikey": anon_key}
        self.run_id = str(uuid4())
        self.fixtures: list[Fixture] = []
        self.uncertain_creations: list[str] = []
        self.checks: list[str] = []
        self.phase = "preflight"

    def call(self, method: str, path: str, *, admin: bool = False, token: str = "", **kwargs):
        base = SUPABASE if admin or path.startswith("/auth/") or path.startswith("/rest/") else self.app
        headers = dict(self.admin if admin else self.public if base == SUPABASE else {})
        if token:
            headers["Authorization"] = "Bearer " + token
        # The privileged key can never be attached to the application URL.
        return self.client.request(method, base + path, headers=headers, **kwargs)

    def expect(self, condition: bool, label: str) -> None:
        if not condition:
            raise SmokeStopped(label)
        self.checks.append(label)

    def create(self, label: str) -> Fixture:
        self.phase = "create_" + label
        email = f"jp-closure-{self.run_id}-{label}@example.test"
        password = secrets.token_urlsafe(40)
        self.uncertain_creations.append(email)
        response = self.call("POST", "/auth/v1/admin/users", admin=True, json={
            "email": email, "password": password, "email_confirm": True,
            "app_metadata": {"journalpulse_closure_test": self.run_id},
        })
        if response.status_code not in {200, 201}:
            if 400 <= response.status_code < 500:
                self.uncertain_creations.remove(email)
            raise SmokeStopped("Fictional account creation failed; no automatic retry.")
        data = response.json()
        owner = str(UUID(data["id"]))
        fixture = Fixture(owner, email, self.run_id, password)
        # Retain only IDs returned by this run's create operation; never list real users.
        self.fixtures.append(fixture)
        if not fixture_identity(data, fixture):
            raise SmokeStopped("Created identity metadata mismatch; manual fixture reconciliation required.")
        self.uncertain_creations.remove(email)
        return fixture

    def signin(self, fixture: Fixture) -> None:
        response = self.call("POST", "/auth/v1/token?grant_type=password", json={
            "email": fixture.email, "password": fixture.password,
        })
        if response.status_code != 200 or response.json().get("user", {}).get("id") != fixture.id:
            raise SmokeStopped("Fictional password sign-in failed.")
        fixture.token = response.json()["access_token"]
        fixture.refresh_token = response.json()["refresh_token"]

    def refresh(self, fixture: Fixture) -> None:
        response = self.call("POST", "/auth/v1/token?grant_type=refresh_token",
                             json={"refresh_token": fixture.refresh_token})
        if response.status_code != 200 or response.json().get("user", {}).get("id") != fixture.id:
            raise SmokeStopped("Fictional token refresh failed; stale-signin behavior remains unverified.")
        fixture.token = response.json()["access_token"]
        fixture.refresh_token = response.json()["refresh_token"]

    @staticmethod
    def receipt() -> dict:
        return {"confirmation": "delete my account", "request_id": str(uuid4()),
                "receipt_secret": secrets.token_hex(32)}

    def save(self, fixture: Fixture) -> None:
        revision = self.call("GET", "/v1/account/data-revision", token=fixture.token)
        if revision.status_code != 200:
            raise SmokeStopped("Fictional account write preflight failed.")
        value = revision.json()["revision"]
        response = self.client.post(self.app + "/v1/journal/entries", headers={
            "Authorization": "Bearer " + fixture.token, "X-JournalPulse-Data-Revision": str(value),
        }, json={"text": "Fictional account-closure verification entry.", "client_request_id": str(uuid4())})
        self.expect(response.status_code == 201, "fictional_write")

    def run(self) -> None:
        readiness = self.call("GET", "/ready")
        self.expect(readiness.status_code == 200, "staged_readiness")
        status = self.call("GET", "/v1/system/status")
        self.expect(status.status_code == 200 and status.json().get("account_closure") == "configured",
                    "closure_configured")
        anonymous = self.call("DELETE", "/v1/account", json=self.receipt())
        self.expect(anonymous.status_code == 401, "anonymous_cannot_start")
        first, second = self.create("a"), self.create("b")
        self.signin(first)
        self.signin(second)
        self.save(first)
        self.save(second)
        body = self.receipt()
        injected = self.call("DELETE", "/v1/account", token=first.token,
                             json={**body, "user_id": second.id})
        self.expect(injected.status_code == 422, "client_owner_field_rejected")
        self.phase = "wait_for_stale_signin"
        print("Waiting just over five minutes to test a genuinely stale sign-in; no model calls.", flush=True)
        deadline = time.monotonic() + 310
        while time.monotonic() < deadline:
            time.sleep(max(0, min(30, deadline - time.monotonic())))
        self.refresh(first)  # A newly minted token must not count as a fresh sign-in.
        stale = self.call("DELETE", "/v1/account", token=first.token, json=body)
        self.expect(stale.status_code == 403, "stale_signin_rejected")
        self.save(first)  # Rejection must not leave this account locked pending deletion.
        self.phase = "fresh_closure"
        self.signin(first)
        self.signin(second)
        result = self.call("DELETE", "/v1/account", token=first.token, json=body)
        self.expect(result.status_code == 204, "fresh_closure_confirmed")
        absent = self.call("GET", f"/auth/v1/admin/users/{first.id}", admin=True)
        self.expect(absent.status_code == 404, "auth_identity_absent")
        crossed = self.call("DELETE", "/v1/account", token=second.token, json=body)
        self.expect(crossed.status_code == 403, "other_account_receipt_rejected")
        repeated = self.call("DELETE", "/v1/account", token=first.token, json=body)
        self.expect(repeated.status_code == 204, "same_receipt_retry_after_identity_deletion")
        proof = hashlib.sha256(body["receipt_secret"].encode()).hexdigest()
        resolved = self.call("POST", "/rest/v1/rpc/jp_resolve_account_closure", admin=True,
                             json={"operation_id": body["request_id"], "proof_hash": proof})
        self.expect(resolved.status_code == 200 and resolved.json().get("state") == "complete"
                    and resolved.json().get("user_id") == first.id, "verified_completion_record")
        self.expect(self.call("GET", "/v1/journal/entries", token=first.token).status_code == 401,
                    "deleted_session_cannot_read")
        untouched = self.call("GET", "/v1/journal/entries", token=second.token)
        self.expect(untouched.status_code == 200 and len(untouched.json()["items"]) == 1,
                    "other_account_data_untouched")
        self.phase = "complete"

    def cleanup(self) -> list[str]:
        unresolved = []
        for fixture in self.fixtures:
            try:
                path = f"/auth/v1/admin/users/{fixture.id}"
                response = self.call("GET", path, admin=True)
                if response.status_code == 404:
                    continue
                if response.status_code != 200 or not fixture_identity(response.json(), fixture):
                    unresolved.append(fixture.id)
                    continue
                self.call("DELETE", path, admin=True, json={"should_soft_delete": False})
                if self.call("GET", path, admin=True).status_code != 404:
                    unresolved.append(fixture.id)
            except (httpx.HTTPError, ValueError, TypeError, AttributeError):
                unresolved.append(fixture.id)
        return unresolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not sys.stdin.isatty():
        raise SystemExit("Use an interactive terminal for hidden credential entry.")
    record = json.loads(args.deployment_record.read_text())
    if record.get("project_id") != PROJECT or not re.fullmatch(r"[a-f0-9]{40}", record.get("commit", "")):
        raise SystemExit("A verified JournalPulse deployment record with exact commit is required.")
    origin = app_origin(record["url"])
    if not verified_recovery():
        raise SystemExit("Complete the encrypted live backup/local restore gate before hosted mutations.")
    if args.output.exists():
        raise SystemExit("Use a new evidence path; do not overwrite a previous hosted attempt.")
    admin = getpass.getpass("Supabase server-only service_role key (hidden): ").strip()
    anon = getpass.getpass("Supabase public anon key (hidden): ").strip()
    if not admin or not anon:
        raise SystemExit("Missing credential; no test started.")
    with httpx.Client(timeout=90, follow_redirects=False) as client:
        smoke = HostedSmoke(client, origin, admin, anon)
        passed = False
        failure = None
        try:
            smoke.run()
            passed = True
        except Exception as error:
            # Raw transport and response diagnostics can contain credentials.
            failure = str(error) if isinstance(error, SmokeStopped) else type(error).__name__
            print("Hosted check stopped; sanitized phase and fixture IDs will be recorded.")
        finally:
            cleanup_needed = smoke.cleanup()
            report = {"status": "PASS" if passed and not cleanup_needed and not smoke.uncertain_creations
                      else "STOPPED",
                      "project_id": PROJECT, "declared_commit": record["commit"], "url": origin,
                      "phase": smoke.phase, "checks": smoke.checks, "run_id": smoke.run_id,
                      "failure": failure,
                      "created_fixture_ids": [item.id for item in smoke.fixtures],
                      "fixture_cleanup_needed": cleanup_needed, "provider_calls": 0,
                      "uncertain_fixture_creations": smoke.uncertain_creations,
                      "fault_injection": "LOCAL_TRANSPORT_TESTS_ONLY",
                      "concurrent_write_orderings": "LOCAL_POSTGRESQL_TESTS_ONLY"}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"status": report["status"], "phase": smoke.phase,
                              "cleanup_needed_count": len(cleanup_needed)}))
        if not passed or cleanup_needed or smoke.uncertain_creations:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
