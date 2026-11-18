"""The onboarding auth stand-in accepts only its own tokens and only anonymous sign-ups."""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("onboarding_auth", ROOT / "scripts" / "onboarding_auth.py")
assert _spec and _spec.loader
auth = importlib.util.module_from_spec(_spec)
sys.modules["onboarding_auth"] = auth
_spec.loader.exec_module(auth)

SECRET = "test-only-secret-0123456789abcdef0123456789"


def test_tokens_round_trip_and_reject_tampering_and_expiry() -> None:
    token = auth.mint(SECRET, {"sub": "x", "role": "authenticated", "exp": int(time.time()) + 60})
    assert auth.verify(SECRET, token)["sub"] == "x"
    assert auth.verify("another-secret-0123456789abcdef0123456789", token) is None
    assert auth.verify(SECRET, token[:-2] + "AA") is None
    assert auth.verify(SECRET, "not-a-token") is None
    expired = auth.mint(SECRET, {"sub": "x", "role": "authenticated", "exp": int(time.time()) - 1})
    assert auth.verify(SECRET, expired) is None
    assert auth.verify(SECRET, auth.anon_key(SECRET))["role"] == "anon"


@pytest.fixture()
def server(monkeypatch):
    inserted: list[str] = []
    monkeypatch.setattr(auth, "_psql", lambda url, sql: inserted.append(sql))
    handler = auth.make_handler(SECRET, "postgresql://unused", "http://127.0.0.1:9")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", inserted
    httpd.shutdown()


def call(url: str, method: str = "GET", body: dict | None = None, headers: dict | None = None):
    request = urllib.request.Request(
        url, method=method, data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"{}")


def test_anonymous_signup_creates_an_anonymous_user_that_user_lookup_accepts(server) -> None:
    base, inserted = server
    status, body = call(f"{base}/auth/v1/signup", "POST", {"data": {}}, {"apikey": auth.anon_key(SECRET)})
    assert status == 200 and body["user"]["is_anonymous"] is True
    assert len(inserted) == 1 and "is_anonymous" in inserted[0] and body["user"]["id"] in inserted[0]
    status, user = call(f"{base}/auth/v1/user", headers={"Authorization": f"Bearer {body['access_token']}"})
    assert status == 200 and user["id"] == body["user"]["id"] and user["is_anonymous"] is True


def test_signup_refuses_a_foreign_key_and_email_sign_ups(server) -> None:
    base, inserted = server
    foreign = auth.anon_key("another-secret-0123456789abcdef0123456789")
    assert call(f"{base}/auth/v1/signup", "POST", {}, {"apikey": foreign})[0] == 401
    email = {"email": "someone@example.test", "password": "x"}
    assert call(f"{base}/auth/v1/signup", "POST", email, {"apikey": auth.anon_key(SECRET)})[0] == 422
    assert inserted == []
    anon_token = {"Authorization": f"Bearer {auth.anon_key(SECRET)}"}
    assert call(f"{base}/auth/v1/user", headers=anon_token)[0] == 401
    assert call(f"{base}/admin")[0] == 404
