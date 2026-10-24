"""No real identities are used or deleted by these transport-level checks."""

import base64
import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from journalpulse.account_closure import AccountClosureRequest, AccountClosureService, require_recent_auth
from journalpulse.api import create_app
from journalpulse.auth import AuthContext
from test_auth_and_rls import USER_ID, configured

NOW = datetime(2026, 10, 14, 21, tzinfo=UTC)
SECRET = "ab" * 32


def token(*, owner=USER_ID, method="otp", stamp=None, **changes):
    body = {
        "sub": owner,
        "iss": "https://project.supabase.co/auth/v1",
        "role": "authenticated",
        "is_anonymous": False,
        "amr": [{"method": method, "timestamp": int(NOW.timestamp()) if stamp is None else stamp}],
        "iat": int(NOW.timestamp()),
        **changes,
    }
    payload = base64.urlsafe_b64encode(json.dumps(body).encode()).decode().rstrip("=")
    return "header." + payload + ".verified-by-fake-auth"


def settings(tmp_path):
    return replace(
        configured(tmp_path, supabase=True),
        supabase_service_role_key="fictional-server-only",
        write_signing_key="fictional-signing-key-at-least-32-characters",
        account_closure_enabled=True,
    )


def request():
    return AccountClosureRequest(confirmation="delete my account", request_id=uuid4(), receipt_secret=SECRET)


class ClosureTransport:
    def __init__(self, payload):
        self.payload = payload
        self.pending = False
        self.deleted = False
        self.complete = False
        self.fail_delete = False
        self.timeout_after_delete = False
        self.resolve_owner = USER_ID
        self.auth_owner = USER_ID
        self.auth_token = token()
        self.calls = []
        self.fresh_checks = 0

    def __call__(self, req):
        self.calls.append((req.method, req.url.path))
        path = req.url.path
        if path.endswith("jp_resolve_account_closure"):
            assert req.headers["apikey"] == "fictional-server-only"
            body = json.loads(req.content)
            valid = (
                body["operation_id"] == str(self.payload.request_id)
                and body["proof_hash"] == hashlib.sha256(SECRET.encode()).hexdigest()
            )
            return httpx.Response(
                200,
                json={
                    "user_id": self.resolve_owner,
                    "request_id": str(self.payload.request_id),
                    "state": "complete" if self.complete else "pending",
                }
                if self.pending and valid
                else None,
            )
        if path == "/auth/v1/user":
            self.fresh_checks += 1
            assert req.headers["apikey"] == "public-anon-key"
            assert req.headers["authorization"] == "Bearer " + self.auth_token
            return httpx.Response(401 if self.deleted else 200, json={"id": self.auth_owner})
        if path.endswith("jp_begin_account_closure"):
            assert req.headers["apikey"] == "public-anon-key"
            envelope = json.loads(json.loads(req.content)["payload"])
            assert envelope["user_id"] == self.auth_owner
            self.pending = True
            return httpx.Response(
                200,
                json={
                    "user_id": self.auth_owner,
                    "request_id": str(self.payload.request_id),
                    "state": "pending",
                },
            )
        if "/auth/v1/admin/users/" in path:
            assert path.endswith("/" + USER_ID)
            assert req.headers["apikey"] == "fictional-server-only"
            if req.method == "DELETE":
                assert self.pending
                assert json.loads(req.content) == {"should_soft_delete": False}
                if self.fail_delete:
                    return httpx.Response(503, json={"message": "fictional failure"})
                self.deleted = True
                if self.timeout_after_delete:
                    self.timeout_after_delete = False
                    raise httpx.ReadTimeout("response lost", request=req)
                return httpx.Response(200, json={"id": USER_ID})
            return httpx.Response(404 if self.deleted else 200, json={"id": USER_ID})
        if path.endswith("jp_finish_account_closure"):
            assert self.deleted
            self.complete = True
            return httpx.Response(200, json={"state": "complete"})
        raise AssertionError(path)


def test_freshness_uses_verified_auth_method_not_refreshed_iat():
    require_recent_auth(AuthContext(UUID(USER_ID), token()), NOW)
    for value in [
        token(method="token_refresh"),
        token(stamp=int((NOW - timedelta(minutes=6)).timestamp())),
        token(stamp=True),
        token(stamp=int(NOW.timestamp()) + 60),
        token(amr=[]),
        token(is_anonymous=True),
        token(owner=str(uuid4())),
    ]:
        with pytest.raises(HTTPException) as error:
            require_recent_auth(AuthContext(UUID(USER_ID), value), NOW)
        assert error.value.status_code == 403


def test_initial_closure_is_authenticated_self_only_and_confirmed(tmp_path):
    payload = request()
    transport = ClosureTransport(payload)
    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        AccountClosureService(settings(tmp_path), client).close(
            payload, "Bearer " + transport.auth_token, NOW
        )
    assert transport.pending and transport.deleted and transport.complete


def test_lost_delete_response_can_resume_after_user_auth_is_gone(tmp_path):
    payload = request()
    transport = ClosureTransport(payload)
    transport.timeout_after_delete = True
    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        service = AccountClosureService(settings(tmp_path), client)
        with pytest.raises(HTTPException) as error:
            service.close(payload, "Bearer " + transport.auth_token, NOW)
        assert error.value.status_code == 503 and not transport.complete
        service.close(payload, "Bearer " + transport.auth_token, NOW)
        assert transport.complete
        count = len(transport.calls)
        service.close(payload, None, NOW)
        assert transport.calls[count:] == [("POST", "/rest/v1/rpc/jp_resolve_account_closure")]


def test_failed_admin_delete_never_claims_completion(tmp_path):
    payload = request()
    transport = ClosureTransport(payload)
    transport.fail_delete = True
    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        with pytest.raises(HTTPException) as error:
            AccountClosureService(settings(tmp_path), client).close(
                payload, "Bearer " + transport.auth_token, NOW
            )
    assert error.value.status_code == 503
    assert transport.pending and not transport.deleted and not transport.complete


def test_other_signed_in_account_cannot_apply_a_previous_accounts_receipt(tmp_path):
    payload = request()
    transport = ClosureTransport(payload)
    transport.pending = True
    transport.auth_owner = str(uuid4())
    transport.auth_token = token(owner=transport.auth_owner)
    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        with pytest.raises(HTTPException) as error:
            AccountClosureService(settings(tmp_path), client).close(
                payload, "Bearer " + transport.auth_token, NOW
            )
    assert error.value.status_code == 403 and not transport.deleted


def test_stale_signin_does_not_mark_closure_pending(tmp_path):
    payload = request()
    transport = ClosureTransport(payload)
    transport.auth_token = token(stamp=int(NOW.timestamp()) - 301)
    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        with pytest.raises(HTTPException) as error:
            AccountClosureService(settings(tmp_path), client).close(
                payload, "Bearer " + transport.auth_token, NOW
            )
    assert error.value.status_code == 403 and not transport.pending


def test_api_rejects_user_id_injection_and_accepts_json_receipt(tmp_path):
    payload = request()
    transport = ClosureTransport(payload)
    with httpx.Client(transport=httpx.MockTransport(transport)) as remote:
        app = create_app(settings=settings(tmp_path), account_closure_client=remote, clock=lambda: NOW)
        with TestClient(app) as client:
            body = payload.model_dump(mode="json")
            injected = client.request("DELETE", "/v1/account", json={**body, "user_id": str(uuid4())})
            assert injected.status_code == 422 and transport.calls == []
            result = client.request(
                "DELETE",
                "/v1/account",
                json=body,
                headers={"Authorization": "Bearer " + transport.auth_token},
            )
            assert result.status_code == 204 and result.content == b""


def test_missing_configuration_never_starts_deletion(tmp_path):
    with pytest.raises(HTTPException) as error:
        AccountClosureService(configured(tmp_path)).close(request(), None, NOW)
    assert error.value.status_code == 503


def test_invalid_request_never_echoes_the_private_receipt(tmp_path, caplog):
    with TestClient(create_app(settings=configured(tmp_path))) as client:
        response = client.request(
            "DELETE",
            "/v1/account",
            json={
                "confirmation": "wrong",
                "request_id": str(uuid4()),
                "receipt_secret": SECRET,
            },
        )
    assert response.status_code == 422
    assert SECRET not in response.text
    assert SECRET not in caplog.text


def test_invalid_receipt_without_a_verified_session_cannot_start_closure(tmp_path):
    payload = request()
    transport = ClosureTransport(payload)
    # The lookup yields no operation and the presented ordinary session is invalid.
    transport.deleted = True
    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        with pytest.raises(HTTPException) as error:
            AccountClosureService(settings(tmp_path), client).close(
                payload, "Bearer " + transport.auth_token, NOW
            )
    assert error.value.status_code == 401
    assert not transport.pending and not transport.complete
