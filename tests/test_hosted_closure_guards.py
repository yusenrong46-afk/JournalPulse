"""Guard the hosted test tool itself; all HTTP responses here are fictional."""
import importlib.util
import json
import sys
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

spec = importlib.util.spec_from_file_location(
    "hosted_closure", Path(__file__).parents[1] / "scripts/verify_hosted_closure.py")
assert spec and spec.loader
smoke = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = smoke
spec.loader.exec_module(smoke)


def test_hosted_tool_rejects_unrelated_or_credential_bearing_origins():
    assert smoke.app_origin("https://journalpulse-stage-team.vercel.app/").endswith(".app")
    for value in ["http://journalpulse.vercel.app", "https://evil.test", "https://journalpulse.vercel.app.evil.test",
                  "https://secret@journalpulse.vercel.app", "https://journalpulse.vercel.app?key=secret"]:
        with pytest.raises(smoke.SmokeStopped):
            smoke.app_origin(value)


def test_cleanup_never_deletes_a_user_with_mismatched_creation_metadata():
    calls = []
    owner = str(uuid4())
    def handler(request):
        calls.append(request.method)
        return httpx.Response(200, json={"id": owner, "email": "real@example.test", "app_metadata": {}})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        runner = smoke.HostedSmoke(client, "https://journalpulse.vercel.app", "fictional-admin", "anon")
        runner.fixtures.append(smoke.Fixture(owner, "fictional@example.test", runner.run_id, "unused"))
        assert runner.cleanup() == [owner]
    assert calls == ["GET"]


def test_unknown_creation_is_recorded_without_retry_or_unproven_cleanup():
    calls = []
    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout("fictional interruption", request=request)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        runner = smoke.HostedSmoke(client, "https://journalpulse.vercel.app", "fictional-admin", "anon")
        with pytest.raises(httpx.ReadTimeout):
            runner.create("a")
        assert runner.cleanup() == []
        assert len(runner.uncertain_creations) == 1
        assert runner.fixtures == []
    assert len(calls) == 1


def test_creation_and_cleanup_are_project_pinned_without_sending_email():
    created = {}
    requests = []
    deleted = False
    def handler(request):
        nonlocal deleted
        requests.append(request)
        assert request.url.host == "cggixttnyprrbaotmjxj.supabase.co"
        assert request.headers["authorization"] == "Bearer fictional-admin"
        if request.method == "POST":
            assert request.url.path == "/auth/v1/admin/users"
            body = json.loads(request.content)
            assert body["email_confirm"] is True and body["email"].endswith("@example.test")
            created.update(id=str(uuid4()), email=body["email"], app_metadata=body["app_metadata"])
            return httpx.Response(200, json=created)
        assert request.url.path == "/auth/v1/admin/users/" + created["id"]
        if request.method == "DELETE":
            assert json.loads(request.content) == {"should_soft_delete": False}
            deleted = True
            return httpx.Response(200, json=created)
        return httpx.Response(404 if deleted else 200, json=created)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        runner = smoke.HostedSmoke(client, "https://journalpulse.vercel.app", "fictional-admin", "anon")
        fixture = runner.create("a")
        assert "password" not in repr(fixture)
        assert runner.uncertain_creations == []
        assert runner.cleanup() == []
    assert [r.method for r in requests] == ["POST", "GET", "DELETE", "GET"]
