import base64
import hashlib
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from journalpulse.api import create_app
from journalpulse.web_security import content_security_policy
from test_research_beta_api import settings


def test_export_policy_allows_only_built_inline_scripts(tmp_path, monkeypatch):
    export = tmp_path / "export"
    export.mkdir()
    script = 'window.example = "safe & sound";'
    body = f'<h1>Public information</h1><script>{script}</script>'.encode()
    (export / "index.html").write_bytes(body)
    monkeypatch.setenv("JOURNALPULSE_WEB_DIST", str(export))
    configured = replace(settings(tmp_path), supabase_url="https://example.supabase.co")
    with TestClient(create_app(settings=configured)) as client:
        response = client.get("/")
        head = client.head("/")
    policy = response.headers["content-security-policy"]
    digest = base64.b64encode(hashlib.sha256(script.encode()).digest()).decode()
    assert f"'sha256-{digest}'" in policy
    assert response.content == body
    assert head.content == b"" and head.headers["content-security-policy"] == policy
    scripts = next(part for part in policy.split(";") if part.strip().startswith("script-src "))
    assert "unsafe-inline" not in scripts and "unsafe-eval" not in scripts
    assert "frame-ancestors 'none'" in policy
    assert "connect-src 'self' https://example.supabase.co" in policy


def test_csp_changes_when_same_size_export_script_changes(tmp_path, monkeypatch):
    export = tmp_path / "export"
    export.mkdir()
    page = export / "index.html"
    monkeypatch.setenv("JOURNALPULSE_WEB_DIST", str(export))
    with TestClient(create_app(settings=settings(tmp_path))) as client:
        page.write_text('<script>window.flag = "A";</script>')
        first = client.get("/").headers["content-security-policy"]
        page.write_text('<script>window.flag = "B";</script>')
        second = client.get("/").headers["content-security-policy"]
    assert first != second


@pytest.mark.parametrize("url", ["https://x;evil.example", "https://[", "https://user:password@evil.example"])
def test_csp_rejects_invalid_or_injected_origin_without_hiding_public_pages(url):
    policy = content_security_policy(b"<h1>Page</h1>", url)
    assert "connect-src 'self'" in policy
    assert "evil.example" not in policy


def test_production_hides_interactive_api_docs_but_schema_generation_still_works(tmp_path):
    app = create_app(settings=replace(settings(tmp_path), environment="production"))
    with TestClient(app) as client:
        for path in ("/docs", "/redoc", "/openapi.json"):
            assert client.get(path).status_code == 404
    assert "/v1/conversations" in app.openapi()["paths"]


def test_development_keeps_api_documentation(tmp_path):
    with TestClient(create_app(settings=settings(tmp_path))) as client:
        assert client.get("/docs").status_code == 200
        assert client.get("/openapi.json").status_code == 200
