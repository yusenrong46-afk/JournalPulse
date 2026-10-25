"""Static build must stop before compiling inconsistent browser Auth configuration."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "build_web", Path(__file__).parents[1] / "scripts/build_vercel_web.py")
assert spec and spec.loader
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


@pytest.mark.parametrize("public", ["https://example.supabase.co", "https://example.supabase.co:443/"])
def test_supported_auth_origin(public):
    build.validate_auth_origins({"SUPABASE_URL": "https://example.supabase.co",
                                 "NEXT_PUBLIC_SUPABASE_URL": public})


@pytest.mark.parametrize("public", ["https://other.supabase.co", "https://example.supabase.co/path",
                                   "https://user:secret@example.supabase.co", "https://x;evil.example",
                                   "http://example.supabase.co", "https://["])
def test_mismatch_or_invalid_origin_stops_before_build(monkeypatch, public):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("NEXT_PUBLIC_SUPABASE_URL", public)
    monkeypatch.setattr(build.subprocess, "run", lambda *a, **k: pytest.fail("Build must not start"))
    with pytest.raises(SystemExit, match="origin"):
        build.main()


def test_local_export_needs_no_auth():
    build.validate_auth_origins({})
    build.validate_auth_origins({"SUPABASE_URL": "http://127.0.0.1:54321",
                                 "NEXT_PUBLIC_SUPABASE_URL": "http://127.0.0.1:54321"})
