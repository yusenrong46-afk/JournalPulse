"""The integration stack must only run the reviewed PostgREST build it pins."""

import importlib
import io
import tarfile
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
FAKE_POSTGREST = b"#!/bin/sh\necho 'PostgREST 12.2.12 (fake)'\n"
WRONG_POSTGREST = b"#!/bin/sh\necho 'PostgREST 16.4 (fake)'\n"


def load_stack(monkeypatch: pytest.MonkeyPatch, home: Path):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    module = importlib.import_module("integration_stack")
    monkeypatch.setattr(module.Path, "home", staticmethod(lambda: home))
    monkeypatch.setattr(module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(module.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(module.shutil, "which", lambda name: None)
    return module


def archive(script: bytes) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:xz") as bundle:
        info = tarfile.TarInfo("postgrest")
        info.size = len(script)
        info.mode = 0o755
        bundle.addfile(info, io.BytesIO(script))
    return buffer.getvalue()


def serve(monkeypatch: pytest.MonkeyPatch, module, payload: bytes) -> list[str]:
    requested: list[str] = []

    def fake_urlopen(url, *args, **kwargs):
        requested.append(url if isinstance(url, str) else url.full_url)
        response = io.BytesIO(payload)
        response.__enter__ = lambda self=response: self  # type: ignore[method-assign]
        response.__exit__ = lambda *a: None  # type: ignore[method-assign]
        response.headers = {}  # type: ignore[attr-defined]
        response.info = lambda: {}  # type: ignore[attr-defined]
        return response

    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)
    return requested


def write_binary(path: Path, script: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(script)
    path.chmod(0o755)
    return path


def test_tampered_download_is_rejected_and_never_installed(monkeypatch, tmp_path):
    module = load_stack(monkeypatch, tmp_path)
    serve(monkeypatch, module, archive(FAKE_POSTGREST))
    with pytest.raises(RuntimeError, match="SHA-256"):
        module.postgrest_binary()
    assert not list(tmp_path.rglob("postgrest"))


def test_pinned_download_is_verified_then_installed(monkeypatch, tmp_path):
    module = load_stack(monkeypatch, tmp_path)
    payload = archive(FAKE_POSTGREST)
    digest = module.hashlib.sha256(payload).hexdigest()
    monkeypatch.setitem(module.POSTGREST_ASSETS, ("Linux", "x86_64"), ("linux-static-x86-64", digest))
    requested = serve(monkeypatch, module, payload)
    binary = module.postgrest_binary()
    assert binary.read_bytes() == FAKE_POSTGREST
    assert requested and requested[0].endswith("postgrest-v12.2.12-linux-static-x86-64.tar.xz")
    assert not list(binary.parent.glob("*.tar.xz")), "the verified archive is not left behind"


def test_wrong_version_on_path_or_in_cache_is_not_used(monkeypatch, tmp_path):
    module = load_stack(monkeypatch, tmp_path)
    on_path = write_binary(tmp_path / "bin" / "postgrest", WRONG_POSTGREST)
    monkeypatch.setattr(module.shutil, "which", lambda name: str(on_path))
    write_binary(tmp_path / ".cache" / "journalpulse" / "v12.2.12" / "postgrest", WRONG_POSTGREST)
    payload = archive(FAKE_POSTGREST)
    digest = module.hashlib.sha256(payload).hexdigest()
    monkeypatch.setitem(module.POSTGREST_ASSETS, ("Linux", "x86_64"), ("linux-static-x86-64", digest))
    serve(monkeypatch, module, payload)
    binary = module.postgrest_binary()
    assert binary != on_path
    assert binary.read_bytes() == FAKE_POSTGREST


def test_matching_version_on_path_is_used_without_download(monkeypatch, tmp_path):
    module = load_stack(monkeypatch, tmp_path)
    on_path = write_binary(tmp_path / "bin" / "postgrest", FAKE_POSTGREST)
    monkeypatch.setattr(module.shutil, "which", lambda name: str(on_path))
    requested = serve(monkeypatch, module, b"")
    assert module.postgrest_binary() == on_path
    assert requested == []


def test_unsupported_platform_names_the_manual_option(monkeypatch, tmp_path):
    module = load_stack(monkeypatch, tmp_path)
    monkeypatch.setattr(module.platform, "system", lambda: "Plan9")
    with pytest.raises(RuntimeError, match="PostgREST 12.2.12 on PATH"):
        module.postgrest_binary()
