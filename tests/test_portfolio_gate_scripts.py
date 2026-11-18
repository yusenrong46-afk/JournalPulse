"""The Portfolio Standard L1 gate scripts must fail on the problems they exist to catch."""

import importlib
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
GOOD_README = """# Example

<!-- readme:purpose -->
Example is a notebook for people who want to keep notes.

<!-- readme:result -->
**Measured 2026-11-18:** 12 tests passed.

<!-- readme:image -->
![A screenshot](shot.png)

<!-- readme:see-it -->
Run `make demo`.

## Details

See [the guide](guide.md) and [details](#details).
"""


@pytest.fixture
def check_readme(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return importlib.import_module("check_readme")


def write(tmp_path: Path, text: str) -> Path:
    (tmp_path / "shot.png").write_bytes(b"png")
    (tmp_path / "guide.md").write_text("guide")
    readme = tmp_path / "README.md"
    readme.write_text(text)
    return readme


def test_complete_first_screen_passes(check_readme, tmp_path: Path) -> None:
    assert check_readme.check(write(tmp_path, GOOD_README)) == []


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("<!-- readme:purpose -->\n", "", "missing the <!-- readme:purpose -->"),
        ("for people who", "so people", "who it is for"),
        ("keep notes.", "keep notes. It is fast.", "one sentence"),
        ("2026-11-18", "today", "date"),
        ("12 tests passed", "tests passed", "number"),
        ("shot.png", "missing.png", "image file does not exist"),
        ("Run `make demo`.", "Run it.", "demo link or a command"),
        ("guide.md", "nowhere.md", "does not resolve"),
        ("(#details)", "(#nothing)", "matches no heading"),
    ],
)
def test_each_missing_part_fails(check_readme, tmp_path: Path, old: str, new: str, message: str) -> None:
    errors = check_readme.check(write(tmp_path, GOOD_README.replace(old, new, 1)))
    assert any(message in error for error in errors), errors


def test_repository_readme_passes(check_readme) -> None:
    assert check_readme.check(SCRIPTS.parent / "README.md") == []


@pytest.fixture
def secret_scan(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return importlib.import_module("secret_scan")


@pytest.mark.parametrize(
    "line",
    [
        "OPENROUTER_API_KEY=sk-or-v1-" + "a1" * 24,
        "token = 'ghp_" + "Z" * 36 + "'",
        "-----BEGIN " + "PRIVATE KEY-----",
        "AKIA" + "ABCDEFGHIJKLMNOP",
        "eyJ" + "a" * 20 + ".eyJ" + "b" * 20 + "." + "c" * 30,
    ],
)
def test_credential_shapes_are_detected(secret_scan, line: str) -> None:
    assert any(pattern.search(line) for pattern in secret_scan.PATTERNS.values())


def test_ordinary_text_is_not_flagged(secret_scan) -> None:
    for line in ["JWT_SECRET = 'integration-only-jwt-secret-0123456789abcdef'", "sk-learn is a library"]:
        assert not any(pattern.search(line) for pattern in secret_scan.PATTERNS.values())
