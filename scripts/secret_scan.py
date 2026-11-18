"""Scan tracked files for credentials and for oversized new files.

Two checks, both read-only and offline:

1. Every file Git tracks is searched for credential shapes (provider keys, tokens,
   private keys, signed JWTs). A match prints the file, line and pattern name only;
   the matched value is never printed.
2. No file added or changed since the merge base with ``--base`` (default
   ``origin/main``) may exceed 5 MB.

A line may carry ``secret-scan: allow`` to mark a documented, non-secret fixture.

    uv run python scripts/secret_scan.py [--base origin/main]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_NEW_FILE_BYTES = 5 * 1024 * 1024
PATTERNS: dict[str, re.Pattern[str]] = {
    "openrouter_or_openai_key": re.compile(r"\bsk-(?:or-v1-|proj-)?[A-Za-z0-9_-]{32,}"),
    "anthropic_key": re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private_key_block": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "github_token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})"),
    "slack_token": re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
    "supabase_secret_key": re.compile(r"\bsb_secret_[A-Za-z0-9_-]{16,}"),
    "signed_jwt": re.compile(r"\beyJ[A-Za-z0-9_-]{15,}\.eyJ[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{20,}"),
    "google_api_key": re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
}
ALLOW = "secret-scan: allow"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout


def candidate_files() -> list[str]:
    """Tracked files plus untracked files Git does not ignore (work not yet committed)."""
    names = git("ls-files", "-z").split("\0") + git("ls-files", "-z", "--others", "--exclude-standard").split(
        "\0"
    )
    return sorted({name for name in names if name})


def scan_secrets() -> list[str]:
    findings: list[str] = []
    for name in candidate_files():
        if not name:
            continue
        path = ROOT / name
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary files (images, databases) are covered by the size check only
        for number, line in enumerate(text.splitlines(), start=1):
            if ALLOW in line:
                continue
            for label, pattern in PATTERNS.items():
                if pattern.search(line):
                    findings.append(f"{name}:{number}: looks like {label}")
    return findings


def oversized_new_files(base: str) -> list[str]:
    try:
        merge_base = git("merge-base", "HEAD", base).strip()
    except subprocess.CalledProcessError:
        return [f"cannot find merge base with {base}; fetch it first"]
    changed = git("diff", "--name-only", "--diff-filter=AM", merge_base, "HEAD").splitlines()
    changed += git("ls-files", "--others", "--exclude-standard").splitlines()
    problems = []
    for name in sorted(set(changed)):
        path = ROOT / name
        if path.is_file() and path.stat().st_size > MAX_NEW_FILE_BYTES:
            problems.append(f"{name}: {path.stat().st_size} bytes exceeds 5 MB")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="origin/main")
    args = parser.parse_args(argv)
    secrets = scan_secrets()
    sizes = oversized_new_files(args.base)
    for line in secrets + sizes:
        print(f"secret scan failed: {line}")
    if secrets or sizes:
        return 1
    print(f"secret scan passed: {len(candidate_files())} files, 0 findings, no new file over 5 MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
