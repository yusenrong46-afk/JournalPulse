"""Check exercises/: three stand-alone exercises whose tests all fail on the shipped stubs.

    uv run python scripts/check_exercises.py

Each ``exercises/NN_name/`` folder must hold a stub module, a ``test_*.py`` file and
``HINTS.md``, and no file whose name mentions a solution. Every test must *fail* (not pass,
error or skip), and fail because a stub raised ``NotImplementedError``: that proves the tests
exercise the stubs and that no solution was committed.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ElementTree
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXERCISES = ROOT / "exercises"
MINIMUM = 3


def main() -> int:
    errors: list[str] = []
    folders = sorted(path for path in EXERCISES.iterdir() if path.is_dir() and path.name[:2].isdigit())
    if len(folders) < MINIMUM:
        errors.append(f"expected at least {MINIMUM} exercises, found {len(folders)}")
    for folder in folders:
        names = [path.name for path in folder.iterdir() if path.is_file()]
        if "HINTS.md" not in names:
            errors.append(f"{folder.name}: HINTS.md is missing")
        if not any(name.startswith("test_") and name.endswith(".py") for name in names):
            errors.append(f"{folder.name}: no test_*.py")
        if not any(name.endswith(".py") and not name.startswith("test_") for name in names):
            errors.append(f"{folder.name}: no stub module")
        if any("solution" in name.lower() for name in names):
            errors.append(f"{folder.name}: a solution file is committed")

    with tempfile.TemporaryDirectory() as scratch:
        report = Path(scratch) / "junit.xml"
        subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-o", "addopts=", "-p", "no:cacheprovider",
             f"--junitxml={report}", str(EXERCISES)],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        if not report.exists():
            errors.append("pytest did not produce a report")
            cases = []
        else:
            cases = list(ElementTree.parse(report).getroot().iter("testcase"))
    for case in cases:
        name = f"{case.get('classname')}::{case.get('name')}"
        failure = case.find("failure")
        if failure is None:
            outcome = "errored" if case.find("error") is not None else (
                "skipped" if case.find("skipped") is not None else "passed")
            errors.append(f"{name} {outcome} on the stubs; it must fail")
        elif "NotImplementedError" not in (failure.get("message", "") + (failure.text or "")):
            errors.append(f"{name} failed for another reason than an unimplemented stub")
    if len(cases) < MINIMUM:
        errors.append(f"only {len(cases)} tests ran")
    for error in errors:
        print(f"exercises: {error}", file=sys.stderr)
    if errors:
        return 1
    print(f"exercises check passed: {len(folders)} exercises, all {len(cases)} tests fail on the stubs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
