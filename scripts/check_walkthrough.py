"""Check docs/WALKTHROUGH.md: ten questions, short answers, pointers that resolve.

    uv run python scripts/check_walkthrough.py [--file docs/WALKTHROUGH.md]

Rules (Portfolio Standard L5):

* exactly ten questions, each a ``## Q<n>. ...`` heading, numbered 1 to 10;
* each answer is under 120 words, not counting its ``Where:`` line;
* each answer has a ``Where:`` line with at least one pointer written as
  ```path` → `symbol```; the file must exist in this repository and the symbol must appear
  in it literally, so a renamed function or a moved file fails the check.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = 10
MAX_WORDS = 120
HEADING = re.compile(r"^## Q(\d+)\. (.+)$")
POINTER = re.compile(r"`([\w./-]+)` → `([\w.:-]+)`")


def sections(text: str) -> list[tuple[int, str, list[str]]]:
    found: list[tuple[int, str, list[str]]] = []
    for line in text.splitlines():
        match = HEADING.match(line)
        if match:
            found.append((int(match.group(1)), match.group(2), []))
        elif line.startswith("## "):
            found.append((-1, line, []))  # a non-question section ends the previous answer
        elif found:
            found[-1][2].append(line)
    return [item for item in found if item[0] != -1]


def check(path: Path) -> list[str]:
    errors: list[str] = []
    found = sections(path.read_text())
    numbers = [number for number, _, _ in found]
    if numbers != list(range(1, QUESTIONS + 1)):
        errors.append(f"expected questions Q1 to Q{QUESTIONS} in order, found {numbers}")
    for number, question, body in found:
        where = [line for line in body if line.startswith("Where:")]
        prose = " ".join(line for line in body if not line.startswith("Where:"))
        words = len(re.findall(r"[A-Za-z0-9][\w'’./-]*", prose))
        if words >= MAX_WORDS:
            errors.append(f"Q{number} answer is {words} words; keep it under {MAX_WORDS}")
        if len(where) != 1:
            errors.append(f"Q{number} needs exactly one 'Where:' line")
            continue
        pointers = POINTER.findall(where[0])
        if not pointers:
            errors.append(f"Q{number} 'Where:' line has no `path` → `symbol` pointer")
        for file, symbol in pointers:
            target = ROOT / file
            if not target.is_file():
                errors.append(f"Q{number}: {file} does not exist")
            elif symbol not in target.read_text(errors="replace"):
                errors.append(f"Q{number}: `{symbol}` does not appear in {file}")
        if not question.endswith("?"):
            errors.append(f"Q{number} heading should be a question")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--file", type=Path, default=ROOT / "docs" / "WALKTHROUGH.md")
    args = parser.parse_args(argv)
    errors = check(args.file)
    for error in errors:
        print(f"walkthrough: {error}", file=sys.stderr)
    if errors:
        return 1
    print(f"walkthrough check passed: {QUESTIONS} questions, every pointer resolves")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
