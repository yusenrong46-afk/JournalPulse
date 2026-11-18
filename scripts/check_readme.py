"""Check that the README's first screen meets the Portfolio Standard L1 rules.

The first screen is everything above the first ``## `` heading. It must hold four
marked parts, each introduced by an HTML comment so the check is exact instead of
guessing at prose:

* ``<!-- readme:purpose -->``  one sentence saying what the app does and for whom
* ``<!-- readme:result -->``   one dated result (YYYY-MM-DD) with a number
* ``<!-- readme:image -->``    one Markdown image whose file exists
* ``<!-- readme:see-it -->``   how to see it: a demo link or a one-line command

Every relative link anywhere in the README must resolve to a file in the repository,
and every in-page anchor must match a heading.

    uv run python scripts/check_readme.py [--readme README.md]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIRST_SCREEN_MAX_LINES = 40
MARKERS = ("purpose", "result", "image", "see-it")
LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
IMAGE = re.compile(r"!\[[^\]]+\]\(([^)\s]+)\)")
DATE = re.compile(r"\b20\d{2}-\d{2}-\d{2}\b")
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def first_screen(text: str) -> list[str]:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if index > 0 and line.startswith("## "):
            return lines[:index]
    return lines


def marked_block(lines: list[str], name: str) -> str | None:
    """Return the text after ``<!-- readme:name -->`` up to the next blank line or marker."""
    tag = f"<!-- readme:{name} -->"
    for index, line in enumerate(lines):
        if line.strip() == tag:
            block: list[str] = []
            for following in lines[index + 1 :]:
                if not following.strip() or following.strip().startswith("<!-- readme:"):
                    break
                block.append(following.strip())
            return " ".join(block)
    return None


def github_anchor(heading: str) -> str:
    slug = heading.strip().lower()
    slug = re.sub(r"[^\w\- ]", "", slug)
    return slug.replace(" ", "-")


def check(readme: Path) -> list[str]:
    errors: list[str] = []
    text = readme.read_text(encoding="utf-8")
    screen = first_screen(text)
    if len(screen) > FIRST_SCREEN_MAX_LINES:
        errors.append(f"first screen is {len(screen)} lines; keep it within {FIRST_SCREEN_MAX_LINES}")

    blocks = {name: marked_block(screen, name) for name in MARKERS}
    for name, block in blocks.items():
        if not block:
            errors.append(f"first screen is missing the <!-- readme:{name} --> block")

    purpose = blocks["purpose"] or ""
    if purpose:
        sentences = [part for part in re.split(r"(?<=[.!?])\s+", purpose) if part]
        if len(sentences) != 1:
            errors.append(f"purpose must be one sentence; found {len(sentences)}")
        if " for " not in f" {purpose} ":
            errors.append("purpose must say who it is for (use the word 'for')")

    result = blocks["result"] or ""
    if result:
        dates = DATE.findall(result)
        if not dates:
            errors.append("result must carry a date in YYYY-MM-DD form")
        numbers = [value for value in NUMBER.findall(DATE.sub("", result)) if value.strip(",")]
        if not numbers:
            errors.append("result must carry a number")

    image = blocks["image"] or ""
    if image:
        match = IMAGE.search(image)
        if not match:
            errors.append("image block must be a Markdown image with alt text")
        elif not (readme.parent / match.group(1)).is_file():
            errors.append(f"image file does not exist: {match.group(1)}")

    see_it = blocks["see-it"] or ""
    if see_it and not (re.search(r"\]\(https?://", see_it) or re.search(r"`[^`]+`", see_it)):
        errors.append("see-it must give a demo link or a command in backticks")

    anchors = {
        github_anchor(line.lstrip("#")) for line in text.splitlines() if re.match(r"#{1,6} ", line)
    }
    in_code = False
    for number, line in enumerate(text.splitlines(), start=1):
        if line.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        for target in LINK.findall(line):
            if re.match(r"^[a-z][a-z0-9+.-]*:", target):
                continue  # http(s), mailto and other absolute links are out of scope
            path, _, anchor = target.partition("#")
            if not path:
                if anchor and anchor not in anchors:
                    errors.append(f"line {number}: anchor #{anchor} matches no heading")
                continue
            if not (readme.parent / path).exists():
                errors.append(f"line {number}: relative link does not resolve: {target}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--readme", type=Path, default=ROOT / "README.md")
    args = parser.parse_args(argv)
    errors = check(args.readme)
    for error in errors:
        print(f"README check failed: {error}")
    if errors:
        return 1
    print(f"README check passed: {args.readme.name} first screen and links")
    return 0


if __name__ == "__main__":
    sys.exit(main())
