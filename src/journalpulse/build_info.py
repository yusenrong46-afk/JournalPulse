"""Build identity: which Git commit this process was built from.

Hosts expose the commit differently; the first valid value wins. The API never runs Git
at request time. ``unknown`` means no host variable was set (for example plain local runs).
"""

from __future__ import annotations

import os
import re

from pydantic import BaseModel

# JOURNALPULSE_GIT_COMMIT is set by Dockerfile.api and the local scripts; the others are
# set by Vercel, Render and GitHub Actions respectively.
COMMIT_VARIABLES = ("JOURNALPULSE_GIT_COMMIT", "VERCEL_GIT_COMMIT_SHA", "RENDER_GIT_COMMIT", "GITHUB_SHA")
_COMMIT = re.compile(r"^[0-9a-f]{7,40}$")


class BuildIdentity(BaseModel):
    commit: str
    source: str


def build_identity() -> BuildIdentity:
    for name in COMMIT_VARIABLES:
        value = os.getenv(name, "").strip().lower()
        if _COMMIT.fullmatch(value):
            return BuildIdentity(commit=value, source=name)
    return BuildIdentity(commit="unknown", source="none")
