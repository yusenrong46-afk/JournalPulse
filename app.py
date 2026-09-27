"""Vercel entrypoint. The package lives in src/, which is not on the default path."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

app = importlib.import_module("journalpulse.api").app
