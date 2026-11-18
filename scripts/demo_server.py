"""Run the portfolio demo on this machine: fictional data, AI and search off, no sign-up.

    uv run python scripts/demo_server.py [--port 8000] [--skip-build]

What runs (all on 127.0.0.1, nothing hosted):

* the FastAPI app with a fresh SQLite database under ``.demo/`` and local identities;
* the exported web app (built with the "Try the demo" button) served by that same API.

Every provider credential is blanked for this process, so no model or search request can
be made even if a local ``.env`` holds keys; demo identities are refused by the API in
any case (``AuthContext.demo``). Ctrl+C stops the server.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / ".demo"
DEMO_WEB = DEMO_DIR / "web"
DEMO_DB = DEMO_DIR / "demo.db"

# Blank values win over .env (python-dotenv never overrides a variable that is set).
DEMO_ENVIRONMENT = {
    "JOURNALPULSE_ENV": "local",
    "JOURNALPULSE_DEMO_ENABLED": "true",
    "JOURNALPULSE_LLM_ENABLED": "false",
    "JOURNALPULSE_SEARCH_ENABLED": "false",
    "JOURNALPULSE_LLM_API_KEY": "",
    "OPENROUTER_API_KEY": "",
    "JOURNALPULSE_SEARCH_API_KEY": "",
    "SUPABASE_URL": "",
    "SUPABASE_ANON_KEY": "",
}
WEB_BUILD_ENVIRONMENT = {
    "JOURNALPULSE_STATIC_EXPORT": "true",
    "NEXT_PUBLIC_JOURNALPULSE_DEMO_ENABLED": "true",
    "NEXT_PUBLIC_API_BASE_URL": "",
    "NEXT_PUBLIC_SUPABASE_URL": "",
    "NEXT_PUBLIC_SUPABASE_ANON_KEY": "",
}


def build_web() -> None:
    environment = {**os.environ, **WEB_BUILD_ENVIRONMENT}
    command = ["npm", "--prefix", str(ROOT / "web"), "run", "build"]
    if subprocess.run(command, env=environment, check=False).returncode != 0:
        # Next.js 16.4 stops the first build in an agent session to show an upgrade
        # notice and asks for the same command again; a real build error fails twice.
        print("web build stopped; retrying once", flush=True)
        subprocess.run(command, env=environment, check=True)
    if DEMO_WEB.exists():
        shutil.rmtree(DEMO_WEB)
    # Keep a private copy: other builds (the integration stack) also write web/out.
    shutil.copytree(ROOT / "web" / "out", DEMO_WEB)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--skip-build", action="store_true", help="reuse .demo/web if it exists")
    args = parser.parse_args(argv)
    DEMO_DIR.mkdir(exist_ok=True)
    if not (args.skip_build and (DEMO_WEB / "index.html").exists()):
        build_web()
    for path in DEMO_DIR.glob("demo.db*"):
        path.unlink()  # every demo run starts from an empty database
    environment = {
        **os.environ,
        **DEMO_ENVIRONMENT,
        "JOURNALPULSE_DB_PATH": str(DEMO_DB),
        "JOURNALPULSE_WEB_DIST": str(DEMO_WEB),
        "JOURNALPULSE_CORS_ORIGINS": f"http://127.0.0.1:{args.port}",
    }
    print(f"JournalPulse demo on http://127.0.0.1:{args.port}/login/ (fictional data, AI off)", flush=True)
    command = [sys.executable, "-m", "uvicorn", "journalpulse.api:app", "--host", "127.0.0.1",
               "--port", str(args.port), "--log-level", "warning"]
    return subprocess.run(command, env=environment, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
