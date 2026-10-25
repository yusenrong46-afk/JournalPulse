"""Build the Next.js PWA into web-dist/ for the FastAPI static mount.

The Python function keeps the API routes. Pages, the service worker, and the
manifest are static files. The anon key is public and is copied from the
server Supabase variables when the Next.js names were not set separately.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
PUBLIC = ROOT / "web-dist"


def auth_origin(value: str) -> str:
    """Static exports support one shared Auth origin; reject ambiguous URLs."""
    try:
        parsed = urlsplit(value)
        port = parsed.port
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path not in {"", "/"} or any(c.isspace() for c in value)
                or not re.fullmatch(r"[A-Za-z0-9.\-:\[\]]+", parsed.netloc)):
            raise ValueError
        if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError
        host = parsed.hostname
        if ":" in host:
            host = f"[{host}]"
        suffix = f":{port}" if port and port != {"https": 443, "http": 80}[parsed.scheme] else ""
        return f"{parsed.scheme}://{host}{suffix}"
    except ValueError:
        raise SystemExit("Invalid Auth origin; use an HTTPS origin or loopback HTTP origin.") from None


def validate_auth_origins(env: dict[str, str]) -> None:
    backend = env.get("SUPABASE_URL", "")
    frontend = env.get("NEXT_PUBLIC_SUPABASE_URL", "")
    if not backend and not frontend:
        return  # The journal-only local export needs no Auth service.
    if not backend or not frontend or auth_origin(backend) != auth_origin(frontend):
        raise SystemExit("Unsupported Auth origin mismatch: frontend and API must use the same origin.")


def main() -> None:
    env = os.environ.copy()
    env["JOURNALPULSE_STATIC_EXPORT"] = "true"
    env.setdefault("NEXT_PUBLIC_API_BASE_URL", "")
    if not env.get("NEXT_PUBLIC_SUPABASE_URL") and env.get("SUPABASE_URL"):
        env["NEXT_PUBLIC_SUPABASE_URL"] = env["SUPABASE_URL"]
    if not env.get("NEXT_PUBLIC_SUPABASE_ANON_KEY") and env.get("SUPABASE_ANON_KEY"):
        env["NEXT_PUBLIC_SUPABASE_ANON_KEY"] = env["SUPABASE_ANON_KEY"]

    validate_auth_origins(env)

    subprocess.run(["npm", "ci"], cwd=WEB, check=True, env=env)
    subprocess.run(["npm", "run", "build"], cwd=WEB, check=True, env=env)
    exported = WEB / "out"
    if not (exported / "index.html").is_file():
        raise SystemExit(f"Static export did not produce {exported / 'index.html'}")
    if PUBLIC.exists():
        shutil.rmtree(PUBLIC)
    shutil.copytree(exported, PUBLIC)


if __name__ == "__main__":
    main()
