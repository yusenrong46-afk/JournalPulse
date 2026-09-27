"""Build the Next.js PWA into public/ so Vercel can serve it from the CDN.

The Python function keeps the API routes. Pages, the service worker, and the
manifest are static files. The anon key is public and is copied from the
server Supabase variables when the Next.js names were not set separately.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
PUBLIC = ROOT / "public"


def main() -> None:
    env = os.environ.copy()
    env["JOURNALPULSE_STATIC_EXPORT"] = "true"
    env.setdefault("NEXT_PUBLIC_API_BASE_URL", "")
    if not env.get("NEXT_PUBLIC_SUPABASE_URL") and env.get("SUPABASE_URL"):
        env["NEXT_PUBLIC_SUPABASE_URL"] = env["SUPABASE_URL"]
    if not env.get("NEXT_PUBLIC_SUPABASE_ANON_KEY") and env.get("SUPABASE_ANON_KEY"):
        env["NEXT_PUBLIC_SUPABASE_ANON_KEY"] = env["SUPABASE_ANON_KEY"]

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
