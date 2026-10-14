"""CSP for the exported frontend, bound to the exact inline bootstrap scripts."""
from __future__ import annotations

import base64
import hashlib
import re
from html.parser import HTMLParser
from urllib.parse import urlsplit


class _Scripts(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.active = False
        self.parts: list[str] = []
        self.hashes: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "script":
            self.active = not any(key == "src" for key, _ in attrs)
            self.parts = []

    def handle_data(self, data: str) -> None:
        if self.active:
            self.parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self.active:
            digest = hashlib.sha256("".join(self.parts).encode()).digest()
            self.hashes.add("'sha256-" + base64.b64encode(digest).decode() + "'")
            self.active = False


def content_security_policy(html: bytes, supabase_url: str | None) -> str:
    scripts = _Scripts()
    scripts.feed(html.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n"))
    connect = "'self'"
    if supabase_url:
        try:
            parsed = urlsplit(supabase_url)
        except ValueError:
            parsed = urlsplit("")
        # Never turn malformed configuration into an extra CSP directive.
        if (parsed.scheme in {"https", "http"} and parsed.netloc
                and re.fullmatch(r"[A-Za-z0-9.\-:\[\]]+", parsed.netloc)):
            connect += f" {parsed.scheme}://{parsed.netloc}"
    return "; ".join([
        "default-src 'self'", "base-uri 'none'", "object-src 'none'", "frame-ancestors 'none'",
        "form-action 'self'", "script-src 'self' " + " ".join(sorted(scripts.hashes)),
        "script-src-attr 'none'", "style-src 'self' 'unsafe-inline'", "img-src 'self' data: blob:",
        "font-src 'self'", "worker-src 'self'", f"connect-src {connect}",
    ])
