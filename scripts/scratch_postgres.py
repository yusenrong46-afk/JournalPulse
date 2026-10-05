"""Keep destructive test helpers on a local PostgreSQL instance."""

from __future__ import annotations

import os
from urllib.parse import parse_qs, unquote, urlsplit


def require_local_postgres_dsn(configured: str | None) -> None:
    """Reject remote routing before a scratch reset, without echoing credentials.

    libpq query parameters and PGHOSTADDR can override a URI's apparent host,
    so checking just ``urlsplit(...).hostname`` does not protect hosted data.
    Local peer authentication and absolute Unix socket directories stay usable.
    """
    hosts: list[str] = []
    service = bool(os.getenv("PGSERVICE"))
    try:
        if configured:
            parts = urlsplit(configured)
            if parts.scheme not in {"postgres", "postgresql"}:
                raise ValueError("Unsupported PostgreSQL URI")
            if parts.hostname:
                hosts.append(unquote(parts.hostname))
            query = parse_qs(parts.query, keep_blank_values=True)
            for parameter in ("host", "hostaddr"):
                hosts.extend(query.get(parameter, []))
            service = service or bool(query.get("service"))
    except ValueError:
        raise SystemExit("Scratch PostgreSQL helpers require a valid local PostgreSQL URI.") from None

    hosts.extend(os.environ[name] for name in ("PGHOST", "PGHOSTADDR") if os.getenv(name))
    if service or any(
        host and host not in {"localhost", "127.0.0.1", "::1"} and not host.startswith("/")
        for value in hosts for host in value.split(",")
    ):
        raise SystemExit("Scratch PostgreSQL helpers are restricted to local PostgreSQL.")
