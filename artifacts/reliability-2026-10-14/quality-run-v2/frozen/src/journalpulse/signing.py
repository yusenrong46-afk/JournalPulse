"""HMAC signatures for writes that carry server provenance.

The database verifies the signature over the exact payload text before storing policy,
model, safety, or catalog-card data, so a person holding their own session token cannot
write those records directly.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import UTC, datetime
from typing import Any
from uuid import UUID


def sign_text(text: str, key: str) -> str:
    return hmac.new(key.encode(), text.encode(), hashlib.sha256).hexdigest()


def signed_payload(purpose: str, user_id: UUID, body: dict[str, Any], key: str) -> dict[str, str]:
    """Return the RPC arguments {payload, signature} for one signed write."""
    envelope = {
        "purpose": purpose,
        "user_id": str(user_id),
        "issued_at": datetime.now(UTC).isoformat(),
        **body,
    }
    text = json.dumps(envelope, separators=(",", ":"), sort_keys=True)
    return {"payload": text, "signature": sign_text(text, key)}


def readiness_probe(key: str) -> dict[str, str]:
    probe = f"readiness:{datetime.now(UTC).isoformat()}"
    return {"probe": probe, "signature": sign_text(probe, key)}
