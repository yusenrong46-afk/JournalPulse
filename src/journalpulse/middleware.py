from __future__ import annotations

import json
import logging
import re
import threading
import time
from uuid import uuid4

from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger("uvicorn.error")
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{8,80}$")


class RequestBodyTooLarge(Exception):
    pass


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp, *, max_request_bytes: int) -> None:
        self.app = app
        self.max_request_bytes = max_request_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        supplied_id = headers.get(b"x-request-id", b"").decode("ascii", errors="ignore")
        request_id = supplied_id if REQUEST_ID_PATTERN.fullmatch(supplied_id) else str(uuid4())
        scope.setdefault("state", {})["request_id"] = request_id
        content_length = headers.get(b"content-length")
        if content_length:
            try:
                if int(content_length) > self.max_request_bytes:
                    await self._too_large(send, request_id)
                    return
            except ValueError:
                await self._bad_length(send, request_id)
                return

        started = time.perf_counter()
        status_code = 500
        received_bytes = 0

        async def receive_with_limit() -> Message:
            nonlocal received_bytes
            message = await receive()
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > self.max_request_bytes:
                    raise RequestBodyTooLarge
            return message

        async def send_with_context(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                response_headers = list(message.get("headers", []))
                response_headers.extend(
                    [
                        (b"x-request-id", request_id.encode()),
                        (b"x-content-type-options", b"nosniff"),
                        (b"referrer-policy", b"no-referrer"),
                        (b"permissions-policy", b"camera=(), microphone=(), geolocation=()"),
                    ]
                )
                message = {**message, "headers": response_headers}
            await send(message)

        try:
            await self.app(scope, receive_with_limit, send_with_context)
        except RequestBodyTooLarge:
            status_code = 413
            await self._too_large(send, request_id)
        finally:
            logger.info(
                json.dumps(
                    {
                        "event": "http_request",
                        "request_id": request_id,
                        "method": scope.get("method"),
                        "path": scope.get("path"),
                        "status": status_code,
                        "latency_ms": round((time.perf_counter() - started) * 1000),
                    },
                    separators=(",", ":"),
                )
            )

    @staticmethod
    async def _too_large(send: Send, request_id: str) -> None:
        body = json.dumps(
            {
                "detail": "Request body is too large",
                "code": "request_too_large",
                "request_id": request_id,
            }
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"x-request-id", request_id.encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    @staticmethod
    async def _bad_length(send: Send, request_id: str) -> None:
        body = json.dumps(
            {"detail": "Invalid content length", "code": "invalid_content_length"}
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 400,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"x-request-id", request_id.encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


class SlidingWindowRateLimiter:
    def __init__(self, *, limit: int, window_seconds: float = 60.0) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._events: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str, *, now: float | None = None) -> tuple[bool, int]:
        current = time.monotonic() if now is None else now
        cutoff = current - self.window_seconds
        with self._lock:
            events = [timestamp for timestamp in self._events.get(key, []) if timestamp > cutoff]
            if len(events) >= self.limit:
                retry_after = max(1, round(events[0] + self.window_seconds - current))
                self._events[key] = events
                return False, retry_after
            events.append(current)
            self._events[key] = events
        return True, 0
