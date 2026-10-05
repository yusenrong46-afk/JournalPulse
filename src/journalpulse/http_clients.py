from collections.abc import Iterator
from contextlib import contextmanager

import httpx


@contextmanager
def managed_http_client(client: httpx.Client | None, *, timeout: float) -> Iterator[httpx.Client]:
    """Close transports we create, while leaving caller-owned transports reusable.

    Create the owned client only when an operation starts. A constructor may never
    be used, and closing inside a retry attempt would break subsequent attempts.
    """
    if client is not None:
        yield client
    else:
        with httpx.Client(timeout=timeout, follow_redirects=False) as owned:
            yield owned
