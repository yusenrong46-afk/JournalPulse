"""Exercise 2: every gate runs before the model can be called.

``answer`` decides what to do with one chat message. Implement it so that, in this order:

1. a message that ``looks_risky`` gets ``Reply("support", SUPPORT_MESSAGE)``, whatever else is
   true: no consent check, no rate-limit charge, no model call;
2. without consent (``consent=False``) the reply is ``Reply("simple", SIMPLE_MESSAGE)``, with no
   rate-limit charge and no model call;
3. otherwise one unit of the rate limit is taken with ``limiter.take()``; if it returns False the
   reply is ``Reply("limited", LIMIT_MESSAGE)`` and the model is not called. If the limiter
   raises ``LimiterUnavailable`` the answer fails closed: ``Reply("unavailable",
   UNAVAILABLE_MESSAGE)`` and no model call;
4. only then ``Reply("ai", model(message))``.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

SUPPORT_MESSAGE = "You deserve support right now. In Canada, call or text 9-8-8."
SIMPLE_MESSAGE = "Simple mode: want to name one small thing that might help?"
LIMIT_MESSAGE = "Let's pause for a minute; you can try again shortly."
UNAVAILABLE_MESSAGE = "Nothing new was saved. Please try again in a moment."

RISK = re.compile(r"\b(kill myself|end my life|hurt myself)\b", re.IGNORECASE)


class LimiterUnavailable(Exception):
    """The shared rate-limit counter could not be reached."""


class Limiter(Protocol):
    def take(self) -> bool: ...


@dataclass(frozen=True)
class Reply:
    kind: str
    text: str


def looks_risky(message: str) -> bool:
    return RISK.search(message) is not None


def answer(message: str, *, consent: bool, limiter: Limiter, model: Callable[[str], str]) -> Reply:
    raise NotImplementedError("exercise 2: answer")
