from __future__ import annotations

import pytest
from gates import (
    LIMIT_MESSAGE,
    SIMPLE_MESSAGE,
    SUPPORT_MESSAGE,
    UNAVAILABLE_MESSAGE,
    LimiterUnavailable,
    Reply,
    answer,
)


def failing_model(message: str) -> str:
    raise AssertionError("the model must not be called here")


class CountingLimiter:
    def __init__(self, allow: bool = True, broken: bool = False) -> None:
        self.allow, self.broken, self.taken = allow, broken, 0

    def take(self) -> bool:
        self.taken += 1
        if self.broken:
            raise LimiterUnavailable
        return self.allow


def test_risky_words_get_support_before_anything_else() -> None:
    limiter = CountingLimiter()
    reply = answer("Some days I want to end my life", consent=True, limiter=limiter, model=failing_model)
    assert reply == Reply("support", SUPPORT_MESSAGE)
    assert limiter.taken == 0


def test_without_consent_the_model_is_never_called_and_nothing_is_charged() -> None:
    limiter = CountingLimiter()
    reply = answer("Rough day", consent=False, limiter=limiter, model=failing_model)
    assert reply == Reply("simple", SIMPLE_MESSAGE)
    assert limiter.taken == 0


def test_over_the_limit_the_model_is_not_called() -> None:
    limiter = CountingLimiter(allow=False)
    reply = answer("Rough day", consent=True, limiter=limiter, model=failing_model)
    assert reply == Reply("limited", LIMIT_MESSAGE)
    assert limiter.taken == 1


def test_a_broken_limiter_fails_closed() -> None:
    limiter = CountingLimiter(broken=True)
    reply = answer("Rough day", consent=True, limiter=limiter, model=failing_model)
    assert reply == Reply("unavailable", UNAVAILABLE_MESSAGE)


@pytest.mark.parametrize("consent", [True, False])
def test_support_wins_even_without_consent(consent: bool) -> None:
    reply = answer("I might hurt myself", consent=consent, limiter=CountingLimiter(), model=failing_model)
    assert reply.kind == "support"


def test_the_model_is_called_once_when_every_gate_passes() -> None:
    calls: list[str] = []

    def model(message: str) -> str:
        calls.append(message)
        return "That sounds like a lot."

    limiter = CountingLimiter()
    reply = answer("Rough day", consent=True, limiter=limiter, model=model)
    assert reply == Reply("ai", "That sounds like a lot.")
    assert calls == ["Rough day"] and limiter.taken == 1
