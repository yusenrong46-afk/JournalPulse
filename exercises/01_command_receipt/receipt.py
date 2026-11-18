"""Exercise 1: an idempotent command receipt that ignores a stale reply.

A chat keeps a list of saved messages and a ``turn`` number that rises by one on every
accepted command. Browsers retry (a flaky network sends the same command twice) and an AI
reply can arrive late (after the person already saved something newer). Implement the two
functions so that:

1. ``apply_command``: a command whose ``request_id`` was seen before returns the receipt
   recorded the first time and changes nothing, but the same ``request_id`` with different
   text raises ``RequestIdReused``. A command whose ``expected_turn`` is not the chat's
   current turn is stale: raise ``StaleCommand`` and change nothing. Otherwise append the text,
   advance the turn, record and return a receipt.
2. ``apply_reply``: an AI reply written for ``for_turn`` is stored only if the chat is still at
   that turn; otherwise it is ignored (return False) and nothing changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class StaleCommand(Exception):
    """The command was prepared against an older turn."""


class RequestIdReused(Exception):
    """The same request id was sent with a different command."""


@dataclass(frozen=True)
class Command:
    request_id: str
    expected_turn: int
    text: str


@dataclass(frozen=True)
class Receipt:
    request_id: str
    turn: int  # the chat's turn after the command was applied
    text: str


@dataclass
class Chat:
    turn: int = 0
    messages: list[str] = field(default_factory=list)
    receipts: dict[str, Receipt] = field(default_factory=dict)


def apply_command(chat: Chat, command: Command) -> Receipt:
    raise NotImplementedError("exercise 1: apply_command")


def apply_reply(chat: Chat, for_turn: int, reply: str) -> bool:
    raise NotImplementedError("exercise 1: apply_reply")
