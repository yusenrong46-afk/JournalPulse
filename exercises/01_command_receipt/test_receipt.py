from __future__ import annotations

import pytest
from receipt import Chat, Command, RequestIdReused, StaleCommand, apply_command, apply_reply


def test_a_new_command_is_applied_once_and_returns_a_receipt() -> None:
    chat = Chat()
    receipt = apply_command(chat, Command("r1", 0, "I went for a walk"))
    assert receipt.turn == 1 and chat.turn == 1
    assert chat.messages == ["I went for a walk"]


def test_a_retried_command_returns_the_first_receipt_and_changes_nothing() -> None:
    chat = Chat()
    first = apply_command(chat, Command("r1", 0, "I went for a walk"))
    again = apply_command(chat, Command("r1", 0, "I went for a walk"))
    assert again == first
    assert chat.turn == 1 and chat.messages == ["I went for a walk"]


def test_reusing_a_request_id_for_different_text_is_refused() -> None:
    chat = Chat()
    apply_command(chat, Command("r1", 0, "I went for a walk"))
    with pytest.raises(RequestIdReused):
        apply_command(chat, Command("r1", 1, "Something else"))
    assert chat.messages == ["I went for a walk"]


def test_a_command_prepared_against_an_older_turn_is_stale() -> None:
    chat = Chat()
    apply_command(chat, Command("r1", 0, "First"))
    with pytest.raises(StaleCommand):
        apply_command(chat, Command("r2", 0, "Written in an old tab"))
    assert chat.turn == 1 and chat.messages == ["First"]


def test_a_late_reply_for_an_older_turn_is_ignored() -> None:
    chat = Chat()
    apply_command(chat, Command("r1", 0, "First"))
    assert apply_reply(chat, 1, "Reply to first") is True
    apply_command(chat, Command("r2", 1, "Second"))
    assert apply_reply(chat, 1, "A slow duplicate reply to first") is False
    assert chat.messages == ["First", "Reply to first", "Second"]
