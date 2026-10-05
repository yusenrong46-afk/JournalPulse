from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from journalpulse.journal_models import CreateJournalEntryRequest, JournalEntry


@pytest.mark.parametrize("text", ["", "  \n\t", "\u2003", "x" * 5001])
def test_journal_rejects_blank_or_oversized_writing(text: str):
    with pytest.raises(ValidationError):
        CreateJournalEntryRequest(text=text)


def test_journal_preserves_exact_writing_and_saved_entry_is_immutable():
    text = "  I enjoyed the rain.\n\nIt gave me a pause.  "
    request = CreateJournalEntryRequest(text=text)
    entry = JournalEntry(
        id=uuid4(),
        user_id=uuid4(),
        created_at=datetime.now(UTC),
        text=request.text,
    )
    assert entry.text == text
    with pytest.raises(ValidationError):
        entry.text = "Rewritten"
