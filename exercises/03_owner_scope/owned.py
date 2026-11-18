"""Exercise 3: owner-scoped data access.

``Store`` holds every person's journal entries. ``OwnedEntries`` is the only way the app
touches it, and it is bound to one owner. Implement it so that:

* ``save(text)`` stores a new entry for the bound owner and returns its id (owner is never taken
  from the caller);
* ``get(entry_id)`` returns the entry's text, or raises ``NotFound`` when the id does not exist
  **or belongs to someone else**: the two cases must be indistinguishable;
* ``list()`` returns only the owner's entries, oldest first, as ``(id, text)`` pairs;
* ``delete(entry_id)`` removes the owner's entry, or raises ``NotFound`` exactly as ``get`` does,
  and never removes another person's entry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count


class NotFound(Exception):
    """No entry with this id is visible to this owner (it may not exist at all)."""


@dataclass
class Row:
    owner: str
    text: str


@dataclass
class Store:
    rows: dict[int, Row] = field(default_factory=dict)
    ids: count = field(default_factory=lambda: count(1))


class OwnedEntries:
    def __init__(self, store: Store, owner: str) -> None:
        self.store = store
        self.owner = owner

    def save(self, text: str) -> int:
        raise NotImplementedError("exercise 3: save")

    def get(self, entry_id: int) -> str:
        raise NotImplementedError("exercise 3: get")

    def list(self) -> list[tuple[int, str]]:
        raise NotImplementedError("exercise 3: list")

    def delete(self, entry_id: int) -> None:
        raise NotImplementedError("exercise 3: delete")
