from __future__ import annotations

import pytest
from owned import NotFound, OwnedEntries, Store


def two_people() -> tuple[Store, OwnedEntries, OwnedEntries]:
    store = Store()
    return store, OwnedEntries(store, "ana"), OwnedEntries(store, "ben")


def test_each_person_sees_only_their_own_entries() -> None:
    _, ana, ben = two_people()
    first = ana.save("Ana's first entry")
    ben.save("Ben's entry")
    second = ana.save("Ana's second entry")
    assert ana.list() == [(first, "Ana's first entry"), (second, "Ana's second entry")]
    assert [text for _, text in ben.list()] == ["Ben's entry"]


def test_someone_elses_entry_looks_exactly_like_a_missing_one() -> None:
    _, ana, ben = two_people()
    entry = ana.save("Private")
    assert ana.get(entry) == "Private"
    with pytest.raises(NotFound) as other:
        ben.get(entry)
    with pytest.raises(NotFound) as missing:
        ben.get(999)
    assert str(other.value) == str(missing.value)


def test_delete_cannot_remove_another_persons_entry() -> None:
    store, ana, ben = two_people()
    entry = ana.save("Private")
    with pytest.raises(NotFound):
        ben.delete(entry)
    assert ana.get(entry) == "Private" and len(store.rows) == 1
    ana.delete(entry)
    assert ana.list() == []
    with pytest.raises(NotFound):
        ana.get(entry)


def test_the_owner_always_comes_from_the_binding() -> None:
    store, ana, _ = two_people()
    entry = ana.save("Mine")
    assert store.rows[entry].owner == "ana"
