"""Bind repository work to the account data revision observed before mutation submission."""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, cast
from uuid import UUID

if TYPE_CHECKING:
    from .persistence import Repository

# A repository may be shared by tests or callers. Keep this per invocation rather
# than mutating its instance, so simultaneous requests cannot overwrite the fence.
write_scope: ContextVar[tuple[UUID, int] | None] = ContextVar("journalpulse_write_scope", default=None)


class DeletedObjectIdentity(ValueError):
    """A logical creation ID has already been erased."""


class AccountDataErased(RuntimeError):
    """An authenticated request predates the latest account data erasure."""


def bind_repository(repository: Repository, user_id: UUID, revision: int | None) -> Repository:
    if revision is None:
        return repository

    scope: tuple[UUID, int] = (user_id, revision)

    class BoundRepository:
        def __getattr__(self, name: str) -> Any:
            attribute = getattr(repository, name)
            if not callable(attribute):
                return attribute

            def invoke(*args: Any, **kwargs: Any) -> Any:
                token = write_scope.set(scope)
                try:
                    return attribute(*args, **kwargs)
                finally:
                    write_scope.reset(token)

            return invoke

    return cast("Repository", BoundRepository())
