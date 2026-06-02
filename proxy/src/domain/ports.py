"""Port/adapter interfaces for infrastructure dependencies.

These protocols define contracts that service layer uses without
importing infrastructure libraries. Concrete implementations are
in src/adapters/ layer.

python.md §1: Hexagonal architecture - dependencies point inward.
"""

from collections.abc import Sequence
from typing import Protocol, TypeVar
from uuid import UUID

T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")


class ScalarResult(Protocol):
    """Abstract query result — matches sqlalchemy.engine.Result API.

    Used by callers after ``db.execute(statement)`` to chain
    ``.scalars().all()`` or ``.scalar()``.
    """

    def scalars(self) -> "ScalarResult": ...

    def all(self) -> Sequence[object]: ...

    def scalar(self) -> object: ...


class AsyncSessionPort(Protocol):
    """Abstract async database session - service layer uses this.

    Concrete implementation: sqlmodel.ext.asyncio.session.AsyncSession
    This protocol decouples service from SQLModel library.
    """

    async def get(
        self, entity_type: type[T_co], ident: str | int | UUID, **kwargs: object
    ) -> T_co | None:
        """Get entity by primary key. Options can include eager loading."""
        ...

    async def execute(
        self, statement: object, **kwargs: object
    ) -> ScalarResult:
        """Execute a SQL statement and return a result with .scalars()."""
        ...

    async def flush(self) -> None:
        """Flush changes to database."""
        ...

    async def commit(self) -> None:
        """Commit transaction."""
        ...

    async def rollback(self) -> None:
        """Rollback transaction."""
        ...

    async def close(self) -> None:
        """Close session."""
        ...

    def add(self, instance: object) -> None:
        """Add instance to session."""
        ...

    def __aenter__(self) -> "AsyncSessionPort":
        """Async context manager entry."""
        ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Async context manager exit."""
        ...
