"""Port/adapter interfaces for infrastructure dependencies.

These protocols define contracts that service layer uses without
importing infrastructure libraries. Concrete implementations are
in src/adapters/ layer.

python.md §1: Hexagonal architecture - dependencies point inward.
"""

from collections.abc import Coroutine
from typing import Protocol, TypeVar

T_co = TypeVar("T_co", covariant=True)


class AsyncSessionPort(Protocol):
    """Abstract async database session - service layer uses this.

    Concrete implementation: sqlmodel.ext.asyncio.session.AsyncSession
    This protocol decouples service from SQLModel library.
    """

    async def get(
        self, entity_type: type[T_co], ident: str | int, **kwargs: object
    ) -> T_co | None:
        """Get entity by primary key. Options can include eager loading."""
        ...

    async def execute(
        self, statement: object, **kwargs: object
    ) -> Coroutine[None, None, object]:
        """Execute a SQL statement."""
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
