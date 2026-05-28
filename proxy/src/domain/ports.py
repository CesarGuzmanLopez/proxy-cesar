"""Port/adapter interfaces for infrastructure dependencies.

These protocols define contracts that service layer uses without
importing infrastructure libraries. Concrete implementations are
in src/adapters/ layer.

python.md §1: Hexagonal architecture - dependencies point inward.
"""

from typing import Protocol, TypeVar, Any


T = TypeVar("T")


class AsyncSessionPort(Protocol):
    """Abstract async database session - service layer uses this.

    Concrete implementation: sqlmodel.ext.asyncio.session.AsyncSession
    This protocol decouples service from SQLModel library.
    """

    async def get(self, entity_type: type[T], ident: Any, **kwargs) -> T | None:
        """Get entity by primary key. Options can include eager loading."""
        ...

    async def execute(self, statement: Any, **kwargs) -> Any:
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

    def add(self, instance: Any) -> None:
        """Add instance to session."""
        ...

    def __aenter__(self) -> "AsyncSessionPort":
        """Async context manager entry."""
        ...

    def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        ...
