"""Affinity port (Protocol) for physical model pinning.

python.md §5.2: Use Protocol for ports/abstractions.
"""

from typing import Protocol


class AffinityPort(Protocol):
    """Interface for storing/retrieving physical model affinity per conversation."""

    async def get(self, conversation_id: str) -> str | None:
        """Get the pinned physical model. Returns None if not set or expired."""
        ...

    async def set(
        self, conversation_id: str, physical_model: str, ttl_seconds: int = 86400
    ) -> None:
        """Pin a physical model to a conversation with a TTL."""
        ...

    async def delete(self, conversation_id: str) -> None:
        """Remove the affinity pin."""
        ...
