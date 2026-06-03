"""Valkey affinity adapter — removed (Valkey no longer available).

All methods are no-ops. Affinity falls back to per-request model selection.
"""

import logging

logger = logging.getLogger(__name__)


class ValkeyAffinityAdapter:
    """Stub — Valkey was removed. No affinity persistence."""

    def __init__(self, client=None):
        pass

    async def get(self, conversation_id: str) -> str | None:
        return None

    async def set(self, conversation_id: str, physical_model: str, ttl_seconds: int = 86400) -> None:
        pass

    async def delete(self, conversation_id: str) -> None:
        pass

    async def get_key_slot(self, conversation_id: str) -> int:
        return 1

    async def set_key_slot(self, conversation_id: str, slot: int, ttl_seconds: int = 86400) -> None:
        pass

    async def record_failure(self, conversation_id: str, model: str, error: str | None = None) -> None:
        pass


async def setup_valkey(settings) -> None:
    """No-op — Valkey was removed."""
    logger.warning("setup_valkey called but Valkey is no longer available")
    return None
