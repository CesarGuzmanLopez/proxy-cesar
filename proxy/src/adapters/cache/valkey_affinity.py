"""Valkey affinity adapter — Valkey removed, graceful degradation.

Stores client if provided (e.g., fakeredis in tests). When client is None
(production), all methods are no-ops and affinity is per-request only.
"""

import logging

logger = logging.getLogger(__name__)


class ValkeyAffinityAdapter:
    """Adapter implementing AffinityPort protocol.

    In production (client=None) all methods are no-ops — affinity is
    per-request only. When a client is provided (e.g., fakeredis in tests)
    it performs real get/set operations for multi-turn affinity.
    """

    def __init__(self, client=None):
        self._client = client

    async def get(self, conversation_id: str) -> str | None:
        if self._client is None:
            return None
        key = f"conv:{conversation_id}:physical_model"
        return await self._client.get(key)

    async def set(self, conversation_id: str, physical_model: str, ttl_seconds: int = 86400) -> None:
        if self._client is None:
            return
        key = f"conv:{conversation_id}:physical_model"
        await self._client.set(key, physical_model, ex=ttl_seconds)

    async def delete(self, conversation_id: str) -> None:
        if self._client is None:
            return
        key = f"conv:{conversation_id}:physical_model"
        await self._client.delete(key)

    async def get_key_slot(self, conversation_id: str) -> int:
        if self._client is None:
            return 1
        key = f"conv:{conversation_id}:key_slot"
        raw = await self._client.get(key)
        if raw is not None:
            try:
                return int(raw)
            except (ValueError, TypeError):
                logger.warning("affinity_key_slot_corrupt conv=%s raw=%s", conversation_id, raw)
                return 1
        return 1

    async def set_key_slot(self, conversation_id: str, slot: int, ttl_seconds: int = 86400) -> None:
        if self._client is None:
            return
        key = f"conv:{conversation_id}:key_slot"
        await self._client.set(key, str(slot), ex=ttl_seconds)

    async def record_failure(self, conversation_id: str, model: str, error: str | None = None) -> None:
        if self._client is None:
            return
        key = f"affinity_metrics:{conversation_id}:{model}"
        try:
            await self._client.incr(key)
            await self._client.expire(key, 3600)
        except Exception as e:
            logger.warning("affinity_record_failure_error model=%s error=%s", model, e)


async def setup_valkey(settings) -> None:
    """No-op — Valkey was removed."""
    logger.warning("setup_valkey called but Valkey is no longer available")
    return None
