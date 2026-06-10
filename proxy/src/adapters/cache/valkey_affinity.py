"""Valkey affinity adapter — persistent multi-turn model affinity.

Stores pinned physical models and failure metrics. Required for content
delegation (PDF extraction, image description). Proxy fails to start if
Valkey is unavailable.
"""

import logging
import time

import valkey.asyncio as valkey_async

from src.config.settings import Settings

logger = logging.getLogger(__name__)


class ValkeyAffinityAdapter:
    """Adapter implementing AffinityPort protocol using Valkey.

    Philosophy: User chooses pseudo-model, proxy respects it and pins the physical model.
    Only change model on FAILURE (via fallback), never on size/capacity.

    Features:
    - Dynamic TTL: extended if conversation stays active (sliding window)
    - Failure tracking: records errors to inform fallback decisions
    - Affinity invalidation: removed only if parallel tools incompatible
    """

    def __init__(self, client: valkey_async.Valkey) -> None:
        self._client = client

    async def get(self, conversation_id: str) -> str | None:
        key = f"conv:{conversation_id}:physical_model"
        model = await self._client.get(key)
        if model:
            await self._client.set(
                f"affinity_last_use:{conversation_id}",
                str(int(time.time())),
                ex=86400,
            )
        return model

    async def set(self, conversation_id: str, physical_model: str, ttl_seconds: int = 86400) -> None:
        key = f"conv:{conversation_id}:physical_model"
        await self._client.set(key, physical_model, ex=ttl_seconds)

    async def delete(self, conversation_id: str) -> None:
        key = f"conv:{conversation_id}:physical_model"
        await self._client.delete(key)

    async def get_key_slot(self, conversation_id: str) -> int:
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
        key = f"conv:{conversation_id}:key_slot"
        await self._client.set(key, str(slot), ex=ttl_seconds)

    async def record_failure(self, conversation_id: str, model: str, error: str | None = None) -> None:
        key = f"affinity_metrics:{conversation_id}:{model}"
        try:
            await self._client.incr(key)
            await self._client.expire(key, 3600)
        except Exception as e:
            logger.warning("affinity_record_failure_error model=%s error=%s", model, e)


async def setup_valkey(settings) -> valkey_async.Valkey:
    """Connect to Valkey. Raises ConnectionError if unavailable."""
    url = settings.valkey_url
    logger.info("valkey_connecting url=%s", url)
    client = valkey_async.Valkey.from_url(url, decode_responses=True)
    try:
        await client.ping()
        logger.info("valkey_connected url=%s", url)
        return client
    except Exception as exc:
        raise ConnectionError(
            f"Valkey is REQUIRED but unavailable at {url}: {exc}. "
            "Start Valkey with: redis-server --port 6380 --daemonize yes"
        ) from exc
