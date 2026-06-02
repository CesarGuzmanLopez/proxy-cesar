"""Valkey implementation of the AffinityPort.

Key schema:
- conv:{conversation_id}:physical_model — pinned model
- affinity_metrics:{conversation_id}:{model} — failure counts
- affinity_last_use:{conversation_id} — last access timestamp

TTL: 86400s (24 hours) default, dynamic based on usage pattern.
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
        """Get pinned physical model, update last access time."""
        key = f"conv:{conversation_id}:physical_model"
        model = await self._client.get(key)

        if model:
            # Update last access timestamp for TTL extension
            await self._client.set(
                f"affinity_last_use:{conversation_id}",
                str(int(time.time())),
                ex=86400,  # Keep timestamp for 24h
            )

        return model

    async def set(
        self, conversation_id: str, physical_model: str, ttl_seconds: int = 86400
    ) -> None:
        """Set pinned model with dynamic TTL."""
        key = f"conv:{conversation_id}:physical_model"

        # Calculate dynamic TTL: if conversation is active (recent access),
        # extend to 72h; otherwise use 24h
        last_use_key = f"affinity_last_use:{conversation_id}"
        last_use = await self._client.get(last_use_key)

        if last_use:
            last_use_ts = int(last_use)
            now = int(time.time())
            time_since_last = now - last_use_ts

            # If conversation had activity in last 4h, extend to 72h
            if time_since_last < 14400:  # 4 hours
                ttl_seconds = 259200  # 72 hours

        await self._client.set(key, physical_model, ex=ttl_seconds)
        await self._client.set(
            last_use_key,
            str(int(time.time())),
            ex=86400,
        )

    async def delete(self, conversation_id: str) -> None:
        """Remove pinned model affinity."""
        key = f"conv:{conversation_id}:physical_model"
        await self._client.delete(key)

    async def get_key_slot(self, conversation_id: str) -> int:
        """Get pinned key slot (1-based). Returns 1 if not set."""
        key = f"conv:{conversation_id}:key_slot"
        raw = await self._client.get(key)
        if raw is not None:
            try:
                return int(raw)
            except (ValueError, TypeError):
                logger.warning("affinity_key_slot_corrupt conv=%s raw=%s", conversation_id, raw)
                return 1
        return 1

    async def set_key_slot(
        self, conversation_id: str, slot: int, ttl_seconds: int = 86400
    ) -> None:
        """Pin a key slot to a conversation with TTL (sliding window)."""
        key = f"conv:{conversation_id}:key_slot"
        await self._client.set(key, str(slot), ex=ttl_seconds)

    # Lua script for atomic read-modify-write of failure metrics.
    # INCR creates the key with value 1 if it doesn't exist (no TOCTOU).
    # EXPIRE only on count == 1 ensures the 1h window starts at first failure
    # and is NOT reset by subsequent failures.
    _LUA_RECORD_FAILURE = """
local key = KEYS[1]
local err = ARGV[1]
local ttl = tonumber(ARGV[2])
local count = redis.call('INCR', key)
if count == 1 then
    redis.call('EXPIRE', key, ttl)
end
if err and err ~= '' then
    local ekey = key .. ':last_error'
    redis.call('SET', ekey, err)
    redis.call('EXPIRE', ekey, ttl)
end
return count
"""

    async def record_failure(
        self,
        conversation_id: str,
        model: str,
        error: str | None = None,
    ) -> None:
        """Record a failure for this model in this conversation.

        Uses an atomic Lua script to avoid TOCTOU races on the error counter.
        TTL is only set on the first failure (count == 1) so subsequent failures
        within the window don't extend it.
        """
        key = f"affinity_metrics:{conversation_id}:{model}"
        try:
            await self._client.eval(
                self._LUA_RECORD_FAILURE,
                1,           # number of keys
                key,         # KEYS[1]
                error[:100] if error else "",
                3600,        # TTL in seconds
            )
        except Exception as e:
            logger.warning("affinity_record_failure_error model=%s error=%s", model, e)


async def setup_valkey(settings: Settings) -> valkey_async.Valkey:
    """Create and verify Valkey client. Called during FastAPI lifespan startup."""
    client = valkey_async.from_url(settings.valkey_url, decode_responses=True)
    try:
        await client.ping()
    except Exception:
        await client.aclose()
        raise  # propagate — FastAPI lifespan will handle it
    return client
