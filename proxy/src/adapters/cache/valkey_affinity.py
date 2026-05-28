"""Valkey implementation of the AffinityPort.

Key schema:
- conv:{conversation_id}:physical_model — pinned model
- affinity_metrics:{conversation_id}:{model} — failure counts
- affinity_last_use:{conversation_id} — last access timestamp

TTL: 86400s (24 hours) default, dynamic based on usage pattern.
"""

import json
import logging
import time
import valkey.asyncio as valkey_async

from src.config.settings import Settings

logger = logging.getLogger(__name__)


class ValkeyAffinityAdapter:
    """Adapter implementing AffinityPort protocol using Valkey.

    Features:
    - Dynamic TTL: extended if conversation stays active
    - Upgrade detection: should_upgrade() triggers model change
    - Failure tracking: records errors per model per conversation
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

    async def record_failure(
        self,
        conversation_id: str,
        model: str,
        error: str | None = None,
    ) -> None:
        """Record a failure for this model in this conversation."""
        key = f"affinity_metrics:{conversation_id}:{model}"

        try:
            metrics = await self._client.get(key)
            if metrics:
                data = json.loads(metrics)
            else:
                data = {"errors_1h": 0, "last_error": None}

            data["errors_1h"] = data.get("errors_1h", 0) + 1
            if error:
                data["last_error"] = error[:100]

            await self._client.set(
                key,
                json.dumps(data),
                ex=3600,  # 1 hour TTL
            )
        except Exception as e:
            logger.warning("affinity_record_failure_error model=%s error=%s", model, e)

    async def get_failure_count(
        self,
        conversation_id: str,
        model: str,
    ) -> int:
        """Get failure count for this model in this conversation (last 1h)."""
        key = f"affinity_metrics:{conversation_id}:{model}"

        try:
            metrics = await self._client.get(key)
            if metrics:
                data = json.loads(metrics)
                return data.get("errors_1h", 0)
        except Exception as e:
            logger.warning("affinity_get_failure_count_error model=%s error=%s", model, e)

        return 0

    async def should_upgrade(
        self,
        conversation_id: str,
        physical_model: str,
        context_window: int,
        input_tokens: int,
    ) -> bool:
        """Determine if pinned model should be upgraded.

        Triggers upgrade if:
        1. Input tokens exceed 70% of context window
        2. Model has failed 2+ times in last hour

        Args:
            conversation_id: Conversation ID
            physical_model: Currently pinned model
            context_window: Context window of pinned model
            input_tokens: Estimated input tokens

        Returns:
            True if should upgrade to next model, False otherwise
        """
        # Check 1: Input size exceeds 70% of context window
        if context_window > 0 and input_tokens > (context_window * 0.7):
            logger.debug(
                "affinity_upgrade_trigger input_exceeds_context "
                "conv=%s model=%s input=%d window=%d",
                str(conversation_id)[:12],
                physical_model,
                input_tokens,
                context_window,
            )
            return True

        # Check 2: Model has failed recently
        failure_count = await self.get_failure_count(conversation_id, physical_model)
        if failure_count >= 2:
            logger.debug(
                "affinity_upgrade_trigger too_many_failures "
                "conv=%s model=%s failures=%d",
                str(conversation_id)[:12],
                physical_model,
                failure_count,
            )
            return True

        return False


async def setup_valkey(settings: Settings) -> valkey_async.Valkey:
    """Create and verify Valkey client. Called during FastAPI lifespan startup."""
    client = valkey_async.from_url(settings.valkey_url, decode_responses=True)
    try:
        await client.ping()
    except Exception:
        await client.aclose()
        raise  # propagate — FastAPI lifespan will handle it
    return client
