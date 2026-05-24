"""Valkey implementation of the AffinityPort.

Key schema: conv:{conversation_id}:physical_model
TTL: 86400s (24 hours), configurable.
"""

import valkey.asyncio as valkey_async

from src.config.settings import Settings


class ValkeyAffinityAdapter:
    """Adapter implementing AffinityPort protocol using Valkey."""

    def __init__(self, client: valkey_async.Valkey) -> None:
        self._client = client

    async def get(self, conversation_id: str) -> str | None:
        key = f"conv:{conversation_id}:physical_model"
        return await self._client.get(key)

    async def set(
        self, conversation_id: str, physical_model: str, ttl_seconds: int = 86400
    ) -> None:
        key = f"conv:{conversation_id}:physical_model"
        await self._client.set(key, physical_model, ex=ttl_seconds)

    async def delete(self, conversation_id: str) -> None:
        key = f"conv:{conversation_id}:physical_model"
        await self._client.delete(key)


async def setup_valkey(settings: Settings) -> valkey_async.Valkey:
    """Create and verify Valkey client. Called during FastAPI lifespan startup."""
    client = valkey_async.from_url(
        settings.valkey_url, decode_responses=True
    )
    await client.ping()
    return client
