"""Provider-specific cache optimizations.

Sprint 7 §3: each provider has a different caching mechanism.
The proxy applies the appropriate strategy based on the physical model's provider.

- Anthropic: cache_control breakpoints (max 4)
- OpenAI/DeepSeek: automatic prefix caching (no action needed)
- Gemini: CachedContent via cache_id (documented limitation)
- Groq/Zhipu/Qwen/MiniMax/Ollama: no caching
"""

import logging
from copy import deepcopy

logger = logging.getLogger(__name__)

# Providers with known caching mechanisms
_PROVIDERS_WITH_CACHE = frozenset({"anthropic", "openai", "deepseek", "google"})
_PROVIDERS_WITH_CACHE_CONTROL = frozenset({"anthropic"})
_PROVIDERS_WITH_AUTO_CACHE = frozenset({"openai", "deepseek"})


# ── Anthropic cache_control ──────────────────────────────────────────────────


def apply_anthropic_cache_control(messages: list[dict]) -> list[dict]:
    """Add cache_control breakpoints to messages for Anthropic provider.

    Strategy (Sprint 7 §3.2):
      - Breakpoint 1: after system message (caches system prompt)
      - Breakpoint 2: at the last message before the final user query
        (caches conversation history prefix)
      - Breakpoint 3-4: reserved for future use

    Max 4 breakpoints per Anthropic's limits.
    Only works with messages in canonical order (system first).

    Returns a deep copy with cache_control annotations added.
    """
    modified = deepcopy(messages)
    breakpoints_placed = 0
    msg_count = len(modified)

    if msg_count < 2:
        return modified  # too few messages for meaningful breakpoints

    # Breakpoint 1: on the first system message
    for i, msg in enumerate(modified):
        if breakpoints_placed >= 4:
            break
        if msg.get("role") == "system" and breakpoints_placed == 0:
            msg["cache_control"] = {"type": "ephemeral"}
            breakpoints_placed += 1
            break  # system found, move to next breakpoint position

    # Breakpoint 2: on the last message before the final user query
    # In canonical order, the penultimate message is the last history entry.
    # We assume the last message is the new user query.
    if breakpoints_placed < 4 and msg_count >= 2:
        # Index of the last message before the final one
        bp_idx = msg_count - 2
        if bp_idx >= 0 and modified[bp_idx].get("role") != "system":
            modified[bp_idx]["cache_control"] = {"type": "ephemeral"}
            breakpoints_placed += 1
        logger.debug(
            "anthropic_cache_control placed=%d messages=%d",
            breakpoints_placed,
            len(modified),
        )

    return modified


def should_apply_cache_control(provider: str) -> bool:
    """Check if cache_control breakpoints should be applied for this provider."""
    return provider.lower() in _PROVIDERS_WITH_CACHE_CONTROL


def provider_supports_cache(provider: str) -> bool:
    """Check if provider has any caching support."""
    return provider.lower() in _PROVIDERS_WITH_CACHE


# ── Gemini CachedContent (documented limitation in Sprint 7) ──────────────────


async def manage_gemini_cache(
    conversation_id: str,
    valkey_client,
    system_prompt: str,
    tool_definitions: list[dict] | None,
    provider: str,
) -> str | None:
    """Manage Gemini CachedContent for a conversation.

    Creates a cache on first turn, reuses on subsequent turns.
    Returns the cache_id or None.

    Sprint 7 limitation: if LiteLLM does not support Gemini CachedContent
    directly, this returns None and the limitation is documented in
    proxy_metadata.
    """
    if provider.lower() != "google":
        return None

    cache_key = f"conv:{conversation_id}:gemini_cache_id"

    try:
        existing_cache_id = await valkey_client.get(cache_key)
        if existing_cache_id:
            return existing_cache_id
    except Exception:
        logger.warning("gemini_cache valkey_read_failed conv=%s", conversation_id)
        return None

    # LiteLLM may not support Gemini CachedContent directly.
    # Document as limitation — Sprint 7 §3.3
    logger.info(
        "gemini_cache not_supported_via_litellm conv=%s "
        "limitation=CachedContent_requires_direct_gemini_api",
        conversation_id,
    )
    return None


# ── Cache metadata extraction ────────────────────────────────────────────────


def build_cache_metadata(
    response: dict,
    provider: str,
    cache_optimization_applied: bool = False,
) -> dict:
    """Extract cache hit information from the provider response.

    Different providers report cache hits differently:
      - OpenAI/DeepSeek: usage.prompt_tokens_details.cached_tokens
      - Anthropic: usage.cache_read_input_tokens, usage.cache_creation_input_tokens
      - Others: may not report cache at all
    """
    metadata: dict = {
        "cache_optimization_applied": cache_optimization_applied,
        "provider": provider,
    }

    if not isinstance(response, dict):
        usage = {}
    else:
        usage = response.get("usage", {})

    if not usage:
        metadata["provider_cache_hit"] = False
        return metadata

    # OpenAI / DeepSeek format
    details = usage.get("prompt_tokens_details", {})
    if details and isinstance(details, dict):
        cached = details.get("cached_tokens", 0)
        if cached > 0:
            metadata["provider_cache_hit"] = True
            metadata["cached_tokens"] = cached
            metadata["total_prompt_tokens"] = usage.get("prompt_tokens", 0)
            metadata["estimated_savings_usd"] = round(
                (cached / 1000) * 0.0025,
                5,
            )
            return metadata

    # Anthropic format
    cache_read = usage.get("cache_read_input_tokens", 0)
    cache_write = usage.get("cache_creation_input_tokens", 0)
    if cache_read > 0 or cache_write > 0:
        metadata["provider_cache_hit"] = cache_read > 0
        metadata["cache_read_tokens"] = cache_read
        metadata["cache_write_tokens"] = cache_write
        if cache_read > 0:
            metadata["estimated_savings_usd"] = round(
                (cache_read / 1000) * 0.0025,
                5,
            )
        return metadata

    # No cache info found
    metadata["provider_cache_hit"] = False
    return metadata


def build_cache_destruction_metadata(
    previous_model: str,
    new_model: str,
    previous_cached_tokens: int = 0,
) -> dict:
    """Build cache destruction metadata when fallback changes the physical model.

    Sprint 7 §3.7: cache is destroyed on fallback — the proxy MUST report this clearly.
    """
    meta: dict = {
        "previous_cache_destroyed": True,
        "previous_model": previous_model,
        "new_model": new_model,
    }

    if previous_cached_tokens > 0:
        meta["previous_cached_tokens_lost"] = previous_cached_tokens
        meta["new_cache_starting"] = True
        meta["estimated_extra_cost_usd"] = round(
            (previous_cached_tokens / 1000) * 0.0025,
            5,
        )

    return meta
