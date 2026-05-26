"""Provider-specific cache optimizations.

Sprint 7 §3: each provider has a different caching mechanism.
The proxy applies the appropriate strategy based on the physical model's provider.

- OpenAI/DeepSeek/Groq: automatic prefix caching (no action needed)
- Ollama: no caching
"""

import logging
from copy import deepcopy

logger = logging.getLogger(__name__)

# Providers with known caching mechanisms
# Z.ai/Zhipu: automatic prefix caching (OpenAI-compatible format, cached input ~18% of regular price)
# Groq: automatic prefix caching, 50% discount on cached tokens, 2-hour TTL
#   Supported models: openai/gpt-oss-20b, openai/gpt-oss-120b
_PROVIDERS_WITH_CACHE = frozenset({"openai", "deepseek", "groq", "anthropic"})
_PROVIDERS_WITH_CACHE_CONTROL = frozenset({"anthropic"})
_PROVIDERS_WITH_AUTO_CACHE = frozenset({"openai", "deepseek", "groq"})


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

    LiteLLM expects cache_control on CONTENT ITEMS, not at the message level:
      {"role": "user", "content": [
          {"type": "text", "text": "...", "cache_control": {"type": "ephemeral"}}
      ]}

    Messages with string content are converted to this format automatically.

    Returns a deep copy with cache_control annotations added.
    """
    modified = deepcopy(messages)
    breakpoints_placed = 0
    msg_count = len(modified)

    if msg_count < 2:
        return modified

    # Helper: ensure content is a list of content items (Anthropic format)
    def _ensure_content_list(msg: dict) -> dict:
        content = msg.get("content", "")
        if isinstance(content, str):
            msg["content"] = [{"type": "text", "text": content}]
        elif isinstance(content, list):
            pass  # already in Anthropic format
        return msg

    # Helper: add cache_control to the last content item
    def _add_cache_control_to_last_content(msg: dict) -> dict:
        content = msg.get("content", [])
        if isinstance(content, list) and content:
            last = content[-1]
            if isinstance(last, dict):
                last["cache_control"] = {"type": "ephemeral"}
        return msg

    # Breakpoint 1: on the first system message
    for i, msg in enumerate(modified):
        if breakpoints_placed >= 4:
            break
        if msg.get("role") == "system" and breakpoints_placed == 0:
            msg = _ensure_content_list(msg)
            msg = _add_cache_control_to_last_content(msg)
            modified[i] = msg
            breakpoints_placed += 1
            break

    # Breakpoint 2: on the last message before the final user query
    if breakpoints_placed < 4 and msg_count >= 2:
        bp_idx = msg_count - 2
        if bp_idx >= 0 and modified[bp_idx].get("role") != "system":
            modified[bp_idx] = _ensure_content_list(modified[bp_idx])
            modified[bp_idx] = _add_cache_control_to_last_content(modified[bp_idx])
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


# ── Gemini CachedContent (deprecated — Google models removed) ──────────────


async def manage_gemini_cache(
    conversation_id: str,
    valkey_client,
    system_prompt: str,
    tool_definitions: list[dict] | None,
    provider: str,
) -> str | None:
    """Stub — Google models have been removed from the proxy configuration."""
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
