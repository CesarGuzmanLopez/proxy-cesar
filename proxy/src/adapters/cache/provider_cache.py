"""Provider-specific cache optimizations.

# Feature: each provider has a different caching mechanism.
The proxy applies the appropriate strategy based on the physical model's provider.

- OpenAI/DeepSeek/Groq: automatic prefix caching (no action needed)
- Ollama: no caching
"""

import functools
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)

# Providers with known caching mechanisms
# Z.ai/Zhipu: automatic prefix caching (OpenAI-compatible format, cached input ~18% of regular price)
# Groq: automatic prefix caching, 50% discount on cached tokens, 2-hour TTL
#   Supported models: openai/gpt-oss-20b, openai/gpt-oss-120b
_PROVIDERS_WITH_CACHE = frozenset({"openai", "deepseek", "groq", "anthropic"})
_PROVIDERS_WITH_CACHE_CONTROL = frozenset({"anthropic"})
# DeepSeek/Groq/OpenAI have AUTOMATIC prefix caching — no cache_control headers needed.
# DeepSeek: Context Caching on Disk (enabled by default, automatic prefix matching).
# Groq: automatic prefix caching, 50% discount, 2-hour TTL.
# OpenAI: automatic prompt caching.
# This is NOT a bug — the proxy correctly identifies which providers need explicit
# cache_control (Anthropic only) vs those with transparent automatic caching.
_PROVIDERS_WITH_AUTO_CACHE = frozenset({"openai", "deepseek", "groq"})


# ── Anthropic cache_control ──────────────────────────────────────────────────


def _ensure_content_list(msg: dict) -> dict:
    """Ensure message content is a list of content items (Anthropic format)."""
    content = msg.get("content", "")
    if isinstance(content, str):
        msg["content"] = [{"type": "text", "text": content}]
    return msg


def _add_cache_control_to_last_content(msg: dict) -> dict:
    """Add cache_control to the last content item of a message."""
    content = msg.get("content", [])
    if isinstance(content, list) and content:
        last = content[-1]
        if isinstance(last, dict):
            last["cache_control"] = {"type": "ephemeral"}
    return msg


def _place_breakpoint_on_system_msg(modified: list[dict]) -> int:
    """Place breakpoint 1 on the first system message. Returns breakpoints placed."""
    for i, msg in enumerate(modified):
        if msg.get("role") == "system":
            modified[i] = _add_cache_control_to_last_content(_ensure_content_list(msg))
            return 1
    return 0


def _place_breakpoint_before_last(modified: list[dict]) -> int:
    """Place breakpoint 2 on the message before the final user query. Returns 1 if placed."""
    msg_count = len(modified)
    if msg_count < 2:
        return 0
    bp_idx = msg_count - 2
    if bp_idx >= 0 and modified[bp_idx].get("role") != "system":
        modified[bp_idx] = _add_cache_control_to_last_content(
            _ensure_content_list(modified[bp_idx])
        )
        return 1
    return 0


def apply_anthropic_cache_control(messages: list[dict]) -> list[dict]:
    """Add cache_control breakpoints to messages for Anthropic provider.

    Strategy (Feature):
      - Breakpoint 1: after system message (caches system prompt)
      - Breakpoint 2: at the last message before the final user query
        (caches conversation history prefix)

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
    if len(modified) < 2:
        return modified

    breakpoints = 0

    # Breakpoint 1: on the first system message
    breakpoints += _place_breakpoint_on_system_msg(modified)

    # Breakpoint 2: on the last message before the final user query
    if breakpoints < 4:
        breakpoints += _place_breakpoint_before_last(modified)

    logger.debug(
        "anthropic_cache_control placed=%d messages=%d",
        breakpoints,
        len(modified),
    )

    return modified


@functools.lru_cache(maxsize=32)
def should_apply_cache_control(provider: str) -> bool:
    """Check if cache_control breakpoints should be applied for this provider."""
    return provider.lower() in _PROVIDERS_WITH_CACHE_CONTROL


@functools.lru_cache(maxsize=32)
def provider_supports_cache(provider: str) -> bool:
    """Check if provider has any caching support."""
    return provider.lower() in _PROVIDERS_WITH_CACHE


# ── Gemini CachedContent (deprecated — Google models removed) ──────────────


def manage_gemini_cache() -> None:
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
        cached = details.get("cached_tokens", 0) or 0
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

    Feature cache is destroyed on fallback — the proxy MUST report this clearly.
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
