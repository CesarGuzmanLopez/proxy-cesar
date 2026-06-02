"""Tests for provider caching — Anthropic, DeepSeek, and compaction caching.

Verifies:
  1. Anthropic cache_control placed on content items (not message level)
  2. DeepSeek automatic prefix caching via canonical ordering
  3. Cache metadata extraction (both Anthropic and DeepSeek formats)
  4. Cache destruction on fallback
  5. Compaction chunking preserves shared prefix for caching
  6. Canonical message ordering
"""

import sys

sys.path.insert(0, "src")

# ── Anthropic cache_control ────────────────────────────────────────────────


def test_anthropic_cache_control_on_system():
    """Breakpoint 1: system message gets cache_control on its last content item."""
    from adapters.cache.provider_cache import apply_anthropic_cache_control

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
    ]
    result = apply_anthropic_cache_control(messages)

    sys_msg = result[0]
    assert isinstance(sys_msg["content"], list), "system content should be list"
    last_item = sys_msg["content"][-1]
    assert last_item.get("cache_control") == {"type": "ephemeral"}, \
        f"expected cache_control on system content, got {last_item}"


def test_anthropic_cache_control_on_penultimate():
    """Breakpoint 2: penultimate message gets cache_control on content item."""
    from adapters.cache.provider_cache import apply_anthropic_cache_control

    messages = [
        {"role": "system", "content": "You are a bot."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "How are you?"},
    ]
    result = apply_anthropic_cache_control(messages)

    penultimate = result[2]  # assistant's "Hello"
    assert penultimate["role"] == "assistant"
    assert isinstance(penultimate["content"], list), "content should be list"
    last_item = penultimate["content"][-1]
    assert last_item.get("cache_control") == {"type": "ephemeral"}, \
        f"expected cache_control on penultimate, got {last_item}"


def test_anthropic_cache_control_less_than_two():
    """Fewer than 2 messages: no breakpoints placed."""
    from adapters.cache.provider_cache import apply_anthropic_cache_control

    result = apply_anthropic_cache_control([{"role": "user", "content": "Hi"}])
    assert len(result) == 1
    content = result[0].get("content", "")
    assert isinstance(content, str), "single message content should remain string"


def test_anthropic_cache_control_content_list_already():
    """Messages already in content-list format should still get cache_control."""
    from adapters.cache.provider_cache import apply_anthropic_cache_control

    messages = [
        {"role": "system", "content": [
            {"type": "text", "text": "You are a bot."},
        ]},
        {"role": "user", "content": [
            {"type": "text", "text": "Hello"},
        ]},
    ]
    result = apply_anthropic_cache_control(messages)
    last_item = result[0]["content"][-1]
    assert last_item.get("cache_control") == {"type": "ephemeral"}


# ── DeepSeek / OpenAI cache metadata ───────────────────────────────────────


def test_deepseek_cache_hit():
    """DeepSeek format: prompt_tokens_details.cached_tokens > 0."""
    from adapters.cache.provider_cache import build_cache_metadata

    response = {
        "usage": {
            "prompt_tokens": 2000,
            "prompt_tokens_details": {"cached_tokens": 1500},
        }
    }
    meta = build_cache_metadata(response, "deepseek")
    assert meta["provider_cache_hit"] is True
    assert meta["cached_tokens"] == 1500
    assert meta["estimated_savings_usd"] > 0


def test_deepseek_no_cache():
    """DeepSeek format: no cached_tokens → no hit."""
    from adapters.cache.provider_cache import build_cache_metadata

    response = {"usage": {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 0}}}
    meta = build_cache_metadata(response, "deepseek")
    assert meta["provider_cache_hit"] is False


def test_anthropic_cache_hit():
    """Anthropic format: cache_read_input_tokens > 0."""
    from adapters.cache.provider_cache import build_cache_metadata

    response = {
        "usage": {
            "cache_read_input_tokens": 800,
            "cache_creation_input_tokens": 2000,
            "prompt_tokens": 3000,
        }
    }
    meta = build_cache_metadata(response, "anthropic", cache_optimization_applied=True)
    assert meta["provider_cache_hit"] is True
    assert meta["cache_read_tokens"] == 800
    assert meta["cache_write_tokens"] == 2000
    assert meta["cache_optimization_applied"] is True


def test_no_cache_info():
    """Response with no cache fields → no hit."""
    from adapters.cache.provider_cache import build_cache_metadata

    meta = build_cache_metadata({"usage": {"prompt_tokens": 50}}, "deepseek")
    assert meta["provider_cache_hit"] is False


# ── Cache destruction on fallback ─────────────────────────────────────────


def test_cache_destruction():
    """Fallback destroys existing cache — metadata reflects this."""
    from adapters.cache.provider_cache import build_cache_destruction_metadata

    d = build_cache_destruction_metadata(
        previous_model="deepseek/deepseek-v4-pro",
        new_model="deepseek/deepseek-v4-flash",
        previous_cached_tokens=1500,
    )
    assert d["previous_cache_destroyed"] is True
    assert d["previous_model"] == "deepseek/deepseek-v4-pro"
    assert d["new_model"] == "deepseek/deepseek-v4-flash"
    assert d["previous_cached_tokens_lost"] == 1500
    assert d["estimated_extra_cost_usd"] > 0


# ── Provider detection ─────────────────────────────────────────────────────


def test_provider_cache_detection():
    """Only known providers report cache control support."""
    from adapters.cache.provider_cache import (
        should_apply_cache_control,
    )

    assert should_apply_cache_control("anthropic") is True   # cache_control breakpoints
    assert should_apply_cache_control("deepseek") is False  # auto-cache
    assert should_apply_cache_control("groq") is False


# ── Canonical message ordering ─────────────────────────────────────────────


def test_canonicalize_message_order():
    """System messages moved to front, rest preserved in order."""
    from adapters.cache.message_ordering import canonicalize_message_order

    msgs = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "system", "content": "You are a bot."},
    ]
    ordered = canonicalize_message_order(msgs)
    assert ordered[0]["role"] == "system"
    assert ordered[1]["role"] == "user"
    assert ordered[2]["role"] == "assistant"
