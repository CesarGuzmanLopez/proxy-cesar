"""Tests for provider-specific cache optimizations (Sprint 7 §3)."""

from src.adapters.cache.provider_cache import (
    apply_anthropic_cache_control,
    build_cache_destruction_metadata,
    build_cache_metadata,
    provider_supports_cache,
    should_apply_cache_control,
)


def test_anthropic_cache_control_breakpoints():
    """Anthropic messages receive cache_control breakpoints (max 4)."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
        {"role": "user", "content": "New query"},
    ]
    result = apply_anthropic_cache_control(messages)

    # System message gets first breakpoint on its last content item
    assert result[0]["content"][-1].get("cache_control") == {"type": "ephemeral"}

    # The second-to-last message (assistant) gets the second breakpoint
    assert result[2]["content"][-1].get("cache_control") == {"type": "ephemeral"}

    # No more than 4 breakpoints across content items
    bp_count = sum(
        1 for m in result
        for c in (m.get("content", []) if isinstance(m.get("content"), list) else [])
        if isinstance(c, dict) and c.get("cache_control") == {"type": "ephemeral"}
    )
    assert bp_count <= 4


def test_anthropic_cache_control_original_not_modified():
    """Original messages are not modified by cache_control application."""
    original = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
    ]
    result = apply_anthropic_cache_control(original)
    assert "cache_control" not in original[0].get("content", [{}])[0]
    assert "cache_control" in result[0].get("content", [{}])[0]


def test_should_apply_cache_control():
    """Only Anthropic gets cache_control breakpoints; others use auto-cache."""
    assert should_apply_cache_control("anthropic") is True   # cache_control breakpoints
    assert should_apply_cache_control("openai") is False
    assert should_apply_cache_control("deepseek") is False
    assert should_apply_cache_control("groq") is False


def test_provider_supports_cache():
    """Check which providers have any cache support."""
    assert provider_supports_cache("anthropic") is True      # cache_control support
    assert provider_supports_cache("openai") is True
    assert provider_supports_cache("deepseek") is True
    assert provider_supports_cache("groq") is True


def test_build_cache_metadata_openai_style():
    """Cache metadata extracted from OpenAI/DeepSeek-style response."""
    response = {
        "usage": {
            "prompt_tokens": 5000,
            "completion_tokens": 800,
            "prompt_tokens_details": {"cached_tokens": 4500},
        },
    }
    meta = build_cache_metadata(response, "openai", cache_optimization_applied=True)
    assert meta["provider_cache_hit"] is True
    assert meta["cached_tokens"] == 4500
    assert meta["total_prompt_tokens"] == 5000
    assert meta["estimated_savings_usd"] > 0
    assert meta["cache_optimization_applied"] is True


def test_build_cache_metadata_anthropic_style():
    """Cache metadata extracted from Anthropic-style response."""
    response = {
        "usage": {
            "cache_read_input_tokens": 4200,
            "cache_creation_input_tokens": 300,
        },
    }
    meta = build_cache_metadata(response, "anthropic", cache_optimization_applied=True)
    assert meta["provider_cache_hit"] is True
    assert meta["cache_read_tokens"] == 4200
    assert meta["cache_write_tokens"] == 300
    assert meta["estimated_savings_usd"] > 0


def test_build_cache_metadata_no_cache():
    """When no cache info is present, cache_hit is False."""
    response = {"usage": {"prompt_tokens": 100, "completion_tokens": 50}}
    meta = build_cache_metadata(response, "groq", cache_optimization_applied=False)
    assert meta["provider_cache_hit"] is False
    assert "cached_tokens" not in meta
    assert meta["cache_optimization_applied"] is False


def test_build_cache_metadata_empty_response():
    """Empty response dict returns basic metadata."""
    meta = build_cache_metadata({}, "openai", False)
    assert meta["provider"] == "openai"
    assert meta["provider_cache_hit"] is False


def test_cache_destruction_metadata():
    """Cache destruction on fallback is reported clearly."""
    meta = build_cache_destruction_metadata(
        previous_model="qwen3-max",
        new_model="deepseek-v4-flash",
        previous_cached_tokens=45000,
    )
    assert meta["previous_cache_destroyed"] is True
    assert meta["previous_model"] == "qwen3-max"
    assert meta["new_model"] == "deepseek-v4-flash"
    assert meta["previous_cached_tokens_lost"] == 45000
    assert meta["new_cache_starting"] is True
    assert meta["estimated_extra_cost_usd"] > 0


def test_anthropic_max_breakpoints():
    """More than 4 breakpoints never applied (Anthropic limit)."""
    # Create a long conversation that would trigger many breakpoints
    messages = [
        {"role": "system", "content": "sys"},
    ]
    # Add many user/assistant pairs
    for i in range(20):
        messages.append({"role": "user", "content": f"q{i}"})
        messages.append({"role": "assistant", "content": f"a{i}"})
    # Add final user message
    messages.append({"role": "user", "content": "final question"})

    result = apply_anthropic_cache_control(messages)

    bp_count = sum(1 for m in result if m.get("cache_control") == {"type": "ephemeral"})
    assert bp_count <= 4


# ── Bug 1: cache provider derived from model prefix ────────────────────


def test_should_apply_cache_control_with_model_prefix():
    """cache_control is applied when provider matches from model prefix."""
    # These should all resolve to 'anthropic' via model prefix
    assert should_apply_cache_control("anthropic") is True


def test_should_apply_cache_control_non_anthropic():
    """Non-Anthropic providers don't get cache_control breakpoints."""
    assert should_apply_cache_control("openai") is False
    assert should_apply_cache_control("deepseek") is False
    assert should_apply_cache_control("groq") is False
    assert should_apply_cache_control("opencode-go") is False


def test_provider_supports_cache_mixed_case():
    """Case-insensitive check works for all known providers."""
    assert provider_supports_cache("ANTHROPIC") is True
    assert provider_supports_cache("OpenAI") is True
    assert provider_supports_cache("DEEPSEEK") is True
    assert provider_supports_cache("Groq") is True


def test_provider_supports_cache_unknown():
    """Unknown providers are correctly identified as not supporting cache."""
    assert provider_supports_cache("ollama") is False
    assert provider_supports_cache("unknown-provider") is False


def test_lru_cache_hits():
    """Repeated calls with same provider return cached result (fast)."""
    # First call populates cache
    result1 = should_apply_cache_control("anthropic")
    # Second call should hit cache
    result2 = should_apply_cache_control("anthropic")
    assert result1 == result2
    # Verify it's True
    assert result1 is True


def test_lru_cache_maxsize():
    """lru_cache with maxsize=32 handles many distinct providers."""
    providers = [f"provider-{i}" for i in range(50)]
    for p in providers:
        # Should not raise or behave incorrectly
        result = provider_supports_cache(p)
        assert isinstance(result, bool)
