"""Tests for Valkey affinity operations — Valkey was removed, all no-ops."""

import pytest

from src.adapters.cache.valkey_affinity import ValkeyAffinityAdapter


@pytest.fixture
def affinity():
    return ValkeyAffinityAdapter()


@pytest.mark.asyncio
async def test_1_set_and_get(affinity):
    """set does nothing, get always returns None (Valkey removed)."""
    await affinity.set("conv-abc", "qwen3-max")
    result = await affinity.get("conv-abc")
    assert result is None


@pytest.mark.asyncio
async def test_2_get_non_existent(affinity):
    """get returns None for any key."""
    result = await affinity.get("conv-nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_3_ttl_respected(affinity):
    """set and get both no-ops (Valkey removed)."""
    await affinity.set("conv-abc", "qwen3-max", ttl_seconds=1)
    result = await affinity.get("conv-abc")
    assert result is None


@pytest.mark.asyncio
async def test_4_delete(affinity):
    """delete does nothing, get still returns None."""
    await affinity.delete("conv-abc")
    result = await affinity.get("conv-abc")
    assert result is None
    await affinity.set("conv-abc", "qwen3-max")
    await affinity.delete("conv-abc")
    result = await affinity.get("conv-abc")
    assert result is None


@pytest.mark.asyncio
async def test_5_multiple_conversations_isolated(affinity):
    """Multiple conversations have isolated keys (all return None without Valkey)."""
    await affinity.set("conv-a", "model-a")
    await affinity.set("conv-b", "model-b")
    assert await affinity.get("conv-a") is None
    assert await affinity.get("conv-b") is None


@pytest.mark.asyncio
async def test_6_overwrite_existing(affinity):
    """Setting affinity overwrites previous value (all no-op without Valkey)."""
    await affinity.set("conv-abc", "model-old")
    await affinity.set("conv-abc", "model-new")
    result = await affinity.get("conv-abc")
    assert result is None
