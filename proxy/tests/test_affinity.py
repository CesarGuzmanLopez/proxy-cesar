"""Tests for Valkey affinity operations.

"""

import pytest

from src.adapters.cache.valkey_affinity import ValkeyAffinityAdapter


@pytest.fixture
def affinity(mock_valkey):
    return ValkeyAffinityAdapter(mock_valkey)


@pytest.mark.asyncio
async def test_1_set_and_get(affinity):
    """set_affinity writes key with correct value."""
    await affinity.set("conv-abc", "qwen3-max")
    result = await affinity.get("conv-abc")
    assert result == "qwen3-max"


@pytest.mark.asyncio
async def test_2_get_non_existent(affinity):
    """get_affinity returns None for non-existent key."""
    result = await affinity.get("conv-nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_3_ttl_respected(affinity):
    """TTL is set correctly."""
    await affinity.set("conv-abc", "qwen3-max", ttl_seconds=1)
    result = await affinity.get("conv-abc")
    assert result == "qwen3-max"


@pytest.mark.asyncio
async def test_4_delete(affinity):
    """delete_affinity removes the key."""
    await affinity.set("conv-abc", "qwen3-max")
    await affinity.delete("conv-abc")
    result = await affinity.get("conv-abc")
    assert result is None


@pytest.mark.asyncio
async def test_5_multiple_conversations_isolated(affinity):
    """Multiple conversations have isolated keys."""
    await affinity.set("conv-a", "model-a")
    await affinity.set("conv-b", "model-b")
    assert await affinity.get("conv-a") == "model-a"
    assert await affinity.get("conv-b") == "model-b"


@pytest.mark.asyncio
async def test_6_overwrite_existing(affinity):
    """Setting affinity overwrites previous value."""
    await affinity.set("conv-abc", "model-old")
    await affinity.set("conv-abc", "model-new")
    result = await affinity.get("conv-abc")
    assert result == "model-new"
