"""Tests for Valkey affinity operations with mock Valkey."""

import pytest
from fakeredis import FakeAsyncValkey

from src.adapters.cache.valkey_affinity import ValkeyAffinityAdapter


@pytest.fixture
def affinity():
    fake = FakeAsyncValkey(decode_responses=True)
    return ValkeyAffinityAdapter(fake)


@pytest.mark.asyncio
async def test_1_set_and_get(affinity):
    """set stores value, get retrieves it."""
    await affinity.set("conv-abc", "qwen3-max")
    result = await affinity.get("conv-abc")
    assert result == "qwen3-max"


@pytest.mark.asyncio
async def test_2_get_non_existent(affinity):
    """get returns None for non-existent key."""
    result = await affinity.get("conv-nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_3_ttl_respected(affinity):
    """set with TTL works correctly."""
    await affinity.set("conv-abc", "qwen3-max", ttl_seconds=1)
    result = await affinity.get("conv-abc")
    assert result == "qwen3-max"


@pytest.mark.asyncio
async def test_4_delete(affinity):
    """delete removes stored value."""
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
