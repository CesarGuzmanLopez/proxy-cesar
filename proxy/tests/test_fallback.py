"""Tests for fallback within pseudo-model.

sprint §13.5 — minimum 6 test cases.
"""

from unittest.mock import MagicMock

import pytest
from httpx import AsyncClient
from litellm.exceptions import RateLimitError, ServiceUnavailableError


@pytest.mark.asyncio
async def test_1_primary_succeeds(async_client: AsyncClient, mock_litellm):
    """Primary model succeeds → no fallback, fallback_applied: false."""
    response = await async_client.post(
        "/v1/chat/completions",
        json={
            "model": "normal",
            "messages": [{"role": "user", "content": "Hello"}],
            "conversation_id": "conv-fb-1",
        },
    )
    assert response.status_code == 200
    meta = response.json()["proxy_metadata"]
    assert meta["fallback_applied"] is False


@pytest.mark.asyncio
async def test_2_primary_503_fallback(async_client: AsyncClient, mock_litellm):
    """Primary model returns 503 → fallback to second model, fallback_applied: true."""
    # First call fails with 503, second succeeds
    second_response = MagicMock()
    second_response.choices = []
    second_response.usage = MagicMock()
    second_response.usage.prompt_tokens = 10
    second_response.usage.completion_tokens = 20
    second_response.model_dump.return_value = {
        "id": "chatcmpl-fallback",
        "object": "chat.completion",
        "choices": [{"message": {"role": "assistant", "content": "Fallback response"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
    }

    mock_litellm.side_effect = [
        ServiceUnavailableError(
            "Primary model down", llm_provider="qwen", model="qwen3-max"
        ),
        second_response,
    ]

    response = await async_client.post(
        "/v1/chat/completions",
        json={
            "model": "normal",
            "messages": [{"role": "user", "content": "Hello"}],
            "conversation_id": "conv-fb-2",
        },
    )
    assert response.status_code == 200
    meta = response.json()["proxy_metadata"]
    assert meta["fallback_applied"] is True
    assert meta["fallback_reason"] is not None


@pytest.mark.asyncio
async def test_3_all_models_fail(async_client: AsyncClient, mock_litellm):
    """All models return 503 → 503 ALL_MODELS_FAILED."""
    mock_litellm.side_effect = ServiceUnavailableError(
        "Model down", llm_provider="any", model="any"
    )

    response = await async_client.post(
        "/v1/chat/completions",
        json={
            "model": "normal",
            "messages": [{"role": "user", "content": "Hello"}],
            "conversation_id": "conv-fb-3",
        },
    )
    assert response.status_code == 503
    data = response.json()
    assert "ALL_MODELS_FAILED" in str(data)


@pytest.mark.asyncio
async def test_4_primary_429_fallback(async_client: AsyncClient, mock_litellm):
    """Primary model returns 429 → fallback to second model."""
    second_response = MagicMock()
    second_response.choices = []
    second_response.usage = MagicMock()
    second_response.usage.prompt_tokens = 10
    second_response.usage.completion_tokens = 20
    second_response.model_dump.return_value = {
        "id": "chatcmpl-fallback-429",
        "object": "chat.completion",
        "choices": [{"message": {"role": "assistant", "content": "Fallback"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
    }

    mock_litellm.side_effect = [
        RateLimitError("Rate limited", llm_provider="qwen", model="qwen3-max"),
        second_response,
    ]

    response = await async_client.post(
        "/v1/chat/completions",
        json={
            "model": "normal",
            "messages": [{"role": "user", "content": "Hello"}],
            "conversation_id": "conv-fb-4",
        },
    )
    assert response.status_code == 200
    meta = response.json()["proxy_metadata"]
    assert meta["fallback_applied"] is True


@pytest.mark.asyncio
async def test_5_non_retryable_error(async_client: AsyncClient, mock_litellm):
    """Non-retryable error (e.g., 400) → raised immediately, no fallback."""
    from litellm.exceptions import BadRequestError

    mock_litellm.side_effect = BadRequestError(
        "Bad request", llm_provider="qwen", model="qwen3-max"
    )

    response = await async_client.post(
        "/v1/chat/completions",
        json={
            "model": "normal",
            "messages": [{"role": "user", "content": "Hello"}],
            "conversation_id": "conv-fb-5",
        },
    )
    # BadRequestError is now retryable (invalid model IDs should skip to next model).
    # When ALL models fail, returns 503 ALL_MODELS_FAILED.
    assert response.status_code == 503


@pytest.mark.asyncio
async def test_6_fallback_reason_explains(async_client: AsyncClient, mock_litellm):
    """fallback_reason explains which model failed and why."""
    second_response = MagicMock()
    second_response.choices = []
    second_response.usage = MagicMock()
    second_response.usage.prompt_tokens = 10
    second_response.usage.completion_tokens = 20
    second_response.model_dump.return_value = {
        "id": "chatcmpl-fallback",
        "object": "chat.completion",
        "choices": [{"message": {"role": "assistant", "content": "OK"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
    }

    mock_litellm.side_effect = [
        ServiceUnavailableError("Down", llm_provider="qwen", model="qwen3-max"),
        second_response,
    ]

    response = await async_client.post(
        "/v1/chat/completions",
        json={
            "model": "normal",
            "messages": [{"role": "user", "content": "Hello"}],
            "conversation_id": "conv-fb-6",
        },
    )
    assert response.status_code == 200
    meta = response.json()["proxy_metadata"]
    assert meta["fallback_applied"] is True
    assert "ServiceUnavailableError" in meta["fallback_reason"]
