"""Tests for SSE streaming.

sprint §13.4 — minimum 5 test cases.
"""

import json
from unittest.mock import MagicMock

import pytest
from httpx import AsyncClient


@pytest.fixture
def mock_streaming_response():
    """Create a mock streaming LiteLLM response."""

    async def async_gen():
        chunk = MagicMock()
        chunk.model_dump_json.return_value = json.dumps(
            {
                "id": "chatcmpl-mock",
                "object": "chat.completion.chunk",
                "choices": [{"delta": {"content": "Hello"}, "finish_reason": None}],
            }
        )
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = "Hello"
        chunk.choices[0].delta.tool_calls = None
        yield chunk

    return async_gen()


@pytest.mark.asyncio
async def test_1_sse_format(
    async_client: AsyncClient, mock_litellm, mock_streaming_response
):
    """Stream response produces valid SSE format (data: ...\\n\\n)."""
    mock_litellm.return_value = mock_streaming_response

    response = await async_client.post(
        "/v1/chat/completions",
        json={
            "model": "normal",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
            "conversation_id": "conv-sse-1",
        },
    )
    assert response.status_code == 200
    content = response.text
    lines = content.strip().split("\n")
    # Each SSE message starts with "data: "
    sse_data_lines = [line for line in lines if line.startswith("data: ")]
    assert len(sse_data_lines) >= 2  # at least one content chunk + final meta + [DONE]


@pytest.mark.asyncio
async def test_2_content_chunks_forwarded(
    async_client: AsyncClient, mock_litellm, mock_streaming_response
):
    """Content chunks are forwarded without modification."""
    mock_litellm.return_value = mock_streaming_response

    response = await async_client.post(
        "/v1/chat/completions",
        json={
            "model": "normal",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
            "conversation_id": "conv-sse-2",
        },
    )
    assert response.status_code == 200
    content = response.text
    # Check that the content "Hello" appears in the stream
    assert "Hello" in content


@pytest.mark.asyncio
async def test_3_final_chunk_has_metadata(
    async_client: AsyncClient, mock_litellm, mock_streaming_response
):
    """Final chunk contains proxy_metadata."""
    mock_litellm.return_value = mock_streaming_response

    response = await async_client.post(
        "/v1/chat/completions",
        json={
            "model": "normal",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
            "conversation_id": "conv-sse-3",
        },
    )
    assert response.status_code == 200
    content = response.text
    # The final chunk before [DONE] should have proxy_metadata
    assert "proxy_metadata" in content


@pytest.mark.asyncio
async def test_4_done_marker(
    async_client: AsyncClient, mock_litellm, mock_streaming_response
):
    """[DONE] marker is present at end of stream."""
    mock_litellm.return_value = mock_streaming_response

    response = await async_client.post(
        "/v1/chat/completions",
        json={
            "model": "normal",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
            "conversation_id": "conv-sse-4",
        },
    )
    assert response.status_code == 200
    assert "data: [DONE]" in response.text


@pytest.mark.asyncio
async def test_5_stream_closes_on_error(async_client: AsyncClient, mock_litellm):
    """Stream handles errors gracefully."""
    mock_litellm.side_effect = Exception("Upstream error")

    response = await async_client.post(
        "/v1/chat/completions",
        json={
            "model": "normal",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
            "conversation_id": "conv-sse-5",
        },
    )
    # Should return an error, not crash
    assert response.status_code in (502, 200)
