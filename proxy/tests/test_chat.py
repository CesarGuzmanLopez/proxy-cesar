"""Tests for POST /v1/chat/completions.

sprint §13.3 — minimum 12 test cases.
"""

import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient

CONFIG_PATH = Path(__file__).resolve().parent.parent / "pseudo_models.yaml"

# Load expected primary model from YAML config so tests don't hardcode model names
import yaml
with open(CONFIG_PATH) as _f:
    _raw_config = yaml.safe_load(_f)
NORMAL_PRIMARY = _raw_config["pseudo_models"]["normal"]["physical_models"][0]["model"]


@pytest.mark.asyncio
async def test_1_new_conversation(async_client: AsyncClient, mock_litellm):
    """New conversation → creates conversation, sets affinity, returns response with proxy_metadata."""
    response = await async_client.post(
        "/v1/chat/completions",
        json={
            "model": "normal",
            "messages": [{"role": "user", "content": "Hello"}],
            "conversation_id": "conv-test-1",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "proxy_metadata" in data
    assert data["proxy_metadata"]["pseudo_model"] == "normal"
    assert data["proxy_metadata"]["physical_model"] == NORMAL_PRIMARY
    assert data["proxy_metadata"]["conversation_id"] == "conv-test-1"


@pytest.mark.asyncio
async def test_2_affinity_maintained(async_client: AsyncClient, mock_litellm):
    """Second turn same pseudo-model → same physical model, affinity_maintained: true."""
    # First turn
    resp1 = await async_client.post(
        "/v1/chat/completions",
        json={
            "model": "normal",
            "messages": [{"role": "user", "content": "Turn 1"}],
            "conversation_id": "conv-affinity",
        },
    )
    assert resp1.status_code == 200
    data1 = resp1.json()
    phys_model = data1["proxy_metadata"]["physical_model"]

    # Second turn — should maintain same physical model
    resp2 = await async_client.post(
        "/v1/chat/completions",
        json={
            "model": "normal",
            "messages": [{"role": "user", "content": "Turn 2"}],
            "conversation_id": "conv-affinity",
        },
    )
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["proxy_metadata"]["physical_model"] == phys_model


@pytest.mark.asyncio
async def test_3_unknown_pseudo_model(async_client: AsyncClient, mock_litellm):
    """Unknown pseudo-model → resolved via default alias to 'normal' (200)."""
    response = await async_client.post(
        "/v1/chat/completions",
        json={
            "model": "nonexistent-model",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    # Sprint 7: default alias maps unknown models to "normal"
    assert response.status_code == 200
    data = response.json()
    assert data["proxy_metadata"]["pseudo_model"] == "normal"


@pytest.mark.asyncio
async def test_4_auto_generated_conversation_id(
    async_client: AsyncClient, mock_litellm
):
    """Auto-generated conversation_id when not provided."""
    response = await async_client.post(
        "/v1/chat/completions",
        json={
            "model": "normal",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    assert response.status_code == 200
    data = response.json()
    # Should either have conversation_id in proxy_metadata or in response body
    assert "conversation_id" in data or data["proxy_metadata"].get("conversation_id")


@pytest.mark.asyncio
async def test_5_proxy_metadata_fields(async_client: AsyncClient, mock_litellm):
    """Response includes all expected proxy_metadata fields."""
    response = await async_client.post(
        "/v1/chat/completions",
        json={
            "model": "normal",
            "messages": [{"role": "user", "content": "Hello"}],
            "conversation_id": "conv-meta",
        },
    )
    assert response.status_code == 200
    meta = response.json()["proxy_metadata"]
    assert "physical_model" in meta
    assert "pseudo_model" in meta
    assert "conversation_id" in meta
    assert "affinity_maintained" in meta
    assert "fallback_applied" in meta
    assert "fallback_reason" in meta
    assert "router_suggestion" in meta
    assert "tools_filter_applied" in meta
    assert "images_described" in meta
    assert "warning" in meta


@pytest.mark.asyncio
async def test_6_messages_forwarded(async_client: AsyncClient, mock_litellm):
    """Messages are forwarded correctly to LiteLLM (verify via mock)."""
    response = await async_client.post(
        "/v1/chat/completions",
        json={
            "model": "normal",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello"},
            ],
            "conversation_id": "conv-msg",
        },
    )
    assert response.status_code == 200

    # Verify mock was called
    mock_litellm.assert_called_once()
    call_kwargs = mock_litellm.call_args.kwargs
    assert "messages" in call_kwargs
    assert len(call_kwargs["messages"]) == 2


@pytest.mark.asyncio
async def test_7_turn_saved(async_client: AsyncClient, mock_litellm):
    """Turn is saved to DB (we can verify no error occurs)."""
    response = await async_client.post(
        "/v1/chat/completions",
        json={
            "model": "normal",
            "messages": [{"role": "user", "content": "Save me"}],
            "conversation_id": "conv-save",
        },
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_8_tools_forwarded(async_client: AsyncClient, mock_litellm):
    """Request with tools is forwarded (no filtering in Sprint 1)."""
    response = await async_client.post(
        "/v1/chat/completions",
        json={
            "model": "normal",
            "messages": [{"role": "user", "content": "Use a tool"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "test_tool",
                        "description": "A test tool",
                        "parameters": {
                            "type": "object",
                            "properties": {"arg": {"type": "string"}},
                        },
                    },
                }
            ],
            "conversation_id": "conv-tools",
        },
    )
    assert response.status_code == 200
    mock_litellm.assert_called()
    # Verify tools were passed to LiteLLM
    call_kwargs = mock_litellm.call_args.kwargs
    assert "tools" in call_kwargs


@pytest.mark.asyncio
async def test_9_streaming_returns_sse(async_client: AsyncClient, mock_litellm):
    """Request with stream: true returns text/event-stream."""

    # Make litellm return an async generator for streaming
    async def mock_stream():
        chunk = MagicMock()
        chunk.model_dump_json.return_value = '{"id":"test","choices":[]}'
        # Provide usage so _stream_response_generator can extract token counts
        chunk.usage = MagicMock()
        chunk.usage.prompt_tokens = 10
        chunk.usage.completion_tokens = 5
        chunk.model_dump.return_value = {
            "id": "test",
            "choices": [{"delta": {"content": "Hello"}}],
        }
        yield chunk

    # Override the mock for this test
    mock_litellm.return_value = mock_stream()

    response = await async_client.post(
        "/v1/chat/completions",
        json={
            "model": "normal",
            "messages": [{"role": "user", "content": "Stream me"}],
            "stream": True,
            "conversation_id": "conv-stream",
        },
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")


def test_10_model_normalization(async_client: AsyncClient):
    """Model name normalization works with prefix."""
    # This test verifies the normalization function directly
    from src.service.model_resolver import normalize_model_name
    from src.config.pseudo_models import load_config

    config = load_config(CONFIG_PATH)

    assert normalize_model_name("normal", config) == "normal"
    assert normalize_model_name("local/normal", config) == "normal"
    assert normalize_model_name("cesar-proxy/normal", config) == "normal"
    assert (
        normalize_model_name("local/pensamiento-profundo-caro", config)
        == "pensamiento-profundo-caro"
    )


@pytest.mark.asyncio
async def test_auto_describe_on_vision_switch(async_client: AsyncClient, mock_litellm):
    """Switch from vision to non-vision model → images_described in proxy_metadata."""
    from src.main import app
    from src.adapters.db.models import Conversation
    from src.domain.capabilities import SessionCapabilities

    mock_session = app.state.db_session_factory()

    conv_id = uuid.uuid5(uuid.NAMESPACE_DNS, "conv-auto-desc")
    conv = Conversation(
        id=conv_id,
        pseudo_model="vision",
        physical_model="gemini/gemini-3.5-flash",
        total_tokens=100,
    )
    turn = MagicMock()
    turn.turn_number = 1
    turn.messages = [{"role": "user", "content": "hello"}]
    conv.turns = [turn]

    mock_session.get = AsyncMock(return_value=conv)

    # Mock db.execute so compaction code doesn't get coroutines from .scalar()
    exec_result = MagicMock()
    exec_result.scalar.return_value = 0
    exec_result.scalars.return_value.all.return_value = []
    mock_session.execute = AsyncMock(return_value=exec_result)

    session_caps = SessionCapabilities(
        conversation_id=str(conv_id),
        has_images=True,
        has_tools=False,
        has_parallel_tools=False,
        total_tokens=100,
    )

    with (
        patch(
            "src.service.chat_service.load_session_capabilities", new_callable=AsyncMock
        ) as mock_load_caps,
        patch(
            "src.service.chat_messages.auto_describe_images", new_callable=AsyncMock
        ) as mock_auto,
    ):
        mock_load_caps.return_value = session_caps
        mock_auto.return_value = (
            [{"role": "user", "content": "described text"}],
            {
                "ok": True,
                "images_described": 2,
                "described_by": "gemini/gemini-3.5-flash",
                "total_description_tokens": 30,
                "status": "completed",
            },
        )

        response = await async_client.post(
            "/v1/chat/completions",
            json={
                "model": "normal-gratis",
                "messages": [{"role": "user", "content": "Tell me about this image"}],
                "conversation_id": "conv-auto-desc",
            },
        )

        assert response.status_code == 200
        data = response.json()
        meta = data["proxy_metadata"]
        assert meta["images_described"] == 2
        assert meta["images_described_by"] == "gemini/gemini-3.5-flash"
