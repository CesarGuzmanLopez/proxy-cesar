"""Tests for conversation state and compatibility endpoints.

Sprint 2 §9.5 — minimum 8 tests.
Tests the GET endpoints on api/conversations.py.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from src.config.pseudo_models import load_config
from src.main import app
from src.adapters.cache.valkey_affinity import ValkeyAffinityAdapter
from src.adapters.db.models import Conversation

CONFIG_PATH = None  # Will be set in fixture


@pytest.fixture
def valid_config():
    return load_config()


@pytest.fixture
async def client_with_conversation(mock_valkey):
    """Test client with a mock conversation in DB."""
    from src.main import app

    config = load_config()
    app.state.config = config
    app.state.valkey = mock_valkey
    app.state.affinity = ValkeyAffinityAdapter(mock_valkey)

    # Create mock conversation with capability flags
    mock_conv = MagicMock(spec=Conversation)
    mock_conv.id = uuid.uuid4()
    mock_conv.pseudo_model = "avanzada-vision"
    mock_conv.physical_model = "gemini-3.5-flash"
    mock_conv.total_tokens = 45230
    mock_conv.created_at = None
    mock_conv.capability_has_images = True
    mock_conv.capability_has_audio = False
    mock_conv.capability_has_pdf = False
    mock_conv.capability_has_video = False
    mock_conv.capability_has_tools = True
    mock_conv.capability_has_parallel_tools = False
    mock_conv.turns = []  # No turns loaded

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock()
    mock_session.get = AsyncMock(return_value=mock_conv)
    mock_session.add = MagicMock()
    mock_session.flush = AsyncMock()
    mock_session.commit = AsyncMock()
    mock_session.close = AsyncMock()
    # Mock execute() for the turn-count query in GET /conversations/{id}
    mock_count_result = MagicMock()
    mock_count_result.scalar = MagicMock(return_value=0)
    mock_session.execute = AsyncMock(return_value=mock_count_result)
    app.state.db_session_factory = MagicMock(return_value=mock_session)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
async def client_no_conversation(mock_valkey):
    """Test client where conversation does not exist."""
    from src.main import app

    config = load_config()
    app.state.config = config
    app.state.valkey = mock_valkey
    app.state.affinity = ValkeyAffinityAdapter(mock_valkey)

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock()
    mock_session.get = AsyncMock(return_value=None)  # No conversation found
    mock_session.close = AsyncMock()
    mock_count_result = MagicMock()
    mock_count_result.scalar = MagicMock(return_value=0)
    mock_session.execute = AsyncMock(return_value=mock_count_result)
    app.state.db_session_factory = MagicMock(return_value=mock_session)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.mark.asyncio
async def test_get_conversation_returns_full_state(client_with_conversation):
    """GET /conversations/{id} returns full state with capabilities."""
    response = await client_with_conversation.get("/conversations/test-conv-id")
    assert response.status_code == 200
    data = response.json()
    assert "capabilities" in data
    assert data["capabilities"]["has_images"] is True
    assert data["capabilities"]["has_tools"] is True
    assert data["pseudo_model"] == "avanzada-vision"


@pytest.mark.asyncio
async def test_get_nonexistent_conversation_returns_404(client_no_conversation):
    """Non-existent conversation → 404."""
    response = await client_no_conversation.get("/conversations/non-existent-id")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_compatible_models_returns_all(client_with_conversation):
    """GET /conversations/{id}/compatible-models returns all pseudo-models."""
    response = await client_with_conversation.get(
        "/conversations/test-conv-id/compatible-models"
    )
    assert response.status_code == 200
    data = response.json()
    assert "compatible_models" in data
    # Should include all pseudo-models
    assert len(data["compatible_models"]) >= 8
    assert data["current_pseudo_model"] == "avanzada-vision"


@pytest.mark.asyncio
async def test_compatible_models_determinism(client_with_conversation):
    """Compatible models determinism: call twice → same result."""
    url = "/conversations/test-conv-id/compatible-models"
    resp1 = await client_with_conversation.get(url)
    resp2 = await client_with_conversation.get(url)
    assert resp1.status_code == 200
    assert resp2.status_code == 200
    # Same pseudo-model statuses
    models1 = {m["pseudo_model"]: m["status"] for m in resp1.json()["compatible_models"]}
    models2 = {m["pseudo_model"]: m["status"] for m in resp2.json()["compatible_models"]}
    assert models1 == models2


@pytest.mark.asyncio
async def test_compatible_models_reflects_capabilities(client_with_conversation):
    """Compatible models properly reflects current capabilities."""
    response = await client_with_conversation.get(
        "/conversations/test-conv-id/compatible-models"
    )
    data = response.json()
    # Since conversation has images: true,
    # non-vision models should be blocked
    caps = data["capabilities"]
    assert caps["has_images"] is True

    # Find a non-vision model that should be blocked
    for m in data["compatible_models"]:
        if m["pseudo_model"] == "normal":
            assert m["status"] == "blocked"
            break


@pytest.mark.asyncio
async def test_get_tools_compatibility(client_with_conversation):
    """GET /conversations/{id}/tools-compatibility returns tool info."""
    response = await client_with_conversation.get(
        "/conversations/test-conv-id/tools-compatibility"
    )
    assert response.status_code == 200
    data = response.json()
    assert "tools_used" in data
    assert "parallel_tools_used" in data
    assert "pseudo_models" in data
    # Should have at least one pseudo-model with tool info
    assert len(data["pseudo_models"]) >= 1


@pytest.mark.asyncio
async def test_tools_compatibility_identifies_parallel_models(client_with_conversation):
    """Tools-compatibility properly identifies parallel-eligible models."""
    response = await client_with_conversation.get(
        "/conversations/test-conv-id/tools-compatibility"
    )
    data = response.json()
    for pm in data["pseudo_models"]:
        assert "tool_support" in pm
        ts = pm["tool_support"]
        assert "parallel_eligible" in ts
        assert "parallel_models" in ts
        assert "strict_models" in ts


@pytest.mark.asyncio
async def test_conversation_404_with_proper_error(client_no_conversation):
    """Non-existent conversation returns proper error structure."""
    response = await client_no_conversation.get(
        "/conversations/non-existent-id/compatible-models"
    )
    assert response.status_code == 404
    data = response.json()
    assert "error" in data["detail"] or "detail" in data
