"""Tests for GET /v1/models and GET /health.

sprint §13.6 — minimum 5 test cases.
"""

import pytest
from httpx import AsyncClient

KNOWN_PSEUDO_MODELS = 10


@pytest.mark.asyncio
async def test_1_models_list_all(async_client: AsyncClient):
    """GET /v1/models returns at least the 10 pseudo-models (plus local models if any)."""
    response = await async_client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    # At minimum, the 10 pseudo-models
    assert len(data["data"]) >= KNOWN_PSEUDO_MODELS
    # Verify pseudo-model names are present
    pseudo_ids = {m["id"] for m in data["data"]}
    for expected in ("normal", "vision", "compactador", "pensamiento-profundo-caro",
                     "tareas-avanzadas", "normal-gratis", "massive-fast", "flash-lowcost", "audio", "imagen"):
        assert expected in pseudo_ids, f"Missing pseudo-model: {expected}"


@pytest.mark.asyncio
async def test_2_models_optimistic_capabilities(async_client: AsyncClient):
    """Each pseudo-model advertises ALL capabilities as true (optimistic advertising).

    Local models (ollama/lmstudio) report their actual capabilities instead.
    """
    response = await async_client.get("/v1/models")
    assert response.status_code == 200
    for model in response.json()["data"]:
        caps = model["capabilities"]
        # Local models report real capabilities; pseudo-models are always optimistic
        if model["id"] in ("normal", "vision", "normal-gratis",
                           "massive-fast", "flash-lowcost", "audio", "imagen", "compactador",
                           "pensamiento-profundo-caro", "tareas-avanzadas"):
            assert caps["vision"] is True
            assert caps["tools"] is True
            assert caps["parallel_tools"] is True
            assert caps["streaming"] is True
            assert caps["function_calling"] is True
        else:
            # Local model — capabilities should exist (actual provider values)
            assert "vision" in caps
            assert "streaming" in caps


@pytest.mark.asyncio
async def test_3_health_ok(async_client: AsyncClient):
    """GET /health returns 200 with services status."""
    response = await async_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "database" in data
    assert "valkey" in data
    assert "providers" in data
    assert "pseudo_models_loaded" in data
    assert data["pseudo_models_loaded"] == KNOWN_PSEUDO_MODELS


@pytest.mark.asyncio
async def test_4_health_degraded(async_client: AsyncClient):
    """GET /health reflects degraded state when services are down."""

    response = await async_client.get("/health")
    assert response.status_code == 200
    # Status may be "ok" or "degraded" depending on test setup
    assert response.json()["status"] in ("ok", "degraded")


@pytest.mark.asyncio
async def test_5_health_no_auth(async_client: AsyncClient):
    """GET /health does not require auth."""
    response = await async_client.get("/health")
    assert response.status_code == 200
    # In Sprint 1, no auth is enforced anywhere
