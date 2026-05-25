"""Tests for authentication middleware (Sprint 8 §2)."""

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.auth import AuthMiddleware


@pytest.fixture
def auth_app():
    """FastAPI test app with only AuthMiddleware registered."""
    app = FastAPI()
    app.add_middleware(AuthMiddleware)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {"data": []}

    @app.post("/v1/chat/completions")
    async def chat():
        return {"choices": []}

    return app


@pytest.fixture
async def auth_client(monkeypatch, auth_app):
    """Async client with PROXY_API_KEY set."""
    monkeypatch.setenv("PROXY_API_KEY", "sk-test-key-12345")
    transport = ASGITransport(app=auth_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
async def auth_client_dev(monkeypatch, auth_app):
    """Async client in dev mode (no PROXY_API_KEY)."""
    monkeypatch.delenv("PROXY_API_KEY", raising=False)
    transport = ASGITransport(app=auth_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


async def test_valid_bearer_token_proceeds(auth_client):
    """Requests with valid Bearer token pass through."""
    response = await auth_client.get(
        "/v1/models",
        headers={"Authorization": "Bearer sk-test-key-12345"},
    )
    assert response.status_code == 200


async def test_missing_auth_header_returns_401(auth_client):
    """Requests without Authorization header are rejected."""
    response = await auth_client.get("/v1/models")
    assert response.status_code == 401
    data = response.json()
    assert data["error"] == "MISSING_AUTH"


async def test_invalid_api_key_returns_401(auth_client):
    """Requests with wrong API key are rejected."""
    response = await auth_client.get(
        "/v1/models",
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert response.status_code == 401
    data = response.json()
    assert data["error"] == "INVALID_API_KEY"


async def test_health_accessible_without_auth(auth_client):
    """Health endpoint is public — no auth required."""
    response = await auth_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_dev_mode_all_endpoints_accessible(auth_client_dev):
    """When PROXY_API_KEY is not set, all endpoints are accessible."""
    response = await auth_client_dev.get("/v1/models")
    assert response.status_code == 200

    response = await auth_client_dev.post(
        "/v1/chat/completions",
        json={"model": "normal", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200


async def test_chat_endpoint_requires_auth(auth_client):
    """Chat completions endpoint requires valid auth."""
    response = await auth_client.post(
        "/v1/chat/completions",
        json={"model": "normal", "messages": [{"role": "user", "content": "hi"}]},
    )
    # Either 200 (if test passes through to unmocked endpoint) or 401
    # The middleware should check auth before routing
    assert response.status_code == 401  # no auth header provided


async def test_wrong_bearer_format(auth_client):
    """Non-Bearer auth schemes are rejected."""
    response = await auth_client.get(
        "/v1/models",
        headers={"Authorization": "Basic dGVzdDp0ZXN0"},
    )
    assert response.status_code == 401
