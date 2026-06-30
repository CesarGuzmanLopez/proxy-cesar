"""Tests for rate limiting middleware (Feature)."""

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from fakeredis import FakeAsyncValkey


@pytest.fixture
def rate_limit_app():
    """FastAPI test app with RateLimitMiddleware and mock Valkey."""
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat():
        return {"choices": [{"message": {"content": "ok"}}]}

    return app


@pytest.fixture
async def rate_limit_client(monkeypatch, rate_limit_app):
    """Client with mock Valkey for rate limiting tests."""
    from src.middleware.rate_limiter import RateLimitMiddleware

    fake_valkey = FakeAsyncValkey(decode_responses=True)
    rate_limit_app.state.valkey = fake_valkey

    monkeypatch.setenv("PROXY_API_KEY", "")  # disable auth for testing

    rate_limit_app.add_middleware(RateLimitMiddleware)

    transport = ASGITransport(app=rate_limit_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


_HEADERS = {"X-Pseudo-Model": "normal"}


async def test_request_within_limit_proceeds(rate_limit_client):
    """First few requests within the limit succeed."""
    response = await rate_limit_client.post(
        "/v1/chat/completions",
        json={"model": "normal", "messages": [{"role": "user", "content": "hi"}]},
        headers=_HEADERS,
    )
    assert response.status_code == 200


async def test_rate_limiter_disabled_no_limit_check(rate_limit_client):
    """Rate limiter is disabled — all requests pass through regardless of count."""
    for _ in range(100):
        response = await rate_limit_client.post(
            "/v1/chat/completions",
            json={"model": "normal", "messages": [{"role": "user", "content": "hi"}]},
            headers=_HEADERS,
        )
        assert response.status_code == 200


async def test_rate_limiter_disabled_all_requests_pass(rate_limit_client):
    """Rate limiter is disabled — all requests pass through."""
    for _ in range(5):
        response = await rate_limit_client.post(
            "/v1/chat/completions",
            json={"model": "normal", "messages": [{"role": "user", "content": "hi"}]},
            headers=_HEADERS,
        )
        assert response.status_code == 200


async def test_rate_limiter_disabled_different_models_all_pass(rate_limit_client):
    """All pseudo-models pass through when rate limiter is disabled."""
    for model in ["normal", "flash", "tareas-avanzadas"]:
        response = await rate_limit_client.post(
            "/v1/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": "hi"}]},
            headers=_HEADERS,
        )
        assert response.status_code == 200
