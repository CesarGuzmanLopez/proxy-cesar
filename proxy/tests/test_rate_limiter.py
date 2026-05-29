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

    # Clear cached rate limits and re-init
    import src.middleware.rate_limiter as rl

    rl._rate_limits = None

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


async def test_rate_limit_exceeded_returns_429(rate_limit_client):
    """Request above the limit returns 429."""
    # Override limit to 2 for testing
    import src.middleware.rate_limiter as rl

    rl._rate_limits = {"normal": 2}

    # First 2 should succeed
    for _ in range(2):
        response = await rate_limit_client.post(
            "/v1/chat/completions",
            json={"model": "normal", "messages": [{"role": "user", "content": "hi"}]},
            headers=_HEADERS,
        )
        assert response.status_code == 200

    # 3rd should be rate limited
    response = await rate_limit_client.post(
        "/v1/chat/completions",
        json={"model": "normal", "messages": [{"role": "user", "content": "hi"}]},
        headers=_HEADERS,
    )
    assert response.status_code == 429
    data = response.json()
    assert data["error"] == "RATE_LIMIT_EXCEEDED"


async def test_retry_after_header_on_429(rate_limit_client):
    """429 response includes Retry-After header."""
    import src.middleware.rate_limiter as rl

    rl._rate_limits = {"normal": 1}

    # Exhaust limit
    await rate_limit_client.post(
        "/v1/chat/completions",
        json={"model": "normal", "messages": [{"role": "user", "content": "hi"}]},
        headers=_HEADERS,
    )
    response = await rate_limit_client.post(
        "/v1/chat/completions",
        json={"model": "normal", "messages": [{"role": "user", "content": "hi"}]},
        headers=_HEADERS,
    )
    assert response.status_code == 429
    assert "Retry-After" in response.headers
    assert int(response.headers["Retry-After"]) > 0


async def test_rate_limit_headers_on_response(rate_limit_client):
    """Successful responses include X-RateLimit-* headers."""
    response = await rate_limit_client.post(
        "/v1/chat/completions",
        json={"model": "normal", "messages": [{"role": "user", "content": "hi"}]},
        headers=_HEADERS,
    )
    assert response.status_code == 200
    assert "X-RateLimit-Limit" in response.headers
    assert "X-RateLimit-Remaining" in response.headers
    assert "X-RateLimit-Reset" in response.headers


async def test_different_pseudo_models_independent_limits(rate_limit_client):
    """Each pseudo-model has its own rate limit counter."""
    import src.middleware.rate_limiter as rl

    rl._rate_limits = {"normal": 1, "deep-flash": 1}

    # Exhaust normal
    await rate_limit_client.post(
        "/v1/chat/completions",
        json={"model": "normal", "messages": [{"role": "user", "content": "hi"}]},
        headers={"X-Pseudo-Model": "normal"},
    )

    # normal should be rate limited
    r1 = await rate_limit_client.post(
        "/v1/chat/completions",
        json={"model": "normal", "messages": [{"role": "user", "content": "hi"}]},
        headers={"X-Pseudo-Model": "normal"},
    )
    assert r1.status_code == 429

    # deep-flash should still work (independent limit)
    r2 = await rate_limit_client.post(
        "/v1/chat/completions",
        json={"model": "deep-flash", "messages": [{"role": "user", "content": "hi"}]},
        headers={"X-Pseudo-Model": "deep-flash"},
    )
    assert r2.status_code == 200
