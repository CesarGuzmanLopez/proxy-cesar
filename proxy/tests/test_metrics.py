"""Tests for metrics endpoint (Sprint 8 §6)."""

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient


@pytest.fixture(autouse=True)
def reset_metrics():
    """Reset the global metrics store before each test."""
    from src.api.metrics import metrics

    metrics.total_requests = 0
    metrics.requests_by_pseudo = {}
    metrics.total_input_tokens = 0
    metrics.total_output_tokens = 0
    metrics.total_cached_tokens = 0
    metrics.total_saved_by_compaction = 0
    metrics.cache_hits = 0
    metrics.pre_compactions = 0
    metrics.continuous_compactions = 0
    metrics.fallbacks = {}
    metrics.errors_4xx = 0
    metrics.errors_5xx = 0
    metrics.errors_by_type = {}

    yield


@pytest.fixture
def metrics_app():
    """FastAPI app with metrics router and required state."""
    app = FastAPI()

    # Set up app state required by metrics endpoint
    app.state.db_session_factory = None

    from src.api.metrics import router as metrics_router

    app.include_router(metrics_router)
    return app


@pytest.fixture
async def metrics_client(metrics_app):
    """Client for metrics endpoint tests."""
    import time
    import src.api.metrics as metrics_module

    metrics_module._START_TIME = time.time()

    transport = ASGITransport(app=metrics_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


async def test_get_metrics_returns_valid_json(metrics_client):
    """GET /metrics returns 200 with valid JSON."""
    response = await metrics_client.get("/metrics")
    assert response.status_code == 200

    data = response.json()
    assert "uptime_seconds" in data
    assert "total_requests" in data
    assert "total_tokens" in data
    assert "cache" in data
    assert "compactions" in data
    assert "fallbacks" in data
    assert "conversations" in data
    assert "errors" in data


async def test_metrics_include_requests_by_pseudo_model(metrics_client):
    """Metrics include requests_by_pseudo_model breakdown."""
    from src.api.metrics import metrics

    # Record some requests
    metrics.record_request("normal")
    metrics.record_request("normal")
    metrics.record_request("deep-flash")

    response = await metrics_client.get("/metrics")
    data = response.json()
    assert data["total_requests"] == 3
    assert data["requests_by_pseudo_model"]["normal"] == 2
    assert data["requests_by_pseudo_model"]["deep-flash"] == 1


async def test_metrics_include_token_counts(metrics_client):
    """Metrics track input, output, and cached tokens."""
    from src.api.metrics import metrics

    metrics.record_request("normal")
    metrics.record_tokens(input_tokens=1000, output_tokens=200, cached_tokens=800)
    metrics.record_request("normal")
    metrics.record_tokens(input_tokens=500, output_tokens=100, cached_tokens=0)

    response = await metrics_client.get("/metrics")
    data = response.json()
    assert data["total_tokens"]["input"] == 1500
    assert data["total_tokens"]["output"] == 300
    assert data["total_tokens"]["cached"] == 800
    assert data["cache"]["total_cache_hits"] == 1
    assert data["cache"]["hit_rate_pct"] == 50.0  # 1 hit / 2 requests = 50%


async def test_metrics_include_errors_breakdown(metrics_client):
    """Metrics track error counts by type."""
    from src.api.metrics import metrics

    metrics.record_error(400, "INPUT_EXCEEDS_THRESHOLD")
    metrics.record_error(409, "PSEUDO_MODEL_INCOMPATIBLE")
    metrics.record_error(503, "ALL_MODELS_FAILED")

    response = await metrics_client.get("/metrics")
    data = response.json()
    assert data["errors"]["4xx"] == 2
    assert data["errors"]["5xx"] == 1
    assert data["errors"]["by_type"]["INPUT_EXCEEDS_THRESHOLD"] == 1
    assert data["errors"]["by_type"]["PSEUDO_MODEL_INCOMPATIBLE"] == 1
    assert data["errors"]["by_type"]["ALL_MODELS_FAILED"] == 1


async def test_metrics_include_fallback_counts(metrics_client):
    """Metrics track fallback events."""
    from src.api.metrics import metrics

    metrics.record_fallback("upstream_503")
    metrics.record_fallback("upstream_503")
    metrics.record_fallback("upstream_429")

    response = await metrics_client.get("/metrics")
    data = response.json()
    assert data["fallbacks"]["total"] == 3
    assert data["fallbacks"]["by_reason"]["upstream_503"] == 2
    assert data["fallbacks"]["by_reason"]["upstream_429"] == 1


async def test_metrics_include_compactions(metrics_client):
    """Metrics track compaction events."""
    from src.api.metrics import metrics

    metrics.record_compaction(72000)
    metrics.record_compaction(15000)
    metrics.record_compaction(45000)

    response = await metrics_client.get("/metrics")
    data = response.json()
    assert data["compactions"]["explicit_compactions"] >= 0
    assert data["compactions"]["total_tokens_saved"] == 132000
