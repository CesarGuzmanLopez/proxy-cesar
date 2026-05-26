"""Coverage gap tests for Sprint 5.

Targets specific uncovered lines to push new code coverage past 80%.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.service.chat_service import (
    _suggest_higher_threshold_models,
    call_with_fallback,
    evaluate_router_suggestion,
)
from src.config.pseudo_models import load_config

_CONFIG = load_config("pseudo_models.yaml")


@pytest.mark.asyncio
async def test_call_with_fallback_non_retryable_propagates():
    """Non-503/429 errors propagate through fallback (not caught)."""
    mock_phys = MagicMock()
    mock_phys.model = "test-model"
    mock_phys.context_window = None
    mock_schema = MagicMock()
    mock_schema.physical_models = [mock_phys]
    mock_schema.display_name = "Test"

    with patch(
        "src.service.chat_service.call_litellm",
        side_effect=ValueError("Non-retryable error"),
    ):
        with pytest.raises(ValueError, match="Non-retryable error"):
            await call_with_fallback(mock_schema, [{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_call_with_fallback_retryable_triggers_fallback():
    """503/429 on ALL models raises 503 ALL_MODELS_FAILED."""
    mock_phys1 = MagicMock()
    mock_phys1.model = "test-model-1"
    mock_phys1.context_window = None
    mock_phys2 = MagicMock()
    mock_phys2.model = "test-model-2"
    mock_phys2.context_window = None
    mock_schema = MagicMock()
    mock_schema.physical_models = [mock_phys1, mock_phys2]
    mock_schema.display_name = "Test"

    from fastapi import HTTPException
    from litellm.exceptions import ServiceUnavailableError

    svc_err = ServiceUnavailableError(
        message="Down",
        llm_provider="test",
        model="test-model",
    )
    with patch("src.service.chat_service.call_litellm", side_effect=svc_err):
        with pytest.raises(HTTPException) as exc:
            await call_with_fallback(mock_schema, [])
        assert exc.value.status_code == 503
        assert "ALL_MODELS_FAILED" in exc.value.detail["error"]


@pytest.mark.asyncio
async def test_evaluate_router_suggestion_disabled():
    """Returns None when router_llm is disabled."""
    pm = _CONFIG.pseudo_models["normal"]  # normal has router_llm disabled
    result = await evaluate_router_suggestion(
        pm_schema=pm,
        messages=[{"role": "user", "content": "test"}],
        current_pseudo_name="normal",
        config=_CONFIG,
    )
    assert result is None


@pytest.mark.asyncio
async def test_evaluate_router_suggestion_no_suggester():
    """Returns None when suggester model not found in config."""
    mock_pm = MagicMock()
    mock_pm.router_llm.enabled = True
    mock_pm.router_llm.suggester = "non-existent-model"

    result = await evaluate_router_suggestion(
        pm_schema=mock_pm,
        messages=[{"role": "user", "content": "test"}],
        current_pseudo_name="test",
        config=_CONFIG,
    )
    assert result is None


def test_suggest_higher_threshold_models():
    """Returns models with higher threshold than estimated tokens."""
    suggestions = _suggest_higher_threshold_models(_CONFIG, 50000)
    assert len(suggestions) > 0
    for s in suggestions:
        assert s["input_token_threshold"] >= 50000


def test_suggest_higher_threshold_models_no_match():
    """No suggestions when threshold exceeds all models."""
    suggestions = _suggest_higher_threshold_models(_CONFIG, 99999999)
    assert len(suggestions) == 0
