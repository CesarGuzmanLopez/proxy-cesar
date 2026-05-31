"""Tests for tool filter (get_eligible_models).

"""

from src.config.pseudo_models import load_config
from src.domain.capabilities import SessionCapabilities
from src.service.tool_filter import get_eligible_models, is_pinned_model_eligible

CONFIG = load_config()


def _make_caps(has_parallel_tools: bool = False) -> SessionCapabilities:
    return SessionCapabilities(
        conversation_id="test",
        has_parallel_tools=has_parallel_tools,
    )


def test_no_parallel_tools_all_models_returned():
    """No parallel tools in session → all models returned."""
    pm = CONFIG.pseudo_models["normal"]
    result = get_eligible_models(
        pm.physical_models, _make_caps(has_parallel_tools=False)
    )
    assert len(result) == len(pm.physical_models)


def test_parallel_tools_only_parallel_models():
    """Parallel tools in session → only parallel_tools: true models."""
    pm = CONFIG.pseudo_models["normal"]
    result = get_eligible_models(
        pm.physical_models, _make_caps(has_parallel_tools=True)
    )
    for m in result:
        assert m.parallel_tools is True


def test_pensamiento_profundo_parallel_filters():
    """pensamiento-profundo-caro: parallel → models with parallel_tools."""
    pm = CONFIG.pseudo_models["pensamiento-profundo-caro"]
    result = get_eligible_models(
        pm.physical_models, _make_caps(has_parallel_tools=True)
    )
    assert len(result) == 2  # deepseek-v4-pro + gemini-3.5-flash
    assert all(m.parallel_tools is True for m in result)


def test_tareas_avanzadas_no_parallel_returns_all():
    """tareas-avanzadas: no parallel → all 2 models eligible."""
    pm = CONFIG.pseudo_models["tareas-avanzadas"]
    result = get_eligible_models(
        pm.physical_models, _make_caps(has_parallel_tools=False)
    )
    models = [m.model for m in result]
    assert len(models) == 2
    assert "openai/minimax-m2.7" in models
    assert "deepseek/deepseek-v4-flash" in models

    assert is_pinned_model_eligible("deepseek/deepseek-v4-flash", result) is True
