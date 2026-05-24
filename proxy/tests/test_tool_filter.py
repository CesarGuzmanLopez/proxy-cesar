"""Tests for tool filter (get_eligible_models).

Sprint 2 §9.3 — minimum 8 tests.
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
    result = get_eligible_models(pm.physical_models, _make_caps(has_parallel_tools=False))
    assert len(result) == len(pm.physical_models)


def test_parallel_tools_only_parallel_models():
    """Parallel tools in session → only parallel_tools: true models."""
    pm = CONFIG.pseudo_models["normal"]
    result = get_eligible_models(pm.physical_models, _make_caps(has_parallel_tools=True))
    for m in result:
        assert m.parallel_tools is True


def test_pensamiento_profundo_parallel_filters():
    """pensamiento-profundo-caro: parallel → only deepseek-v4-pro."""
    pm = CONFIG.pseudo_models["pensamiento-profundo-caro"]
    result = get_eligible_models(pm.physical_models, _make_caps(has_parallel_tools=True))
    # Only deepseek-v4-pro has parallel_tools: true
    assert len(result) == 1
    assert result[0].model == "openrouter/deepseek-v4-pro"


def test_tareas_avanzadas_no_parallel_includes_minimax():
    """tareas-avanzadas: no parallel → all models including MiniMax."""
    pm = CONFIG.pseudo_models["tareas-avanzadas"]
    result = get_eligible_models(pm.physical_models, _make_caps(has_parallel_tools=False))
    models = [m.model for m in result]
    assert "openrouter/minimax-m2.5" in models


def test_normal_with_parallel_only_deepseek():
    """normal with parallel → only deepseek-v4-flash."""
    pm = CONFIG.pseudo_models["normal"]
    result = get_eligible_models(pm.physical_models, _make_caps(has_parallel_tools=True))
    models = [m.model for m in result]
    assert "openrouter/deepseek-v4-flash" in models
    assert "openrouter/qwen3-max" not in models


def test_flash_lowcost_with_parallel_returns_all():
    """flash-lowcost with parallel → pool empty → returns all models."""
    pm = CONFIG.pseudo_models["flash-lowcost"]
    result = get_eligible_models(pm.physical_models, _make_caps(has_parallel_tools=True))
    # No model has parallel_tools: true, so returns all
    assert len(result) == len(pm.physical_models)


def test_deep_flash_with_parallel_only_deepseek():
    """deep-flash with parallel → only deepseek-v4-flash."""
    pm = CONFIG.pseudo_models["deep-flash"]
    result = get_eligible_models(pm.physical_models, _make_caps(has_parallel_tools=True))
    models = [m.model for m in result]
    assert "openrouter/deepseek-v4-flash" in models
    assert "openrouter/glm-4.5-flash" not in models


def test_is_pinned_model_eligible():
    """is_pinned_model_eligible correctly identifies eligible pinned models."""
    pm = CONFIG.pseudo_models["normal"]
    eligible = get_eligible_models(pm.physical_models, _make_caps(has_parallel_tools=True))

    assert is_pinned_model_eligible("openrouter/deepseek-v4-flash", eligible) is True
    assert is_pinned_model_eligible("openrouter/qwen3-max", eligible) is False
