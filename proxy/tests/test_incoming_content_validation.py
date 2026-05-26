"""Tests for incoming content validation.

Verifies that validate_incoming_content() returns appropriate signals:
- None if content is supported
- delegation signal if images can be delegated to tools
- transform_unsupported signal if content type needs URL transformation
- HTTPException for unrecoverable errors (parallel tools)
"""

import pytest
from fastapi import HTTPException

from src.config.pseudo_models import load_config
from src.domain.capabilities import TurnCapabilities
from src.service.compatibility import validate_incoming_content

CONFIG = load_config()


def _get_pm(name: str):
    return CONFIG.pseudo_models[name]


def test_image_sent_to_no_vision_model_returns_transform():
    """Image sent to 'tareas-avanzadas' (no vision, no tools) → transform signal."""
    turn_caps = TurnCapabilities(has_images=True)
    result = validate_incoming_content(
        turn_caps, _get_pm("tareas-avanzadas"), "tareas-avanzadas", CONFIG
    )
    assert result == {"action": "transform_unsupported"}


def test_image_sent_to_vision_model_proceeds():
    """Image sent to 'vision' (has vision models) → no signal."""
    turn_caps = TurnCapabilities(has_images=True)
    result = validate_incoming_content(
        turn_caps, _get_pm("vision"), "vision", CONFIG
    )
    assert result is None


def test_image_sent_to_no_vision_with_tool_returns_delegation():
    """Image sent to 'tareas-avanzadas' with a compatible tool → delegation signal."""
    turn_caps = TurnCapabilities(has_images=True)
    tools = [{"function": {"name": "describe", "parameters": {"properties": {"url": {"type": "string"}}}}}]
    result = validate_incoming_content(
        turn_caps, _get_pm("tareas-avanzadas"), "tareas-avanzadas", CONFIG, tools
    )
    assert result is not None
    assert result["action"] == "delegate_images"
    assert "tool_name" in result
    assert "param_name" in result


def test_audio_sent_to_audio_model_passes():
    """Audio sent to 'audio' pseudo-model (has whisper) → no signal."""
    turn_caps = TurnCapabilities(has_audio=True)
    result = validate_incoming_content(
        turn_caps, _get_pm("audio"), "audio", CONFIG
    )
    assert result is None  # audio model supports audio


def test_audio_sent_to_no_audio_model_returns_transform():
    """Audio sent to 'normal' (no audio) → transform signal."""
    turn_caps = TurnCapabilities(has_audio=True)
    result = validate_incoming_content(
        turn_caps, _get_pm("normal"), "normal", CONFIG
    )
    assert result == {"action": "transform_unsupported"}


def test_pdf_sent_to_no_vision_model_returns_transform():
    """PDF sent to 'tareas-avanzadas' (no vision) → transform signal."""
    turn_caps = TurnCapabilities(has_pdf=True)
    result = validate_incoming_content(
        turn_caps, _get_pm("tareas-avanzadas"), "tareas-avanzadas", CONFIG
    )
    assert result == {"action": "transform_unsupported"}


def test_pdf_sent_to_vision_model_proceeds():
    """PDF sent to 'vision' (has vision) → proceeds (PDFs treated as images)."""
    turn_caps = TurnCapabilities(has_pdf=True)
    result = validate_incoming_content(
        turn_caps, _get_pm("vision"), "vision", CONFIG
    )
    assert result is None


def test_video_sent_to_any_model_returns_transform():
    """Video sent to any model without video → transform signal."""
    turn_caps = TurnCapabilities(has_video=True)
    for name in ("normal", "vision", "flash-lowcost"):
        result = validate_incoming_content(
            turn_caps, _get_pm(name), name, CONFIG
        )
        assert result == {"action": "transform_unsupported"}


def test_parallel_tools_sent_to_no_parallel_model_returns_400():
    """Parallel tools sent to 'flash-lowcost' (no parallel models) → 400 error."""
    turn_caps = TurnCapabilities(has_parallel_tools=True)
    with pytest.raises(HTTPException) as exc:
        validate_incoming_content(
            turn_caps, _get_pm("flash-lowcost"), "flash-lowcost", CONFIG
        )
    assert exc.value.status_code == 400
    assert "PARALLEL_TOOLS_NOT_SUPPORTED_BY_PSEUDO_MODEL" in str(
        exc.value.detail["error"]
    )


def test_transform_signal_returned_for_all_unsupported_types():
    """All unsupported content types return transform_unsupported signal instead of error."""
    caps_list = [
        ("images", TurnCapabilities(has_images=True), "tareas-avanzadas"),
        ("audio", TurnCapabilities(has_audio=True), "normal"),
        ("pdf", TurnCapabilities(has_pdf=True), "tareas-avanzadas"),
        ("video", TurnCapabilities(has_video=True), "normal"),
    ]
    for content_type, caps, pm_name in caps_list:
        result = validate_incoming_content(
            caps, _get_pm(pm_name), pm_name, CONFIG
        )
        assert result == {"action": "transform_unsupported"}, (
            f"{content_type} should return transform signal, got {result}"
        )
