"""Tests for content validation against physical models.

Verifies that validate_physical_model_content() returns appropriate signals:
- None if content is supported
- delegation signal if content type needs URL transformation
"""

import pytest

from src.config.pseudo_models import load_config
from src.domain.capabilities import TurnCapabilities
from src.service.compatibility import validate_physical_model_content

CONFIG = load_config()


def _get_phys(pm_name: str, index: int = 0):
    """Get a physical model from a pseudo-model by name."""
    return CONFIG.pseudo_models[pm_name].physical_models[index]


def test_image_sent_to_no_vision_model_returns_transform():
    """Image sent to non-vision physical model → transform signal."""
    turn_caps = TurnCapabilities(has_images=True)
    result = validate_physical_model_content(
        turn_caps, _get_phys("tareas-avanzadas")
    )
    assert result == {"action": "transform_unsupported"}


def test_image_sent_to_vision_model_proceeds():
    """Image sent to vision-capable physical model → no signal."""
    turn_caps = TurnCapabilities(has_images=True)
    # Find a vision model
    for pm_name in ("vision", "normal"):
        for phys in CONFIG.pseudo_models[pm_name].physical_models:
            if getattr(phys, "vision", False):
                result = validate_physical_model_content(turn_caps, phys)
                assert result is None
                return
    pytest.skip("No vision model found in config")


def test_audio_sent_to_model_with_audio_passes():
    """Audio sent to physical model with audio capability → no signal."""
    turn_caps = TurnCapabilities(has_audio=True)
    # Find an audio model
    # Find an audio model across all pseudo-models
    for pm_name in CONFIG.pseudo_models:
        for phys in CONFIG.pseudo_models[pm_name].physical_models:
            if getattr(phys, "audio", False):
                result = validate_physical_model_content(turn_caps, phys)
                assert result is None
                return
    pytest.skip("No audio model found in config")


def test_audio_sent_to_no_audio_model_returns_transform():
    """Audio sent to physical model without audio → transform signal."""
    turn_caps = TurnCapabilities(has_audio=True)
    result = validate_physical_model_content(
        turn_caps, _get_phys("normal")
    )
    assert result == {"action": "transform_unsupported"}


def test_pdf_sent_to_no_pdf_model_returns_transform():
    """PDF sent to model without pdf capability → transform signal."""
    turn_caps = TurnCapabilities(has_pdf=True)
    result = validate_physical_model_content(
        turn_caps, _get_phys("tareas-avanzadas")
    )
    assert result == {"action": "transform_unsupported"}


def test_video_sent_to_any_model_returns_transform():
    """Video sent to any model without video → transform signal."""
    turn_caps = TurnCapabilities(has_video=True)
    for name in ("normal", "vision", "flash"):
        result = validate_physical_model_content(
            turn_caps, _get_phys(name)
        )
        assert result == {"action": "transform_unsupported"}


def test_transform_signal_returned_for_all_unsupported_types():
    """All unsupported content types return transform_unsupported signal."""
    caps_list = [
        ("images", TurnCapabilities(has_images=True), "tareas-avanzadas"),
        ("audio", TurnCapabilities(has_audio=True), "normal"),
        ("pdf", TurnCapabilities(has_pdf=True), "tareas-avanzadas"),
        ("video", TurnCapabilities(has_video=True), "normal"),
    ]
    for content_type, caps, pm_name in caps_list:
        result = validate_physical_model_content(
            caps, _get_phys(pm_name)
        )
        assert result == {"action": "transform_unsupported"}, (
            f"{content_type} should return transform signal, got {result}"
        )


def test_no_content_returns_none():
    """No content capabilities → None."""
    turn_caps = TurnCapabilities()
    result = validate_physical_model_content(
        turn_caps, _get_phys("normal")
    )
    assert result is None
