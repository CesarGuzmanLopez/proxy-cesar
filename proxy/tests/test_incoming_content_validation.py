"""Tests for incoming content validation.

Sprint 2 §9.1b — minimum 9 tests.
Verifies that validate_incoming_content() rejects unsupported content
on the CURRENT pseudo-model (not just on switch).
"""

import pytest
from fastapi import HTTPException

from src.config.pseudo_models import load_config
from src.domain.capabilities import TurnCapabilities
from src.service.compatibility import validate_incoming_content

CONFIG = load_config()


def _get_pm(name: str):
    return CONFIG.pseudo_models[name]


def test_image_sent_to_normal_returns_400():
    """Image sent to 'normal' (no vision models) → 400 IMAGES_NOT_SUPPORTED_BY_PSEUDO_MODEL."""
    turn_caps = TurnCapabilities(has_images=True)
    with pytest.raises(HTTPException) as exc:
        validate_incoming_content(turn_caps, _get_pm("normal"), "normal", CONFIG)
    assert exc.value.status_code == 400
    assert "IMAGES_NOT_SUPPORTED_BY_PSEUDO_MODEL" in str(exc.value.detail["error"])


def test_image_sent_to_avanzada_vision_proceeds():
    """Image sent to 'avanzada-vision' (has vision models) → no exception."""
    turn_caps = TurnCapabilities(has_images=True)
    # Should not raise
    result = validate_incoming_content(turn_caps, _get_pm("avanzada-vision"), "avanzada-vision", CONFIG)
    assert result is None


def test_audio_sent_to_any_model_returns_400():
    """Audio sent to any pseudo-model → 400 AUDIO_NOT_SUPPORTED."""
    turn_caps = TurnCapabilities(has_audio=True)
    for name in ("normal", "avanzada-vision", "deep-flash"):
        with pytest.raises(HTTPException) as exc:
            validate_incoming_content(turn_caps, _get_pm(name), name, CONFIG)
        assert exc.value.status_code == 400
        assert "AUDIO_NOT_SUPPORTED" in str(exc.value.detail["error"])


def test_pdf_sent_to_deep_flash_returns_400():
    """PDF sent to 'deep-flash' (no vision) → 400 PDF_NOT_SUPPORTED."""
    turn_caps = TurnCapabilities(has_pdf=True)
    with pytest.raises(HTTPException) as exc:
        validate_incoming_content(turn_caps, _get_pm("deep-flash"), "deep-flash", CONFIG)
    assert exc.value.status_code == 400
    assert "PDF_NOT_SUPPORTED" in str(exc.value.detail["error"])


def test_pdf_sent_to_avanzada_vision_proceeds():
    """PDF sent to 'avanzada-vision' (has vision) → proceeds (PDFs treated as images)."""
    turn_caps = TurnCapabilities(has_pdf=True)
    result = validate_incoming_content(turn_caps, _get_pm("avanzada-vision"), "avanzada-vision", CONFIG)
    assert result is None


def test_video_sent_to_any_model_returns_400():
    """Video sent to any pseudo-model → 400 VIDEO_NOT_SUPPORTED."""
    turn_caps = TurnCapabilities(has_video=True)
    for name in ("normal", "avanzada-vision", "deep-flash"):
        with pytest.raises(HTTPException) as exc:
            validate_incoming_content(turn_caps, _get_pm(name), name, CONFIG)
        assert exc.value.status_code == 400
        assert "VIDEO_NOT_SUPPORTED" in str(exc.value.detail["error"])


def test_parallel_tools_sent_to_flash_lowcost_returns_400():
    """Parallel tools sent to 'flash-lowcost' (no parallel models) → 400."""
    turn_caps = TurnCapabilities(has_parallel_tools=True)
    with pytest.raises(HTTPException) as exc:
        validate_incoming_content(turn_caps, _get_pm("flash-lowcost"), "flash-lowcost", CONFIG)
    assert exc.value.status_code == 400
    assert "PARALLEL_TOOLS_NOT_SUPPORTED_BY_PSEUDO_MODEL" in str(exc.value.detail["error"])


def test_error_responses_include_remediation():
    """Error responses include 'remediation' array with actionable options."""
    turn_caps = TurnCapabilities(has_images=True)
    with pytest.raises(HTTPException) as exc:
        validate_incoming_content(turn_caps, _get_pm("normal"), "normal", CONFIG)
    detail = exc.value.detail
    assert "remediation" in detail
    assert isinstance(detail["remediation"], list)
    assert len(detail["remediation"]) > 0


def test_error_responses_include_vision_capable_list():
    """Error responses include 'vision_capable_pseudo_models' list."""
    turn_caps = TurnCapabilities(has_images=True)
    with pytest.raises(HTTPException) as exc:
        validate_incoming_content(turn_caps, _get_pm("normal"), "normal", CONFIG)
    detail = exc.value.detail
    assert "vision_capable_pseudo_models" in detail
    assert isinstance(detail["vision_capable_pseudo_models"], list)
