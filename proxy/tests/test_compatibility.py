"""Tests for compatibility validation (validate_switch).

Sprint 2 §9.2 — minimum 30 tests covering the full compatibility matrix.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config.pseudo_models import load_config
from src.domain.capabilities import (
    CompatibilityStatus,
    SessionCapabilities,
)
from src.service.compatibility import validate_switch

CONFIG = load_config()


def _make_caps(
    has_images: bool = False,
    has_audio: bool = False,
    has_pdf: bool = False,
    has_video: bool = False,
    has_tools: bool = False,
    has_parallel_tools: bool = False,
    total_tokens: int = 0,
) -> SessionCapabilities:
    return SessionCapabilities(
        conversation_id="test-conv",
        has_images=has_images,
        has_audio=has_audio,
        has_pdf=has_pdf,
        has_video=has_video,
        has_tools=has_tools,
        has_parallel_tools=has_parallel_tools,
        total_tokens=total_tokens,
    )


# ── SAFE cases ──────────────────────────────────────────────────────────────


def test_normal_to_tareas_avanzadas_no_multimedia_no_tools():
    """normal → tareas-avanzadas (no multimedia, no tools) → SAFE."""
    result = validate_switch(
        "normal",
        "tareas-avanzadas",
        CONFIG.pseudo_models["tareas-avanzadas"],
        _make_caps(),
        CONFIG,
    )
    assert result.status == CompatibilityStatus.SAFE


def test_normal_to_tareas_avanzadas_with_tools():
    """normal → tareas-avanzadas (tools, no parallel) → SAFE."""
    result = validate_switch(
        "normal",
        "tareas-avanzadas",
        CONFIG.pseudo_models["tareas-avanzadas"],
        _make_caps(has_tools=True),
        CONFIG,
    )
    assert result.status == CompatibilityStatus.SAFE


def test_normal_to_pensamiento_profundo():
    """normal → pensamiento-profundo-caro → SAFE (superset)."""
    result = validate_switch(
        "normal",
        "pensamiento-profundo-caro",
        CONFIG.pseudo_models["pensamiento-profundo-caro"],
        _make_caps(),
        CONFIG,
    )
    assert result.status == CompatibilityStatus.SAFE


def test_tareas_avanzadas_to_normal_no_parallel():
    """tareas-avanzadas → normal (no parallel tools) → SAFE."""
    result = validate_switch(
        "tareas-avanzadas",
        "normal",
        CONFIG.pseudo_models["normal"],
        _make_caps(),
        CONFIG,
    )
    assert result.status == CompatibilityStatus.SAFE


def test_tareas_avanzadas_to_pensamiento_profundo():
    """tareas-avanzadas → pensamiento-profundo-caro → SAFE."""
    result = validate_switch(
        "tareas-avanzadas",
        "pensamiento-profundo-caro",
        CONFIG.pseudo_models["pensamiento-profundo-caro"],
        _make_caps(),
        CONFIG,
    )
    assert result.status == CompatibilityStatus.SAFE


def test_pensamiento_profundo_to_tareas_avanzadas():
    """pensamiento-profundo-caro → tareas-avanzadas → SAFE."""
    result = validate_switch(
        "pensamiento-profundo-caro",
        "tareas-avanzadas",
        CONFIG.pseudo_models["tareas-avanzadas"],
        _make_caps(),
        CONFIG,
    )
    assert result.status == CompatibilityStatus.SAFE


def test_flash_vision_to_avanzada_vision_with_images():
    """flash-vision → avanzada-vision (with images) → SAFE (upgrade)."""
    result = validate_switch(
        "vision",
        "vision",
        CONFIG.pseudo_models["vision"],
        _make_caps(has_images=True),
        CONFIG,
    )
    assert result.status == CompatibilityStatus.SAFE


def test_flash_lowcost_to_normal():
    """flash-lowcost → normal → SAFE."""
    result = validate_switch(
        "flash-lowcost",
        "normal",
        CONFIG.pseudo_models["normal"],
        _make_caps(),
        CONFIG,
    )
    assert result.status == CompatibilityStatus.SAFE


def test_any_to_compactador():
    """Any → compactador → SAFE (it's an operation)."""
    result = validate_switch(
        "normal",
        "compactador",
        CONFIG.pseudo_models["compactador"],
        _make_caps(has_images=True, has_tools=True, has_parallel_tools=True),
        CONFIG,
    )
    assert result.status == CompatibilityStatus.SAFE


# ── WARNING cases ────────────────────────────────────────────────────────────


def test_normal_to_massive_fast_no_multimedia():
    """normal → massive-fast (no multimedia, no tools) → WARNING."""
    result = validate_switch(
        "normal",
        "massive-fast",
        CONFIG.pseudo_models["massive-fast"],
        _make_caps(),
        CONFIG,
    )
    assert result.status == CompatibilityStatus.WARNING


def test_normal_to_massive_fast_with_tools():
    """normal → massive-fast (with tools) → WARNING."""
    result = validate_switch(
        "normal",
        "massive-fast",
        CONFIG.pseudo_models["massive-fast"],
        _make_caps(has_tools=True),
        CONFIG,
    )
    assert result.status == CompatibilityStatus.WARNING


def test_normal_to_flash_lowcost():
    """normal → flash-lowcost → WARNING (capacity loss)."""
    result = validate_switch(
        "normal",
        "flash-lowcost",
        CONFIG.pseudo_models["flash-lowcost"],
        _make_caps(),
        CONFIG,
    )
    assert result.status == CompatibilityStatus.WARNING


def test_tareas_avanzadas_to_normal_with_parallel():
    """tareas-avanzadas → normal (parallel tools) → WARNING (2→1 parallel models)."""
    result = validate_switch(
        "tareas-avanzadas",
        "normal",
        CONFIG.pseudo_models["normal"],
        _make_caps(has_parallel_tools=True),
        CONFIG,
    )
    assert result.status == CompatibilityStatus.WARNING


def test_tareas_avanzadas_to_flash_lowcost_with_tools():
    """tareas-avanzadas → flash-lowcost (with tools) → WARNING."""
    result = validate_switch(
        "tareas-avanzadas",
        "flash-lowcost",
        CONFIG.pseudo_models["flash-lowcost"],
        _make_caps(has_tools=True),
        CONFIG,
    )
    assert result.status == CompatibilityStatus.WARNING


def test_pensamiento_profundo_to_flash_lowcost_with_tools():
    """pensamiento-profundo-caro → flash-lowcost (with tools) → WARNING."""
    result = validate_switch(
        "pensamiento-profundo-caro",
        "flash-lowcost",
        CONFIG.pseudo_models["flash-lowcost"],
        _make_caps(has_tools=True),
        CONFIG,
    )
    assert result.status == CompatibilityStatus.WARNING


def test_normal_to_vision_with_images_safe():
    """normal → vision (with images) → WARNING (context shrink 500K→120K)."""
    result = validate_switch(
        "normal",
        "vision",
        CONFIG.pseudo_models["vision"],
        _make_caps(has_images=True),
        CONFIG,
    )
    assert result.status == CompatibilityStatus.WARNING
    assert "context" in result.reason.lower()


def test_vision_to_tareas_avanzadas_images_blocked():
    """vision → tareas-avanzadas (with images) → BLOCKED (no vision, block)."""
    result = validate_switch(
        "vision",
        "tareas-avanzadas",
        CONFIG.pseudo_models["tareas-avanzadas"],
        _make_caps(has_images=True),
        CONFIG,
    )
    assert result.status == CompatibilityStatus.BLOCKED


def test_images_to_normal_gratis_auto_describe():
    """normal → normal-gratis (images) → WARNING (auto-describe on downgrade)."""
    # normal-gratis has no vision but on_downgrade=auto_describe → WARNING.
    result = validate_switch(
        "normal",
        "normal-gratis",
        CONFIG.pseudo_models["normal-gratis"],
        _make_caps(has_images=True),
        CONFIG,
    )
    assert result.status == CompatibilityStatus.WARNING


# ── BLOCKED cases ────────────────────────────────────────────────────────────


def test_normal_to_flash_lowcost_parallel_tools():
    """normal → flash-lowcost (with parallel tools) → BLOCKED."""
    result = validate_switch(
        "normal",
        "flash-lowcost",
        CONFIG.pseudo_models["flash-lowcost"],
        _make_caps(has_parallel_tools=True),
        CONFIG,
    )
    assert result.status == CompatibilityStatus.BLOCKED


def test_vision_to_flash_lowcost_images_allowed():
    """vision → flash-lowcost (with images) → WARNING (budget model, but images pass)."""
    result = validate_switch(
        "vision",
        "flash-lowcost",
        CONFIG.pseudo_models["flash-lowcost"],
        _make_caps(has_images=True),
        CONFIG,
    )
    assert result.status == CompatibilityStatus.WARNING


def test_vision_to_massive_fast_with_images_blocked():
    """vision → massive-fast (with images) → BLOCKED (no vision, on_downgrade=block)."""
    result = validate_switch(
        "vision",
        "massive-fast",
        CONFIG.pseudo_models["massive-fast"],
        _make_caps(has_images=True),
        CONFIG,
    )
    assert result.status == CompatibilityStatus.BLOCKED
    assert len(result.remediation) > 0


def test_context_exceeds_destination_window():
    """Context exceeds destination window → BLOCKED ('CONTEXT_TOO_LARGE')."""
    result = validate_switch(
        "normal",
        "vision",
        CONFIG.pseudo_models["vision"],
        _make_caps(total_tokens=999999),
        CONFIG,
    )
    assert result.status == CompatibilityStatus.BLOCKED


def test_parallel_tools_no_parallel_in_destination():
    """Parallel tools → destination has no parallel models → BLOCKED."""
    result = validate_switch(
        "normal",
        "flash-lowcost",
        CONFIG.pseudo_models["flash-lowcost"],
        _make_caps(has_parallel_tools=True),
        CONFIG,
    )
    assert result.status == CompatibilityStatus.BLOCKED


def test_audio_in_history():
    """Audio in history → destination has no audio → BLOCKED."""
    result = validate_switch(
        "normal",
        "flash-lowcost",
        CONFIG.pseudo_models["flash-lowcost"],
        _make_caps(has_audio=True),
        CONFIG,
    )
    assert result.status == CompatibilityStatus.BLOCKED


def test_video_in_history():
    """Video in history → always BLOCKED."""
    result = validate_switch(
        "normal",
        "tareas-avanzadas",
        CONFIG.pseudo_models["tareas-avanzadas"],
        _make_caps(has_video=True),
        CONFIG,
    )
    assert result.status == CompatibilityStatus.BLOCKED


def test_pdf_in_history_no_vision_destination():
    """PDF in history → destination no vision → BLOCKED."""
    result = validate_switch(
        "vision",
        "tareas-avanzadas",
        CONFIG.pseudo_models["tareas-avanzadas"],
        _make_caps(has_pdf=True),
        CONFIG,
    )
    assert result.status == CompatibilityStatus.BLOCKED


# ── Edge cases ───────────────────────────────────────────────────────────────


def test_same_pseudo_model_safe():
    """Same pseudo-model → always SAFE (no switch)."""
    result = validate_switch(
        "normal",
        "normal",
        CONFIG.pseudo_models["normal"],
        _make_caps(has_images=True, has_parallel_tools=True),
        CONFIG,
    )
    assert result.status == CompatibilityStatus.SAFE


def test_determinism_same_inputs_same_result():
    """Determinism: same inputs → same result (run twice)."""
    caps = _make_caps(has_images=True, has_tools=True)
    result1 = validate_switch(
        "vision",
        "normal",
        CONFIG.pseudo_models["normal"],
        caps,
        CONFIG,
    )
    result2 = validate_switch(
        "vision",
        "normal",
        CONFIG.pseudo_models["normal"],
        caps,
        CONFIG,
    )
    assert result1.status == result2.status
    assert result1.reason == result2.reason
    assert result1.remediation == result2.remediation


def test_warning_on_tools_strict_downgrade():
    """WARNING on tools strict downgrade."""
    # normal has tools_strict: true on deepseek-v4-flash
    # flash-lowcost has NO strict models
    result = validate_switch(
        "normal",
        "flash-lowcost",
        CONFIG.pseudo_models["flash-lowcost"],
        _make_caps(has_tools=True),
        CONFIG,
    )
    assert result.status == CompatibilityStatus.WARNING


def test_blocked_with_remediation():
    """BLOCKED always includes remediation options."""
    result = validate_switch(
        "vision",
        "tareas-avanzadas",
        CONFIG.pseudo_models["tareas-avanzadas"],
        _make_caps(has_images=True),
        CONFIG,
    )
    assert result.status == CompatibilityStatus.BLOCKED
    assert len(result.remediation) > 0
    # Each remediation should be a non-empty string
    for r in result.remediation:
        assert isinstance(r, str) and len(r) > 0
