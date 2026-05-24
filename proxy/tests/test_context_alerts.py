"""Tests for context alert service.

Sprint 6 §2: Alert thresholds and CONTEXT_UNUSABLE.
Minimum 9 tests per sprint spec.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.service.context_alert import get_context_alert, ContextAlert


# ── Pure function tests (no mocks needed) ────────────────────────────────


def test_context_below_60_normal():
    """Context < 60% → normal alert level, no warning."""
    alert = get_context_alert(total_tokens=30000, context_window=100000, conversation_id="test-conv")
    assert alert.alert_level == "normal"
    assert alert.context_usage_pct == 30.0
    assert alert.warning is None
    assert alert.error_code is None


def test_context_60_to_80_moderate():
    """Context 60-80% → moderate alert with warning message."""
    alert = get_context_alert(total_tokens=70000, context_window=100000, conversation_id="test-conv")
    assert alert.alert_level == "moderate"
    assert alert.context_usage_pct == 70.0
    assert alert.warning is not None
    assert "compacting soon" in alert.warning
    assert alert.compaction_endpoint is not None
    assert "test-conv" in (alert.compaction_endpoint or "")


def test_context_80_to_99_high():
    """Context 80-99% → high alert with strong warning."""
    alert = get_context_alert(total_tokens=90000, context_window=100000, conversation_id="test-conv")
    assert alert.alert_level == "high"
    assert alert.context_usage_pct == 90.0
    assert alert.warning is not None
    assert "Compact recommended" in alert.warning
    assert alert.compaction_endpoint is not None


def test_context_exactly_100_unusable():
    """Context exactly at 100% → unusable alert level."""
    alert = get_context_alert(total_tokens=100000, context_window=100000, conversation_id="test-conv")
    assert alert.alert_level == "unusable"
    assert alert.context_usage_pct == 100.0
    assert alert.error_code == "CONTEXT_UNUSABLE"
    assert alert.warning is not None


def test_context_above_100_unusable():
    """Context > 100% → unusable alert level with error code."""
    alert = get_context_alert(total_tokens=150000, context_window=100000, conversation_id="test-conv")
    assert alert.alert_level == "unusable"
    assert alert.context_usage_pct == 150.0
    assert alert.error_code == "CONTEXT_UNUSABLE"
    assert alert.compaction_endpoint is not None


def test_context_null_window_none():
    """No context_window (compactador) → 'none' alert level."""
    alert = get_context_alert(total_tokens=50000, context_window=None, conversation_id="test-conv")
    assert alert.alert_level == "none"
    assert alert.context_usage_pct is None


def test_context_zero_tokens():
    """Zero tokens → normal alert at 0%."""
    alert = get_context_alert(total_tokens=0, context_window=100000, conversation_id="test-conv")
    assert alert.alert_level == "normal"
    assert alert.context_usage_pct == 0.0


def test_context_rounding():
    """Context usage percentage is rounded to 1 decimal."""
    alert = get_context_alert(total_tokens=12345, context_window=100000, conversation_id="test-conv")
    assert alert.context_usage_pct == 12.3  # Rounds to 1 decimal
    assert alert.alert_level == "normal"


def test_context_edge_59_5_percent():
    """Context at 59.5% → still normal (below 60% threshold)."""
    alert = get_context_alert(total_tokens=59500, context_window=100000, conversation_id="test-conv")
    assert alert.alert_level == "normal"
    assert alert.context_usage_pct == 59.5


def test_context_edge_60_percent():
    """Context exactly at 60% → moderate (at threshold)."""
    alert = get_context_alert(total_tokens=60000, context_window=100000, conversation_id="test-conv")
    assert alert.alert_level == "moderate"


def test_context_edge_80_percent():
    """Context exactly at 80% → high (at threshold)."""
    alert = get_context_alert(total_tokens=80000, context_window=100000, conversation_id="test-conv")
    assert alert.alert_level == "high"


# ── ContextAlert dataclass immutability ──────────────────────────────────


def test_context_alert_frozen():
    """ContextAlert dataclass is frozen (immutable)."""
    alert = ContextAlert(alert_level="normal", context_usage_pct=30.0)
    with pytest.raises(AttributeError):
        alert.alert_level = "high"  # type: ignore[misc]
