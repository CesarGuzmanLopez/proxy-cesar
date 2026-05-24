"""Tests for input threshold guard.

Sprint 2 §9.4 — minimum 6 tests.
"""

from src.domain.errors import InputExceedsThreshold
from src.service.threshold_guard import check_input_threshold


def test_input_below_threshold():
    """Input below threshold → Ok."""
    result = check_input_threshold("normal", 96000, 50000)
    assert result.success is True


def test_input_above_threshold_without_pre_compaction():
    """Input above threshold without pre_compaction → Err."""
    result = check_input_threshold("normal", 96000, 97000, pre_compaction_enabled=False)
    assert result.success is False
    assert isinstance(result.error, InputExceedsThreshold)
    assert result.error.estimated == 97000
    assert result.error.threshold == 96000


def test_input_above_threshold_with_pre_compaction():
    """Input above threshold with pre_compaction → passes (deferred to Sprint 4)."""
    result = check_input_threshold("pensamiento-profundo-caro", 32000, 50000, pre_compaction_enabled=True)
    assert result.success is True


def test_compactador_null_threshold():
    """compactador (null threshold) → always passes."""
    result = check_input_threshold("compactador", None, 999999)
    assert result.success is True


def test_exact_threshold_boundary():
    """Exact threshold boundary → passes."""
    result = check_input_threshold("normal", 96000, 96000)
    assert result.success is True


def test_threshold_plus_one_fails():
    """Threshold + 1 token → fails."""
    result = check_input_threshold("normal", 96000, 96001)
    assert result.success is False
    assert isinstance(result.error, InputExceedsThreshold)
