"""Context alert service for Sprint 6.

plan-proxy.md §11.1: Warns user about context usage before it becomes critical.
python.md §3: Pure data types for alert results.
python.md §4: Pure functions, no side effects.
"""

from dataclasses import dataclass
from typing import Literal


type AlertLevel = Literal["normal", "moderate", "high", "unusable", "none"]


@dataclass(frozen=True, slots=True)
class ContextAlert:
    alert_level: AlertLevel
    context_usage_pct: float | None
    warning: str | None = None
    compaction_endpoint: str | None = None
    error_code: str | None = None


def get_context_alert(
    total_tokens: int,
    context_window: int | None,
    conversation_id: str,
) -> ContextAlert:
    """Determine the context alert level based on token usage.

    Pure function: no I/O, no side effects, deterministic.
    Used by both streaming and non-streaming paths.

    Thresholds:
      < 60%   → normal (informational only)
      60-80%  → moderate (warning + compaction endpoint)
      80-99%  → high (strong warning + compaction endpoint)
      100%+   → unusable (HTTP 400, feature available)
      None    → none (compactador pseudo-model, no window)

    Args:
        total_tokens: Current total tokens in the conversation.
        context_window: Context window of current pseudo-model (None if unknown).
        conversation_id: For building the compaction endpoint URL.

    Returns:
        ContextAlert with the computed alert level and metadata.
    """
    if context_window is None:
        return ContextAlert(alert_level="none", context_usage_pct=None)

    pct = round((total_tokens / context_window) * 100, 1)
    endpoint = f"POST /conversations/{conversation_id}/compact"

    if pct >= 100:
        return ContextAlert(
            alert_level="unusable",
            context_usage_pct=pct,
            warning="CONTEXT_UNUSABLE: History exceeds all available model windows. Compaction is the only available action.",
            compaction_endpoint=endpoint,
            error_code="CONTEXT_UNUSABLE",
        )

    if pct >= 80:
        return ContextAlert(
            alert_level="high",
            context_usage_pct=pct,
            warning=f"CONTEXT_HIGH: {pct}% of context window used. Compact recommended.",
            compaction_endpoint=endpoint,
        )

    if pct >= 60:
        return ContextAlert(
            alert_level="moderate",
            context_usage_pct=pct,
            warning=f"CONTEXT_MODERATE: {pct}% of context window used. Consider compacting soon.",
            compaction_endpoint=endpoint,
        )

    return ContextAlert(alert_level="normal", context_usage_pct=pct)
