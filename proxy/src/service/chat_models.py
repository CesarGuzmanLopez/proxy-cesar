"""Chat result models and proxy_metadata builder.

Sprint 1–4: DTOs for chat orchestration output.
python.md §3: dataclasses for error/data types.
python.md §4: pure functions, no side effects.
"""

from dataclasses import dataclass, field

from src.domain.capabilities import SessionCapabilities


@dataclass
class FallbackInfo:
    applied: bool = False
    reason: str | None = None
    attempted_models: list[str] = field(default_factory=list)


@dataclass
class ChatResult:
    conversation_id: str
    pseudo_model: str
    physical_model: str
    response: dict
    fallback_info: FallbackInfo
    is_new_conversation: bool
    affinity_maintained: bool
    total_tokens: int
    context_window: int | None

    # Sprint 2 fields
    session_caps: SessionCapabilities | None = None
    compatibility_warning: str | None = None
    compatibility_details: dict | None = None
    tools_filter_applied: bool = False
    tools_filter_reason: str | None = None

    # Sprint 3 fields
    tools_level_used: int = 0
    tools_incomplete: bool = False
    thinking_content: str | None = None
    tool_result_truncated: bool = False

    # Sprint 4 fields
    pre_compaction_applied: bool = False
    pre_compaction_metadata: dict | None = None
    continuous_compaction_applied: bool = False
    continuous_compaction_metadata: dict | None = None
    external_compaction_detected: bool = False
    external_compaction_metadata: dict | None = None


def build_proxy_metadata(
    pseudo_model: str,
    physical_model: str,
    conversation_id: str,
    context_tokens: int = 0,
    context_window: int | None = None,
    fallback_info: FallbackInfo | None = None,
    affinity_maintained: bool = True,
    *,
    # Sprint 2 fields
    session_caps: SessionCapabilities | None = None,
    compatibility_warning: str | None = None,
    compatibility_details: dict | None = None,
    tools_filter_applied: bool = False,
    tools_filter_reason: str | None = None,
    # Sprint 4 fields
    pre_compaction_applied: bool = False,
    pre_compaction_metadata: dict | None = None,
    continuous_compaction_applied: bool = False,
    continuous_compaction_metadata: dict | None = None,
    external_compaction_detected: bool = False,
    external_compaction_metadata: dict | None = None,
) -> dict:
    """Build proxy_metadata dict for API response.

    Sprint 1: basic fields (physical_model, pseudo_model, affinity, fallback).
    Sprint 2: +capabilities_detected, warning, tools_filter.
    Sprint 4: +pre_compaction, +continuous_compaction, +external_compaction.
    """
    metadata: dict = {
        "physical_model": physical_model,
        "pseudo_model": pseudo_model,
        "conversation_id": conversation_id,
        "affinity_maintained": affinity_maintained,
        "fallback_applied": fallback_info.applied if fallback_info else False,
        "fallback_reason": fallback_info.reason if fallback_info else None,
    }

    if context_window:
        metadata["context_tokens_total"] = context_tokens
        metadata["context_usage_pct"] = (
            round((context_tokens / context_window) * 100, 1) if context_window else None
        )
    else:
        metadata["context_tokens_total"] = context_tokens
        metadata["context_usage_pct"] = None

    # Sprint 2: capabilities detected
    if session_caps:
        metadata["capabilities_detected"] = {
            "has_images": session_caps.has_images,
            "has_tools": session_caps.has_tools,
        }
    else:
        metadata["capabilities_detected"] = None

    # Sprint 2: compatibility warning
    metadata["warning"] = compatibility_warning
    if compatibility_details:
        metadata["warning_details"] = compatibility_details

    # Sprint 2: tool filter
    metadata["tools_filter_applied"] = tools_filter_applied
    metadata["tools_filter_reason"] = tools_filter_reason

    # Sprint 4: pre-compaction
    metadata["pre_compaction_applied"] = pre_compaction_applied
    if pre_compaction_applied and pre_compaction_metadata:
        metadata["pre_compaction"] = {
            "original_input_tokens": pre_compaction_metadata.get("original_input_tokens", 0),
            "compacted_input_tokens": pre_compaction_metadata.get("compacted_input_tokens", 0),
            "compactor_model": pre_compaction_metadata.get("compactor_model", ""),
            "compactor_pseudo_model": pre_compaction_metadata.get("compactor_pseudo_model", ""),
            "savings_tokens": pre_compaction_metadata.get("savings_tokens", 0),
        }
    else:
        metadata["pre_compaction"] = None

    # Sprint 4: continuous compaction
    metadata["continuous_compaction_applied"] = continuous_compaction_applied
    if continuous_compaction_applied and continuous_compaction_metadata:
        metadata["continuous_compaction"] = {
            "tokens_before": continuous_compaction_metadata.get("tokens_before", 0),
            "tokens_after_snapshot": continuous_compaction_metadata.get("tokens_after", 0),
            "compactor_model": continuous_compaction_metadata.get("compactor_model", ""),
            "turns_compacted": continuous_compaction_metadata.get("turns_compacted", 0),
            "turns_preserved": continuous_compaction_metadata.get("turns_preserved", 0),
            "snapshot_id": continuous_compaction_metadata.get("snapshot_id", ""),
            "snapshot_type": continuous_compaction_metadata.get("snapshot_type", "continuous"),
        }
    else:
        metadata["continuous_compaction"] = None

    # Sprint 4: external compaction
    metadata["external_compaction_detected"] = external_compaction_detected
    if external_compaction_detected and external_compaction_metadata:
        metadata["external_compaction"] = external_compaction_metadata

    # Placeholders for future sprints
    metadata["router_suggestion"] = None
    metadata["images_described"] = 0

    return metadata



