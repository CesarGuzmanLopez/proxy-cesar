"""Chat result models and proxy_metadata builder.

python.md §3: dataclasses for error/data types.
python.md §4: pure functions, no side effects.
"""

from dataclasses import dataclass, field
from typing import Any

from src.domain.capabilities import SessionCapabilities
from src.service.context_alert import ContextAlert


@dataclass
class FallbackInfo:
    applied: bool = False
    reason: str | None = None
    attempted_models: list[str] = field(default_factory=list)


@dataclass
class StreamContext:
    """Context for streaming response generation — reduces params from 31 to 1."""

    litellm_response: Any
    conversation_id: str
    pseudo_model: str
    physical_model: str
    fallback_info: FallbackInfo | None = None
    affinity_maintained: bool = True
    context_window: int | None = None
    session_caps: Any = None
    compatibility_warning: str | None = None
    compatibility_details: dict | None = None
    tools_filter_applied: bool = False
    tools_filter_reason: str | None = None
    images_described: int = 0
    images_described_by: str | None = None
    router_suggestion: dict | None = None
    context_alert: ContextAlert | None = None
    cache_metadata: dict | None = None
    images_degraded_manually: bool = False
    db: Any = None
    conv: Any = None
    conv_uuid: Any = None
    turn_caps: Any = None
    provider: str | None = None
    messages: list | None = None
    tools: list | None = None
    tool_choice: str | dict | None = None
    resolved_model: str | None = None
    is_new: bool | None = None


@dataclass
class SaveContext:
    """Context for saving a turn and returning ChatResult — reduces params from 23 to 1."""

    db: Any
    conv: Any
    conv_uuid: Any
    conv_id: str
    pseudo_model_name: str
    physical_model: str
    provider: str | None
    turn_caps: SessionCapabilities
    messages: list[dict]
    response: Any
    fallback_info: FallbackInfo
    is_new_conversation: bool
    existing_affinity: str | None
    pm_schema: Any
    session_caps: SessionCapabilities
    tools: list[dict] | None
    tool_choice: str | dict | None
    tools_filter: dict
    compatibility: dict = field(default_factory=dict)
    images_described: int = 0
    images_described_by: str | None = None
    router_suggestion: dict | None = None
    context_alert: ContextAlert | None = None
    cache_metadata: dict | None = None


@dataclass
class MetadataContext:
    """Context for building proxy_metadata — reduces params from 22 to 1."""

    pseudo_model: str
    physical_model: str
    conversation_id: str
    context_tokens: int = 0
    context_window: int | None = None
    fallback_info: FallbackInfo | None = None
    affinity_maintained: bool = True
    session_caps: Any = None
    compatibility_warning: str | None = None
    compatibility_details: dict | None = None
    tools_filter_applied: bool = False
    tools_filter_reason: str | None = None
    images_described: int = 0
    images_described_by: str | None = None
    images_degraded_manually: bool = False
    router_suggestion: dict | None = None
    context_alert: ContextAlert | None = None
    cache_metadata: dict | None = None


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

    # Sprint 5 fields
    images_described: int = 0
    images_described_by: str | None = None
    images_degraded_manually: bool = False
    router_suggestion: dict | None = None

    # Sprint 6 fields
    context_alert: ContextAlert | None = None
    cache_metadata: dict | None = None


def build_proxy_metadata(ctx: MetadataContext) -> dict:
    """Build proxy_metadata dict for API response.

    Sprint 1: basic fields (physical_model, pseudo_model, affinity, fallback).
    Sprint 2: +capabilities_detected, warning, tools_filter.
    Sprint 5: +images_described, +images_described_by, +router_suggestion.
    """
    metadata: dict = {
        "physical_model": ctx.physical_model,
        "pseudo_model": ctx.pseudo_model,
        "conversation_id": ctx.conversation_id,
        "affinity_maintained": ctx.affinity_maintained,
        "fallback_applied": ctx.fallback_info.applied if ctx.fallback_info else False,
        "fallback_reason": ctx.fallback_info.reason if ctx.fallback_info else None,
    }

    metadata["context_tokens_total"] = ctx.context_tokens
    if ctx.context_window:
        metadata["context_usage_pct"] = round(
            (ctx.context_tokens / ctx.context_window) * 100, 1
        )
    else:
        metadata["context_usage_pct"] = None

    # Sprint 2: capabilities detected
    if ctx.session_caps:
        metadata["capabilities_detected"] = {
            "has_images": ctx.session_caps.has_images,
            "has_tools": ctx.session_caps.has_tools,
        }
    else:
        metadata["capabilities_detected"] = None

    # Sprint 2: compatibility warning
    metadata["warning"] = ctx.compatibility_warning
    if ctx.compatibility_details:
        metadata["warning_details"] = ctx.compatibility_details

    # Sprint 2: tool filter
    metadata["tools_filter_applied"] = ctx.tools_filter_applied
    metadata["tools_filter_reason"] = ctx.tools_filter_reason

    metadata["images_described"] = ctx.images_described
    metadata["images_described_by"] = ctx.images_described_by
    metadata["images_degraded_manually"] = ctx.images_degraded_manually

    metadata["router_suggestion"] = ctx.router_suggestion

    # Sprint 7: provider cache
    if ctx.cache_metadata:
        metadata["cache"] = ctx.cache_metadata

    # Sprint 6: context alerts
    if ctx.context_alert:
        alert_dict: dict = {
            "alert_level": ctx.context_alert.alert_level,
            "context_usage_pct": ctx.context_alert.context_usage_pct,
        }
        if ctx.context_alert.warning:
            alert_dict["warning"] = ctx.context_alert.warning
        if ctx.context_alert.compaction_endpoint:
            alert_dict["compaction_endpoint"] = ctx.context_alert.compaction_endpoint
        if ctx.context_alert.error_code:
            alert_dict["error_code"] = ctx.context_alert.error_code
        metadata["context_alert"] = alert_dict

    return metadata
