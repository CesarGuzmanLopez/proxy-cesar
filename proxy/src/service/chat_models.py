"""Chat result models and proxy_metadata builder.

python.md §3: dataclasses for error/data types.
python.md §4: pure functions, no side effects.
"""

from dataclasses import dataclass, field
from uuid import UUID

from litellm.types.utils import ModelResponse

from src.adapters.db.models import Conversation
from src.config.pseudo_models import PseudoModelSchema
from src.domain.capabilities import SessionCapabilities, TurnCapabilities
from src.domain.ports import AsyncSessionPort
from src.service.context_alert import ContextAlert
from src.service.pipeline_trace import PipelineTrace


@dataclass
class FallbackInfo:
    applied: bool = False
    reason: str | None = None
    attempted_models: list[str] = field(default_factory=list)


@dataclass
class StreamContext:
    """Context for streaming response generation — reduces params from 31 to 1."""

    litellm_response: ModelResponse
    conversation_id: str
    pseudo_model: str
    physical_model: str
    fallback_info: FallbackInfo | None = None
    affinity_maintained: bool = True
    context_window: int | None = None
    session_caps: SessionCapabilities | None = None
    compatibility_warning: str | None = None
    compatibility_details: dict | None = None
    tools_filter_applied: bool = False
    tools_filter_reason: str | None = None
    images_described: int = 0
    images_described_by: str | None = None
    pdfs_analyzed: int = 0
    audios_transcribed: int = 0
    documents_processed: int = 0
    router_suggestion: dict | None = None
    context_alert: ContextAlert | None = None
    cache_metadata: dict | None = None
    images_degraded_manually: bool = False
    db: AsyncSessionPort | None = None
    conv: Conversation | None = None
    conv_uuid: UUID | None = None
    turn_caps: TurnCapabilities | None = None
    provider: str | None = None
    messages: list | None = None
    tools: list | None = None
    tool_choice: str | dict | None = None
    resolved_model: str | None = None
    is_new: bool | None = None
    # feature fields for token-limit fallback continuation
    pm_schema: PseudoModelSchema | None = None
    """Pseudo-model schema with physical_models list — needed for continuing."""
    call_kwargs: dict | None = None
    """Kwargs to pass when calling the next model for continuation."""
    active_messages: list | None = None
    """Full assembled message list (history + current) for continuation context."""
    trace: PipelineTrace | None = None
    """Pipeline trace for observability — logs LLM in/out events."""
    timeout: float | None = None
    """LLM call timeout in seconds — propagated to _try_physical_model for continuation."""


@dataclass
class SaveContext:
    """Context for saving a turn and returning ChatResult — reduces params from 23 to 1."""

    db: AsyncSessionPort
    conv: Conversation
    conv_uuid: UUID
    conv_id: str
    pseudo_model_name: str
    physical_model: str
    provider: str | None
    turn_caps: TurnCapabilities | SessionCapabilities
    messages: list[dict]
    response: ModelResponse | dict
    fallback_info: FallbackInfo
    is_new_conversation: bool
    existing_affinity: str | None
    pm_schema: PseudoModelSchema
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
class StreamingRequestContext:
    """Context for streaming request setup — reduces params from 15 to 1."""

    config: object
    affinity: object
    db_session_factory: object
    conversation_id: str
    pseudo_model_name: str
    messages: list[dict]
    stream: bool = True
    temperature: float | None = None
    max_tokens: int | None = None
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None
    stream_options: dict | None = None
    thinking: dict | str | bool | None = None
    trace: PipelineTrace | None = None
    request: object | None = None


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
    session_caps: SessionCapabilities | None = None
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

    # Featurefields
    session_caps: SessionCapabilities | None = None
    compatibility_warning: str | None = None
    compatibility_details: dict | None = None
    tools_filter_applied: bool = False
    tools_filter_reason: str | None = None

    # Featurefields
    tools_level_used: int = 0
    tools_incomplete: bool = False
    thinking_content: str | None = None
    tool_result_truncated: bool = False

    # Compaction feature fields
    pre_compaction_applied: bool = False
    pre_compaction_metadata: dict | None = None
    continuous_compaction_applied: bool = False
    continuous_compaction_metadata: dict | None = None
    external_compaction_detected: bool = False
    external_compaction_metadata: dict | None = None

    # Featurefields
    images_described: int = 0
    images_described_by: str | None = None
    images_degraded_manually: bool = False
    router_suggestion: dict | None = None

    # Featurefields
    context_alert: ContextAlert | None = None
    cache_metadata: dict | None = None


def build_proxy_metadata(ctx: MetadataContext) -> dict:
    """Build proxy_metadata dict for API response.

    feature basic fields (physical_model, pseudo_model, affinity, fallback).
    feature +capabilities_detected, warning, tools_filter.
    feature +images_described, +images_described_by, +router_suggestion.
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

    # feature capabilities detected
    if ctx.session_caps:
        metadata["capabilities_detected"] = {
            "has_images": ctx.session_caps.has_images,
            "has_tools": ctx.session_caps.has_tools,
        }
    else:
        metadata["capabilities_detected"] = None

    # feature compatibility warning
    metadata["warning"] = ctx.compatibility_warning
    if ctx.compatibility_details:
        metadata["warning_details"] = ctx.compatibility_details

    # feature tool filter
    metadata["tools_filter_applied"] = ctx.tools_filter_applied
    metadata["tools_filter_reason"] = ctx.tools_filter_reason

    metadata["images_described"] = ctx.images_described
    metadata["images_described_by"] = ctx.images_described_by
    metadata["images_degraded_manually"] = ctx.images_degraded_manually

    metadata["router_suggestion"] = ctx.router_suggestion

    # feature provider cache
    if ctx.cache_metadata:
        metadata["cache"] = ctx.cache_metadata

    # feature context alerts
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
