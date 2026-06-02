"""Database persistence for chat turns and response metadata.

Handles saving turns, extracting usage/metrics/cache metadata, and
suggesting higher-threshold models on input overflow.
"""

import copy
import logging

from sqlalchemy import func

from src.adapters.cache.provider_cache import (
    build_cache_destruction_metadata,
    build_cache_metadata,
)
from src.adapters.db.models import ConversationTurn
from src.api.metrics import metrics
from src.config.pseudo_models import ProxyConfigSchema
from src.service.capability_detector import accumulate_capabilities
from src.service.chat_models import ChatResult, FallbackInfo, SaveContext
from src.service.tools_canonical import (
    determine_tool_level_for_turn,
    extract_tool_calls_from_response,
    validate_tool_call_ids,
)
from src.service.tools_edge_cases import (
    enforce_tool_choice,
    extract_thinking_content,
    truncate_tool_result,
)

logger = logging.getLogger(__name__)

__all__ = [
    "_save_and_return",
    "_parse_usage",
    "_process_tool_metadata",
    "_extract_cache_metadata",
    "_suggest_higher_threshold_models",
]


# ── Token / Usage helpers ────────────────────────────────────────────────────


def _parse_usage(response_dict: dict) -> tuple[int, int]:
    """Extract prompt and completion tokens from response."""
    if not isinstance(response_dict, dict):
        return 0, 0
    usage = response_dict.get("usage", {})
    if not usage:
        return 0, 0
    return (
        usage.get("prompt_tokens", 0) or 0,
        usage.get("completion_tokens", 0) or 0,
    )


def _process_tool_metadata(response_dict: dict, ctx) -> dict:
    """Extract tool calls, validate, determine level, extract thinking."""
    tool_defs = ctx.tools
    tool_calls = extract_tool_calls_from_response(response_dict) if tool_defs else []

    if tool_calls:
        try:
            validate_tool_call_ids(tool_calls)
        except ValueError:
            ctx.turn_caps.tools_incomplete = True

    if (
        ctx.tool_choice == "required"
        and not enforce_tool_choice(response_dict, ctx.tool_choice)
    ):
        ctx.turn_caps.tools_incomplete = True

    thinking_content = extract_thinking_content(response_dict, ctx.provider)
    tools_incomplete = ctx.turn_caps.tools_incomplete
    tools_level = determine_tool_level_for_turn(
        tool_calls=tool_calls,
        tool_definitions=tool_defs,
        tools_incomplete=tools_incomplete,
    )

    return {
        "tool_calls": tool_calls,
        "tools_incomplete": tools_incomplete,
        "tools_level": tools_level,
        "thinking_content": thinking_content,
    }


# ── Cache metadata ───────────────────────────────────────────────────────────


def _extract_cache_metadata(
    response,
    provider: str,
    fallback_info: FallbackInfo,
) -> dict:
    """feature Extract cache hit/miss metadata from the provider response.

    Uses provider_cache.build_cache_metadata() for standard extraction and
    adds cache destruction info when fallback occurred.
    """
    response_dict = (
        response.model_dump() if hasattr(response, "model_dump") else response
    )
    if not isinstance(response_dict, dict):
        response_dict = {}

    cache_applied = getattr(response, "_proxy_cache_optimization_applied", False)
    meta = build_cache_metadata(response_dict, provider, cache_applied)

    if fallback_info.applied and fallback_info.attempted_models:
        prev_model = (
            fallback_info.attempted_models[0]
            if len(fallback_info.attempted_models) > 1
            else ""
        )
        new_model = fallback_info.attempted_models[-1]
        destruction = build_cache_destruction_metadata(
            previous_model=prev_model,
            new_model=new_model,
            previous_cached_tokens=meta.get("cached_tokens", 0),
        )
        meta["fallback_cache_destruction"] = destruction

    return meta


# ── Suggestion helper ────────────────────────────────────────────────────────


def _suggest_higher_threshold_models(
    config: ProxyConfigSchema,
    estimated_tokens: int,
) -> list[dict]:
    """Suggest pseudo-models with higher input_token_threshold."""
    suggestions = []
    for name, pm in config.pseudo_models.items():
        if (
            pm.input_token_threshold is not None
            and pm.input_token_threshold >= estimated_tokens
        ):
            suggestions.append(
                {
                    "pseudo_model": name,
                    "display_name": pm.display_name,
                    "input_token_threshold": pm.input_token_threshold,
                }
            )
    suggestions.sort(key=lambda x: int(x["input_token_threshold"] or 0))  # type: ignore[arg-type]  # justification: dict values are object; int cast is safe at runtime
    return suggestions


# ── Persist turn and return ChatResult ───────────────────────────────────────


async def _save_and_return(ctx: SaveContext) -> ChatResult:
    """Steps 14-20: Build turn, save to DB, accumulate capabilities, return result."""
    response_dict = (
        ctx.response.model_dump()
        if hasattr(ctx.response, "model_dump")
        else ctx.response
    )

    provider_headers = getattr(ctx.response, "_provider_response_headers", None)
    if provider_headers:
        response_dict["provider_headers"] = provider_headers

    input_tokens, output_tokens = _parse_usage(response_dict)
    tool_meta = _process_tool_metadata(response_dict, ctx)

    tool_defs = ctx.tools
    tools_incomplete = tool_meta["tools_incomplete"]
    tools_level = tool_meta["tools_level"]
    thinking_content = tool_meta["thinking_content"]

    tool_result_truncated = False
    truncated_messages = copy.deepcopy(ctx.messages)
    for msg in truncated_messages:
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            truncated = truncate_tool_result(content)
            if truncated != content:
                tool_result_truncated = True
                msg["content"] = truncated

    provider = ctx.provider or ""
    cache_meta = _extract_cache_metadata(ctx.response, provider, ctx.fallback_info)

    turn_number = 1
    if ctx.conv.turns:
        turn_number = max(t.turn_number for t in ctx.conv.turns) + 1

    turn = ConversationTurn(
        conversation_id=ctx.conv_uuid,
        turn_number=turn_number,
        pseudo_model=ctx.pseudo_model_name,
        physical_model=ctx.physical_model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        messages=truncated_messages,
        response=response_dict,
        fallback_applied=ctx.fallback_info.applied,
        fallback_reason=ctx.fallback_info.reason,
        turn_type="normal",
        had_images=ctx.turn_caps.has_images,
        had_tools=ctx.turn_caps.has_tools,
        had_parallel_tools=ctx.turn_caps.has_parallel_tools,
        tool_definitions=tool_defs,
        thinking_blocks={"content": thinking_content} if thinking_content else None,
        tools_incomplete=tools_incomplete,
        tools_level_used=tools_level,
    )
    ctx.db.add(turn)

    ctx.conv.physical_model = ctx.physical_model
    ctx.conv.total_tokens += input_tokens + output_tokens
    ctx.conv.updated_at = func.now()  # type: ignore[assignment]  # justification: SQLAlchemy server_default expression; validated at DB level

    updated_caps = await accumulate_capabilities(
        ctx.db, ctx.conv_uuid, ctx.turn_caps, ctx.session_caps  # type: ignore[arg-type]  # justification: SessionCapabilities/TurnCapabilities compatible at runtime; protocol boundary
    )

    await ctx.db.commit()

    metrics.record_tokens(input_tokens, output_tokens, input_tokens)
    if ctx.fallback_info.applied:
        metrics.record_fallback(ctx.fallback_info.reason or "unknown")

    # Affinity maintained only if:
    # 1. Not a new conversation
    # 2. Existing affinity model matches current physical model
    # 3. No fallback occurred (fallback breaks affinity even if end result matches)
    affinity_maintained = (
        not ctx.is_new_conversation
        and ctx.existing_affinity == ctx.physical_model
        and not ctx.fallback_info.applied
    )

    return ChatResult(
        conversation_id=ctx.conv_id,
        pseudo_model=ctx.pseudo_model_name,
        physical_model=ctx.physical_model,
        response=response_dict,
        fallback_info=ctx.fallback_info,
        is_new_conversation=ctx.is_new_conversation,
        affinity_maintained=affinity_maintained,
        total_tokens=ctx.conv.total_tokens,
        context_window=ctx.pm_schema.context_window,
        session_caps=updated_caps,
        compatibility_warning=ctx.compatibility.get("warning"),
        compatibility_details=ctx.compatibility.get("details"),
        tools_filter_applied=ctx.tools_filter["applied"],
        tools_filter_reason=ctx.tools_filter["reason"],
        tools_level_used=tools_level,
        tools_incomplete=tools_incomplete,
        thinking_content=thinking_content,
        tool_result_truncated=tool_result_truncated,
        images_described=ctx.images_described,
        images_described_by=ctx.images_described_by,
        router_suggestion=ctx.router_suggestion,
        context_alert=ctx.context_alert,
        cache_metadata=cache_meta,
    )
