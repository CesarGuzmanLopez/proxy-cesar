"""Chat orchestration service.

Central business logic for POST /v1/chat/completions.
Uses Result monad for domain errors, HTTPException for transport errors (routers only).
python.md §3: errors as data in domain, exceptions at boundary.

Sprint 1: basic pseudo-model resolution, affinity, fallback.
Sprint 2: +capability detection, compatibility validation, threshold guard, tool filter.
Sprint 3: +canonical tool storage, tiktoken, thinking blocks, tool edge cases.
"""

import uuid
from dataclasses import dataclass, field

from fastapi import HTTPException
from litellm.exceptions import RateLimitError, ServiceUnavailableError
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy import func

from src.adapters.db.models import Conversation, ConversationTurn
from src.adapters.litellm import call_litellm
from src.adapters.cache.valkey_affinity import ValkeyAffinityAdapter
from src.config.pseudo_models import ProxyConfigSchema
from src.domain.capabilities import SessionCapabilities
from src.service.model_resolver import normalize_model_name
from src.service.capability_detector import (
    accumulate_capabilities,
    detect_turn_capabilities,
    estimate_tokens,
    load_session_capabilities,
)
from src.service.compatibility import validate_incoming_content, validate_switch
from src.service.threshold_guard import check_input_threshold
from src.service.tool_filter import get_eligible_models, is_pinned_model_eligible
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


async def process_chat_request(
    model: str,
    messages: list[dict],
    conversation_id: str | None,
    stream: bool,
    config: ProxyConfigSchema,
    affinity: ValkeyAffinityAdapter,
    db: AsyncSession,
    temperature: float | None = None,
    max_tokens: int | None = None,
    tools: list[dict] | None = None,
    tool_choice: str | dict | None = None,
) -> ChatResult:
    """Execute the full chat completion flow with Sprint 2 capability checks.

    Steps:
    1. Normalize + validate pseudo-model
    2. Detect capabilities in incoming messages
    3. Validate incoming content against current pseudo-model
    4. Get existing affinity (if any)
    5. Load existing session capabilities
    6. Check if pseudo-model changed → validate switch
    7. Filter pool by parallel_tools if needed
    8. Resolve physical model
    9. Load or create conversation
    10. Set affinity
    11. Check input threshold
    12. Call LiteLLM with fallback
    13. Save turn with capability flags
    14. Accumulate capabilities in DB
    15. Return result with proxy_metadata
    """
    # Step 1: Normalize model name
    resolved_model = normalize_model_name(model, config)
    if resolved_model not in config.pseudo_models:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "UNKNOWN_PSEUDO_MODEL",
                "message": f"Unknown pseudo-model: '{resolved_model}'",
                "available": list(config.pseudo_models.keys()),
            },
        )

    pseudo_model_name = resolved_model
    conv_id = conversation_id or str(uuid.uuid4())
    pm_schema = config.pseudo_models[pseudo_model_name]

    # Step 2: Detect capabilities in incoming messages
    turn_caps = detect_turn_capabilities(messages, tools)

    # Step 3: Validate incoming content against current pseudo-model
    validate_incoming_content(turn_caps, pm_schema, pseudo_model_name, config)

    # Step 4: Get existing affinity
    existing_affinity = await affinity.get(conv_id)

    # Step 5: Resolve conversation ID for DB
    try:
        conv_uuid = uuid.UUID(conv_id)
    except ValueError:
        conv_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, conv_id)

    # Step 6: Load existing session capabilities (if conversation exists)
    session_caps = await load_session_capabilities(db, conv_uuid)

    # Step 7: Check if pseudo-model changed → validate switch
    compatibility_warning: str | None = None
    compatibility_details: dict | None = None

    conv = await db.get(Conversation, conv_uuid)
    if conv is not None and conv.pseudo_model != pseudo_model_name:
        # Pseudo-model switch detected
        switch_result = validate_switch(
            from_pseudo_name=conv.pseudo_model,
            to_pseudo_name=pseudo_model_name,
            to_pseudo=pm_schema,
            caps=session_caps,
            config=config,
        )
        if switch_result.status.value == "blocked":
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "PSEUDO_MODEL_INCOMPATIBLE",
                    "message": switch_result.reason,
                    "remediation": switch_result.remediation,
                    "details": switch_result.details,
                    "from_pseudo_model": conv.pseudo_model,
                    "to_pseudo_model": pseudo_model_name,
                },
            )
        if switch_result.status.value == "warning":
            compatibility_warning = switch_result.reason
            compatibility_details = switch_result.details

    # Step 8: Filter pool by parallel_tools if needed
    tools_filter_applied = False
    tools_filter_reason: str | None = None

    eligible_models = get_eligible_models(pm_schema.physical_models, session_caps)

    if session_caps.has_parallel_tools and len(eligible_models) < len(
        pm_schema.physical_models
    ):
        tools_filter_applied = True
        tools_filter_reason = "parallel_tools_required"

    # Step 9: Resolve physical model
    # If pinned model exists but is not eligible, use first eligible model
    pinned_model = existing_affinity
    if pinned_model and session_caps.has_parallel_tools:
        if not is_pinned_model_eligible(pinned_model, eligible_models):
            pinned_model = None  # Force re-resolve

    if pinned_model is None:
        # Use first eligible model (or first overall if no filtering)
        pinned_model = eligible_models[0].model if eligible_models else pm_schema.physical_models[0].model

    physical_model = pinned_model

    # Step 10: Load or create conversation
    is_new_conversation = conv is None

    if is_new_conversation:
        conv = Conversation(
            id=conv_uuid,
            pseudo_model=pseudo_model_name,
            physical_model=physical_model,
            total_tokens=0,
        )
        db.add(conv)
        await db.flush()

    # Step 11: Set affinity
    await affinity.set(conv_id, physical_model)

    # Step 12: Check input threshold
    estimated_input = estimate_tokens(messages)
    threshold_check = check_input_threshold(
        pseudo_model_name=pseudo_model_name,
        input_token_threshold=pm_schema.input_token_threshold,
        estimated_tokens=estimated_input,
        pre_compaction_enabled=pm_schema.pre_compaction.enabled,
    )

    if not threshold_check.ok:
        # InputExceedsThreshold
        error = threshold_check.error
        raise HTTPException(
            status_code=400,
            detail={
                "error": "INPUT_EXCEEDS_THRESHOLD",
                "message": (
                    f"Input ({error.estimated} tokens) exceeds threshold "
                    f"({error.threshold} tokens) for pseudo-model "
                    f"'{pm_schema.display_name}'."
                ),
                "suggestions": _suggest_higher_threshold_models(config, error.estimated),
            },
        )

    # Step 13: Call LiteLLM with fallback
    response, fallback_info = await call_with_fallback(
        pseudo_model_schema=pm_schema,
        messages=messages,
        stream=stream,
        temperature=temperature,
        max_tokens=max_tokens,
        tools=tools,
        tool_choice=tool_choice,
    )

    # Step 14: Save turn with Sprint 3 canonical tool metadata
    turn_number = 1
    if conv.turns:
        turn_number = max(t.turn_number for t in conv.turns) + 1

    input_tokens = 0
    output_tokens = 0
    response_dict = response.model_dump() if hasattr(response, "model_dump") else response
    if isinstance(response_dict, dict):
        usage = response_dict.get("usage", {})
        if usage:
            input_tokens = usage.get("prompt_tokens", 0) or 0
            output_tokens = usage.get("completion_tokens", 0) or 0

    # Sprint 3: Extract tool metadata from response
    tool_calls_in_response = extract_tool_calls_from_response(response_dict)
    tool_definitions_stored: list[dict] | None = None
    if tools:
        tool_definitions_stored = tools
        tool_calls_in_response = extract_tool_calls_from_response(response_dict)

    # Validate tool call IDs if present
    if tool_calls_in_response:
        try:
            validate_tool_call_ids(tool_calls_in_response)
        except ValueError:
            # If validation fails, log and continue without tool data
            turn_caps.tools_incomplete = True

    # Check tool_choice enforcement
    if tool_choice == "required" and tool_calls_in_response:
        if not enforce_tool_choice(response_dict, tool_choice):
            # The model ignored tool_choice. Force fallback will handle it.
            turn_caps.tools_incomplete = True

    # Determine provider for thinking block extraction
    provider = None
    if physical_model:
        if "deepseek" in physical_model.lower():
            provider = "deepseek"
        elif "claude" in physical_model.lower() or "anthropic" in physical_model.lower():
            provider = "anthropic"
        elif "gemini" in physical_model.lower():
            provider = "google"
        elif "gpt" in physical_model.lower() or "o3" in physical_model.lower():
            provider = "openai"

    thinking_content = extract_thinking_content(response_dict, provider)

    tools_incomplete = turn_caps.tools_incomplete
    tools_level_used = determine_tool_level_for_turn(
        tool_calls=tool_calls_in_response,
        tool_definitions=tool_definitions_stored,
        tools_incomplete=tools_incomplete,
    )

    # Check for large tool results in the messages
    tool_result_truncated = False
    for msg in messages:
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            truncated = truncate_tool_result(content)
            if truncated != content:
                tool_result_truncated = True
                msg["content"] = truncated

    turn = ConversationTurn(
        conversation_id=conv_uuid,
        turn_number=turn_number,
        pseudo_model=pseudo_model_name,
        physical_model=physical_model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        messages=messages,
        response=response_dict,
        fallback_applied=fallback_info.applied,
        fallback_reason=fallback_info.reason,
        # Sprint 2: capability flags on turn
        turn_type="normal",
        had_images=turn_caps.has_images,
        had_tools=turn_caps.has_tools,
        had_parallel_tools=turn_caps.has_parallel_tools,
        # Sprint 3: canonical tool storage
        tool_definitions=tool_definitions_stored,
        thinking_blocks={"content": thinking_content} if thinking_content else None,
        tools_incomplete=tools_incomplete,
        tools_level_used=tools_level_used,
    )
    db.add(turn)

    # Step 15: Update conversation
    conv.physical_model = physical_model
    conv.total_tokens += input_tokens + output_tokens
    conv.updated_at = func.now()

    await db.commit()

    # Step 16: Accumulate capabilities in DB
    session_caps = await accumulate_capabilities(
        db, conv_uuid, turn_caps, session_caps
    )

    affinity_maintained = not is_new_conversation and existing_affinity == physical_model

    return ChatResult(
        conversation_id=conv_id,
        pseudo_model=pseudo_model_name,
        physical_model=physical_model,
        response=response_dict,
        fallback_info=fallback_info,
        is_new_conversation=is_new_conversation,
        affinity_maintained=affinity_maintained,
        total_tokens=conv.total_tokens,
        context_window=pm_schema.context_window,
        session_caps=session_caps,
        compatibility_warning=compatibility_warning,
        compatibility_details=compatibility_details,
        tools_filter_applied=tools_filter_applied,
        tools_filter_reason=tools_filter_reason,
        # Sprint 3
        tools_level_used=tools_level_used,
        tools_incomplete=tools_incomplete,
        thinking_content=thinking_content,
        tool_result_truncated=tool_result_truncated,
    )


async def call_with_fallback(
    pseudo_model_schema,
    messages: list[dict],
    stream: bool = False,
    **kwargs,
) -> tuple:
    """Try each physical model in order. On 503/429, move to next.

    Fallback strategy: sequential (Sprint 1 only).
    Returns (response, FallbackInfo).
    """
    fallback_info = FallbackInfo()
    last_error: Exception | None = None

    for phys in pseudo_model_schema.physical_models:
        try:
            response = await call_litellm(
                model=phys.model,
                messages=messages,
                stream=stream,
                **{k: v for k, v in kwargs.items() if v is not None},
            )
            fallback_info.attempted_models.append(phys.model)
            return response, fallback_info
        except (ServiceUnavailableError, RateLimitError) as e:
            last_error = e
            fallback_info.attempted_models.append(phys.model)
            fallback_info.applied = True
            fallback_info.reason = f"{type(e).__name__}: {phys.model}"
            continue
        except Exception:
            raise

    all_attempted = tuple(fallback_info.attempted_models)
    raise HTTPException(
        status_code=503,
        detail={
            "error": "ALL_MODELS_FAILED",
            "message": (
                f"All models for pseudo-model "
                f"'{pseudo_model_schema.display_name}' failed."
            ),
            "attempted": list(all_attempted),
            "last_error": str(last_error),
        },
    )


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
) -> dict:
    """Build proxy_metadata dict for API response.

    Sprint 1: basic fields (physical_model, pseudo_model, affinity, fallback).
    Sprint 2: +capabilities_detected, warning, tools_filter.
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

    # Placeholders for future sprints (unchanged from Sprint 1)
    metadata["pre_compaction_applied"] = False
    metadata["continuous_compaction_applied"] = False
    metadata["router_suggestion"] = None
    metadata["images_described"] = 0

    return metadata


def _suggest_higher_threshold_models(
    config: ProxyConfigSchema,
    estimated_tokens: int,
) -> list[dict]:
    """Suggest pseudo-models with higher input_token_threshold."""
    suggestions = []
    for name, pm in config.pseudo_models.items():
        if pm.input_token_threshold is not None and pm.input_token_threshold >= estimated_tokens:
            suggestions.append({
                "pseudo_model": name,
                "display_name": pm.display_name,
                "input_token_threshold": pm.input_token_threshold,
            })
    suggestions.sort(key=lambda x: x["input_token_threshold"])
    return suggestions
