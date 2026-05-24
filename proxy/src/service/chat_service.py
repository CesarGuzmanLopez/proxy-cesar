"""Chat orchestration service.

Central business logic for POST /v1/chat/completions.
Uses Result monad for domain errors, HTTPException for transport errors (routers only).
python.md §3: errors as data in domain, exceptions at boundary.

Sprint 1: basic pseudo-model resolution, affinity, fallback.
Sprint 2: +capability detection, compatibility validation, threshold guard, tool filter.
Sprint 3: +canonical tool storage, tiktoken, thinking blocks, tool edge cases.
Sprint 4: +pre-compaction, continuous compaction, external compaction detection.
"""

import uuid

from fastapi import HTTPException
from litellm.exceptions import RateLimitError, ServiceUnavailableError
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy import func
from sqlalchemy.orm import selectinload

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
from src.service.compatibility import (
    validate_incoming_content,
    validate_switch,
)
from src.service.compatibility import _any_vision as _any_vision_comp
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
from src.service.compactor.pre_compactor import pre_compact_input
from src.service.compactor.continuous import (
    assemble_context,
    continuous_compact,
    detect_external_compaction,
    handle_external_compaction,
)
from src.service.multimedia.image_describer import auto_describe_images
from src.service.router_llm.suggester import evaluate_complexity, is_downgrade
from src.service.chat_models import ChatResult, FallbackInfo, SaveContext


# ── Orchestrator ──────────────────────────────────────────────────────────────


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
    """Execute the full chat completion flow.

    Each logical step is delegated to a focused helper function.
    """
    # Step 1-3: Resolve model + detect capabilities + validate content
    pseudo_model_name, pm_schema, turn_caps = _resolve_and_validate(
        model, messages, tools, config,
    )
    conv_id = conversation_id or str(uuid.uuid4())
    conv_uuid = _parse_uuid(conv_id)

    # Step 4-11: Load session, check switch, load/create conversation, resolve model, set affinity
    existing_affinity, session_caps, compatibility, physical_model, provider, tools_filter, conv, is_new = (
        await _resolve_session_conv_and_models(
            db, affinity, conv_id, conv_uuid, pseudo_model_name, pm_schema, config,
        )
    )
    await affinity.set(conv_id, physical_model)

    # Sprint 5: Auto-describe images on pseudo-model switch
    auto_describe_meta: dict | None = None
    if conv is not None and not is_new and pseudo_model_name != conv.pseudo_model:
        auto_describe_meta = await handle_auto_describe(
            conv=conv,
            current_pseudo_name=conv.pseudo_model,
            new_pm_schema=pm_schema,
            config=config,
            db=db,
            pinned_physical_model=physical_model,
        )

    # Step 12: Check input threshold
    est_input = estimate_tokens(messages)
    _raise_if_exceeds_threshold(est_input, pm_schema, pseudo_model_name, config)

    # Sprint 4: Pre-compaction, external detection, continuous compaction
    prep = await _apply_compaction(
        conv, is_new, messages, pm_schema, config, db, est_input,
    )
    active_messages = prep["active_messages"]

    # Sprint 5: Router LLM — evaluate complexity (non-blocking, never changes model)
    # SAFETY: evaluate_complexity() internally extracts ONLY the last user message.
    # The full message array is passed for structural context, but the prompt
    # sent to the evaluator contains only the last user text (MAX_TASK_CHARS=2000).
    router_suggestion: dict | None = await evaluate_router_suggestion(
        pm_schema=pm_schema,
        messages=active_messages,
        current_pseudo_name=pseudo_model_name,
        config=config,
    )

    # Step 13: Call LiteLLM with fallback
    response, fallback_info = await call_with_fallback(
        pseudo_model_schema=pm_schema,
        messages=active_messages,
        stream=stream,
        temperature=temperature,
        max_tokens=max_tokens,
        tools=tools,
        tool_choice=tool_choice,
    )

    # Compute Sprint 5 metadata
    images_described = 0
    images_described_by: str | None = None
    if auto_describe_meta:
        images_described = auto_describe_meta.get("images_described", 0)
        images_described_by = auto_describe_meta.get("described_by")

    return await _save_and_return(SaveContext(
        db=db,
        conv=conv,
        conv_uuid=conv_uuid,
        conv_id=conv_id,
        pseudo_model_name=pseudo_model_name,
        physical_model=physical_model,
        provider=provider,
        turn_caps=turn_caps,
        messages=messages,
        response=response,
        fallback_info=fallback_info,
        is_new_conversation=is_new,
        existing_affinity=existing_affinity,
        pm_schema=pm_schema,
        session_caps=session_caps,
        tools=tools,
        tool_choice=tool_choice,
        prep=prep,
        compatibility=compatibility,
        tools_filter=tools_filter,
        images_described=images_described,
        images_described_by=images_described_by,
        router_suggestion=router_suggestion,
    ))


# ── Step helpers ──────────────────────────────────────────────────────────────


def _resolve_and_validate(
    model: str,
    messages: list[dict],
    tools: list[dict] | None,
    config: ProxyConfigSchema,
) -> tuple[str, object, SessionCapabilities]:
    """Steps 1-3: Normalize model, detect capabilities, validate content."""
    resolved = normalize_model_name(model, config)
    if resolved not in config.pseudo_models:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "UNKNOWN_PSEUDO_MODEL",
                "message": f"Unknown pseudo-model: '{resolved}'",
                "available": list(config.pseudo_models.keys()),
            },
        )
    pm = config.pseudo_models[resolved]
    caps = detect_turn_capabilities(messages, tools)
    validate_incoming_content(caps, pm, resolved, config)
    return resolved, pm, caps


def _parse_uuid(conv_id: str) -> uuid.UUID:
    """Parse conversation ID string to UUID."""
    try:
        return uuid.UUID(conv_id)
    except ValueError:
        return uuid.uuid5(uuid.NAMESPACE_DNS, conv_id)


async def _resolve_session_conv_and_models(
    db: AsyncSession,
    affinity: ValkeyAffinityAdapter,
    conv_id: str,
    conv_uuid: uuid.UUID,
    pseudo_model_name: str,
    pm_schema: object,
    config: ProxyConfigSchema,
) -> tuple:
    """Steps 4-11: Load/create conversation, session, check switch, resolve physical model.

    Merged from _resolve_session_and_models + _load_or_create_conv
    to avoid loading Conversation twice (and to eagerly load .turns).
    """
    existing_affinity = await affinity.get(conv_id)

    # Load conversation FIRST with turns eagerly loaded.
    # Must happen BEFORE load_session_capabilities to avoid the identity-map issue:
    # if load_session_capabilities loads Conversation (without selectinload) first,
    # the subsequent db.get with selectinload returns the same object without eager loads.
    conv = await db.get(Conversation, conv_uuid, options=[selectinload(Conversation.turns)])
    is_new = conv is None

    # Load session capabilities (uses identity-mapped conv, no extra DB trip)
    session_caps = await load_session_capabilities(db, conv_uuid)
    compatibility_warning: str | None = None
    compatibility_details: dict | None = None

    if conv is not None and conv.pseudo_model != pseudo_model_name:
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

    eligible = get_eligible_models(pm_schema.physical_models, session_caps)
    tools_filter_applied = bool(
        session_caps.has_parallel_tools and len(eligible) < len(pm_schema.physical_models)
    )
    tools_filter_reason = "parallel_tools_required" if tools_filter_applied else None

    pinned = existing_affinity
    if pinned and session_caps.has_parallel_tools and not is_pinned_model_eligible(pinned, eligible):
        pinned = None
    if pinned:
        selected_phys = next(
            (p for p in pm_schema.physical_models if p.model == pinned),
            pm_schema.physical_models[0],
        )
    elif eligible:
        selected_phys = eligible[0]
    else:
        selected_phys = pm_schema.physical_models[0]
    physical = selected_phys.model
    provider = selected_phys.provider

    if is_new:
        conv = Conversation(
            id=conv_uuid,
            pseudo_model=pseudo_model_name,
            physical_model=physical,
            total_tokens=0,
        )
        conv.turns = []  # prevent lazy-load trigger outside greenlet context
        db.add(conv)
        await db.flush()

    return (
        existing_affinity,
        session_caps,
        {"warning": compatibility_warning, "details": compatibility_details},
        physical,
        provider,
        {"applied": tools_filter_applied, "reason": tools_filter_reason},
        conv,
        is_new,
    )


def _raise_if_exceeds_threshold(
    estimated_input: int,
    pm_schema: object,
    pseudo_model_name: str,
    config: ProxyConfigSchema,
) -> None:
    """Step 12: Raise 400 if input exceeds threshold (no pre-compaction)."""
    check = check_input_threshold(
        pseudo_model_name=pseudo_model_name,
        input_token_threshold=pm_schema.input_token_threshold,
        estimated_tokens=estimated_input,
        pre_compaction_enabled=pm_schema.pre_compaction.enabled,
    )
    if not check.success:
        error = check.error
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


async def handle_auto_describe(
    conv: Conversation,
    current_pseudo_name: str,
    new_pm_schema: object,
    config,
    db: AsyncSession,
    pinned_physical_model: str,
) -> dict | None:
    """Execute auto-describe when switching from vision to non-vision model.

    Shared between streaming (``api/chat.py``) and non-streaming
    (``chat_service.py``) paths to eliminate code duplication.

    Runs after ``validate_switch()`` returns WARNING with
    ``IMAGES_WILL_BE_DESCRIBED``. Describes all images in the conversation
    history using the current vision model and stores a degradation_event turn.

    Args:
        conv: Current conversation (with turns loaded).
        current_pseudo_name: Current pseudo-model name (vision-capable).
        new_pm_schema: Target pseudo-model schema.
        config: Proxy config with all pseudo-models.
        db: DB session.
        pinned_physical_model: Currently pinned physical model.

    Returns:
        Metadata dict with ``images_described``, ``described_by``, etc.
        ``None`` if auto-describe is not needed or not possible.
    """
    # Check destination has auto_describe enabled
    if new_pm_schema.image_handling.on_downgrade != "auto_describe":
        return None

    # Check destination lacks vision (that's why we're describing)
    if _any_vision_comp(new_pm_schema.physical_models):
        return None  # Destination has vision — no describe needed

    # Find vision model in current pseudo-model
    current_pm = config.pseudo_models.get(current_pseudo_name)
    if current_pm is None:
        return None

    vision_models = [m for m in current_pm.physical_models if m.vision]
    if not vision_models:
        return None  # Current model has no vision — can't describe

    # Determine which model to use for describing
    vision_model: str = (
        pinned_physical_model
        if any(m.model == pinned_physical_model and m.vision for m in current_pm.physical_models)
        else vision_models[0].model
    )

    # Load all conversation messages from turns
    all_messages: list[dict] = []
    for turn in sorted(conv.turns, key=lambda t: t.turn_number):
        turn_msgs = turn.messages
        if isinstance(turn_msgs, list):
            all_messages.extend(turn_msgs)

    if not all_messages:
        return None

    # Auto-describe images
    described_messages, desc_meta = await auto_describe_images(
        all_messages, vision_model,
    )

    described_count = desc_meta.get("images_described", 0)
    if described_count == 0:
        return desc_meta  # No images found

    # Store degradation_event turn
    turn_number: int = 1
    if conv.turns:
        turn_number = max(t.turn_number for t in conv.turns) + 1

    deg_turn = ConversationTurn(
        conversation_id=conv.id,
        turn_number=turn_number,
        pseudo_model=current_pseudo_name,
        physical_model=vision_model,
        input_tokens=0,
        output_tokens=desc_meta.get("total_description_tokens", 0),
        messages=described_messages,
        response={"metadata": desc_meta},
        turn_type="degradation_event",
        had_images=False,
        had_tools=False,
        had_parallel_tools=False,
    )
    db.add(deg_turn)

    # Update conversation tracking
    conv.images_described = max(conv.images_described or 0, 0) + described_count

    return desc_meta


async async def _assemble_snapshot_context(
    conv: Conversation,
    db: AsyncSession,
    active_messages: list[dict],
) -> list[dict] | None:
    """If conversation has an active snapshot, assemble context with snapshot + last user message.

    Returns the assembled context, or None if no snapshot exists.
    """
    if not conv.active_snapshot_id:
        return None

    context = await assemble_context(conv, db)

    # Find and append the last user message
    for m in reversed(active_messages):
        if m.get("role") == "user":
            context.append(m)
            break

    return context


async def _apply_compaction(
    conv: Conversation,
    is_new_conversation: bool,
    messages: list[dict],
    pm_schema: object,
    config: ProxyConfigSchema,
    db: AsyncSession,
    estimated_input: int,
) -> dict:
    """Sprint 4: Run pre-compaction, external detection, continuous compaction.

    Returns a dict with all compaction state to pass forward.
    """
    state: dict = {
        "pre_compaction_applied": False,
        "pre_compaction_metadata": None,
        "active_messages": messages,
        "external_compaction_detected": False,
        "external_compaction_metadata": None,
        "continuous_compaction_applied": False,
        "continuous_compaction_metadata": None,
    }

    # Pre-compaction
    if pm_schema.pre_compaction.enabled and estimated_input > (pm_schema.pre_compaction.threshold or 0):
        compacted, meta = await pre_compact_input(
            messages=messages, pseudo_model=pm_schema, config=config,
        )
        state["pre_compaction_applied"] = meta.get("applied", False)
        state["pre_compaction_metadata"] = meta
        if meta.get("applied", False):
            state["active_messages"] = compacted

    active = state["active_messages"]

    # External compaction detection
    skip_continuous = False
    if conv is not None and not is_new_conversation:
        ext_info = await detect_external_compaction(active, conv, db)
        if ext_info is not None:
            ext_meta = await handle_external_compaction(active, conv, ext_info, db)
            state["external_compaction_detected"] = True
            state["external_compaction_metadata"] = ext_meta
            skip_continuous = True

    # Continuous compaction
    if not skip_continuous and conv is not None and pm_schema.continuous_compaction.enabled:
        cc_meta = await continuous_compact(
            conversation=conv, pseudo_model=pm_schema, config=config, db=db,
        )
        state["continuous_compaction_applied"] = cc_meta.get("applied", False)
        state["continuous_compaction_metadata"] = cc_meta

    # Assemble context if snapshot exists
    snapshot_context = await _assemble_snapshot_context(conv, db, active)
    if snapshot_context is not None:
        state["active_messages"] = snapshot_context

    return state


async def _save_and_return(ctx: SaveContext) -> ChatResult:
    """Steps 14-20: Build turn, save to DB, accumulate capabilities, return result."""
    response_dict = ctx.response.model_dump() if hasattr(ctx.response, "model_dump") else ctx.response

    input_tokens, output_tokens = _parse_usage(response_dict)
    tool_meta = _process_tool_metadata(response_dict, ctx)

    tool_defs = ctx.tools
    tools_incomplete = tool_meta["tools_incomplete"]
    tools_level = tool_meta["tools_level"]
    thinking_content = tool_meta["thinking_content"]

    tool_result_truncated = False
    for msg in ctx.messages:
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            truncated = truncate_tool_result(content)
            if truncated != content:
                tool_result_truncated = True
                msg["content"] = truncated

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
        messages=ctx.messages,
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
    ctx.conv.updated_at = func.now()
    await ctx.db.commit()

    updated_caps = await accumulate_capabilities(ctx.db, ctx.conv_uuid, ctx.turn_caps, ctx.session_caps)

    affinity_maintained = not ctx.is_new_conversation and ctx.existing_affinity == ctx.physical_model

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
        compatibility_warning=ctx.compatibility["warning"],
        compatibility_details=ctx.compatibility["details"],
        tools_filter_applied=ctx.tools_filter["applied"],
        tools_filter_reason=ctx.tools_filter["reason"],
        tools_level_used=tools_level,
        tools_incomplete=tools_incomplete,
        thinking_content=thinking_content,
        tool_result_truncated=tool_result_truncated,
        pre_compaction_applied=ctx.prep["pre_compaction_applied"],
        pre_compaction_metadata=ctx.prep["pre_compaction_metadata"],
        continuous_compaction_applied=ctx.prep["continuous_compaction_applied"],
        continuous_compaction_metadata=ctx.prep["continuous_compaction_metadata"],
        external_compaction_detected=ctx.prep["external_compaction_detected"],
        external_compaction_metadata=ctx.prep["external_compaction_metadata"],
        images_described=ctx.images_described,
        images_described_by=ctx.images_described_by,
        router_suggestion=ctx.router_suggestion,
    )


# ── Fallback logic ─────────────────────────────────────────────────────────────


async def call_with_fallback(
    pseudo_model_schema,
    messages: list[dict],
    stream: bool = False,
    **kwargs,
) -> tuple:
    """Try each physical model in order. On 503/429, move to next."""
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


# ── Shared helpers (exported for streaming path) ──────────────────────────────


async def evaluate_router_suggestion(
    pm_schema,
    messages: list[dict],
    current_pseudo_name: str,
    config,
) -> dict | None:
    """Evaluate task complexity via Router LLM — shared between paths.

    Non-blocking: if the feature is disabled or evaluation fails, returns
    ``None`` and the request continues unchanged.
    Only evaluates the **last user message** (never the full conversation).

    Args:
        pm_schema: Current pseudo-model schema.
        messages: Full message list (only last user evaluated internally).
        current_pseudo_name: Current pseudo-model name.
        config: Proxy config.

    Returns:
        Suggestion dict with ``complexity``, ``suggested``, ``reason``.
        ``None`` if disabled or evaluation fails.
    """
    if not pm_schema.router_llm.enabled:
        return None

    suggester_pm = config.pseudo_models.get(pm_schema.router_llm.suggester)
    if not suggester_pm or not suggester_pm.physical_models:
        return None

    suggester_model = suggester_pm.physical_models[0].model
    suggestion = await evaluate_complexity(
        messages=messages,
        suggester_model=suggester_model,
    )

    if not suggestion or not suggestion.get("suggested"):
        return None

    if pm_schema.router_llm.suggest_on_downgrade_only:
        if is_downgrade(
            suggestion["suggested"],
            current_pseudo_name,
            config,
        ):
            return suggestion
        return None

    return suggestion


# ── Internal helpers ──────────────────────────────────────────────────────────


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

    if ctx.tool_choice == "required" and tool_calls and not enforce_tool_choice(response_dict, ctx.tool_choice):
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
