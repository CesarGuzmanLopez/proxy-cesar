"""Chat orchestration service.

Central business logic for POST /v1/chat/completions.
Uses Result monad for domain errors, HTTPException for transport errors (routers only).
python.md §3: errors as data in domain, exceptions at boundary.

Sprint 1: basic pseudo-model resolution, affinity, fallback.
Sprint 2: +capability detection, compatibility validation, threshold guard, tool filter.
Sprint 3: +canonical tool storage, tiktoken, thinking blocks, tool edge cases.
Sprint 4: +pre-compaction, continuous compaction, external compaction detection.
"""

import logging
import time
import uuid

from fastapi import HTTPException
from litellm.exceptions import (
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    RateLimitError,
    ServiceUnavailableError,
)
from sqlalchemy import func
from sqlalchemy.orm import selectinload
from sqlmodel.ext.asyncio.session import AsyncSession

from src.adapters.cache.message_ordering import (
    canonicalize_message_order,
    sort_tool_definitions,
)
from src.adapters.cache.provider_cache import (
    apply_anthropic_cache_control,
)
from src.adapters.cache.valkey_affinity import ValkeyAffinityAdapter
from src.adapters.db.models import Conversation, ConversationTurn
from src.adapters.litellm import call_litellm
from src.api.metrics import metrics
from src.config.pseudo_models import ProxyConfigSchema
from src.domain.capabilities import SessionCapabilities
from src.service.capability_detector import (
    accumulate_capabilities,
    detect_turn_capabilities,
    estimate_tokens,
    load_session_capabilities,
)
from src.adapters.cache.provider_cache import (
    build_cache_destruction_metadata,
    build_cache_metadata,
    should_apply_cache_control,
)
from src.service.chat_models import ChatResult, FallbackInfo, SaveContext
from src.service.compatibility import (
    validate_incoming_content,
    validate_switch,
    _any_vision as _any_vision_comp,
)
from src.service.context_alert import ContextAlert, get_context_alert
from src.service.inline_commands import handle_inline_command
from src.service.model_resolver import (
    build_passthrough_pseudo_model,
    normalize_model_name,
)
from src.service.multimedia.image_describer import auto_describe_images
from src.service.router_llm.suggester import evaluate_complexity, is_downgrade
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

logger = logging.getLogger(__name__)


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
    valkey=None,
) -> ChatResult:
    """Execute the full chat completion flow.

    Each logical step is delegated to a focused helper function.
    """
    _req_id = str(uuid.uuid4())[:8]
    _t0 = time.monotonic()
    logger.info(
        "req_start | trace=%s model=%s conv=%s messages=%d tools=%s stream=%s",
        _req_id,
        model,
        conversation_id or "new",
        len(messages),
        bool(tools),
        stream,
    )

    # Step 1-3: Resolve model + detect capabilities + validate content
    pseudo_model_name, pm_schema, turn_caps, delegation = _resolve_and_validate(
        model,
        messages,
        tools,
        config,
    )
    # Apply image→tool delegation or blob storage for unsupported content
    if delegation:
        from src.service.tool_detector import (
            delegate_images_to_tool,
            replace_base64_with_blob_refs,
        )

        if delegation.get("action") == "delegate_images":
            messages = delegate_images_to_tool(
                messages,
                delegation["tool_name"],
                delegation["param_name"],
            )
        elif delegation.get("action") == "transform_unsupported":
            messages = await replace_base64_with_blob_refs(
                messages,
                conversation_id,
                valkey or getattr(affinity, "_client", None),
                config,
            )
    # Sprint 8: track request metrics
    metrics.record_request(pseudo_model_name)
    conv_id = conversation_id or str(uuid.uuid4())
    conv_uuid = _parse_uuid(conv_id)

    # Sprint 9: Check for inline commands BEFORE any LLM processing.
    # If the user typed @compact, @degrade, @status, or @help, handle
    # it here and return immediately without calling the LLM.
    cmd_result = await handle_inline_command(
        messages=messages,
        conversation_id=conversation_id,
        db=db,
    )
    if cmd_result.handled and cmd_result.skip_llm:
        logger.info(
            "inline_cmd | trace=%s conv=%s cmd=%s",
            _req_id,
            conv_id[:12],
            _extract_cmd_name(messages),
        )
        # Build a minimal ChatResult with the command output
        return ChatResult(
            conversation_id=conv_id,
            pseudo_model=pseudo_model_name,
            physical_model="(command)",
            response={
                "id": f"chatcmpl-{_req_id}",
                "object": "chat.completion",
                "model": pseudo_model_name,
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": cmd_result.response_text,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                "proxy_metadata": cmd_result.response_metadata,
            },
            fallback_info=FallbackInfo(),
            is_new_conversation=False,
            affinity_maintained=False,
            total_tokens=0,
            context_window=pm_schema.context_window
            if hasattr(pm_schema, "context_window")
            else None,
            session_caps=SessionCapabilities(conversation_id=conv_id),
            compatibility_warning=None,
            compatibility_details=None,
            tools_filter_applied=False,
            tools_filter_reason=None,
            tools_level_used=0,
            tools_incomplete=False,
            thinking_content=None,
            tool_result_truncated=False,
            pre_compaction_applied=False,
            pre_compaction_metadata=None,
            continuous_compaction_applied=False,
            continuous_compaction_metadata=None,
            external_compaction_detected=False,
            external_compaction_metadata=None,
            images_described=0,
            images_described_by=None,
            router_suggestion=None,
            context_alert=ContextAlert(alert_level="none", context_usage_pct=None),
            cache_metadata={},
        )

    # Step 4-11: Load session, check switch, load/create conversation, resolve model, set affinity
    (
        existing_affinity,
        session_caps,
        compatibility,
        physical_model,
        provider,
        tools_filter,
        conv,
        is_new,
    ) = await _resolve_session_conv_and_models(
        db,
        affinity,
        conv_id,
        conv_uuid,
        pseudo_model_name,
        pm_schema,
        config,
    )
    await affinity.set(conv_id, physical_model)

    # Sprint 5: Auto-describe images on pseudo-model switch
    auto_describe_meta: dict | None = None
    messages_for_llm: list[dict] = messages  # May be replaced by described version
    if conv is not None and not is_new and pseudo_model_name != conv.pseudo_model:
        desc_in_flight, auto_describe_meta = await handle_auto_describe(
            conv=conv,
            current_pseudo_name=conv.pseudo_model,
            new_pm_schema=pm_schema,
            config=config,
            db=db,
            pinned_physical_model=physical_model,
            in_flight_messages=messages,
        )
        if desc_in_flight is not None:
            messages_for_llm = desc_in_flight

    # Step 12: Check input threshold
    est_input = estimate_tokens(messages_for_llm)
    _raise_if_exceeds_threshold(est_input, pm_schema, pseudo_model_name, config)

    # No automatic compaction — if threshold is exceeded, error is returned above.
    active_messages = messages_for_llm

    # Sprint 6: Context alerts — warn before context becomes unusable
    context_alert = get_context_alert(
        total_tokens=conv.total_tokens if conv else 0,
        context_window=pm_schema.context_window,
        conversation_id=conv_id,
    )
    if context_alert.alert_level == "unusable":
        raise HTTPException(
            status_code=400,
            detail={
                "error": "CONTEXT_UNUSABLE",
                "message": context_alert.warning,
                "context_tokens": conv.total_tokens if conv else 0,
                "context_window": pm_schema.context_window,
                "remediation": {
                    "action": "compact",
                    "endpoint": f"POST /conversations/{conv_id}/compact",
                    "description": (
                        "Compact the conversation history into a snapshot. "
                        "Original history is preserved."
                    ),
                },
            },
        )

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
        estimated_input=est_input,
        temperature=temperature,
        max_tokens=max_tokens,
        tools=tools,
        tool_choice=tool_choice,
    )
    _elapsed = time.monotonic() - _t0
    logger.info(
        "req_end   | trace=%s conv=%s pseudo=%s physical=%s "
        "stream=%s fallback=%s tokens_in=%s elapsed=%.1fs",
        _req_id,
        conv_id[:12],
        pseudo_model_name,
        physical_model,
        stream,
        fallback_info.reason if fallback_info.applied else "none",
        est_input,
        _elapsed,
    )

    # Compute Sprint 5 metadata
    images_described = 0
    images_described_by: str | None = None
    if auto_describe_meta:
        images_described = auto_describe_meta.get("images_described", 0)
        images_described_by = auto_describe_meta.get("described_by")

    return await _save_and_return(
        SaveContext(
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
            compatibility=compatibility,
            tools_filter=tools_filter,
            images_described=images_described,
            images_described_by=images_described_by,
            router_suggestion=router_suggestion,
            context_alert=context_alert,
        )
    )


# ── Step helpers ──────────────────────────────────────────────────────────────


def _resolve_and_validate(
    model: str,
    messages: list[dict],
    tools: list[dict] | None,
    config: ProxyConfigSchema,
) -> tuple[str, object, SessionCapabilities, dict | None]:
    """Steps 1-3: Normalize model, detect capabilities, validate content.

    If the model name is not a known pseudo-model, creates a direct passthrough.
    Returns (resolved_model, pseudo_model, capabilities, delegation_signal).
    delegation_signal is None if OK, or dict with "action": "delegate_images" etc.
    """
    resolved = normalize_model_name(model, config)
    if resolved not in config.pseudo_models:
        pm = build_passthrough_pseudo_model(resolved)
    else:
        pm = config.pseudo_models[resolved]
    caps = detect_turn_capabilities(messages, tools)
    delegation = validate_incoming_content(caps, pm, resolved, config, tools)
    return resolved, pm, caps, delegation


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
    conv = await db.get(
        Conversation, conv_uuid, options=[selectinload(Conversation.turns)]
    )
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
        session_caps.has_parallel_tools
        and len(eligible) < len(pm_schema.physical_models)
    )
    tools_filter_reason = "parallel_tools_required" if tools_filter_applied else None

    pinned = existing_affinity
    if (
        pinned
        and session_caps.has_parallel_tools
        and not is_pinned_model_eligible(pinned, eligible)
    ):
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
    """Step 12: Raise 400 if input exceeds threshold (auto-compaction removed)."""
    check = check_input_threshold(
        pseudo_model_name=pseudo_model_name,
        input_token_threshold=pm_schema.input_token_threshold,
        estimated_tokens=estimated_input,
        pre_compaction_enabled=False,
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
                "suggestions": _suggest_higher_threshold_models(
                    config, error.estimated
                ),
            },
        )


def _resolve_auto_describe_params(
    config,
    current_pseudo_name: str,
    new_pm_schema: object,
    pinned_physical_model: str,
) -> tuple[str | None, str | None, str | None]:
    """Check if auto-describe should run and resolve the vision model.

    Returns (vision_model, current_pseudo_name, skip_reason).
    If vision_model is None, skip_reason explains why auto-describe was skipped.
    """
    if new_pm_schema.image_handling.on_downgrade != "auto_describe":
        return (None, None, "destination_on_downgrade_not_auto_describe")
    if _any_vision_comp(new_pm_schema.physical_models):
        return (None, None, "destination_has_vision_no_describe_needed")

    current_pm = config.pseudo_models.get(current_pseudo_name)
    if current_pm is None:
        return (None, None, f"source_pseudo_model_not_found:{current_pseudo_name}")

    vision_models = [m for m in current_pm.physical_models if m.vision]
    if not vision_models:
        return (None, None, f"source_{current_pseudo_name}_has_no_vision_models")

    vision_model: str = (
        pinned_physical_model
        if any(
            m.model == pinned_physical_model and m.vision
            for m in current_pm.physical_models
        )
        else vision_models[0].model
    )
    return (vision_model, current_pseudo_name, None)


def _load_messages_from_turns(conv: Conversation) -> list[dict]:
    """Load all messages from conversation turns in order."""
    all_messages: list[dict] = []
    for turn in sorted(conv.turns, key=lambda t: t.turn_number):
        turn_msgs = turn.messages
        if isinstance(turn_msgs, list):
            all_messages.extend(turn_msgs)
    return all_messages


async def handle_auto_describe(
    conv: Conversation,
    current_pseudo_name: str,
    new_pm_schema: object,
    config,
    db: AsyncSession,
    pinned_physical_model: str,
    in_flight_messages: list[dict] | None = None,
) -> tuple[list[dict] | None, dict | None]:
    """Execute auto-describe when switching from vision to non-vision model."""
    # Resolve vision model (also validates auto-describe is needed)
    vision_model, current_pseudo_name, skip_reason = _resolve_auto_describe_params(
        config,
        current_pseudo_name,
        new_pm_schema,
        pinned_physical_model,
    )
    if vision_model is None:
        if skip_reason:
            logger.debug("auto_describe_skipped reason=%s", skip_reason)
        return (
            None,
            None
            if not skip_reason
            else {"auto_describe_skipped": True, "skip_reason": skip_reason},
        )

    # Load and describe messages from DB history
    all_messages = _load_messages_from_turns(conv)
    if not all_messages:
        return (None, None)

    described_history, desc_meta = await auto_describe_images(
        all_messages, vision_model
    )
    described_count = desc_meta.get("images_described", 0)
    if described_count == 0:
        return (None, desc_meta)

    # Store degradation_event turn
    turn_number = max(t.turn_number for t in conv.turns) + 1 if conv.turns else 1
    deg_turn = ConversationTurn(
        conversation_id=conv.id,
        turn_number=turn_number,
        pseudo_model=current_pseudo_name,
        physical_model=vision_model,
        input_tokens=0,
        output_tokens=desc_meta.get("total_description_tokens", 0),
        messages=described_history,
        response={"metadata": desc_meta},
        turn_type="degradation_event",
        had_images=False,
        had_tools=False,
        had_parallel_tools=False,
    )
    db.add(deg_turn)
    conv.images_described = max(conv.images_described or 0, 0) + described_count
    conv.capability_has_images = False  # Reset after successful auto-describe

    # Describe in-flight messages if present
    described_in_flight: list[dict] | None = None
    if in_flight_messages:
        desc_in_flight, _ = await auto_describe_images(in_flight_messages, vision_model)
        described_in_flight = desc_in_flight

    return (described_in_flight, desc_meta)


async def _save_and_return(ctx: SaveContext) -> ChatResult:
    """Steps 14-20: Build turn, save to DB, accumulate capabilities, return result."""
    response_dict = (
        ctx.response.model_dump()
        if hasattr(ctx.response, "model_dump")
        else ctx.response
    )

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

    # Sprint 7: extract provider cache metadata from response
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

    # Accumulate capabilities BEFORE commit — same transaction as turn save
    updated_caps = await accumulate_capabilities(
        ctx.db, ctx.conv_uuid, ctx.turn_caps, ctx.session_caps
    )

    await ctx.db.commit()

    # Sprint 8: record metrics
    metrics.record_tokens(input_tokens, output_tokens, input_tokens)
    if ctx.fallback_info.applied:
        metrics.record_fallback(ctx.fallback_info.reason or "unknown")

    affinity_maintained = (
        not ctx.is_new_conversation and ctx.existing_affinity == ctx.physical_model
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
        compatibility_warning=ctx.compatibility["warning"],
        compatibility_details=ctx.compatibility["details"],
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


async def call_with_fallback(
    pseudo_model_schema,
    messages: list[dict],
    stream: bool = False,
    estimated_input: int | None = None,
    **kwargs,
) -> tuple:
    """Try each physical model in order. On retryable errors, move to next.

    Retryable errors: ServiceUnavailableError (503), RateLimitError (429),
    NotFoundError (404 — e.g. model not available via provider), and
    AuthenticationError (401 — expired / invalid key for a given provider).
    Any other exception propagates immediately (the request fails fast).

    Sprint 7: applies provider-specific cache optimizations (Anthropic cache_control)
    and tracks cache destruction on fallback.
    """
    fallback_info = FallbackInfo()
    last_error: Exception | None = None
    _trace_id = str(uuid.uuid4())[:8]
    _start = time.monotonic()

    _RETRYABLE = (
        ServiceUnavailableError,
        RateLimitError,
        NotFoundError,
        AuthenticationError,
        BadRequestError,  # e.g. invalid model ID, model not found via provider
    )

    # Sprint 7: track cache optimization for metadata extraction
    cache_optimization_applied = False

    # Estimate input tokens once (caller may have pre-computed)
    _est_input = (
        estimated_input if estimated_input is not None else estimate_tokens(messages)
    )
    _context_skipped: list[str] = []  # Models skipped due to context too large

    # Sprint 7: canonical ordering + tool sorting — compute ONCE before the loop
    ordered_messages = canonicalize_message_order(messages)
    raw_tools = kwargs.get("tools")
    if raw_tools:
        sorted_tools = sort_tool_definitions(raw_tools)
        kwargs["tools"] = sorted_tools
        first_name = sorted_tools[0].get("function", {}).get("name", "?")
        logger.debug(
            "tools_sorted | trace=%s count=%d first=%s",
            _trace_id,
            len(sorted_tools),
            first_name,
        )

    for idx, phys in enumerate(pseudo_model_schema.physical_models):
        try:
            # Sprint 9: Check if this model's context window is large enough.
            # If the input exceeds the model's context_window, skip it and
            # report clearly. Prevents silent failures when fallback models
            # have smaller context than the primary.
            if phys.context_window is not None and _est_input > phys.context_window:
                logger.warning(
                    "llm_skip  | trace=%s model=%s context_window=%d "
                    "input_est=%d reason=context_too_large",
                    _trace_id,
                    phys.model,
                    phys.context_window,
                    _est_input,
                )
                _context_skipped.append(phys.model)
                fallback_info.attempted_models.append(
                    f"{phys.model} (skipped: context too small)"
                )
                continue

            logger.info(
                "llm_call  | trace=%s attempt=%d/%d model=%s stream=%s",
                _trace_id,
                idx + 1,
                len(pseudo_model_schema.physical_models),
                phys.model,
                stream,
            )

            # Sprint 7: apply provider-specific cache optimizations
            call_messages = ordered_messages
            provider = phys.provider.lower()
            if should_apply_cache_control(provider):
                call_messages = apply_anthropic_cache_control(ordered_messages)
                cache_optimization_applied = True
                logger.debug(
                    "cache_control applied provider=%s messages=%d",
                    provider,
                    len(call_messages),
                )

            response = await call_litellm(
                model=phys.model,
                messages=call_messages,
                stream=stream,
                **{k: v for k, v in kwargs.items() if v is not None},
            )
            fallback_info.attempted_models.append(phys.model)

            # Sprint 7: attach provider for cache metadata extraction (non-streaming only)
            if not stream:
                try:
                    response._proxy_provider = provider
                    response._proxy_cache_optimization_applied = (
                        cache_optimization_applied
                    )
                except (AttributeError, TypeError):
                    pass  # some response objects are immutable (e.g., async generators)

            elapsed = time.monotonic() - _start
            # Log response summary for non-streaming
            if not stream:
                try:
                    c = response.choices[0].message.content or ""
                    logger.info(
                        "llm_ok    | trace=%s model=%s elapsed=%.1fs "
                        "content_len=%d finish=%s",
                        _trace_id,
                        phys.model,
                        elapsed,
                        len(c),
                        response.choices[0].finish_reason,
                    )
                except (AttributeError, IndexError):
                    logger.warning(
                        "llm_ok    | trace=%s model=%s (unexpected format)",
                        _trace_id,
                        phys.model,
                    )
            else:
                logger.info(
                    "llm_ok    | trace=%s model=%s elapsed=%.1fs (streaming)",
                    _trace_id,
                    phys.model,
                    elapsed,
                )
            return response, fallback_info
        except _RETRYABLE as e:
            last_error = e
            fallback_info.attempted_models.append(phys.model)
            fallback_info.applied = True
            fallback_info.reason = f"{type(e).__name__}: {phys.model}"
            logger.warning(
                "llm_fallback | trace=%s model=%s error=%s elapsed=%.1fs",
                _trace_id,
                phys.model,
                type(e).__name__,
                time.monotonic() - _start,
            )
            continue

    elapsed = time.monotonic() - _start

    # Sprint 9: If all models were skipped due to context too large,
    # return a 413 error suggesting compaction instead of a generic 503.
    if len(_context_skipped) == len(pseudo_model_schema.physical_models):
        logger.error(
            "llm_fail   | trace=%s elapsed=%.1fs reason=all_context_too_large "
            "input_est=%d models=%s",
            _trace_id,
            elapsed,
            _est_input,
            _context_skipped,
        )
        raise HTTPException(
            status_code=413,
            detail={
                "error": "CONTEXT_TOO_LARGE_FOR_ALL_MODELS",
                "message": (
                    f"The conversation ({_est_input:,} tokens) exceeds the "
                    f"context window of all remaining models in "
                    f"'{pseudo_model_schema.display_name}'. "
                    f"Type 'compact' or '/compact' in your message to compact."
                ),
                "estimated_tokens": _est_input,
                "remediation": {
                    "action": "compact",
                    "description": "Compact the conversation into a snapshot.",
                    "command": "/compact",
                },
                "context_skipped": _context_skipped,
            },
        )

    # Some models were skipped due to context, some failed — report partial
    context_skipped_note = ""
    if _context_skipped:
        context_skipped_note = f" ({len(_context_skipped)} skipped: context too large)"

    logger.error(
        "llm_fail   | trace=%s elapsed=%.1fs models=%s last_error=%s",
        _trace_id,
        elapsed,
        fallback_info.attempted_models,
        last_error,
    )
    raise HTTPException(
        status_code=503,
        detail={
            "error": "ALL_MODELS_FAILED",
            "message": (
                f"All {len(fallback_info.attempted_models)} model(s) for "
                f"pseudo-model '{pseudo_model_schema.display_name}' failed"
                f"{context_skipped_note}."
            ),
            "attempted": fallback_info.attempted_models,
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

    if (
        ctx.tool_choice == "required"
        and tool_calls
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
    suggestions.sort(key=lambda x: x["input_token_threshold"])
    return suggestions


def _extract_cache_metadata(
    response,
    provider: str,
    fallback_info: FallbackInfo,
) -> dict:
    """Sprint 7: Extract cache hit/miss metadata from the provider response.

    Uses provider_cache.build_cache_metadata() for standard extraction and
    adds cache destruction info when fallback occurred.
    """
    response_dict = (
        response.model_dump() if hasattr(response, "model_dump") else response
    )
    if not isinstance(response_dict, dict):
        response_dict = {}

    # Check if cache optimization was applied (attached in call_with_fallback)
    cache_applied = getattr(response, "_proxy_cache_optimization_applied", False)
    meta = build_cache_metadata(response_dict, provider, cache_applied)

    # If fallback occurred, add cache destruction metadata
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


def _extract_cmd_name(messages: list[dict]) -> str:
    """Extract the command name from the last user message for logging."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str) and content.startswith("@"):
                return content.split()[0][1:]  # "@compact foo" → "compact"
    return "?"
