"""POST /v1/chat/completions — Main endpoint.

OpenAI-compatible format. Supports streaming SSE and non-streaming.
"""

import json
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import selectinload

from src.adapters.db.models import Conversation, ConversationTurn
from src.service.chat_models import (
    MetadataContext,
    StreamContext,
    build_proxy_metadata,
)
from src.service.chat_service import (
    call_with_fallback,
    evaluate_router_suggestion,
    handle_auto_describe,
    process_chat_request,
)
from src.service.capability_detector import (
    accumulate_capabilities,
    detect_turn_capabilities,
    estimate_tokens,
    load_session_capabilities,
)
from src.service.compatibility import validate_incoming_content, validate_switch
from src.service.model_resolver import normalize_model_name
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


router = APIRouter()


# ── Request/Response schemas ────────────────────────────────────────────────


class Message(BaseModel, extra="forbid"):
    role: str
    content: str | list[dict] | None = None
    name: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None


class ChatRequest(BaseModel, extra="forbid"):
    model: str
    messages: list[Message]
    conversation_id: str | None = None
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None


# ── Endpoint ────────────────────────────────────────────────────────────────


@router.post(
    "/v1/chat/completions",
    responses={
        400: {"description": "Bad request - unknown pseudo-model, input exceeds threshold, unsupported content"},
        409: {"description": "Conflict - pseudo-model switch blocked due to incompatibility"},
        502: {"description": "Proxy error - upstream LLM call failed"},
        503: {"description": "All physical models for the pseudo-model failed"},
    },
)
async def chat_completions(
    request: ChatRequest,
    fastapi_request: Request,
):
    """Main chat completions endpoint with Sprint 2 capability checks.

    Flow:
    1. Resolve conversation + detect capabilities
    2. Validate incoming content
    3. Validate pseudo-model switch compatibility
    4. Filter tool pool if needed
    5. Check input threshold
    6. Call LiteLLM with fallback
    7. Save turn with capability flags
    8. Build proxy_metadata with Sprint 2 fields
    """
    app_state = fastapi_request.app.state
    config = app_state.config
    db = app_state.db_session_factory()
    affinity = app_state.affinity

    # Determine conversation ID
    conversation_id = request.conversation_id or str(uuid.uuid4())

    # Prepare messages as dicts
    messages = [msg.model_dump(exclude_none=True) for msg in request.messages]

    if request.stream:
        # Streaming path: session lifecycle is managed inside _handle_streaming
        # because _stream_response_generator runs lazily after this function returns.
        # The generator needs the session to persist the turn after streaming ends.
        return await _handle_streaming(
            config=config,
            affinity=affinity,
            db_session_factory=app_state.db_session_factory,
            conversation_id=conversation_id,
            pseudo_model_name=request.model,
            messages=messages,
            stream=True,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            tools=request.tools,
            tool_choice=request.tool_choice,
        )

    # Non-streaming path: session lifecycle managed here with try/finally
    try:
        return await _handle_non_streaming(
            config=config,
            affinity=affinity,
            db=db,
            conversation_id=conversation_id,
            request=request,
            messages=messages,
        )
    except HTTPException:
        await db.rollback()
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=502,
            detail={"error": "PROXY_ERROR", "message": str(e)},
        ) from e
    finally:
        await db.close()


async def _handle_non_streaming(
    config,
    affinity,
    db,
    conversation_id: str,
    request: ChatRequest,
    messages: list[dict],
) -> dict:
    """Non-streaming request: call LLM, save turn, return response."""
    result = await process_chat_request(
        model=request.model,
        messages=messages,
        conversation_id=conversation_id,
        stream=False,
        config=config,
        affinity=affinity,
        db=db,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        tools=request.tools,
        tool_choice=request.tool_choice,
    )

    # Build response with Sprint 2 + Sprint 4 proxy_metadata
    response_dict = result.response
    response_dict["proxy_metadata"] = build_proxy_metadata(MetadataContext(
        pseudo_model=result.pseudo_model,
        physical_model=result.physical_model,
        conversation_id=result.conversation_id,
        context_tokens=result.total_tokens,
        context_window=result.context_window,
        fallback_info=result.fallback_info,
        affinity_maintained=result.affinity_maintained,
        session_caps=result.session_caps,
        compatibility_warning=result.compatibility_warning,
        compatibility_details=result.compatibility_details,
        tools_filter_applied=result.tools_filter_applied,
        tools_filter_reason=result.tools_filter_reason,
        pre_compaction_applied=result.pre_compaction_applied,
        pre_compaction_metadata=result.pre_compaction_metadata,
        continuous_compaction_applied=result.continuous_compaction_applied,
        continuous_compaction_metadata=result.continuous_compaction_metadata,
        external_compaction_detected=result.external_compaction_detected,
        external_compaction_metadata=result.external_compaction_metadata,
        images_described=result.images_described,
        images_described_by=result.images_described_by,
        images_degraded_manually=result.images_degraded_manually,
        router_suggestion=result.router_suggestion,
    ))

    if not request.conversation_id:
        response_dict["conversation_id"] = result.conversation_id

    return response_dict


async def _handle_streaming(
    config,
    affinity,
    db_session_factory,
    conversation_id: str,
    pseudo_model_name: str,
    messages: list[dict],
    stream: bool,
    temperature: float | None = None,
    max_tokens: int | None = None,
    tools: list[dict] | None = None,
    tool_choice: str | dict | None = None,
):
    """Streaming request: return SSE StreamingResponse.

    Capability detection, threshold check, and compaction run synchronously
    before the stream starts. Turn persistence happens after the stream
    completes, inside _stream_response_generator.

    Session lifecycle: creates its own DB session (not the one from the
    caller) because the generator runs lazily after this function returns.
    The generator is responsible for closing the session.
    """
    db = db_session_factory()
    try:
        return await _handle_streaming_with_db(
            db=db, config=config, affinity=affinity,
            conversation_id=conversation_id,
            pseudo_model_name=pseudo_model_name,
            messages=messages,
            temperature=temperature, max_tokens=max_tokens,
            tools=tools, tool_choice=tool_choice,
        )
    except HTTPException:
        await db.close()
        raise
    except Exception as e:
        await db.close()
        raise HTTPException(
            status_code=502,
            detail={"error": "PROXY_ERROR", "message": str(e)},
        ) from e


def _resolve_physical_model(
    existing_affinity: str | None,
    session_caps,
    eligible_models: list,
    pm_schema,
) -> tuple[str, str | None]:
    """Resolve physical model + provider based on affinity, caps, and eligibility."""
    pinned_model = existing_affinity
    if pinned_model and session_caps.has_parallel_tools:
        if not is_pinned_model_eligible(pinned_model, eligible_models):
            pinned_model = None
    if pinned_model:
        selected_phys = next(
            (p for p in pm_schema.physical_models if p.model == pinned_model),
            pm_schema.physical_models[0],
        )
    elif eligible_models:
        selected_phys = eligible_models[0]
    else:
        selected_phys = pm_schema.physical_models[0]
    return selected_phys.model, selected_phys.provider


def _validate_switch_and_filter_pool(
    conv,
    resolved_model: str,
    pm_schema,
    session_caps,
    config,
) -> tuple[str | None, dict | None, list, bool, str | None]:
    """Validate pseudo-model switch compatibility and filter eligible models.

    Returns:
        Tuple of (compatibility_warning, compatibility_details,
                  eligible_models, tools_filter_applied, tools_filter_reason).
    """
    compatibility_warning: str | None = None
    compatibility_details: dict | None = None

    if conv is not None and conv.pseudo_model != resolved_model:
        switch_result = validate_switch(
            from_pseudo_name=conv.pseudo_model,
            to_pseudo_name=resolved_model,
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
                    "to_pseudo_model": resolved_model,
                },
            )
        if switch_result.status.value == "warning":
            compatibility_warning = switch_result.reason
            compatibility_details = switch_result.details

    eligible_models = get_eligible_models(pm_schema.physical_models, session_caps)
    tools_filter_applied = session_caps.has_parallel_tools and len(eligible_models) < len(
        pm_schema.physical_models
    )
    tools_filter_reason = "parallel_tools_required" if tools_filter_applied else None

    return (
        compatibility_warning,
        compatibility_details,
        eligible_models,
        tools_filter_applied,
        tools_filter_reason,
    )


async def _run_compaction_pipeline(
    conv,
    is_new: bool,
    messages: list[dict],
    pm_schema,
    config,
    db,
    estimated_input: int,
) -> dict:
    """Run pre-compaction, external detection, continuous compaction, and snapshot assembly.

    Returns a dict with compaction state and active_messages.
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
    if conv is not None and not is_new:
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
    if conv is not None and conv.active_snapshot_id:
        state["active_messages"] = await _assemble_snapshot_context_sync(conv, db, active)

    return state


async def _assemble_snapshot_context_sync(conv, db, active_messages: list[dict]) -> list[dict]:
    """Assemble snapshot context with last user message appended."""
    context = await assemble_context(conv, db)
    for m in reversed(active_messages):
        if m.get("role") == "user":
            context.append(m)
            break
    return context


async def _handle_streaming_with_db(
    db,
    config,
    affinity,
    conversation_id: str,
    pseudo_model_name: str,
    messages: list[dict],
    temperature: float | None = None,
    max_tokens: int | None = None,
    tools: list[dict] | None = None,
    tool_choice: str | dict | None = None,
):
    """Pre-stream logic (runs synchronously before SSE starts)."""
    # Resolve model
    resolved_model = normalize_model_name(pseudo_model_name, config)
    if resolved_model not in config.pseudo_models:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "UNKNOWN_PSEUDO_MODEL",
                "message": f"Unknown pseudo-model: '{resolved_model}'",
                "available": list(config.pseudo_models.keys()),
            },
        )

    pm_schema = config.pseudo_models[resolved_model]

    # Detect capabilities in incoming messages
    turn_caps = detect_turn_capabilities(messages, tools)

    # Validate incoming content
    validate_incoming_content(turn_caps, pm_schema, resolved_model, config)

    # Resolve conversation ID
    try:
        conv_uuid = uuid.UUID(conversation_id)
    except ValueError:
        conv_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, conversation_id)

    # Load conversation FIRST (with eager-loaded turns) to avoid identity-map
    # issue when load_session_capabilities loads it next.
    conv = await db.get(Conversation, conv_uuid, options=[selectinload(Conversation.turns)])
    is_new = conv is None

    # Load session capabilities (uses identity-mapped conv, no extra DB trip)
    session_caps = await load_session_capabilities(db, conv_uuid)

    # Validate pseudo-model switch and filter eligible models
    (
        compatibility_warning, compatibility_details,
        eligible_models, tools_filter_applied, tools_filter_reason,
    ) = _validate_switch_and_filter_pool(
        conv=conv, resolved_model=resolved_model, pm_schema=pm_schema,
        session_caps=session_caps, config=config,
    )

    # Check input threshold
    estimated_input = estimate_tokens(messages)
    threshold_check = check_input_threshold(
        pseudo_model_name=resolved_model,
        input_token_threshold=pm_schema.input_token_threshold,
        estimated_tokens=estimated_input,
        pre_compaction_enabled=pm_schema.pre_compaction.enabled,
    )
    if not threshold_check.success:
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
            },
        )

    # Create or get existing conversation
    existing_affinity = await affinity.get(conversation_id)

    physical_model, provider = _resolve_physical_model(
        existing_affinity, session_caps, eligible_models, pm_schema,
    )

    if is_new:
        conv = Conversation(
            id=conv_uuid,
            pseudo_model=resolved_model,
            physical_model=physical_model,
            total_tokens=0,
        )
        conv.turns = []  # prevent lazy-load trigger outside greenlet context
        db.add(conv)
        await db.flush()

    await affinity.set(conversation_id, physical_model)

    # ── Sprint 5: Auto-describe images on pseudo-model switch ─────────────
    auto_describe_meta: dict | None = None
    if conv is not None and not is_new and resolved_model != conv.pseudo_model:
        auto_describe_meta = await handle_auto_describe(
            conv=conv,
            current_pseudo_name=conv.pseudo_model,
            new_pm_schema=pm_schema,
            config=config,
            db=db,
            pinned_physical_model=physical_model,
        )

    # ── Sprint 4: Compaction pipeline ─────────────────────────────────────
    comp_state = await _run_compaction_pipeline(
        conv=conv, is_new=is_new, messages=messages, pm_schema=pm_schema,
        config=config, db=db, estimated_input=estimated_input,
    )
    active_messages = comp_state["active_messages"]

    # ── Sprint 5: Router LLM ──────────────────────────────────────────────
    # Shared with non-streaming path via evaluate_router_suggestion().
    router_suggestion: dict | None = await evaluate_router_suggestion(
        pm_schema=pm_schema,
        messages=active_messages,
        current_pseudo_name=resolved_model,
        config=config,
    )

    # Call LiteLLM (streaming) with active_messages
    litellm_response, fallback_info = await call_with_fallback(
        pseudo_model_schema=pm_schema,
        messages=active_messages,
        stream=True,
        temperature=temperature,
        max_tokens=max_tokens,
        tools=tools,
        tool_choice=tool_choice,
    )

    # Sprint 5 metadata
    images_described: int = auto_describe_meta.get("images_described", 0) if auto_describe_meta else 0
    images_described_by: str | None = auto_describe_meta.get("described_by") if auto_describe_meta else None

    _streaming_response = StreamingResponse(
        _stream_response_generator(StreamContext(
            litellm_response=litellm_response,
            conversation_id=conversation_id,
            pseudo_model=resolved_model,
            physical_model=physical_model,
            fallback_info=fallback_info,
            affinity_maintained=not is_new and existing_affinity == physical_model,
            context_window=pm_schema.context_window,
            session_caps=session_caps,
            compatibility_warning=compatibility_warning,
            compatibility_details=compatibility_details,
            tools_filter_applied=tools_filter_applied,
            tools_filter_reason=tools_filter_reason,
            pre_compaction_applied=comp_state["pre_compaction_applied"],
            pre_compaction_metadata=comp_state["pre_compaction_metadata"],
            continuous_compaction_applied=comp_state["continuous_compaction_applied"],
            continuous_compaction_metadata=comp_state["continuous_compaction_metadata"],
            external_compaction_detected=comp_state["external_compaction_detected"],
            external_compaction_metadata=comp_state["external_compaction_metadata"],
            images_described=images_described,
            images_described_by=images_described_by,
            router_suggestion=router_suggestion,
            db=db,
            conv=conv,
            conv_uuid=conv_uuid,
            turn_caps=turn_caps,
            provider=provider,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            resolved_model=resolved_model,
            is_new=is_new,
        )),
        media_type="text/event-stream",
    )
    return _streaming_response


def _build_turn_tool_metadata(
    response_dict: dict,
    tool_defs: list[dict] | None,
    tool_choice: str | dict | None,
    turn_caps,
    provider: str | None,
) -> tuple[list[dict], bool, int, str | None]:
    """Extract tool metadata from response — extracted for cognitive complexity."""
    tool_calls = extract_tool_calls_from_response(response_dict) if tool_defs else []
    if tool_calls:
        try:
            validate_tool_call_ids(tool_calls)
        except ValueError:
            turn_caps.tools_incomplete = True
    if tool_choice == "required" and tool_calls and not enforce_tool_choice(response_dict, tool_choice):
        turn_caps.tools_incomplete = True
    thinking_content = extract_thinking_content(response_dict, provider)
    tools_incomplete = turn_caps.tools_incomplete
    tools_level = determine_tool_level_for_turn(
        tool_calls=tool_calls, tool_definitions=tool_defs, tools_incomplete=tools_incomplete,
    )
    return tool_calls, tools_incomplete, tools_level, thinking_content


def _truncate_tool_results(messages: list[dict] | None) -> None:
    """Truncate long tool result content in-place."""
    for msg in messages or []:
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            truncated = truncate_tool_result(content)
            if truncated != content:
                msg["content"] = truncated


async def _persist_stream_turn(
    ctx: StreamContext,
    response_dict: dict,
    input_tokens: int,
    output_tokens: int,
) -> tuple[Any, Any, Any]:
    """Persist turn after successful stream — extracted for cognitive complexity."""
    db = ctx.db
    conv = ctx.conv
    if not (db and conv is not None and ctx.conv_uuid is not None and ctx.turn_caps is not None):
        return db, conv, ctx.session_caps
    try:
        tool_defs: list[dict] | None = ctx.tools
        _, tools_incomplete, tools_level, thinking_content = _build_turn_tool_metadata(
            response_dict=response_dict,
            tool_defs=tool_defs,
            tool_choice=ctx.tool_choice,
            turn_caps=ctx.turn_caps,
            provider=ctx.provider,
        )
        _truncate_tool_results(ctx.messages)
        turn_number = 1
        if conv.turns:
            turn_number = max(t.turn_number for t in conv.turns) + 1
        turn = ConversationTurn(
            conversation_id=ctx.conv_uuid, turn_number=turn_number,
            pseudo_model=ctx.pseudo_model, physical_model=ctx.physical_model,
            input_tokens=input_tokens, output_tokens=output_tokens,
            messages=ctx.messages or [], response=response_dict,
            fallback_applied=ctx.fallback_info.applied if ctx.fallback_info else False,
            fallback_reason=ctx.fallback_info.reason if ctx.fallback_info else None,
            turn_type="normal", had_images=ctx.turn_caps.has_images,
            had_tools=ctx.turn_caps.has_tools, had_parallel_tools=ctx.turn_caps.has_parallel_tools,
            tool_definitions=tool_defs,
            thinking_blocks={"content": thinking_content} if thinking_content else None,
            tools_incomplete=tools_incomplete, tools_level_used=tools_level,
        )
        db.add(turn)
        conv.physical_model = ctx.physical_model
        conv.total_tokens += input_tokens + output_tokens
        conv.updated_at = func.now()
        await db.commit()
        updated_caps = await accumulate_capabilities(db, ctx.conv_uuid, ctx.turn_caps, ctx.session_caps)
        return db, conv, updated_caps
    except Exception:
        await db.rollback()
        return db, conv, ctx.session_caps


def _extract_tokens_from_chunks(chunks: list) -> tuple[int, int, dict]:
    """Extract input/output tokens and response dict from collected chunks."""
    input_tokens = 0
    output_tokens = 0
    response_dict: dict = {}

    if not chunks:
        return 0, 0, {}

    last_chunk = chunks[-1]

    # Extract usage
    try:
        usage = getattr(last_chunk, "usage", None)
        if usage is not None:
            pt = getattr(usage, "prompt_tokens", 0)
            ct = getattr(usage, "completion_tokens", 0)
            input_tokens = int(pt) if isinstance(pt, (int, float)) else 0
            output_tokens = int(ct) if isinstance(ct, (int, float)) else 0
    except (TypeError, ValueError):
        pass

    # Extract response dict
    try:
        raw = last_chunk.model_dump() if hasattr(last_chunk, "model_dump") else {}
        if isinstance(raw, dict):
            response_dict = raw
    except Exception:
        pass

    return input_tokens, output_tokens, response_dict


def _build_final_metadata_chunk(ctx: StreamContext, conv, session_caps, input_tokens: int, output_tokens: int) -> dict:
    """Build the final SSE chunk with complete proxy_metadata."""
    final_context_tokens = conv.total_tokens if conv is not None else (input_tokens + output_tokens)
    metadata = build_proxy_metadata(MetadataContext(
        pseudo_model=ctx.pseudo_model, physical_model=ctx.physical_model,
        conversation_id=ctx.conversation_id, context_tokens=final_context_tokens,
        context_window=ctx.context_window, fallback_info=ctx.fallback_info,
        affinity_maintained=ctx.affinity_maintained, session_caps=session_caps,
        compatibility_warning=ctx.compatibility_warning, compatibility_details=ctx.compatibility_details,
        tools_filter_applied=ctx.tools_filter_applied, tools_filter_reason=ctx.tools_filter_reason,
        pre_compaction_applied=ctx.pre_compaction_applied, pre_compaction_metadata=ctx.pre_compaction_metadata,
        continuous_compaction_applied=ctx.continuous_compaction_applied, continuous_compaction_metadata=ctx.continuous_compaction_metadata,
        external_compaction_detected=ctx.external_compaction_detected, external_compaction_metadata=ctx.external_compaction_metadata,
        images_described=ctx.images_described, images_described_by=ctx.images_described_by,
        router_suggestion=ctx.router_suggestion,
    ))
    return {
        "id": f"chatcmpl-{ctx.conversation_id[:12]}",
        "object": "chat.completion.chunk",
        "choices": [{"delta": {}, "finish_reason": "stop"}],
        "proxy_metadata": metadata,
    }


async def _stream_response_generator(ctx: StreamContext):
    """SSE streaming: forward chunks, persist turn on success, append metadata."""
    chunks: list = []
    try:
        async for chunk in ctx.litellm_response:
            chunks.append(chunk)
            yield f"data: {chunk.model_dump_json()}\n\n"
    except Exception as e:
        try:
            await ctx.db.rollback()
        except Exception:
            pass
        error_payload = {
            "error": "PROXY_STREAM_ERROR", "message": str(e),
            "physical_model": ctx.physical_model, "pseudo_model": ctx.pseudo_model,
        }
        yield f"data: {json.dumps({'id': f'chatcmpl-{ctx.conversation_id[:12]}', 'object': 'chat.completion.chunk', 'choices': [{'delta': {}, 'finish_reason': 'error'}], 'proxy_metadata': error_payload})}\n\n"
        yield "data: [DONE]\n\n"
        return

    input_tokens, output_tokens, response_dict = _extract_tokens_from_chunks(chunks)

    db, conv, session_caps = await _persist_stream_turn(ctx, response_dict, input_tokens, output_tokens)

    final_chunk = _build_final_metadata_chunk(ctx, conv, session_caps, input_tokens, output_tokens)
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"
    if db is not None:
        try:
            await db.close()
        except Exception:
            pass
