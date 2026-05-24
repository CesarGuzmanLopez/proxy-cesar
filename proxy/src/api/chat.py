"""POST /v1/chat/completions — Main endpoint.

OpenAI-compatible format. Supports streaming SSE and non-streaming.
"""

import json
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import selectinload

from src.adapters.db.models import Conversation, ConversationTurn
from src.service.chat_models import build_proxy_metadata
from src.service.chat_service import (
    call_with_fallback,
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


@router.post("/v1/chat/completions")
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
    response_dict["proxy_metadata"] = build_proxy_metadata(
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
        # Sprint 4
        pre_compaction_applied=result.pre_compaction_applied,
        pre_compaction_metadata=result.pre_compaction_metadata,
        continuous_compaction_applied=result.continuous_compaction_applied,
        continuous_compaction_metadata=result.continuous_compaction_metadata,
        external_compaction_detected=result.external_compaction_detected,
        external_compaction_metadata=result.external_compaction_metadata,
    )

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
            db, config, affinity, db_session_factory,
            conversation_id, pseudo_model_name, messages,
            stream, temperature, max_tokens, tools, tool_choice,
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


async def _handle_streaming_with_db(
    db,
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

    # Check pseudo-model switch
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

    # Filter pool
    eligible_models = get_eligible_models(pm_schema.physical_models, session_caps)
    tools_filter_applied = session_caps.has_parallel_tools and len(eligible_models) < len(
        pm_schema.physical_models
    )
    tools_filter_reason = "parallel_tools_required" if tools_filter_applied else None

    # Check input threshold
    estimated_input = estimate_tokens(messages)
    threshold_check = check_input_threshold(
        pseudo_model_name=resolved_model,
        input_token_threshold=pm_schema.input_token_threshold,
        estimated_tokens=estimated_input,
        pre_compaction_enabled=pm_schema.pre_compaction.enabled,
    )
    if not threshold_check.ok:
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

    # Resolve physical model + extract provider
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
    physical_model = selected_phys.model
    provider = selected_phys.provider

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

    # ── Pre-compaction ────────────────────────────────────────────────────
    pre_compaction_applied = False
    pre_compaction_metadata: dict | None = None
    active_messages = messages

    if pm_schema.pre_compaction.enabled and estimated_input > (pm_schema.pre_compaction.threshold or 0):
        compacted, meta = await pre_compact_input(
            messages=messages, pseudo_model=pm_schema, config=config,
        )
        pre_compaction_applied = meta.get("applied", False)
        pre_compaction_metadata = meta
        if meta.get("applied", False):
            active_messages = compacted

    # ── External compaction detection ─────────────────────────────────────
    external_compaction_detected = False
    external_compaction_metadata: dict | None = None
    skip_continuous = False

    if conv is not None and not is_new:
        ext_info = await detect_external_compaction(active_messages, conv, db)
        if ext_info is not None:
            ext_meta = await handle_external_compaction(active_messages, conv, ext_info, db)
            external_compaction_detected = True
            external_compaction_metadata = ext_meta
            skip_continuous = True

    # ── Continuous compaction ─────────────────────────────────────────────
    continuous_compaction_applied = False
    continuous_compaction_metadata: dict | None = None

    if not skip_continuous and conv is not None and pm_schema.continuous_compaction.enabled:
        cc_meta = await continuous_compact(
            conversation=conv, pseudo_model=pm_schema, config=config, db=db,
        )
        continuous_compaction_applied = cc_meta.get("applied", False)
        continuous_compaction_metadata = cc_meta

    # Assemble context if snapshot exists
    if conv is not None and conv.active_snapshot_id:
        context = await assemble_context(conv, db)
        last_user = None
        for m in reversed(active_messages):
            if m.get("role") == "user":
                last_user = m
                break
        if last_user:
            context.append(last_user)
        active_messages = context

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

    _streaming_response = StreamingResponse(
        _stream_response_generator(
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
            pre_compaction_applied=pre_compaction_applied,
            pre_compaction_metadata=pre_compaction_metadata,
            continuous_compaction_applied=continuous_compaction_applied,
            continuous_compaction_metadata=continuous_compaction_metadata,
            external_compaction_detected=external_compaction_detected,
            external_compaction_metadata=external_compaction_metadata,
            # Persistence context
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
        ),
        media_type="text/event-stream",
    )
    return _streaming_response


async def _stream_response_generator(
    litellm_response,
    conversation_id: str,
    pseudo_model: str,
    physical_model: str,
    fallback_info,
    affinity_maintained: bool,
    context_window: int | None,
    *,
    session_caps=None,
    compatibility_warning: str | None = None,
    compatibility_details: dict | None = None,
    tools_filter_applied: bool = False,
    tools_filter_reason: str | None = None,
    pre_compaction_applied: bool = False,
    pre_compaction_metadata: dict | None = None,
    continuous_compaction_applied: bool = False,
    continuous_compaction_metadata: dict | None = None,
    external_compaction_detected: bool = False,
    external_compaction_metadata: dict | None = None,
    # Persistence context (for saving turn after stream completes)
    db=None,
    conv=None,
    conv_uuid=None,
    turn_caps=None,
    provider=None,
    messages=None,
    tools=None,
    tool_choice=None,
    resolved_model=None,
    is_new=None,
):
    """SSE streaming: forward chunks, persist turn on success, append metadata.

    On success: saves the ConversationTurn to DB, updates conversation
    total_tokens and capabilities, then yields [DONE].
    On error: yields PROXY_STREAM_ERROR chunk + [DONE] without persisting.
    """
    chunks: list = []
    try:
        async for chunk in litellm_response:
            chunks.append(chunk)
            chunk_json = chunk.model_dump_json()
            yield f"data: {chunk_json}\n\n"
    except Exception as e:
        # Yield error chunk instead of crashing the stream
        try:
            await db.rollback()
        except Exception:
            pass
        error_payload = {
            "error": "PROXY_STREAM_ERROR",
            "message": str(e),
            "physical_model": physical_model,
            "pseudo_model": pseudo_model,
        }
        error_chunk = {
            "id": f"chatcmpl-{conversation_id[:12]}",
            "object": "chat.completion.chunk",
            "choices": [{"delta": {}, "finish_reason": "error"}],
            "proxy_metadata": error_payload,
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"
        return

    # ── Persist turn after successful stream ──────────────────────────
    input_tokens = 0
    output_tokens = 0
    response_dict: dict = {}

    if chunks:
        last_chunk = chunks[-1]

        # Extract usage — guard against MagicMock in tests
        try:
            usage = getattr(last_chunk, "usage", None)
            if usage is not None:
                pt = getattr(usage, "prompt_tokens", 0)
                ct = getattr(usage, "completion_tokens", 0)
                input_tokens = int(pt) if isinstance(pt, (int, float)) else 0
                output_tokens = int(ct) if isinstance(ct, (int, float)) else 0
        except (TypeError, ValueError):
            input_tokens = 0
            output_tokens = 0

        # Build a coarse response dict from chunks (for tools metadata)
        try:
            raw = last_chunk.model_dump() if hasattr(last_chunk, "model_dump") else {}
            if isinstance(raw, dict):
                response_dict = raw
        except Exception:
            response_dict = {}

    if db and conv is not None and conv_uuid is not None and turn_caps is not None:
        try:
            # Tool metadata
            tool_defs: list[dict] | None = tools
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
                tool_calls=tool_calls,
                tool_definitions=tool_defs,
                tools_incomplete=tools_incomplete,
            )

            tool_result_truncated = False
            for msg in messages or []:
                if msg.get("role") == "tool":
                    content = msg.get("content", "")
                    truncated = truncate_tool_result(content)
                    if truncated != content:
                        tool_result_truncated = True
                        msg["content"] = truncated

            turn_number = 1
            if conv.turns:
                turn_number = max(t.turn_number for t in conv.turns) + 1

            turn = ConversationTurn(
                conversation_id=conv_uuid,
                turn_number=turn_number,
                pseudo_model=pseudo_model,
                physical_model=physical_model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                messages=messages or [],
                response=response_dict,
                fallback_applied=fallback_info.applied if fallback_info else False,
                fallback_reason=fallback_info.reason if fallback_info else None,
                turn_type="normal",
                had_images=turn_caps.has_images,
                had_tools=turn_caps.has_tools,
                had_parallel_tools=turn_caps.has_parallel_tools,
                tool_definitions=tool_defs,
                thinking_blocks={"content": thinking_content} if thinking_content else None,
                tools_incomplete=tools_incomplete,
                tools_level_used=tools_level,
            )
            db.add(turn)

            conv.physical_model = physical_model
            conv.total_tokens += input_tokens + output_tokens
            conv.updated_at = func.now()
            await db.commit()

            session_caps = await accumulate_capabilities(db, conv_uuid, turn_caps, session_caps)

        except Exception:
            await db.rollback()
            # Non-fatal: stream already sent, just log

    # ── Final metadata chunk ──────────────────────────────────────────────
    final_context_tokens = conv.total_tokens if conv is not None else (input_tokens + output_tokens)
    metadata = build_proxy_metadata(
        pseudo_model=pseudo_model,
        physical_model=physical_model,
        conversation_id=conversation_id,
        context_tokens=final_context_tokens,
        context_window=context_window,
        fallback_info=fallback_info,
        affinity_maintained=affinity_maintained,
        session_caps=session_caps,
        compatibility_warning=compatibility_warning,
        compatibility_details=compatibility_details,
        tools_filter_applied=tools_filter_applied,
        tools_filter_reason=tools_filter_reason,
        pre_compaction_applied=pre_compaction_applied,
        pre_compaction_metadata=pre_compaction_metadata,
        continuous_compaction_applied=continuous_compaction_applied,
        continuous_compaction_metadata=continuous_compaction_metadata,
        external_compaction_detected=external_compaction_detected,
        external_compaction_metadata=external_compaction_metadata,
    )
    final_chunk = {
        "id": f"chatcmpl-{conversation_id[:12]}",
        "object": "chat.completion.chunk",
        "choices": [{"delta": {}, "finish_reason": "stop"}],
        "proxy_metadata": metadata,
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"
    if db is not None:
        try:
            await db.close()
        except Exception:
            pass
