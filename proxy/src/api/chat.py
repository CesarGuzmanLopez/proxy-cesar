"""POST /v1/chat/completions — Main endpoint.

OpenAI-compatible format. Supports streaming SSE and non-streaming.
sprint §9 — exact flow.

Sprint 2 extensions:
- Incoming content validation (images, audio, PDF, video, parallel tools)
- Capability detection per turn
- Compatibility validation on pseudo-model switch
- Tool filter by parallel_tools
- Input threshold guard
"""

import json
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.service.chat_service import (
    build_proxy_metadata,
    call_with_fallback,
    process_chat_request,
)
from src.service.capability_detector import detect_turn_capabilities, estimate_tokens

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

    try:
        # Determine conversation ID
        conversation_id = request.conversation_id or str(uuid.uuid4())

        # Prepare messages as dicts
        messages = [msg.model_dump(exclude_none=True) for msg in request.messages]

        if request.stream:
            return await _handle_streaming(
                config=config,
                affinity=affinity,
                db=db,
                conversation_id=conversation_id,
                pseudo_model_name=request.model,
                messages=messages,
                stream=True,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                tools=request.tools,
                tool_choice=request.tool_choice,
            )

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

    # Build response with Sprint 2 proxy_metadata
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
    )

    if not request.conversation_id:
        response_dict["conversation_id"] = result.conversation_id

    return response_dict


async def _handle_streaming(
    config,
    affinity,
    db,
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

    For streaming, capability detection and threshold check still run
    synchronously before the stream starts. Accumulation happens after
    the stream ends (in the generator callback).
    """
    from src.service.model_resolver import normalize_model_name, resolve_physical_model
    from src.adapters.db.models import Conversation
    from src.service.capability_detector import (
        accumulate_capabilities,
        load_session_capabilities,
    )
    from src.service.compatibility import validate_incoming_content, validate_switch
    from src.service.threshold_guard import check_input_threshold
    from src.service.tool_filter import get_eligible_models, is_pinned_model_eligible

    import uuid as uuid_mod

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
        conv_uuid = uuid_mod.UUID(conversation_id)
    except ValueError:
        conv_uuid = uuid_mod.uuid5(uuid_mod.NAMESPACE_DNS, conversation_id)

    # Load existing capabilities
    session_caps = await load_session_capabilities(db, conv_uuid)

    # Check pseudo-model switch
    compatibility_warning: str | None = None
    compatibility_details: dict | None = None

    conv = await db.get(Conversation, conv_uuid)
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
    is_new = conv is None

    # Resolve physical model
    pinned_model = existing_affinity
    if pinned_model and session_caps.has_parallel_tools:
        if not is_pinned_model_eligible(pinned_model, eligible_models):
            pinned_model = None

    physical_model = pinned_model or (eligible_models[0].model if eligible_models else pm_schema.physical_models[0].model)

    if is_new:
        conv = Conversation(
            id=conv_uuid,
            pseudo_model=resolved_model,
            physical_model=physical_model,
            total_tokens=0,
        )
        db.add(conv)
        await db.flush()

    await affinity.set(conversation_id, physical_model)

    # Call LiteLLM (streaming)
    litellm_response, fallback_info = await call_with_fallback(
        pseudo_model_schema=pm_schema,
        messages=messages,
        stream=True,
        temperature=temperature,
        max_tokens=max_tokens,
        tools=tools,
        tool_choice=tool_choice,
    )

    return StreamingResponse(
        _stream_response_generator(
            litellm_response=litellm_response,
            conversation_id=conversation_id,
            pseudo_model=resolved_model,
            physical_model=physical_model,
            fallback_info=fallback_info,
            affinity_maintained=not is_new and existing_affinity == physical_model,
            context_tokens=conv.total_tokens,
            context_window=pm_schema.context_window,
            session_caps=session_caps,
            compatibility_warning=compatibility_warning,
            compatibility_details=compatibility_details,
            tools_filter_applied=tools_filter_applied,
            tools_filter_reason=tools_filter_reason,
        ),
        media_type="text/event-stream",
    )


async def _stream_response_generator(
    litellm_response,
    conversation_id: str,
    pseudo_model: str,
    physical_model: str,
    fallback_info,
    affinity_maintained: bool,
    context_tokens: int,
    context_window: int | None,
    *,
    session_caps=None,
    compatibility_warning: str | None = None,
    compatibility_details: dict | None = None,
    tools_filter_applied: bool = False,
    tools_filter_reason: str | None = None,
):
    """SSE streaming: forward chunks, append proxy_metadata on [DONE].

    sprint §9.3.
    """
    async for chunk in litellm_response:
        chunk_json = chunk.model_dump_json()
        yield f"data: {chunk_json}\n\n"

    # Final chunk: proxy_metadata with Sprint 2 fields
    metadata = build_proxy_metadata(
        pseudo_model=pseudo_model,
        physical_model=physical_model,
        conversation_id=conversation_id,
        context_tokens=context_tokens,
        context_window=context_window,
        fallback_info=fallback_info,
        affinity_maintained=affinity_maintained,
        session_caps=session_caps,
        compatibility_warning=compatibility_warning,
        compatibility_details=compatibility_details,
        tools_filter_applied=tools_filter_applied,
        tools_filter_reason=tools_filter_reason,
    )
    final_chunk = {
        "id": f"chatcmpl-{conversation_id[:12]}",
        "object": "chat.completion.chunk",
        "choices": [{"delta": {}, "finish_reason": "stop"}],
        "proxy_metadata": metadata,
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"
