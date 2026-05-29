"""Streaming handlers for /v1/chat/completions.

Extracted from chat.py to keep individual files under 600 lines.
"""

import json
import logging
import re
import uuid

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import selectinload

from src.api.metrics import metrics
from src.adapters.cache.message_ordering import canonicalize_message_order
from src.adapters.cache.provider_cache import (
    build_cache_destruction_metadata,
    build_cache_metadata,
)
from src.adapters.db.models import Conversation
from src.domain.types import Err
from src.service.chat_models import StreamContext, StreamingRequestContext
from src.service.context_alert import get_context_alert
from src.service.chat_fallback import _try_physical_model, call_with_fallback
from src.service.chat_messages import handle_auto_describe
from src.service.chat_service import evaluate_router_suggestion
from src.service.capability_detector import (
    detect_turn_capabilities,
    estimate_tokens,
    load_session_capabilities,
)
from src.service.compatibility import validate_physical_model_content
from src.service.model_resolver import (
    build_passthrough_pseudo_model,
    normalize_model_name,
)
from src.service.threshold_guard import check_input_threshold
from src.service.pipeline_trace import PipelineTrace
from src.adapters.litellm.client import normalise_stream_chunk
from src.api.chat_stream_persistence import (
    _build_final_metadata_chunk,
    _extract_tokens_from_chunks,
    _filter_eligible_models,
    _persist_stream_turn,
    _resolve_physical_model,
)


logger = logging.getLogger(__name__)


type _ContentType = str


_CONTENT_TYPE_MAP: dict[_ContentType, list[_ContentType]] = {
    "images": ["image_url", "image"],
    "pdfs": ["pdf"],
    "audios": ["audio", "wav", "mp3"],
    "documents": ["word", "docx", "text", "csv", "excel"],
}


def _classify_file(mime_type: str) -> str:
    """Classify a file mime_type into a content category."""
    lower = mime_type.lower()
    for category, keywords in _CONTENT_TYPE_MAP.items():
        if category == "images":
            continue
        if any(k in lower for k in keywords):
            return category
    return "documents"


def _count_content_types(messages: list[dict]) -> dict[str, int]:
    """Count images, PDFs, audios, and other content types in messages."""
    counts = {"images": 0, "pdfs": 0, "audios": 0, "documents": 0}

    for msg in messages or []:
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue

        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type", "")
            if item_type in _CONTENT_TYPE_MAP["images"]:
                counts["images"] += 1
            elif item_type == "file":
                mime_type = item.get("file", {}).get("mime_type", "")
                counts[_classify_file(mime_type)] += 1

    return counts


def _build_analysis_message(counts: dict[str, int], images_described: int) -> str:
    """Build analysis message based on content types being processed."""
    parts = []

    if images_described > 0:
        parts.append(f"imagen{'s' if images_described > 1 else ''}")
    if counts.get("pdfs", 0) > 0:
        parts.append(f"PDF{'s' if counts['pdfs'] > 1 else ''}")
    if counts.get("audios", 0) > 0:
        parts.append(f"audio{'s' if counts['audios'] > 1 else ''}")
    if counts.get("documents", 0) > 0:
        parts.append(f"documento{'s' if counts['documents'] > 1 else ''}")

    if not parts:
        return "Procesando contenido..."

    return f"Analizando {', '.join(parts)}..."


async def _handle_streaming(ctx: StreamingRequestContext):
    """Streaming request: return SSE StreamingResponse.

    Capability detection, threshold check, and compaction run synchronously
    before the stream starts. Turn persistence happens after the stream
    completes, inside _stream_response_generator.

    Session lifecycle: creates its own DB session (not the one from the
    caller) because the generator runs lazily after this function returns.
    The generator is responsible for closing the session.
    """
    db = ctx.db_session_factory()
    try:
        return await _handle_streaming_with_db(
            db=db,
            config=ctx.config,
            affinity=ctx.affinity,
            conversation_id=ctx.conversation_id,
            pseudo_model_name=ctx.pseudo_model_name,
            messages=ctx.messages,
            temperature=ctx.temperature,
            max_tokens=ctx.max_tokens,
            tools=ctx.tools,
            tool_choice=ctx.tool_choice,
            stream_options=ctx.stream_options,
            thinking=ctx.thinking,
            trace=ctx.trace,
            request=ctx.request,
        )
    except HTTPException:
        try:
            await db.close()
        except Exception as exc:
            logger.debug("stream_db_close_error err=%s", exc)
        if ctx.trace:
            ctx.trace.proxy_out(http_status=400, stream=True)
        raise
    except ValueError as e:
        try:
            await db.close()
        except Exception as exc:
            logger.debug("stream_db_close_error err=%s", exc)
        error_msg = str(e)
        status_code, error_detail = _map_stream_domain_error(error_msg)
        if ctx.trace:
            ctx.trace.proxy_out(http_status=status_code, stream=True)
        raise HTTPException(
            status_code=status_code,
            detail=error_detail,
        ) from e
    except Exception as e:
        try:
            await db.close()
        except Exception as exc:
            logger.debug("stream_db_close_error err=%s", exc)
        metrics.record_error(502, "PROXY_ERROR")
        if ctx.trace:
            ctx.trace.proxy_out(http_status=502, stream=True)
        raise HTTPException(
            status_code=502,
            detail={
                "error": {
                    "message": str(e),
                    "type": "server_error",
                    "param": None,
                    "code": "server_error",
                },
            },
        ) from e


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
    stream_options: dict | None = None,
    thinking: dict | str | bool | None = None,
    trace: PipelineTrace | None = None,
    request = None,
):
    """Pre-stream logic (runs synchronously before SSE starts)."""
    # Resolve model
    resolved_model = normalize_model_name(pseudo_model_name, config)
    if resolved_model not in config.pseudo_models:
        pm_schema = build_passthrough_pseudo_model(resolved_model)
    else:
        pm_schema = config.pseudo_models[resolved_model]

    # Detect capabilities in incoming messages
    content_details = []
    for i, msg in enumerate(messages):
        content = msg.get("content")
        if isinstance(content, list):
            part_types = [part.get("type", "?") for part in content]
            content_details.append(f"msg{i}:list[{','.join(part_types)}]")
            # Also log first 100 chars of each part to see what's inside
            for j, part in enumerate(content):
                text_preview = ""
                if part.get("type") == "text" and "text" in part:
                    preview = str(part.get("text", ""))[:100]
                    text_preview = f" preview='{preview}'"
                elif part.get("type") == "image":
                    text_preview = f" image_data_len={len(str(part.get('image', '')))}"
                if part.get("type") == "text":
                    full_text = str(part.get("text", ""))[:200]
                    logger.info(
                        "message_part_detail msg=%d part=%d type=%s text_len=%d preview=%s",
                        i, j, part.get("type", "?"), len(str(part.get("text", ""))), full_text
                    )
                else:
                    logger.info(
                        "message_part_detail msg=%d part=%d type=%s%s",
                        i, j, part.get("type", "?"), text_preview
                    )
        elif isinstance(content, str):
            content_details.append(f"msg{i}:str")
        else:
            content_details.append(f"msg{i}:{type(content).__name__}")
    turn_caps = detect_turn_capabilities(messages, tools)

    # Resolve conversation ID
    try:
        conv_uuid = uuid.UUID(conversation_id)
    except ValueError:
        conv_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, conversation_id)

    logger.info(
        "stream_req_start conv=%s pseudo=%s messages=%d tools=%s",
        conversation_id[:12],
        resolved_model,
        len(messages),
        bool(tools),
    )

    # Load conversation FIRST (with eager-loaded turns) to avoid identity-map
    # issue when load_session_capabilities loads it next.
    conv = await db.get(
        Conversation, conv_uuid, options=[selectinload(Conversation.turns)]
    )
    is_new = conv is None

    # Load session capabilities (uses identity-mapped conv, no extra DB trip)
    session_caps = await load_session_capabilities(db, conv_uuid)

    # Filter eligible models based on session capabilities
    (
        eligible_models,
        tools_filter_applied,
        tools_filter_reason,
    ) = _filter_eligible_models(
        pm_schema=pm_schema,
        session_caps=session_caps,
    )

    # Create or get existing conversation
    existing_affinity = await affinity.get(conversation_id)

    physical_model, provider, selected_phys_model = _resolve_physical_model(
        existing_affinity,
        session_caps,
        eligible_models,
        pm_schema,
    )

    logger.info(
        "DIAGNOSTIC_1_after_physical_model_resolution trace=%s phys=%s has_vision=%s",
        trace.trace_id if trace else "?",
        physical_model,
        getattr(selected_phys_model, "vision", "?"),
    )

    # NEW: Validate that the SELECTED physical model can handle the incoming content
    # (this happens AFTER model selection, not before like the old logic)
    delegation = validate_physical_model_content(turn_caps, selected_phys_model)

    # Debug logging for content delegation
    logger.info(
        "content_validation_stream trace=%s conv=%s model=%s has_images=%s model_vision=%s delegation=%s",
        trace.trace_id if trace else "?",
        conversation_id[:12],
        physical_model,
        getattr(turn_caps, "has_images", False),
        getattr(selected_phys_model, "vision", False),
        bool(delegation),
    )

    # Apply content delegation (images → descriptions, audio → transcriptions, etc.)
    if delegation:
        logger.info(
            "content_delegation_applying_stream trace=%s conv=%s model=%s action=%s",
            trace.trace_id if trace else "?",
            conversation_id[:12],
            physical_model,
            delegation.get("action"),
        )
        from src.service.tool_detector import replace_base64_with_blob_refs, inject_blob_extraction_guidance

        valkey_client = getattr(affinity, "_client", None)
        messages = await replace_base64_with_blob_refs(
            messages, conversation_id, valkey_client, config
        )
        # Inject guidance about blob extraction and available tools
        messages = inject_blob_extraction_guidance(messages)

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

    # feature Auto-describe images on pseudo-model switch (Bug 6 fix)
    auto_describe_meta: dict | None = None
    messages_for_llm: list[dict] = messages
    if conv is not None and not is_new and resolved_model != conv.pseudo_model:
        desc_in_flight, auto_describe_meta = await handle_auto_describe(
            conv=conv,
            current_pseudo_name=conv.pseudo_model,
            config=config,
            db=db,
            pinned_physical_model=physical_model,
            in_flight_messages=messages,
        )
        if desc_in_flight is not None:
            messages_for_llm = desc_in_flight

    # ── Load conversation history for streaming context ──────────
    # NOTE: build_conversation_messages is NOT called here because the client
    # (opencode, Continue, etc.) already sends the full conversation context
    # in every request. Loading history from DB doubles the token count
    # (e.g., 6 client messages → 59 after loading DB history = 128K tokens).
    # History is still stored in DB for auditing, compaction, and affinity.
    if not is_new and conv is not None and conv.turns:
        logger.debug("DBG conv=%s step=skip_build_msgs client_msgs=%d db_turns=%d",
                     conversation_id[:12], len(messages), len(conv.turns))
    # No automatic compaction — if threshold is exceeded, error is returned.
    active_messages = messages_for_llm

    # ── Token estimation AFTER content delegation ─────────────────────
    # estimate_tokens must run AFTER replace_base64_with_blob_refs so the
    # count reflects what the LLM actually receives (text descriptions
    # can be much larger than the original base64 blobs).
    estimated_input = await estimate_tokens(active_messages)

    # ── Check input threshold (post-delegation) ───────────────────────
    threshold_check = check_input_threshold(
        pseudo_model_name=resolved_model,
        input_token_threshold=pm_schema.input_token_threshold,
        estimated_tokens=estimated_input,
        pre_compaction_enabled=False,
    )
    if not threshold_check.success:
        error = threshold_check.error
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": (
                        f"This model's maximum context length is "
                        f"{error.threshold} tokens. However, your messages "
                        f"resulted in {error.estimated} tokens. Please reduce "
                        f"the length of the messages."
                    ),
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "context_length_exceeded",
                },
            },
        )

    # ── feature Context alerts (total = history + current request) ────
    _total_for_alert = (conv.total_tokens if conv else 0) + estimated_input
    context_alert = get_context_alert(
        total_tokens=_total_for_alert,
        context_window=pm_schema.context_window,
        conversation_id=conversation_id,
    )
    if context_alert.alert_level == "unusable":
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": context_alert.warning,
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "context_length_exceeded",
                },
            },
        )

    # ── feature Router LLM ──────────────────────────────────────────────
    # Shared with non-streaming path via evaluate_router_suggestion().
    router_suggestion: dict | None = await evaluate_router_suggestion(
        pm_schema=pm_schema,
        messages=active_messages,
        current_pseudo_name=resolved_model,
        config=config,
    )

    # Apply canonical message ordering before LLM call
    active_messages = canonicalize_message_order(active_messages)

    logger.info(
        "stream_llm_call conv=%s pseudo=%s physical=%s provider=%s "
        "messages=%d tools=%s est_tokens=%d",
        conversation_id[:12],
        resolved_model,
        physical_model,
        provider or "none",
        len(active_messages),
        bool(tools),
        estimated_input,
    )

    # Call LiteLLM (streaming) with active_messages
    litellm_response, fallback_info = await call_with_fallback(
        pseudo_model_schema=pm_schema,
        messages=active_messages,
        stream=True,
        estimated_input=estimated_input,
        temperature=temperature,
        max_tokens=max_tokens,
        tools=tools,
        tool_choice=tool_choice,
        stream_options=stream_options,
        thinking=thinking,
        conversation_id=conversation_id,
        valkey_client=getattr(affinity, "_client", None),
        affinity=affinity,
    )

    # feature Update physical_model from actual response when fallback occurred
    if fallback_info.applied:
        if hasattr(litellm_response, "model") and litellm_response.model:
            logger.debug(
                "physical_model_update (stream) | old=%s new=%s",
                physical_model,
                litellm_response.model,
            )
            physical_model = litellm_response.model
            provider = (
                physical_model.split("/")[0] if "/" in physical_model else provider
            )

    logger.info(
        "stream_llm_ok conv=%s physical=%s provider=%s fallback=%s attempted=%s",
        conversation_id[:12],
        physical_model,
        provider or "none",
        fallback_info.applied,
        fallback_info.attempted_models or "none",
    )

    # Bug 5 fix: update affinity AFTER successful LLM call
    await affinity.set(conversation_id, physical_model)

    images_described: int = 0
    images_described_by: str | None = None
    if auto_describe_meta:
        images_described = auto_describe_meta.get("images_described", 0)
        images_described_by = auto_describe_meta.get("described_by")

    # Extract provider response headers (cache, rate-limit, request-id, etc.)
    # and include them in the SSE response so the client can observe
    # provider-level metadata.
    _sse_headers = {"X-Conversation-Id": conversation_id}
    if hasattr(litellm_response, "_provider_response_headers"):
        provider_headers = litellm_response._provider_response_headers
        if provider_headers:
            for h, v in provider_headers.items():
                _sse_headers[h] = str(v)

    # Extract keyvault secrets from request state for re-injection in generator
    keyvault_secrets = getattr(request.state, "keyvault_secrets", {}) if request else {}
    if keyvault_secrets:
        logger.info("stream_keyvault_active conv=%s secrets=%s",
                     conversation_id[:12], list(keyvault_secrets.keys()))

    _streaming_response = StreamingResponse(
        _stream_response_generator(
            StreamContext(
                litellm_response=litellm_response,
                conversation_id=conversation_id,
                pseudo_model=resolved_model,
                physical_model=physical_model,
                fallback_info=fallback_info,
                affinity_maintained=not is_new and existing_affinity == physical_model and not fallback_info.applied,
                context_window=pm_schema.context_window,
                session_caps=session_caps,
                compatibility_warning=None,
                compatibility_details=None,
                tools_filter_applied=tools_filter_applied,
                tools_filter_reason=tools_filter_reason,
                images_described=images_described,
                images_described_by=images_described_by,
                router_suggestion=router_suggestion,
                context_alert=context_alert,
                db=db,
                conv=conv,
                conv_uuid=conv_uuid,
                turn_caps=turn_caps,
                provider=provider,
                messages=messages,
                active_messages=active_messages,
                tools=tools,
                tool_choice=tool_choice,
                resolved_model=resolved_model,
                is_new=is_new,
                # feature pass schema + kwargs for token-limit continuation
                pm_schema=pm_schema,
                call_kwargs={
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "tools": tools,
                    "tool_choice": tool_choice,
                    "stream_options": stream_options,
                    "thinking": thinking,
                },
                trace=trace,
                timeout=60.0,  # Use same default as DEFAULT_LLM_TIMEOUT_SECONDS
            ),
            keyvault_secrets=keyvault_secrets,
        ),
        media_type="text/event-stream",
        headers=_sse_headers,
    )
    return _streaming_response


def _chunk_to_dict(chunk) -> dict:
    """Convert a streaming chunk to a dict (single serialization, no string intermediate)."""
    try:
        return chunk.model_dump(exclude_none=False)
    except (AttributeError, TypeError):
        try:
            return dict(chunk)
        except Exception:
            return {}


async def _stream_response_generator(ctx: StreamContext, keyvault_secrets: dict[str, str] | None = None):
    """SSE streaming: forward chunks, persist turn on success, append metadata.

    feature supports token-limit continuation — when a model finishes with
    ``finish_reason="length"`` and more physical models are available, the
    generator suppresses the ``"length"`` finish_reason (sets to ``null``),
    appends the accumulated content as an assistant message, and seamlessly
    continues streaming from the next model.

    keyvault_secrets: dict mapping [KEYVAULT:hash] to real values for re-injection.
    """
    import uuid

    stream_id = str(uuid.uuid4())[:8]  # Unique ID for this streaming session
    last_chunk = None  # Track last chunk for token extraction (avoids memory accumulation)
    db = ctx.db

    # feature prepare for token-limit continuation across multiple models
    current_stream = ctx.litellm_response
    phys_models = (
        list(ctx.pm_schema.physical_models) if ctx.pm_schema and ctx.call_kwargs else []
    )
    current_idx: int = 0
    accumulated_content: str = ""
    tool_calls_by_index: dict[int, dict] = {}

    logger.info(
        "stream_gen_start stream_id=%s conv=%s physical=%s models_available=%d",
        stream_id,
        ctx.conversation_id[:12],
        ctx.physical_model,
        len(phys_models),
    )
    _first_chunk_logged = False
    _analysis_message_sent = False

    try:
        # ── Multi-stream loop (continues when a model hits token limit) ──
        while True:
            finish_reason: str | None = None
            try:
                # Send initial message if content is being analyzed (first iteration only)
                if not _analysis_message_sent and current_idx == 0:
                    content_counts = _count_content_types(ctx.messages)
                    has_content = ctx.images_described > 0 or any(content_counts.values())
                    if has_content:
                        _analysis_message_sent = True
                        analysis_msg = _build_analysis_message(content_counts, ctx.images_described)
                        analysis_chunk = {
                            "id": f"chatcmpl-{ctx.conversation_id[:12]}",
                            "object": "chat.completion.chunk",
                            "choices": [{"delta": {"content": analysis_msg}, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(analysis_chunk)}\n\n"
                        logger.info(
                            "stream_analysis_msg conv=%s images=%d pdfs=%d audios=%d docs=%d",
                            ctx.conversation_id[:12],
                            ctx.images_described,
                            content_counts.get("pdfs", 0),
                            content_counts.get("audios", 0),
                            content_counts.get("documents", 0),
                        )

                async for chunk in current_stream:
                    last_chunk = chunk  # Track last chunk; avoids storing all chunks in memory

                    # Accumulate text content and tool_calls for potential continuation
                    try:
                        delta = chunk.choices[0].delta
                    except (AttributeError, IndexError):
                        delta = None

                    if delta:
                        # Content: safe to process
                        if hasattr(delta, "content") and delta.content:
                            accumulated_content += delta.content

                        # Tool calls: accumulate with defensive extraction
                        tc_deltas = getattr(delta, "tool_calls", None)
                        if tc_deltas:
                            for tcd in tc_deltas:
                                try:
                                    _accumulate_tool_call_delta(tcd, tool_calls_by_index)
                                except Exception as e:
                                    logger.warning(
                                        "stream_tool_call_delta_error conv=%s | error=%s",
                                        ctx.conversation_id[:12],
                                        str(e),
                                    )
                                    continue

                    if not _first_chunk_logged:
                        _first_chunk_logged = True
                        _content_preview = (
                            (delta.content or "")[:80]
                            if hasattr(chunk.choices[0], "delta")
                            and hasattr(chunk.choices[0].delta, "content")
                            else ""
                        )
                        logger.info(
                            "stream_chunk_start conv=%s physical=%s preview=%s",
                            ctx.conversation_id[:12],
                            ctx.physical_model,
                            _content_preview or "(no content)",
                        )

                    # Detect finish_reason — intercept "length" if more models
                    try:
                        fr = chunk.choices[0].finish_reason if chunk.choices else None
                    except (AttributeError, IndexError):
                        fr = None

                    if fr:
                        logger.info(
                            "stream_chunk_finish conv=%s physical=%s reason=%s "
                            "accumulated_len=%d",
                            ctx.conversation_id[:12],
                            ctx.physical_model,
                            fr,
                            len(accumulated_content),
                        )

                    # Convert chunk to dict ONCE — reuse for all processing
                    chunk_dict = _chunk_to_dict(chunk)

                    if (
                        fr == "length"
                        and current_idx < len(phys_models) - 1
                        and getattr(ctx.pm_schema, "continue_on_length", False)
                    ):
                        # Remove reasoning_content before sending "length" suppression chunk
                        for choice in chunk_dict.get("choices", []):
                            if isinstance(choice, dict) and "delta" in choice:
                                choice["delta"].pop("reasoning_content", None)
                        # Suppress "length" — set to null so client continues
                        if chunk_dict.get("choices"):
                            chunk_dict["choices"][0]["finish_reason"] = None
                        # Final safeguard: remove reasoning_content with regex before sending
                        chunk_json = json.dumps(chunk_dict)
                        chunk_json = re.sub(r', "reasoning_content": "[^"]*"', '', chunk_json)
                        yield f"data: {chunk_json}\n\n"
                        finish_reason = "length"
                        break

                    # Pass reasoning_content through as-is — clients that support it
                    # (opencode, Continue) display it in a separate thinking section.
                    # The normalise_stream_chunk is intentionally a no-op for this reason.
                    normalise_stream_chunk(chunk_dict)

                    # Re-inject secrets: cross-chunk placeholder buffering
                    # LLM tokenizers split [KEYVAULT:hash] across 3-6 SSE chunks.
                    # We check for complete placeholders first, then buffer open brackets.
                    if keyvault_secrets:
                        delta_content = ""
                        for choice in chunk_dict.get("choices", []):
                            delta = choice.get("delta", {})
                            dc = delta.get("content", "") or ""
                            delta_content += dc

                        if delta_content:
                            buf = getattr(ctx, "_keyvault_buf", "")
                            buf += delta_content

                            # 1. Check for COMPLETE placeholder in current chunk
                            chunk_json = json.dumps(chunk_dict)
                            replaced = False
                            for secret_hash, real_value in keyvault_secrets.items():
                                placeholder = f"[KEYVAULT:{secret_hash}]"
                                if placeholder in chunk_json:
                                    chunk_json = chunk_json.replace(placeholder, real_value)
                                    replaced = True
                                    logger.info(
                                        "stream_reinject_ok conv=%s hash=%s chunk_delta=%s",
                                        ctx.conversation_id[:12], secret_hash,
                                        repr(delta_content)[:80],
                                    )

                            # 2. Check for cross-chunk placeholder (fragmented across chunks)
                            if not replaced:
                                has_open = "[" in buf.rpartition("]")[2]
                                if has_open:
                                    ctx._keyvault_buf = buf[-256:]
                                    for secret_hash, real_value in keyvault_secrets.items():
                                        placeholder = f"[KEYVAULT:{secret_hash}]"
                                        if placeholder in ctx._keyvault_buf:
                                            chunk_json = json.dumps(chunk_dict)
                                            chunk_json = chunk_json.replace(placeholder, real_value)
                                            replaced = True
                                            logger.info(
                                                "stream_reinject_ok conv=%s hash=%s (cross-chunk)",
                                                ctx.conversation_id[:12], secret_hash,
                                            )
                                            break
                                    if not replaced:
                                        ctx._keyvault_buf = buf[-256:]
                                else:
                                    ctx._keyvault_buf = ""  # No potential placeholder
                            else:
                                ctx._keyvault_buf = ""

                            if replaced:
                                chunk_dict = json.loads(chunk_json)

                    # Use chunk_dict (may have keyvault secrets re-injected)
                    yield f"data: {json.dumps(chunk_dict)}\n\n"

                    if fr:
                        finish_reason = fr
                        break

            except GeneratorExit:
                logger.info(
                    "stream_gen_disconnect conv=%s physical=%s",
                    ctx.conversation_id[:12],
                    ctx.physical_model,
                )
                if db is not None:
                    await db.close()
                    db = None  # prevent double-close in outer finally
                return
            except Exception as e:
                try:
                    await ctx.db.rollback()
                except Exception as rollback_err:
                    logger.debug("stream_db_rollback_error err=%s", rollback_err)
                error_payload = {
                    "error": "PROXY_STREAM_ERROR",
                    "message": str(e),
                    "physical_model": ctx.physical_model,
                    "pseudo_model": ctx.pseudo_model,
                }
                yield f"data: {json.dumps({'id': f'chatcmpl-{ctx.conversation_id[:12]}', 'object': 'chat.completion.chunk', 'choices': [{'delta': {}, 'finish_reason': 'error'}], 'proxy_metadata': error_payload})}\n\n"
                # NOTE: [DONE] will be sent by finally block below, not here
                return

            # ── Decide whether to continue with next model ──────────────
            if finish_reason != "length":
                break  # Natural stop — we're done

            # All models exhausted — stop
            if current_idx >= len(phys_models) - 1:
                break

            # ── Prepare and call next model ─────────────────────────────
            current_idx += 1
            next_phys = phys_models[current_idx]

            # Build continuation messages: full history + partial assistant
            cont_messages = list(ctx.active_messages or ctx.messages or [])
            assistant_msg: dict[str, object] = {
                "role": "assistant",
                "content": accumulated_content,
            }
            if tool_calls_by_index:
                assembled_tcs: list[dict] = []
                for _idx in sorted(tool_calls_by_index.keys()):
                    _entry = tool_calls_by_index[_idx]
                    _args = "".join(_entry["function"]["arguments_parts"])
                    assembled_tcs.append(
                        {
                            "id": _entry["id"],
                            "type": "function",
                            "function": {
                                "name": _entry["function"]["name"],
                                "arguments": _args,
                            },
                        }
                    )
                if assembled_tcs:
                    assistant_msg["tool_calls"] = assembled_tcs
            cont_messages.append(assistant_msg)
            tool_calls_by_index = {}  # Reset for next model stream

            _trace_id = str(uuid.uuid4())[:8]
            # NOTE: ctx.call_kwargs contains generic OpenAI-compatible parameters
            # (temperature, max_tokens, tools, tool_choice, stream_options, thinking).
            # These are NOT provider-specific — they apply across models without
            # modification. Provider-specific handling (api_key, api_base, thinking
            # stripping for non-Anthropic) is done inside _try_physical_model().
            # The kwargs are the same for all models in the fallback chain and do
            # NOT contaminate the next model call. This is NOT a bug.
            new_stream, skip_reason = await _try_physical_model(
                next_phys,
                canonicalize_message_order(cont_messages),
                True,  # stream
                ctx.call_kwargs or {},
                await estimate_tokens(cont_messages),
                _trace_id,
                timeout=ctx.timeout,
            )

            if new_stream is None:
                logger.warning(
                    "stream_continuation_skip | trace=%s model=%s reason=%s",
                    _trace_id,
                    next_phys.model,
                    skip_reason,
                )
                break  # Next model also skipped — give up

            logger.info(
                "stream_continuation conv=%s trace=%s model=%s accumulated=%d",
                ctx.conversation_id[:12],
                _trace_id,
                next_phys.model,
                len(accumulated_content),
            )
            # Update context to reflect the active model
            ctx.physical_model = next_phys.model
            ctx.provider = (
                next_phys.provider.lower() if next_phys.provider else ctx.provider
            )
            current_stream = new_stream
            # Continue the outer while loop to iterate the new stream

        # ── All streams consumed — persist and finalize ─────────────────
        input_tokens, output_tokens, response_dict = _extract_tokens_from_chunks(
            last_chunk, accumulated_content
        )

        logger.info(
            "stream_complete conv=%s physical=%s tokens=%d+%d",
            ctx.conversation_id[:12],
            ctx.physical_model,
            input_tokens,
            output_tokens,
        )

        # Log the assembled response for debugging
        _msg = response_dict.get("choices", [{}])[0].get("message", {})
        _content = _msg.get("content") or ""
        _reasoning = _msg.get("reasoning_content") or ""
        logger.info(
            "stream_response_assembled conv=%s physical=%s "
            "content_len=%d reasoning_len=%d finish=%s preview=%s",
            ctx.conversation_id[:12],
            ctx.physical_model,
            len(_content),
            len(_reasoning),
            response_dict.get("choices", [{}])[0].get("finish_reason", "?"),
            _content[:2000],
        )

        # feature extract cache metadata from streaming response
        if ctx.cache_metadata is None:
            provider = ctx.provider or ""
            cache_applied = _should_stream_cache_be_applied(ctx)
            cache_meta = build_cache_metadata(response_dict, provider, cache_applied)
            # Add fallback cache destruction if applicable
            if (
                ctx.fallback_info
                and ctx.fallback_info.applied
                and ctx.fallback_info.attempted_models
            ):
                prev = (
                    ctx.fallback_info.attempted_models[0]
                    if len(ctx.fallback_info.attempted_models) > 1
                    else ""
                )
                new_m = ctx.fallback_info.attempted_models[-1]
                destruction = build_cache_destruction_metadata(
                    previous_model=prev,
                    new_model=new_m,
                    previous_cached_tokens=cache_meta.get("cached_tokens", 0),
                )
                cache_meta["fallback_cache_destruction"] = destruction
            ctx.cache_metadata = cache_meta

        # Set defaults for final metadata chunk (before persistence runs)
        session_caps = ctx.session_caps
        conv = ctx.conv

        # Build and send final metadata chunk with fallback for json.dumps failure
        try:
            final_chunk = _build_final_metadata_chunk(
                ctx, conv, session_caps, input_tokens, output_tokens
            )
            final_json = json.dumps(final_chunk)
            yield f"data: {final_json}\n\n"
        except Exception as e:
            logger.error(
                "stream_final_chunk_error conv=%s error=%s",
                ctx.conversation_id,
                str(e),
            )
            # Fallback: send empty delta chunk without metadata
            fallback_chunk = {
                "id": f"chatcmpl-{ctx.conversation_id[:12]}",
                "object": "chat.completion.chunk",
                "choices": [{"delta": {}, "finish_reason": "stop"}],
            }
            try:
                yield f"data: {json.dumps(fallback_chunk)}\n\n"
            except Exception as fallback_err:
                logger.error(
                    "stream_fallback_chunk_error conv=%s error=%s",
                    ctx.conversation_id,
                    str(fallback_err),
                )

        # Persist AFTER metadata chunk, BEFORE [DONE] marker.
        # This ensures [DONE] is sent immediately after content chunks,
        # reducing the gap between last content and stream termination.
        try:
            persist_result = await _persist_stream_turn(
                ctx, response_dict, input_tokens, output_tokens
            )
            match persist_result:
                case Err(error=error):
                    logger.error(
                        "stream_persist_failed conv=%s turn=%s reason=%s",
                        error.conversation_id,
                        error.turn_number,
                        error.reason,
                    )
        except Exception as e:
            logger.error(
                "stream_persist_error conv=%s error=%s",
                ctx.conversation_id,
                str(e),
            )

        logger.info(
            "stream_done conv=%s physical=%s",
            ctx.conversation_id[:12],
            ctx.physical_model,
        )

        # Send [DONE] marker ONCE to signal end of stream
        logger.info(
            "stream_sending_done_marker conv=%s physical=%s stream_id=%s",
            ctx.conversation_id[:12],
            ctx.physical_model,
            stream_id,
        )
        try:
            yield "data: [DONE]\n\n"
            logger.debug(
                "stream_done_marker_sent conv=%s",
                ctx.conversation_id[:12],
            )
        except Exception as e:
            logger.warning(
                "stream_done_yield_error conv=%s error=%s",
                ctx.conversation_id,
                str(e),
            )
    finally:
        # Close DB session, but DO NOT yield [DONE] again (already sent above)
        # Ensure DB session is closed even on client disconnect (GeneratorExit)
        if db is not None:
            try:
                await db.close()
            except Exception as exc:
                logger.debug("stream_db_close_finally_error err=%s", exc)


def _should_stream_cache_be_applied(ctx: StreamContext) -> bool:
    """feature check if cache optimization was applied for the streaming path."""
    if not ctx.provider:
        return False
    from src.adapters.cache.provider_cache import should_apply_cache_control

    # Bug 1 fix: derive cache provider from physical model prefix
    cache_provider = ctx.provider
    if ctx.physical_model and "/" in ctx.physical_model:
        model_prefix = ctx.physical_model.split("/")[0].lower()
        if model_prefix in ("anthropic",):
            cache_provider = model_prefix
    if cache_provider.lower() == "opencode-go":
        cache_provider = "opencode-go"
    return should_apply_cache_control(cache_provider)


def _map_stream_domain_error(error_msg: str) -> tuple[int, dict]:
    """Map domain error ValueError messages to OpenAI-compatible error detail for streaming path."""
    if error_msg.startswith("AllModelsFailed:"):
        return 503, _openai_error_detail("All physical models in the fallback chain failed.", "server_error")
    if error_msg.startswith("ContextTooLargeForAllModels:"):
        return 400, _openai_error_detail(error_msg.split(":", 1)[1].strip(), "context_length_exceeded")
    if error_msg.startswith("ContextUnusable:"):
        return 400, _openai_error_detail(error_msg.split(":", 1)[1].strip(), "context_length_exceeded")
    if error_msg.startswith("InputExceedsThreshold:"):
        return 400, _openai_error_detail(error_msg.split(":", 1)[1].strip(), "context_length_exceeded")
    if error_msg.startswith("ParallelToolsNotSupported:"):
        return 400, _openai_error_detail("Parallel tool calls are not supported by any physical model.", "unsupported_parameters")
    return 502, _openai_error_detail(str(error_msg), "server_error")


def _accumulate_tool_call_delta(tcd, tool_calls_by_index: dict[int, dict]) -> None:
    """Accumulate a single tool_call delta into the running index map."""
    idx = getattr(tcd, "index", None) or 0
    if idx not in tool_calls_by_index:
        tool_calls_by_index[idx] = {
            "id": getattr(tcd, "id", "") or "",
            "type": "function",
            "function": {"name": "", "arguments_parts": []},
        }
    entry = tool_calls_by_index[idx]
    tc_id = getattr(tcd, "id", None)
    if tc_id:
        entry["id"] = tc_id
    func = getattr(tcd, "function", None)
    if func:
        func_name = getattr(func, "name", None)
        if func_name:
            entry["function"]["name"] += func_name
        func_args = getattr(func, "arguments", None)
        if func_args:
            entry["function"]["arguments_parts"].append(func_args)


def _openai_error_detail(message: str, code: str) -> dict:
    """Build OpenAI-compatible error detail dict."""
    return {
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "param": None,
            "code": code,
        },
    }
