"""Streaming handlers for /v1/chat/completions.

Extracted from chat.py to keep individual files under 600 lines.
"""

import json
import logging
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
from src.domain.types import Ok, Err
from src.service.chat_models import StreamContext
from src.service.context_alert import get_context_alert
from src.service.chat_fallback import _try_physical_model, call_with_fallback
from src.service.chat_messages import build_conversation_messages, handle_auto_describe
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
from src.api.chat_stream_persistence import (
    _build_final_metadata_chunk,
    _extract_tokens_from_chunks,
    _filter_eligible_models,
    _persist_stream_turn,
    _resolve_physical_model,
)


logger = logging.getLogger(__name__)


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
    stream_options: dict | None = None,
    thinking: dict | str | bool | None = None,
    trace: PipelineTrace | None = None,
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
            db=db,
            config=config,
            affinity=affinity,
            conversation_id=conversation_id,
            pseudo_model_name=pseudo_model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            stream_options=stream_options,
            thinking=thinking,
            trace=trace,
        )
    except HTTPException:
        try:
            await db.close()
        except Exception as exc:
            logger.debug("stream_db_close_error err=%s", exc)
        if trace:
            trace.proxy_out(http_status=400, stream=True)
        raise
    except ValueError as e:
        try:
            await db.close()
        except Exception as exc:
            logger.debug("stream_db_close_error err=%s", exc)
        error_msg = str(e)
        status_code, error_type = _map_stream_domain_error(error_msg)
        if trace:
            trace.proxy_out(http_status=status_code, stream=True)
        raise HTTPException(
            status_code=status_code,
            detail={"error": error_type, "message": error_msg},
        ) from e
    except Exception as e:
        try:
            await db.close()
        except Exception as exc:
            logger.debug("stream_db_close_error err=%s", exc)
        metrics.record_error(502, "PROXY_ERROR")
        if trace:
            trace.proxy_out(http_status=502, stream=True)
        raise HTTPException(
            status_code=502,
            detail={"error": "PROXY_ERROR", "message": str(e)},
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
):
    """Pre-stream logic (runs synchronously before SSE starts)."""
    # Resolve model
    resolved_model = normalize_model_name(pseudo_model_name, config)
    if resolved_model not in config.pseudo_models:
        pm_schema = build_passthrough_pseudo_model(resolved_model)
    else:
        pm_schema = config.pseudo_models[resolved_model]

    # Detect capabilities in incoming messages
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

    # Check input threshold
    estimated_input = estimate_tokens(messages)
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

    physical_model, provider, selected_phys_model = _resolve_physical_model(
        existing_affinity,
        session_caps,
        eligible_models,
        pm_schema,
    )

    # NEW: Validate that the SELECTED physical model can handle the incoming content
    # (this happens AFTER model selection, not before like the old logic)
    delegation = validate_physical_model_content(turn_caps, selected_phys_model)

    # Debug logging for content delegation
    logger.info(
        "content_validation_stream trace=%s conv=%s model=%s has_images=%s model_vision=%s delegation=%s",
        trace.request_id if trace else "?",
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
            trace.request_id if trace else "?",
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

    # Sprint 5: Auto-describe images on pseudo-model switch (Bug 6 fix)
    auto_describe_meta: dict | None = None
    messages_for_llm: list[dict] = messages
    if conv is not None and not is_new and resolved_model != conv.pseudo_model:
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

    # ── Load conversation history for streaming context ──────────
    if not is_new and conv is not None and conv.turns:
        messages_for_llm = build_conversation_messages(conv, messages_for_llm)

    # No automatic compaction — if threshold is exceeded, error is returned.
    active_messages = messages_for_llm

    # ── Sprint 6: Context alerts ──────────────────────────────────────────
    context_alert = get_context_alert(
        total_tokens=conv.total_tokens if conv else 0,
        context_window=pm_schema.context_window,
        conversation_id=conversation_id,
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
                    "endpoint": f"POST /conversations/{conversation_id}/compact",
                    "description": (
                        "Compact the conversation history into a snapshot. "
                        "Original history is preserved."
                    ),
                },
            },
        )

    # ── Sprint 5: Router LLM ──────────────────────────────────────────────
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
    )

    # Sprint 11: Update physical_model from actual response when fallback occurred
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

    _streaming_response = StreamingResponse(
        _stream_response_generator(
            StreamContext(
                litellm_response=litellm_response,
                conversation_id=conversation_id,
                pseudo_model=resolved_model,
                physical_model=physical_model,
                fallback_info=fallback_info,
                affinity_maintained=not is_new and existing_affinity == physical_model,
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
                # Sprint 11: pass schema + kwargs for token-limit continuation
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
            )
        ),
        media_type="text/event-stream",
        headers=_sse_headers,
    )
    return _streaming_response


def _dump_chunk_for_sse(chunk) -> str:
    """Serialize a streaming chunk to JSON, ensuring both content and
    reasoning_content are included in the delta as separate fields."""
    try:
        d = json.loads(chunk.model_dump_json())
    except (AttributeError, TypeError, ValueError):
        return chunk.model_dump_json() if hasattr(chunk, "model_dump_json") else "{}"
    try:
        delta = chunk.choices[0].delta
        if delta:
            rc = getattr(delta, "reasoning_content", None)
            if rc is not None and rc != "":
                choice = d.get("choices", [{}])[0]
                if "delta" not in choice:
                    choice["delta"] = {}
                choice["delta"]["reasoning_content"] = rc
    except (AttributeError, IndexError, TypeError):
        pass
    return json.dumps(d)


async def _stream_response_generator(ctx: StreamContext):
    """SSE streaming: forward chunks, persist turn on success, append metadata.

    Sprint 11: supports token-limit continuation — when a model finishes with
    ``finish_reason="length"`` and more physical models are available, the
    generator suppresses the ``"length"`` finish_reason (sets to ``null``),
    appends the accumulated content as an assistant message, and seamlessly
    continues streaming from the next model.
    """
    import uuid

    stream_id = str(uuid.uuid4())[:8]  # Unique ID for this streaming session
    chunks: list = []
    db = ctx.db

    # Sprint 11: prepare for token-limit continuation across multiple models
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

    try:
        # ── Multi-stream loop (continues when a model hits token limit) ──
        while True:
            finish_reason: str | None = None
            try:
                async for chunk in current_stream:
                    chunks.append(chunk)

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
                                    # Extract index safely (default to 0)
                                    idx = getattr(tcd, "index", None)
                                    if idx is None:
                                        idx = 0

                                    # Initialize tool_calls_by_index entry if needed
                                    if idx not in tool_calls_by_index:
                                        tool_calls_by_index[idx] = {
                                            "id": getattr(tcd, "id", "") or "",
                                            "type": "function",
                                            "function": {
                                                "name": "",
                                                "arguments_parts": [],
                                            },
                                        }

                                    entry = tool_calls_by_index[idx]

                                    # Update id if present
                                    tc_id = getattr(tcd, "id", None)
                                    if tc_id:
                                        entry["id"] = tc_id

                                    # Extract and accumulate function details
                                    func = getattr(tcd, "function", None)
                                    if func:
                                        func_name = getattr(func, "name", None)
                                        if func_name:
                                            entry["function"]["name"] += func_name

                                        func_args = getattr(func, "arguments", None)
                                        if func_args:
                                            entry["function"]["arguments_parts"].append(
                                                func_args
                                            )
                                except Exception as e:
                                    logger.warning(
                                        "stream_tool_call_delta_error conv=%s | error=%s",
                                        ctx.conversation_id[:12],
                                        str(e),
                                    )
                                    # Continue processing other tool_calls
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

                    if (
                        fr == "length"
                        and current_idx < len(phys_models) - 1
                        and getattr(ctx.pm_schema, "continue_on_length", False)
                    ):
                        # Suppress "length" — set to null so client continues
                        chunk_dict = json.loads(_dump_chunk_for_sse(chunk))
                        if chunk_dict.get("choices"):
                            chunk_dict["choices"][0]["finish_reason"] = None
                        yield f"data: {json.dumps(chunk_dict)}\n\n"
                        finish_reason = "length"
                        break

                    yield f"data: {_dump_chunk_for_sse(chunk)}\n\n"

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
                estimate_tokens(cont_messages),
                _trace_id,
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
        input_tokens, output_tokens, response_dict = _extract_tokens_from_chunks(chunks)

        logger.info(
            "stream_complete conv=%s physical=%s chunks=%d tokens=%d+%d",
            ctx.conversation_id[:12],
            ctx.physical_model,
            len(chunks),
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

        # Sprint 7: extract cache metadata from streaming response
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

        # Handle Result[..., StreamPersistenceFailed] from _persist_stream_turn
        persist_result = await _persist_stream_turn(
            ctx, response_dict, input_tokens, output_tokens
        )

        # Determine session_caps and conv for final metadata — use defaults if persistence failed
        session_caps = ctx.session_caps
        conv = ctx.conv
        match persist_result:
            case Ok(value=(db, conv, session_caps)):
                pass
            case Err(error=error):
                logger.error(
                    "stream_persist_failed conv=%s turn=%s reason=%s",
                    error.conversation_id,
                    error.turn_number,
                    error.reason,
                )

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

        logger.info(
            "stream_done conv=%s physical=%s",
            ctx.conversation_id[:12],
            ctx.physical_model,
        )
    finally:
        # Always send [DONE] to signal end of stream, even if metadata failed
        try:
            logger.info(
                "stream_sending_done_marker conv=%s physical=%s stream_id=%s",
                ctx.conversation_id[:12],
                ctx.physical_model,
                stream_id,
            )
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

        # Ensure DB session is closed even on client disconnect (GeneratorExit)
        if db is not None:
            try:
                await db.close()
            except Exception as exc:
                logger.debug("stream_db_close_finally_error err=%s", exc)


def _should_stream_cache_be_applied(ctx: StreamContext) -> bool:
    """Sprint 7: check if cache optimization was applied for the streaming path."""
    if not ctx.provider:
        return False
    from src.adapters.cache.provider_cache import should_apply_cache_control

    # Bug 1 fix: derive cache provider from physical model prefix
    cache_provider = ctx.provider
    if ctx.physical_model and "/" in ctx.physical_model:
        model_prefix = ctx.physical_model.split("/")[0].lower()
        if model_prefix in ("anthropic",):
            cache_provider = model_prefix
    return should_apply_cache_control(cache_provider)


def _map_stream_domain_error(error_msg: str) -> tuple[int, str]:
    """Map domain error ValueError messages to HTTP status codes for streaming path."""
    if error_msg.startswith("AllModelsFailed:"):
        return 503, "ALL_MODELS_FAILED"
    if error_msg.startswith("ContextTooLargeForAllModels:"):
        return 400, "CONTEXT_TOO_LARGE"
    if error_msg.startswith("ContextUnusable:"):
        return 400, "CONTEXT_UNUSABLE"
    if error_msg.startswith("InputExceedsThreshold:"):
        return 400, "INPUT_EXCEEDS_THRESHOLD"
    if error_msg.startswith("ParallelToolsNotSupported:"):
        return 400, "PARALLEL_TOOLS_NOT_SUPPORTED"
    return 502, "PROXY_ERROR"
