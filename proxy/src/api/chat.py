"""POST /v1/chat/completions — Main endpoint.

OpenAI-compatible format. Supports streaming SSE and non-streaming.
"""

import logging
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.api.metrics import metrics
from src.api.chat_streaming import _handle_streaming
from src.service.chat_models import (
    MetadataContext,
    build_proxy_metadata,
)
from src.service.chat_service import (
    process_chat_request,
)
from src.service.inline_commands import handle_inline_command
from src.service.pipeline_trace import PipelineTrace


router = APIRouter()
logger = logging.getLogger(__name__)


# ── Request/Response schemas ────────────────────────────────────────────────


class Message(BaseModel, extra="ignore"):
    role: str
    content: str | list[dict] | None = None
    name: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None


class ChatRequest(BaseModel, extra="ignore"):
    model: str
    messages: list[Message]
    conversation_id: str | None = None
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None
    stream_options: dict | None = None
    """OpenAI-compatible stream options (e.g. {"include_usage": true})."""
    thinking: dict | str | bool | None = None
    """Extended thinking toggle (Anthropic-compatible).
    ``{"type": "enabled", "budget_tokens": 16000}`` enables deep thinking;
    ``{"type": "disabled"}`` or ``False`` disables it.
    Currently supported by ``pensamiento-profundo-caro`` (qwen3.7-max)."""


# ── Endpoint ────────────────────────────────────────────────────────────────


@router.post(
    "/v1/chat/completions",
    responses={
        400: {
            "description": "Bad request - unknown pseudo-model, input exceeds threshold, unsupported content"
        },
        409: {
            "description": "Conflict - pseudo-model switch blocked due to incompatibility"
        },
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
    request_id = str(uuid.uuid4())[:8]  # Unique ID for this request

    # Create request trace for observability
    trace = PipelineTrace.create(conversation_id, request.model)

    logger.info(
        "chat_request_received request_id=%s conv=%s model=%s stream=%s messages=%d",
        request_id,
        conversation_id[:12],
        request.model,
        request.stream,
        len(request.messages),
    )

    trace.proxy_in(
        messages_count=len(request.messages),
        tools_count=len(request.tools) if request.tools else 0,
        stream=request.stream,
    )

    # Prepare messages as dicts
    messages = [msg.model_dump(exclude_none=True) for msg in request.messages]

    # If KeyVault middleware sanitized the body, use the sanitized messages
    if hasattr(fastapi_request.state, "_keyvault_body"):
        try:
            import json as _json
            sanitized = _json.loads(fastapi_request.state._keyvault_body)
            sanitized_msgs = sanitized.get("messages", [])
            if sanitized_msgs:
                messages = sanitized_msgs
        except Exception:
            pass

    # For streaming: detect secrets natively (middleware skips streaming)
    if request.stream and not hasattr(fastapi_request.state, "_keyvault_body"):
        try:
            from src.middleware.keyvault import _mask_messages, _KEYVAULT_SYSTEM_PROMPT
            secrets: dict[str, str] = {}
            body_copy = {"messages": [dict(m) for m in messages]}
            _mask_messages(body_copy, secrets)
            if secrets:
                # Inject KeyVault system prompt
                msgs = body_copy["messages"]
                insert_pos = 0
                for i, m in enumerate(msgs):
                    if m.get("role") != "system":
                        break
                    insert_pos = i + 1
                msgs.insert(insert_pos, {"role": "system", "content": _KEYVAULT_SYSTEM_PROMPT})
                messages = msgs
                fastapi_request.state.keyvault_secrets = secrets
                logger.info("keyvault_handler_stream secrets=%d", len(secrets))
        except Exception:
            pass

    # Log full request details including message content
    _msg_summaries = []
    for m in messages:
        _role = m.get("role", "?")
        _c = m.get("content", "")
        if isinstance(_c, str):
            _preview = _c[:2000].replace("\n", " ")
        elif isinstance(_c, list):
            _preview = f"[{len(_c)} parts]"
        else:
            _preview = str(_c)[:2000]
        _msg_summaries.append(f"{_role}={_preview}")
    logger.info(
        "chat_request_full request_id=%s conv=%s model=%s stream=%s "
        "messages=%d tools=%d thinking=%s msgs=%s",
        request_id,
        conversation_id[:12],
        request.model,
        request.stream,
        len(messages),
        len(request.tools) if request.tools else 0,
        str(request.thinking)[:50] if request.thinking else "none",
        " | ".join(_msg_summaries),
    )

    # Sprint 9: Check for inline commands early — if detected, respond
    # with the command output instead of calling the LLM.
    cmd_result = await handle_inline_command(
        messages=messages,
        conversation_id=request.conversation_id,
        db=db,
    )
    if cmd_result.handled and cmd_result.skip_llm:
        proxy_meta = {
            "command": cmd_result.response_metadata,
            "handled": True,
            "physical_model": "(command)",
            "pseudo_model": request.model,
            "conversation_id": conversation_id,
        }
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "model": request.model,
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
            "proxy_metadata": proxy_meta,
        }

    if request.stream:
        # Streaming path: session lifecycle is managed inside _handle_streaming
        # because _stream_response_generator runs lazily after this function returns.
        # The generator needs the session to persist the turn after streaming ends.
        logger.info(
            "chat_request_streaming request_id=%s conv=%s model=%s returning_generator",
            request_id,
            conversation_id[:12],
            request.model,
        )
        from src.service.chat_models import StreamingRequestContext
        result = await _handle_streaming(
            StreamingRequestContext(
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
                stream_options=request.stream_options,
                thinking=request.thinking,
                trace=trace,
                request=fastapi_request,
            )
        )
        logger.info(
            "chat_request_streaming_returned request_id=%s conv=%s",
            request_id,
            conversation_id[:12],
        )
        return result

    # Non-streaming path: session lifecycle managed here with try/finally
    try:
        response = await _handle_non_streaming(
            config=config,
            affinity=affinity,
            db=db,
            conversation_id=conversation_id,
            request=request,
            messages=messages,
            valkey=app_state.valkey,
            thinking=request.thinking,
            trace=trace,
        )
        trace.proxy_out(
            http_status=200, stream=False, details={"response_len": len(response.body)}
        )
        return response
    except HTTPException as e:
        await db.rollback()
        trace.proxy_out(http_status=e.status_code, stream=False)
        raise
    except ValueError as e:
        await db.rollback()
        error_msg = str(e)
        status_code, error_detail = _map_domain_error(error_msg)
        trace.proxy_out(http_status=status_code, stream=False)
        raise HTTPException(
            status_code=status_code,
            detail=error_detail,
        ) from e
    except Exception as e:
        await db.rollback()
        metrics.record_error(502, "PROXY_ERROR")
        trace.proxy_out(http_status=502, stream=False)
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
    finally:
        await db.close()


def _map_domain_error(error_msg: str) -> tuple[int, dict]:
    """Map domain error ValueError messages to OpenAI-compatible error detail.

    Service layer wraps domain errors as ``ValueError(f"ErrorName: {detail}")``.
    Returns (status_code, error_detail_dict) for HTTPException.
    """
    if error_msg.startswith("AllModelsFailed:"):
        return 503, _openai_error("All physical models in the fallback chain failed.", "server_error")
    if error_msg.startswith("ContextTooLargeForAllModels:"):
        return 400, _openai_error(error_msg.split(":", 1)[1].strip(), "context_length_exceeded")
    if error_msg.startswith("ContextUnusable:"):
        return 400, _openai_error(error_msg.split(":", 1)[1].strip(), "context_length_exceeded")
    if error_msg.startswith("InputExceedsThreshold:"):
        return 400, _openai_error(error_msg.split(":", 1)[1].strip(), "context_length_exceeded")
    if error_msg.startswith("ParallelToolsNotSupported:"):
        return 400, _openai_error("Parallel tool calls are not supported by any physical model in this pseudo-model.", "unsupported_parameters")
    return 502, _openai_error(str(error_msg), "server_error")


def _openai_error(message: str, code: str) -> dict:
    """Build OpenAI-compatible error detail dict."""
    return {
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "param": None,
            "code": code,
        },
    }


async def _handle_non_streaming(
    config,
    affinity,
    db,
    conversation_id: str,
    request: ChatRequest,
    messages: list[dict],
    valkey=None,
    thinking: dict | str | bool | None = None,
    trace: PipelineTrace | None = None,
) -> JSONResponse:
    """Non-streaming request: call LLM, save turn, return JSONResponse with headers."""
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
        valkey=valkey,
        thinking=thinking,
        trace=trace,
    )

    # Build response with Sprint 2 + Sprint 4 proxy_metadata
    response_dict = result.response
    response_dict["proxy_metadata"] = build_proxy_metadata(
        MetadataContext(
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
            images_described=result.images_described,
            images_described_by=result.images_described_by,
            images_degraded_manually=result.images_degraded_manually,
            router_suggestion=result.router_suggestion,
            context_alert=result.context_alert,
            cache_metadata=result.cache_metadata,
        )
    )

    # Always return conversation_id so the client can persist and reuse it
    response_dict["conversation_id"] = result.conversation_id

    # Forward provider response headers to HTTP response
    # Exclude Content-Encoding and Transfer-Encoding to avoid httpx double-decompression
    _excluded = {"content-encoding", "transfer-encoding"}
    headers: dict[str, str] = {"X-Conversation-Id": conversation_id}
    provider_headers = response_dict.get("provider_headers")
    if provider_headers and isinstance(provider_headers, dict):
        for h, v in provider_headers.items():
            if h.lower() not in _excluded:
                headers[h] = str(v)

    # Log full response for debugging
    resp_content = (
        response_dict.get("choices", [{}])[0].get("message", {}).get("content") or ""
    )
    logger.info(
        "chat_response_full conv=%s model=%s physical=%s "
        "content_len=%d finish=%s reasoning=%s preview=%s",
        conversation_id[:12],
        response_dict.get("model", "?"),
        result.physical_model,
        len(resp_content),
        response_dict.get("choices", [{}])[0].get("finish_reason", "?"),
        bool(
            response_dict.get("choices", [{}])[0]
            .get("message", {})
            .get("reasoning_content")
        ),
        resp_content[:2000],
    )

    return JSONResponse(content=response_dict, headers=headers)
