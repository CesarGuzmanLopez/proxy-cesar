"""Stream-turn persistence helpers for /v1/chat/completions.

Extracted from chat.py to keep individual files under 600 lines.
"""

import copy
import logging

from sqlalchemy import func
from sqlmodel.ext.asyncio.session import AsyncSession

from src.adapters.db.models import Conversation, ConversationTurn
from src.domain.capabilities import SessionCapabilities
from src.domain.errors import StreamPersistenceFailed
from src.domain.types import Ok, Err, Result
from src.service.capability_detector import accumulate_capabilities
from src.service.chat_models import (
    MetadataContext,
    StreamContext,
    build_proxy_metadata,
)
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


def _resolve_physical_model(
    existing_affinity: str | None,
    session_caps,
    eligible_models: list,
    pm_schema,
) -> tuple[str, str | None, object]:
    """Resolve physical model + provider based on affinity, caps, and eligibility.

    Returns: (physical_model_name: str, provider: str | None, selected_phys_obj: PhysicalModelSchema)
    """
    pinned_model = existing_affinity
    if pinned_model and session_caps.has_parallel_tools:
        from src.service.tool_filter import is_pinned_model_eligible

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
    return selected_phys.model, selected_phys.provider, selected_phys


def _filter_eligible_models(
    pm_schema,
    session_caps,
) -> tuple[list, bool, str | None]:
    """Filter eligible physical models based on session capabilities (parallel tools)."""
    from src.service.tool_filter import get_eligible_models

    eligible_models = get_eligible_models(pm_schema.physical_models, session_caps)
    tools_filter_applied = session_caps.has_parallel_tools and len(
        eligible_models
    ) < len(pm_schema.physical_models)
    tools_filter_reason = "parallel_tools_required" if tools_filter_applied else None
    return eligible_models, tools_filter_applied, tools_filter_reason


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
    if (
        tool_choice == "required"
        and tool_calls
        and not enforce_tool_choice(response_dict, tool_choice)
    ):
        turn_caps.tools_incomplete = True
    thinking_content = extract_thinking_content(response_dict, provider)
    tools_incomplete = turn_caps.tools_incomplete
    tools_level = determine_tool_level_for_turn(
        tool_calls=tool_calls,
        tool_definitions=tool_defs,
        tools_incomplete=tools_incomplete,
    )
    return tool_calls, tools_incomplete, tools_level, thinking_content


def _truncate_tool_results(messages: list[dict] | None) -> list[dict]:
    """Truncate long tool result content without mutating originals."""
    if not messages:
        return messages or []
    result = list(messages)
    for i, msg in enumerate(result):
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            truncated = truncate_tool_result(content)
            if truncated != content:
                new_msg = copy.deepcopy(msg)
                new_msg["content"] = truncated
                result[i] = new_msg
    return result


async def _persist_stream_turn(
    ctx: StreamContext,
    response_dict: dict,
    input_tokens: int,
    output_tokens: int,
) -> Result[
    tuple[AsyncSession, Conversation, SessionCapabilities], StreamPersistenceFailed
]:
    """Persist turn after successful stream — extracted for cognitive complexity.

    Returns Ok[db, conv, updated_caps] on success, or Err[StreamPersistenceFailed] on error.
    The caller must handle the Result type and close the DB session if needed.
    """
    db = ctx.db
    conv = ctx.conv
    if not (
        db
        and conv is not None
        and ctx.conv_uuid is not None
        and ctx.turn_caps is not None
    ):
        logger.warning(
            "persist_stream_turn guard_triggered conv=%s missing_db=%s missing_conv=%s",
            ctx.conversation_id,
            db is None,
            conv is None,
        )
        return Err(
            StreamPersistenceFailed(
                conversation_id=ctx.conversation_id,
                turn_number=0,
                reason="missing_context_for_persistence",
            )
        )

    turn_number = 1
    try:
        tool_defs: list[dict] | None = ctx.tools
        _, tools_incomplete, tools_level, thinking_content = _build_turn_tool_metadata(
            response_dict=response_dict,
            tool_defs=tool_defs,
            tool_choice=ctx.tool_choice,
            turn_caps=ctx.turn_caps,
            provider=ctx.provider,
        )
        truncated_messages = _truncate_tool_results(ctx.messages)
        if conv.turns:
            turn_number = max(t.turn_number for t in conv.turns) + 1
        turn = ConversationTurn(
            conversation_id=ctx.conv_uuid,
            turn_number=turn_number,
            pseudo_model=ctx.pseudo_model,
            physical_model=ctx.physical_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            messages=truncated_messages or [],
            response=response_dict,
            fallback_applied=ctx.fallback_info.applied if ctx.fallback_info else False,
            fallback_reason=ctx.fallback_info.reason if ctx.fallback_info else None,
            turn_type="normal",
            had_images=ctx.turn_caps.has_images,
            had_tools=ctx.turn_caps.has_tools,
            had_parallel_tools=ctx.turn_caps.has_parallel_tools,
            tool_definitions=tool_defs,
            thinking_blocks={"content": thinking_content} if thinking_content else None,
            tools_incomplete=tools_incomplete,
            tools_level_used=tools_level,
        )
        db.add(turn)
        conv.physical_model = ctx.physical_model
        conv.total_tokens += input_tokens + output_tokens
        conv.updated_at = func.now()

        # Accumulate capabilities BEFORE commit — same transaction as turn save
        updated_caps = await accumulate_capabilities(
            db, ctx.conv_uuid, ctx.turn_caps, ctx.session_caps
        )

        await db.commit()
        return Ok((db, conv, updated_caps))
    except Exception as e:
        logger.error(
            "persist_stream_turn_error conv=%s turn_number=%s error=%s",
            ctx.conversation_id,
            turn_number,
            str(e),
        )
        try:
            await db.rollback()
        except Exception as rollback_err:
            logger.warning(
                "persist_stream_turn_rollback_error conv=%s error=%s",
                ctx.conversation_id,
                str(rollback_err),
            )
        return Err(
            StreamPersistenceFailed(
                conversation_id=ctx.conversation_id,
                turn_number=turn_number,
                reason=str(e),
            )
        )


def _extract_usage(chunks: list) -> tuple[int, int]:
    """Extract prompt/completion tokens from the last chunk."""
    if not chunks:
        return 0, 0
    last = chunks[-1]
    try:
        usage = getattr(last, "usage", None)
        if usage is not None:
            pt = getattr(usage, "prompt_tokens", 0)
            ct = getattr(usage, "completion_tokens", 0)
            return (
                int(pt) if isinstance(pt, (int, float)) else 0,
                int(ct) if isinstance(ct, (int, float)) else 0,
            )
    except (TypeError, ValueError):
        pass
    return 0, 0


def _extract_chunk_metadata(chunk) -> tuple[str | None, int | None, str | None]:
    """Extract id, created, model from a chunk if not already found."""
    return (
        getattr(chunk, "id", None),
        getattr(chunk, "created", None),
        getattr(chunk, "model", None),
    )


def _accumulate_tool_call_delta(tc, tool_calls_map: dict[int, dict]) -> None:
    """Accumulate a single tool_call delta into the index map."""
    idx = getattr(tc, "index", None) or 0
    if idx not in tool_calls_map:
        tool_calls_map[idx] = {
            "id": None,
            "type": "function",
            "function": {"name": None, "arguments": ""},
        }
    tc_id = getattr(tc, "id", None)
    if tc_id and isinstance(tc_id, str):
        tool_calls_map[idx]["id"] = tc_id
    func = getattr(tc, "function", None)
    if func:
        func_name = getattr(func, "name", None)
        if func_name and isinstance(func_name, str):
            tool_calls_map[idx]["function"]["name"] = func_name
        func_args = getattr(func, "arguments", None)
        if func_args and isinstance(func_args, str):
            tool_calls_map[idx]["function"]["arguments"] += func_args


def _build_tool_calls_list(tool_calls_map: dict[int, dict]) -> list[dict[str, object]]:
    """Convert accumulated tool_calls map to validated list."""
    result: list[dict[str, object]] = []
    for idx in sorted(tool_calls_map.keys()):
        tc = tool_calls_map[idx]
        if tc["id"] and tc["function"]["name"]:
            result.append({
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"],
                },
            })
    return result


def _extract_message_and_metadata(chunks: list) -> tuple[dict, str | None, int | None, str | None, str | None]:
    """Reconstruct message, tool_calls, finish_reason and metadata from ALL chunks."""
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls_map: dict[int, dict[str, object]] = {}
    finish_reason: str | None = None
    response_id: str | None = None
    response_created: int | None = None
    response_model: str | None = None

    for chunk in chunks:
        try:
            chunk_id, chunk_created, chunk_model = _extract_chunk_metadata(chunk)
            if response_id is None:
                response_id = chunk_id
            if response_created is None:
                response_created = chunk_created
            if response_model is None:
                response_model = chunk_model

            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            choice = choices[0]
            delta = getattr(choice, "delta", None)

            if delta:
                content_val = getattr(delta, "content", None)
                if isinstance(content_val, str):
                    content_parts.append(content_val)

                reasoning_val = getattr(delta, "reasoning_content", None)
                if isinstance(reasoning_val, str) and reasoning_val:
                    reasoning_parts.append(reasoning_val)

                tc_deltas = getattr(delta, "tool_calls", None)
                if tc_deltas:
                    for tc in tc_deltas:
                        try:
                            _accumulate_tool_call_delta(tc, tool_calls_map)
                        except Exception as e:
                            logger.warning(
                                "extract_tool_delta_error | idx=%s error=%s",
                                getattr(tc, "index", "unknown"),
                                str(e),
                            )
                            continue

            fr = getattr(choice, "finish_reason", None)
            if fr is not None:
                finish_reason = fr
        except (AttributeError, IndexError, TypeError):
            pass

    content = "".join(content_parts) if content_parts else None
    reasoning_content = "".join(reasoning_parts) if reasoning_parts else None
    tool_calls_list = _build_tool_calls_list(tool_calls_map)

    message: dict[str, object] = {"role": "assistant", "content": content}
    if reasoning_content:
        message["reasoning_content"] = reasoning_content
    if tool_calls_list:
        message["tool_calls"] = tool_calls_list

    return message, response_id, response_created, response_model, finish_reason


def _extract_usage_dict(chunks: list) -> dict[str, object] | None:
    """Extract usage dict from the last chunk."""
    if not chunks:
        return None
    last = chunks[-1]
    try:
        usage = getattr(last, "usage", None)
        if usage is not None:
            result: dict[str, object] = {}
            for attr in ("prompt_tokens", "completion_tokens", "total_tokens"):
                val = getattr(usage, attr, None)
                if val is not None:
                    result[attr] = int(val) if isinstance(val, (int, float)) else val
            return result
    except Exception:
        pass
    return None


def _extract_tokens_from_chunks(
    last_chunk_or_list,
    accumulated_content: str | None = None,
    tool_calls: list[dict] | None = None,
) -> tuple[int, int, dict]:
    """Extract input/output tokens and response dict.

    Supports two call patterns for backward compatibility:
      1. New streaming path: (last_chunk, accumulated_content, tool_calls)
      2. Legacy test path: (chunks_list,) — uses old accumulation logic

    Args:
        last_chunk_or_list: Final streaming chunk OR legacy list of chunks.
        accumulated_content: Re-assembled content string (new path).
        tool_calls: Accumulated tool_calls list (new path).
    """
    # ── Legacy test path: called with a list of chunks ───────────────────────
    if isinstance(last_chunk_or_list, list):
        chunks = last_chunk_or_list
        if not chunks:
            return 0, 0, {}
        input_tokens, output_tokens = _extract_usage(chunks)
        message, response_id, response_created, response_model, finish_reason = _extract_message_and_metadata(
            chunks
        )
        usage_dict = _extract_usage_dict(chunks)
        response_dict: dict[str, object] = {}
        if response_id:
            response_dict["id"] = response_id
        if response_created:
            response_dict["created"] = response_created
        if response_model:
            response_dict["model"] = response_model
        response_dict["object"] = "chat.completion"
        response_dict["choices"] = [
            {"message": message, "finish_reason": finish_reason or "stop"}
        ]
        if usage_dict:
            response_dict["usage"] = usage_dict
        return input_tokens, output_tokens, response_dict

    # ── New streaming path ───────────────────────────────────────────────────
    last_chunk = last_chunk_or_list
    _input_tokens = 0
    _output_tokens = 0

    if last_chunk is not None:
        _input_tokens, _output_tokens = _extract_usage_from_chunk(last_chunk)

    _usage_dict: dict = {}
    if last_chunk is not None:
        _usage_dict = _extract_usage_dict_from_chunk(last_chunk)

    response_id = getattr(last_chunk, "id", None) if last_chunk else None
    response_created = getattr(last_chunk, "created", None) if last_chunk else None
    response_model = getattr(last_chunk, "model", None) if last_chunk else None

    _response_dict: dict[str, object] = {}
    if response_id:
        _response_dict["id"] = response_id
    if response_created:
        _response_dict["created"] = response_created
    if response_model:
        _response_dict["model"] = response_model
    _response_dict["object"] = "chat.completion"

    _message: dict[str, object] = {"content": accumulated_content or ""}
    if tool_calls:
        _message["tool_calls"] = tool_calls
    _response_dict["choices"] = [
        {"message": _message, "finish_reason": "stop"}
    ]
    if _usage_dict:
        _response_dict["usage"] = _usage_dict

    return _input_tokens, _output_tokens, _response_dict


def _extract_usage_from_chunk(chunk) -> tuple[int, int]:
    """Extract prompt/completion tokens from a single chunk."""
    try:
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            pt = getattr(usage, "prompt_tokens", 0)
            ct = getattr(usage, "completion_tokens", 0)
            return (
                int(pt) if isinstance(pt, (int, float)) else 0,
                int(ct) if isinstance(ct, (int, float)) else 0,
            )
    except (TypeError, ValueError):
        pass
    return 0, 0


def _extract_usage_dict_from_chunk(chunk) -> dict:
    """Extract usage dict from a single chunk."""
    try:
        usage = getattr(chunk, "usage", None)
        if usage is None:
            return {}
        result = {}
        pt = getattr(usage, "prompt_tokens", None)
        if pt is not None:
            result["prompt_tokens"] = int(pt)
        ct = getattr(usage, "completion_tokens", None)
        if ct is not None:
            result["completion_tokens"] = int(ct)
        tt = getattr(usage, "total_tokens", None)
        if tt is not None:
            result["total_tokens"] = int(tt)
        return result
    except (TypeError, ValueError):
        return {}


def _build_final_metadata_chunk(
    ctx: StreamContext, conv, session_caps, input_tokens: int, output_tokens: int
) -> dict:
    """Build the final SSE chunk with complete proxy_metadata."""
    final_context_tokens = (
        conv.total_tokens if conv is not None else (input_tokens + output_tokens)
    )
    metadata = build_proxy_metadata(
        MetadataContext(
            pseudo_model=ctx.pseudo_model,
            physical_model=ctx.physical_model,
            conversation_id=ctx.conversation_id,
            context_tokens=final_context_tokens,
            context_window=ctx.context_window,
            fallback_info=ctx.fallback_info,
            affinity_maintained=ctx.affinity_maintained,
            session_caps=session_caps,
            compatibility_warning=ctx.compatibility_warning,
            compatibility_details=ctx.compatibility_details,
            tools_filter_applied=ctx.tools_filter_applied,
            tools_filter_reason=ctx.tools_filter_reason,
            images_described=ctx.images_described,
            images_described_by=ctx.images_described_by,
            images_degraded_manually=ctx.images_degraded_manually,
            router_suggestion=ctx.router_suggestion,
            context_alert=ctx.context_alert,
            cache_metadata=ctx.cache_metadata,
        )
    )
    return {
        "id": f"chatcmpl-{ctx.conversation_id[:12]}",
        "object": "chat.completion.chunk",
        "choices": [{"delta": {}, "finish_reason": "stop"}],
        "proxy_metadata": metadata,
    }
