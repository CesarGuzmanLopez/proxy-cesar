"""SSE streaming helpers extracted from chat_streaming.py for file size compliance.

Contains: content type counting, analysis message building, chunk formatting,
cache metadata helpers, and error detail formatting.
"""

import json
import logging

from src.service.chat_models import StreamContext
from src.service.tools_edge_cases import truncate_tool_result

logger = logging.getLogger(__name__)

# ── Content type map for SSE analysis messages ───────────────────────────

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
    counts: dict[str, int] = {"images": 0, "pdfs": 0, "audios": 0, "documents": 0}
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
    parts: list[str] = []
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
    return f"Analizando {' y '.join(parts)} desde archivos adjuntos..."


def _chunk_to_dict(chunk) -> dict:
    """Convert a streaming chunk to a dict (single serialization, no string intermediate).

    Uses model_dump with exclude_none=False first. Falls back to dict() for
    non-Pydantic responses (edge case from some providers).
    """
    try:
        return chunk.model_dump(exclude_none=False)
    except (AttributeError, TypeError):
        try:
            return dict(chunk)
        except Exception:
            return {}


# ── Cache metadata helpers ───────────────────────────────────────────────


def _should_stream_cache_be_applied(ctx: StreamContext) -> bool:
    """Determine if streaming cache metadata should be applied to the final response."""
    from src.adapters.cache.provider_cache import CACHE_ELIGIBLE_ARGS

    if not ctx.call_kwargs:
        return False
    if not any(k in ctx.call_kwargs for k in CACHE_ELIGIBLE_ARGS):
        return False
    return True


def _map_stream_domain_error(error_msg: str) -> tuple[int, dict]:
    """Map domain error to HTTP status code and error detail."""
    if "ban" in error_msg.lower():
        return 429, _openai_error_detail(error_msg, "rate_limited")
    if "tool_use_failed" in error_msg:
        return 400, _openai_error_detail(
            "The model attempted to call a tool not provided in the request. "
            "This typically means the client did not send the `tools` parameter.",
            "unsupported_parameters",
        )
    if "parallel_tool_calls_not_supported" in error_msg:
        return 400, _openai_error_detail(
            "Parallel tool calls are not supported by any physical model.",
            "unsupported_parameters",
        )
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
