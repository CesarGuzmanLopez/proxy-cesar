"""Capability detection for incoming messages.

Detects images, audio, PDF, video, tools, and parallel tools in messages.
Also handles accumulation of capability flags in DB and token estimation.

python.md §4: pure functions, immutable data, declarative style.
Sprint 3: tiktoken-based token counting replaces 4-char heuristic.
"""

import functools
import uuid

import tiktoken
from sqlalchemy import update
from sqlmodel.ext.asyncio.session import AsyncSession

from src.adapters.db.models import Conversation
from src.domain.capabilities import SessionCapabilities, TurnCapabilities

# Default encoding for token counting.
# Uses o200k_base (GPT-4o) as a general-purpose encoding for all providers.
_TIKTOKEN_ENCODING = "o200k_base"


@functools.lru_cache(maxsize=1)
def _get_encoding():
    """Get the tiktoken encoding, with fallback to cl100k_base."""
    try:
        return tiktoken.get_encoding(_TIKTOKEN_ENCODING)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def _scan_multimodal_content(content: list, caps: TurnCapabilities) -> None:
    """Scan multimodal content parts for images, audio, PDF, video."""
    for part in content:
        part_type = part.get("type", "")

        if part_type == "image_url":
            caps.has_images = True
        elif part_type == "input_audio":
            caps.has_audio = True
        elif part_type == "file":
            mime = part.get("mime_type", part.get("mimetype", ""))
            mime_lower = mime.lower()
            if "pdf" in mime_lower:
                caps.has_pdf = True
            elif any(v in mime_lower for v in ("video/", "mp4", "webm", "avi")):
                caps.has_video = True
        elif part_type in ("video_url", "video"):
            caps.has_video = True


def _scan_tool_calls(msg: dict, caps: TurnCapabilities) -> None:
    """Scan a message for tool calls and parallel tool calls."""
    tool_calls = msg.get("tool_calls")
    if tool_calls and isinstance(tool_calls, list) and len(tool_calls) > 0:
        caps.has_tools = True
        if len(tool_calls) > 1:
            caps.has_parallel_tools = True


def detect_turn_capabilities(
    messages: list[dict],
    tools: list[dict] | None = None,
) -> TurnCapabilities:
    """Scan all messages in this turn to detect capabilities.

    Rules are applied deterministically — no ML, no heuristics.
    Scans ALL messages, not just the last one.

    Args:
        messages: List of message dicts (OpenAI format).
        tools: Optional list of tool definitions from the request.

    Returns:
        TurnCapabilities with flags set based on message content.
    """
    caps = TurnCapabilities()

    # 1. Tool definitions in the request
    if tools and len(tools) > 0:
        caps.has_tools = True

    for msg in messages:
        content = msg.get("content")

        # 2. Content is an array (multimodal)
        if isinstance(content, list):
            _scan_multimodal_content(content, caps)

        # 3. Tool calls in assistant messages
        _scan_tool_calls(msg, caps)

        # 4. Tool results (role: "tool")
        if msg.get("role") == "tool":
            caps.has_tools = True

    return caps


async def load_session_capabilities(
    db: AsyncSession,
    conversation_id: uuid.UUID,
    total_tokens: int = 0,
) -> SessionCapabilities:
    """Load accumulated capabilities from DB for a conversation."""
    conv = await db.get(Conversation, conversation_id)
    if conv is None:
        return SessionCapabilities(
            conversation_id=str(conversation_id),
            total_tokens=total_tokens,
        )

    return SessionCapabilities(
        conversation_id=str(conversation_id),
        has_images=conv.capability_has_images,
        has_audio=conv.capability_has_audio,
        has_pdf=conv.capability_has_pdf,
        has_video=conv.capability_has_video,
        has_tools=conv.capability_has_tools,
        has_parallel_tools=conv.capability_has_parallel_tools,
        total_tokens=conv.total_tokens,
        max_tools_level=getattr(conv, "max_tools_level", 0),
        # Sprint 5
        images_described=getattr(conv, "images_described", 0),
        images_degraded_manually=getattr(conv, "images_degraded_manually", False),
    )


async def accumulate_capabilities(
    db: AsyncSession,
    conversation_id: uuid.UUID,
    turn_caps: TurnCapabilities,
    existing: SessionCapabilities,
) -> SessionCapabilities:
    """Merge turn capabilities into session. Update DB row.

    Flags are additive — once True, never reset.
    Returns the updated SessionCapabilities.
    """
    updated = existing.merge(turn_caps)

    await db.execute(
        update(Conversation)
        .where(Conversation.id == conversation_id)
        .values(
            capability_has_images=updated.has_images,
            capability_has_audio=updated.has_audio,
            capability_has_pdf=updated.has_pdf,
            capability_has_video=updated.has_video,
            capability_has_tools=updated.has_tools,
            capability_has_parallel_tools=updated.has_parallel_tools,
            max_tools_level=updated.max_tools_level,
            # Sprint 5
            images_described=updated.images_described,
            images_degraded_manually=updated.images_degraded_manually,
        )
    )

    return updated


def _count_string_content(encoding, content: str) -> int:
    """Count tokens in a plain string content."""
    return len(encoding.encode(content))


def _count_multimodal_content(encoding, content: list) -> int:
    """Count tokens in multimodal content (array of parts)."""
    total = 0
    for part in content:
        text = part.get("text", "")
        if text:
            total += len(encoding.encode(text))
    return total


def _count_tool_arguments(encoding, msg: dict) -> int:
    """Count tokens in tool call arguments of a message."""
    total = 0
    for tc in msg.get("tool_calls") or []:
        args = tc.get("function", {}).get("arguments", "")
        if args:
            total += len(encoding.encode(args))
    return total


def _count_tool_result(encoding, msg: dict) -> int:
    """Count tokens in tool result content."""
    if msg.get("role") != "tool":
        return 0
    result_content = msg.get("content", "")
    if not result_content:
        return 0
    return len(encoding.encode(result_content))


def _tiktoken_count(encoding, messages: list[dict]) -> int:
    """Count tokens using tiktoken encoding."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += _count_string_content(encoding, content)
        elif isinstance(content, list):
            total += _count_multimodal_content(encoding, content)

        total += _count_tool_arguments(encoding, msg)
        total += _count_tool_result(encoding, msg)
        total += 4  # Per-message overhead

    return max(1, total)


def _char_fallback_count(messages: list[dict]) -> int:
    """Fallback: 4 chars = 1 token heuristic."""
    total_chars = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part.get("text"), str):
                    total_chars += len(part["text"])
        for tc in msg.get("tool_calls") or []:
            args = tc.get("function", {}).get("arguments", "")
            total_chars += len(args)
    return max(1, total_chars // 4)


def estimate_tokens(messages: list[dict]) -> int:
    """Count tokens in messages using tiktoken.

    plan-proxy.md §2: Uses tiktoken for deterministic token counting.
    Falls back to 4-char heuristic if tiktoken is unavailable.

    Counts contents of all message parts:
    - text content (string or text parts in arrays)
    - tool call arguments
    - tool result content
    - adds overhead per message (~4 tokens for message framing)
    """
    try:
        encoding = _get_encoding()
        return _tiktoken_count(encoding, messages)
    except Exception:
        return _char_fallback_count(messages)
