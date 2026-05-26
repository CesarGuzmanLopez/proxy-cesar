"""Tool and content detection utilities.

- Finds tools compatible with image content (delegation)
- Transforms unsupported content to text with blob references
- Stores base64 blobs in Valkey for tool retrieval
"""

import hashlib
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

BLOB_PREFIX = "BLOB"
BLOB_TTL = 7200  # 2 hours
_MAX_BLOB_SIZE = 10 * 1024 * 1024  # 10 MB


def _hash_content(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def _extract_mime(data_uri: str) -> str | None:
    """Extract MIME from data URI: 'data:image/png;base64,...' → 'image/png'."""
    match = re.match(r"data:([a-z]+/[a-z0-9+-.]+)", data_uri)
    return match.group(1) if match else None


def find_image_compatible_tool(tools: list[dict] | None) -> tuple[str, str] | None:
    """Find first tool with a string parameter that could accept an image path.

    Returns (tool_name, param_name) or None if no compatible tool found.
    """
    if not tools:
        return None
    for tool in tools:
        func = tool.get("function", {}) if isinstance(tool, dict) else {}
        name = func.get("name", "")
        params = func.get("parameters", {}).get("properties", {})
        if isinstance(params, dict):
            for param_name, param_schema in params.items():
                if isinstance(param_schema, dict) and param_schema.get("type") == "string":
                    return (name, param_name)
    return None


async def _store_blob(valkey, key: str, raw_data: str) -> None:
    """Store base64 blob in Valkey if within size limits."""
    if len(raw_data) > _MAX_BLOB_SIZE:
        logger.warning("blob_too_large key=%s size=%d", key, len(raw_data))
        return
    try:
        await valkey.set(key, raw_data, ex=BLOB_TTL)
    except Exception:
        pass


async def replace_base64_with_blob_refs(
    messages: list[dict[str, Any]],
    conversation_id: str | None = None,
    valkey=None,
) -> list[dict[str, Any]]:
    """Replace base64 content parts with [BLOB:hash:mime] references.

    Stores the actual base64 data in Valkey so tools can retrieve it later.
    Real URLs (non-base64) pass through unchanged.
    """
    if valkey is None:
        return messages

    prefix = f"blob:{conversation_id or 'anon'}"
    new_messages: list[dict[str, Any]] = []

    for msg in messages:
        if msg.get("role") != "user":
            new_messages.append(msg)
            continue

        content = msg.get("content", "")
        if not isinstance(content, list):
            new_messages.append(msg)
            continue

        new_content: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict):
                new_content.append(part)
                continue

            ptype = part.get("type", "")

            if ptype == "image_url":
                raw = part.get("image_url", {}).get("url", "")
                if raw.startswith("data:"):
                    h = _hash_content(raw)
                    mime = _extract_mime(raw) or "image/unknown"
                    await _store_blob(valkey, f"{prefix}:{h}", raw)
                    new_content.append({
                        "type": "text",
                        "text": f"[The user sent an image. blob: {BLOB_PREFIX}:{h}:{mime}]",
                    })
                else:
                    new_content.append(part)

            elif ptype == "input_audio":
                audio = part.get("input_audio", {})
                raw = audio.get("data", "")
                if raw.startswith("data:"):
                    h = _hash_content(raw)
                    mime = _extract_mime(raw) or "audio/unknown"
                    await _store_blob(valkey, f"{prefix}:{h}", raw)
                    new_content.append({
                        "type": "text",
                        "text": f"[The user sent an audio file. blob: {BLOB_PREFIX}:{h}:{mime}]",
                    })
                else:
                    new_content.append(part)

            elif ptype == "file":
                file_data = part.get("file", {})
                raw = file_data.get("data", "")
                if raw.startswith("data:"):
                    h = _hash_content(raw)
                    mime = _extract_mime(raw) or "application/octet-stream"
                    await _store_blob(valkey, f"{prefix}:{h}", raw)
                    new_content.append({
                        "type": "text",
                        "text": f"[The user sent a file. blob: {BLOB_PREFIX}:{h}:{mime}]",
                    })
                else:
                    new_content.append(part)

            else:
                new_content.append(part)

        new_messages.append({**msg, "content": new_content})

    return new_messages


def delegate_images_to_tool(
    messages: list[dict[str, Any]],
    tool_name: str,
    param_name: str,
) -> list[dict[str, Any]]:
    """Replace image_url content parts with text instructions for tool use."""
    new_messages: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") != "user":
            new_messages.append(msg)
            continue
        content = msg.get("content", "")
        if not isinstance(content, list):
            new_messages.append(msg)
            continue
        has_image = any(
            isinstance(p, dict) and p.get("type") == "image_url" for p in content
        )
        if not has_image:
            new_messages.append(msg)
            continue
        new_content: list[dict[str, Any]] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                new_content.append({
                    "type": "text",
                    "text": (
                        f"[Image path delegated to tool '{tool_name}' "
                        f"as parameter '{param_name}']: {url}"
                    ),
                })
            else:
                new_content.append(part)
        new_messages.append({**msg, "content": new_content})
    return new_messages
