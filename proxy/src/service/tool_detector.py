"""Tool detector: finds tools compatible with image content.

When the user sends images to a model without vision, the proxy can
delegate image processing to a tool instead of rejecting the request.
Unsupported content types are transformed to text URL references.
"""

import re
from typing import Any

_CONTENT_TYPE_LABELS: dict[str, str] = {
    "image_url": "image",
    "input_audio": "audio file",
    "file": "file (PDF/video)",
}


def find_image_compatible_tool(tools: list[dict] | None) -> tuple[str, str] | None:
    """Find first tool with a string parameter that could accept an image URL.

    Scans tool function schemas for a parameter of type "string".
    Returns (tool_name, param_name) or None if no compatible tool found.

    The heuristic is deliberately permissive: any string parameter could
    potentially accept an image URL. Future iterations may check for
    format: "uri" or "binary" annotations in the JSON Schema.
    """
    if not tools:
        return None

    for tool in tools:
        func = tool.get("function", {}) if isinstance(tool, dict) else {}
        name = func.get("name", "")
        params = func.get("parameters", {}).get("properties", {})
        if isinstance(params, dict):
            for param_name, param_schema in params.items():
                if (
                    isinstance(param_schema, dict)
                    and param_schema.get("type") == "string"
                ):
                    return (name, param_name)

    return None


def _extract_mime(raw_path: str) -> str | None:
    """Extract MIME type from a data URI prefix.

    e.g. 'data:image/png;base64,...' → 'image/png'
    """
    match = re.match(r"data:([a-z]+/[a-z0-9+-.]+)", raw_path)
    return match.group(1) if match else None


def _format_content_part(part: dict) -> str:
    """Format a content part dict into a text explanation for the model.

    If the content is a real URL, include the path.
    If it's base64 data, extract and include the MIME type so the model
    knows what kind of content the user provided.
    """
    ptype = part.get("type", "")
    label = _CONTENT_TYPE_LABELS.get(ptype, ptype)

    # Extract raw path/data
    if ptype == "image_url":
        img_data = part.get("image_url", {})
        raw_path = img_data.get("url", "")
        detail = img_data.get("detail", "")
    elif ptype == "input_audio":
        audio_data = part.get("input_audio", {})
        raw_path = audio_data.get("url", "") or audio_data.get("data", "")
        audio_format = audio_data.get("format", "")
    elif ptype == "file":
        file_data = part.get("file", {})
        raw_path = file_data.get("url", "") or file_data.get("data", "")
        filename = file_data.get("filename", "")
        mime = file_data.get("mime_type", "")
    else:
        return ""

    if not raw_path:
        return f"[The user sent an {label}. (no path provided)]"

    # Real URL → include the path
    if not raw_path.startswith("data:"):
        return f"[The user sent an {label}. path: {raw_path}]"

    # Base64 inline data → extract MIME and metadata
    mime_type = _extract_mime(raw_path) or "unknown"
    meta_parts = [f"type: {mime_type}"]

    if ptype == "image_url" and detail and detail != "auto":
        meta_parts.append(f"detail: {detail}")
    if ptype == "input_audio" and audio_format:
        meta_parts.append(f"format: {audio_format}")
    if ptype == "file":
        if filename:
            meta_parts.append(f"filename: {filename}")
        if mime:
            meta_parts.append(f"mime: {mime}")

    meta_str = ", ".join(meta_parts)
    return f"[The user sent an {label} as base64. {meta_str}]"


def transform_unsupported_content(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Replace unsupported content types with text explanations.

    For each user message containing content types the model can't process
    (image_url, input_audio, file), replaces the content part with a text
    explanation so the model knows what the user sent.

    Real URLs are passed as-is. Base64 inline data is described by its
    MIME type and metadata (format, dimensions, filename, etc.) instead
    of the raw blob.

    Examples:
      [The user sent an image. path: https://example.com/img.png]
      [The user sent an image as base64. type: image/png]
      [The user sent an audio file as base64. type: audio/wav, format: wav]
    """
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
            if isinstance(part, dict) and part.get("type") in (
                "image_url",
                "input_audio",
                "file",
            ):
                new_content.append({"type": "text", "text": _format_content_part(part)})
            else:
                new_content.append(part)

        new_messages.append({**msg, "content": new_content})

    return new_messages


def delegate_images_to_tool(
    messages: list[dict[str, Any]],
    tool_name: str,
    param_name: str,
) -> list[dict[str, Any]]:
    """Replace image_url content parts with text instructions for tool use.

    For each user message containing image_url parts:
    1. Extracts the image URL
    2. Replaces with a text part containing the URL + instruction
    3. Returns modified messages

    Non-image messages and non-image content parts are untouched.
    """
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
            isinstance(part, dict) and part.get("type") == "image_url"
            for part in content
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
                        f"[Image URL delegated to tool '{tool_name}' "
                        f"as parameter '{param_name}']: {url}"
                    ),
                })
            else:
                new_content.append(part)

        new_messages.append({**msg, "content": new_content})

    return new_messages
