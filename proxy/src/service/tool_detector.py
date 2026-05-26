"""Tool detector: finds tools compatible with image content.

When the user sends images to a model without vision, the proxy can
delegate image processing to a tool instead of rejecting the request.
Unsupported content types are transformed to text URL references.
"""

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


def _content_type_label(content_type: str) -> str:
    """Return human-readable label for a content type."""
    return _CONTENT_TYPE_LABELS.get(content_type, content_type)


def transform_unsupported_content(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Replace unsupported content types with text explanations.

    For each user message containing content types the model can't process
    (image_url, input_audio, file), extracts the path and replaces the
    content part with a text explanation so the model knows what the
    user sent and can respond appropriately.

    The model receives something like:
      [The user sent an image. path: https://...]
      [The user sent an audio file. path: data:audio/wav;base64,...]
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
            if not isinstance(part, dict):
                new_content.append(part)
                continue

            ptype = part.get("type", "")

            # image_url → extract content path
            if ptype == "image_url":
                url = part.get("image_url", {}).get("url", "")
                label = _content_type_label(ptype)
                new_content.append({
                    "type": "text",
                    "text": (
                        f"[The user sent an {label}. "
                        f"path: {url}]"
                    ),
                })

            # input_audio → extract URL/data
            elif ptype == "input_audio":
                audio_data = part.get("input_audio", {})
                url = audio_data.get("url", "") or audio_data.get("data", "")
                label = _content_type_label(ptype)
                new_content.append({
                    "type": "text",
                    "text": (
                        f"[The user sent an {label}. "
                        f"path: {url}]"
                    ),
                })

            # file (PDF/video) → extract URL
            elif ptype == "file":
                file_data = part.get("file", {})
                url = file_data.get("url", "") or file_data.get("data", "")
                label = _content_type_label(ptype)
                new_content.append({
                    "type": "text",
                    "text": (
                        f"[The user sent a {label}. "
                        f"path: {url}]"
                    ),
                })

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
