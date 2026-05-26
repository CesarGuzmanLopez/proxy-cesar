"""Tool detector: finds tools compatible with image content.

When the user sends images to a model without vision, the proxy can
delegate image processing to a tool instead of rejecting the request.
"""

from typing import Any


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
                new_content.append(
                    {
                        "type": "text",
                        "text": (
                            f"[Image URL delegated to tool '{tool_name}' "
                            f"as parameter '{param_name}']: {url}"
                        ),
                    }
                )
            else:
                new_content.append(part)

        new_messages.append({**msg, "content": new_content})

    return new_messages
