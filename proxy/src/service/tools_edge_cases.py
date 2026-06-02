"""Tool edge cases: streaming partial, mixed content, errors, thinking blocks.

plan-proxy.md §6.7: comprehensive edge case handling for tool calling.
All edge cases are handled deterministically — no ML, no heuristics.
"""

import json
import logging

MAX_TOOL_RESULT_TOKENS = 8000

logger = logging.getLogger(__name__)


def _process_tool_delta(
    tc_delta: dict,
    tool_calls_by_index: dict[int, dict],
) -> None:
    """Process a single tool call delta from a streaming chunk."""
    idx = tc_delta.get("index", 0)
    if idx not in tool_calls_by_index:
        tool_calls_by_index[idx] = {
            "id": tc_delta.get("id", ""),
            "type": "function",
            "function": {"name": "", "arguments_parts": []},
        }

    entry = tool_calls_by_index[idx]
    if tc_delta.get("id"):
        entry["id"] = tc_delta["id"]
    if tc_delta.get("function"):
        func = tc_delta["function"]
        if func.get("name"):
            entry["function"]["name"] += func["name"]
        if func.get("arguments"):
            entry["function"]["arguments_parts"].append(func["arguments"])


def _assemble_tool_call(
    idx: int,
    entry: dict,
    was_incomplete: bool,
) -> tuple[dict | None, bool]:
    """Assemble a single tool call from accumulated deltas, validating JSON."""
    args = "".join(entry["function"]["arguments_parts"])

    if args:
        try:
            json.loads(args)
        except json.JSONDecodeError:
            return None, True

    return {
        "id": entry["id"] or f"call_incomplete_{idx}",
        "type": "function",
        "function": {
            "name": entry["function"]["name"],
            "arguments": args,
        },
    }, was_incomplete


def truncate_tool_result(content: str, max_tokens: int = MAX_TOOL_RESULT_TOKENS) -> str:
    """Truncate tool result to max_tokens, adding a truncation marker.

    plan-proxy.md §6.7: Large tool results (>8K tokens) are truncated
    with a clear marker. The full result should be stored in audit log.

    Args:
        content: The raw tool result content.
        max_tokens: Maximum tokens allowed (default 8000).

    Returns:
        Truncated content with marker, or original if within limit.
    """
    max_chars = max_tokens * 4  # ~4 chars per token heuristic
    if len(content) <= max_chars:
        return content
    return (
        content[:max_chars]
        + f"\n\n[...truncated to {max_tokens} tokens. Full result in audit log...]"
    )


def _extract_deepseek_thinking(message: dict) -> str | None:
    """Extract DeepSeek reasoning_content field."""
    reasoning = message.get("reasoning_content")
    return str(reasoning) if reasoning else None


def _extract_anthropic_thinking(content) -> str | None:
    """Extract Anthropic thinking blocks from content."""
    if not isinstance(content, list):
        return None
    for block in content:
        if isinstance(block, dict) and block.get("type") == "thinking":
            thinking_text = block.get("thinking", "")
            if thinking_text:
                return str(thinking_text)
    return None


def _extract_gemini_thinking(content) -> str | None:
    """Extract Google Gemini thought parts from content."""
    if not isinstance(content, list):
        return None
    thoughts: list[str] = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "thought":
            text = part.get("text", "")
            if text:
                thoughts.append(str(text))
    return "\n".join(thoughts) if thoughts else None


def _extract_openai_thinking(response: dict, provider: str | None) -> str | None:
    """Extract OpenAI o-series reasoning token info."""
    if provider != "openai":
        return None
    usage = response.get("usage", {})
    reasoning_tokens = usage.get("reasoning_tokens", 0)
    if reasoning_tokens:
        return (
            f"[OpenAI reasoning tokens: {reasoning_tokens}. "
            f"Raw reasoning content not exposed by API.]"
        )
    return None


def extract_thinking_content(response: dict, provider: str | None = None) -> str | None:
    """Extract thinking/reasoning content from a provider-specific response.

    plan-proxy.md §6.7: Different providers expose thinking content differently.
    The proxy extracts and stores it to preserve cache affinity.
    """
    choices = response.get("choices", [])
    if not choices:
        return None
    message = choices[0].get("message", {})

    # DeepSeek
    result = _extract_deepseek_thinking(message)
    if result is not None:
        return result

    content = message.get("content", "")

    # Anthropic
    result = _extract_anthropic_thinking(content)
    if result is not None:
        return result

    # Gemini
    result = _extract_gemini_thinking(content)
    if result is not None:
        return result

    # OpenAI
    return _extract_openai_thinking(response, provider)


def enforce_tool_choice(response: dict, tool_choice: str | None) -> bool:
    """Check if the model respected tool_choice.

    plan-proxy.md §6.7: When tool_choice is "required" but the model
    responds without tool calls, the proxy must force fallback.

    Args:
        response: The response dict from LiteLLM.
        tool_choice: The tool_choice value from the request.

    Returns:
        True if the model respected tool_choice, False if it ignored it.
    """
    if tool_choice != "required":
        return True  # No enforcement needed

    choices = response.get("choices", [])
    if not choices:
        return False
    message = choices[0].get("message", {})
    tool_calls = message.get("tool_calls", [])

    return len(tool_calls) > 0
