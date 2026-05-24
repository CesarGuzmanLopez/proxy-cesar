"""Tool edge cases: streaming partial, mixed content, errors, thinking blocks.

plan-proxy.md §6.7: comprehensive edge case handling for tool calling.
All edge cases are handled deterministically — no ML, no heuristics.
"""

import json

MAX_TOOL_RESULT_TOKENS = 8000


async def accumulate_streaming_tool_calls(
    stream_generator,
) -> tuple[list[dict], bool]:
    """Accumulate tool call deltas from streaming chunks.

    plan-proxy.md §6.7: During SSE streaming, tool call arguments arrive
    in multiple chunks. If the stream is interrupted, the tool call is
    incomplete and marked as such.

    Args:
        stream_generator: Async generator yielding streaming chunks.

    Returns:
        Tuple of (complete_tool_calls, was_incomplete).
        - complete_tool_calls: List of assembled tool call dicts (OpenAI format).
        - was_incomplete: True if any tool call was incomplete or JSON-invalid.
    """
    tool_calls_by_index: dict[int, dict] = {}
    was_incomplete = False

    try:
        async for chunk in stream_generator:
            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            tool_call_deltas = delta.get("tool_calls", [])
            if not tool_call_deltas:
                continue

            for tc_delta in tool_call_deltas:
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

    except Exception:
        was_incomplete = True

    # Assemble final tool calls
    complete: list[dict] = []
    for idx in sorted(tool_calls_by_index.keys()):
        entry = tool_calls_by_index[idx]
        args = "".join(entry["function"]["arguments_parts"])

        # Validate JSON
        if args:
            try:
                json.loads(args)
            except json.JSONDecodeError:
                was_incomplete = True
                continue

        complete.append({
            "id": entry["id"] or f"call_incomplete_{idx}",
            "type": "function",
            "function": {
                "name": entry["function"]["name"],
                "arguments": args,
            },
        })

    if not complete:
        was_incomplete = True

    return complete, was_incomplete


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


def extract_thinking_content(response: dict, provider: str | None = None) -> str | None:
    """Extract thinking/reasoning content from a provider-specific response.

    plan-proxy.md §6.7: Different providers expose thinking content differently.
    The proxy extracts and stores it to preserve cache affinity.

    Args:
        response: The response dict from LiteLLM.
        provider: The provider name (deepseek, anthropic, google, openai).

    Returns:
        The extracted thinking text, or None if not available.
    """
    choices = response.get("choices", [])
    if not choices:
        return None
    message = choices[0].get("message", {})

    # DeepSeek: reasoning_content field
    reasoning = message.get("reasoning_content")
    if reasoning:
        return str(reasoning)

    # Anthropic: content blocks with type="thinking"
    content = message.get("content", "")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "thinking":
                thinking_text = block.get("thinking", "")
                if thinking_text:
                    return str(thinking_text)

    # Google Gemini: content parts with type="thought"
    if isinstance(content, list):
        thoughts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "thought":
                text = part.get("text", "")
                if text:
                    thoughts.append(str(text))
        if thoughts:
            return "\n".join(thoughts)

    # OpenAI o-series: usage.reasoning_tokens (just token count)
    usage = response.get("usage", {})
    reasoning_tokens = usage.get("reasoning_tokens", 0)
    if reasoning_tokens and provider == "openai":
        return (
            f"[OpenAI reasoning tokens: {reasoning_tokens}. "
            f"Raw reasoning content not exposed by API.]"
        )

    return None


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


def is_mixed_content(message: dict) -> bool:
    """Check if an assistant message has both text and tool calls.

    plan-proxy.md §6.7: Text + tool_calls in the same turn is valid.
    Both are stored as separate fields of the same assistant object.

    Args:
        message: An assistant message dict.

    Returns:
        True if the message has both content and tool_calls.
    """
    return bool(
        message.get("content")
        and message.get("role") == "assistant"
        and message.get("tool_calls")
    )
