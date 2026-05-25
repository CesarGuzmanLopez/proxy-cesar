"""Canonical tool format storage for conversation turns.

plan-proxy.md §6.5: The proxy ALWAYS stores and retrieves tools history
in OpenAI format, regardless of provider. LiteLLM handles translations.

python.md §4: pure functions, immutable data where possible.
python.md §3: Result monad for error handling.
"""

import json
from dataclasses import dataclass

from src.domain.tools import ToolLevel


def determine_tools_level(
    tool_calls: list[dict] | None,
    tool_definitions: list[dict] | None = None,
) -> int:
    """Determine the tool complexity level from tool calls and definitions.

    plan-proxy.md §6.3: models have different tool capability levels.

    Args:
        tool_calls: List of tool calls from the assistant response.
        tool_definitions: Optional list of tool definitions from the request.

    Returns:
        ToolLevel as int (0=NONE, 1=BASIC, 2=STANDARD, 3=PARALLEL_STRICT).
    """
    if not tool_calls:
        return ToolLevel.NONE

    # Multiple tool calls → PARALLEL_STRICT
    if len(tool_calls) > 1:
        return ToolLevel.PARALLEL_STRICT

    # Check for strict mode in tool definitions
    if tool_definitions:
        for td in tool_definitions:
            func = td.get("function", {})
            if func.get("strict"):
                return ToolLevel.PARALLEL_STRICT

    # Single tool call: check schema complexity
    if len(tool_calls) == 1:
        tc = tool_calls[0]
        args_str = tc.get("function", {}).get("arguments", "{}")
        try:
            args = json.loads(args_str)
            if args and len(args) > 1:
                return ToolLevel.STANDARD
        except (json.JSONDecodeError, TypeError):
            pass
        return ToolLevel.BASIC

    return ToolLevel.NONE


def validate_tool_call_ids(tool_calls: list[dict]) -> None:
    """Validate that each tool call has a non-empty, unique ID.

    plan-proxy.md §6.5: tool_call_id is used EXACTLY as returned by the model.
    No prefix, no suffix, no modification — but it MUST exist and be unique.

    Args:
        tool_calls: List of tool call dicts from the response.

    Raises:
        ValueError: If any tool call is missing an ID or IDs are not unique.
    """
    seen_ids: set[str] = set()
    for i, tc in enumerate(tool_calls):
        tc_id = tc.get("id", "")
        if not tc_id:
            raise ValueError(f"Tool call at index {i} is missing 'id' field")
        if tc_id in seen_ids:
            raise ValueError(
                f"Duplicate tool_call_id '{tc_id}' at index {i}. "
                f"IDs must be unique within a turn."
            )
        seen_ids.add(tc_id)

        args = tc.get("function", {}).get("arguments", "")
        if not args:
            raise ValueError(f"Tool call '{tc_id}' has empty 'arguments' field")


def validate_arguments_json(arguments: str) -> bool:
    """Verify that the arguments string is valid JSON.

    plan-proxy.md §6.5: arguments is stored as a JSON string, not parsed.
    This function only validates — it does NOT modify the string.

    Args:
        arguments: The JSON string of tool call arguments.

    Returns:
        True if valid JSON, False otherwise.
    """
    if not arguments:
        return False
    try:
        json.loads(arguments)
        return True
    except ValueError:
        return False


def extract_tool_calls_from_response(response: dict) -> list[dict]:
    """Extract tool calls from a LiteLLM response dict.

    plan-proxy.md §6.5: LiteLLM normalizes responses back to OpenAI format.
    The proxy extracts tool_calls from the normalized response.

    Args:
        response: The response dict from LiteLLM (already in OpenAI format).

    Returns:
        List of tool call dicts, or empty list if none.
    """
    choices = response.get("choices", [])
    if not choices:
        return []
    message = choices[0].get("message", {})
    return message.get("tool_calls", [])


def determine_tool_level_for_turn(
    tool_calls: list[dict],
    tool_definitions: list[dict] | None = None,
    tools_incomplete: bool = False,
) -> int:
    """Determine the tool level for a turn, accounting for incompleteness.

    If tools were incomplete, the level is capped at BASIC since the
    full tool call was not received.

    Args:
        tool_calls: Tool calls from this turn.
        tool_definitions: Tool definitions from the request.
        tools_incomplete: Whether the tool call was interrupted mid-stream.

    Returns:
        ToolLevel as int.
    """
    if tools_incomplete or not tool_calls:
        return ToolLevel.NONE if not tool_calls else ToolLevel.BASIC
    return determine_tools_level(tool_calls, tool_definitions)


@dataclass
class TurnToolMetadata:
    """Metadata about tool usage in a turn, ready for DB storage.

    plan-proxy.md §6.5: all tool history stored in canonical OpenAI format.
    """

    tool_definitions: list[dict] | None = None
    thinking_blocks: dict | None = None
    tools_incomplete: bool = False
    tools_level_used: int = 0
