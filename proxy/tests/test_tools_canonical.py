"""Tests for canonical tool format storage (tools_canonical.py).

Sprint 3 §6.1 — minimum 10 tests.
"""

from src.service.tools_canonical import (
    determine_tools_level,
    determine_tool_level_for_turn,
    extract_tool_calls_from_response,
    validate_arguments_json,
    validate_tool_call_ids,
)


# ---------------------------------------------------------------------------
# determine_tools_level
# ---------------------------------------------------------------------------

def test_no_tool_calls_level_zero():
    """No tool calls → tools_level_used: 0."""
    assert determine_tools_level(None) == 0
    assert determine_tools_level([]) == 0


def test_single_basic_tool_call_level_basic():
    """Single tool call with simple args → BASIC (1)."""
    calls = [{"function": {"arguments": '{"query": "hello"}'}}]
    assert determine_tools_level(calls) == 1


def test_single_tool_call_with_multiple_args_standard():
    """Single tool call with multiple args → STANDARD (2)."""
    calls = [{"function": {"arguments": '{"query": "hello", "limit": 10}'}}]
    assert determine_tools_level(calls) == 2


def test_parallel_calls_level_parallel():
    """Multiple tool calls in one turn → PARALLEL_STRICT (3)."""
    calls = [
        {"function": {"arguments": "{}"}},
        {"function": {"arguments": "{}"}},
    ]
    assert determine_tools_level(calls) == 3


def test_strict_mode_tool_level_parallel():
    """Tool definition with strict: true → PARALLEL_STRICT (3)."""
    calls = [{"function": {"arguments": "{}"}}]
    definitions = [{"function": {"name": "search", "strict": True}}]
    assert determine_tools_level(calls, definitions) == 3


# ---------------------------------------------------------------------------
# determine_tool_level_for_turn
# ---------------------------------------------------------------------------

def test_turn_incomplete_caps_at_basic():
    """Tools incomplete with calls → capped at BASIC (1)."""
    calls = [{"function": {"arguments": '{"a":1, "b":2}'}}]
    assert determine_tool_level_for_turn(calls, tools_incomplete=True) == 1


def test_turn_incomplete_no_calls_none():
    """Tools incomplete with no calls → NONE (0)."""
    assert determine_tool_level_for_turn([], tools_incomplete=True) == 0


# ---------------------------------------------------------------------------
# validate_tool_call_ids
# ---------------------------------------------------------------------------

def test_valid_tool_call_ids_pass():
    """Valid tool calls with unique IDs → passes."""
    calls = [
        {"id": "call_1", "function": {"arguments": "{}"}},
        {"id": "call_2", "function": {"arguments": "{}"}},
    ]
    validate_tool_call_ids(calls)  # should not raise


def test_missing_id_raises():
    """Tool call without 'id' field → raises ValueError."""
    calls = [{"function": {"arguments": "{}"}}]
    try:
        validate_tool_call_ids(calls)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "missing 'id'" in str(e)


def test_duplicate_id_raises():
    """Duplicate tool_call_id → raises ValueError."""
    calls = [
        {"id": "call_same", "function": {"arguments": "{}"}},
        {"id": "call_same", "function": {"arguments": "{}"}},
    ]
    try:
        validate_tool_call_ids(calls)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "Duplicate" in str(e)


def test_empty_arguments_raises():
    """Empty arguments string → raises ValueError."""
    calls = [{"id": "call_1", "function": {"arguments": ""}}]
    try:
        validate_tool_call_ids(calls)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "empty" in str(e).lower()


# ---------------------------------------------------------------------------
# validate_arguments_json
# ---------------------------------------------------------------------------

def test_validate_valid_json():
    """Valid JSON arguments string → True."""
    assert validate_arguments_json('{"key": "value"}') is True


def test_validate_invalid_json():
    """Invalid JSON arguments string → False."""
    assert validate_arguments_json("{invalid}") is False


def test_validate_empty_json():
    """Empty arguments string → False."""
    assert validate_arguments_json("") is False


# ---------------------------------------------------------------------------
# extract_tool_calls_from_response
# ---------------------------------------------------------------------------

def test_extract_tool_calls_from_response():
    """Extract tool calls from a standard LiteLLM response."""
    response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Searching...",
                    "tool_calls": [
                        {"id": "call_1", "type": "function", "function": {"name": "search", "arguments": "{}"}},
                    ],
                }
            }
        ]
    }
    calls = extract_tool_calls_from_response(response)
    assert len(calls) == 1
    assert calls[0]["id"] == "call_1"


def test_extract_no_tool_calls():
    """Response without tool calls → empty list."""
    response = {"choices": [{"message": {"role": "assistant", "content": "Hello"}}]}
    assert extract_tool_calls_from_response(response) == []


def test_extract_empty_response():
    """Empty response → empty list."""
    assert extract_tool_calls_from_response({}) == []
