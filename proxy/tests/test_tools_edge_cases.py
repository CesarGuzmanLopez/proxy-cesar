"""Tests for tool edge cases (tools_edge_cases.py).

"""

import json

import pytest

from src.service.tools_edge_cases import (
    accumulate_streaming_tool_calls,
    enforce_tool_choice,
    extract_thinking_content,
    is_mixed_content,
    truncate_tool_result,
)


# ---------------------------------------------------------------------------
# Streaming partial tool calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_complete_tool_call():
    """Streaming: complete tool call → stored correctly (not incomplete)."""

    async def _stream():
        yield {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {"name": "search", "arguments": '{"que'},
                            }
                        ]
                    }
                }
            ]
        }
        yield {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": 'ry": "test"}'}}
                        ]
                    }
                }
            ]
        }

    calls, incomplete = await accumulate_streaming_tool_calls(_stream())
    assert incomplete is False
    assert len(calls) == 1
    assert calls[0]["id"] == "call_1"
    assert calls[0]["function"]["name"] == "search"
    assert json.loads(calls[0]["function"]["arguments"]) == {"query": "test"}


@pytest.mark.asyncio
async def test_streaming_partial_tool_call_incomplete():
    """Streaming: partial tool call → marked incomplete."""

    async def _stream():
        yield {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {
                                    "name": "search",
                                    "arguments": '{"incomplete',
                                },
                            }
                        ]
                    }
                }
            ]
        }

    calls, incomplete = await accumulate_streaming_tool_calls(_stream())
    assert incomplete is True
    assert len(calls) == 0  # JSON invalid → discarded


@pytest.mark.asyncio
async def test_streaming_exception_marked_incomplete():
    """Streaming: exception during iteration → marked incomplete."""

    async def _broken_stream():
        yield {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {"name": "search", "arguments": "{}"},
                            }
                        ]
                    }
                }
            ]
        }
        raise RuntimeError("Connection lost")

    calls, incomplete = await accumulate_streaming_tool_calls(_broken_stream())
    assert incomplete is True


@pytest.mark.asyncio
async def test_streaming_no_tool_calls():
    """Streaming: no tool calls in stream → empty, not incomplete."""

    async def _stream():
        yield {"choices": [{"delta": {"content": "Hello"}}]}

    calls, incomplete = await accumulate_streaming_tool_calls(_stream())
    assert calls == []
    assert incomplete is True  # No complete calls → incomplete


@pytest.mark.asyncio
async def test_streaming_multiple_parallel():
    """Streaming: multiple parallel tool calls accumulated."""

    async def _stream():
        yield {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_a",
                                "function": {"name": "search", "arguments": "{}"},
                            }
                        ]
                    }
                }
            ]
        }
        yield {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 1,
                                "id": "call_b",
                                "function": {"name": "read", "arguments": "{}"},
                            }
                        ]
                    }
                }
            ]
        }

    calls, incomplete = await accumulate_streaming_tool_calls(_stream())
    assert len(calls) == 2
    assert calls[0]["id"] == "call_a"
    assert calls[1]["id"] == "call_b"


# ---------------------------------------------------------------------------
# Mixed content (text + tool calls)
# ---------------------------------------------------------------------------


def test_mixed_content_detected():
    """Message with text + tool_calls → mixed content."""
    msg = {
        "role": "assistant",
        "content": "Let me search.",
        "tool_calls": [{"id": "call_1", "function": {"arguments": "{}"}}],
    }
    assert is_mixed_content(msg) is True


def test_mixed_content_text_only():
    """Message with text only → not mixed."""
    msg = {"role": "assistant", "content": "Hello"}
    assert is_mixed_content(msg) is False


def test_mixed_content_tool_only():
    """Message with tool_calls only → not mixed."""
    msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "call_1", "function": {"arguments": "{}"}}],
    }
    assert is_mixed_content(msg) is False


# ---------------------------------------------------------------------------
# Tool error result
# ---------------------------------------------------------------------------


def test_truncate_tool_result_normal():
    """Normal-sized tool result → not truncated."""
    content = "x" * 100
    assert truncate_tool_result(content) == content


def test_truncate_tool_result_large():
    """Large tool result → truncated with marker."""
    content = "x" * 100000  # 100K chars → ~25K tokens → way over 8K
    truncated = truncate_tool_result(content)
    assert len(truncated) < len(content)
    assert "truncated" in truncated
    assert "audit log" in truncated


def test_truncate_tool_result_boundary():
    """Tool result at exact boundary → not truncated."""
    max_chars = 8000 * 4  # 32000 chars
    content = "x" * max_chars
    assert truncate_tool_result(content) == content


# ---------------------------------------------------------------------------
# Thinking blocks extraction
# ---------------------------------------------------------------------------


def test_thinking_deepseek():
    """Extract reasoning_content from DeepSeek response."""
    response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Final answer",
                    "reasoning_content": "Step-by-step reasoning...",
                }
            }
        ]
    }
    thinking = extract_thinking_content(response, "deepseek")
    assert thinking == "Step-by-step reasoning..."


def test_thinking_anthropic():
    """Extract thinking block from Claude response."""
    response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "Claude is thinking..."},
                        {"type": "text", "text": "Final answer"},
                    ],
                }
            }
        ]
    }
    thinking = extract_thinking_content(response, "anthropic")
    assert thinking == "Claude is thinking..."


def test_thinking_gemini():
    """Extract thought parts from Gemini response."""
    response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "thought", "text": "Step 1: analyze"},
                        {"type": "thought", "text": "Step 2: solve"},
                        {"type": "text", "text": "Final answer"},
                    ],
                }
            }
        ]
    }
    thinking = extract_thinking_content(response, "google")
    assert thinking == "Step 1: analyze\nStep 2: solve"


def test_thinking_openai():
    """Extract reasoning_tokens from OpenAI o-series response."""
    response = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "Answer"},
            }
        ],
        "usage": {"reasoning_tokens": 500},
    }
    thinking = extract_thinking_content(response, "openai")
    assert thinking is not None
    assert "500" in thinking
    assert "reasoning tokens" in thinking.lower()


def test_thinking_not_present():
    """No thinking content → None."""
    response = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "Simple answer"},
            }
        ]
    }
    assert extract_thinking_content(response, "deepseek") is None
    assert extract_thinking_content(response, "anthropic") is None


# ---------------------------------------------------------------------------
# enforce_tool_choice
# ---------------------------------------------------------------------------


def test_tool_choice_required_respected():
    """tool_choice='required' with tool calls → OK."""
    response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "tool_calls": [{"id": "call_1", "function": {"arguments": "{}"}}],
                }
            }
        ]
    }
    assert enforce_tool_choice(response, "required") is True


def test_tool_choice_required_ignored():
    """tool_choice='required' without tool calls → False."""
    response = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "I refuse to call tools"},
            }
        ]
    }
    assert enforce_tool_choice(response, "required") is False


def test_tool_choice_not_required():
    """tool_choice is not 'required' → always True."""
    response = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "OK"},
            }
        ]
    }
    assert enforce_tool_choice(response, None) is True
    assert enforce_tool_choice(response, "auto") is True
    assert enforce_tool_choice(response, "any") is True


def test_tool_choice_required_empty_choices():
    """Empty choices with required → False."""
    assert enforce_tool_choice({}, "required") is False
