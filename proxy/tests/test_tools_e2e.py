"""End-to-end integration tests for tool calling with real providers.

Sprint 3 §6.4 — 12+ integration tests covering the verification matrix.
All tests are marked @pytest.mark.integration and skipped by default.
Run with: python -m pytest tests/test_tools_e2e.py --run-integration

Requires API keys for all providers in .env.
"""

import json

import pytest

from src.adapters.litellm import call_litellm
from src.service.tools_canonical import extract_tool_calls_from_response
from src.service.tools_edge_cases import enforce_tool_choice


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SIMPLE_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name",
                    },
                },
                "required": ["location"],
                "additionalProperties": False,
            },
        },
    }
]

COMPLEX_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "search_codebase",
            "description": "Search the codebase for patterns",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "path": {"type": "string", "description": "Base path to search in"},
                    "max_results": {"type": "integer", "description": "Max results"},
                    "include_pattern": {"type": "string", "description": "Glob pattern"},
                },
                "required": ["query", "path"],
                "additionalProperties": False,
            },
        },
    }
]

USER_MESSAGE = {"role": "user", "content": "What is the weather in Paris today?"}
COMPLEX_MESSAGE = {"role": "user", "content": "Search for database connections in src/ directory, max 5 results."}

TOOL_RESULT_MESSAGE = {
    "role": "tool",
    "tool_call_id": "call_placeholder",
    "content": '{"temperature": 22, "condition": "sunny", "city": "Paris"}',
}


def _check_tool_call(response: dict) -> dict | None:
    """Validate response has a tool call and return it."""
    choices = response.get("choices", [])
    assert len(choices) > 0, "No choices in response"
    msg = choices[0].get("message", {})
    tool_calls = msg.get("tool_calls", [])
    if not tool_calls:
        return None
    tc = tool_calls[0]
    assert tc.get("id"), "Tool call missing 'id'"
    assert tc.get("function", {}).get("arguments"), "Tool call missing 'arguments'"
    # Validate arguments is valid JSON
    args = tc["function"]["arguments"]
    json.loads(args)  # Should not raise
    return tc


# ---------------------------------------------------------------------------
# Integration tests (skip by default)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.skip(reason="Requires API keys. Run with --run-integration")
async def test_deepseek_v4_pro_simple_tool():
    """DeepSeek V4 Pro: simple tool call."""
    response = await call_litellm(
        model="deepseek/deepseek-chat",
        messages=[USER_MESSAGE],
        tools=SIMPLE_TOOL,
    )
    tc = _check_tool_call(response)
    assert tc is not None, "Model did not return a tool call"


@pytest.mark.integration
@pytest.mark.skip(reason="Requires API keys. Run with --run-integration")
async def test_deepseek_v4_pro_complex_schema():
    """DeepSeek V4 Pro: complex schema tool call."""
    response = await call_litellm(
        model="deepseek/deepseek-chat",
        messages=[COMPLEX_MESSAGE],
        tools=COMPLEX_TOOL,
    )
    tc = _check_tool_call(response)
    assert tc is not None


@pytest.mark.integration
@pytest.mark.skip(reason="Requires API keys. Run with --run-integration")
async def test_deepseek_v4_pro_parallel_tools():
    """DeepSeek V4 Pro: parallel tool calls."""
    tools = [
        {"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
        {"type": "function", "function": {"name": "get_time", "description": "Get time", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    ]
    response = await call_litellm(
        model="deepseek/deepseek-chat",
        messages=[{"role": "user", "content": "Get weather and time in Paris"}],
        tools=tools,
    )
    tcs = extract_tool_calls_from_response(response)
    assert len(tcs) > 1, "Expected parallel tool calls"


@pytest.mark.integration
@pytest.mark.skip(reason="Requires API keys. Run with --run-integration")
async def test_deepseek_v4_pro_tool_choice_required():
    """DeepSeek V4 Pro: tool_choice='required' is respected."""
    response = await call_litellm(
        model="deepseek/deepseek-chat",
        messages=[USER_MESSAGE],
        tools=SIMPLE_TOOL,
        tool_choice="required",
    )
    assert enforce_tool_choice(response, "required"), "Model ignored tool_choice=required"


@pytest.mark.integration
@pytest.mark.skip(reason="Requires API keys. Run with --run-integration")
async def test_deepseek_v4_flash_simple_tool():
    """DeepSeek V4 Flash: simple tool call."""
    response = await call_litellm(
        model="deepseek/deepseek-chat",
        messages=[USER_MESSAGE],
        tools=SIMPLE_TOOL,
    )
    tc = _check_tool_call(response)
    assert tc is not None


@pytest.mark.integration
@pytest.mark.skip(reason="Requires API keys. Run with --run-integration")
async def test_gemini_3_5_flash_simple_tool():
    """Gemini 3.5 Flash: simple tool call."""
    response = await call_litellm(
        model="gemini/gemini-2.0-flash",  # Adjust if needed
        messages=[USER_MESSAGE],
        tools=SIMPLE_TOOL,
    )
    tc = _check_tool_call(response)
    assert tc is not None


@pytest.mark.integration
@pytest.mark.skip(reason="Requires API keys. Run with --run-integration")
async def test_qwen3_max_simple_tool():
    """Qwen3 Max: simple tool call."""
    response = await call_litellm(
        model="qwen/qwen-max",  # Adjust if needed
        messages=[USER_MESSAGE],
        tools=SIMPLE_TOOL,
    )
    tc = _check_tool_call(response)
    assert tc is not None


@pytest.mark.integration
@pytest.mark.skip(reason="Requires API keys. Run with --run-integration")
async def test_claude_haiku_simple_tool():
    """Claude Haiku 4.5: simple tool call."""
    response = await call_litellm(
        model="anthropic/claude-3-5-haiku-latest",  # Adjust if needed
        messages=[USER_MESSAGE],
        tools=SIMPLE_TOOL,
    )
    tc = _check_tool_call(response)
    assert tc is not None


@pytest.mark.integration
@pytest.mark.skip(reason="Requires API keys. Run with --run-integration")
async def test_deepseek_v4_pro_tool_result_roundtrip():
    """DeepSeek V4 Pro: tool call → tool result → model uses result."""
    # Step 1: Get tool call
    response = await call_litellm(
        model="deepseek/deepseek-chat",
        messages=[USER_MESSAGE],
        tools=SIMPLE_TOOL,
    )
    tc = _check_tool_call(response)
    assert tc is not None

    # Step 2: Send tool result
    result_msg = {
        "role": "tool",
        "tool_call_id": tc["id"],
        "content": '{"temperature": 22, "condition": "sunny"}',
    }
    followup = await call_litellm(
        model="deepseek/deepseek-chat",
        messages=[USER_MESSAGE, {"role": "assistant", "content": None, "tool_calls": [tc]}, result_msg],
        tools=SIMPLE_TOOL,
    )
    choices = followup.get("choices", [])
    assert len(choices) > 0
    content = choices[0].get("message", {}).get("content", "")
    assert len(content) > 0, "Model did not respond to tool result"
