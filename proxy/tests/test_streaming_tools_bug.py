"""Test for bug: streaming + tools lose response with OpenCode.

Reproduces the issue where streaming responses with tool_calls
disappear or don't reach the client properly.
"""

import json
from unittest.mock import MagicMock
import pytest
from src.api.chat_stream_persistence import _extract_tokens_from_chunks


class MockChunk:
    """Simulate a real LiteLLM streaming chunk (Pydantic object)."""
    def __init__(self, content=None, tool_calls=None, finish_reason=None,
                 prompt_tokens=None, completion_tokens=None):
        self.id = "chatcmpl-test"
        self.created = 1234567890
        self.model = "openai/kimi-k2.5"
        self.object = "chat.completion.chunk"

        self.choices = []
        if content is not None or tool_calls is not None or finish_reason is not None:
            choice = MagicMock()
            delta = MagicMock()

            if content is not None:
                delta.content = content
            else:
                delta.content = None

            if tool_calls is not None:
                delta.tool_calls = tool_calls
            else:
                delta.tool_calls = None

            choice.delta = delta
            choice.finish_reason = finish_reason
            self.choices = [choice]

        self.usage = None
        if prompt_tokens is not None or completion_tokens is not None:
            self.usage = MagicMock()
            self.usage.prompt_tokens = prompt_tokens or 0
            self.usage.completion_tokens = completion_tokens or 0

    def model_dump_json(self):
        """Serialize to JSON (Pydantic-style)."""
        return json.dumps({
            "id": self.id,
            "created": self.created,
            "model": self.model,
            "object": self.object,
            "choices": [
                {
                    "delta": {
                        "content": self.choices[0].delta.content if self.choices and hasattr(self.choices[0].delta, "content") else None,
                        "tool_calls": self._serialize_tool_calls(),
                    } if self.choices else {},
                    "finish_reason": self.choices[0].finish_reason if self.choices else None,
                }
            ] if self.choices else [],
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens if self.usage else None,
                "completion_tokens": self.usage.completion_tokens if self.usage else None,
            } if self.usage else None,
        })

    def _serialize_tool_calls(self):
        if not self.choices or not self.choices[0].delta.tool_calls:
            return None
        return [
            {
                "index": tc.index,
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function.name if hasattr(tc.function, "name") else None,
                    "arguments": tc.function.arguments if hasattr(tc.function, "arguments") else None,
                }
            }
            for tc in self.choices[0].delta.tool_calls
        ]


def create_tool_call_delta(index=0, tc_id="call_1", name="", arguments=""):
    """Create a tool_calls delta for streaming."""
    tc = MagicMock()
    tc.index = index
    tc.id = tc_id
    tc.type = "function"
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


@pytest.mark.asyncio
async def test_streaming_tool_calls_complete_reconstruction():
    """Simulate streaming with tool calls - should reconstruct properly."""
    # Simulate chunks from OpenCode (OpenAI-compatible)
    chunks = [
        # First chunk: partial tool call + text
        MockChunk(
            content="Let me search...",
            tool_calls=[
                create_tool_call_delta(index=0, tc_id="call_search", name="search", arguments='{"q')
            ]
        ),
        # Second chunk: more arguments
        MockChunk(
            tool_calls=[
                create_tool_call_delta(index=0, tc_id=None, name="", arguments='uery": "test"}')
            ]
        ),
        # Final chunk with usage
        MockChunk(
            finish_reason="tool_calls",
            prompt_tokens=100,
            completion_tokens=50
        )
    ]

    # Extract tokens and response
    input_tokens, output_tokens, response_dict = _extract_tokens_from_chunks(chunks)

    # Verify extraction
    assert input_tokens == 100
    assert output_tokens == 50
    assert response_dict["object"] == "chat.completion"

    # Verify tool calls are present
    message = response_dict["choices"][0]["message"]
    assert message["role"] == "assistant"
    assert message["content"] == "Let me search..."

    # BUG: Tool calls should be extracted but may be missing
    assert "tool_calls" in message, "BUG: tool_calls missing from response!"
    assert len(message["tool_calls"]) > 0
    assert message["tool_calls"][0]["id"] == "call_search"
    assert message["tool_calls"][0]["function"]["name"] == "search"


@pytest.mark.asyncio
async def test_streaming_tool_calls_multiple_parallel():
    """Simulate multiple parallel tool calls in streaming."""
    chunks = [
        # First tool
        MockChunk(
            tool_calls=[
                create_tool_call_delta(index=0, tc_id="call_a", name="search", arguments='{}')
            ]
        ),
        # Second tool (parallel)
        MockChunk(
            tool_calls=[
                create_tool_call_delta(index=1, tc_id="call_b", name="fetch", arguments='{}')
            ]
        ),
        # Final
        MockChunk(
            finish_reason="tool_calls",
            prompt_tokens=100,
            completion_tokens=50
        )
    ]

    input_tokens, output_tokens, response_dict = _extract_tokens_from_chunks(chunks)
    message = response_dict["choices"][0]["message"]

    # Should have both tool calls
    assert "tool_calls" in message
    assert len(message["tool_calls"]) == 2
    assert message["tool_calls"][0]["id"] == "call_a"
    assert message["tool_calls"][1]["id"] == "call_b"


@pytest.mark.asyncio
async def test_streaming_empty_chunks():
    """Simulate when chunks have no content."""
    chunks = [
        MockChunk(),  # Empty chunk
        MockChunk(
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=5
        )
    ]

    input_tokens, output_tokens, response_dict = _extract_tokens_from_chunks(chunks)

    assert input_tokens == 10
    assert output_tokens == 5
    assert response_dict["object"] == "chat.completion"
    message = response_dict["choices"][0]["message"]
    assert message["content"] is None
    assert "tool_calls" not in message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
