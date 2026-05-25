"""Pydantic v2 schemas for tool definitions, calls, and results.

plan-proxy.md §6.5: canonical OpenAI format.
python.md §6.3: Pydantic v2 with extra="forbid".
All schemas validate deterministic, unambiguous tool data.
"""

from pydantic import BaseModel, Field
from typing import Literal


class ToolFunctionSchema(BaseModel, extra="forbid"):
    """A single tool function definition (OpenAI format)."""

    name: str = Field(min_length=1, max_length=256)
    description: str = Field(default="", max_length=4096)
    parameters: dict = Field(default_factory=dict)
    strict: bool | None = None


class ToolDefinitionSchema(BaseModel, extra="forbid"):
    """Wrapper for a tool definition (OpenAI format)."""

    type: Literal["function"] = "function"
    function: ToolFunctionSchema


class ToolCallFunctionSchema(BaseModel, extra="forbid"):
    """The function part of a tool call.

    `arguments` is a JSON string — stored exactly as the model returns it.
    plan-proxy.md §6.5: arguments must NOT be parsed into an object.
    """

    name: str = Field(min_length=1, max_length=256)
    arguments: str = Field(min_length=1)


class ToolCallSchema(BaseModel, extra="forbid"):
    """A single tool call within an assistant response.

    `id` is used EXACTLY as returned by the model via LiteLLM.
    plan-proxy.md §6.5: no prefix, no suffix, no modification.
    """

    id: str = Field(min_length=1, max_length=128)
    type: Literal["function"] = "function"
    function: ToolCallFunctionSchema


class ToolResultSchema(BaseModel, extra="forbid"):
    """A tool result message (role: 'tool').

    `tool_call_id` must match the assistant's tool_call id.
    """

    role: Literal["tool"] = "tool"
    tool_call_id: str = Field(min_length=1, max_length=128)
    name: str | None = Field(default=None, max_length=256)
    content: str = Field(default="", max_length=131072)


class NormalizeToolsRequest(BaseModel, extra="forbid"):
    """Request body for POST /conversations/{id}/normalize-tools."""

    dry_run: bool = False


class NormalizeToolsResponse(BaseModel, extra="forbid"):
    """Response from POST /conversations/{id}/normalize-tools.

    plan-proxy.md §6.8: original history is preserved.
    A normalization_event turn is created with the serialized history.
    """

    conversation_id: str
    normalized_turns: int = Field(ge=0)
    parallel_calls_serialized: int = Field(ge=0)
    turns_affected: list[int] = Field(default_factory=list)
    original_history_preserved: bool = True
    normalization_event_id: str | None = None
    preview: str | None = None
