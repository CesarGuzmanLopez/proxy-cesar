"""Domain types for tool calling — pure dataclasses, no infrastructure.

plan-proxy.md §6.5: canonical OpenAI format for tool storage.
python.md §1.1: domain must not import FastAPI, SQLAlchemy, or Pydantic.
python.md §4: pure functions, immutable data.
"""

from dataclasses import dataclass
from enum import IntEnum


class ToolLevel(IntEnum):
    """Tool complexity level used in a conversation turn.

    plan-proxy.md §6.3: models have different tool capability levels.
    - NONE: no tools used in this turn
    - BASIC: single tool call, simple schema, no strict
    - STANDARD: single tool call with optional/required params
    - PARALLEL_STRICT: multiple parallel calls OR strict mode enabled
    """

    NONE = 0
    BASIC = 1
    STANDARD = 2
    PARALLEL_STRICT = 3


@dataclass(frozen=True, slots=True)
class ToolDef:
    """A single tool function definition — canonical OpenAI format.

    plan-proxy.md §6.5: stored exactly as sent by the client.
    The `parameters` field is a JSON Schema object (dict).
    """

    name: str
    description: str
    parameters: dict  # JSON Schema object
    strict: bool | None = None


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A tool call made by the assistant — canonical OpenAI format.

    plan-proxy.md §6.5: `arguments` is a JSON STRING, not a parsed object.
    The `id` is used EXACTLY as returned by the model via LiteLLM.
    No prefix, no suffix, no modification.
    """

    id: str
    name: str
    arguments: str  # JSON string — stored as-is from model


@dataclass(frozen=True, slots=True)
class ToolResult:
    """A tool result message — canonical OpenAI format.

    plan-proxy.md §6.5: `tool_call_id` must match the assistant's tool_call id.
    Content is the raw result or error message.
    """

    tool_call_id: str
    content: str
    name: str | None = None


@dataclass(frozen=True, slots=True)
class ThinkingBlock:
    """Thinking/reasoning content from models that support it.

    plan-proxy.md §6.7: Thinking blocks preserve cache affinity
    when switching between models with reasoning support.
    """

    content: str
    provider: str | None = None  # deepseek, anthropic, google, openai
