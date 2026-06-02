"""Domain types for tool calling — pure dataclasses, no infrastructure.

plan-proxy.md §6.5: canonical OpenAI format for tool storage.
python.md §1.1: domain must not import FastAPI, SQLAlchemy, or Pydantic.
python.md §4: pure functions, immutable data.
"""

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
