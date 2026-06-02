"""Pydantic v2 schemas for tool normalization.

plan-proxy.md §6.5: canonical OpenAI format.
python.md §6.3: Pydantic v2 with extra="forbid".
"""

from pydantic import BaseModel, Field


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
