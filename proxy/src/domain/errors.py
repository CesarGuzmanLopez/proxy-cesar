"""Domain error types — pure data, no infrastructure dependencies.

python.md §3: errors as data in domain, exceptions at boundary.
Only types that are actually raised/returned by the service layer are kept.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PseudoModelNotFound:
    name: str


@dataclass(frozen=True, slots=True)
class PhysicalModelNotInList:
    model: str
    pseudo_model: str


@dataclass(frozen=True, slots=True)
class InputExceedsThreshold:
    """Input tokens exceed the pseudo-model's input_token_threshold."""

    estimated: int
    threshold: int
    pseudo_model: str


@dataclass(frozen=True, slots=True)
class StreamPersistenceFailed:
    """Turn persistence failed after streaming completed.

    Indicates a DB error during _persist_stream_turn, causing the turn
    (and its response) to be lost from conversation history.
    """

    conversation_id: str
    turn_number: int
    reason: str  # e.g. "database connection lost", "constraint violation"


@dataclass(frozen=True, slots=True)
class StreamInterrupted:
    """Streaming interrupted before [DONE] was reached.

    Client disconnection or internal error during chunk iteration.
    """

    conversation_id: str
    chunks_received: int
    reason: str


@dataclass(frozen=True, slots=True)
class ContextContaminated:
    """Conversation context contains duplicated or corrupted messages.

    Detected when turn_type is 'degradation_event' and would cause
    the full historical context to be replayed.
    """

    conversation_id: str
    turn_type: str
    reason: str


@dataclass(frozen=True, slots=True)
class AllModelsFailed:
    """All physical models in the fallback chain failed.

    Moved from HTTPException to domain error to represent as Result[T, E].
    """

    pseudo_model: str
    attempted: list[str]  # list of model names tried
    last_error: str  # the last error message from the LLM provider
