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
class AllModelsFailed:
    """All physical models in the fallback chain failed.

    Moved from HTTPException to domain error to represent as Result[T, E].
    """

    pseudo_model: str
    attempted: list[str]  # list of model names tried
    last_error: str  # the last error message from the LLM provider


@dataclass(frozen=True, slots=True)
class ContextTooLargeForAllModels:
    """Conversation context exceeds context window of all remaining models.

    Indicates the estimated input tokens exceed the context window of every
    physical model in the fallback chain.
    """

    estimated_tokens: int
    context_skipped: list[str]  # models that were skipped
    pseudo_model: str


@dataclass(frozen=True, slots=True)
class ParallelToolsNotSupported:
    """Pseudo-model has no physical models with parallel tool support.

    Request contains parallel tool calls but the chosen pseudo-model
    cannot process them.
    """

    pseudo_model: str
    has_parallel_tools: bool


@dataclass(frozen=True, slots=True)
class ConversationNotFound:
    """Conversation with the given ID does not exist."""

    conversation_id: str


@dataclass(frozen=True, slots=True)
class EmptyConversation:
    """Conversation has no turns to compact."""

    conversation_id: str


@dataclass(frozen=True, slots=True)
class HistoryTooLargeForCompactor:
    """No compactor model has a context window large enough for the history."""

    total_tokens: int
    max_compactor_window: int


@dataclass(frozen=True, slots=True)
class CompactionFailed:
    """Compactor model failed to process the conversation."""

    conversation_id: str
    compactor_model: str
    reason: str  # error message from the compactor API


@dataclass(frozen=True, slots=True)
class ContextUnusable:
    """Conversation context is unusable and requires compaction."""

    conversation_id: str
    context_tokens: int
    context_window: int
    warning_message: str
