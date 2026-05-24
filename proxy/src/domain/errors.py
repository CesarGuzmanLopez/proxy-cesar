"""Domain error types — pure data, no infrastructure dependencies.

python.md §3: errors as data in domain, exceptions at boundary.
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
class AllModelsFailed:
    attempted: tuple[str, ...]
    last_error: str


@dataclass(frozen=True, slots=True)
class ConversationNotFound:
    conversation_id: str


@dataclass(frozen=True, slots=True)
class CapabilityIncompatible:
    """A pseudo-model switch is blocked due to capability mismatch."""

    reason: str
    remediation: list[str]
    details: dict


@dataclass(frozen=True, slots=True)
class InputExceedsThreshold:
    """Input tokens exceed the pseudo-model's input_token_threshold."""

    estimated: int
    threshold: int
    pseudo_model: str


@dataclass(frozen=True, slots=True)
class ContentNotSupported:
    """Incoming content is not supported by the current pseudo-model."""

    content_type: str
    pseudo_model: str
    remediation: list[str]


@dataclass(frozen=True, slots=True)
class CompactorFailed:
    """The compactor model failed to generate a snapshot."""

    compactor_model: str
    reason: str
    original_input_preserved: bool = True


@dataclass(frozen=True, slots=True)
class CompactorNotFound:
    """The compactor pseudo-model was not found in config."""

    compactor_name: str
    pseudo_model: str


@dataclass(frozen=True, slots=True)
class ContextUnusable:
    """History exceeds all available model windows."""

    total_tokens: int
    max_context_window: int
    conversation_id: str


@dataclass(frozen=True, slots=True)
class HistoryTooLargeForCompactor:
    """No compactor model can handle this history size."""

    total_tokens: int
    max_compactor_window: int


type DomainError = (
    PseudoModelNotFound
    | PhysicalModelNotInList
    | AllModelsFailed
    | ConversationNotFound
    | CapabilityIncompatible
    | InputExceedsThreshold
    | ContentNotSupported
    | CompactorFailed
    | CompactorNotFound
    | ContextUnusable
    | HistoryTooLargeForCompactor
)
