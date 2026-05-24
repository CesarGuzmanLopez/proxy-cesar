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


type DomainError = (
    PseudoModelNotFound
    | PhysicalModelNotInList
    | AllModelsFailed
    | ConversationNotFound
    | CapabilityIncompatible
    | InputExceedsThreshold
    | ContentNotSupported
)
