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
