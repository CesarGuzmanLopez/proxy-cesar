"""Result monad for explicit error handling (python.md §3)."""

from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")
E = TypeVar("E")


@dataclass(frozen=True, slots=True)
class Ok[T]:
    value: T
    success: bool = True


@dataclass(frozen=True, slots=True)
class Err[E]:
    error: E
    success: bool = False


type Result[T, E] = Ok[T] | Err[E]
