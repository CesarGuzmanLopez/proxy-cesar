"""Pure domain entities for pseudo-models and physical models.

These are plain dataclasses — no FastAPI, no SQLAlchemy, no Pydantic.
python.md §1.1: domain must not import infrastructure.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PhysicalModel:
    provider: str
    model_id: str
    openai_tools_compatible: bool = True
    tools_strict: bool = False
    parallel_tools: bool = False
    vision: bool = False
    context_window: int | None = None
    note: str | None = None


@dataclass(frozen=True, slots=True)
class PseudoModel:
    name: str
    display_name: str
    description: str
    input_token_threshold: int | None
    context_window: int | None
    physical_models: tuple[PhysicalModel, ...]
    fallback_strategy: str = "sequential"
