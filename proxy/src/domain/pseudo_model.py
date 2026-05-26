"""Pure domain entities for pseudo-models and physical models.

These are plain dataclasses — no FastAPI, no SQLAlchemy, no Pydantic.
python.md §1.1: domain must not import infrastructure.
"""

from dataclasses import dataclass, field


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
class RouterLLMConfig:
    """Router LLM configuration for a pseudo-model.

    plan-proxy.md §12: Optional evaluation of task complexity.
    """

    enabled: bool = False
    suggester: str | None = None
    suggest_on_downgrade_only: bool = True


@dataclass(frozen=True, slots=True)
class ImageHandlingConfig:
    """Image handling configuration for a pseudo-model.

    plan-proxy.md §7.2: On downgrade to a model without vision,
    either auto_describe or block.
    """

    on_downgrade: str = "block"


@dataclass(frozen=True, slots=True)
class PseudoModel:
    """A pseudo-model — a user-facing intent that resolves to physical models.

    Sprint 1: name, display_name, description, thresholds, physical_models.
    Sprint 4: +pre_compaction, +continuous_compaction, +image_handling (removed).
    Sprint 5 (planned): +router_llm.
    """

    name: str
    display_name: str
    description: str
    input_token_threshold: int | None
    context_window: int | None
    physical_models: tuple[PhysicalModel, ...]
    fallback_strategy: str = "sequential"

    # Sub-configurations
    image_handling: ImageHandlingConfig = field(default_factory=ImageHandlingConfig)
    router_llm: RouterLLMConfig = field(default_factory=RouterLLMConfig)
