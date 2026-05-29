"""Pseudo-models YAML loader with strict Pydantic validation at startup.

Fail-fast: any validation error → SystemExit(1) with clear FATAL message.
Exact 14 validation rules from feature
"""

import logging
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError

logger = logging.getLogger(__name__)


# ── Physical model ──────────────────────────────────────────────────────────


class PhysicalModelSchema(BaseModel, extra="forbid"):
    provider: str
    model: str
    openai_tools_compatible: bool = True
    tools_strict: bool = False
    parallel_tools: bool = False
    vision: bool = False
    audio: bool = False
    context_window: int | None = None
    note: str | None = None
    api_base: str | None = None
    """Custom API base URL (e.g. https://opencode.ai/zen/go/v1)."""
    api_key_env: str | None = None
    """Env var name holding the API key (e.g. OPENCODE_API_KEY)."""
    temperature: float | None = None
    """Default temperature — overrides client value when set."""
    top_p: float | None = None
    """Default top_p — overrides client value when set."""
    system_prompt: str | None = None
    """System prompt injected before user messages for this model."""


# ── Sub-configs ─────────────────────────────────────────────────────────────


class RouterLLMConfig(BaseModel, extra="forbid"):
    enabled: bool = False
    suggester: str | None = None
    suggest_on_downgrade_only: bool = True


class ImageHandlingConfig(BaseModel, extra="forbid"):
    on_downgrade: str = "block"

    @field_validator("on_downgrade")
    @classmethod
    def _valid_on_downgrade(cls, v: str) -> str:
        if v not in ("auto_describe", "block"):
            raise ValueError(
                f"image_handling.on_downgrade must be 'auto_describe' or 'block', got '{v}'"
            )
        return v


# ── Pseudo-model ────────────────────────────────────────────────────────────


class PseudoModelSchema(BaseModel, extra="forbid"):
    display_name: str
    description: str
    input_token_threshold: int | None = None
    context_window: int | None = None
    router_llm: RouterLLMConfig = Field(default_factory=RouterLLMConfig)
    image_handling: ImageHandlingConfig = Field(default_factory=ImageHandlingConfig)
    physical_models: list[PhysicalModelSchema]
    fallback_strategy: str = "sequential"
    default_thinking: dict | str | bool | None = None
    """Default thinking configuration when the client doesn't specify one.
    Accepts the same formats as the top-level ``thinking`` parameter:
    ``{"type": "enabled", "budget_tokens": 16000}``, ``{"type": "disabled"}``,
    ``True``, ``False``, ``"enabled"``, ``"disabled"``.
    Only applies when ``thinking`` is absent from the client request."""
    continue_on_length: bool = False
    """If ``True`` and a non-streaming model finishes with ``finish_reason="length"``
    while more physical models are available, the partial response is appended as
    an assistant message and the next model is called to continue (composite
    response). Streaming does the same seamlessly.

    Default ``False`` — ``length`` is treated as a normal completion."""

    @field_validator("display_name")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("display_name must not be empty")
        return v

    @field_validator("physical_models")
    @classmethod
    def _non_empty_physical_models(cls, v: list) -> list:
        if not v:
            raise ValueError("physical_models must have at least one model")
        return v

    @field_validator("fallback_strategy")
    @classmethod
    def _valid_strategy(cls, v: str) -> str:
        if v not in ("sequential", "by_context_window"):
            raise ValueError(
                f"fallback_strategy must be 'sequential' or 'by_context_window', got '{v}'"
            )
        return v


# ── Top-level config ────────────────────────────────────────────────────────


class ProxyConfigSchema(BaseModel, extra="forbid"):
    pseudo_models: dict[str, PseudoModelSchema]
    model_aliases: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _all_models_tools_compatible(self):
        for name, pm in self.pseudo_models.items():
            for i, phys in enumerate(pm.physical_models):
                if not phys.openai_tools_compatible:
                    raise ValueError(
                        f"pseudo_model '{name}' physical_model[{i}] '{phys.model}' "
                        f"has openai_tools_compatible: false. All models must be true."
                    )
        return self

    @model_validator(mode="after")
    def _router_references(self):
        for name, pm in self.pseudo_models.items():
            if pm.router_llm.enabled and pm.router_llm.suggester:
                if pm.router_llm.suggester not in self.pseudo_models:
                    raise ValueError(
                        f"pseudo_model '{name}' router_llm.suggester "
                        f"'{pm.router_llm.suggester}' references unknown pseudo-model"
                    )
        return self


# ── Validation helpers ──────────────────────────────────────────────────────


def _validate_pseudo_model_names(schema: ProxyConfigSchema) -> None:
    """Rule 3: names must be alphanumeric with hyphens only."""
    pattern = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9-]*$")
    for name in schema.pseudo_models:
        if not pattern.match(name):
            logger.critical(
                "Invalid pseudo-model name: '%s'. Only alphanumeric and hyphens allowed.",
                name,
            )
            raise SystemExit(1)


# ── Public loader ───────────────────────────────────────────────────────────


def load_config(path: Path = Path("pseudo_models.yaml")) -> ProxyConfigSchema:
    """Load and validate pseudo_models.yaml. Exits with code 1 on any error."""
    # Rule 1: parseable YAML
    try:
        with open(path) as f:
            raw = yaml.safe_load(f)
    except FileNotFoundError:
        logger.critical("%s not found", path)
        raise SystemExit(1) from None
    except yaml.YAMLError as e:
        logger.critical("%s is not valid YAML: %s", path, e)
        raise SystemExit(1) from None

    # Rule 2: pseudo_models key must exist and be a dict
    if not isinstance(raw, dict) or "pseudo_models" not in raw:
        logger.critical("pseudo_models must be a mapping at root level")
        raise SystemExit(1)

    # Validate with Pydantic
    try:
        schema = ProxyConfigSchema.model_validate(raw)
    except ValidationError as e:
        logger.critical("%s validation failed:", path)
        for error in e.errors():
            loc = " -> ".join(str(p) for p in error["loc"])
            logger.critical("  - %s: %s", loc, error["msg"])
        raise SystemExit(1) from None

    # Additional imperative validations
    _validate_pseudo_model_names(schema)

    return schema
