"""Local model discovery — Ollama and LM Studio.

Queries local providers for available models and their capabilities.
Applies a 30% safety factor to reported context windows: if a model
reports 128K context, the proxy advertises 38K to leave headroom for
conversation turns, tool calls, and system prompts.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)

OLLAMA_BASE = "http://localhost:11434"
LMSTUDIO_BASE = "http://localhost:1234"

# 30% of reported context window is used as the effective limit
CONTEXT_SAFETY_FACTOR = 0.30

# Default context window when the model doesn't report one
DEFAULT_CONTEXT_WINDOW = 32768


@dataclass
class LocalModelInfo:
    """Discovered local model with resolved capabilities."""

    id: str
    provider: str  # "ollama" or "lmstudio"
    display_name: str
    context_window: int
    vision: bool
    tools: bool
    parameter_size: str | None = None
    architecture: str | None = None
    raw_info: dict[str, Any] = field(default_factory=dict)


async def discover_local_models() -> list[LocalModelInfo]:
    """Query all local providers and return discovered models."""
    models: list[LocalModelInfo] = []

    try:
        ollama_models = await _discover_ollama()
        models.extend(ollama_models)
    except Exception as e:
        logger.debug("ollama discovery failed: %s", e)

    try:
        lmstudio_models = await _discover_lmstudio()
        models.extend(lmstudio_models)
    except Exception as e:
        logger.debug("lmstudio discovery failed: %s", e)

    return models


# ── Ollama ────────────────────────────────────────────────────────────────


async def _discover_ollama() -> list[LocalModelInfo]:
    """Discover models from a local Ollama instance."""
    async with httpx.AsyncClient(base_url=OLLAMA_BASE, timeout=3) as client:
        resp = await client.get("/api/tags")
        resp.raise_for_status()
        data = resp.json()

    models: list[LocalModelInfo] = []
    for entry in data.get("models", []):
        name = entry.get("name", "")
        if not name:
            continue

        ctx_window = DEFAULT_CONTEXT_WINDOW
        has_vision = False
        has_tools = False
        param_size = entry.get("details", {}).get("parameter_size", "")
        arch = entry.get("details", {}).get("family", "")

        # Try to get detailed info for context window and capabilities
        try:
            detailed = await _get_ollama_model_info(name)
            if detailed:
                ctx_window = detailed.get("context_window", ctx_window)
                has_vision = detailed.get("vision", False)
        except Exception as e:
            logger.debug("ollama show failed for %s: %s", name, e)

        effective_ctx = max(int(ctx_window * CONTEXT_SAFETY_FACTOR), 4096)

        models.append(
            LocalModelInfo(
                id=f"ollama/{name}",
                provider="ollama",
                display_name=name,
                context_window=effective_ctx,
                vision=has_vision,
                tools=has_tools,
                parameter_size=param_size,
                architecture=arch,
            )
        )

    return models


async def _get_ollama_model_info(model_name: str) -> dict | None:
    """Fetch detailed model info from Ollama's /api/show endpoint."""
    async with httpx.AsyncClient(base_url=OLLAMA_BASE, timeout=3) as client:
        resp = await client.post("/api/show", json={"model": model_name})
        resp.raise_for_status()
        data = resp.json()

    info: dict = {}
    model_info = data.get("model_info", {})

    # Context window is in model_info as "<arch>.context_length"
    ctx = None
    for key, value in model_info.items():
        if key.endswith(".context_length") and isinstance(value, (int, float)):
            ctx = int(value)
            break
    info["context_window"] = ctx or DEFAULT_CONTEXT_WINDOW

    # Vision capability
    capabilities = data.get("capabilities", [])
    info["vision"] = "vision" in capabilities

    return info


# ── LM Studio ─────────────────────────────────────────────────────────────


async def _discover_lmstudio() -> list[LocalModelInfo]:
    """Discover models from a local LM Studio instance."""
    async with httpx.AsyncClient(base_url=LMSTUDIO_BASE, timeout=3) as client:
        resp = await client.get("/api/v1/models")
        resp.raise_for_status()
        data = resp.json()

    models: list[LocalModelInfo] = []
    for entry in data.get("models", []):
        if entry.get("type") != "llm":
            continue

        model_id = entry.get("key", "")
        if not model_id:
            continue

        # Context window: prefer max_context_length, fall back to loaded context
        ctx_max = entry.get("max_context_length")
        loaded = entry.get("loaded_instances", [])
        ctx_loaded = (
            loaded[0].get("config", {}).get("context_length") if loaded else None
        )
        ctx_raw = ctx_max or ctx_loaded or DEFAULT_CONTEXT_WINDOW

        capabilities = entry.get("capabilities", {})
        has_vision = capabilities.get("vision", False)
        has_tools = capabilities.get("trained_for_tool_use", False)

        effective_ctx = max(int(ctx_raw * CONTEXT_SAFETY_FACTOR), 4096)

        models.append(
            LocalModelInfo(
                id=f"lmstudio/{model_id}",
                provider="lmstudio",
                display_name=entry.get("display_name", model_id),
                context_window=effective_ctx,
                vision=has_vision,
                tools=has_tools,
                parameter_size=entry.get("params_string", ""),
                architecture=entry.get("architecture", ""),
            )
        )

    return models
