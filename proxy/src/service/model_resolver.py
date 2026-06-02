"""Pseudo-model to physical model resolution logic.

Exact logic from feature No prefix manipulation, no string concatenation.
"""

from src.config.pseudo_models import (
    ImageHandlingConfig,
    PhysicalModelSchema,
    PseudoModelSchema,
    ProxyConfigSchema,
    RouterLLMConfig,
)


def normalize_model_name(raw_model: str, config: ProxyConfigSchema) -> str:
    """Normalize incoming model name by stripping provider/ prefix and resolving aliases.

    Rules (feature + Feature):
    1. Exact pseudo-model match → as-is
    2. Contains '/' → strip prefix before last '/' and try remainder
    3. Exact alias match → resolved pseudo-model name
    4. Default alias → if "default" in config.model_aliases
    5. No match → return original for error handling upstream
    """
    # 1. Exact pseudo-model match
    if raw_model in config.pseudo_models:
        return raw_model

    # 2. Strip provider prefix (e.g., "local/normal", "cesar-proxy/normal")
    if "/" in raw_model:
        candidate = raw_model.rsplit("/", 1)[-1]
        if candidate in config.pseudo_models:
            return candidate
        # Also try alias resolution on the stripped name
        if candidate in config.model_aliases:
            return config.model_aliases[candidate]

    # 3. Exact alias match (e.g., "gpt-4o" → "normal")
    if raw_model in config.model_aliases:
        return config.model_aliases[raw_model]

    # 4. Default fallback alias — only applies to simple names, not provider-prefixed models
    if "default" in config.model_aliases and "/" not in raw_model:
        return config.model_aliases["default"]

    # 5. No match — return original for passthrough (e.g. "ollama/llama3.2")
    return raw_model


def build_passthrough_pseudo_model(model_name: str) -> PseudoModelSchema:
    """Build a minimal passthrough pseudo-model for direct model calls.

    Supports models not defined in pseudo_models.yaml (e.g. ollama/llama3.2).
    No compaction, no router, no thresholds — just pass through to LiteLLM as-is.
    The response is reported exactly as the local model returns it.
    """
    provider = model_name.split("/")[0] if "/" in model_name else "openai"
    return PseudoModelSchema(
        display_name=model_name,
        description=f"Direct passthrough: {model_name}",
        input_token_threshold=None,
        context_window=None,
        router_llm=RouterLLMConfig(enabled=False),
        image_handling=ImageHandlingConfig(on_downgrade="auto_describe"),
        physical_models=[
            PhysicalModelSchema(
                provider=provider,
                model=model_name,
                openai_tools_compatible=True,
            )
        ],
        fallback_strategy="sequential",
    )
