"""Tool filter: filter physical model pool based on session capabilities.

All models in config already have openai_tools_compatible: true
(validated at startup in Sprint 1). The main filter in Sprint 2 is
parallel_tools: filter to only models that support parallel tool calls.

python.md §4: pure functions, declarative style.
"""

from src.config.pseudo_models import PhysicalModelSchema
from src.domain.capabilities import SessionCapabilities


def get_eligible_models(
    physical_models: list[PhysicalModelSchema],
    session_caps: SessionCapabilities,
) -> list[PhysicalModelSchema]:
    """Filter physical models based on session capabilities.

    Args:
        physical_models: Full list of physical models from the pseudo-model.
        session_caps: Accumulated session capabilities.

    Returns:
        Filtered list. If session has parallel tools, only models with
        parallel_tools: True are returned. If none qualify, returns all
        (compatibility validator will block the switch separately).
    """
    if not session_caps.has_parallel_tools:
        return physical_models  # No filtering needed

    parallel_eligible = [m for m in physical_models if m.parallel_tools]

    if parallel_eligible:
        return parallel_eligible

    # No model supports parallel tools — return all with warning.
    # The compatibility validator blocks the switch if this would break things.
    return physical_models


def is_pinned_model_eligible(
    pinned_model: str,
    eligible_models: list[PhysicalModelSchema],
) -> bool:
    """Check if the pinned model is in the eligible list.

    Args:
        pinned_model: Current physical model ID from affinity.
        eligible_models: Filtered list of eligible models.

    Returns:
        True if pinned model is still eligible.
    """
    return any(m.model == pinned_model for m in eligible_models)
