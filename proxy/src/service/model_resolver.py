"""Pseudo-model to physical model resolution logic.

Exact logic from sprint §7. No prefix manipulation, no string concatenation.
"""

from src.config.pseudo_models import ProxyConfigSchema
from src.domain.errors import PhysicalModelNotInList, PseudoModelNotFound
from src.domain.types import Err, Ok, Result


def normalize_model_name(
    raw_model: str, config: ProxyConfigSchema
) -> str:
    """Normalize incoming model name by stripping provider/ prefix.

    Rules (sprint §7.3):
    1. Exact match → as-is
    2. Contains '/' → strip prefix before last '/' and try remainder
    3. No match → return original for error handling upstream
    """
    if raw_model in config.pseudo_models:
        return raw_model

    if "/" in raw_model:
        candidate = raw_model.rsplit("/", 1)[-1]
        if candidate in config.pseudo_models:
            return candidate

    return raw_model


def resolve_physical_model(
    pseudo_model_name: str,
    config: ProxyConfigSchema,
    existing_affinity: str | None = None,
) -> Result[str, PseudoModelNotFound | PhysicalModelNotInList]:
    """Resolve a pseudo-model name to an exact LiteLLM model ID.

    Priority:
    1. existing_affinity → use if still in physical_models list
    2. otherwise → first physical_model in list (priority 1)

    Returns Ok(model) | Err(...)
    """
    pm_schema = config.pseudo_models.get(pseudo_model_name)
    if pm_schema is None:
        return Err(PseudoModelNotFound(name=pseudo_model_name))

    if existing_affinity:
        for phys in pm_schema.physical_models:
            if phys.model == existing_affinity:
                return Ok(existing_affinity)

    return Ok(pm_schema.physical_models[0].model)
