"""Content validation for incoming requests.

Checks that the current pseudo-model can handle the content type
in the incoming request. Unsupported content is routed to the Blob
Vault for transformation (images, audio, PDF) or rejected (video).

python.md §4: pure functions with explicit Result monad.
python.md §3: errors as data in domain, exceptions at boundary.
"""

from src.config.pseudo_models import (
    PseudoModelSchema,
    ProxyConfigSchema,
)
from src.domain.capabilities import TurnCapabilities
from src.domain.errors import ParallelToolsNotSupported


def validate_incoming_content(
    turn_caps: TurnCapabilities,
    pseudo_model: PseudoModelSchema,
    pseudo_model_name: str,
    config: ProxyConfigSchema,
    tools: list[dict] | None = None,
) -> dict | None:
    """Validate that the current pseudo-model can handle the incoming content.

    Returns:
      - None if everything is OK
      - {"action": "transform_unsupported"} if content should be blobified
    Raises HTTPException for unrecoverable errors (parallel tools mismatch).
    """
    phys = pseudo_model.physical_models

    # Images → model without vision
    result = _check_content_support(
        turn_caps,
        "has_images",
        phys,
        "vision",
    )
    if result is not None:
        return result

    # Audio → model without audio
    result = _check_content_support(
        turn_caps,
        "has_audio",
        phys,
        "audio",
    )
    if result is not None:
        return result

    # PDF → model without vision
    result = _check_content_support(
        turn_caps,
        "has_pdf",
        phys,
        "vision",
    )
    if result is not None:
        return result

    # Video → model without video
    result = _check_content_support(
        turn_caps,
        "has_video",
        phys,
        "video",
    )
    if result is not None:
        return result

    # Parallel tools → model without parallel support
    parallel_error = _check_parallel_tools_support(turn_caps, phys, pseudo_model_name, config)
    if parallel_error is not None:
        raise ValueError(f"ParallelToolsNotSupported: {parallel_error}")


# ── Internal helpers ────────────────────────────────────────────────────────


def _check_content_support(
    turn_caps: TurnCapabilities,
    cap_attr: str,
    physical_models: list,
    capability: str,
) -> dict | None:
    """Check if any physical model supports a required capability.

    Returns None if supported, or {"action": "transform_unsupported"}.
    """
    if not getattr(turn_caps, cap_attr, False):
        return None

    has_capability = any(getattr(m, capability, False) for m in physical_models)
    if has_capability:
        return None

    return {"action": "transform_unsupported"}


def _check_parallel_tools_support(
    turn_caps: TurnCapabilities,
    physical_models: list,
    pseudo_model_name: str,
    config,
) -> ParallelToolsNotSupported | None:
    """Check if parallel tools are supported.

    Returns ParallelToolsNotSupported error if parallel tools are required
    but not supported by the pseudo-model. Returns None if OK.
    """
    if not turn_caps.has_parallel_tools:
        return None

    has_parallel = any(m.parallel_tools for m in physical_models)
    if has_parallel:
        return None

    return ParallelToolsNotSupported(
        pseudo_model=pseudo_model_name,
        has_parallel_tools=turn_caps.has_parallel_tools,
    )
