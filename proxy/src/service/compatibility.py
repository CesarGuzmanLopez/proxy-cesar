"""Content validation for incoming requests.

Checks that the current pseudo-model can handle the content type
in the incoming request. Unsupported content is routed to the Blob
Vault for transformation (images, audio, PDF) or rejected (video).

python.md §4: pure functions with explicit Result monad.
"""

from fastapi import HTTPException

from src.config.pseudo_models import (
    PhysicalModelSchema,
    PseudoModelSchema,
    ProxyConfigSchema,
)
from src.domain.capabilities import TurnCapabilities


_SPECIALIZED_MODEL_ERRORS: dict[str, dict[str, str]] = {}


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
    Raises HTTPException for unrecoverable errors (specialized model, etc.).
    """
    phys = pseudo_model.physical_models

    _check_specialized_model_mismatch(turn_caps, pseudo_model_name)

    # Images → model without vision
    result = _check_content_support(
        turn_caps, "has_images", phys, "vision",
    )
    if result is not None:
        return result

    # Audio → model without audio
    result = _check_content_support(
        turn_caps, "has_audio", phys, "audio",
    )
    if result is not None:
        return result

    # PDF → model without vision
    result = _check_content_support(
        turn_caps, "has_pdf", phys, "vision",
    )
    if result is not None:
        return result

    # Video → model without video
    result = _check_content_support(
        turn_caps, "has_video", phys, "video",
    )
    if result is not None:
        return result

    # Parallel tools → model without parallel support
    _check_parallel_tools_support(turn_caps, phys, pseudo_model_name, config)


# ── Internal helpers ────────────────────────────────────────────────────────


def _check_specialized_model_mismatch(
    turn_caps: TurnCapabilities,
    pseudo_model_name: str,
) -> None:
    """Raise HTTPException if content type is incompatible with specialized model."""
    content_type_map = {
        "has_images": "image_url",
        "has_audio": "input_audio",
        "has_video": "video",
    }
    for cap_attr, error_type in content_type_map.items():
        if not getattr(turn_caps, cap_attr, False):
            continue
        if pseudo_model_name not in _SPECIALIZED_MODEL_ERRORS:
            continue
        error_info = _SPECIALIZED_MODEL_ERRORS[pseudo_model_name]
        if error_type not in error_info:
            continue
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"{error_type.upper()}_NOT_SUPPORTED_BY_SPECIALIZED_MODEL",
                "message": error_info[error_type],
                "remediation": [
                    f"Use a different model for {error_type} content",
                    "Try 'model': 'vision' for images, or 'model': 'audio' for audio",
                ],
                "current_pseudo_model": pseudo_model_name,
            },
        )


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
) -> None:
    """Raise HTTPException if parallel tools are not supported."""
    if not turn_caps.has_parallel_tools:
        return

    has_parallel = any(m.parallel_tools for m in physical_models)
    if has_parallel:
        return

    parallel_pseudos = [
        name
        for name, pm in config.pseudo_models.items()
        if any(m.parallel_tools for m in pm.physical_models)
    ]
    raise HTTPException(
        status_code=400,
        detail={
            "error": "PARALLEL_TOOLS_NOT_SUPPORTED_BY_PSEUDO_MODEL",
            "message": (
                f"Pseudo-model '{pseudo_model_name}' has no physical models "
                f"with parallel_tools: true. The incoming request contains "
                f"parallel tool calls that cannot be processed."
            ),
            "remediation": [
                f"Switch to a pseudo-model with parallel tool support: {parallel_pseudos}",
                "Use POST /conversations/{id}/normalize-tools to serialize parallel calls (Sprint 3)",
            ],
        },
    )


def _any_vision(models: list[PhysicalModelSchema]) -> bool:
    """Check if any physical model in the list has vision capability."""
    return any(m.vision for m in models)
