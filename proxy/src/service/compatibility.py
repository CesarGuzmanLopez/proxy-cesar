"""Compatibility validation for pseudo-model switches and incoming content.

Implements the compatibility matrix from analisis.md §8 and SPEC.md §3.2.
All checks are deterministic — no ML, no heuristics.

python.md §4: pure functions with explicit Result monad.
"""

from fastapi import HTTPException

from src.config.pseudo_models import (
    PhysicalModelSchema,
    PseudoModelSchema,
    ProxyConfigSchema,
)
from src.domain.capabilities import (
    CompatibilityResult,
    CompatibilityStatus,
    SessionCapabilities,
    TurnCapabilities,
)


def _check_images(
    from_pseudo_name: str,
    to_pseudo_name: str,
    to_pseudo: PseudoModelSchema,
    caps: SessionCapabilities,
    config: ProxyConfigSchema,
) -> CompatibilityResult | None:
    """CHECK 1: Images → vision compatibility."""
    if not caps.has_images:
        return None

    dest_has_vision = _any_vision(to_pseudo.physical_models)
    if not dest_has_vision:
        if to_pseudo.image_handling.on_downgrade == "auto_describe":
            return CompatibilityResult(
                status=CompatibilityStatus.WARNING,
                reason="Images in history will be auto-described textually before migration (Sprint 5).",
                details={"images_described_by": "current_vision_model"},
            )
        return CompatibilityResult(
            status=CompatibilityStatus.BLOCKED,
            reason=(
                "Conversation contains images but destination pseudo-model "
                "lacks vision support. Use a vision-capable model to describe "
                "images via tools, then switch."
            ),
            remediation=[
                "Switch to a vision-capable pseudo-model first",
                "Use a vision model as a tool to describe images as text",
            ],
        )

    # Both have vision — check for reduced visual capacity
    from_pm = config.pseudo_models.get(from_pseudo_name)
    if from_pm and _any_vision(from_pm.physical_models):
        from_cw = from_pm.context_window
        to_cw = to_pseudo.context_window
        if from_cw and to_cw and to_cw < from_cw:
            return CompatibilityResult(
                status=CompatibilityStatus.WARNING,
                reason=(
                    f"Destination pseudo-model '{to_pseudo_name}' has "
                    f"smaller context window ({to_cw} vs {from_cw}). "
                    f"Reduced capacity for image processing."
                ),
                details={
                    "from_context_window": from_cw,
                    "to_context_window": to_cw,
                },
            )
    return None


def _check_audio(
    to_pseudo: PseudoModelSchema,
    caps: SessionCapabilities,
) -> CompatibilityResult | None:
    """CHECK 2: Audio in history → destination must support audio."""
    if not caps.has_audio:
        return None

    to_has_audio = any(getattr(m, "audio", False) for m in to_pseudo.physical_models)
    if not to_has_audio:
        return CompatibilityResult(
            status=CompatibilityStatus.BLOCKED,
            reason=(
                "Conversation contains audio content. No destination model "
                "supports audio. Audio degradation is not available in v1."
            ),
            remediation=[
                "Start a new conversation without audio content",
                "Audio support is planned for v2",
            ],
        )
    return None


def _check_pdf(
    to_pseudo: PseudoModelSchema,
    caps: SessionCapabilities,
) -> CompatibilityResult | None:
    """CHECK 3: PDF in history → model without vision."""
    if caps.has_pdf and not _any_vision(to_pseudo.physical_models):
        return CompatibilityResult(
            status=CompatibilityStatus.BLOCKED,
            reason=(
                "Conversation contains PDF files. Destination lacks vision support. "
                "In v1, PDFs require vision models."
            ),
            remediation=[
                "Use a vision-capable pseudo-model (vision)",
                "PDF text extraction is planned for v2",
            ],
        )
    return None


def _check_video(
    caps: SessionCapabilities,
) -> CompatibilityResult | None:
    """CHECK 4: Video in history — not supported in v1."""
    if caps.has_video:
        return CompatibilityResult(
            status=CompatibilityStatus.BLOCKED,
            reason="Video content is not supported in any pseudo-model in v1.",
            remediation=["Video support is planned for a future version"],
        )
    return None


def _check_parallel_tools(
    from_pseudo_name: str,
    to_pseudo_name: str,
    to_pseudo: PseudoModelSchema,
    caps: SessionCapabilities,
    config: ProxyConfigSchema,
) -> CompatibilityResult | None:
    """CHECK 5: Parallel tools → destination must have parallel-capable models."""
    if not caps.has_parallel_tools:
        return None

    parallel_eligible = [m for m in to_pseudo.physical_models if m.parallel_tools]
    if not parallel_eligible:
        return CompatibilityResult(
            status=CompatibilityStatus.BLOCKED,
            reason=(
                "Conversation history contains parallel tool calls. "
                "No model in the destination pseudo-model supports parallel tools."
            ),
            remediation=[
                "POST /conversations/{id}/normalize-tools — serialize parallel calls to sequential (Sprint 3)",
                "Switch to a pseudo-model with parallel_tools support",
            ],
        )

    # Destination has parallel tools but fewer than source → WARNING
    from_pm = config.pseudo_models.get(from_pseudo_name)
    if from_pm:
        from_parallel_count = sum(
            1 for m in from_pm.physical_models if m.parallel_tools
        )
        if from_parallel_count > len(parallel_eligible):
            return CompatibilityResult(
                status=CompatibilityStatus.WARNING,
                reason=(
                    f"Destination pseudo-model '{to_pseudo_name}' has "
                    f"fewer parallel-tool models ({len(parallel_eligible)}) "
                    f"than source '{from_pseudo_name}' ({from_parallel_count}). "
                    f"Existing parallel tool calls may be less reliable."
                ),
                details={
                    "from_parallel_count": from_parallel_count,
                    "to_parallel_count": len(parallel_eligible),
                },
            )
    return None


def _check_context(
    to_pseudo: PseudoModelSchema,
    caps: SessionCapabilities,
) -> CompatibilityResult | None:
    """CHECK 6: Context too large for destination."""
    if to_pseudo.context_window and caps.total_tokens > to_pseudo.context_window:
        return CompatibilityResult(
            status=CompatibilityStatus.BLOCKED,
            reason=(
                f"Accumulated context ({caps.total_tokens} tokens) exceeds "
                f"destination window ({to_pseudo.context_window} tokens)."
            ),
            remediation=[
                "POST /conversations/{id}/compact — compact the conversation before switching (Sprint 6)",
                "Switch to a pseudo-model with larger context window",
            ],
        )
    return None


def _check_tools_downgrade(
    from_pseudo_name: str,
    to_pseudo: PseudoModelSchema,
    caps: SessionCapabilities,
    config: ProxyConfigSchema,
) -> CompatibilityResult | None:
    """CHECK 7: Tools downgrade warning."""
    if not caps.has_tools:
        return None

    to_strict_count = sum(1 for m in to_pseudo.physical_models if m.tools_strict)
    from_pm = config.pseudo_models.get(from_pseudo_name)
    from_strict_count = (
        sum(1 for m in from_pm.physical_models if m.tools_strict) if from_pm else 0
    )
    if to_strict_count == 0 and from_strict_count > 0:
        return CompatibilityResult(
            status=CompatibilityStatus.WARNING,
            reason=(
                "Destination pseudo-model lacks models with tools_strict support. "
                "Tool call parameter validation may be less reliable."
            ),
            details={
                "from_strict_models": from_strict_count,
                "to_strict_models": 0,
            },
        )
    return None


def _check_capacity_loss(
    from_pseudo_name: str,
    to_pseudo_name: str,
) -> CompatibilityResult | None:
    """CHECK 8: General capacity loss when switching to budget models."""
    _BUDGET_MODELS: set[str] = {"flash-lowcost", "massive-fast"}
    if to_pseudo_name in _BUDGET_MODELS and from_pseudo_name not in _BUDGET_MODELS:
        return CompatibilityResult(
            status=CompatibilityStatus.WARNING,
            reason=(
                f"Switching to '{to_pseudo_name}', a budget model with "
                f"reduced reasoning capacity and/or tool quality compared "
                f"to '{from_pseudo_name}'."
            ),
            details={
                "from_pseudo": from_pseudo_name,
                "to_pseudo": to_pseudo_name,
            },
        )
    return None


def validate_switch(
    from_pseudo_name: str,
    to_pseudo_name: str,
    to_pseudo: PseudoModelSchema,
    caps: SessionCapabilities,
    config: ProxyConfigSchema,
) -> CompatibilityResult:
    """Determine if switching from one pseudo-model to another is safe.

    Returns SAFE, WARNING, or BLOCKED with reason and remediation options.
    Implements the logic from analisis.md §8.1 and the matrix in §8.2.
    """
    # ---- CHECK 0: Same pseudo-model → always SAFE ----
    if from_pseudo_name == to_pseudo_name:
        return CompatibilityResult(
            status=CompatibilityStatus.SAFE,
            reason="Same pseudo-model. No switch needed.",
        )

    # ---- CHECK 0b: compactador → always SAFE (it's an operation) ----
    if to_pseudo_name == "compactador":
        return CompatibilityResult(
            status=CompatibilityStatus.SAFE,
            reason="Compactador is an operation, not a conversation model.",
        )

    # Run all compatibility checks in order
    checks = [
        _check_images(from_pseudo_name, to_pseudo_name, to_pseudo, caps, config),
        _check_audio(to_pseudo, caps),
        _check_pdf(to_pseudo, caps),
        _check_video(caps),
        _check_parallel_tools(
            from_pseudo_name, to_pseudo_name, to_pseudo, caps, config
        ),
        _check_context(to_pseudo, caps),
        _check_tools_downgrade(from_pseudo_name, to_pseudo, caps, config),
        _check_capacity_loss(from_pseudo_name, to_pseudo_name),
    ]

    for result in checks:
        if result is not None:
            return result

    # All checks passed → SAFE
    return CompatibilityResult(
        status=CompatibilityStatus.SAFE,
        reason="All capabilities compatible.",
    )


def validate_incoming_content(
    turn_caps: TurnCapabilities,
    pseudo_model: PseudoModelSchema,
    pseudo_model_name: str,
    config: ProxyConfigSchema,
    tools: list[dict] | None = None,
) -> dict | None:
    """Validate that the current pseudo-model can handle the incoming content.

    This check runs on EVERY turn, not just on pseudo-model switches.

    Returns:
      - None if everything is OK
      - {"action": "delegate_images", ...} if images can be delegated to tool
      - {"action": "transform_unsupported"} if content should be blobified
    Raises HTTPException on unrecoverable errors (wrong content for model, etc.).
    """
    phys = pseudo_model.physical_models

    _check_specialized_model_mismatch(turn_caps, pseudo_model_name)

    # Images → model without vision
    result = _check_content_support(
        turn_caps, "has_images", phys, "vision",
        pseudo_model_name, tools, can_delegate=True,
    )
    if result is not None:
        return result

    # Audio → model without audio
    result = _check_content_support(
        turn_caps, "has_audio", phys, "audio", pseudo_model_name,
    )
    if result is not None:
        return result

    # PDF → model without vision
    result = _check_content_support(
        turn_caps, "has_pdf", phys, "vision", pseudo_model_name,
    )
    if result is not None:
        return result

    # Video → model without video
    result = _check_content_support(
        turn_caps, "has_video", phys, "video", pseudo_model_name,
    )
    if result is not None:
        return result

    # Parallel tools → model without parallel support
    _check_parallel_tools_support(turn_caps, phys, pseudo_model_name, config)


# ── Internal content validators ────────────────────────────────────────────────

_SPECIALIZED_MODEL_ERRORS: dict[str, dict[str, str]] = {
    "imagen": {
        "image_url": (
            "The 'imagen' model generates images from text. "
            "It cannot process image input."
        ),
        "input_audio": (
            "The 'imagen' model generates images from text. "
            "It cannot process audio input."
        ),
    },
    "audio": {
        "image_url": (
            "The 'audio' model transcribes audio to text. "
            "It cannot process image input."
        ),
    },
}


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
    pseudo_model_name: str,
    tools: list[dict] | None = None,
    can_delegate: bool = False,
) -> dict | None:
    """Check if any physical model supports a required capability.

    Returns None if supported, or {"action": "transform_unsupported"}.
    For images with can_delegate=True, also checks tool delegation.
    """
    if not getattr(turn_caps, cap_attr, False):
        return None

    has_capability = any(getattr(m, capability, False) for m in physical_models)
    if has_capability:
        return None

    # Try tool delegation for images
    if can_delegate:
        from src.service.tool_detector import find_image_compatible_tool

        match = find_image_compatible_tool(tools)
        if match:
            return {
                "action": "delegate_images",
                "tool_name": match[0],
                "param_name": match[1],
            }

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
