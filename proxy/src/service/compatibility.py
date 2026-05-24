"""Compatibility validation for pseudo-model switches and incoming content.

Implements the compatibility matrix from analisis.md §8 and SPEC.md §3.2.
All checks are deterministic — no ML, no heuristics.

python.md §4: pure functions with explicit Result monad.
"""

from fastapi import HTTPException

from src.config.pseudo_models import PhysicalModelSchema, PseudoModelSchema, ProxyConfigSchema
from src.domain.capabilities import (
    CompatibilityResult,
    CompatibilityStatus,
    SessionCapabilities,
    TurnCapabilities,
)


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

    # ---- CHECK 1: Images → vision compatibility ----
    if caps.has_images:
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
                    "lacks vision support."
                ),
                remediation=[
                    "Enable 'auto_describe' on destination pseudo-model in pseudo_models.yaml",
                    "POST /conversations/{id}/degrade-images (available in Sprint 5)",
                ],
            )
        # Both source and destination have vision — check for reduced
        # visual capacity (smaller context_window → fewer image tokens).
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

    # ---- CHECK 2: Audio in history ----
    if caps.has_audio:
        to_has_audio = any(
            getattr(m, "audio", False) for m in to_pseudo.physical_models
        )
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

    # ---- CHECK 3: PDF in history → model without vision ----
    if caps.has_pdf and not _any_vision(to_pseudo.physical_models):
        return CompatibilityResult(
            status=CompatibilityStatus.BLOCKED,
            reason=(
                "Conversation contains PDF files. Destination lacks vision support. "
                "In v1, PDFs require vision models."
            ),
            remediation=[
                "Use a vision-capable pseudo-model (avanzada-vision, flash-vision)",
                "PDF text extraction is planned for v2",
            ],
        )

    # ---- CHECK 4: Video in history ----
    if caps.has_video:
        return CompatibilityResult(
            status=CompatibilityStatus.BLOCKED,
            reason="Video content is not supported in any pseudo-model in v1.",
            remediation=["Video support is planned for a future version"],
        )

    # ---- CHECK 5: Parallel tools → destination lacks parallel models ----
    if caps.has_parallel_tools:
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

    # ---- CHECK 6: Context too large for destination ----
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

    # ---- CHECK 7: Tools downgrade warning ----
    if caps.has_tools:
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

    # ---- CHECK 8: General capacity loss warning ----
    # Warn when switching TO a budget model (deep-flash / flash-lowcost)
    # from a more capable model. These are the only pseudo-models that
    # represent a meaningful capacity downgrade warranting a warning
    # (per compatibility matrix in analisis.md §8.2).
    _BUDGET_MODELS: set[str] = {"deep-flash", "flash-lowcost"}
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

    # ---- All checks passed → SAFE ----
    return CompatibilityResult(
        status=CompatibilityStatus.SAFE,
        reason="All capabilities compatible.",
    )


def validate_incoming_content(
    turn_caps: TurnCapabilities,
    pseudo_model: PseudoModelSchema,
    pseudo_model_name: str,
    config: ProxyConfigSchema,
) -> None:
    """Validate that the current pseudo-model can handle the incoming content.

    This check runs on EVERY turn, not just on pseudo-model switches.
    It prevents silent data loss when the client sends content the model
    can't process.

    Raises HTTPException with descriptive error and remediation on failure.
    """
    physical_models = pseudo_model.physical_models

    # ---- CHECK: Images → model without vision ----
    if turn_caps.has_images:
        has_vision_model = any(m.vision for m in physical_models)
        if not has_vision_model:
            vision_pseudos = [
                name
                for name, pm in config.pseudo_models.items()
                if any(m.vision for m in pm.physical_models)
            ]
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "IMAGES_NOT_SUPPORTED_BY_PSEUDO_MODEL",
                    "message": (
                        f"Pseudo-model '{pseudo_model_name}' "
                        f"({pseudo_model.display_name}) has no vision-capable "
                        f"physical models. The incoming request contains images "
                        f"that cannot be processed. No content was lost — "
                        f"the request was rejected with this error."
                    ),
                    "remediation": [
                        f"Switch to a vision-capable pseudo-model: {vision_pseudos}",
                        "Use auto_describe to downgrade images to text (Sprint 5)",
                    ],
                    "current_pseudo_model": pseudo_model_name,
                    "vision_capable_pseudo_models": vision_pseudos,
                },
            )

    # ---- CHECK: Audio → no model supports audio in v1 ----
    if turn_caps.has_audio:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "AUDIO_NOT_SUPPORTED",
                "message": (
                    f"Pseudo-model '{pseudo_model_name}' does not support audio "
                    f"content. No pseudo-model in v1 supports audio processing. "
                    f"The request was rejected rather than silently dropping the audio."
                ),
                "remediation": [
                    "Remove audio content from the request",
                    "Audio transcription support is planned for v2",
                ],
            },
        )

    # ---- CHECK: PDF → model without vision ----
    if turn_caps.has_pdf:
        has_vision_model = any(m.vision for m in physical_models)
        if not has_vision_model:
            vision_pseudos = [
                name
                for name, pm in config.pseudo_models.items()
                if any(m.vision for m in pm.physical_models)
            ]
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "PDF_NOT_SUPPORTED",
                    "message": (
                        f"Pseudo-model '{pseudo_model_name}' has no vision-capable "
                        f"physical models. PDFs are treated as images in v1 and "
                        f"require a vision model. The request was rejected — "
                        f"no content was silently lost."
                    ),
                    "remediation": [
                        f"Switch to a vision-capable pseudo-model: {vision_pseudos}",
                        "Extract text from the PDF before sending (manual, or v2 feature)",
                    ],
                },
            )

    # ---- CHECK: Video → not supported in v1 ----
    if turn_caps.has_video:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "VIDEO_NOT_SUPPORTED",
                "message": (
                    "Video content is not supported in any pseudo-model in v1. "
                    "The request was rejected rather than silently dropping the video."
                ),
                "remediation": [
                    "Extract key frames as images and send those instead",
                    "Video frame extraction is planned for v2",
                ],
            },
        )

    # ---- CHECK: Parallel tools → model without parallel support ----
    if turn_caps.has_parallel_tools:
        has_parallel_models = any(m.parallel_tools for m in physical_models)
        if not has_parallel_models:
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
