"""Content validation for incoming requests.

Checks that the current pseudo-model can handle the content type
in the incoming request. Unsupported content is routed to the Blob
Vault for transformation (images, audio, PDF) or rejected (video).

python.md §4: pure functions with explicit Result monad.
python.md §3: errors as data in domain, exceptions at boundary.
"""

from src.domain.capabilities import TurnCapabilities


def validate_physical_model_content(
    turn_caps: TurnCapabilities,
    physical_model,
) -> dict | None:
    """Validate if a SPECIFIC physical model can handle the incoming content.

    Used AFTER physical model selection to determine if content delegation
    (image description, audio transcription, etc.) is needed.

    Args:
        turn_caps: Detected capabilities in the turn
        physical_model: The specific PhysicalModelSchema that will handle the request

    Returns:
      - None if the physical model can handle all content
      - {"action": "transform_unsupported"} if content needs delegation
    """
    checks = [
        ("has_images", "vision"),
        ("has_audio", "audio"),
        ("has_pdf", "pdf"),
        ("has_documents", "documents"),
        ("has_video", "video"),
    ]

    for cap_attr, phys_attr in checks:
        if getattr(turn_caps, cap_attr, False):
            if not getattr(physical_model, phys_attr, False):
                return {"action": "transform_unsupported"}

    return None
