"""Multimedia processing services.

Sprint 5: Image description for auto-describe on pseudo-model switch
and manual POST /degrade-images endpoint.
"""

from src.service.multimedia.image_describer import auto_describe_images, find_image_refs

__all__ = [
    "auto_describe_images",
    "find_image_refs",
]
