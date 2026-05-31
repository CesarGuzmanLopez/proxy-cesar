"""Image processing utilities: degradation, token estimation, batch sizing.

When sending images to vision models for description, we degrade quality
to reduce token consumption and split into batches when there are many
images to avoid exceeding the model's context window.
"""

import asyncio
import base64
import io
import logging
import re

from PIL import Image

from src.config.constants import (
    MAX_IMAGE_DIMENSION,
    IMAGE_JPEG_QUALITY,
    VISION_TOKENS_PER_TILE,
    VISION_LOW_DETAIL_TOKENS,
)

logger = logging.getLogger(__name__)

# Local overrides (currently same as config/constants, but can be tuned separately)
# Max safe ratio of context window to use for images
_MAX_IMAGE_TOKENS_RATIO = 0.7
# Max images per batch (safety limit even if context would fit more)
_MAX_IMAGES_PER_BATCH = 5


def _is_data_url(url: str) -> bool:
    return bool(re.match(r"^data:image/", url))


def _decode_data_url(data_url: str) -> tuple[bytes, str]:
    """Decode a data URL into raw bytes and MIME type."""
    match = re.match(r"data:(image/[a-z0-9+.-]+);base64,(.+)", data_url)
    if not match:
        raise ValueError(f"Not a valid image data URL: {data_url[:50]}...")
    mime = match.group(1)
    raw = base64.b64decode(match.group(2))
    return raw, mime


def _encode_data_url(raw: bytes, mime: str = "image/jpeg") -> str:
    """Encode raw bytes into a data URL."""
    b64 = base64.b64encode(raw).decode()
    return f"data:{mime};base64,{b64}"


def _is_svg_data_url(data_url: str) -> bool:
    """Check if a data URL contains an SVG image."""
    return bool(re.match(r"^data:image/svg\+xml", data_url))


def _convert_svg_to_png(svg_bytes: bytes) -> bytes | None:
    """Convert SVG bytes to PNG using cairosvg if available.

    Returns PNG bytes on success, None on failure.
    """
    try:
        import cairosvg  # noqa: F811

        return cairosvg.svg2png(bytestring=svg_bytes)
    except Exception:
        logger.warning("svg_conversion_error: cairosvg not available or failed")
        return None


def degrade_image(data_url: str) -> str:
    """Downscale and compress an image to reduce token consumption.

    - SVG images are converted to PNG via cairosvg first
    - Resizes to max MAX_IMAGE_DIMENSION px on the longest side (maintains ratio)
    - Converts to JPEG at IMAGE_JPEG_QUALITY quality
    - Returns a new data URL

    Args:
        data_url: Original base64 image data URL.

    Returns:
        New data URL with degraded image, or original if degradation fails.
    """
    if not _is_data_url(data_url):
        return data_url

    try:
        if _is_svg_data_url(data_url):
            raw, _mime = _decode_data_url(data_url)
            png_raw = _convert_svg_to_png(raw)
            if png_raw:
                # Re-encode as PNG data URL for PIL processing
                data_url = _encode_data_url(png_raw, "image/png")
            # If conversion failed, continue with SVG (will fail at PIL)

        raw, _mime = _decode_data_url(data_url)
        img = Image.open(io.BytesIO(raw))

        # Convert to RGB if necessary (JPEG doesn't support RGBA)
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")

        # Resize if larger than max dimension
        w, h = img.size
        if w > MAX_IMAGE_DIMENSION or h > MAX_IMAGE_DIMENSION:
            ratio = min(MAX_IMAGE_DIMENSION / w, MAX_IMAGE_DIMENSION / h)
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            img = img.resize((new_w, new_h), Image.LANCZOS)

        # Save as JPEG with quality compression
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=IMAGE_JPEG_QUALITY)
        raw = buf.getvalue()

        return _encode_data_url(raw, "image/jpeg")
    except Exception as e:
        logger.warning("image_processing_error error=%s", str(e))
        return data_url


async def degrade_image_async(data_url: str) -> str:
    """Async wrapper for degrade_image — runs PIL I/O in a thread pool.

    Use this instead of degrade_image() inside async functions to avoid
    blocking the event loop.
    """
    return await asyncio.to_thread(degrade_image, data_url)


def estimate_image_tokens(data_url: str, detail: str = "auto") -> int:
    """Estimate the token cost of an image for a vision model.

    Based on OpenAI's token calculation heuristic:
    - Low detail: VISION_LOW_DETAIL_TOKENS tokens
    - Auto/High detail: tiles = ceil(w/512) * ceil(h/512), tokens = VISION_TOKENS_PER_TILE * tiles + VISION_LOW_DETAIL_TOKENS

    Args:
        data_url: Image data URL.
        detail: Detail level ('low', 'high', 'auto').

    Returns:
        Estimated token count.
    """
    if not _is_data_url(data_url):
        return 0

    if detail == "low":
        return VISION_LOW_DETAIL_TOKENS

    try:
        raw, _mime = _decode_data_url(data_url)
        img = Image.open(io.BytesIO(raw))
        w, h = img.size

        if detail == "auto":
            # Auto: if both dimensions < 512, treat as low detail
            if w < 512 and h < 512:
                return VISION_LOW_DETAIL_TOKENS
            # Otherwise, use high detail calculation
            tiles_x = (w + 511) // 512
            tiles_y = (h + 511) // 512
            return VISION_TOKENS_PER_TILE * tiles_x * tiles_y + VISION_LOW_DETAIL_TOKENS
        else:
            # High detail
            tiles_x = (w + 511) // 512
            tiles_y = (h + 511) // 512
            return VISION_TOKENS_PER_TILE * tiles_x * tiles_y + VISION_LOW_DETAIL_TOKENS
    except Exception:
        # Conservative default for unparseable images
        return 1000


def can_batch_fit(
    image_count: int,
    estimated_tokens_per_image: int,
    context_window: int,
    text_tokens: int = 200,
) -> bool:
    """Check if N images at estimated tokens each fit in the context window.

    Args:
        image_count: Number of images.
        estimated_tokens_per_image: Token cost per image.
        context_window: Model's context window.
        text_tokens: Tokens for system prompt + user text.

    Returns:
        True if the batch fits within _MAX_IMAGE_TOKENS_RATIO of context.
    """
    total_image_tokens = image_count * estimated_tokens_per_image
    max_allowed = int(context_window * _MAX_IMAGE_TOKENS_RATIO)
    return (text_tokens + total_image_tokens) <= max_allowed
