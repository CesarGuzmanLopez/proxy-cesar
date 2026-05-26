"""Tool and content detection utilities.

- Finds tools compatible with image content (delegation)
- Stores base64 blobs in Valkey with auto-description via cheap vision model
- Replaces unsupported content with [BLOB:hash:mime:description] references
"""

import hashlib
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

BLOB_PREFIX = "BLOB"
BLOB_TTL = 86400  # 24 hours
_MAX_BLOB_SIZE = 10 * 1024 * 1024  # 10 MB
_DESCRIBE_PROMPT = (
    "Describe this image in one brief paragraph (max 3 sentences). "
    "Focus on what a developer would need to know: "
    "what is shown, any text/code visible, and the overall context."
)


def _hash_content(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def _extract_mime(data_uri: str) -> str | None:
    """Extract MIME from data URI: 'data:image/png;base64,...' → 'image/png'."""
    match = re.match(r"data:([a-z]+/[a-z0-9+-.]+)", data_uri)
    return match.group(1) if match else None


def find_image_compatible_tool(tools: list[dict] | None) -> tuple[str, str] | None:
    """Find first tool with a string parameter that could accept an image path."""
    if not tools:
        return None
    for tool in tools:
        func = tool.get("function", {}) if isinstance(tool, dict) else {}
        name = func.get("name", "")
        params = func.get("parameters", {}).get("properties", {})
        if isinstance(params, dict):
            for param_name, param_schema in params.items():
                if isinstance(param_schema, dict) and param_schema.get("type") == "string":
                    return (name, param_name)
    return None


def _find_model_with_capability(config, cap: str = "vision") -> str | None:
    """Find any configured physical model with a given capability.

    Provider-agnostic: scans ALL pseudo-models for the first model
    with the required capability (vision, audio, etc.).
    """
    for pm in config.pseudo_models.values():
        for phys in pm.physical_models:
            if getattr(phys, cap, False):
                return phys.model
    return None


_DESCRIBE_PROMPTS: dict[str, str] = {
    "image": (
        "Describe this image in one brief paragraph (max 3 sentences). "
        "Focus on what a developer would need to know: "
        "what is shown, any text/code visible, and the overall context."
    ),
    "audio": (
        "Transcribe and summarize this audio in one paragraph. "
        "Include any key points, decisions, or action items mentioned."
    ),
    "file": (
        "Summarize this document in one paragraph. "
        "Include the document type, key sections, and any actionable content."
    ),
}


async def _describe_content(
    raw_data: str, mime: str, config, ptype: str
) -> str:
    """Generate a brief description of any content using the cheapest capable model.

    For images: uses a vision model.
    For audio: uses an audio/whisper model for transcription.
    For files/PDF: uses a vision model if available, otherwise extracts metadata.

    Provider-agnostic: the model is selected from config based on capability,
    not hardcoded.
    """
    prompt_key = "image" if "image" in mime else ("audio" if "audio" in mime else "file")
    prompt = _DESCRIBE_PROMPTS.get(prompt_key, _DESCRIBE_PROMPTS["file"])

    # Image → use any vision model
    if prompt_key == "image":
        vision_model = _find_model_with_capability(config, "vision")
        if not vision_model:
            return ""
        try:
            from src.adapters.litellm import call_litellm

            msg = {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": raw_data}},
                ],
            }
            response = await call_litellm(
                model=vision_model, messages=[msg], max_tokens=200, temperature=0.1
            )
            resp = response.model_dump() if hasattr(response, "model_dump") else response
            if isinstance(resp, dict):
                choices = resp.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "")[:500]
            return ""
        except Exception as exc:
            logger.warning("blob_describe_failed model=%s error=%s", vision_model, str(exc))
            return ""

    # Audio → use any audio model (whisper)
    if prompt_key == "audio":
        audio_model = _find_model_with_capability(config, "audio")
        if not audio_model:
            return ""
        try:
            from src.adapters.litellm import call_litellm

            msg = {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "input_audio", "input_audio": {"data": raw_data}},
                ],
            }
            response = await call_litellm(
                model=audio_model, messages=[msg], max_tokens=300, temperature=0.1
            )
            resp = response.model_dump() if hasattr(response, "model_dump") else response
            if isinstance(resp, dict):
                choices = resp.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "")[:500]
            return ""
        except Exception as exc:
            logger.warning("blob_describe_failed model=%s error=%s", audio_model, str(exc))
            return ""

    # Files/PDF → try Python text extraction (no model needed)
    # If extraction fails, metadata is returned so the model can decide
    # whether to use a tool (vision, convert to images, etc.)
    if "pdf" in mime:
        return _try_extract_pdf_text(raw_data)

    return ""


def _try_extract_pdf_text(base64_data: str) -> str:
    """Try to extract text from a base64-encoded PDF using Python libraries.

    Returns the extracted text if successful, or metadata about the PDF
    (page count, size) if text extraction fails, so the model can decide
    whether to use a tool to process it further.
    """
    try:
        import fitz  # PyMuPDF
        import base64

        pdf_bytes = base64.b64decode(base64_data.split(",", 1)[-1])
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page_count = len(doc)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()

        text = text.strip()
        if text:
            return text[:1000]
        return f"[PDF with {page_count} pages — no extractable text found. Consider using a vision model or OCR tool.]"
    except ImportError:
        import base64
        pdf_bytes = base64.b64decode(base64_data.split(",", 1)[-1])
        size_kb = len(pdf_bytes) // 1024
        return f"[PDF file ({size_kb} KB). PyMuPDF not available for text extraction. Use a tool to process this PDF.]"
    except Exception as exc:
        return f"[PDF file — could not parse: {str(exc)[:100]}]"


async def _store_blob_with_description(
    valkey, blob_key: str, desc_key: str, raw_data: str, mime: str, ptype: str, config
) -> str:
    """Store base64 blob in Valkey and generate a text description.

    The description is generated using the cheapest available model
    that can handle the content type (vision for images, audio for
    speech, etc.). Provider-agnostic — never hardcodes a provider.

    If the blob was already stored (same hash), reuse existing description.
    Returns the description text (empty string if generation fails).
    """
    if len(raw_data) > _MAX_BLOB_SIZE:
        logger.warning("blob_too_large key=%s size=%d", blob_key, len(raw_data))
        return ""

    # Check if already stored (deduplicate by hash)
    try:
        existing = await valkey.get(desc_key)
        if existing:
            return existing
    except Exception:
        pass

    # Store the raw blob data
    try:
        await valkey.set(blob_key, raw_data, ex=BLOB_TTL)
    except Exception:
        return ""

    # Generate description using the cheapest capable model
    description = await _describe_content(raw_data, mime, config, ptype)
    description = description.strip().replace("\n", " ")[:500]

    # Store description
    if description:
        try:
            await valkey.set(desc_key, description, ex=BLOB_TTL)
        except Exception:
            pass

    return description


async def replace_base64_with_blob_refs(
    messages: list[dict[str, Any]],
    conversation_id: str | None = None,
    valkey=None,
    config=None,
) -> list[dict[str, Any]]:
    """Replace base64 content parts with [BLOB:hash:mime:description] references.

    Stores the actual base64 data in Valkey, generates a brief description
    using a cheap vision model, and includes it in the reference so the
    main model knows what the content contains without needing vision.

    The model receives:
      [The user sent an image. blob: BLOB:hash:mime | description: ...]

    Real URLs (non-base64) pass through unchanged.
    """
    if valkey is None:
        return messages

    prefix = f"blob:{conversation_id or 'anon'}"
    new_messages: list[dict[str, Any]] = []

    for msg in messages:
        if msg.get("role") != "user":
            new_messages.append(msg)
            continue

        content = msg.get("content", "")
        if not isinstance(content, list):
            new_messages.append(msg)
            continue

        new_content: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict):
                new_content.append(part)
                continue

            ptype = part.get("type", "")

            if ptype == "image_url":
                raw = part.get("image_url", {}).get("url", "")
                if raw.startswith("data:"):
                    h = _hash_content(raw)
                    mime = _extract_mime(raw) or "image/unknown"
                    blob_key = f"{prefix}:{h}"
                    desc_key = f"{prefix}:{h}:desc"
                    description = await _store_blob_with_description(
                        valkey, blob_key, desc_key, raw, config
                    )
                    ref = f"{BLOB_PREFIX}:{h}:{mime}"
                    text = f"[The user sent an image. blob: {ref}"
                    if description:
                        text += f" | {description}"
                    text += "]"
                    new_content.append({"type": "text", "text": text})
                else:
                    new_content.append(part)

            elif ptype == "input_audio":
                audio = part.get("input_audio", {})
                raw = audio.get("data", "")
                if raw.startswith("data:"):
                    h = _hash_content(raw)
                    mime = _extract_mime(raw) or "audio/unknown"
                    blob_key = f"{prefix}:{h}"
                    desc_key = f"{prefix}:{h}:desc"
                    description = await _store_blob_with_description(
                        valkey, blob_key, desc_key, raw, config
                    )
                    ref = f"{BLOB_PREFIX}:{h}:{mime}"
                    text = f"[The user sent an audio file. blob: {ref}"
                    if description:
                        text += f" | {description}"
                    text += "]"
                    new_content.append({"type": "text", "text": text})
                else:
                    new_content.append(part)

            elif ptype == "file":
                file_data = part.get("file", {})
                raw = file_data.get("data", "")
                if raw.startswith("data:"):
                    h = _hash_content(raw)
                    mime = _extract_mime(raw) or "application/octet-stream"
                    blob_key = f"{prefix}:{h}"
                    desc_key = f"{prefix}:{h}:desc"
                    description = await _store_blob_with_description(
                        valkey, blob_key, desc_key, raw, mime, ptype, config
                    )
                    ref = f"{BLOB_PREFIX}:{h}:{mime}"
                    text = f"[The user sent an image. blob: {ref}"

            elif ptype == "input_audio":
                audio = part.get("input_audio", {})
                raw = audio.get("data", "")
                if raw.startswith("data:"):
                    h = _hash_content(raw)
                    mime = _extract_mime(raw) or "audio/unknown"
                    blob_key = f"{prefix}:{h}"
                    desc_key = f"{prefix}:{h}:desc"
                    description = await _store_blob_with_description(
                        valkey, blob_key, desc_key, raw, mime, ptype, config
                    )
                    ref = f"{BLOB_PREFIX}:{h}:{mime}"
                    text = f"[The user sent an audio file. blob: {ref}"

            elif ptype == "file":
                file_data = part.get("file", {})
                raw = file_data.get("data", "")
                if raw.startswith("data:"):
                    h = _hash_content(raw)
                    mime = _extract_mime(raw) or "application/octet-stream"
                    blob_key = f"{prefix}:{h}"
                    desc_key = f"{prefix}:{h}:desc"
                    description = await _store_blob_with_description(
                        valkey, blob_key, desc_key, raw, mime, ptype, config
                    )
                    ref = f"{BLOB_PREFIX}:{h}:{mime}"
                    text = f"[The user sent a file. blob: {ref}"
                    if description:
                        text += f" | {description}"
                    text += "]"
                    new_content.append({"type": "text", "text": text})
                else:
                    new_content.append(part)

            else:
                new_content.append(part)

        new_messages.append({**msg, "content": new_content})

    return new_messages


def delegate_images_to_tool(
    messages: list[dict[str, Any]],
    tool_name: str,
    param_name: str,
) -> list[dict[str, Any]]:
    """Replace image_url content parts with text instructions for tool use."""
    new_messages: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") != "user":
            new_messages.append(msg)
            continue
        content = msg.get("content", "")
        if not isinstance(content, list):
            new_messages.append(msg)
            continue
        has_image = any(
            isinstance(p, dict) and p.get("type") == "image_url" for p in content
        )
        if not has_image:
            new_messages.append(msg)
            continue
        new_content: list[dict[str, Any]] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                new_content.append({
                    "type": "text",
                    "text": (
                        f"[Image path delegated to tool '{tool_name}' "
                        f"as parameter '{param_name}']: {url}"
                    ),
                })
            else:
                new_content.append(part)
        new_messages.append({**msg, "content": new_content})
    return new_messages
