"""Tool and content detection utilities.

- Finds tools compatible with image content (delegation)
- Stores base64 blobs in Valkey with auto-description via cheap vision model
- Replaces unsupported content with [BLOB:hash:mime:description] references
"""

import base64
import hashlib
import logging
import re
from typing import Any

import fitz  # PyMuPDF — mandatory dependency for PDF text extraction

logger = logging.getLogger(__name__)

BLOB_PREFIX = "BLOB"
BLOB_TTL = 86400  # 24 hours
_MAX_BLOB_SIZE = 10 * 1024 * 1024  # 10 MB
_CONTENT_PREVIEW_MAX = 500  # max chars for content preview in description


def _hash_content(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def _extract_mime(data_uri: str) -> str | None:
    """Extract MIME from data URI: 'data:image/png;base64,...' → 'image/png'."""
    match = re.match(r"data:([a-z]+/[a-z0-9+.-]+)", data_uri)
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
                if (
                    isinstance(param_schema, dict)
                    and param_schema.get("type") == "string"
                ):
                    return (name, param_name)
    return None


def _find_model_with_capability(config, cap: str = "vision") -> str | None:
    """Find any configured physical model with a given capability.

    Provider-agnostic: scans ALL pseudo-models for the first model
    with the required capability (vision, audio, etc.).

    TODO: This currently returns the FIRST model found, not the cheapest.
    For production, sort by cost/performance and pick the most economical
    one. E.g. for descriptions, a fast cheap model like Groq's
    llama-4-scout is better than an expensive reasoning model.
    Cost data is in pseudo_models.yaml notes for each physical model.
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


async def _call_model_for_content(
    raw_data: str, prompt: str, model_name: str, max_tokens: int, content_type: str
) -> str:
    """Call a model to describe content and extract the text response."""
    from src.adapters.litellm import call_litellm

    content_parts: list[dict] = [{"type": "text", "text": prompt}]
    if content_type == "image":
        content_parts.append(
            {"type": "image_url", "image_url": {"url": raw_data}}
        )
    elif content_type == "audio":
        content_parts.append(
            {"type": "input_audio", "input_audio": {"data": raw_data}}
        )

    response = await call_litellm(
        model=model_name, messages=[{"role": "user", "content": content_parts}],
        max_tokens=max_tokens, temperature=0.1,
    )
    resp = (
        response.model_dump() if hasattr(response, "model_dump") else response
    )
    if isinstance(resp, dict):
        choices = resp.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")[:500]
    return ""


async def _describe_content(raw_data: str, mime: str, config) -> str:
    """Generate a brief description of any content using the cheapest capable model.

    For images: uses a vision model.
    For audio: uses an audio/whisper model for transcription.
    For files/PDF: uses a vision model if available, otherwise extracts metadata.

    Provider-agnostic: the model is selected from config based on capability,
    not hardcoded.
    """
    if "image" in mime:
        prompt_key = "image"
    elif "audio" in mime:
        prompt_key = "audio"
    else:
        prompt_key = "file"
    prompt = _DESCRIBE_PROMPTS.get(prompt_key, _DESCRIBE_PROMPTS["file"])

    # Image → use any vision model
    if prompt_key == "image":
        model_name = _find_model_with_capability(config, "vision")
        if not model_name:
            return ""
        try:
            return await _call_model_for_content(
                raw_data, prompt, model_name, max_tokens=200, content_type="image"
            )
        except Exception as exc:
            logger.warning(
                "blob_describe_failed model=%s error=%s", model_name, str(exc)
            )
            return ""

    # Audio → use any audio model (whisper)
    if prompt_key == "audio":
        model_name = _find_model_with_capability(config, "audio")
        if not model_name:
            return ""
        try:
            return await _call_model_for_content(
                raw_data, prompt, model_name, max_tokens=300, content_type="audio"
            )
        except Exception as exc:
            logger.warning(
                "blob_describe_failed model=%s error=%s", model_name, str(exc)
            )
            return ""

    # Files/PDF → try Python text extraction (no model needed)
    if "pdf" in mime:
        return _try_extract_pdf_text(raw_data)

    return ""


def _try_extract_pdf_text(base64_data: str) -> str:
    """Extract text from a base64-encoded PDF using PyMuPDF.

    PyMuPDF is a mandatory dependency — checked at startup.

    Returns extracted text (truncated with ...{N} more if large),
    or a message explaining that PyMuPDF could not extract text
    (e.g., scanned/image-based PDF), so the model can decide
    whether to use a vision tool to process it further.
    """
    try:
        pdf_bytes = base64.b64decode(base64_data.split(",", 1)[-1].strip())
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page_count = len(doc)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()

        text = text.strip()
        size_kb = len(pdf_bytes) // 1024
        meta = f"[PDF: {page_count} pages, {size_kb} KB."

        if text:
            if len(text) > _CONTENT_PREVIEW_MAX:
                remainder = len(text) - _CONTENT_PREVIEW_MAX
                preview = text[:_CONTENT_PREVIEW_MAX]
                return f"{meta}\n\n{preview}\n...{{{remainder} more chars}}]"
            return f"{meta}\n\n{text}]"

        return f"{meta} PyMuPDF could not extract text — the PDF may be scanned or image-based. Try a vision model or OCR tool.]"
    except Exception as exc:
        return f"[PDF — could not parse: {str(exc)[:100]}]"


async def _store_blob_with_description(
    valkey, blob_key: str, desc_key: str, raw_data: str, mime: str, config
) -> str:
    """Store base64 blob in Valkey and generate a text description.

    Uses SETNX for the description key to avoid duplicate LLM calls
    when concurrent requests process the same blob hash.
    """
    if len(raw_data) > _MAX_BLOB_SIZE:
        logger.warning("blob_too_large key=%s size=%d", blob_key, len(raw_data))
        return ""

    # Check if description already exists (fast path)
    try:
        existing = await valkey.get(desc_key)
        if existing:
            return existing
    except Exception:
        pass

    # Store raw blob data first
    try:
        await valkey.set(blob_key, raw_data, ex=BLOB_TTL)
    except Exception:
        return ""

    # Generate description
    description = await _describe_content(raw_data, mime, config)
    description = description.strip().replace("\n", " ")[:500]

    # Use SETNX so only the first writer persists — prevents duplicate LLM calls
    if description:
        try:
            await valkey.setnx(desc_key, description)
            # Set TTL on the description key (setnx doesn't support ex)
            await valkey.expire(desc_key, BLOB_TTL)
        except Exception:
            pass

    return description


async def _process_content_parts(
    content: list[dict[str, Any]],
    prefix: str,
    valkey,
    config,
) -> list[dict[str, Any]]:
    """Process content parts of a user message, replacing base64 with blob refs."""
    new_parts: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            new_parts.append(part)
            continue

        ptype = part.get("type", "")
        raw, label = _extract_blob_info(part, ptype)

        if not raw or not raw.startswith("data:"):
            new_parts.append(part)
            continue

        # Base64 → store as blob reference
        h = _hash_content(raw)
        mime = _extract_mime(raw) or f"{ptype}/unknown"
        blob_key = f"{prefix}:{h}"
        desc_key = f"{prefix}:{h}:desc"
        size_kb = len(raw) // 1024

        description = await _store_blob_with_description(
            valkey, blob_key, desc_key, raw, mime, config
        )

        ref = f"{BLOB_PREFIX}:{h}:{mime}"
        text = f"[The user sent an {label}. blob: {ref} | {size_kb} KB"
        if description:
            text += f"\n{description}"
        text += "]"
        new_parts.append({"type": "text", "text": text})

    return new_parts


def _extract_blob_info(part: dict, ptype: str) -> tuple[str, str]:
    """Extract raw data and a human-readable label from a content part."""
    raw: str = ""
    label: str = ""

    if ptype == "image_url":
        raw = part.get("image_url", {}).get("url", "")
        label = "image"
    elif ptype == "input_audio":
        audio = part.get("input_audio", {})
        raw = audio.get("data", "") or audio.get("url", "")
        label = "audio file"
    elif ptype == "file":
        file_data = part.get("file", {})
        raw = file_data.get("data", "") or file_data.get("url", "")
        label = "file"

    return raw, label


async def replace_base64_with_blob_refs(
    messages: list[dict[str, Any]],
    conversation_id: str | None = None,
    valkey=None,
    config=None,
) -> list[dict[str, Any]]:
    """Replace base64 content parts with [BLOB:hash:mime] references.

    Stores the actual base64 data in Valkey, generates a brief description
    using the cheapest capable model, and includes it in the reference so
    the main model knows what the content contains without needing vision.

    Real URLs pass through unchanged so the model can download them.

    Format examples:
      [The user sent an image. blob: BLOB:hash:image/png | screenshot of login]
      [The user sent an audio file. blob: BLOB:hash:audio/wav | meeting notes...]
      [PDF: 12 pages, 240 KB.
       Introduction to machine learning...
       ...{3450 more chars}]
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

        processed = await _process_content_parts(content, prefix, valkey, config)
        new_messages.append({**msg, "content": processed})

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
                new_content.append(
                    {
                        "type": "text",
                        "text": (
                            f"[Image path delegated to tool '{tool_name}' "
                            f"as parameter '{param_name}']: {url}"
                        ),
                    }
                )
            else:
                new_content.append(part)
        new_messages.append({**msg, "content": new_content})
    return new_messages
