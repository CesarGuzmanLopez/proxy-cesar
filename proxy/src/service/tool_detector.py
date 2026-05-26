"""Content handling utilities for unsupported content types.

When the user sends content (images, audio, files) to a model that lacks
the required capability, the proxy stores the base64 data as a blob in
Valkey and replaces it with a text reference containing size, type, and
a brief description so the model knows what the user sent.
"""

import base64
import hashlib
import logging
import re
from typing import Any

import fitz  # PyMuPDF — mandatory for PDF text extraction

logger = logging.getLogger(__name__)

BLOB_PREFIX = "BLOB"
BLOB_TTL = 86400  # 24 hours
_MAX_BLOB_SIZE = 10 * 1024 * 1024  # 10 MB
_CONTENT_PREVIEW_MAX = 500  # max chars for content preview

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
}


def _hash_content(data: str) -> str:
    """SHA-256 hash (16 hex chars) for deduplication."""
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def _extract_mime(data_uri: str) -> str | None:
    """Extract MIME from data URI: 'data:image/png;base64,...' → 'image/png'."""
    match = re.match(r"data:([a-z]+/[a-z0-9+.-]+)", data_uri)
    return match.group(1) if match else None


def _find_model_with_capability(config, cap: str = "vision") -> str | None:
    """Find any configured physical model with a given capability.

    TODO: pick the cheapest, not the first. Cost data in pseudo_models.yaml notes.
    """
    for pm in config.pseudo_models.values():
        for phys in pm.physical_models:
            if getattr(phys, cap, False):
                return phys.model
    return None


async def _describe_content(raw_data: str, mime: str, config) -> str:
    """Generate a brief description using the cheapest capable model."""
    prompt_key = "image" if "image" in mime else ("audio" if "audio" in mime else None)
    if prompt_key is None:
        return ""

    prompt = _DESCRIBE_PROMPTS.get(prompt_key, "")
    cap = "vision" if prompt_key == "image" else "audio"
    model = _find_model_with_capability(config, cap)
    if not model:
        return ""

    try:
        from src.adapters.litellm import call_litellm

        content_block = (
            {"type": "image_url", "image_url": {"url": raw_data}}
            if prompt_key == "image"
            else {"type": "input_audio", "input_audio": {"data": raw_data}}
        )
        msg = {"role": "user", "content": [{"type": "text", "text": prompt}, content_block]}
        response = await call_litellm(model=model, messages=[msg], max_tokens=200, temperature=0.1)
        resp = response.model_dump() if hasattr(response, "model_dump") else response
        if isinstance(resp, dict):
            choices = resp.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")[:500]
    except Exception as exc:
        logger.warning("blob_describe_failed model=%s error=%s", model, str(exc))
    return ""


def _try_extract_pdf_text(base64_data: str) -> str:
    """Extract text from a base64-encoded PDF using PyMuPDF (mandatory dep)."""
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
                return f"{meta}\n\n{text[:_CONTENT_PREVIEW_MAX]}\n...{{{remainder} more chars}}]"
            return f"{meta}\n\n{text}]"
        return f"{meta} PyMuPDF could not extract text — scanned or image-based PDF.]"
    except Exception as exc:
        return f"[PDF could not parse: {str(exc)[:100]}]"


async def _store_blob_with_description(
    valkey, blob_key: str, desc_key: str, raw_data: str, mime: str, ptype: str, config
) -> str:
    """Store base64 blob in Valkey and generate a text description (idempotent via SETNX)."""
    if len(raw_data) > _MAX_BLOB_SIZE:
        logger.warning("blob_too_large key=%s", blob_key)
        return ""

    existing = await valkey.get(desc_key) if valkey else None
    if existing:
        return existing

    try:
        await valkey.set(blob_key, raw_data, ex=BLOB_TTL)
    except Exception:
        return ""

    description = await _describe_content(raw_data, mime, config)
    description = description.strip().replace("\n", " ")[:500]
    if description:
        try:
            await valkey.setnx(desc_key, description)
            await valkey.expire(desc_key, BLOB_TTL)
        except Exception:
            pass
    return description


async def replace_base64_with_blob_refs(
    messages: list[dict[str, Any]],
    conversation_id: str | None = None,
    valkey=None,
    config=None,
) -> list[dict[str, Any]]:
    """Replace base64 content parts with [BLOB:hash:mime] references.

    Real URLs pass through unchanged so the model can download them.
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
            raw: str = ""
            label = ""

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

            if not raw or not raw.startswith("data:"):
                new_content.append(part)
                continue

            h = _hash_content(raw)
            mime = _extract_mime(raw) or f"{ptype}/unknown"
            size_kb = len(raw) // 1024
            description = await _store_blob_with_description(
                valkey, f"{prefix}:{h}", f"{prefix}:{h}:desc", raw, mime, ptype, config
            )

            text = f"[The user sent an {label}. blob: {BLOB_PREFIX}:{h}:{mime} | {size_kb} KB"
            if description:
                text += f"\n{description}"
            text += "]"
            new_content.append({"type": "text", "text": text})

        new_messages.append({**msg, "content": new_content})

    return new_messages
