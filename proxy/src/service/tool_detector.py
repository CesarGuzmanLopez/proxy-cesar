"""Content handling utilities for unsupported content types.

When the user sends content (images, audio, files) to a model that lacks
the required capability, the proxy stores the base64 data as a blob in
Valkey and replaces it with a text reference containing size, type, and
a brief description so the model knows what the user sent.

Images from the same message are described together in a single vision
model call, using the user's original text prompt for context.
"""

import asyncio
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
_CONTENT_PREVIEW_MAX = 500


def _hash_content(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def _extract_mime(data_uri: str) -> str | None:
    match = re.match(r"data:([a-z]+/[a-z0-9+.-]+)", data_uri)
    return match.group(1) if match else None


def _find_model_with_capability(config, cap: str = "vision"):
    """Find any configured physical model with a given capability.

    Returns the full ``PhysicalModelSchema`` object (not just the model string)
    so callers can access ``api_base`` and ``api_key_env`` for custom endpoints.

    TODO: pick the cheapest, not the first. Cost data in pseudo_models.yaml notes.
    """
    for pm in config.pseudo_models.values():
        for phys in pm.physical_models:
            if getattr(phys, cap, False):
                return phys
    return None


def _extract_user_text(content: list[dict]) -> str:
    """Join all text parts from a user message for context."""
    return " ".join(
        str(p.get("text", "")) for p in content
        if isinstance(p, dict) and p.get("type") == "text"
    )


def _resolve_api_key(phys) -> str | None:
    """Resolve API key from environment if the physical model has api_key_env set."""
    if not phys or not phys.api_key_env:
        return None
    import os
    return os.environ.get(phys.api_key_env) or None


async def _describe_image_batch(
    images: list[tuple[str, str]], user_prompt: str, config
) -> list[str]:
    """Describe multiple images together in one vision model call.

    Sends ALL images + the user's text prompt together so the vision
    model gives contextual descriptions. Returns one per image.
    """
    vision_phys = _find_model_with_capability(config, "vision")
    if not vision_phys:
        return [""] * len(images)
    if not images:
        return []

    vision_model = vision_phys.model
    system = (
        "You receive images that the user sent to a model without vision. "
        "Describe each briefly (1-2 sentences). "
        "Return a JSON array of strings in image order. "
        'Example: ["A login screen", "A bar chart of Q1 sales"]'
    )
    content: list[dict] = [{
        "type": "text",
        "text": f"User: {user_prompt}\nDescribe each image below in JSON array format:"
    }]
    for _, raw in images:
        content.append({"type": "image_url", "image_url": {"url": raw}})

    try:
        from src.adapters.litellm import call_litellm
        response = await call_litellm(
            model=vision_model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
            api_base=vision_phys.api_base or None,
            api_key=_resolve_api_key(vision_phys),
            max_tokens=500 * len(images),
            temperature=0.1,
        )
        resp = response.model_dump() if hasattr(response, "model_dump") else response
        if isinstance(resp, dict):
            text = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
            if text:
                text = text.strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                import json
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return [str(d)[:500] for d in parsed]
    except Exception as exc:
        logger.warning("batch_describe_failed model=%s error=%s", vision_model, str(exc))
    return [""] * len(images)


async def _transcribe_audio(raw_data: str, config) -> str:
    """Transcribe audio using Whisper via litellm.atranscription()."""
    audio_phys = _find_model_with_capability(config, "audio")
    if not audio_phys or not raw_data.startswith("data:"):
        return ""
    audio_model = audio_phys.model
    try:
        audio_bytes = base64.b64decode(raw_data.split(",", 1)[-1])
        from litellm import atranscription
        from io import BytesIO
        audio_file = BytesIO(audio_bytes)
        audio_file.name = "audio.wav"
        response = await atranscription(
            model=audio_model,
            file=audio_file,
            api_base=audio_phys.api_base or None,
            api_key=_resolve_api_key(audio_phys),
            temperature=0.1,
        )
        return (getattr(response, "text", None) or "")[:500]
    except Exception as exc:
        logger.warning("audio_transcribe_failed model=%s error=%s", audio_model, str(exc))
        return ""


async def _try_extract_pdf_text(base64_data: str) -> str:
    """Extract text from a base64-encoded PDF using PyMuPDF (offloaded to thread pool)."""
    def _sync_extract() -> str:
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
    return await asyncio.to_thread(_sync_extract)


def _classify_content_parts(  # noqa: S3776 — 3 content types × multiple checks
    content: list,
) -> tuple[list[str], list[tuple[str, str, str, str]], list[tuple[str, str, str, str]], list[tuple[str, str, str, str]], list[dict]]:
    """Classify content parts into images, audio, files, and others."""
    user_text = _extract_user_text(content)
    images: list[tuple[str, str, str, str]] = []
    audios: list[tuple[str, str, str, str]] = []
    files: list[tuple[str, str, str, str]] = []
    others: list[dict] = []

    for part in content:
        if not isinstance(part, dict):
            others.append(part)
            continue
        ptype = part.get("type", "")
        raw = ""
        if ptype == "image_url":
            raw = part.get("image_url", {}).get("url", "")
        elif ptype == "input_audio":
            raw = (part.get("input_audio", {}) or {}).get("data", "")
        elif ptype == "file":
            raw = (part.get("file", {}) or {}).get("data", "")
        else:
            others.append(part)
            continue
        if not raw or not raw.startswith("data:"):
            others.append(part)
            continue
        h = _hash_content(raw)
        mime = _extract_mime(raw) or f"{ptype}/unknown"
        sz = str(len(raw) // 1024)
        info = (h, raw, mime, sz)
        if ptype == "image_url":
            images.append(info)
        elif ptype == "input_audio":
            audios.append(info)
        elif ptype == "file":
            files.append(info)
    return user_text, images, audios, files, others


def _build_blob_output(others, images, descs, audios, aresults, files, fresults):
    """Build output content list from classified parts and descriptions."""
    out: list[dict[str, Any]] = list(others)

    def _bt(h, mime, sz, label, desc=""):
        t = f"[The user sent an {label}. blob: {BLOB_PREFIX}:{h}:{mime} | {sz} KB"
        if desc:
            t += f"\n{desc}"
        t += "]"
        return {"type": "text", "text": t}

    for (h, _, mime, sz), d in zip(images, descs):
        out.append(_bt(h, mime, sz, "image", d))
    for (h, _, mime, sz), d in zip(audios, aresults):
        out.append(_bt(h, mime, sz, "audio file", d))
    for (h, _, mime, sz), d in zip(files, fresults):
        out.append(_bt(h, mime, sz, "file", d))
    return out


async def _process_msg_blobs(
    msg: dict[str, Any],
    prefix: str,
    valkey,
    config,
) -> list[dict[str, Any]]:
    """Process one user message: classify, store, describe, build output."""
    content = msg.get("content", "")
    if not isinstance(content, list):
        return [msg]

    # Pass 1: classify and store
    user_text, image_blobs, audio_blobs, file_blobs, other_parts = _classify_content_parts(content)
    store_tasks = [_store_blob_if_missing(valkey, f"{prefix}:{h}", r) for h, r, _, _ in image_blobs + audio_blobs + file_blobs]
    if store_tasks:
        await asyncio.gather(*store_tasks)

    # Pass 2: describe
    descriptions = await _describe_images(valkey, prefix, image_blobs, user_text, config)
    audio_results = await asyncio.gather(*[
        _describe_audio(valkey, f"{prefix}:{h}:desc", r, config) for h, r, _, _ in audio_blobs
    ]) if audio_blobs else []
    pdf_results = await asyncio.gather(*[
        _describe_pdf(valkey, f"{prefix}:{h}:desc", r) for h, r, _, _ in file_blobs
    ]) if file_blobs else []

    # Pass 3: build output
    out = _build_blob_output(other_parts, image_blobs, descriptions, audio_blobs, audio_results, file_blobs, pdf_results)
    return [{**msg, "content": out}]


async def _describe_images(valkey, prefix, blobs, user_text, config):
    """Batch describe images or return cached descriptions."""
    if not blobs:
        return []
    cached = await asyncio.gather(*[
        _get_cached(valkey, f"{prefix}:{h}:desc") for h, _, _, _ in blobs
    ])
    if all(cached):
        return cached
    descs = await _describe_image_batch([(h, r) for h, r, _, _ in blobs], user_text, config)
    while len(descs) < len(blobs):
        descs.append("")
    store = [_store_desc(valkey, f"{prefix}:{h}:desc", d) for (h, _, _, _), d in zip(blobs, descs) if d]
    if store:
        await asyncio.gather(*store)
    return descs


async def _get_cached(valkey, key: str) -> str:
    try:
        v = await valkey.get(key)
        return v or ""
    except Exception:
        return ""


async def replace_base64_with_blob_refs(
    messages: list[dict[str, Any]],
    conversation_id: str | None = None,
    valkey=None,
    config=None,
) -> list[dict[str, Any]]:
    """Replace base64 content with [BLOB:hash:mime] references.

    Groups images from the same message and describes them together
    using the user's original prompt for context.
    Real URLs pass through unchanged.
    """
    if valkey is None:
        return messages
    prefix = f"blob:{conversation_id or 'anon'}"
    results: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") != "user":
            results.append(msg)
        else:
            results.extend(await _process_msg_blobs(msg, prefix, valkey, config))
    return results


async def _store_blob_if_missing(valkey, key: str, raw: str) -> None:
    """Store blob only if it doesn't exist yet."""
    try:
        exists = await valkey.exists(key)
        if not exists:
            await valkey.set(key, raw, ex=BLOB_TTL)
    except Exception:
        pass


async def _store_desc(valkey, key: str, desc: str) -> None:
    """Store description with SETNX + EXPIRE."""
    try:
        await valkey.setnx(key, desc)
        await valkey.expire(key, BLOB_TTL)
    except Exception:
        pass


async def _describe_audio(valkey, desc_key: str, raw: str, config) -> str:
    """Transcribe audio if not already cached."""
    try:
        cached = await valkey.get(desc_key)
        if cached:
            return cached
    except Exception:
        pass
    desc = await _transcribe_audio(raw, config)
    if desc:
        await _store_desc(valkey, desc_key, desc)
    return desc


async def _describe_pdf(valkey, desc_key: str, raw: str) -> str:
    """Extract PDF text if not already cached."""
    try:
        cached = await valkey.get(desc_key)
        if cached:
            return cached
    except Exception:
        pass
    desc = await _try_extract_pdf_text(raw)
    if desc:
        await _store_desc(valkey, desc_key, desc)
    return desc
