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


def _find_model_with_capability(config, cap: str = "vision") -> str | None:
    """Find any configured physical model with a given capability.

    TODO: pick the cheapest, not the first. Cost data in pseudo_models.yaml notes.
    """
    for pm in config.pseudo_models.values():
        for phys in pm.physical_models:
            if getattr(phys, cap, False):
                return phys.model
    return None


def _extract_user_text(content: list[dict]) -> str:
    """Join all text parts from a user message for context."""
    return " ".join(
        str(p.get("text", "")) for p in content
        if isinstance(p, dict) and p.get("type") == "text"
    )


async def _describe_image_batch(
    images: list[tuple[str, str]], user_prompt: str, config
) -> list[str]:
    """Describe multiple images together in one vision model call.

    Sends ALL images + the user's text prompt together so the vision
    model gives contextual descriptions. Returns one per image.
    """
    if not _find_model_with_capability(config, "vision"):
        return [""] * len(images)
    if not images:
        return []

    vision_model = _find_model_with_capability(config, "vision")
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
    audio_model = _find_model_with_capability(config, "audio")
    if not audio_model or not raw_data.startswith("data:"):
        return ""
    try:
        audio_bytes = base64.b64decode(raw_data.split(",", 1)[-1])
        from litellm import atranscription
        from io import BytesIO
        audio_file = BytesIO(audio_bytes)
        audio_file.name = "audio.wav"
        response = await atranscription(
            model=audio_model,
            file=audio_file,
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
    Optimized with cache checks and parallel operations.
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

        # ── Pass 1: classify parts, check cache, store only new blobs ──
        user_text = _extract_user_text(content)
        image_blobs: list[tuple[str, str, str, str]] = []
        audio_blobs: list[tuple[str, str, str, str]] = []
        file_blobs: list[tuple[str, str, str, str]] = []
        other_parts: list[dict] = []
        store_tasks = []

        for part in content:
            if not isinstance(part, dict):
                other_parts.append(part)
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
                other_parts.append(part)
                continue

            if not raw or not raw.startswith("data:"):
                other_parts.append(part)
                continue

            h = _hash_content(raw)
            mime = _extract_mime(raw) or f"{ptype}/unknown"
            size_kb = str(len(raw) // 1024)
            info = (h, raw, mime, size_kb)

            # Check if blob already exists before storing
            store_tasks.append(_store_blob_if_missing(valkey, f"{prefix}:{h}", raw))

            if ptype == "image_url":
                image_blobs.append(info)
            elif ptype == "input_audio":
                audio_blobs.append(info)
            elif ptype == "file":
                file_blobs.append(info)

        if store_tasks:
            await asyncio.gather(*store_tasks)

        # ── Pass 2: describe/transcribe/extract with cache check ──
        async def _get_cached(key: str) -> str:
            try:
                v = await valkey.get(key)
                return v or ""
            except Exception:
                return ""

        # Images: batch describe (always regenerate with new context)
        descriptions: list[str] = []
        if image_blobs:
            # Check if ALL descriptions already cached (skip vision call)
            cached_descs = await asyncio.gather(*[
                _get_cached(f"{prefix}:{h}:desc") for h, _, _, _ in image_blobs
            ])
            if all(cached_descs):
                descriptions = cached_descs
            else:
                descs = await _describe_image_batch(
                    [(h, r) for h, r, _, _ in image_blobs], user_text, config
                )
                while len(descs) < len(image_blobs):
                    descs.append("")
                descriptions = descs
                # Store only new descriptions
                store = []
                for (h, _, _, _), d in zip(image_blobs, descriptions):
                    if d:
                        store.append(_store_desc(valkey, f"{prefix}:{h}:desc", d))
                if store:
                    await asyncio.gather(*store)

        # Audio: transcribe if not cached
        audio_tasks = []
        for h, raw, mime, sz in audio_blobs:
            audio_tasks.append(_describe_audio(valkey, f"{prefix}:{h}:desc", raw, config))
        audio_results = await asyncio.gather(*audio_tasks) if audio_tasks else []

        # PDF: extract text if not cached
        pdf_tasks = []
        for h, raw, mime, sz in file_blobs:
            pdf_tasks.append(_describe_pdf(valkey, f"{prefix}:{h}:desc", raw))
        pdf_results = await asyncio.gather(*pdf_tasks) if pdf_tasks else []

        # ── Pass 3: build output ──
        out: list[dict[str, Any]] = list(other_parts)

        def _blob_text(h, mime, sz, label, desc=""):
            t = f"[The user sent an {label}. blob: {BLOB_PREFIX}:{h}:{mime} | {sz} KB"
            if desc:
                t += f"\n{desc}"
            t += "]"
            return {"type": "text", "text": t}

        for i, (h, _, mime, sz) in enumerate(image_blobs):
            out.append(_blob_text(h, mime, sz, "image", descriptions[i] if i < len(descriptions) else ""))

        for (h, _, mime, sz), desc in zip(audio_blobs, audio_results):
            out.append(_blob_text(h, mime, sz, "audio file", desc))

        for (h, _, mime, sz), desc in zip(file_blobs, pdf_results):
            out.append(_blob_text(h, mime, sz, "file", desc))

        new_messages.append({**msg, "content": out})

    return new_messages


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
