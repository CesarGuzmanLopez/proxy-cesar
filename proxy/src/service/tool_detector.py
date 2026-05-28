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

import fitz  # PyMuPDF — mandatory for PDF text extraction

from src.config.constants import BLOB_STORAGE_TTL_SECONDS

logger = logging.getLogger(__name__)

BLOB_PREFIX = "BLOB"
BLOB_TTL = BLOB_STORAGE_TTL_SECONDS  # Configurable in src/config/constants.py
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
        str(p.get("text", ""))
        for p in content
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

    Degrades images before sending (reduces quality to save tokens).
    Checks context window capacity and splits into batches if needed.
    Returns one description per image. Returns empty strings on failure.

    If a single image exceeds the model's context capacity, returns an
    error description so the user knows the images are too large.
    """
    vision_phys = _find_model_with_capability(config, "vision")
    if not vision_phys:
        return [""] * len(images)
    if not images:
        return []

    from src.service.multimedia.image_processor import (
        can_batch_fit,
        degrade_image,
        estimate_image_tokens,
    )

    vision_model = vision_phys.model
    context_window = vision_phys.context_window or 128000
    # Degrade all images first
    degraded: list[tuple[str, str]] = []
    for h, raw in images:
        degraded_raw = degrade_image(raw)
        degraded.append((h, degraded_raw))

    system = (
        "You receive images that the user sent to a model without vision. "
        "For EACH image, analyze exhaustively for a text-only AI model:\n"
        "1. Extract ALL visible text exactly as it appears (including crossed out, "
        "strikethrough, handwritten, annotations, headers, footers, labels, watermarks). "
        "Preserve exact spelling and original language.\n"
        "2. Describe the image in rich detail: scene, objects, people, actions, "
        "expressions, clothing, environment, lighting, colors, composition, style, "
        "spatial relationships, perspective.\n"
        "3. For UI/code/diagrams: describe layout, structure, hierarchy, "
        "interactive elements, data values, axes, legends, trends.\n"
        "4. Note imperfections: blur, glare, artifacts, damage, low resolution areas.\n"
        "5. If multiple panels/sections, describe each separately.\n\n"
        "Return a JSON array of strings in image order, one full analysis per image. "
        'Format per element: "[EXTRACTED TEXT]: ...\\n[DESCRIPTION]: ...\\n[TECHNICAL DETAILS]: ..."'
    )

    # Estimate if all images fit in one batch
    sample_tokens = estimate_image_tokens(degraded[0][1], "high")
    text_tokens = len(user_prompt) // 4 + 200  # rough text estimate

    if not can_batch_fit(len(degraded), sample_tokens, context_window, text_tokens):
        # Split into smaller batches
        from src.service.multimedia.image_processor import _MAX_IMAGES_PER_BATCH

        results: list[str] = []
        for i in range(0, len(degraded), _MAX_IMAGES_PER_BATCH):
            batch = degraded[i : i + _MAX_IMAGES_PER_BATCH]
            batch_tokens = estimate_image_tokens(batch[0][1], "high")
            if not can_batch_fit(len(batch), batch_tokens, context_window, text_tokens):
                # Single batch still too large — return error descriptions
                logger.warning(
                    "batch_too_large model=%s images=%d ctx=%d",
                    vision_model,
                    len(batch),
                    context_window,
                )
                results.extend(
                    [
                        "[ERROR: Image too large to process. Reduce size or use a vision-capable model directly.]"
                    ]
                    * len(batch)
                )
                continue
            batch_descs = await _describe_single_batch(
                batch,
                user_prompt,
                system,
                vision_model,
                vision_phys,
            )
            results.extend(batch_descs)
        return results

    return await _describe_single_batch(
        degraded,
        user_prompt,
        system,
        vision_model,
        vision_phys,
    )


async def _describe_single_batch(
    images: list[tuple[str, str]],
    user_prompt: str,
    system: str,
    vision_model: str,
    vision_phys,
) -> list[str]:
    """Send a single batch of images to the vision model for description."""
    content: list[dict] = [
        {
            "type": "text",
            "text": f"User: {user_prompt}\nDescribe each image below in JSON array format:",
        }
    ]
    for _, raw in images:
        content.append({"type": "image_url", "image_url": {"url": raw}})

    try:
        from src.adapters.litellm import call_litellm

        response = await call_litellm(
            model=vision_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            api_base=vision_phys.api_base or None,
            api_key=_resolve_api_key(vision_phys),
            max_tokens=2048 * len(images),
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
                    return [str(d) for d in parsed]
    except Exception as exc:
        logger.warning(
            "batch_describe_failed model=%s images=%d error=%s",
            vision_model,
            len(images),
            str(exc),
        )
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
        logger.warning(
            "audio_transcribe_failed model=%s error=%s", audio_model, str(exc)
        )
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
            return (
                f"{meta} PyMuPDF could not extract text — scanned or image-based PDF.]"
            )
        except Exception as exc:
            return f"[PDF could not parse: {str(exc)[:100]}]"

    return await asyncio.to_thread(_sync_extract)


async def _try_extract_docx_text(base64_data: str) -> str:
    """Extract text from a base64-encoded DOCX file (offloaded to thread pool)."""

    def _sync_extract() -> str:
        try:
            from docx import Document
            from io import BytesIO

            docx_bytes = base64.b64decode(base64_data.split(",", 1)[-1].strip())
            doc = Document(BytesIO(docx_bytes))

            # Extract text from paragraphs
            text_parts = [para.text for para in doc.paragraphs if para.text.strip()]
            text = "\n".join(text_parts)
            text = text.strip()

            size_kb = len(docx_bytes) // 1024
            para_count = len(doc.paragraphs)
            table_count = len(doc.tables)

            meta = f"[DOCX: {para_count} paragraphs, {table_count} tables, {size_kb} KB."

            if text:
                if len(text) > _CONTENT_PREVIEW_MAX:
                    remainder = len(text) - _CONTENT_PREVIEW_MAX
                    return f"{meta}\n\n{text[:_CONTENT_PREVIEW_MAX]}\n...{{{remainder} more chars}}]"
                return f"{meta}\n\n{text}]"
            return f"{meta} Could not extract text from document.]"
        except ImportError:
            return "[DOCX: Could not process — python-docx library not available.]"
        except Exception as exc:
            return f"[DOCX could not parse: {str(exc)[:100]}]"

    return await asyncio.to_thread(_sync_extract)


def _classify_content_parts(  # noqa: S3776 — multiple content types × multiple formats
    content: list,
) -> tuple[
    str,
    list[tuple[str, str, str, str]],
    list[tuple[str, str, str, str]],
    list[tuple[str, str, str, str]],
    list[dict],
]:
    """Classify content parts into images, audio, files, and others.

    Supports multiple image formats:
    - OpenAI: {"type": "image_url", "image_url": {"url": "..."}}
    - Base64: {"type": "image", "image": "data:image/..."}
    - Text with data URL: {"type": "text", "text": "data:image/..."}
    """
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
        content_type = None

        # Extract raw data from various formats
        if ptype == "image_url":
            # OpenAI standard format
            raw = part.get("image_url", {}).get("url", "")
            content_type = "image"
        elif ptype == "image":
            # Base64 inline format (used by Anthropic, others)
            raw = part.get("image", "")
            content_type = "image"
        elif ptype == "text":
            # Some clients send images as text with data: URL
            text_content = part.get("text", "")
            if isinstance(text_content, str) and text_content.startswith("data:image/"):
                raw = text_content
                content_type = "image"
            else:
                # Regular text, not an image
                others.append(part)
                continue
        elif ptype == "input_audio":
            raw = (part.get("input_audio", {}) or {}).get("data", "")
            content_type = "audio"
        elif ptype == "file":
            raw = (part.get("file", {}) or {}).get("data", "")
            content_type = "file"
        else:
            others.append(part)
            continue

        # Validate and process extracted data
        if not raw or not raw.startswith("data:"):
            others.append(part)
            continue

        h = _hash_content(raw)
        mime = _extract_mime(raw) or f"{content_type}/unknown"
        sz = str(len(raw) // 1024)
        info = (h, raw, mime, sz)

        # Classify into appropriate category
        if content_type == "image":
            images.append(info)
        elif content_type == "audio":
            audios.append(info)
        elif content_type == "file":
            files.append(info)

    return user_text, images, audios, files, others


def _build_blob_output(others, images, descs, audios, aresults, files, fresults):
    """Build output content list from classified parts and descriptions.

    Each blob is sent with:
    1. Metadata: blob hash, MIME type, size
    2. Auto-extracted information (description, transcription, text)
    3. Guidance: model can use tools to extract alternative information if needed

    This allows models with specialized tools (PDF extraction, image analysis,
    audio processing) to decide whether to use the proxy's extraction or
    invoke their own tools for custom analysis.
    """
    out: list[dict[str, object]] = list(others)

    def _bt(h, mime, sz, label, desc="", extraction_method="", filename=""):
        # Header: what was sent and how to access it
        t = f"[Content provided: {label}\n"
        t += f"  blob_ref: {BLOB_PREFIX}:{h}:{mime} | size: {sz} KB"
        if filename:
            t += f" | filename: {filename}"

        # Extraction info: how it was processed
        if extraction_method:
            t += f"\n  extraction: {extraction_method}"

        # Content: the extracted/described information
        if desc:
            t += f"\n\nExtracted content:\n{desc}"
            t += "\n\nNote: If you have specialized tools for analyzing this content,"
            t += " you can:"
            t += "\n  • Use the blob reference to retrieve raw data if needed"
            t += "\n  • Apply your own analysis tools for custom extraction"
            t += "\n  • Combine your analysis with the provided extraction"
        else:
            t += "\n\nWarning: Extraction failed or produced empty result."
            t += "\n  • Check if a specialized tool is available for this content type"

        t += "\n]"
        return {"type": "text", "text": t}

    for (h, _, mime, sz), d in zip(images, descs):
        extraction = "Vision model (description via Groq/similar)"
        out.append(_bt(h, mime, sz, "image", d, extraction))

    for (h, _, mime, sz), d in zip(audios, aresults):
        extraction = "Speech-to-text (Whisper/similar transcription)"
        out.append(_bt(h, mime, sz, "audio", d, extraction))

    for (h, _, mime, sz), d in zip(files, fresults):
        # Determine extraction method based on MIME type
        if "pdf" in mime.lower():
            extraction = "PDF text extraction"
        elif "wordprocessingml" in mime.lower() or "word" in mime.lower():
            extraction = "DOCX text extraction"
        else:
            extraction = "Document text extraction"
        out.append(_bt(h, mime, sz, "document", d, extraction))

    return out


async def _process_msg_blobs(
    msg: dict[str, object],
    prefix: str,
    valkey,
    config,
) -> list[dict[str, object]]:
    """Process one user message: classify, store, describe, build output."""
    content = msg.get("content", "")
    if not isinstance(content, list):
        return [msg]

    # Pass 1: classify and store
    user_text, image_blobs, audio_blobs, file_blobs, other_parts = (
        _classify_content_parts(content)
    )
    store_tasks = [
        _store_blob_if_missing(valkey, f"{prefix}:{h}", r)
        for h, r, _, _ in image_blobs + audio_blobs + file_blobs
    ]
    if store_tasks:
        await asyncio.gather(*store_tasks)

    # Pass 2: describe
    descriptions = await _describe_images(
        valkey, prefix, image_blobs, user_text, config
    )
    audio_results = (
        await asyncio.gather(
            *[
                _describe_audio(valkey, f"{prefix}:{h}:desc", r, config)
                for h, r, _, _ in audio_blobs
            ]
        )
        if audio_blobs
        else []
    )
    pdf_results = (
        await asyncio.gather(
            *[
                _describe_pdf(valkey, f"{prefix}:{h}:desc", r)
                if "pdf" in mime.lower()
                else _describe_docx(valkey, f"{prefix}:{h}:desc", r)
                for h, r, mime, _ in file_blobs
            ]
        )
        if file_blobs
        else []
    )

    # Pass 3: build output
    out = _build_blob_output(
        other_parts,
        image_blobs,
        descriptions,
        audio_blobs,
        audio_results,
        file_blobs,
        pdf_results,
    )
    return [{**msg, "content": out}]


async def _describe_images(valkey, prefix, blobs, user_text, config):
    """Batch describe images or return cached descriptions.

    Cache key includes a hash of the user prompt so that the same image
    asked with a different question gets a different (correct) description.
    Key format: {prefix}:{hash}:desc:{prompt_hash}
    """
    if not blobs:
        return []
    prompt_hash = hashlib.sha256((user_text or "").encode()).hexdigest()[:8]
    cached = await asyncio.gather(
        *[
            _get_cached(valkey, f"{prefix}:{h}:desc:{prompt_hash}")
            for h, _, _, _ in blobs
        ]
    )
    if all(cached):
        return cached
    descs = await _describe_image_batch(
        [(h, r) for h, r, _, _ in blobs], user_text, config
    )
    while len(descs) < len(blobs):
        descs.append("")
    store = [
        _store_desc(valkey, f"{prefix}:{h}:desc:{prompt_hash}", d)
        for (h, _, _, _), d in zip(blobs, descs)
        if d
    ]
    if store:
        await asyncio.gather(*store)
    return descs


async def _get_cached(valkey, key: str) -> str:
    try:
        v = await valkey.get(key)
        return v or ""
    except Exception as exc:
        logger.debug("blob_cache_get_error key=%s err=%s", key, exc)
        return ""


async def replace_base64_with_blob_refs(
    messages: list[dict[str, object]],
    conversation_id: str | None = None,
    valkey=None,
    config=None,
) -> list[dict[str, object]]:
    """Replace base64 content with [BLOB:hash:mime] references.

    Groups images from the same message and describes them together
    using the user's original prompt for context.
    Real URLs pass through unchanged.
    """
    if valkey is None:
        return messages
    prefix = f"blob:{conversation_id or 'anon'}"
    results: list[dict[str, object]] = []
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
    except Exception as exc:
        logger.warning("blob_store_error key=%s err=%s", key, exc)


async def _store_desc(valkey, key: str, desc: str) -> None:
    """Store description with SETNX + EXPIRE."""
    try:
        await valkey.setnx(key, desc)
        await valkey.expire(key, BLOB_TTL)
    except Exception as exc:
        logger.warning("blob_desc_store_error key=%s err=%s", key, exc)


async def _describe_audio(valkey, desc_key: str, raw: str, config) -> str:
    """Transcribe audio if not already cached."""
    try:
        cached = await valkey.get(desc_key)
        if cached:
            return cached
    except Exception as exc:
        logger.warning("blob_audio_cache_error key=%s err=%s", desc_key, exc)
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
    except Exception as exc:
        logger.warning("blob_pdf_cache_error key=%s err=%s", desc_key, exc)
    desc = await _try_extract_pdf_text(raw)
    if desc:
        await _store_desc(valkey, desc_key, desc)
    return desc


async def _describe_docx(valkey, desc_key: str, raw: str) -> str:
    """Extract DOCX text if not already cached."""
    try:
        cached = await valkey.get(desc_key)
        if cached:
            return cached
    except Exception as exc:
        logger.warning("blob_docx_cache_error key=%s err=%s", desc_key, exc)
    desc = await _try_extract_docx_text(raw)
    if desc:
        await _store_desc(valkey, desc_key, desc)
    return desc


def inject_blob_extraction_guidance(messages: list[dict]) -> list[dict]:
    """Inject system message explaining blob extraction and tool availability.

    When the proxy auto-extracts content (describes images, transcribes audio,
    extracts PDF text), it sends the extracted information in the message.
    This system message tells the model:

    1. Content was auto-extracted from blobs (multimodal content the model can't process)
    2. The model can choose to use its own specialized tools for alternative analysis
    3. How to access the blob reference if needed for custom processing

    This allows models with specialized tools (PDF parsers, image recognition,
    audio analysis) to decide whether to use the proxy's extraction or invoke
    their own tools for custom or more accurate analysis.

    Returns:
        messages with injected system message if blobs detected, otherwise unchanged
    """
    # Check if any message contains blob references (extracted content)
    has_blobs = False
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            if "[Content provided:" in content or f"[{BLOB_PREFIX}:" in content:
                has_blobs = True
                break
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    if "[Content provided:" in part["text"]:
                        has_blobs = True
                        break
        if has_blobs:
            break

    if not has_blobs:
        return messages

    # Check if system message already exists
    has_system = any(m.get("role") == "system" for m in messages)

    system_message = (
        "**Blob Content Processing Guide**\n\n"
        "Some of the content in this conversation was auto-extracted from multimodal files "
        "(images, audio, PDFs, Word documents) that the current model cannot process natively. "
        "The proxy has automatically:\n"
        "  • Described images using vision models\n"
        "  • Transcribed audio using speech-to-text\n"
        "  • Extracted text from PDFs and Word documents (DOCX)\n\n"
        "**If you have specialized tools available**, you can:\n"
        "  1. Use the blob reference (format: BLOB:hash:mimetype) to access raw data\n"
        "  2. Apply your own analysis for custom extraction or more detailed processing\n"
        "  3. Combine your specialized analysis with the provided extraction\n\n"
        "The extracted content is provided as text context above. "
        "Use your tools if you need alternative analysis or higher precision."
    )

    # Insert system message at the beginning if not already present
    if has_system:
        # System message exists, don't add another (user may have custom instructions)
        return messages
    else:
        # Add as first message
        return [{"role": "system", "content": system_message}] + messages
