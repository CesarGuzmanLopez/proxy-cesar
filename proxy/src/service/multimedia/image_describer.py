"""Image description service — Sprint 5.

Auto-describes images in conversation history when switching from a vision
pseudo-model to a non-vision one with ``on_downgrade: auto_describe``.

Design follows the compactor pattern from Sprint 4 (pre_compactor.py):
- ``call_litellm`` adapter for all model calls
- Pure async functions with immutable data
- ``Result`` monad pattern for error handling (python.md §3)

python.md §4: Pure functions, immutable data (deepcopy), declarative style.
python.md §5.2: Adapter pattern — uses LiteLLM adapter, not raw calls.
"""

import logging
from copy import deepcopy

from src.adapters.litellm import call_litellm
from src.config.constants import BLOB_STORAGE_TTL_SECONDS

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

DESCRIPTION_PROMPT: str = (
    "Analyze this image exhaustively for a text-only AI model. "
    "You must:\n"
    "1. Extract ALL visible text exactly as it appears (including text that is crossed out, "
    "strikethrough, handwritten, annotations, side notes, headers, footers, watermarks, labels). "
    "Preserve the original language and exact spelling.\n"
    "2. Describe the image in rich detail: overall scene, objects, people, actions, "
    "expressions, clothing, environment, lighting, colors, composition, perspective, "
    "spatial relationships, style (photograph, illustration, screenshot, diagram, chart, etc.).\n"
    "3. For UI/code/diagrams: describe layout, structure, hierarchy, interactive elements, "
    "data visualizations, axes, legends, trends, values shown.\n"
    "4. Note any imperfections: blur, glare, artifacts, damage, low resolution areas.\n"
    "5. If there are multiple panels/sections, describe each separately.\n\n"
    "Format your response with clear sections:\n"
    "[EXTRACTED TEXT]: All visible text verbatim\n"
    "[DESCRIPTION]: Detailed visual description\n"
    "[TECHNICAL DETAILS]: Format, layout, data if applicable"
)
"""Prompt sent to the vision model for each image.

Designed for exhaustive text extraction (including crossed-out/strikethrough text)
and comprehensive visual description. The model is instructed to preserve exact text
including annotations and handwritten notes.
"""

MAX_TOKENS_PER_IMAGE: int = 2048
"""Maximum completion tokens per image description."""

TAG_PREFIX: str = "IMAGE_DESCRIBED"
"""Prefix for the annotation tag inserted before each description."""


# ── Public API ─────────────────────────────────────────────────────────────────


def _extract_image_metadata(url: str) -> dict:
    """Extract metadata from image URL or base64 data.

    Returns dict with: type, size (if available), format, etc.
    """
    metadata = {"type": "image"}

    if url.startswith("data:"):
        # Base64 data URL: data:image/png;base64,... or data:image/jpeg;base64,...
        try:
            mime_part = url.split(";")[0].replace("data:", "")
            if mime_part:
                metadata["mime_type"] = mime_part
                ext = mime_part.split("/")[-1]
                metadata["format"] = ext.upper()

            # Rough size estimation from base64
            if "base64," in url:
                b64_data = url.split("base64,")[1]
                # Each base64 char represents ~6 bits, ~4 chars = 3 bytes
                estimated_bytes = (len(b64_data) * 3) // 4
                metadata["estimated_size_kb"] = str(round(estimated_bytes / 1024, 1))
                metadata["source"] = "uploaded_image"
        except Exception:
            pass
    else:
        # HTTPS URL
        metadata["url"] = url
        try:
            if "png" in url.lower():
                metadata["format"] = "PNG"
                metadata["mime_type"] = "image/png"
            elif "jpg" in url.lower() or "jpeg" in url.lower():
                metadata["format"] = "JPEG"
                metadata["mime_type"] = "image/jpeg"
            elif "webp" in url.lower():
                metadata["format"] = "WEBP"
                metadata["mime_type"] = "image/webp"
            elif "gif" in url.lower():
                metadata["format"] = "GIF"
                metadata["mime_type"] = "image/gif"

            # Extract filename from URL path
            from urllib.parse import urlparse
            parsed = urlparse(url)
            path = parsed.path
            if path:
                filename = path.split("/")[-1]
                if filename and "." in filename:
                    metadata["filename"] = filename
                    metadata["source"] = "url"
        except Exception:
            pass

    return metadata


def _extract_user_context(messages: list[dict]) -> str | None:
    """Extract the last user message's text content for vision context.

    Returns the user's prompt so the vision model understands the context
    of what the user is asking about.
    """
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, list):
                # Find text parts in the content
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text = part.get("text", "").strip()
                        if text and not text.startswith("data:image/"):
                            return text
            elif isinstance(content, str):
                return content.strip()
    return None


def find_image_refs(messages: list[dict]) -> list[dict]:
    """Find all image references in a message list (supports multiple formats).

    Detects images in all common formats:
    - OpenAI standard: {"type": "image_url", "image_url": {"url": "..."}}
    - Base64: {"type": "image", "image": "data:image/...;base64,..."}
    - Text with data URL: {"type": "text", "text": "data:image/..."}

    Deduplicates identical URLs — the same image in multiple turns or
    multiple parts produces a single description call.

    Args:
        messages: List of OpenAI-format message dicts.

    Returns:
        List of ref dicts with keys:
        - ``msg_idx``: index in messages array
        - ``part_idx``: index in content parts array
        - ``url``: the image URL or base64 data
        - ``detail``: detail level (auto/low/high)
        - ``is_duplicate``: ``True`` if this exact URL was seen before
    """
    refs: list[dict] = []
    seen_urls: set[str] = set()

    for msg_idx, msg in enumerate(messages):
        content = msg.get("content")
        if not isinstance(content, list):
            continue

        for part_idx, part in enumerate(content):
            if not isinstance(part, dict):
                continue

            part_type = part.get("type", "")
            url = None
            detail = "auto"

            # Format 1: OpenAI standard image_url
            if part_type == "image_url":
                url = part.get("image_url", {}).get("url", "")
                detail = part.get("image_url", {}).get("detail", "auto")

            # Format 2: Base64 inline image (type="image")
            elif part_type == "image":
                url = part.get("image", "")

            # Format 3: Text field containing data URL (some clients send it this way)
            elif part_type == "text":
                text_content = part.get("text", "")
                if isinstance(text_content, str) and text_content.startswith("data:image/"):
                    url = text_content

            # If we found an image URL, track it
            if url:
                is_duplicate = url in seen_urls
                seen_urls.add(url)

                refs.append(
                    {
                        "msg_idx": msg_idx,
                        "part_idx": part_idx,
                        "url": url,
                        "detail": detail,
                        "is_duplicate": is_duplicate,
                    }
                )

    return refs


async def describe_image(
    image_url: str,
    detail: str,
    vision_model: str,
    api_base: str | None = None,
    api_key: str | None = None,
) -> tuple[str, int]:
    """Describe a single image using a vision model via LiteLLM.

    Automatically degrades images (downscale + JPEG compression) before
    sending to reduce token consumption. High-resolution images (4K) are
    resized to a max of 1024px on the longest side.

    Args:
        image_url: Data URL or HTTPS URL of the image.
        detail: Detail level (``auto``, ``low``, or ``high``).
        vision_model: LiteLLM model identifier (e.g. ``openrouter/gemini-3.5-flash``).
        api_base: Custom API base URL (e.g. for OpenCode Go models).
        api_key: Custom API key (resolved from api_key_env).

    Returns:
        Tuple of ``(description_text, tokens_used)``.
        On failure, returns an error placeholder with 0 tokens.
    """
    from src.service.multimedia.image_processor import degrade_image

    degraded_url = degrade_image(image_url)
    img_messages: list[dict] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": DESCRIPTION_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": degraded_url, "detail": "high"},
                },
            ],
        }
    ]

    try:
        response = await call_litellm(
            model=vision_model,
            messages=img_messages,
            api_base=api_base,
            api_key=api_key,
            max_tokens=MAX_TOKENS_PER_IMAGE,
            temperature=0.0,  # Deterministic descriptions
        )
        content: str = response.choices[0].message.content or ""
        tokens: int = response.usage.completion_tokens if response.usage else 0
        return content.strip(), tokens
    except Exception:
        # Non-fatal: single image failure produces a placeholder
        return f"[{TAG_PREFIX} — DESCRIPTION FAILED for this image]", 0


async def auto_describe_images(
    messages: list[dict],
    vision_model: str,
    api_base: str | None = None,
    api_key: str | None = None,
    valkey=None,
) -> tuple[list[dict], dict]:
    """Auto-describe all images in a message list using BATCH calls + cache.

    Instead of describing each image one by one (legacy behavior), groups
    all uncached images and sends them in a single batch to the vision model
    WITH the user's prompt for context. Also returns a general summary if
    multiple images are present.

    Caches descriptions in Valkey under ``blob:desc:generic:{url_hash}`` so
    the same image across conversations reuses the description.

    Args:
        messages: Original message list (read-only — not modified).
        vision_model: LiteLLM model identifier with vision capability.
        api_base: Custom API base URL (e.g. for OpenCode Go models).
        api_key: Custom API key (resolved from api_key_env).
        valkey: Optional Valkey client for description cache.

    Returns:
        Tuple of ``(modified_messages, metadata_dict)``.
        Original messages are never mutated — a deep copy is returned.
    """
    import hashlib
    import json as json_mod

    refs = find_image_refs(messages)
    if not refs:
        return messages, {
            "ok": True,
            "images_described": 0,
            "reason": "no_images_found",
            "status": "no_images_found",
        }

    # Extract user prompt for context
    user_prompt = _extract_user_context(messages)

    unique_refs = [r for r in refs if not r["is_duplicate"]]
    duplicate_refs = [r for r in refs if r["is_duplicate"]]
    total_tokens: int = 0
    url_cache: dict[str, str] = {}
    metadata_cache: dict[str, dict] = {}  # Store metadata per URL

    # Separate cached vs uncached images
    uncached: list[dict] = []
    for ref in unique_refs:
        url_hash = hashlib.sha256(ref["url"].encode()).hexdigest()[:16]
        # Extract metadata for ALL images (cached or not)
        metadata = _extract_image_metadata(ref["url"])
        metadata_cache[ref["url"]] = metadata

        cached_desc: str | None = None
        if valkey is not None:
            try:
                cached_desc = await valkey.get(f"blob:desc:generic:{url_hash}")
            except Exception as e:
                logger.warning("image_cache_retrieval_error error=%s", str(e))
        if cached_desc:
            url_cache[ref["url"]] = cached_desc
            total_tokens += len(cached_desc) // 4  # rough token estimate
        else:
            uncached.append({**ref, "url_hash": url_hash})

    # Describe uncached images in a single BATCH call
    if uncached:
        from src.service.multimedia.image_processor import degrade_image

        # Build instruction with user context
        instruction = (
            "Analyze each image exhaustively for a text-only AI model. For EACH image you must:\n"
            "1. Extract ALL visible text verbatim (including crossed out, strikethrough, "
            "handwritten, annotations, headers, footers, labels, watermarks). "
            "Preserve exact spelling and original language.\n"
            "2. Provide rich visual description: scene, objects, people, actions, expressions, "
            "clothing, environment, lighting, colors, composition, style, spatial relationships.\n"
            "3. For UI/code/diagrams: layout, structure, hierarchy, interactive elements, "
            "data values, axes, legends, trends.\n"
            "4. Note imperfections: blur, glare, artifacts, damage.\n"
            "5. Multiple panels: describe each separately.\n\n"
            "Format per image:\n"
            "[EXTRACTED TEXT]: All visible text\n"
            "[DESCRIPTION]: Detailed visual description\n"
            "[TECHNICAL DETAILS]: Format, layout, data"
        )
        if user_prompt:
            instruction += f"\n\nUser context / what the user is asking about: {user_prompt}"

        if len(uncached) > 1:
            instruction += (
                "\n\nReturn a JSON object with:"
                '\n- "descriptions": array of strings (one per image, in order, each containing '
                "the full analysis with [EXTRACTED TEXT], [DESCRIPTION], [TECHNICAL DETAILS])"
                '\n- "summary": synthesis (2-3 sentences) noting relationships, patterns, or comparisons'
                '\n\nExample: {"descriptions": ["[EXTRACTED TEXT]: ...\\n[DESCRIPTION]: ...", '
                '"[EXTRACTED TEXT]: ...\\n[DESCRIPTION]: ..."], '
                '"summary": "Both images show..."}'
            )
        else:
            instruction += (
                "\n\nReturn a JSON array of strings in the order the images were sent. "
                'Example: ["[EXTRACTED TEXT]: ...\\n[DESCRIPTION]: ...\\n[TECHNICAL DETAILS]: ..."]'
            )

        batch_content: list[dict] = [
            {
                "type": "text",
                "text": instruction,
            }
        ]
        for ref in uncached:
            degraded = degrade_image(ref["url"])
            batch_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": degraded, "detail": "high"},
                }
            )
            # Collect metadata from URL/base64
            metadata = _extract_image_metadata(ref["url"])
            metadata_cache[ref["url"]] = metadata

        batch_tokens: int = 0
        general_summary: str | None = None
        try:
            response = await call_litellm(
                model=vision_model,
                messages=[{"role": "user", "content": batch_content}],
                api_base=api_base,
                api_key=api_key,
                max_tokens=2048 * len(uncached),
                temperature=0.1,
            )
            # Extract usage tokens from response
            try:
                if hasattr(response, "usage") and response.usage:
                    batch_tokens = getattr(response.usage, "completion_tokens", 0) or 0
                elif isinstance(response, dict):
                    batch_tokens = (response.get("usage") or {}).get(
                        "completion_tokens", 0
                    ) or 0
            except (AttributeError, TypeError, KeyError) as e:
                logger.warning("image_metadata_extraction_failed error=%s", str(e))

            # Extract text from response (handles ModelResponse, dict, and MagicMock)
            text = ""
            try:
                if hasattr(response, "choices"):
                    choice = response.choices[0]
                    if hasattr(choice, "message"):
                        text = choice.message.content or ""
                elif isinstance(response, dict):
                    text = (response.get("choices") or [{}])[0].get("message", {}).get(
                        "content", ""
                    ) or ""
            except (AttributeError, IndexError, TypeError, KeyError) as e:
                logger.warning("image_description_parsing_failed error=%s", str(e))
            if text and isinstance(text, str):
                text = text.strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                try:
                    parsed = json_mod.loads(text)

                    if isinstance(parsed, list):
                        # Format: ["desc1", "desc2", ...]
                        tokens_per_image = (
                            max(batch_tokens // len(parsed), 1) if batch_tokens else 0
                        )
                        for i, ref in enumerate(uncached):
                            desc = str(parsed[i]) if i < len(parsed) else ""
                            url_cache[ref["url"]] = desc
                            total_tokens += (
                                tokens_per_image if tokens_per_image else len(desc) // 4
                            )
                            if valkey is not None and desc:
                                try:
                                    prompt_hash = hashlib.sha256(
                                        (user_prompt or "").encode()
                                    ).hexdigest()[:8]
                                    await valkey.set(
                                        f"blob:desc:{prompt_hash}:{ref['url_hash']}",
                                        desc,
                                        ex=BLOB_STORAGE_TTL_SECONDS,
                                    )
                                except Exception as e:
                                    logger.warning(
                                        "blob_storage_error error=%s", str(e)
                                    )
                    elif isinstance(parsed, dict):
                        # Format: {"descriptions": [...], "summary": "..."}
                        descriptions = parsed.get("descriptions", [])
                        general_summary = parsed.get("summary", None)

                        tokens_per_image = (
                            max(batch_tokens // len(descriptions), 1)
                            if batch_tokens and descriptions
                            else 0
                        )
                        for i, ref in enumerate(uncached):
                            desc = str(descriptions[i]) if i < len(descriptions) else ""
                            url_cache[ref["url"]] = desc
                            total_tokens += (
                                tokens_per_image if tokens_per_image else len(desc) // 4
                            )
                            if valkey is not None and desc:
                                try:
                                    prompt_hash = hashlib.sha256(
                                        (user_prompt or "").encode()
                                    ).hexdigest()[:8]
                                    await valkey.set(
                                        f"blob:desc:{prompt_hash}:{ref['url_hash']}",
                                        desc,
                                        ex=BLOB_STORAGE_TTL_SECONDS,
                                    )
                                except Exception as e:
                                    logger.warning(
                                        "blob_storage_error error=%s", str(e)
                                    )
                except (json_mod.JSONDecodeError, IndexError, TypeError):
                    pass

        except Exception as exc:
            import logging

            logging.getLogger(__name__).warning(
                "auto_describe_batch_failed model=%s images=%d error=%s",
                vision_model,
                len(uncached),
                str(exc),
            )

        # Fallback: any image still missing description gets described individually
        for ref in uncached:
            if ref["url"] not in url_cache or not url_cache[ref["url"]]:
                desc, tokens = await describe_image(
                    ref["url"],
                    ref["detail"],
                    vision_model,
                    api_base=api_base,
                    api_key=api_key,
                )
                url_cache[ref["url"]] = desc
                total_tokens += tokens

    # Build modified message list - deepcopy all messages that will be modified
    # to avoid corrupting the caller's original conversation history
    modified_indices: set[int] = {
        ref["msg_idx"] for ref in refs if url_cache.get(ref["url"])
    }
    # Also include the last image message if general summary will be added
    if general_summary and len([r for r in refs if url_cache.get(r["url"])]) > 1:
        last_img_msg_idx = max([ref["msg_idx"] for ref in refs]) if refs else -1
        if last_img_msg_idx >= 0:
            modified_indices.add(last_img_msg_idx)

    modified: list[dict] = [
        deepcopy(msg) if i in modified_indices else msg
        for i, msg in enumerate(messages)
    ]
    described_count: int = 0

    for ref in refs:
        description = url_cache.get(ref["url"])
        if not description:
            continue
        described_count += 1

        # Build full text with metadata
        tag: str = f"[{TAG_PREFIX} #{described_count} — described by {vision_model}]"
        full_text: str = f"{tag}\n\n{description}"

        # Add metadata if available (from cache, works for all images)
        meta = metadata_cache.get(ref["url"])
        if meta:
            meta_lines = []
            if meta.get("format"):
                meta_lines.append(f"Format: {meta['format']}")
            if meta.get("estimated_size_kb"):
                meta_lines.append(f"Size: ~{meta['estimated_size_kb']}KB")
            if meta.get("mime_type"):
                meta_lines.append(f"MIME: {meta['mime_type']}")
            if meta.get("filename"):
                meta_lines.append(f"Filename: {meta['filename']}")
            if meta_lines:
                full_text += f"\n[Metadata: {', '.join(meta_lines)}]"

        msg = modified[ref["msg_idx"]]
        content_list = msg["content"]
        content_list[ref["part_idx"]] = {"type": "text", "text": full_text}

    # Add general summary after all descriptions if available
    if general_summary and described_count > 1:
        # Append summary to the last message that had images
        last_img_msg_idx = max([ref["msg_idx"] for ref in refs]) if refs else -1
        if last_img_msg_idx >= 0 and last_img_msg_idx < len(modified):
            msg = modified[last_img_msg_idx]
            summary_text = f"\n\n[{TAG_PREFIX} General Summary]\n{general_summary}"
            # Append to existing content
            if isinstance(msg.get("content"), list):
                # Find last text part and append, or add new text part
                found_text = False
                for part in reversed(msg["content"]):
                    if isinstance(part, dict) and part.get("type") == "text":
                        part["text"] = part["text"] + summary_text
                        found_text = True
                        break
                if not found_text:
                    msg["content"].append({"type": "text", "text": summary_text})

    metadata: dict = {
        "ok": True,
        "images_described": described_count,
        "unique_images_described": len(unique_refs),
        "duplicate_images_skipped": len(duplicate_refs),
        "described_by": vision_model,
        "total_description_tokens": total_tokens,
        "general_summary_provided": general_summary is not None,
        "image_metadata_included": len(metadata_cache) > 0,
        "status": "completed",
    }

    return modified, metadata
