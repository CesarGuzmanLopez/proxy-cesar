"""Image description service — feature

Auto-describes images in conversation history when switching from a vision
pseudo-model to a non-vision one with ``on_downgrade: auto_describe``.
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
    "1. Extract ALL visible text exactly as it appears (including crossed out, "
    "strikethrough, handwritten, annotations, side notes, headers, footers, watermarks, labels). "
    "Preserve the original language and exact spelling.\n"
    "2. Describe the image in rich detail: overall scene, objects, people, actions, "
    "expressions, clothing, environment, lighting, colors, composition, perspective, "
    "spatial relationships, style.\n"
    "3. For UI/code/diagrams: describe layout, structure, hierarchy, interactive elements, "
    "data visualizations, axes, legends, trends, values shown.\n"
    "4. Note any imperfections: blur, glare, artifacts, damage, low resolution areas.\n"
    "5. If there are multiple panels/sections, describe each separately.\n\n"
    "Format your response with clear sections:\n"
    "[EXTRACTED TEXT]: All visible text verbatim\n"
    "[DESCRIPTION]: Detailed visual description\n"
    "[TECHNICAL DETAILS]: Format, layout, data if applicable"
)

MAX_TOKENS_PER_IMAGE: int = 2048
TAG_PREFIX: str = "IMAGE_DESCRIBED"


# ── Metadata & context extraction ──────────────────────────────────────────────


def _extract_image_metadata(url: str) -> dict:
    """Extract metadata from image URL or base64 data."""
    metadata = {"type": "image"}
    if url.startswith("data:"):
        try:
            mime_part = url.split(";")[0].replace("data:", "")
            if mime_part:
                metadata["mime_type"] = mime_part
                metadata["format"] = mime_part.split("/")[-1].upper()
            if "base64," in url:
                b64_data = url.split("base64,")[1]
                estimated_bytes = (len(b64_data) * 3) // 4
                metadata["estimated_size_kb"] = str(round(estimated_bytes / 1024, 1))
                metadata["source"] = "uploaded_image"
        except Exception:
            pass
    else:
        metadata["url"] = url
        try:
            for fmt, mime in (
                ("png", "image/png"),
                ("jpg", "image/jpeg"),
                ("jpeg", "image/jpeg"),
                ("webp", "image/webp"),
                ("gif", "image/gif"),
            ):
                if fmt in url.lower():
                    metadata["format"] = fmt.upper()
                    metadata["mime_type"] = mime
                    break
            from urllib.parse import urlparse
            path = urlparse(url).path
            if path:
                filename = path.split("/")[-1]
                if filename and "." in filename:
                    metadata["filename"] = filename
                    metadata["source"] = "url"
        except Exception:
            pass
    return metadata


def _extract_user_context(messages: list[dict]) -> str | None:
    """Extract the last user message's text content for vision context."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text = part.get("text", "").strip()
                        if text and not text.startswith("data:image/"):
                            return text
            elif isinstance(content, str):
                return content.strip()
    return None


def find_image_refs(messages: list[dict]) -> list[dict]:
    """Find all image references in a message list (deduplicates by URL)."""
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
            if part_type == "image_url":
                url = part.get("image_url", {}).get("url", "")
                detail = part.get("image_url", {}).get("detail", "auto")
            elif part_type == "image":
                url = part.get("image", "")
            elif part_type == "text":
                text_content = part.get("text", "")
                if isinstance(text_content, str) and text_content.startswith("data:image/"):
                    url = text_content
            if url:
                is_duplicate = url in seen_urls
                seen_urls.add(url)
                refs.append({
                    "msg_idx": msg_idx,
                    "part_idx": part_idx,
                    "url": url,
                    "detail": detail,
                    "is_duplicate": is_duplicate,
                })
    return refs


# ── Individual image description ───────────────────────────────────────────────


async def describe_image(
    image_url: str,
    detail: str,
    vision_model: str,
    api_base: str | None = None,
    api_key: str | None = None,
) -> tuple[str, int]:
    """Describe a single image using a vision model via LiteLLM."""
    from src.service.multimedia.image_processor import degrade_image

    degraded_url = degrade_image(image_url)
    img_messages: list[dict] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": DESCRIPTION_PROMPT},
                {"type": "image_url", "image_url": {"url": degraded_url, "detail": "high"}},
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
            temperature=0.0,
        )
        content: str = response.choices[0].message.content or ""
        tokens: int = response.usage.completion_tokens if response.usage else 0
        return content.strip(), tokens
    except Exception:
        return f"[{TAG_PREFIX} — DESCRIPTION FAILED for this image]", 0


# ── Batch description helpers ──────────────────────────────────────────────────


async def _collect_cached_descriptions(
    refs: list[dict],
    valkey,
) -> tuple[dict[str, str], dict[str, dict], list[dict], int]:
    """Collect cached descriptions; return (url_cache, metadata_cache, uncached, total_tokens)."""
    import hashlib

    url_cache: dict[str, str] = {}
    metadata_cache: dict[str, dict] = {}
    uncached: list[dict] = []
    total_tokens: int = 0

    for ref in refs:
        if ref["is_duplicate"]:
            continue
        url_hash = hashlib.sha256(ref["url"].encode()).hexdigest()[:16]
        metadata_cache[ref["url"]] = _extract_image_metadata(ref["url"])
        cached_desc: str | None = None
        if valkey is not None:
            try:
                cached_desc = await valkey.get(f"blob:desc:generic:{url_hash}")
            except Exception as e:
                logger.warning("image_cache_retrieval_error error=%s", str(e))
        if cached_desc:
            url_cache[ref["url"]] = cached_desc
            total_tokens += len(cached_desc) // 4
        else:
            uncached.append({**ref, "url_hash": url_hash})

    return url_cache, metadata_cache, uncached, total_tokens


def _build_batch_instruction(user_prompt: str | None, image_count: int) -> str:
    """Build instruction for batch image description."""
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
    if image_count > 1:
        instruction += (
            "\n\nReturn a JSON object with:"
            '\n- "descriptions": array of strings (one per image, in order)'
            '\n- "summary": synthesis (2-3 sentences) noting relationships, patterns, or comparisons'
            '\n\nExample: {"descriptions": ["...", "..."], "summary": "Both images show..."}'
        )
    else:
        instruction += (
            '\n\nReturn a JSON array of strings in order. Example: ["[EXTRACTED TEXT]: ..."]'
        )
    return instruction


async def _parse_batch_response(
    response,
    uncached: list[dict],
    url_cache: dict[str, str],
    total_tokens: int,
    valkey,
    user_prompt: str | None,
) -> tuple[dict[str, str], int, str | None]:
    """Parse batch vision response, store in cache."""
    import json as json_mod

    # Extract text from response
    text = ""
    try:
        if hasattr(response, "choices"):
            choice = response.choices[0]
            if hasattr(choice, "message"):
                text = choice.message.content or ""
        elif isinstance(response, dict):
            text = (response.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
    except (AttributeError, IndexError, TypeError, KeyError) as e:
        logger.warning("image_description_parsing_failed error=%s", str(e))

    if not text or not isinstance(text, str):
        return url_cache, total_tokens, None

    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    # Extract usage
    batch_tokens = 0
    try:
        if hasattr(response, "usage") and response.usage:
            batch_tokens = getattr(response.usage, "completion_tokens", 0) or 0
        elif isinstance(response, dict):
            batch_tokens = (response.get("usage") or {}).get("completion_tokens", 0) or 0
    except (AttributeError, TypeError, KeyError) as e:
        logger.warning("image_metadata_extraction_failed error=%s", str(e))

    general_summary: str | None = None
    try:
        parsed = json_mod.loads(text)
        if isinstance(parsed, list):
            for i, ref in enumerate(uncached):
                desc = str(parsed[i]) if i < len(parsed) else ""
                url_cache[ref["url"]] = desc
                total_tokens += max(batch_tokens // len(parsed), 1) if batch_tokens else len(desc) // 4
                await _cache_description(valkey, ref, desc, user_prompt)
        elif isinstance(parsed, dict):
            descriptions = parsed.get("descriptions", [])
            general_summary = parsed.get("summary")
            for i, ref in enumerate(uncached):
                desc = str(descriptions[i]) if i < len(descriptions) else ""
                url_cache[ref["url"]] = desc
                total_tokens += max(batch_tokens // len(descriptions), 1) if batch_tokens and descriptions else len(desc) // 4
                await _cache_description(valkey, ref, desc, user_prompt)
    except (json_mod.JSONDecodeError, IndexError, TypeError):
        pass

    return url_cache, total_tokens, general_summary


async def _cache_description(
    valkey,
    ref: dict,
    desc: str,
    user_prompt: str | None,
) -> None:
    """Store a description in Valkey cache."""
    import hashlib

    if valkey is not None and desc:
        try:
            prompt_hash = hashlib.sha256((user_prompt or "").encode()).hexdigest()[:8]
            await valkey.set(
                f"blob:desc:{prompt_hash}:{ref['url_hash']}",
                desc,
                ex=BLOB_STORAGE_TTL_SECONDS,
            )
        except Exception as e:
            logger.warning("blob_storage_error error=%s", str(e))


async def _describe_individually(
    uncached: list[dict],
    url_cache: dict[str, str],
    vision_model: str,
    api_base: str | None,
    api_key: str | None,
    total_tokens: int,
) -> tuple[dict[str, str], int]:
    """Fallback: describe any uncached images individually."""
    for ref in uncached:
        if ref["url"] not in url_cache or not url_cache[ref["url"]]:
            desc, tokens = await describe_image(
                ref["url"], ref["detail"], vision_model, api_base, api_key
            )
            url_cache[ref["url"]] = desc
            total_tokens += tokens
    return url_cache, total_tokens


def _build_modified_messages(
    messages: list[dict],
    refs: list[dict],
    url_cache: dict[str, str],
    metadata_cache: dict[str, dict],
    general_summary: str | None,
) -> tuple[list[dict], int]:
    """Build modified messages with descriptions substituted."""
    modified_indices: set[int] = {ref["msg_idx"] for ref in refs if url_cache.get(ref["url"])}
    if general_summary and len([r for r in refs if url_cache.get(r["url"])]) > 1:
        last_img_msg_idx = max(ref["msg_idx"] for ref in refs) if refs else -1
        if last_img_msg_idx >= 0:
            modified_indices.add(last_img_msg_idx)

    modified: list[dict] = [
        deepcopy(msg) if i in modified_indices else msg
        for i, msg in enumerate(messages)
    ]

    described_count = 0
    for ref in refs:
        description = url_cache.get(ref["url"])
        if not description:
            continue
        described_count += 1
        tag = f"[{TAG_PREFIX} #{described_count} — described by vision model]"
        full_text = f"{tag}\n\n{description}"
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
        msg["content"][ref["part_idx"]] = {"type": "text", "text": full_text}

    # Add general summary after all descriptions if available
    if general_summary and described_count > 1:
        last_img_msg_idx = max(ref["msg_idx"] for ref in refs) if refs else -1
        if 0 <= last_img_msg_idx < len(modified):
            msg = modified[last_img_msg_idx]
            summary_text = f"\n\n[{TAG_PREFIX} General Summary]\n{general_summary}"
            if isinstance(msg.get("content"), list):
                for part in reversed(msg["content"]):
                    if isinstance(part, dict) and part.get("type") == "text":
                        part["text"] = part["text"] + summary_text
                        break
                else:
                    msg["content"].append({"type": "text", "text": summary_text})

    return modified, described_count


# ── Public API ─────────────────────────────────────────────────────────────────


async def auto_describe_images(
    messages: list[dict],
    vision_model: str,
    api_base: str | None = None,
    api_key: str | None = None,
    valkey=None,
) -> tuple[list[dict], dict]:
    """Auto-describe all images in a message list using BATCH calls + cache.

    Returns (modified_messages, metadata_dict). Original messages are never mutated.
    """
    refs = find_image_refs(messages)
    if not refs:
        return messages, {"ok": True, "images_described": 0, "reason": "no_images_found", "status": "no_images_found"}

    user_prompt = _extract_user_context(messages)
    url_cache, metadata_cache, uncached, total_tokens = await _collect_cached_descriptions(refs, valkey)
    unique_refs = [r for r in refs if not r["is_duplicate"]]
    duplicate_refs = [r for r in refs if r["is_duplicate"]]

    # Batch describe uncached images
    if uncached:
        from src.service.multimedia.image_processor import degrade_image, can_batch_fit, estimate_image_tokens

        instruction = _build_batch_instruction(user_prompt, len(uncached))
        batch_content: list[dict] = [{"type": "text", "text": instruction}]
        for ref in uncached:
            degraded = degrade_image(ref["url"])
            batch_content.append({"type": "image_url", "image_url": {"url": degraded, "detail": "high"}})
            metadata_cache[ref["url"]] = _extract_image_metadata(ref["url"])

        # Safety check
        first_degraded = degrade_image(uncached[0]["url"])
        sample_tokens = estimate_image_tokens(first_degraded, "high")
        instruction_tokens = len(instruction) // 4
        if not can_batch_fit(len(uncached), sample_tokens, 128000, instruction_tokens):
            logger.warning(
                "image_batch_large vision_model=%s images=%d estimated_tokens=%d",
                vision_model, len(uncached), len(uncached) * sample_tokens,
            )

        try:
            response = await call_litellm(
                model=vision_model,
                messages=[{"role": "user", "content": batch_content}],
                api_base=api_base,
                api_key=api_key,
                max_tokens=2048 * len(uncached),
                temperature=0.1,
            )
            url_cache, total_tokens, general_summary = await _parse_batch_response(
                response, uncached, url_cache, total_tokens, valkey, user_prompt
            )
        except Exception as exc:
            logger.warning(
                "auto_describe_batch_failed model=%s images=%d error=%s",
                vision_model, len(uncached), str(exc),
            )
            general_summary = None

        # Fallback: individual descriptions
        url_cache, total_tokens = await _describe_individually(
            uncached, url_cache, vision_model, api_base, api_key, total_tokens
        )
    else:
        general_summary = None

    modified, described_count = _build_modified_messages(
        messages, refs, url_cache, metadata_cache, general_summary
    )

    metadata = {
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
