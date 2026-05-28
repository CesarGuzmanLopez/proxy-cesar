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
    "Describe this image in detail for a text-only AI model. "
    "Include: what is shown, layout, visible text, colors, "
    "key elements, and any technical details (UI, code, diagram). "
    "Be thorough but concise — max 200 words."
)
"""Prompt sent to the vision model for each image.

Designed for completeness (covers visual + technical details) and
conciseness (max 200 words to avoid bloating the context).
"""

MAX_TOKENS_PER_IMAGE: int = 512
"""Maximum completion tokens per image description."""

TAG_PREFIX: str = "IMAGE_DESCRIBED"
"""Prefix for the annotation tag inserted before each description."""


# ── Public API ─────────────────────────────────────────────────────────────────


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
    all uncached images and sends them in a single batch to the vision model.
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

    unique_refs = [r for r in refs if not r["is_duplicate"]]
    duplicate_refs = [r for r in refs if r["is_duplicate"]]
    total_tokens: int = 0
    url_cache: dict[str, str] = {}

    # Separate cached vs uncached images
    uncached: list[dict] = []
    for ref in unique_refs:
        url_hash = hashlib.sha256(ref["url"].encode()).hexdigest()[:16]
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

        batch_content: list[dict] = [
            {
                "type": "text",
                "text": (
                    "Describe each image briefly (1-2 sentences). "
                    "Return a JSON array of strings in the order the images were sent. "
                    'Example: ["A login screen", "A bar chart of Q1 sales"]'
                ),
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

        batch_tokens: int = 0
        try:
            response = await call_litellm(
                model=vision_model,
                messages=[{"role": "user", "content": batch_content}],
                api_base=api_base,
                api_key=api_key,
                max_tokens=512 * len(uncached),
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
                        tokens_per_image = (
                            max(batch_tokens // len(parsed), 1) if batch_tokens else 0
                        )
                        for i, ref in enumerate(uncached):
                            desc = str(parsed[i])[:500] if i < len(parsed) else ""
                            url_cache[ref["url"]] = desc
                            total_tokens += (
                                tokens_per_image if tokens_per_image else len(desc) // 4
                            )
                            if valkey is not None and desc:
                                try:
                                    await valkey.set(
                                        f"blob:desc:generic:{ref['url_hash']}",
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

    # Build modified message list (shallow copy list, deep copy only modified msgs)
    # Optimization: avoid deepcopy of entire list if only a few messages change
    modified_indices: set[int] = {
        ref["msg_idx"] for ref in refs if url_cache.get(ref["url"])
    }
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
        tag: str = f"[{TAG_PREFIX} #{described_count} — described by {vision_model}]"
        full_text: str = f"{tag}\n\n{description}"
        msg = modified[ref["msg_idx"]]
        content_list = msg["content"]
        content_list[ref["part_idx"]] = {"type": "text", "text": full_text}

    metadata: dict = {
        "ok": True,
        "images_described": described_count,
        "unique_images_described": len(unique_refs),
        "duplicate_images_skipped": len(duplicate_refs),
        "described_by": vision_model,
        "total_description_tokens": total_tokens,
        "status": "completed",
    }

    return modified, metadata
