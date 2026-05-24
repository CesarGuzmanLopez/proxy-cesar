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

from copy import deepcopy

from src.adapters.litellm import call_litellm

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
    """Find all ``image_url`` references in a message list.

    Deduplicates identical URLs — the same image in multiple turns or
    multiple parts produces a single description call.

    Args:
        messages: List of OpenAI-format message dicts.

    Returns:
        List of ref dicts with keys:
        - ``msg_idx``: index in messages array
        - ``part_idx``: index in content parts array
        - ``url``: the image URL
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
            if part.get("type") != "image_url":
                continue

            url = part.get("image_url", {}).get("url", "")
            if not url:
                continue

            is_duplicate = url in seen_urls
            seen_urls.add(url)

            refs.append({
                "msg_idx": msg_idx,
                "part_idx": part_idx,
                "url": url,
                "detail": part.get("image_url", {}).get("detail", "auto"),
                "is_duplicate": is_duplicate,
            })

    return refs


async def describe_image(
    image_url: str,
    detail: str,
    vision_model: str,
) -> tuple[str, int]:
    """Describe a single image using a vision model via LiteLLM.

    Args:
        image_url: Data URL or HTTPS URL of the image.
        detail: Detail level (``auto``, ``low``, or ``high``).
        vision_model: LiteLLM model identifier (e.g. ``openrouter/gemini-3.5-flash``).

    Returns:
        Tuple of ``(description_text, tokens_used)``.
        On failure, returns an error placeholder with 0 tokens.
    """
    img_messages: list[dict] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": DESCRIPTION_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": image_url, "detail": detail},
                },
            ],
        }
    ]

    try:
        response = await call_litellm(
            model=vision_model,
            messages=img_messages,
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
) -> tuple[list[dict], dict]:
    """Auto-describe all images in a message list.

    Iterates over all messages, finds ``image_url`` content parts,
    describes each unique image using the given vision model, and
    replaces the URL parts with ``[IMAGE_DESCRIBED #N]`` text annotations.

    Args:
        messages: Original message list (read-only — not modified).
        vision_model: LiteLLM model identifier with vision capability.

    Returns:
        Tuple of ``(modified_messages, metadata_dict)``.
        Original messages are never mutated — a deep copy is returned.
        Metadata keys:
        - ``ok``: always ``True`` (errors are per-image, not aborting)
        - ``images_described``: total images described (including duplicates)
        - ``unique_images_described``: unique URLs described
        - ``duplicate_images_skipped``: repeated URLs reused
        - ``described_by``: the vision model that described them
        - ``total_description_tokens``: sum of all description tokens
        - ``status``: ``"completed"`` or ``"no_images_found"``
    """
    refs = find_image_refs(messages)

    if not refs:
        return deepcopy(messages), {
            "ok": True,
            "images_described": 0,
            "reason": "no_images_found",
            "status": "no_images_found",
        }

    # Separate unique vs duplicate refs
    unique_refs = [r for r in refs if not r["is_duplicate"]]
    duplicate_refs = [r for r in refs if r["is_duplicate"]]

    # Build URL→description cache by describing unique images
    url_cache: dict[str, str] = {}
    total_tokens: int = 0

    for idx, ref in enumerate(unique_refs):
        desc, tokens = await describe_image(ref["url"], ref["detail"], vision_model)
        url_cache[ref["url"]] = desc
        total_tokens += tokens

    # Build modified message list
    modified: list[dict] = deepcopy(messages)
    described_count: int = 0

    for ref in refs:
        description = url_cache.get(ref["url"])
        if description is None:
            continue  # Safety — should not happen

        described_count += 1
        tag: str = f"[{TAG_PREFIX} #{described_count} — described by {vision_model}]"
        full_text: str = f"{tag}\n\n{description}"

        msg = modified[ref["msg_idx"]]
        content_list = msg["content"]
        content_list[ref["part_idx"]] = {
            "type": "text",
            "text": full_text,
        }

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
