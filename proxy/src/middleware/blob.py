"""Blob Vault middleware — transparent binary content storage and reference.

Intercepts chat completion requests to:
1. Detect base64-encoded binary content (images, audio, files) in messages
2. Store them in Valkey per conversation with a content hash
3. Replace with [BLOB:hash:type] placeholders
4. Provide a GET /blobs/{hash} endpoint to retrieve the real data

When a model discovers it can use a tool to process the content, the
tool handler can fetch the real base64 data from the blob endpoint.
"""

import hashlib
import logging
import re

from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

BLOB_TTL = 7200  # 2 hours
BLOB_PREFIX = "blob"
_PLACEHOLDER_RE = re.compile(rf"\[{BLOB_PREFIX}:([a-f0-9]{{16}}):([^\]]+)\]")
_CHAT_PATH = "/v1/chat/completions"

_MAX_BLOB_SIZE = 10 * 1024 * 1024  # 10MB max blob size


def _hash_content(data: str) -> str:
    """SHA-256 hash of blob content (16 hex chars)."""
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def _make_placeholder(content_hash: str, mime_type: str) -> str:
    return f"[{BLOB_PREFIX}:{content_hash}:{mime_type}]"


def _extract_base64_parts(messages: list) -> list[tuple[str, str, str]]:
    """Scan messages for base64 content parts.

    Returns list of (content_hash, raw_data, mime_type) tuples.
    """
    blobs: list[tuple[str, str, str]] = []
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type", "")

            if ptype == "image_url":
                raw = part.get("image_url", {}).get("url", "")
                if raw.startswith("data:"):
                    mime = _extract_mime(raw) or "image/unknown"
                    blobs.append((_hash_content(raw), raw, mime))

            elif ptype == "input_audio":
                audio = part.get("input_audio", {})
                raw = audio.get("data", "")
                if raw.startswith("data:"):
                    mime = _extract_mime(raw) or "audio/unknown"
                    blobs.append((_hash_content(raw), raw, mime))

            elif ptype == "file":
                file_data = part.get("file", {})
                raw = file_data.get("data", "")
                if raw.startswith("data:"):
                    mime = _extract_mime(raw) or "application/octet-stream"
                    blobs.append((_hash_content(raw), raw, mime))

    return blobs


def _extract_mime(data_uri: str) -> str | None:
    """Extract MIME type from data URI: 'data:image/png;base64,...' → 'image/png'."""
    match = re.match(r"data:([a-z]+/[a-z0-9+-.]+)", data_uri)
    return match.group(1) if match else None


def _replace_base64_with_placeholder(messages: list, blob_map: dict[str, tuple[str, str]]) -> list:
    """Replace base64 content with [BLOB:hash:type] placeholders.

    blob_map: {hash: (raw_data, mime_type)}
    """
    hash_to_mime = {h: m for h, (_, m) in blob_map.items()}
    new_messages = []
    for msg in messages:
        if not isinstance(msg, dict):
            new_messages.append(msg)
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            new_messages.append(msg)
            continue
        new_content = []
        for part in content:
            if not isinstance(part, dict):
                new_content.append(part)
                continue
            ptype = part.get("type", "")

            if ptype == "image_url":
                raw = part.get("image_url", {}).get("url", "")
                if raw.startswith("data:"):
                    h = _hash_content(raw)
                    if h in hash_to_mime:
                        new_content.append({
                            "type": "text",
                            "text": f"[The user sent an image. blob: {_make_placeholder(h, hash_to_mime[h])}]",
                        })
                        continue
            elif ptype == "input_audio":
                raw = part.get("input_audio", {}).get("data", "")
                if raw.startswith("data:"):
                    h = _hash_content(raw)
                    if h in hash_to_mime:
                        new_content.append({
                            "type": "text",
                            "text": f"[The user sent an audio file. blob: {_make_placeholder(h, hash_to_mime[h])}]",
                        })
                        continue
            elif ptype == "file":
                raw = part.get("file", {}).get("data", "")
                if raw.startswith("data:"):
                    h = _hash_content(raw)
                    if h in hash_to_mime:
                        new_content.append({
                            "type": "text",
                            "text": f"[The user sent a file. blob: {_make_placeholder(h, hash_to_mime[h])}]",
                        })
                        continue

            new_content.append(part)
        new_messages.append({**msg, "content": new_content})
    return new_messages


class BlobVaultMiddleware(BaseHTTPMiddleware):
    """Transparent blob vault middleware.

    Detects base64 binary content in chat messages, stores it in Valkey,
    and replaces with [BLOB:hash:type] placeholders.

    The model receives a reference it can pass to tools. Tool handlers
    can fetch the real data from GET /blobs/{hash}.
    """

    async def dispatch(self, request, call_next):
        if request.url.path != _CHAT_PATH:
            return await call_next(request)

        valkey = request.app.state.valkey
        if valkey is None:
            return await call_next(request)

        # ── Request: extract blobs, store in Valkey, replace with placeholders
        try:
            body_bytes = await request.body()
            import json
            body = json.loads(body_bytes)
        except Exception:
            return await call_next(request)

        conversation_id = body.get("conversation_id") or "anon"
        messages = body.get("messages", [])
        if not isinstance(messages, list):
            return await call_next(request)

        # Find base64 content and store in Valkey
        blobs = _extract_base64_parts(messages)
        if not blobs:
            return await call_next(request)

        blob_map: dict[str, tuple[str, str]] = {}
        for content_hash, raw_data, mime_type in blobs:
            blob_map[content_hash] = (raw_data, mime_type)
            if len(raw_data) > _MAX_BLOB_SIZE:
                logger.warning("blob_too_large conv=%s hash=%s size=%d",
                               conversation_id[:12], content_hash, len(raw_data))
                continue
            try:
                await valkey.set(
                    f"{BLOB_PREFIX}:{conversation_id}:{content_hash}",
                    raw_data,
                    ex=BLOB_TTL,
                )
                logger.debug("blob_store conv=%s hash=%s mime=%s",
                             conversation_id[:12], content_hash, mime_type)
            except Exception:
                continue

        # Replace base64 with placeholders
        body["messages"] = _replace_base64_with_placeholder(messages, blob_map)
        request._body = json.dumps(body).encode()

        # ── Call handler ─────────────────────────────────────────────────
        response = await call_next(request)

        return response
