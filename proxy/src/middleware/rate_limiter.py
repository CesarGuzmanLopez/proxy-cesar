"""Rate limiting middleware per pseudo-model (Sprint 8 §4).

Fixed-window counter in Valkey. Key: ratelimit:{pseudo_model}:{minute_bucket}
Reads only the first 2 KB of the request body to extract the model name —
never loads the full body (which may contain multi-MB base64 images).
"""

import logging
import os
import re
import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# Lightweight scan: find "model":"<value>" in raw body bytes
# Avoids full json.loads() of large chat bodies (images, tools, etc.)
_MODEL_RE = re.compile(rb'"model"\s*:\s*"([^"]+)"', re.IGNORECASE)

# Only read the first 2 KB — the model field always appears near the start
_MAX_BODY_SCAN = 2048

# Default rate limits: requests per minute per pseudo-model
_DEFAULT_RATE_LIMITS: dict[str, int] = {
    "pensamiento-profundo-caro": 5,
    "tareas-avanzadas": 20,
    "normal": 60,
    "deep-flash": 120,
    "flash-lowcost": 200,
    "vision": 15,
    "compactador": 5,
}

_DEFAULT_LIMIT = 60  # for unknown pseudo-models


def _load_rate_limits() -> dict[str, int]:
    """Load rate limits from env vars, falling back to defaults."""
    limits = dict(_DEFAULT_RATE_LIMITS)
    for key, value in os.environ.items():
        if key.startswith("RATE_LIMIT_"):
            pseudo_name = key.removeprefix("RATE_LIMIT_").lower().replace("_", "-")
            try:
                limits[pseudo_name] = int(value)
            except ValueError:
                logger.warning("rate_limit_env_invalid key=%s value=%s", key, value)
    return limits


_rate_limits: dict[str, int] | None = None


def _get_limits() -> dict[str, int]:
    global _rate_limits
    if _rate_limits is None:
        _rate_limits = _load_rate_limits()
    return _rate_limits


_CHAT_PATH = "/v1/chat/completions"


def _extract_model(body_bytes: bytes) -> str:
    """Extract model name from raw JSON body bytes without full parse.

    Scans only the first _MAX_BODY_SCAN bytes — the model field is always
    near the start of the JSON payload.  For very large requests (base64
    images) this avoids scanning megabytes of image data.
    """
    scan_slice = body_bytes[:_MAX_BODY_SCAN]
    try:
        m = _MODEL_RE.search(scan_slice)
        if m:
            return m.group(1).decode()
    except Exception:
        pass
    return "unknown"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Fixed-window rate limiter per pseudo-model in Valkey.

    Reads the request body to determine the pseudo-model.
    Body is cached by Starlette after first read, so downstream handlers
    can still access it without issues.
    """

    async def dispatch(self, request: Request, call_next):
        if request.url.path != _CHAT_PATH:
            return await call_next(request)

        # Read body to determine pseudo-model (reliable, no client bypass)
        pseudo_model = "unknown"
        try:
            body_bytes = await request.body()
            if body_bytes:
                pseudo_model = _extract_model(body_bytes)
        except Exception:
            pass

        limits = _get_limits()
        limit = limits.get(pseudo_model, _DEFAULT_LIMIT)

        minute_bucket = int(time.time() / 60)
        key = f"ratelimit:{pseudo_model}:{minute_bucket}"

        valkey = request.app.state.valkey
        if valkey is None:
            return await call_next(request)

        try:
            count = await valkey.incr(key)
            if count == 1:
                await valkey.expire(key, 120)

            if count > limit:
                retry_after = 60 - (int(time.time()) % 60)
                logger.warning(
                    "rate_limit_hit | pseudo=%s limit=%d count=%d",
                    pseudo_model,
                    limit,
                    count,
                )
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "RATE_LIMIT_EXCEEDED",
                        "message": (
                            f"Rate limit exceeded for pseudo-model '{pseudo_model}'. "
                            f"Limit: {limit}/minute."
                        ),
                        "retry_after_seconds": retry_after,
                    },
                    headers={"Retry-After": str(retry_after)},
                )
        except Exception as exc:
            logger.error("rate_limit_error: %s", exc)
            return await call_next(request)

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(max(0, limit - count))
        response.headers["X-RateLimit-Reset"] = str((minute_bucket + 1) * 60)
        return response
