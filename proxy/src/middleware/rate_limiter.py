"""Rate limiting middleware per pseudo-model (Sprint 8 §4).

Sliding-window rate limiter using Valkey sorted sets.
Key: ratelimit:{ip}:{pseudo_model}
Scans first 64 KB of request body to extract model name —
avoids full parse of multi-MB base64 payloads.
"""

import logging
import os
import re
import time
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# Lightweight scan: find "model":"<value>" in raw body bytes
# Avoids full json.loads() of large chat bodies (images, tools, etc.)
_MODEL_RE = re.compile(rb'"model"\s*:\s*"([^"]+)"', re.IGNORECASE)

# Scan up to 64 KB — covers virtually all real-world payloads.
# The model field always appears near the start of the JSON body.
# 2 KB was too small for requests with very long system prompts.
_MAX_BODY_SCAN = 65536

# Default rate limits: requests per minute per pseudo-model
_DEFAULT_RATE_LIMITS: dict[str, int] = {
    "pensamiento-profundo-caro": 5,
    "tareas-avanzadas": 20,
    "normal": 60,
    "codigo-preciso": 40,
    "massive-fast": 200,
    "flash-lowcost": 200,
    "vision": 15,
    "compactador": 5,
}

_DEFAULT_LIMIT = 60  # for unknown pseudo-models

_WINDOW_SECONDS = 60


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
    """Sliding-window rate limiter per pseudo-model in Valkey sorted sets.

    Reads the request body to determine the pseudo-model.
    Body is cached by Starlette after first read, so downstream handlers
    can still access it without issues.
    """

    async def dispatch(self, request: Request, call_next):
        _trace = str(uuid.uuid4())[:8]

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

        # Get client IP for per-user rate limiting (B13).
        # Backward-compatible: installations without proxy will use client.host.
        forwarded = request.headers.get("x-forwarded-for")
        client_ip = (forwarded or request.client.host) if request.client else "0.0.0.0"

        key = f"ratelimit:{client_ip}:{pseudo_model}"

        valkey = request.app.state.valkey
        if valkey is None:
            return await call_next(request)

        try:
            now = time.time()
            window_start = now - _WINDOW_SECONDS

            await valkey.zremrangebyscore(key, "-inf", window_start)

            member = str(now)
            await valkey.zadd(key, {member: now})

            count = await valkey.zcard(key)
            await valkey.expire(key, 120)

            reset_seconds = _WINDOW_SECONDS
            oldest = await valkey.zrange(key, 0, 0, withscores=True)
            if oldest:
                oldest_score = oldest[0][1]
                reset_seconds = max(1, int(oldest_score + _WINDOW_SECONDS - now))

            if count > limit:
                logger.warning(
                    "rate_limit_hit trace=%s | pseudo=%s limit=%d count=%d",
                    _trace,
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
                        "retry_after_seconds": reset_seconds,
                    },
                    headers={"Retry-After": str(reset_seconds)},
                )
        except Exception as exc:
            logger.error("rate_limit_error trace=%s: %s", _trace, exc)
            return await call_next(request)

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(max(0, limit - count))
        response.headers["X-RateLimit-Reset"] = str(int(now + reset_seconds))
        return response
