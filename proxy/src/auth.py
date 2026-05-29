"""Bearer token authentication middleware.

Feature validates PROXY_API_KEY / PROXY_API_KEYS on all non-public endpoints.
PROXY_API_KEYS supports multiple comma-separated keys for multi-user access.
When all keys are empty/missing → dev mode (auth disabled).
"""

import logging
import os

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from src.config.settings import settings

logger = logging.getLogger(__name__)

# Paths that do NOT require authentication
PUBLIC_PATHS: frozenset[str] = frozenset(
    {
        "/health",
        "/docs",
        "/openapi.json",
        "/redoc",
    }
)


def _get_api_keys() -> list[str]:
    """Resolve all valid API keys.

    Sources (in order):
    1. settings.proxy_api_keys (comma-separated from .env)
    2. settings.proxy_api_key (single key, backward compat)
    3. os.environ (for test compatibility)
    """
    raw = os.environ.get("PROXY_API_KEYS", "") or os.environ.get("PROXY_API_KEY", "")
    if not raw:
        raw = getattr(settings, "proxy_api_keys", "") or getattr(settings, "proxy_api_key", "")
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    return keys


class AuthMiddleware(BaseHTTPMiddleware):
    """Validate Bearer token on all non-public endpoints.

    Accepts any token that matches one of the configured PROXY_API_KEYS.
    Registration order in main.py: AuthMiddleware FIRST.
    """

    async def dispatch(self, request: Request, call_next):
        # Public paths — no auth required (handle trailing slash)
        normalized_path = request.url.path.rstrip("/") or "/"
        if normalized_path in PUBLIC_PATHS:
            return await call_next(request)

        # Dev mode — auth disabled if no api keys configured
        valid_keys = _get_api_keys()
        if not valid_keys:
            return await call_next(request)

        # Validate Bearer token
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            logger.warning(
                "auth_failure | path=%s ip=%s reason=missing_header",
                request.url.path,
                request.client.host if request.client else "unknown",
            )
            return JSONResponse(
                status_code=401,
                content={
                    "error": "MISSING_AUTH",
                    "message": (
                        "Authorization header required. "
                        "Use: Authorization: Bearer <PROXY_API_KEY>"
                    ),
                },
            )

        token = auth_header.removeprefix("Bearer ").strip()
        if token not in valid_keys:
            logger.warning(
                "auth_failure | path=%s ip=%s reason=invalid_key",
                request.url.path,
                request.client.host if request.client else "unknown",
            )
            return JSONResponse(
                status_code=401,
                content={
                    "error": "INVALID_API_KEY",
                    "message": "The provided API key is invalid.",
                },
            )

        return await call_next(request)
