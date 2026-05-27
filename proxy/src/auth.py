"""Bearer token authentication middleware.

Sprint 8 §2: validates PROXY_API_KEY on all non-public endpoints.
When PROXY_API_KEY is empty/missing → dev mode (auth disabled).
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


def _get_api_key() -> str:
    """Resolve API key with runtime override support.

    Primary source is settings.proxy_api_key (from .env / env vars at startup).
    Falls back to os.environ for test compatibility (monkeypatch after import).
    """
    key = settings.proxy_api_key
    if not key:
        key = os.environ.get("PROXY_API_KEY", "")
    return key


class AuthMiddleware(BaseHTTPMiddleware):
    """Validate Bearer token on all non-public endpoints.

    Registration order in main.py: AuthMiddleware FIRST.
    """

    async def dispatch(self, request: Request, call_next):
        # Public paths — no auth required (handle trailing slash)
        normalized_path = request.url.path.rstrip("/") or "/"
        if normalized_path in PUBLIC_PATHS:
            return await call_next(request)

        # Dev mode — auth disabled if proxy_api_key is not set
        api_key = _get_api_key()
        if not api_key:
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
        if token != api_key:
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
