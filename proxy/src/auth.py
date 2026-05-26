"""Bearer token authentication middleware.

Sprint 8 §2: validates PROXY_API_KEY on all non-public endpoints.
Accepts OPENCODE_API_KEY as an additional valid credential so OpenCode
clients can authenticate without a separate PROXY_API_KEY env var.
When neither key is set → dev mode (auth disabled).
"""

import logging
import os

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

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


def _valid_api_keys() -> list[str]:
    """Return all accepted API keys (non-empty values only)."""
    return [
        k for k in [
            os.getenv("PROXY_API_KEY", ""),
            os.getenv("OPENCODE_API_KEY", ""),
        ] if k
    ]


class AuthMiddleware(BaseHTTPMiddleware):
    """Validate Bearer token on all non-public endpoints.

    Registration order in main.py: AuthMiddleware FIRST.
    """

    async def dispatch(self, request: Request, call_next):
        # Public paths — no auth required (handle trailing slash)
        normalized_path = request.url.path.rstrip("/") or "/"
        if normalized_path in PUBLIC_PATHS:
            return await call_next(request)

        # Dev mode — auth disabled if no API key is configured
        valid_keys = _valid_api_keys()
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
