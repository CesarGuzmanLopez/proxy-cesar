"""Rate limiting middleware — currently disabled.

All requests pass through unconditionally.
To re-enable, restore the dispatch logic and imports from git history.
"""

import logging

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiter — currently disabled. All requests pass through.

    To re-enable, restore the dispatch logic from git history.
    """

    async def dispatch(self, request: Request, call_next):
        return await call_next(request)
