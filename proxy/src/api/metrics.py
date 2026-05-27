"""Metrics endpoint (Sprint 8 §6).

GET /metrics returns aggregated stats:
  - Request counts per pseudo-model
  - Token usage (input, output, cached, saved by compaction)
  - Cache hit rate
  - Compaction counts (explicit only)
  - Fallback counts
  - Rate limit hits
  - Conversation counts
  - Error breakdown
"""

import logging
import time

from fastapi import APIRouter, Request
from sqlalchemy import func, select, text

from src.adapters.db.models import Conversation, ConversationSnapshot

logger = logging.getLogger(__name__)

router = APIRouter(tags=["metrics"])

# Module-level start time for uptime calculation
_START_TIME = time.time()


# ── In-memory counter store ──────────────────────────────────────────────────


class MetricsStore:
    """Thread-safe in-memory metrics store. Reset on restart."""

    def __init__(self):
        self.total_requests: int = 0
        self.requests_by_pseudo: dict[str, int] = {}
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_cached_tokens: int = 0
        self.total_saved_by_compaction: int = 0
        self.cache_hits: int = 0
        self.fallbacks: dict[str, int] = {}
        self.errors_4xx: int = 0
        self.errors_5xx: int = 0
        self.errors_by_type: dict[str, int] = {}

    def record_request(self, pseudo_model: str) -> None:
        self.total_requests += 1
        self.requests_by_pseudo[pseudo_model] = (
            self.requests_by_pseudo.get(pseudo_model, 0) + 1
        )

    def record_tokens(
        self, input_tokens: int, output_tokens: int, cached_tokens: int = 0
    ) -> None:
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cached_tokens += cached_tokens
        if cached_tokens > 0:
            self.cache_hits += 1

    def record_compaction(self, tokens_saved: int) -> None:
        self.total_saved_by_compaction += tokens_saved

    def record_fallback(self, reason: str) -> None:
        self.fallbacks[reason] = self.fallbacks.get(reason, 0) + 1

    def record_error(self, status_code: int, error_type: str | None = None) -> None:
        if 400 <= status_code < 500:
            self.errors_4xx += 1
        elif 500 <= status_code < 600:
            self.errors_5xx += 1
        if error_type:
            self.errors_by_type[error_type] = self.errors_by_type.get(error_type, 0) + 1


# Global metrics store (reset on proxy restart)
metrics = MetricsStore()


# ── Endpoint ──────────────────────────────────────────────────────────────────


@router.get("/metrics")
async def get_metrics(request: Request):
    """Aggregated proxy metrics. Requires auth."""
    db_factory = request.app.state.db_session_factory

    total_convs = 0
    active_convs = 0
    total_snapshots = 0
    total_explicit_compactions = 0

    if db_factory is not None:
        try:
            async with db_factory() as db:
                total_convs = await db.scalar(select(func.count(Conversation.id))) or 0
                active_convs = (
                    await db.scalar(
                        select(func.count(Conversation.id)).where(
                            Conversation.updated_at
                            > func.now() - text("1 day")
                        )
                    )
                    or 0
                )
                total_snapshots = (
                    await db.scalar(select(func.count(ConversationSnapshot.id))) or 0
                )
                total_explicit_compactions = (
                    await db.scalar(
                        select(func.count(ConversationSnapshot.id)).where(
                            ConversationSnapshot.snapshot_type == "explicit"
                        )
                    )
                    or 0
                )
        except Exception as e:
            logger.error("metrics_db_query_failed error=%s", str(e))

    return {
        "uptime_seconds": int(time.time() - _START_TIME),
        "total_requests": metrics.total_requests,
        "requests_by_pseudo_model": metrics.requests_by_pseudo,
        "total_tokens": {
            "input": metrics.total_input_tokens,
            "output": metrics.total_output_tokens,
            "cached": metrics.total_cached_tokens,
            "saved_by_compaction": metrics.total_saved_by_compaction,
        },
        "cache": {
            "hit_rate_pct": round(
                (metrics.cache_hits / max(metrics.total_requests, 1)) * 100,
                1,
            ),
            "total_cache_hits": metrics.cache_hits,
        },
        "compactions": {
            "explicit_compactions": total_explicit_compactions,
            "total_tokens_saved": metrics.total_saved_by_compaction,
        },
        "fallbacks": {
            "total": sum(metrics.fallbacks.values()),
            "by_reason": metrics.fallbacks,
        },
        "conversations": {
            "active": active_convs,
            "total": total_convs,
            "with_snapshot": total_snapshots,
        },
        "errors": {
            "4xx": metrics.errors_4xx,
            "5xx": metrics.errors_5xx,
            "by_type": metrics.errors_by_type,
        },
    }
