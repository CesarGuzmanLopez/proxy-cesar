"""Metrics endpoint (Sprint 8 §6).

GET /metrics returns aggregated stats persisted to Valkey (survives restarts).
In-memory counters serve as the fast path; Valkey persists every write
asynchronously so metrics survive proxy redeploys.
"""

import asyncio
import logging
import threading
import time

from fastapi import APIRouter, Request
from sqlalchemy import func, select, text

from src.adapters.db.models import Conversation, ConversationSnapshot

logger = logging.getLogger(__name__)

router = APIRouter(tags=["metrics"])

_START_TIME = time.time()
_METRICS_TTL = 7 * 86400  # 7 days


class MetricsStore:
    """Thread-safe metrics store with optional Valkey persistence."""

    def __init__(self):
        self._lock = threading.Lock()
        self._valkey: object | None = None
        self._bg_tasks: set[asyncio.Task[None]] = set()
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

    def set_valkey(self, client) -> None:
        """Attach a Valkey client for persistent metric storage."""
        self._valkey = client

    def record_request(self, pseudo_model: str) -> None:
        with self._lock:
            self.total_requests += 1
            self.requests_by_pseudo[pseudo_model] = (
                self.requests_by_pseudo.get(pseudo_model, 0) + 1
            )
        self._sync_key("total_requests", 1)
        self._sync_hash("requests_by_pseudo", pseudo_model)

    def record_tokens(
        self, input_tokens: int, output_tokens: int, cached_tokens: int = 0
    ) -> None:
        with self._lock:
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_cached_tokens += cached_tokens
            if cached_tokens > 0:
                self.cache_hits += 1
        self._sync_key("total_input_tokens", input_tokens)
        self._sync_key("total_output_tokens", output_tokens)
        self._sync_key("total_cached_tokens", cached_tokens)
        if cached_tokens > 0:
            self._sync_key("cache_hits", 1)

    def record_compaction(self, tokens_saved: int) -> None:
        with self._lock:
            self.total_saved_by_compaction += tokens_saved
        self._sync_key("total_saved_by_compaction", tokens_saved)

    def record_fallback(self, reason: str) -> None:
        with self._lock:
            self.fallbacks[reason] = self.fallbacks.get(reason, 0) + 1
        self._sync_hash("fallbacks", reason)

    def record_error(self, status_code: int, error_type: str | None = None) -> None:
        with self._lock:
            if 400 <= status_code < 500:
                self.errors_4xx += 1
            elif 500 <= status_code < 600:
                self.errors_5xx += 1
            if error_type:
                self.errors_by_type[error_type] = (
                    self.errors_by_type.get(error_type, 0) + 1
                )
        if 400 <= status_code < 500:
            self._sync_key("errors_4xx", 1)
        elif 500 <= status_code < 600:
            self._sync_key("errors_5xx", 1)

    # ── Valkey sync (best-effort, non-blocking) ──────────────────────

    def _add_bg_task(self, coro) -> None:
        """Schedule a background coroutine, keeping a reference to prevent GC."""
        task = asyncio.create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    def _sync_key(self, key: str, delta: int) -> None:
        v = self._valkey
        if v is None:
            return
        try:
            self._add_bg_task(self._incrby(v, f"metrics:{key}", delta, _METRICS_TTL))
        except RuntimeError:
            logger.debug("metrics_sync_key_no_event_loop key=%s", key)

    def _sync_hash(self, base: str, field: str) -> None:
        v = self._valkey
        if v is None:
            return
        try:
            self._add_bg_task(self._hincrby(v, f"metrics:{base}", field, _METRICS_TTL))
        except RuntimeError:
            logger.debug(
                "metrics_sync_hash_no_event_loop base=%s field=%s", base, field
            )

    @staticmethod
    async def _incrby(v, key: str, delta: int, ttl: int) -> None:
        try:
            await v.incrby(key, delta)
            await v.expire(key, ttl, nx=True)
        except Exception as exc:
            logger.debug("metrics_incrby_error key=%s err=%s", key, exc)

    @staticmethod
    async def _hincrby(v, key: str, field: str, ttl: int) -> None:
        try:
            await v.hincrby(key, field, 1)
            await v.expire(key, ttl, nx=True)
        except Exception as exc:
            logger.debug(
                "metrics_hincrby_error key=%s field=%s err=%s", key, field, exc
            )

    async def restore_from_valkey(self) -> None:
        """Restore in-memory counters from Valkey on startup."""
        v = self._valkey
        if v is None:
            return
        try:
            with self._lock:
                t = await v.get("metrics:total_requests")
                if t:
                    self.total_requests = int(t)
                for k in (
                    "total_input_tokens",
                    "total_output_tokens",
                    "total_cached_tokens",
                    "total_saved_by_compaction",
                    "cache_hits",
                    "errors_4xx",
                    "errors_5xx",
                ):
                    val = await v.get(f"metrics:{k}")
                    if val:
                        setattr(self, k, int(val))

                rbp = await v.hgetall("metrics:requests_by_pseudo")
                self.requests_by_pseudo = {
                    k.decode(): int(v) for k, v in (rbp or {}).items()
                }

                fb = await v.hgetall("metrics:fallbacks")
                self.fallbacks = {k.decode(): int(v) for k, v in (fb or {}).items()}

                eb = await v.hgetall("metrics:errors_by_type")
                self.errors_by_type = {
                    k.decode(): int(v) for k, v in (eb or {}).items()
                }
        except Exception as exc:
            logger.debug("metrics_valkey_restore_failed error=%s", exc)


# Global metrics store
metrics = MetricsStore()


# ── Endpoint ──────────────────────────────────────────────────────


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
                            Conversation.updated_at > func.now() - text("1 day")
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
