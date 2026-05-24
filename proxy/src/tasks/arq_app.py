"""arq worker for async compaction of large conversations (>500K tokens).

Sprint 6 §4: Replaces Celery with arq (MIT, async-native, by Pydantic creator).
Uses Valkey (Redis-compatible) as broker — already deployed.

python.md §7: async-first.

Usage:
    arq src.tasks.arq_app.WorkerSettings
"""

from arq import create_pool
from arq.connections import RedisSettings


# ── Worker function (discovered by name) ──────────────────────────────────


async def compact_conversation_async(ctx, conversation_id: str, compactor_model: str):
    """Async compaction task for very large conversations.

    Called by arq worker when a compaction job is dequeued.
    Creates its own DB session since it runs in a separate process.

    Args:
        ctx: arq worker context (contains db_session_factory, config).
        conversation_id: UUID string of the conversation to compact.
        compactor_model: Physical model name to use for compaction.

    Returns:
        Dict with compaction result metadata.
    """
    from src.service.compactor.explicit import _compact_async

    db_session_factory = ctx.get("db_session_factory")
    config = ctx.get("config")
    if not db_session_factory or not config:
        return {
            "status": "failed",
            "error": "arq context missing db_session_factory or config",
        }

    result = await _compact_async(
        conversation_id=conversation_id,
        compactor_model=compactor_model,
        db_session_factory=db_session_factory,
        config=config,
    )
    return result


# ── Worker settings for CLI ──────────────────────────────────────────────


class WorkerSettings:
    """arq worker settings for `arq src.tasks.arq_app.WorkerSettings`."""

    functions = [compact_conversation_async]
    redis_settings = RedisSettings.from_dsn("valkey://localhost:6379")
    keep_result = 3600  # Keep results for 1 hour
    max_jobs = 4  # Max concurrent compaction jobs
    job_timeout = 300  # 5 minutes max per compaction job


# ── Helper to create pool in FastAPI lifespan ────────────────────────────


async def create_arq_pool():
    """Create arq pool if Valkey is available.

    Returns:
        ArqRedis pool or None if Valkey is unavailable (compaction runs synchronously).
    """
    try:
        pool = await create_pool(RedisSettings.from_dsn("valkey://localhost:6379"))
        return pool
    except Exception:
        return None
