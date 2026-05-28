"""arq worker for async compaction of large conversations (>500K tokens).

Sprint 6 §4: Replaces Celery with arq (MIT, async-native, by Pydantic creator).
Uses Valkey (Redis-compatible) as broker — already deployed.

python.md §7: async-first.

Usage:
    arq src.tasks.arq_app.WorkerSettings
"""

import logging

try:
    from arq import create_pool as _create_pool_impl
    from arq.connections import RedisSettings as _RedisSettingsImpl

    HAS_ARQ = True
except ImportError:
    HAS_ARQ = False
    _create_pool_impl = None
    _RedisSettingsImpl = None

logger = logging.getLogger(__name__)


# ── Worker function (discovered by name) ──────────────────────────────────


async def compact_conversation_async(
    ctx,
    conversation_id: str,
    compactor_model: str,
    api_base: str | None = None,
    api_key: str | None = None,
):
    """Async compaction task for very large conversations.

    Called by arq worker when a compaction job is dequeued.
    Self-contained: recreates DB factory and config because the
    worker runs in a *separate process* (arq-recommended pattern).

    Args:
        ctx: arq worker context (unused — kept for arq signature).
        conversation_id: UUID string of the conversation to compact.
        compactor_model: Physical model name to use for compaction.
        api_base: Custom API base URL (e.g. for OpenCode Go models).
        api_key: Custom API key (resolved from api_key_env).

    Returns:
        Dict with compaction result metadata.
    """
    from pathlib import Path

    from sqlmodel.ext.asyncio.session import AsyncSession
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from src.config.pseudo_models import load_config
    from src.config.settings import settings
    from src.service.compactor.explicit import _compact_async

    config = load_config(
        Path(__file__).resolve().parent.parent.parent / "pseudo_models.yaml"
    )
    engine = create_async_engine(
        settings.database_url,
        echo=False,
        connect_args={"check_same_thread": False}
        if "sqlite" in settings.database_url
        else {},
    )
    db_session_factory = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    try:
        result = await _compact_async(
            conversation_id=conversation_id,
            compactor_model=compactor_model,
            db_session_factory=db_session_factory,
            config=config,
            api_base=api_base,
            api_key=api_key,
        )
        return result
    finally:
        await engine.dispose()


# ── Worker settings for CLI (only defined when arq is available) ──────────


if HAS_ARQ:
    from arq.connections import RedisSettings

    class WorkerSettings:
        """arq worker settings for `arq src.tasks.arq_app.WorkerSettings`."""

        functions = [compact_conversation_async]
        redis_settings = RedisSettings.from_dsn("valkey://localhost:6379")
        keep_result = 3600  # Keep results for 1 hour
        max_jobs = 4  # Max concurrent compaction jobs
        job_timeout = 300  # 5 minutes max per compaction job


# ── Helper to create pool in FastAPI lifespan ────────────────────────────


async def create_arq_pool():
    """Create arq pool if arq and Valkey are available.

    Returns:
        ArqRedis pool or None if arq/Valkey unavailable (compaction runs synchronously).
    """
    if not HAS_ARQ:
        return None
    from arq import create_pool
    from arq.connections import RedisSettings

    try:
        pool = await create_pool(RedisSettings.from_dsn("valkey://localhost:6379"))
        return pool
    except Exception as exc:
        logger.debug("arq_pool_create_failed err=%s", exc)
        return None
