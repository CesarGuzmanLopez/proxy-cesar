"""GET /health — Service health check.

sprint §11 — no auth required, never returns 500.
"""

import os
import time
from urllib.parse import urlparse

from fastapi import APIRouter, Request
from sqlalchemy import select, text

from src.adapters.db.models import ConversationSnapshot
from src.config.settings import settings

router = APIRouter()


@router.get("/health")
async def health(request: Request):
    """Return health status of proxy and its dependencies."""
    app_state = request.app.state
    config = app_state.config

    # Check database
    db_status = "connected"
    try:
        async with app_state.db_session_factory() as session:
            await session.execute(text("SELECT 1"))
    except Exception:
        db_status = "disconnected"

    # Check Valkey
    valkey_status = "connected"
    valkey_latency_ms = None
    try:
        t0 = time.monotonic()
        await app_state.valkey.ping()
        valkey_latency_ms = round((time.monotonic() - t0) * 1000, 2)
    except Exception:
        valkey_status = "disconnected"

    # DB file size
    db_size_kb = None
    parsed = urlparse(settings.database_url)
    if parsed.path and os.path.exists(parsed.path):
        db_size_kb = round(os.path.getsize(parsed.path) / 1024, 2)

    # arq worker status
    arq_worker = "available" if getattr(app_state, "arq_pool", None) else "unavailable"

    # Last compaction timestamp
    last_compaction = None
    try:
        async with app_state.db_session_factory() as session:
            result = await session.execute(
                select(ConversationSnapshot.created_at)
                .order_by(ConversationSnapshot.created_at.desc())
                .limit(1)
            )
            row = result.scalar_one_or_none()
            last_compaction = row.isoformat() if row else None
    except Exception:
        pass

    # Check providers (API keys configured)
    providers = {}
    for provider, key_env in [
        ("pruna", "PRUNA_API_KEY"),
        ("openrouter", "OPENROUTER_API_KEY"),
        ("deepseek", "DEEPSEEK_API_KEY"),
        ("groq", "GROQ_API_KEY"),
        ("opencode-go", "OPENCODE_API_KEY"),
    ]:
        providers[provider] = "configured" if os.getenv(key_env) else "not configured"

    overall = (
        "ok"
        if db_status == "connected" and valkey_status == "connected"
        else "degraded"
    )

    return {
        "status": overall,
        "database": db_status,
        "valkey": valkey_status,
        "valkey_latency_ms": valkey_latency_ms,
        "db_size_kb": db_size_kb,
        "arq_worker": arq_worker,
        "last_compaction": last_compaction,
        "providers": providers,
        "disabled_providers": settings.disabled_providers or "none",
        "pseudo_models_loaded": len(config.pseudo_models),
    }
