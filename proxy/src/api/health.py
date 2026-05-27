"""GET /health — Service health check.

sprint §11 — no auth required, never returns 500.
"""

import os

from fastapi import APIRouter, Request
from sqlalchemy import text

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
    try:
        await app_state.valkey.ping()
    except Exception:
        valkey_status = "disconnected"

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
        "providers": providers,
        "disabled_providers": settings.disabled_providers or "none",
        "pseudo_models_loaded": len(config.pseudo_models),
    }
