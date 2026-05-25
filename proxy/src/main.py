"""FastAPI application entry point.

sprint §12 — Lifespan manages startup/shutdown of all services.
Sprint 8 — Auth, CORS, rate limiting, structured logging, metrics.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy import text as sa_text

from src.adapters.cache.valkey_affinity import ValkeyAffinityAdapter, setup_valkey
from src.adapters.litellm.client import setup_litellm
from src.api.chat import router as chat_router
from src.api.conversations import router as conversations_router
from src.api.health import router as health_router
from src.api.metrics import router as metrics_router
from src.api.models import router as models_router
from src.auth import AuthMiddleware
from src.config.pseudo_models import load_config
from src.config.settings import settings
from src.logging_config import setup_logging
from src.middleware.rate_limiter import RateLimitMiddleware

# Configure structured JSON logging (Sprint 8)
setup_logging(level=os.getenv("LOG_LEVEL", "INFO"))

# Determine config path relative to project root
CONFIG_PATH = Path(__file__).resolve().parent.parent / "pseudo_models.yaml"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan — startup and shutdown."""
    # ── STARTUP ──────────────────────────────────────────────────────────
    print(f"Loading pseudo_models.yaml from {CONFIG_PATH}...")
    config = load_config(CONFIG_PATH)
    app.state.config = config
    print(f"Loaded {len(config.pseudo_models)} pseudo-models")

    # Database
    engine = create_async_engine(
        settings.database_url,
        echo=False,
    )
    # Create all tables (SQLite-friendly) + migrate existing DB
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
        # Migrate existing SQLite DB: add columns that may be missing
        _MIGRATIONS_SQLITE = [
            "ALTER TABLE conversations ADD COLUMN images_described INTEGER DEFAULT 0",
            "ALTER TABLE conversations ADD COLUMN images_degraded_manually INTEGER DEFAULT 0",
        ]
        for stmt in _MIGRATIONS_SQLITE:
            try:
                await conn.execute(sa_text(stmt))
            except Exception:
                pass  # column already exists — ignore
    app.state.db_session_factory = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    # Valkey
    valkey_client = await setup_valkey(settings)
    app.state.valkey = valkey_client
    app.state.affinity = ValkeyAffinityAdapter(valkey_client)

    # LiteLLM
    setup_litellm(settings)

    # Sprint 6: Optional arq pool for async compaction
    arq_pool = None
    try:
        from src.tasks.arq_app import create_arq_pool as _create_arq_pool

        arq_pool = await _create_arq_pool()
        if arq_pool:
            print("arq pool created — async compaction available")
        else:
            print("arq pool not available — compaction runs synchronously")
    except Exception:
        print("arq not available — compaction runs synchronously")
    app.state.arq_pool = arq_pool

    # Sprint 5: Optional BERT router classifier
    from src.service.router_llm.suggester import load_bert_classifier

    bert_loaded = load_bert_classifier()
    if bert_loaded:
        print("BERT router classifier loaded — fast local routing enabled")
    else:
        print("BERT router classifier not loaded — using LLM-based routing (slower)")

    print(f"Proxy ready on port {settings.proxy_port}")

    yield

    # ── SHUTDOWN ─────────────────────────────────────────────────────────
    if hasattr(app.state, "arq_pool") and app.state.arq_pool:
        await app.state.arq_pool.close()
    await valkey_client.close()
    await engine.dispose()
    print("Proxy shut down")


app = FastAPI(
    title="Proxy Determinista Multi-Modelo",
    version="0.1.0",
    lifespan=lifespan,
)

# ── Sprint 8: Middleware registration (order matters!) ──────────────────────

# FastAPI wraps middleware in LIFO order: last registered = outermost (runs first).
# Current order: 1st=RateLimit(inner) → 2nd=Auth(middle) → 3rd=CORS(outer/first)
# Execution: CORS → Auth → RateLimit → handler → RateLimit → Auth → CORS
# Auth rejects unauthenticated requests BEFORE RateLimit counts them.

# 1. Rate limiting — innermost (runs last, after auth)
app.add_middleware(RateLimitMiddleware)

# 2. Auth — middle (runs before rate limiting)
app.add_middleware(AuthMiddleware)

# 3. CORS — outermost (runs first)
origins_str = os.getenv("CORS_ORIGINS", "http://localhost:3000")
origins = [o.strip() for o in origins_str.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Conversation-ID"],
)

# ── Routers ─────────────────────────────────────────────────────────────────

app.include_router(chat_router)
app.include_router(conversations_router)
app.include_router(metrics_router)
app.include_router(models_router)
app.include_router(health_router)


def main() -> None:
    """Entry point for `uvicorn src.main:app` or `python -m src.main`."""
    uvicorn.run(
        "src.main:app",
        host=settings.proxy_host,
        port=settings.proxy_port,
        reload=True,
    )


if __name__ == "__main__":
    main()
