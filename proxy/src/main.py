"""FastAPI application entry point.

sprint §12 — Lifespan manages startup/shutdown of all services.
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn

# ── Logging configuration ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-36s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
# Reduce noisy third-party logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
from fastapi import FastAPI
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.adapters.cache.valkey_affinity import ValkeyAffinityAdapter, setup_valkey
from src.adapters.litellm.client import setup_litellm
from src.api.chat import router as chat_router
from src.api.conversations import router as conversations_router
from src.api.health import router as health_router
from src.api.models import router as models_router
from src.config.pseudo_models import load_config
from src.config.settings import settings

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
        settings.database_url, echo=False,
    )
    # Create all tables (SQLite-friendly, no Alembic needed)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
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
    from src.tasks.arq_app import create_arq_pool
    arq_pool = await create_arq_pool()
    app.state.arq_pool = arq_pool
    if arq_pool:
        print("arq pool created — async compaction available")
    else:
        print("arq pool not available — compaction runs synchronously")

    # Sprint 5: Optional BERT router classifier (loaded at startup, fast local eval)
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

# Register routers
app.include_router(chat_router)
app.include_router(conversations_router)
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
