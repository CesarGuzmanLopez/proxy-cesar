"""FastAPI application entry point.

sprint §12 — Lifespan manages startup/shutdown of all services.
Sprint 8 — Auth, CORS, rate limiting, structured logging, metrics.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# ── SSL_CERT_FILE must be set *before* any module imports httpx/litellm ────
# Check KeyClaw combined CA first, then system CA
_keyclaw_combined = Path.home() / ".keyclaw" / "combined-ca.pem"
if not os.environ.get("SSL_CERT_FILE"):
    for candidate in (
        os.environ.get("NIX_SSL_CERT_FILE", ""),
        str(_keyclaw_combined) if _keyclaw_combined.exists() else "",
        "/etc/ssl/certs/ca-certificates.crt",
        "/etc/ssl/cert.pem",
    ):
        if candidate and Path(candidate).exists():
            os.environ["SSL_CERT_FILE"] = candidate
            break

from contextlib import asynccontextmanager  # noqa: E402 — SSL code above must run first

import uvicorn  # noqa: E402

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from sqlmodel import SQLModel  # noqa: E402
from sqlmodel.ext.asyncio.session import AsyncSession  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402
from sqlalchemy import inspect as sa_inspect  # noqa: E402

from src.adapters.cache.valkey_affinity import ValkeyAffinityAdapter, setup_valkey  # noqa: E402
from src.adapters.litellm.client import setup_litellm  # noqa: E402
from src.api.chat import router as chat_router  # noqa: E402
from src.api.conversation_operations import router as conversation_operations_router  # noqa: E402
from src.api.conversations import router as conversations_router  # noqa: E402
from src.api.health import router as health_router  # noqa: E402
from src.api.metrics import metrics, router as metrics_router  # noqa: E402
from src.api.models import router as models_router  # noqa: E402

from src.auth import AuthMiddleware  # noqa: E402
from src.config.pseudo_models import load_config  # noqa: E402
from src.config.settings import settings  # noqa: E402
from src.logging_config import setup_logging  # noqa: E402
from src.middleware.rate_limiter import RateLimitMiddleware  # noqa: E402
from src.middleware.keyvault import KeyVaultMiddleware  # noqa: E402
from src.utils.sanitize import sanitize, sanitize_dict  # noqa: E402

# Configure structured JSON logging (Sprint 8)
setup_logging(level=os.getenv("LOG_LEVEL", "INFO"))

# Determine config path relative to project root
CONFIG_PATH = Path(__file__).resolve().parent.parent / "pseudo_models.yaml"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan — startup and shutdown."""
    # ── STARTUP ──────────────────────────────────────────────────────────
    # Verify mandatory dependencies
    try:
        import fitz  # noqa: F401 — PyMuPDF for PDF text extraction
    except ImportError:
        print("FATAL: PyMuPDF (fitz) is required. Install with: pip install PyMuPDF")
        raise SystemExit(1) from None

    print(f"Loading pseudo_models.yaml from {CONFIG_PATH}...")
    config = load_config(CONFIG_PATH)
    app.state.config = config
    print(f"Loaded {len(config.pseudo_models)} pseudo-models")

    # Database — SQLite WAL mode + busy timeout for concurrent access
    _db_url = settings.database_url
    if "sqlite" in _db_url:
        _db_url = _db_url.split("?")[0] + "?timeout=30"
    engine = create_async_engine(
        _db_url,
        echo=False,
        connect_args={"check_same_thread": False} if "sqlite" in _db_url else {},
    )
    # Enable WAL mode for SQLite
    if "sqlite" in _db_url:
        async with engine.begin() as conn:
            await conn.exec_driver_sql("PRAGMA journal_mode=WAL")
    # Create all tables (SQLite-friendly) + migrate existing DB
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

        def _migrate_columns(sync_conn):
            """Add columns that may be missing on existing SQLite databases.
            Uses inspector to check column existence first (safe pattern).

            This function is the single source of truth for schema migrations.
            SQLite does not support transactional DDL like PostgreSQL, and for
            this project's scale inline ALTER TABLE is simpler than maintaining
            a separate Alembic migration chain. All schema changes (new columns,
            new tables managed by SQLModel.metadata.create_all) live here.
            """
            inspector = sa_inspect(sync_conn)
            if "conversations" not in inspector.get_table_names():
                return
            existing_cols = {c["name"] for c in inspector.get_columns("conversations")}
            for col_name in ("images_described", "images_degraded_manually"):
                if col_name not in existing_cols:
                    sync_conn.exec_driver_sql(
                        f"ALTER TABLE conversations ADD COLUMN {col_name} "
                        "INTEGER DEFAULT 0"
                    )
                    logger.info("db_migration_applied column=%s", col_name)

        await conn.run_sync(_migrate_columns)
    app.state.db_session_factory = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    # Valkey
    valkey_client = await setup_valkey(settings)
    app.state.valkey = valkey_client
    app.state.affinity = ValkeyAffinityAdapter(valkey_client)

    # Metrics — persist to Valkey so counters survive restarts
    from src.api.metrics import metrics as _metrics
    _metrics.set_valkey(valkey_client)
    await _metrics.restore_from_valkey()

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
# Current order: 1st=KeyVault(inner) → 2nd=RateLimit → 3rd=Auth(middle) → 4th=CORS(outer)
# Execution: CORS → Auth → RateLimit → KeyVault → handler → KeyVault → RateLimit → Auth → CORS
# Auth rejects unauthenticated requests BEFORE RateLimit counts them.
# KeyVault sanitizes secrets closest to the handler.

# 1. KeyVault — innermost (closest to handler, sanitizes request/response bodies)
app.add_middleware(KeyVaultMiddleware)

# 2. Rate limiting — after keyvault, before auth
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

# ── Global exception handlers: sanitize API keys from all error responses ──


@app.exception_handler(HTTPException)
async def sanitize_http_exception(request, exc: HTTPException):
    metrics.record_error(exc.status_code, "HTTPException")
    detail = exc.detail
    if isinstance(detail, dict):
        detail = sanitize_dict(detail)
    elif isinstance(detail, str):
        detail = sanitize(detail)
    status = exc.status_code
    headers = getattr(exc, "headers", None)
    return JSONResponse(
        status_code=status,
        content={"detail": detail},
        headers=headers,
    )


@app.exception_handler(Exception)
async def sanitize_generic_exception(request, exc: Exception):
    metrics.record_error(500, "INTERNAL_ERROR")
    return JSONResponse(
        status_code=500,
        content={
            "detail": {"error": "INTERNAL_ERROR", "message": "Internal server error"}
        },
    )


# ── Routers ─────────────────────────────────────────────────────────────────

app.include_router(chat_router)
app.include_router(conversations_router)
app.include_router(conversation_operations_router)
app.include_router(metrics_router)
app.include_router(models_router)
app.include_router(health_router)


def main() -> None:
    """Entry point for `uvicorn src.main:app` or `python -m src.main`."""
    uvicorn.run(
        "src.main:app",
        host=settings.proxy_host,
        port=settings.proxy_port,
        reload=False,
    )


if __name__ == "__main__":
    main()
