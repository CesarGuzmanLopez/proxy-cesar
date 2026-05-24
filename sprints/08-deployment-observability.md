# Sprint 8 — Deployment & Observability

> **Duration:** 1 week
> **Goal:** The proxy is production-ready. Auth, CORS, rate limiting, structured logging, and metrics. A new user sets it up in under 5 minutes.
> **Success criterion:** New user copies README, configures proxy in <5 min, connects OpenCode without modifying OpenCode. Rate limiting works. Metrics show token savings and cache hit rate.

---

## 1. Dependencies from Previous Sprints

| Dependency | Source | Status |
|---|---|---|
| All API endpoints | Sprint 1-7 | Complete |
| Valkey connection | Sprint 1 | Already working |
| `proxy_metadata` in responses | Sprint 1-7 | Complete |
| OpenCode config example | Sprint 7 | Ready for README |
| `.env.example` | Sprint 1 | Ready for extension |

### 1.1 New files/modules

```
src/
├── auth.py                      # NEW — Bearer token validation middleware
│
├── middleware/
│   └── rate_limiter.py          # NEW — rate limiting by pseudo-model in Valkey
│
├── api/
│   └── metrics.py               # NEW — GET /metrics endpoint
│
├── logging_config.py            # NEW — structured JSON logging setup
│
├── Caddyfile.example            # NEW — Caddy reverse proxy config for HTTPS
│
└── tests/
    ├── test_auth.py             # NEW
    ├── test_rate_limiter.py     # NEW
    └── test_metrics.py          # NEW

README.md                        # UPDATE — full deployment guide
```

### 1.2 Env changes

Add to `.env.example`:
```bash
# Auth
PROXY_API_KEY=sk-proxy-change-me-in-production

# CORS
CORS_ORIGINS=http://localhost:3000,https://tudominio.com

# Rate limiting (tokens per minute per pseudo-model)
RATE_LIMIT_pensamiento-profundo-caro=5
RATE_LIMIT_tareas-avanzadas=20
RATE_LIMIT_normal=60
RATE_LIMIT_deep-flash=120
RATE_LIMIT_flash-lowcost=200
RATE_LIMIT_avanzada-vision=10
RATE_LIMIT_flash-vision=30
RATE_LIMIT_compactador=5
```

---

## 2. Authentication (`auth.py`)

### 2.1 Middleware

```python
# src/auth.py

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import os

PUBLIC_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}


class AuthMiddleware(BaseHTTPMiddleware):
    """Validate Bearer token on all non-public endpoints."""

    async def dispatch(self, request: Request, call_next):
        # Skip auth for public paths
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        # Skip auth if PROXY_API_KEY is not set (dev mode)
        api_key = os.getenv("PROXY_API_KEY", "")
        if not api_key:
            return await call_next(request)

        # Validate Bearer token
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "MISSING_AUTH",
                    "message": "Authorization header required. Use: Authorization: Bearer <PROXY_API_KEY>"
                }
            )

        token = auth_header.removeprefix("Bearer ").strip()
        if token != api_key:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "INVALID_API_KEY",
                    "message": "The provided API key is invalid."
                }
            )

        return await call_next(request)
```

### 2.2 Registration in main.py

```python
from src.auth import AuthMiddleware

app.add_middleware(AuthMiddleware)
```

### 2.3 Behavior

- `/health` — NO auth (always public)
- `/docs`, `/openapi.json`, `/redoc` — NO auth (dev convenience, can be toggled)
- All `/v1/*` and `/conversations/*` endpoints — REQUIRE Bearer token
- If `PROXY_API_KEY` is empty/missing → auth disabled (development mode)
- Invalid key → 401 with descriptive error
- Missing header → 401 with descriptive error

### 2.4 What auth does NOT do

- No user management (single API key)
- No JWT, OAuth, or session tokens
- No role-based access (single key = full access)
- No key rotation mechanism
- No rate limiting per API key (only global per pseudo-model)

---

## 3. CORS

### 3.1 Configuration

```python
# src/main.py

from fastapi.middleware.cors import CORSMiddleware

origins_str = os.getenv("CORS_ORIGINS", "http://localhost:3000")
origins = [o.strip() for o in origins_str.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Conversation-ID"],
)
```

### 3.2 Default origins

- `http://localhost:3000` — local development
- `vscode-webview://*` — VS Code webviews
- Production origins configured via `CORS_ORIGINS` env var

---

## 4. Rate Limiting (`middleware/rate_limiter.py`)

### 4.1 Strategy

Rate limiting is per pseudo-model, using a sliding window in Valkey:

```
Key:     ratelimit:{pseudo_model}:{minute_bucket}
Value:   counter (integer)
TTL:     120 seconds (2 minutes — covers the window)
```

### 4.2 Implementation

```python
# src/middleware/rate_limiter.py

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import os
import time

RATE_LIMITS = {
    "pensamiento-profundo-caro": 5,
    "tareas-avanzadas": 20,
    "normal": 60,
    "deep-flash": 120,
    "flash-lowcost": 200,
    "avanzada-vision": 10,
    "flash-vision": 30,
    "compactador": 5,
}


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Only rate-limit chat completions
        if request.url.path != "/v1/chat/completions":
            return await call_next(request)

        # Read body to determine pseudo-model
        # FastAPI caches the body, so we can read it here
        body = await request.json()
        pseudo_model = body.get("model", "unknown")

        limit = RATE_LIMITS.get(pseudo_model, 60)  # Default 60/min

        valkey = request.app.state.valkey
        minute_bucket = int(time.time() / 60)
        key = f"ratelimit:{pseudo_model}:{minute_bucket}"

        # Increment counter
        count = await valkey.incr(key)
        if count == 1:
            await valkey.expire(key, 120)  # TTL 2 minutes

        # Check limit
        if count > limit:
            retry_after = 60 - (int(time.time()) % 60)
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "RATE_LIMIT_EXCEEDED",
                    "message": f"Rate limit exceeded for pseudo-model '{pseudo_model}'. Limit: {limit}/minute.",
                    "retry_after_seconds": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )

        # Add rate limit headers to response
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(max(0, limit - count))
        response.headers["X-RateLimit-Reset"] = str((minute_bucket + 1) * 60)

        return response
```

### 4.3 Registration order in main.py

```python
# Order matters!
app.add_middleware(AuthMiddleware)         # 1. Auth first
app.add_middleware(RateLimitMiddleware)    # 2. Rate limiting second
app.add_middleware(CORSMiddleware)         # 3. CORS last (outermost)
```

### 4.4 What rate limiting does NOT do

- No per-user or per-API-key limits (single global limit)
- No burst allowance
- No IP-based limiting
- No dynamic limit adjustment
- No rate limit persistence across proxy restarts (Valkey handles this)

---

## 5. Structured Logging (`logging_config.py`)

### 5.1 Configuration

```python
# src/logging_config.py

import logging
import json
import sys
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields if present
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)

        return json.dumps(log_entry, default=str)


def setup_logging(level: str = "INFO"):
    """Configure structured JSON logging to stdout."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    root_logger.handlers = [handler]  # Replace default handlers

    # Suppress noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    return root_logger
```

### 5.2 What gets logged (structured)

Every significant proxy decision must be logged as a JSON line:

```json
{"timestamp": "2026-01-15T10:00:00Z", "level": "INFO", "message": "Affinity maintained", "conversation_id": "abc-123", "physical_model": "qwen3-max", "pseudo_model": "normal", "turn": 12}
{"timestamp": "2026-01-15T10:00:01Z", "level": "WARNING", "message": "Fallback applied", "conversation_id": "abc-456", "failed_model": "qwen3-max", "fallback_model": "deepseek-v4-flash", "reason": "upstream_503"}
{"timestamp": "2026-01-15T10:00:02Z", "level": "INFO", "message": "Pre-compaction applied", "conversation_id": "abc-789", "original_tokens": 80000, "compacted_tokens": 6000, "compactor": "glm-4.5-flash"}
{"timestamp": "2026-01-15T10:00:03Z", "level": "ERROR", "message": "Switch blocked", "conversation_id": "abc-999", "from": "avanzada-vision", "to": "normal", "reason": "IMAGES_INCOMPATIBLE"}
```

### 5.3 Logged events

| Event | When | Fields |
|---|---|---|
| `conversation_created` | New conversation | conversation_id, pseudo_model, physical_model |
| `affinity_maintained` | Turn uses same physical model | conversation_id, physical_model, turn |
| `affinity_changed` | Model changed (switch, fallback) | conversation_id, from, to, reason |
| `fallback_applied` | Model returned 503/429 | conversation_id, failed_model, fallback_model, reason |
| `switch_validated` | Pseudo-model switch checked | conversation_id, from, to, result (safe/warning/blocked) |
| `switch_blocked` | Pseudo-model switch blocked | conversation_id, from, to, reason |
| `pre_compaction_applied` | Pre-compaction triggered | tokens_before, tokens_after, compactor |
| `continuous_compaction_applied` | Continuous compaction triggered | tokens_before, tokens_after, turns_compacted |
| `explicit_compaction` | User invokes /compact | tokens_before, tokens_after, compactor |
| `images_described` | Auto-describe runs | images_count, described_by |
| `tools_normalized` | normalize-tools called | parallel_calls_serialized, turns_affected |
| `tools_incomplete` | Streaming tool call interrupted | conversation_id, turn |
| `router_suggestion` | Router LLM evaluated | complexity, suggested, reason |
| `rate_limit_hit` | Rate limit exceeded | pseudo_model, limit, client_ip |
| `auth_failure` | Invalid API key | client_ip |
| `provider_error` | Provider returned non-retryable error | provider, model, error_type |

---

## 6. Metrics Endpoint (`GET /metrics`)

### 6.1 Endpoint

```
GET /metrics  → requires auth
```

### 6.2 Response format

```json
{
  "uptime_seconds": 86400,
  "total_requests": 15234,
  "requests_by_pseudo_model": {
    "normal": 8000,
    "deep-flash": 4000,
    "flash-lowcost": 2000,
    "tareas-avanzadas": 1000,
    "pensamiento-profundo-caro": 200,
    "avanzada-vision": 30,
    "flash-vision": 4
  },
  "total_tokens": {
    "input": 45000000,
    "output": 8200000,
    "cached": 12000000,
    "saved_by_compaction": 8000000
  },
  "cache": {
    "hit_rate_pct": 72.5,
    "total_cache_hits": 11000,
    "estimated_savings_usd": 28.45
  },
  "compactions": {
    "pre_compactions": 150,
    "continuous_compactions": 45,
    "explicit_compactions": 12,
    "total_tokens_saved": 8200000
  },
  "fallbacks": {
    "total": 23,
    "by_reason": {
      "upstream_503": 18,
      "upstream_429": 5
    }
  },
  "rate_limits": {
    "total_hits": 5,
    "by_pseudo_model": {
      "pensamiento-profundo-caro": 3,
      "avanzada-vision": 2
    }
  },
  "conversations": {
    "active": 45,
    "total": 890,
    "with_snapshot": 34
  },
  "errors": {
    "4xx": 120,
    "5xx": 8,
    "INPUT_EXCEEDS_THRESHOLD": 45,
    "CONTEXT_UNUSABLE": 3,
    "ALL_MODELS_FAILED": 8
  }
}
```

### 6.3 Implementation

The metrics endpoint aggregates data from:
1. In-memory counters (reset on restart)
2. DB queries (conversation counts, compactions)
3. Valkey (rate limit hits — these are ephemeral)

```python
# src/api/metrics.py

from fastapi import APIRouter, Request
from sqlalchemy import text, func

router = APIRouter()

# In-memory counters (reset on restart)
class MetricsStore:
    def __init__(self):
        self.total_requests = 0
        self.requests_by_pseudo: dict[str, int] = {}
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cached_tokens = 0
        self.total_saved_by_compaction = 0
        self.cache_hits = 0
        self.pre_compactions = 0
        self.continuous_compactions = 0
        self.fallbacks: dict[str, int] = {}
        self.errors_4xx = 0
        self.errors_5xx = 0
        self.errors_by_type: dict[str, int] = {}

metrics = MetricsStore()


@router.get("/metrics")
async def get_metrics(request: Request):
    db = request.app.state.db_session_factory()

    # DB queries for persistent data
    total_convs = await db.scalar(select(func.count(Conversation.id)))
    active_convs = await db.scalar(
        select(func.count(Conversation.id)).where(Conversation.updated_at > func.now() - text("INTERVAL '24 hours'"))
    )
    total_snapshots = await db.scalar(select(func.count(ConversationSnapshot.id)))
    total_explicit_compactions = await db.scalar(
        select(func.count(ConversationSnapshot.id)).where(ConversationSnapshot.snapshot_type == "explicit")
    )

    return {
        "uptime_seconds": int(time.time() - START_TIME),
        "total_requests": metrics.total_requests,
        "requests_by_pseudo_model": metrics.requests_by_pseudo,
        "total_tokens": {
            "input": metrics.total_input_tokens,
            "output": metrics.total_output_tokens,
            "cached": metrics.total_cached_tokens,
            "saved_by_compaction": metrics.total_saved_by_compaction,
        },
        "cache": {
            "hit_rate_pct": round((metrics.cache_hits / max(metrics.total_requests, 1)) * 100, 1),
            "total_cache_hits": metrics.cache_hits,
        },
        "compactions": {
            "pre_compactions": metrics.pre_compactions,
            "continuous_compactions": metrics.continuous_compactions,
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
```

---

## 7. Caddy HTTPS Configuration

### 7.1 Caddyfile.example

```
# /etc/caddy/Caddyfile
# Save this as Caddyfile and run: caddy run

proxy.tudominio.com {
    reverse_proxy localhost:9110

    # Caddy automatically obtains and renews Let's Encrypt certificates
    # No manual certificate management needed

    # Optional: rate limit at the reverse proxy level
    # rate_limit {
    #     zone dynamic {
    #         key {remote_host}
    #         events 100
    #         window 1m
    #     }
    # }

    # Security headers
    header {
        X-Content-Type-Options "nosniff"
        X-Frame-Options "DENY"
        Referrer-Policy "strict-origin-when-cross-origin"
    }

    # Logs
    log {
        output file /var/log/caddy/proxy-access.log
        format json
    }
}
```

### 7.2 Caddy setup

```bash
# Install Caddy
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update
sudo apt install caddy

# Start Caddy with the Caddyfile
sudo caddy run --config /etc/caddy/Caddyfile
```

---

## 8. Systemd Service (Production)

### 8.1 Service file

```
# /etc/systemd/system/proxy-cesar.service

[Unit]
Description=Proxy Determinista Multi-Modelo
After=network.target postgresql.service valkey.service

[Service]
Type=simple
User=proxy
WorkingDirectory=/opt/proxy-cesar
EnvironmentFile=/opt/proxy-cesar/.env
ExecStart=/opt/proxy-cesar/.venv/bin/uvicorn src.main:app --host 127.0.0.1 --port 9110
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### 8.2 Celery service

```
# /etc/systemd/system/proxy-cesar-celery.service

[Unit]
Description=Proxy Cesar Celery Worker
After=network.target valkey.service

[Service]
Type=simple
User=proxy
WorkingDirectory=/opt/proxy-cesar
EnvironmentFile=/opt/proxy-cesar/.env
ExecStart=/opt/proxy-cesar/.venv/bin/celery -A src.tasks.celery_app worker --loglevel=info --concurrency=4
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

---

## 9. README.md — Deployment Guide

### 9.1 Sections to include

The README must cover:

1. **What is this?** — 2-3 sentence description
2. **Quick start (5 minutes)**
   - Clone repo
   - Install deps (`poetry install` or `uv sync`)
   - Copy `.env.example` → `.env`, fill in API keys
   - Generate `PROXY_API_KEY`: `openssl rand -hex 32`
   - Start: `uvicorn src.main:app --port 9110`
   - Test: `curl` example
3. **Configuration**
   - `pseudo_models.yaml` — what it is, how to add/remove models
   - `.env` — all vars explained
4. **API Reference**
   - All endpoints with curl examples
   - `proxy_metadata` fields explained
5. **Production deployment**
   - PostgreSQL setup
   - Valkey setup
   - Systemd services
   - Caddy HTTPS
6. **OpenCode integration**
   - Copy-paste config
   - Environment variables

### 9.2 What the README must NOT contain

- No marketing language
- No AI-generated filler text
- No emojis
- No promises about model quality or performance
- No comparisons with other proxies

---

## 10. Deployment Checklist

Before declaring Sprint 8 done, verify ALL of:

- [ ] `PROXY_API_KEY` generated and set in `.env`
- [ ] Auth middleware returns 401 for missing/invalid keys
- [ ] `/health` remains public (no auth required)
- [ ] CORS headers set correctly for configured origins
- [ ] Rate limiting enforces per-pseudo-model limits
- [ ] Rate limit headers (`X-RateLimit-*`) in responses
- [ ] Structured JSON logs to stdout
- [ ] All proxy decisions logged with relevant fields
- [ ] `GET /metrics` returns accurate aggregated data
- [ ] `Caddyfile.example` ready
- [ ] Systemd service files ready
- [ ] `README.md` complete with all sections
- [ ] New user can set up in <5 minutes (timed test)
- [ ] OpenCode integration works (tested with real OpenCode or equivalent client)

---

## 11. Tests (Sprint 8)

### 11.1 test_auth.py (minimum 5 tests)

1. Valid Bearer token → request proceeds
2. Missing Authorization header → 401
3. Invalid API key → 401
4. `/health` accessible without auth
5. Dev mode (no PROXY_API_KEY) → all endpoints accessible

### 11.2 test_rate_limiter.py (minimum 5 tests)

1. Request within limit → proceeds
2. Request above limit → 429
3. `Retry-After` header present on 429
4. `X-RateLimit-*` headers present on responses
5. Different pseudo-models have independent limits

### 11.3 test_metrics.py (minimum 5 tests)

1. `GET /metrics` returns valid JSON
2. Metrics include `total_requests` and `requests_by_pseudo_model`
3. Metrics include `total_tokens` with cached tokens
4. Metrics endpoint requires auth
5. Metrics include `errors` breakdown

### 11.4 test_e2e.py — stress test (minimum 3 tests)

1. 50 concurrent conversations → each maintains correct affinity
2. Rate limiting: burst of 100 requests → some get 429
3. 24h+ uptime simulation → Valkey TTL works correctly

---

## 12. Acceptance Criteria (Sprint 8)

- [ ] Auth middleware active and correct
- [ ] CORS configured for browser clients
- [ ] Rate limiting per pseudo-model in Valkey
- [ ] Structured JSON logs to stdout/stderr
- [ ] `GET /metrics` returns accurate stats
- [ ] `Caddyfile.example` and systemd service files ready
- [ ] `README.md` complete with full deployment guide
- [ ] New user setup timed: <5 minutes
- [ ] OpenCode integration tested
- [ ] All 18+ tests pass
- [ ] No regression on Sprint 1-7 tests
- [ ] All dependencies verified with free licenses (MIT/BSD/Apache 2.0)

---

## 13. Explicitly OUT OF SCOPE for Sprint 8

Sprint 8 is the **final planned sprint** for v1. Anything not listed in this or previous sprints is out of scope for v1:

| Feature | Status |
|---|---|
| Audio support (transcription, degradation) | v2 |
| PDF text extraction | v2 |
| Video support | v2 |
| Multi-user with individual API keys | v2 |
| OAuth / JWT auth | v2 |
| Web UI / dashboard | v2 |
| Email alerts on errors | v2 |
| Horizontal scaling (multiple proxy instances) | v2 (design is already stateless) |
| Auto-scaling based on load | v2 |
| Custom model pricing in config | v2 |
| Model performance benchmarks | Never (out of scope) |
| Model recommendation engine | Never (proxy does not decide) |

---

## 14. Final v1.0 Definition of Done

The proxy is **v1.0 ready** when:

1. All 8 sprints are complete and all acceptance criteria are met
2. All tests pass (estimated 200+ total across all sprints)
3. A new user can deploy and use the proxy following only the README
4. The proxy has been running for 24+ hours without crashes or memory leaks
5. All provider API keys are on the server only — never in client config
6. All dependencies are verified MIT/BSD/Apache 2.0 licensed
7. `proxy_metadata` is present in every response
8. Every proxy decision is logged in structured JSON
