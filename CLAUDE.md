# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

**proxy-cesar** is a deterministic multi-model LLM proxy in production at `chat.guzman-lopez.com`. It abstracts 10 pseudo-models over 30+ physical models from multiple providers (OpenCode Go, DeepSeek, Groq, OpenRouter) with automatic fallback, conversation state management, context compaction, and security features (KeyVault for secret detection).

**Stack:** Python 3.13+, FastAPI, SQLite (conversations), Redis :6380 (affinity/cache/rate-limiting), async-first architecture, result monad for error handling.

---

## Common Commands

```bash
# Setup
cd proxy
python3.13 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run (localhost :9110)
python -m src.main

# Tests
pytest                              # all tests
pytest tests/test_e2e.py -x         # stop on first failure
pytest tests/test_chat.py -k streaming  # single file + filter
pytest tests/ --cov=src             # coverage report

# Lint & format
ruff check src/
ruff format src/

# Health check (running proxy required)
curl http://localhost:9110/health
curl -H "Content-Type: application/json" \
  -d '{"model":"normal","messages":[{"role":"user","content":"hola"}]}' \
  http://localhost:9110/v1/chat/completions

# Environment
cp .env.example .env                # edit API keys before running
```

---

## Architecture

### Directory Structure

```
proxy/
├── src/
│   ├── api/                      # FastAPI routers (endpoints)
│   │   ├── chat.py                   streaming + non-streaming completions
│   │   ├── conversations.py          conversation state
│   │   ├── conversation_operations.py  compaction, audit logs
│   │   ├── health.py                 health check
│   │   ├── metrics.py                aggregated metrics
│   │   └── models.py                 list pseudo-models
│   ├── service/                  # Business logic (pure domain layer)
│   │   ├── chat_service.py           main orchestrator (fallback, capability detection)
│   │   ├── model_resolver.py         pseudo-model → physical model + aliases
│   │   ├── capability_detector.py    turn-level/session-level capabilities
│   │   ├── threshold_guard.py        token limit guards (Result monad)
│   │   ├── tool_filter.py            filter by parallel tool support
│   │   ├── tools_normalizer.py       serialize parallel tool calls
│   │   ├── context_alert.py          60/80/100% context usage alerts
│   │   ├── compactor/                context compaction (3 strategies)
│   │   │   ├── pre_compactor.py      summarize turn if input exceeds threshold
│   │   │   ├── continuous.py         compact old turns when context grows
│   │   │   └── explicit.py           POST /compact endpoint
│   │   ├── multimedia/               content transformation
│   │   │   └── image_describer.py    describe images via vision model
│   │   └── router_llm/               task routing (unused)
│   ├── domain/                   # Pure domain types (no FastAPI/SQLModel imports)
│   │   ├── types.py                  Result monad: Ok[T] / Err[E]
│   │   ├── errors.py                 domain errors (11 types)
│   │   ├── capabilities.py           session capability flags
│   │   ├── affinity.py               AffinityPort protocol
│   │   └── tools.py                  tool definitions
│   ├── adapters/                 # Infrastructure (DB, cache, LLM clients)
│   │   ├── db/
│   │   │   ├── models.py             SQLModel ORM (Conversation, ConversationTurn, ConversationSnapshot)
│   │   │   └── engine.py             async SQLAlchemy setup
│   │   ├── litellm/
│   │   │   └── client.py             LiteLLM wrapper (maps physical models to providers)
│   │   └── cache/
│   │       ├── valkey_affinity.py     physical model affinity (24h TTL)
│   │       ├── message_ordering.py    canonical message ordering
│   │       └── provider_cache.py      provider-specific cache optimization
│   ├── middleware/
│   │   ├── rate_limiter.py           per-pseudo-model fixed-window rate limiting
│   │   └── keyvault.py               secret detection + replacement (22 patterns)
│   ├── config/
│   │   ├── pseudo_models.py          YAML loader + pydantic validation
│   │   └── settings.py               pydantic-settings from .env
│   ├── auth.py                   # Bearer token authentication
│   ├── logging_config.py         # Structured JSON logging
│   ├── tasks/
│   │   └── arq_app.py                async task queue (compaction)
│   └── main.py                   # App entrypoint + lifespan setup
├── tests/
│   ├── conftest.py                   fixtures (fake Redis, in-memory DB, mocked APIs)
│   ├── test_e2e.py                   end-to-end tests (streaming, fallback, etc.)
│   ├── test_chat.py                  chat service tests
│   └── test_*.py                     unit tests per module
├── pseudo_models.yaml            # Configuration: 10 pseudo-models + fallback chains
├── pyproject.toml                # Dependencies: FastAPI, SQLModel, LiteLLM, Valkey, etc.
└── .env.example                  # Template for API keys (OPENCODE_API_KEY, DEEPSEEK_API_KEY, etc.)
```

### Key Patterns

**1. Hexagonal Architecture (Mandatory)**
- `domain/` is pure — no FastAPI, no SQLModel, no external imports
- `service/` orchestrates logic using domain types + adapters (Result monad)
- `adapters/` wraps infrastructure (DB, cache, LLM providers)
- `api/` (routers) maps HTTP to service calls, error handling happens here

**2. Result Monad for Error Handling**
```python
# Service returns Result, not exceptions
def resolve_model(pseudo: str) -> Result[PhysicalModel, DomainError]:
    if pseudo not in PSEUDO_MODELS:
        return Err(UnknownPseudoModel(pseudo))
    return Ok(PSEUDO_MODELS[pseudo])

# Router consumes with match/case
match service.resolve_model(request.model):
    case Ok(value=model):
        return handle(model)
    case Err(error=err):
        raise HTTPException(status_code=400, detail=str(err))
```

**3. Async-First**
- All endpoints are `async def`
- AsyncSession for DB operations
- httpx.AsyncClient for external calls

**4. Pseudo-Models + Fallback Logic**
- 10 pseudo-models defined in `pseudo_models.yaml` (normal, pensamiento-profundo-caro, etc.)
- Each pseudo-model has 1+ physical models (primary + fallbacks)
- Fallback strategy is sequential (try primary, then fallbacks in order) except `compactador` which uses `by_context_window`
- Model aliases (gpt-4o → normal, o3 → pensamiento-profundo-caro, etc.)

**5. Affinity: Conversation Pinning**
- First request in a conversation pins to a physical model (Redis `conv:{id}:physical_model`, 24h TTL)
- Subsequent requests use the pinned model unless capabilities force a switch

**6. KeyVault: Secret Detection**
- Detects 22 patterns (API keys, PEM, SSH, JWT, crypto wallets)
- Stores real values in Redis, replaces with `[KEYVAULT:hash]` before sending to LLM
- Re-injects real values in response (LLM never sees them)

**7. Blob Vault: Content Transformation**
- Images → described by vision model → `[BLOB:hash]` for non-vision models
- PDFs → text extracted → `[BLOB:hash]`
- Stored in Redis (24h TTL)

---

## Code Guidelines

Reference: `python.md` (comprehensive guide, includes all rules).

**Key rules (summary):**

1. **Strict typing** — no `Any`, no `# type: ignore`, parameterize `list[T]`, `dict[K, V]`, etc.
2. **Result monad** — errors as data (`Ok[T] | Err[E]`), not exceptions in business logic
3. **Hexagonal** — domain layer pure, services orchestrate via adapters
4. **Async-first** — `async def` everywhere
5. **match/case** — extract Result values
6. **File size** — ideal 300–400 lines, max 600 lines
7. **No error silencing** — propagate explicitly or convert to Result
8. **DI with FastAPI** — use `Depends()` for injection, not `new`

**Python version:** 3.13+ (use `|` for unions, `match/case`, `type` aliases, etc.)

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `PROXY_PORT` | 9110 | HTTP port |
| `PROXY_API_KEY` | — | Bearer token (empty = dev mode) |
| `DATABASE_URL` | sqlite+aiosqlite:///./proxy.db | SQLite |
| `VALKEY_URL` | valkey://localhost:6380 | Redis native |
| `OPENCODE_API_KEY` | — | OpenCode Go (primary) |
| `DEEPSEEK_API_KEY` | — | DeepSeek (fallbacks) |
| `GROQ_API_KEY` | — | Groq (vision, whisper) |
| `OPENROUTER_API_KEY` | — | OpenRouter |
| `LOG_LEVEL` | INFO | log level |

---

## Testing Strategy

- **Unit tests** (`tests/test_*.py`): service logic, Result monad handling, edge cases
- **E2E tests** (`tests/test_e2e.py`): streaming, fallback chains, capability detection
- **Fixtures** (`tests/conftest.py`): fake Redis (fakeredis), in-memory DB (aiosqlite), mocked LLM clients
- **Coverage target:** > 80% (enforce via CI)

Run with `pytest tests/ --cov=src` or single file with `pytest tests/test_chat.py -x`.

---

## Deployment

**Target:** Ubuntu 22.04 server (`plata`)

**CI/CD:** GitHub Actions on push to `main`:
1. Clone with `--depth 1`
2. Backup SQLite DB
3. Install + migrate
4. `systemctl restart proxy-cesar`
5. Health check (fail → rollback)

**Service files:** `proxy-cesar.service`, `proxy-cesar-arq.service` (background compaction tasks)

**Database:** SQLite (persisted between deploys), 3 tables:
- `conversations` — state per conversation
- `conversation_turns` — individual turns + token counts
- `conversation_snapshots` — compaction history

**Monitoring:** `/health` endpoint, `/metrics` for aggregates, structured JSON logging.

---

## Key Decision Points

**When adding a feature:**
- **Error handling?** → Return `Result[T, E]` from service, convert to HTTPException in router
- **New pseudo-model?** → Add to `pseudo_models.yaml`, test fallback chain with mocked providers
- **Content transformation?** → Add to `multimedia/`, return Result
- **Rate limiting?** → Use existing middleware, config in settings
- **Conversation state?** → Update SQLModel tables, test with real DB in tests

**Gotchas:**
- `domain/` layer must not import FastAPI, SQLModel, or external libs
- KeyVault runs *before* the LLM call, re-inject happens *after* response
- Affinity is per-conversation, not per-user
- Fallback is sequential, test all chains in pytest
- Compaction modifies conversation history, version carefully

---

## References

- **Architecture & Python rules:** `python.md`
- **API & error codes:** `proxy/README.md`
- **Pseudo-models & system errors:** main `README.md`
- **Diagrams:** `diagramas.md`
- **Bug verification stories & commands:** `BUG_VERIFICATION_FLOW.md`
- **Production server rules:** `README-plata.md`
