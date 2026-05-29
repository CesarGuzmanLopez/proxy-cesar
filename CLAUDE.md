# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

**proxy-cesar v1.0** is a deterministic multi-model LLM proxy in production at `chat.guzman-lopez.com`. It abstracts 10 pseudo-models over 30+ physical models from multiple providers (OpenCode Go, DeepSeek, Groq, OpenRouter) with automatic fallback, conversation state management, context compaction, Bearer auth, and security features (KeyVault for secret detection).

**Tests:** 406 total, 73% coverage.
**Auth:** `PROXY_API_KEY` required in production (401 without Bearer token). Public endpoints: `/health`, `/docs`, `/openapi.json`, `/redoc`.

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
pytest                              # all tests (406)
pytest tests/test_tool_detector.py  # content delegation tests (82)
pytest tests/test_e2e.py -x         # stop on first failure
pytest tests/test_chat.py -k streaming  # single file + filter
pytest tests/ --cov=src             # coverage report (72%)

# Lint & format
ruff check src/
ruff format src/

# Health check (running proxy required)
curl http://localhost:9110/health

# Chat (auth required if PROXY_API_KEY is set)
curl -H "Content-Type: application/json" \
  -H "Authorization: Bearer $PROXY_API_KEY" \
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
│   │   ├── chat_fallback.py          fallback loop with SmartFallback scoring
│   │   ├── chat_messages.py          conversation message building + history management
│   │   ├── chat_streaming.py         streaming response generator
│   │   ├── chat_stream_persistence.py token extraction + metadata chunk building
│   │   ├── chat_models.py            context dataclasses (SaveContext, StreamingRequestContext, MetadataContext)
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
- Context windows actualizados a valores reales de cada modelo:
  kimi-k2.5/k2.6 = 256K, MiMo v2.5/V2 = 1M, GLM-5.1 = 198K, etc.

**5. Affinity: Conversation Pinning (Respects User Choice)**
- User chooses pseudo-model → proxy respects it (never changes automatically)
- First request pins physical model (Redis `conv:{id}:physical_model`, 24h TTL with sliding expiry)
- Model only changes on FAILURE (via fallback chain), never on size/capacity
- TTL extends if conversation stays active (sliding window)
- Failure metrics tracked for smart fallback decisions
- Subsequent requests use the pinned model unless capabilities force a switch

**6. KeyVault: Secret Detection**
- Detects 22 patterns (API keys, PEM, SSH, JWT, crypto wallets)
- Stores real values in Redis, replaces with `[KEYVAULT:hash]` before sending to LLM
- Re-injects real values in response (LLM never sees them)

**7. Auth: Bearer Token (Required in Production)**
- `PROXY_API_KEY` env var → validada en cada request (excepto `/health`, `/docs`, etc.)
- Si la variable está vacía → dev mode (auth deshabilitado)
- Error `401 MISSING_AUTH` / `401 INVALID_API_KEY` si la key no coincide
- Middleware registrado PRIMERO en main.py para ejecutarse antes que cualquier otro

**8. Blob Vault: Content Transformation**
- Images → described by vision model → `[BLOB:hash]` for non-vision models
- PDFs → text extracted → `[BLOB:hash]`
- Stored in Redis (24h TTL)

---

## Proxy Core Philosophy & Implementation

**Mission:** Transparent multi-provider LLM coordination. User should NOT notice they're using a proxy.

### How It Works: Complete Request Flow

#### **INCOMING REQUEST**
```
User sends: {
  "model": "normal",           ← User chooses pseudo-model
  "messages": [...],
  "stream": true,
  "tools": [...]
}
```

#### **STEP 1-3: VALIDATION & CONTENT PREP**
- **Model Resolution** (`chat_service.py:_resolve_and_validate`):
  - Validates "normal" exists in pseudo_models.yaml
  - Resolves to pseudo-model schema with 30+ physical models
  - Detects capabilities in messages (images, audio, PDF, video, tools)

- **Content Delegation** (`chat_service.py:_apply_content_delegation`):
  - **Images**: If pseudo-model lacks vision → describe via vision model, cache in Valkey, inject descriptions as `[BLOB:hash]`
  - **PDFs**: Extract text via specialized model → store as `[BLOB:hash]`
  - **Audio**: Transcribe via audio model → store as `[BLOB:hash]`
  - **Video**: Description + keyframes → store as `[BLOB:hash]`
  - **Result**: Messages sent to LLM only contain text + metadata, actual files cached separately

- **KeyVault Secret Detection** (`middleware/keyvault.py`):
  - 27 regex patterns detect API keys, PEM files, JWTs, crypto wallets, SSH keys, etc.
  - Found secrets → stored in Valkey with TTL, replaced with `[KEYVAULT:hash]`
  - Injects system prompt: "When you see [KEYVAULT:*], understand it's a secret placeholder"
  - **LLM never sees real secrets**

#### **STEP 4-11: CONVERSATION & MODEL SELECTION**
- **Load Conversation** (`chat_service.py:_resolve_session_conv_and_models`):
  - Get conversation history from SQLite
  - Load accumulated capabilities (images, tools, audio, etc. in session history)
  - Load affinity: is there a pinned physical model?

- **Affinity Check** (`valkey_affinity.py`):
  - If pinned model exists AND compatible → use it
  - Extends TTL if conversation active (sliding window)
  - **Does NOT upgrade** based on size/capacity (respects user choice)
  - Only invalidates if incompatible (parallel tools required but not supported)

- **Fallback Chain Selection** (`model_resolver.py`):
  - Gets fallback chain for the pseudo-model user chose
  - Chain order: primary → fallbacks (or by_context_window for compactador)
  - SmartFallback scoring: choose model with best success_rate, skip if >3 errors/1h

#### **STEP 12: INPUT VALIDATION**
- **Token Estimation** (`capability_detector.py:estimate_tokens`):
  - tiktoken o200k_base encoding (GPT-4o)
  - Counts text + tool arguments + tool results
  - Fallback to 4-char=1-token heuristic if tiktoken fails

- **Token Threshold** (`threshold_guard.py`):
  - Check: input_tokens < pseudo_model.input_token_threshold
  - If exceeds → FAIL (InputExceedsThreshold) — proxy doesn't hide overages
  - User should handle via compaction or shorter messages

- **Context Usability** (`context_alert.py`):
  - Total history tokens vs context window
  - Alert levels: 60% (warning), 80% (danger), 100%+ (unusable)
  - If unsuable → optionally trigger auto-compaction

#### **STEP 13: CALL LLM WITH FALLBACK**
```
call_with_fallback(
  messages: [system, user history, current request],
  tools: [...],
  ...
)
```

**Fallback Loop** (`chat_fallback.py:call_with_fallback`):
1. Try first physical model (pinned if available)
2. If retryable error (429, 503, 401, 404, 400):
   - Log error
   - **SmartFallback**: Record failure metrics
   - Continue to next model
3. If non-retryable error: fail immediately (fail fast)
4. If success: return response
5. If all models fail: AllModelsFailed error

**SmartFallback Scoring** (`smart_fallback.py`):
- Per conversation, per model: track success_rate, latency, errors_1h
- Score = success_rate - (errors_1h * 0.1) - (latency_ms / 1000)
- Skip models with >3 errors in last 1 hour
- Metrics stored in Valkey (TTL 1h)

#### **STEP 14: RESPONSE PROCESSING**
- **Extract Response Metadata**:
  - finish_reason: "stop", "length", "tool_calls", etc.
  - tokens: prompt_tokens, completion_tokens
  - provider headers (Anthropic cache info, etc.)
  - actual model name (may differ from pinned if fallback occurred)

- **Reconstruct Tool Calls** (for streaming):
  - Aggregate tool_call deltas from stream chunks
  - Validate tool IDs match request definitions
  - Store as structured objects in turn

- **Token Limit Continuation** (if finish_reason="length"):
  - Append partial response as assistant message
  - Continue with next model in chain
  - Build composite response from accumulated parts

#### **STEP 15: RE-INJECT SECRETS**
- Find all `[KEYVAULT:hash]` in response
- Look up hash in Valkey
- Replace with original secret
- **LLM thinks it generated real secrets, but we fixed it before sending to user**

#### **STEP 16-17: SAVE & RETURN**
- **Save Turn to DB**:
  - conversation_id, turn_number, pseudo_model, physical_model
  - input/output tokens, messages, response
  - turn_type: "normal" (not "degradation_event")
  - tool_calls, tool_definitions, thinking_blocks

- **Save Conversation Metadata**:
  - Accumulate capability flags (has_images, has_tools, etc.)
  - Max tools level used
  - Images described count
  - Total tokens

- **Return to User**:
  - If streaming: SSE chunks + final metadata chunk + `[DONE]`
  - If non-streaming: JSON response
  - Both include `proxy_metadata`:
    ```json
    {
      "physical_model": "openai/gpt-4o-mini",
      "pseudo_model": "normal",
      "provider": "openai",
      "affinity_maintained": true,
      "fallback_applied": false,
      "context_usage": "45.2%",
      "images_described": 2,
      "cache_info": {...},
      "elapsed_ms": 1250
    }
    ```

### When Things Go Wrong

#### **No Input Tokens Left** (context full)
- ✗ Proxy does NOT silently delete old turns
- ✗ Proxy does NOT change model
- ✓ Returns ContextUnusable error → User can request compaction

#### **Model Fails (429, 503, etc.)**
- ✓ Try next in fallback chain (SmartFallback scores models)
- ✓ Log which model failed + reason
- ✓ Record metrics for future skipping
- User sees correct response OR AllModelsFailed error

#### **User Sends Incompatible Content**
- Image to non-vision model: Auto-describe via vision model ✓
- PDF to model that can't read: Extract text ✓
- Tool use to model without tool support: Return error ✓

#### **Streaming Interrupted**
- ✓ Sends StreamPersistenceFailed error + `[DONE]`
- ✓ Partial response saved to DB
- Turn persisted (incremental)

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
9. **Context objects** — use `@dataclass` to collapse 15+ parameters into 1 (`StreamingRequestContext`, `SaveContext`, `MetadataContext`). See `chat_models.py`

**Python version:** 3.13+ (use `|` for unions, `match/case`, `type` aliases, etc.)

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `PROXY_PORT` | 9110 | HTTP port |
| `PROXY_API_KEY` | — | Bearer token (empty = dev mode; **required in production**) |
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
