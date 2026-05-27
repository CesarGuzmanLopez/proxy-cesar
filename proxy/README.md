# Proxy César — Technical Documentation

**Deterministic multi-model proxy for LLMs.** A transparent HTTP proxy between your LLM client and multiple providers. Translates pseudo-model names to physical models, manages conversation state, enforces content compatibility, handles tool normalization, and applies context compaction.

---

## Architecture

```
src/
├── api/                  # FastAPI routers
│   ├── chat.py               POST /v1/chat/completions (SSE streaming)
│   ├── conversations.py      GET/POST /conversations/{id}/*
│   ├── health.py             GET /health
│   ├── metrics.py            GET /metrics
│   └── models.py             GET /v1/models
├── service/              # Business logic
│   ├── chat_service.py            Main chat orchestrator
│   ├── chat_models.py             ChatResult, proxy_metadata
│   ├── model_resolver.py          Pseudo-model → physical + aliases
│   ├── capability_detector.py     Turn/session capability detection
│   ├── threshold_guard.py         Input threshold guard (Result monad)
│   ├── tool_filter.py             Filter by parallel tool support
│   ├── tools_canonical.py         Tool level determination
│   ├── tools_normalizer.py        Parallel tool call serialization
│   ├── context_alert.py           Context usage alerts (60/80/100%)
│   ├── compactor/
│   │   ├── pre_compactor.py       Pre-request summarization
│   │   ├── continuous.py          Continuous compaction
│   │   ├── explicit.py            POST /compact
│   │   └── prompts.py             Compaction prompts
│   ├── multimedia/
│   │   └── image_describer.py     Auto-describe images
│   └── router_llm/
│       └── suggester.py           Task complexity evaluation
├── domain/               # Pure domain types
│   ├── types.py                Ok[T] / Err[E] Result monad
│   ├── errors.py               Domain errors (11 types)
│   ├── capabilities.py         Session capabilities
│   └── affinity.py             AffinityPort protocol
├── adapters/             # Infrastructure
│   ├── db/models.py            SQLModel ORM (3 tables)
│   ├── db/engine.py            Async SQLAlchemy engine
│   ├── litellm/client.py       LiteLLM adapter
│   └── cache/
│       ├── valkey_affinity.py      Physical model affinity (Redis)
│       ├── message_ordering.py     Canonical ordering
│       └── provider_cache.py       Provider-specific cache
├── config/
│   ├── pseudo_models.py       YAML loader + Pydantic validation
│   └── settings.py            pydantic-settings
├── middleware/
│   └── rate_limiter.py        Per-pseudo-model rate limiter
├── auth.py                Bearer token auth
├── logging_config.py      Structured JSON logging
├── tasks/
│   └── arq_app.py         Async compaction via arq
└── schemas/
    └── tools.py            Pydantic request/response schemas
```

---

## Key Features

### Pseudo-Models + Fallback

10 pseudo-models defined in `pseudo_models.yaml`, each mapping to 1+ physical models. Fallback is `sequential` (try primary, then fallbacks in order) except `compactador` which uses `by_context_window`.

**Model aliases** bridge common names: `gpt-4o` → `normal`, `o3` → `pensamiento-profundo-caro`, etc.

### KeyVault (Security)

Intercepts `POST /v1/chat/completions`:
1. Detects 22 patterns (API keys, PEM, SSH, JWT, crypto wallets)
2. Stores in Redis `keyvault:{conv}:{hash}` (TTL 1h)
3. Replaces with `[KEYVAULT:hash]` before sending to LLM
4. Re-injects real values in response (streaming + non-streaming)

The LLM **never sees** real keys. The client **always sees** real keys.

### Blob Vault (Content Transformation)

When a model doesn't support the received content type:

| Type | Transformation | Helper |
|---|---|---|
| Image | Describe with vision model → `[BLOB:hash]` | Vision-capable model |
| Audio | Transcribe with whisper → `[BLOB:hash]` | Whisper models |
| PDF | Extract text with PyMuPDF → `[BLOB:hash]` | N/A |

Blobs stored in Redis (24h TTL).

### Content Validation

Incoming content validated before reaching any provider:

- **Images** → blocked unless pseudo-model has vision capability
- **Audio** → blocked (not supported in v1 for chat)
- **Video** → blocked (not supported in v1)
- **Parallel tools** → blocked unless a physical model supports them

### Context Compaction

Three strategies:
- **Pre-compaction**: Summarizes latest user message if input exceeds threshold
- **Continuous**: Compacts old turns into `ConversationSnapshot` when context exceeds `trigger_pct`
- **External**: Detects client-side compaction (message count drops >60%)

### Physical Model Affinity

Conversations pinned to their first physical model via Redis (`conv:{id}:physical_model`, 24h TTL). Consistent behavior across turns unless capability filter forces a change.

### Provider Cache Optimization

- Layer 1 — Redis affinity: Same physical model per conversation
- Layer 2 — Canonical ordering: `system → tools → history → query`
- Layer 3 — Provider-specific: Anthropic `cache_control`, DeepSeek auto-caching

### Rate Limiting

Per-pseudo-model fixed-window rate limiter in Redis. Headers: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`.

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `PROXY_PORT` | `9110` | HTTP port |
| `PROXY_API_KEY` | — | Bearer token (empty = dev mode) |
| `DATABASE_URL` | `sqlite+aiosqlite:///./proxy.db` | SQLite |
| `VALKEY_URL` | `valkey://localhost:6380` | Redis native port 6380 |
| `KEYCLAW_ENABLED` | `false` | KeyClaw disabled |
| `LOG_LEVEL` | `INFO` | Log level |
| `OPENCODE_API_KEY` | — | OpenCode Go (primary) |
| `DEEPSEEK_API_KEY` | — | DeepSeek (fallbacks) |
| `GROQ_API_KEY` | — | Groq (vision, whisper) |
| `OPENROUTER_API_KEY` | — | OpenRouter (normal-gratis) |
| `PRUNA_API_KEY` | — | Pruna |

---

## API Reference

### `GET /health`
```json
{"status":"ok","database":"connected","valkey":"connected","pseudo_models_loaded":10}
```

### `GET /v1/models`
Lists all 10 pseudo-models with capabilities.

### `POST /v1/chat/completions`
OpenAI-compatible. Supports `stream: true` (SSE) and `stream: false`.

**Request:**
```json
{"model":"normal","messages":[{"role":"user","content":"Hello"}],"stream":false}
```

**Response includes `proxy_metadata`:**
```json
{
  "proxy_metadata": {
    "physical_model": "openai/kimi-k2.5",
    "pseudo_model": "normal",
    "fallback_applied": false,
    "capabilities_detected": {"has_tools": true},
    "context_alert": "normal",
    "cache": {"strategy": "...", "savings": 0.5}
  }
}
```

---

## Error Dictionary

| Error Code | Status | Trigger |
|---|---|---|
| `UNKNOWN_PSEUDO_MODEL` | 400 | Model name not found |
| `PSEUDO_MODEL_INCOMPATIBLE` | 409 | Switch blocked by capabilities |
| `INPUT_EXCEEDS_THRESHOLD` | 400 | Input > token threshold |
| `IMAGES_NOT_SUPPORTED` | 400 | Image in non-vision model |
| `AUDIO_NOT_SUPPORTED` | 400 | Audio in chat (v1) |
| `VIDEO_NOT_SUPPORTED` | 400 | Video (v1) |
| `PARALLEL_TOOLS_NOT_SUPPORTED` | 400 | Parallel tools unsupported |
| `ALL_MODELS_FAILED` | 502 | All physical models failed |
| `PROXY_ERROR` | 502 | Unexpected internal error |
| `CONVERSATION_NOT_FOUND` | 404 | Unknown conversation ID |
| `RATE_LIMIT_EXCEEDED` | 429 | Rate limit hit |

---

## Data Model

| Table | Purpose | Key Columns |
|---|---|---|
| `conversations` | Per-conversation state | `pseudo_model`, `physical_model`, `total_tokens`, capability flags |
| `conversation_turns` | Individual turns | `turn_number`, `messages`, `response`, token counts, capability flags |
| `conversation_snapshots` | Compaction snapshots | `snapshot_type`, `tokens_before`, `tokens_after`, `compactor_model` |

---

## Development

```bash
# Setup
python3.14 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Tests
pytest                          # All tests
pytest tests/ -x                # Stop on first failure
pytest tests/test_e2e.py -k streaming  # Streaming only

# Lint
ruff check src/
ruff format src/

# Run
python -m src.main              # :9110
```

---

## Deploy

GitHub Actions → `chat.guzman-lopez.com`:
- Target: `plata` (Ubuntu 22.04)
- Service user: `proxy` (hardcoded)
- Cache: Redis native :6380
- DB: SQLite (preserved between deploys)
- Verification: Health check post-deploy

Config: [.github/workflows/deploy.yml](../.github/workflows/deploy.yml)

---

## Coding Guidelines

This project uses:
- **Result monad** `Ok[T]` / `Err[E]` from `src/domain/types.py` — errors as data, not exceptions
- **match/case** for Result consumption
- **Architectura hexagonal** — domain has no infra imports
- **async-first** — `async def` in all endpoints
- **Strict typing** — no `Any`, no `# type: ignore`

See `python.md` for the full coding guide.
