# Proxy César

**Deterministic multi-model proxy for LLMs.**

A transparent HTTP proxy that sits between your LLM client (Continue, LibreChat, OpenCode, etc.) and multiple LLM providers. It translates abstract pseudo-model names into concrete physical models, manages conversation state, enforces content compatibility, handles tool normalization, and applies context compaction — all while exposing a standard OpenAI-compatible API.

---

## Philosophy

This project follows a **Result monad** pattern for error handling (`Ok[T]` / `Err[E]` from `src/domain/types.py`). Domain errors are represented as data — not exceptions — and are only converted to HTTP errors at the boundary layer (routers). This keeps business logic pure, testable, and free of infrastructure concerns.

See `python.md` for the full coding guidelines.

---

## Architecture

```
src/
├── api/              # FastAPI routers — HTTP boundary
│   ├── chat.py           POST /v1/chat/completions (incl. SSE streaming)
│   ├── conversations.py  GET/POST /conversations/{id}/* (compact, degrade, normalize, audit)
│   ├── health.py         GET /health (public, no auth)
│   ├── metrics.py        GET /metrics (auth required)
│   └── models.py         GET /v1/models
├── service/          # Business logic orchestration
│   ├── chat_service.py      Main chat flow orchestrator
│   ├── chat_models.py       ChatResult, FallbackInfo, proxy_metadata builder
│   ├── model_resolver.py    Pseudo-model → physical model resolution + aliases
│   ├── capability_detector.py  Turn/session capability detection + token counting
│   ├── compatibility.py     Content validation + pseudo-model switch validation
│   ├── threshold_guard.py   Input threshold guard (Result monad)
│   ├── tool_filter.py       Filter physical models by parallel tool support
│   ├── tools_canonical.py   Tool level determination, ID validation
│   ├── tools_normalizer.py  Parallel tool call serialization
│   ├── tools_edge_cases.py  Streaming tool assembly, thinking extraction, truncation
│   ├── context_alert.py     Context usage alerts (60%, 80%, 100%)
│   ├── compactor/
│   │   ├── pre_compactor.py    Pre-request input summarization
│   │   ├── continuous.py       Continuous compaction + external detection
│   │   ├── explicit.py         POST /compact
│   │   └── prompts.py          Compaction prompts
│   ├── multimedia/
│   │   └── image_describer.py  Auto-describe images
│   └── router_llm/
│       └── suggester.py        Task complexity evaluation
├── domain/           # Pure domain types — no infrastructure imports
│   ├── types.py            Ok[T] / Err[E] Result monad (frozen dataclasses)
│   ├── errors.py           Domain error types (11 types)
│   ├── capabilities.py     SessionCapabilities, CompatibilityResult
│   └── affinity.py         AffinityPort protocol (get/set/delete)
├── adapters/         # Infrastructure implementations
│   ├── db/models.py        SQLModel ORM (3 tables)
│   ├── db/engine.py        Async SQLAlchemy engine
│   ├── litellm/client.py   LiteLLM adapter (call_litellm)
│   └── cache/
│       ├── valkey_affinity.py   Physical model affinity store (Redis/Valkey)
│       ├── message_ordering.py  Canonical ordering
│       └── provider_cache.py    Provider-specific cache
├── config/           # Configuration
│   ├── pseudo_models.py    YAML loader + strict Pydantic validation
│   └── settings.py         Environment settings (pydantic-settings)
├── middleware/
│   └── rate_limiter.py     Per-pseudo-model rate limiter (Redis)
├── auth.py                Bearer token auth middleware
├── logging_config.py       Structured JSON logging
├── tasks/
│   └── arq_app.py          Async compaction via arq
└── schemas/
    └── tools.py            Pydantic request/response schemas
```

---

## Key Features

### Pseudo-Models

Abstract model identities that map to 1+ physical models (actual LiteLLM provider IDs). Example: `normal` maps to `openai/kimi-k2.5` → `deepseek/deepseek-v4-flash` with automatic fallback. **10 pseudo-models** are defined in `pseudo_models.yaml`, each validated at startup against strict Pydantic rules.

**Model aliases** bridge common model names to pseudo-models (`gpt-4o` → `normal`, `o3` → `pensamiento-profundo-caro`, etc.). Unknown models fall back to the default alias.

### Content Compatibility

Incoming content is validated before reaching any provider:
- **Images** → blocked unless the pseudo-model has a vision-capable physical model
- **Audio / Video** → blocked (not supported in v1)
- **PDF** → treated as images, requires vision
- **Parallel tools** → blocked unless a physical model supports them (with a normalization escape hatch)

### Tool Normalization

`POST /conversations/{id}/normalize-tools` serializes parallel tool calls into sequential `[TOOL_SERIALIZED]` messages. Supports `dry_run` preview. The original history is always preserved — a `normalization_event` turn is appended.

### Context Compaction

Three complementary strategies keep context within model limits:
- **Pre-compaction**: Before calling the target model, a cheap compactor model summarizes the latest user message if input exceeds threshold
- **Continuous compaction**: When context usage exceeds `trigger_pct` of `context_window`, old turns are compacted into a `ConversationSnapshot` while preserving recent turns
- **External compaction detection**: Detects when the client has already compacted history (message count drops >60%, first message is long system/user) and integrates into the snapshot chain

### Capability Detection

Per-turn scanning of messages detects capabilities (images, audio, PDF, video, tools, parallel tools). Flags are **additive** — once set, they never reset. Accumulated capabilities are stored on the `Conversation` row and used for compatibility validation on pseudo-model switches.

### Physical Model Affinity

Conversations are pinned to their first physical model via Redis cache (`conv:{id}:physical_model`, 24h TTL). This ensures consistent behavior across turns — the same model handles the entire conversation unless a capability-based filter forces a change.

### Model Aliases

Bridges client model names to pseudo-models without modifying clients. Configured in `pseudo_models.yaml`:
- `gpt-4o` / `gpt-4o-mini` → `normal`
- `gpt-4.1` → `tareas-avanzadas`
- `o3` → `pensamiento-profundo-caro`
- `o4-mini` → `pensamiento-rapido`
- `gemini-2.5-flash` → `vision`
- `gemini-2.5-pro` / `claude-sonnet-4-20250514` → `codigo-preciso`
- `claude-haiku-3-5-20241022` → `flash-lowcost`
- `default: normal` catches unknown model names (client-friendly)

### Provider Cache Optimization

Three-layer strategy maximizes cache hits:
- **Layer 1 — Redis affinity**: Every turn uses the same physical model
- **Layer 2 — Canonical ordering**: Messages as `system → tools(sorted) → history → new query`
- **Layer 3 — Provider-specific**: Anthropic `cache_control` breakpoints, DeepSeek auto-caching

Every response includes `proxy_metadata.cache` with hit/miss info and estimated savings.

### Authentication

Bearer `Authorization: Bearer <PROXY_API_KEY>` on all endpoints except `/health`. Dev mode (no auth) when `PROXY_API_KEY` is empty. 401 errors with descriptive messages.

### Rate Limiting

Per-pseudo-model fixed-window rate limiter in Redis. `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset` headers on every response. Configurable via `RATE_LIMIT_*` env vars.

### Structured Logging

All proxy decisions logged as JSON lines to stdout: conversation creation, affinity, fallbacks, switches, compactions, auth failures, rate limit hits, provider errors.

### Metrics

`GET /metrics` returns aggregated telemetry: requests, tokens (input/output/cached), cache hit rate, compaction savings, fallbacks, active conversations, error breakdown.

### Fallback

If a physical model returns `503 ServiceUnavailable` or `429 RateLimit`, the proxy automatically falls back to the next physical model in the pseudo-model's list. If all models fail, a `502 ALL_MODELS_FAILED` error is returned with the list of attempted models.

### Streaming

`POST /v1/chat/completions` with `stream: true` returns SSE chunks. Every stream terminates with a `proxy_metadata` chunk containing compaction info, capability context, and fallback history. Errors during streaming are reported as SSE error chunks (not connection drops).

---

## Quick Start

```bash
# Install
python3.14 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Configure
cp .env.example .env  # Set your API keys

# Run (creates SQLite DB + tables on first start)
python -m src.main

# Verify
curl http://localhost:9110/health
```

### Configuration

| Variable | Default | Description |
|---|---|---|
| `PROXY_PORT` | `9110` | HTTP listen port |
| `PROXY_API_KEY` | — | Bearer token for API access (empty = dev mode) |
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated allowed origins |
| `DATABASE_URL` | `sqlite+aiosqlite:///./proxy.db` | SQLite database (PostgreSQL also supported) |
| `VALKEY_URL` | `valkey://localhost:6380` | Redis/Valkey connection (native, port 6380) |
| `KEYCLAW_ENABLED` | `false` | KeyClaw MITM proxy (disabled by default) |
| `LOG_LEVEL` | `INFO` | Log level (DEBUG, INFO, WARNING, ERROR) |
| `OPENCODE_API_KEY` | — | OpenCode Go API key (primary provider) |
| `DEEPSEEK_API_KEY` | — | DeepSeek API key (fallbacks) |
| `GROQ_API_KEY` | — | Groq API key (whisper, vision fallback) |
| `OPENROUTER_API_KEY` | — | OpenRouter API key (normal-gratis) |
| `PRUNA_API_KEY` | — | Pruna API key |

Pseudo-models are defined in `pseudo_models.yaml`. See the file for available models and their physical model mappings.

---

## API Reference

### `GET /health`

Service health check. Never returns 500.

```json
{
  "status": "ok",
  "database": "connected",
  "valkey": "connected",
  "providers": {
    "opencode-go": "configured",
    "deepseek": "configured",
    "groq": "configured",
    "openrouter": "configured",
    "pruna": "configured"
  },
  "disabled_providers": "none",
  "pseudo_models_loaded": 10
}
```

### `GET /v1/models`

List all pseudo-models with optimistic capability advertising (prevents clients from silently stripping content).

### `POST /v1/chat/completions`

OpenAI-compatible chat completion. Supports `stream: true` (SSE) and `stream: false`.

**Request:**
```json
{
  "model": "normal",
  "messages": [{"role": "user", "content": "Hello"}],
  "stream": false,
  "conversation_id": null,
  "tools": null,
  "tool_choice": null
}
```

**Response (non-streaming):**
```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "model": "openai/kimi-k2.5",
  "choices": [{"index": 0, "message": {...}, "finish_reason": "stop"}],
  "usage": {...},
  "proxy_metadata": {
    "physical_model": "openai/kimi-k2.5",
    "pseudo_model": "normal",
    "fallback_applied": false,
    ...
  }
}
```

### `GET /conversations/{id}`

Full conversation state with capabilities and turn count.

### `GET /conversations/{id}/compatible-models`

Compatibility matrix for switching to another pseudo-model. Returns `safe`, `warning`, or `blocked` for each model with remediation guidance.

### `GET /conversations/{id}/tools-compatibility`

Tool-specific analysis per pseudo-model: parallel eligibility, strict vs non-strict models.

### `POST /conversations/{id}/normalize-tools`

Serializes parallel tool calls. Use `dry_run: true` to preview without modifying history.

---

## Error Dictionary

All errors follow the structure:
```json
{
  "detail": {
    "error": "ERROR_CODE",
    "message": "Human-readable explanation",
    "...": "Additional context fields"
  }
}
```

### Chat Completion Errors

| Error Code | Status | Source | Trigger | Additional Fields |
|---|---|---|---|---|
| `UNKNOWN_PSEUDO_MODEL` | 400 | `_resolve_and_validate`, `_handle_streaming` | The `model` field in the request does not match any known pseudo-model or alias | `available: list[str]` — all valid pseudo-model names |
| `PSEUDO_MODEL_INCOMPATIBLE` | 409 | `_resolve_session_conv_and_models`, `_handle_streaming` | Switching pseudo-models is blocked by accumulated capabilities (images on non-vision, parallel tools, audio, etc.) | `remediation: list[str]`, `details: dict`, `from_pseudo_model: str`, `to_pseudo_model: str` |
| `INPUT_EXCEEDS_THRESHOLD` | 400 | `_raise_if_exceeds_threshold` | Input token count exceeds the pseudo-model's `input_token_threshold` and pre-compaction is disabled | `suggestions: list[dict]` — pseudo-models with higher thresholds (non-streaming path only) |
| `INPUT_EXCEEDS_THRESHOLD` | 400 | `_handle_streaming` | Same as above, for streaming requests | *(no suggestions field in streaming path)* |
| `IMAGES_NOT_SUPPORTED_BY_PSEUDO_MODEL` | 400 | `validate_incoming_content` | Request contains images but the pseudo-model has no vision-capable physical models | `remediation: list[str]`, `current_pseudo_model: str`, `vision_capable_pseudo_models: list[str]` |
| `AUDIO_NOT_SUPPORTED` | 400 | `validate_incoming_content` | Request contains audio content (no model supports audio in v1) | `remediation: list[str]` |
| `PDF_NOT_SUPPORTED` | 400 | `validate_incoming_content` | Request contains a PDF but the pseudo-model has no vision-capable physical models | `remediation: list[str]` |
| `VIDEO_NOT_SUPPORTED` | 400 | `validate_incoming_content` | Request contains video content (not supported in v1) | `remediation: list[str]` |
| `PARALLEL_TOOLS_NOT_SUPPORTED_BY_PSEUDO_MODEL` | 400 | `validate_incoming_content` | Request has parallel tool calls but the pseudo-model has no models with `parallel_tools: true` | `remediation: list[str]` |
| `ALL_MODELS_FAILED` | 502 | `call_with_fallback` | All physical models in the pseudo-model failed with `ServiceUnavailableError` or `RateLimitError` | `attempted: list[str]`, `last_error: str` |
| `PROXY_ERROR` | 502 | `chat.py` catch-all | An unexpected exception escaped the chat completion pipeline | `message: str` |
| `PROXY_STREAM_ERROR` | N/A (SSE) | `_stream_response_generator` | An error occurred during streaming — reported as an SSE chunk with `finish_reason: "error"` followed by `[DONE]` | `physical_model: str`, `pseudo_model: str` |

### Conversation Endpoint Errors

| Error Code | Status | Source | Trigger | Additional Fields |
|---|---|---|---|---|
| `CONVERSATION_NOT_FOUND` | 404 | All 4 `/conversations/{id}` endpoints | The conversation ID does not exist in the database | `message: str` |
| `NO_TURNS` | 400 | `POST /conversations/{id}/normalize-tools` | The conversation exists but has no turns to normalize | `message: str` |
| `NORMALIZATION_FAILED` | 500 | `POST /conversations/{id}/normalize-tools` catch-all | An unexpected error during tool normalization | `message: str` |

### Domain Error Types (Internal)

These are never returned directly in HTTP responses but drive the business logic internally via the `Result` monad:

| Error Type | Fields | Used By |
|---|---|---|
| `PseudoModelNotFound` | `name: str` | `model_resolver.py` |
| `PhysicalModelNotInList` | `model: str`, `pseudo_model: str` | `model_resolver.py` |
| `AllModelsFailed` | `attempted: tuple[str, ...]`, `last_error: str` | `call_with_fallback` |
| `ConversationNotFound` | `conversation_id: str` | Conversation endpoints |
| `CapabilityIncompatible` | `reason: str`, `remediation: list[str]`, `details: dict` | Switch validation |
| `InputExceedsThreshold` | `estimated: int`, `threshold: int`, `pseudo_model: str` | `threshold_guard.py` |
| `ContentNotSupported` | `content_type: str`, `pseudo_model: str`, `remediation: list[str]` | Content validation |
| `CompactorFailed` | `compactor_model: str`, `reason: str`, `original_input_preserved: bool` | Compactor pipeline |
| `CompactorNotFound` | `compactor_name: str`, `pseudo_model: str` | Compactor pipeline |

---

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check src/
ruff format src/

# The Result monad (src/domain/types.py)
# Errors as data, not exceptions:
#   Ok(value)  → success path
#   Err(error) → failure path
# Use match/case to consume:
#   match result:
#       case Ok(value=val): ...
#       case Err(error=err): ...
```

### Test Suite

```bash
pytest                          # All tests (401+ tests)
pytest tests/ -x                # Stop on first failure
pytest tests/ -v                # Verbose output
pytest tests/test_streaming.py  # Streaming-specific tests
pytest tests/ -k "not streaming"  # All non-streaming tests
```

---

## Deploy

The project deploys to `chat.guzman-lopez.com` via GitHub Actions on push to `main`:

- **Runner**: `ubuntu-latest`
- **Target**: `plata` server (Ubuntu 22.04)
- **Service user**: `proxy` (hardcoded, not dynamic)
- **Cache**: Native Redis on port 6380 (not Docker)
- **Database**: SQLite, preserved between deploys (backup/restore)
- **Verification**: Health check automatically after restart

Config: [.github/workflows/deploy.yml](../.github/workflows/deploy.yml)

---

## Data Model

| Table | Purpose | Key Columns |
|---|---|---|
| `conversations` | Per-conversation state | `pseudo_model`, `physical_model`, `total_tokens`, capability flags, `max_tools_level`, `active_snapshot_id` |
| `conversation_turns` | Individual turns | `turn_number`, `messages`, `response`, `input_tokens`, `output_tokens`, capability flags, `tool_definitions`, `thinking_blocks`, `tools_level_used` |
| `conversation_snapshots` | Compaction snapshots | `snapshot_type`, `tokens_before`, `tokens_after`, `compactor_model`, `snapshot_content`, `turn_number_at_compaction` |

All tables are auto-created on startup via `SQLModel.metadata.create_all` (no Alembic required for SQLite).
