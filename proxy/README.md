# Proxy César v1.1 — Technical Documentation

**Deterministic multi-model proxy for LLMs.** A transparent HTTP proxy between your LLM client and 4 providers. Translates 7 pseudo-model names to 25+ physical models, manages conversation state with multi-turn caching, auto-describes images for non-vision models, enforces content compatibility, handles tool normalization, and applies context compaction.

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

7 pseudo-models defined in `pseudo_models.yaml`, each mapping to 1+ physical models. Fallback is `sequential` (try primary, then fallbacks in order) except `compactador` which uses `by_context_window`.

**Model aliases** bridge common names: `gpt-4o` → `normal`, `o3` → `pensamiento-profundo-caro`, etc.

### KeyVault (Security)

Intercepts `POST /v1/chat/completions`:
1. Detects 22 patterns (API keys, PEM, SSH, JWT, crypto wallets)
2. Stores in Redis `keyvault:{conv}:{hash}` (TTL 1h)
3. Replaces with `[KEYVAULT:hash]` before sending to LLM
4. Re-injects real values in response (streaming + non-streaming)

The LLM **never sees** real keys. The client **always sees** real keys.

### Content Extraction (Blob Vault)

When the user sends a file to a model that can't process it natively, the proxy extracts the content and injects it as text:

| Type | Extraction | Tool | Label |
|---|---|---|---|
| Image | Describe via vision model | Llama 4 Scout / MiMo Omni | `[v2][File extracted: image ...]` |
| PDF | Extract text | PyMuPDF (Python) | `[v2][File extracted: document ...]` |
| Word (DOCX) | Extract text | python-docx | `[v2][File extracted: document ...]` |
| Excel (XLSX) | Extract text | openpyxl | `[v2][File extracted: spreadsheet ...]` |
| PowerPoint (PPTX) | Extract text | python-pptx | `[v2][File extracted: presentation ...]` |
| Audio | Transcribe | Whisper speech-to-text | `[v2][File extracted: audio ...]` |
| Plain text | Decode as UTF-8 | text decode | `[v2][File extracted: document ...]` |

Format: extracted content is versioned (`[v2]`) and includes the filename,
size, and extraction tool. The original file is NOT accessible on the proxy
server — agents should read the extracted content directly or use tools if
the file exists in their local workspace.

Extraction is cached in Redis (24h TTL) with a composite key that includes
the content hash + prompt hash (images) or just content hash (documents).

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

### Provider Cache Optimization (Multi-Turn Prompt Caching)

Three-layer caching strategy that reduces provider costs by 60-80% in multi-turn conversations:

**Layer 1 — DB History as Stable Prefix**
When a client sends a new message in an existing conversation, the proxy reconstructs the full message array as `[DB_history] + [new_messages]`. This creates a stable prefix that providers cache across requests. *Previously broken: proxy sent only the client's single new message.*

**Layer 2 — Anthropic-Style cache_control Markers**
Applied to ALL models in the cache control set (`anthropic`, `opencode-go`):
- Breakpoint 1: on system prompt → caches across all turns
- Breakpoint 2: on penultimate message → caches conversation prefix
- Models routed through Anthropic API (`anthropic/qwen3.7-max`, `anthropic/minimax-m2.7`) and Go OpenAI-route models (`openai/mimo-v2.5`, `openai/glm-5.1`) both receive markers. *Previously missing for `opencode-go` models.*

**Layer 3 — Provider-Specific Monitoring**
- **Anthropic/Go**: `cache_read_input_tokens`, `cache_creation_input_tokens` in response
- **DeepSeek**: `prompt_cache_hit_tokens`, `prompt_cache_miss_tokens` — cache misses exposed in metadata
- **Groq**: Automatic prefix caching (>1024 tokens), tracked via `cached_tokens`
- **OpenRouter**: Transparent pass-through

**Log visibility (INFO level):**
```
⚡ cache_control provider=anthropic model=anthropic/qwen3.7-max messages=3
llm_ok  | model=openai/mimo-v2.5 prompt_tokens=249 cache_hit=192
llm_ok  | model=deepseek/deepseek-v4-flash prompt_tokens=65 cache_miss=65
```

### Provider Reasoning Support

The proxy accepts a `thinking` parameter (OpenAI-compatible) and maps it to the format each provider understands:

| Client sends | Anthropic | OpenAI | Others |
|---|---|---|---|
| `"low"` | `budget_tokens: 2048` | `reasoning_effort: "low"` | auto |
| `"medium"` | `budget_tokens: 8192` | `reasoning_effort: "medium"` | auto |
| `"high"` | `budget_tokens: 16000` | `reasoning_effort: "high"` | auto |
| `"xhigh"` | `budget_tokens: 32000` | `reasoning_effort: "high"` | auto |
| `"max"` | `budget_tokens: 64000` | `reasoning_effort: "high"` | auto |
| `"auto"` / `None` | auto (provider decides) | auto | auto |
| `True` / `"enabled"` | `thinking: {type: enabled}` | auto | auto |
| `False` / `"disabled"` | `thinking: {type: disabled}` | auto | auto |
| Dict, e.g. `{"type":"enabled","budget_tokens":2000}` | Passthrough | auto | auto |

The capability is detected per physical model:
- If `provider == "anthropic"` or `model` starts with `anthropic/` → `thinking` capability
- If `provider == "openai"` or `model` starts with `openai/` → `reasoning_effort` capability
- If model is `opencode-go` with `openai/` prefix (MiMo-V2.5, Kimi, etc.) → **both** `thinking` + `reasoning_effort` (verified: MiMo-V2.5 returns `reasoning_content` with both params)
- Otherwise → auto (no param sent, provider decides)

The `/v1/models` endpoint advertises both capabilities separately per pseudo-model.

### Rate Limiting

Per-pseudo-model fixed-window rate limiter in Redis. Headers: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`.

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `PROXY_PORT` | `9110` | HTTP port |
| `PROXY_API_KEY` | — | Bearer token (empty = dev mode) |
| `DATABASE_URL` | `postgresql+asyncpg://...` | PostgreSQL (producción) / SQLite (dev) |
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
{"status":"ok","database":"connected","valkey":"connected","pseudo_models_loaded":7}
```

### `GET /v1/models`
Lists all pseudo-models with capabilities, including:
- `thinking: bool` — models routed to Anthropic-compatible endpoints support the `thinking` dict parameter
- `reasoning_effort: bool` — models routed to OpenAI endpoints support the `reasoning_effort` string param

### `POST /v1/chat/completions`
OpenAI-compatible. Supports `stream: true` (SSE) and `stream: false`.

**Request:**
```json
{"model":"normal","messages":[{"role":"user","content":"Hello"}],"stream":false}
```

**Optional `thinking` parameter** controls reasoning effort. Accepted values:
- `"low"`, `"medium"`, `"high"`, `"xhigh"`, `"max"` — effort levels mapped per provider
- `"auto"` / `None` — let the provider decide
- `True` / `"enabled"` — enabled with provider default budget
- `False` / `"disabled"` — explicitly disabled (Anthropic only, others → auto)
- Dict `{"type": "enabled", "budget_tokens": N}` — explicit Anthropic budget (passthrough)

**Response includes `proxy_metadata`:**
```json
{
  "proxy_metadata": {
    "physical_model": "openai/mimo-v2.5",
    "pseudo_model": "normal",
    "fallback_applied": false,
    "capabilities_detected": {"has_tools": true, "has_images": false},
    "context_alert": "normal",
    "cache": {"provider_cache_hit": true, "cached_tokens": 192},
    "images_described": 0
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
| `IMAGES_NOT_SUPPORTED` | — | Removed in v1.1 — images are auto-described via vision model for non-vision models |
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

GitHub Actions deploy automático en cada push a `main`.
Salud: Health check post-deploy.

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

---

## v1.1 Changelog

| Area | Change |
|---|---|
| **Pseudo-models** | Reduced from 10 to 7. Removed: `pensamiento-rapido`, `codigo-preciso`, `massive-fast`, `flash-lowcost`. Added: `flash` (GPT-OSS 20B) |
| **Model re-assignments** | `normal` → MiMo-V2.5, `tareas-avanzadas` → MiniMax M2.7, `flash` → GPT-OSS 20B + DS V4 Flash |
| **Multi-turn caching** | New `build_conversation_messages()` includes DB history as stable prefix. `opencode-go` added to `_PROVIDERS_WITH_CACHE_CONTROL`. Cache hit/miss logged per request. DeepSeek `prompt_cache_miss_tokens` captured |
| **System prompts** | Groq models inject tool-calling instructions. GPT-OSS forces `temperature: 0.1`, `top_p: 0.9`, `parallel_tool_calls: false` |
| **Vision** | All models advertise `vision: true`. Images auto-described via vision model for non-vision models |
| **Delegation** | Documents (Word, text, CSV, etc.) now detected and delegated via `has_documents` capability |
| **Content extraction** | Centralized `_classify_content_type()`. Supports PDF (PyMuPDF), DOCX (python-docx), XLSX (openpyxl), PPTX (python-pptx), plain text. Versioned output format `[v2][File extracted: ...]` |
| **Blob references** | Removed fake `BLOB:hash` references. Agents told to read extracted content directly or use tools if file exists locally |
| **Logs** | `⚡ cache_control` at INFO. `cache_hit=N`, `cache_miss=N` in `llm_ok`. Extraction methods logged per-request |
| **Default model** | Changed from `normal` to `flash` (GPT-OSS 20B, cheaper and faster) |
| **Database** | Migrated from SQLite to PostgreSQL (production). Added `busy_timeout=15s` for SQLite dev mode |
