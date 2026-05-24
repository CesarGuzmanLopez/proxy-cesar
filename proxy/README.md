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
├── api/            # FastAPI routers — HTTP boundary, request/response
│   ├── chat.py         POST /v1/chat/completions
│   ├── conversations.py  GET/POST /conversations/{id}
│   ├── health.py       GET /health
│   └── models.py       GET /v1/models
├── service/        # Business logic orchestration
│   ├── chat_service.py      Main chat flow (resolve, validate, fallback, save)
│   ├── chat_models.py       ChatResult, FallbackInfo, proxy metadata
│   ├── model_resolver.py    Pseudo-model → physical model resolution
│   ├── capability_detector.py  Turn/session capability detection + token counting
│   ├── compatibility.py     Content validation + pseudo-model switch validation
│   ├── threshold_guard.py   Input threshold guard (uses Result monad)
│   ├── tool_filter.py       Filter physical models by parallel tool support
│   ├── tools_normalizer.py  Parallel tool call serialization
│   ├── tools_canonical.py   Tool level determination, ID validation
│   ├── tools_edge_cases.py  Streaming tool assembly, thinking extraction, truncation
│   └── compactor/           Context compaction pipeline
│       ├── pre_compactor.py    Pre-request input summarization
│       └── continuous.py       Continuous + external compaction detection
├── domain/         # Pure domain types — no infrastructure imports
│   ├── types.py            Ok[T] / Err[E] Result monad
│   ├── errors.py           Domain error types (frozen dataclasses)
│   ├── capabilities.py     SessionCapabilities, TurnCapabilities, CompatibilityResult
│   └── affinity.py         AffinityPort protocol
├── adapters/       # Infrastructure implementations
│   ├── db/models.py        SQLModel ORM (Conversation, ConversationTurn, ConversationSnapshot)
│   ├── litellm/client.py   LiteLLM adapter (call_litellm)
│   └── cache/valkey_affinity.py  Valkey physical model affinity store
├── config/         # Configuration
│   ├── pseudo_models.py    YAML loader + Pydantic validation
│   └── settings.py         Environment settings (pydantic-settings)
└── schemas/        # Pydantic request/response schemas
    └── tools.py            NormalizeToolsRequest, NormalizeToolsResponse
```

---

## Key Features

### Pseudo-Models

Abstract model identities that map to 1+ physical models (actual LiteLLM provider IDs). Example: `normal` maps to `openrouter/qwen3-max` → `openrouter/deepseek-v4-flash` with automatic fallback. 8 pseudo-models are defined in `pseudo_models.yaml`, each validated at startup against 14 strict rules.

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

Conversations are pinned to their first physical model via Valkey cache (`conv:{id}:physical_model`, 24h TTL). This ensures consistent behavior across turns — the same model handles the entire conversation unless a capability-based filter forces a change.

### Fallback

If a physical model returns `503 ServiceUnavailable` or `429 RateLimit`, the proxy automatically falls back to the next physical model in the pseudo-model's list. If all models fail, a `503 ALL_MODELS_FAILED` error is returned with the list of attempted models.

### Streaming

`POST /v1/chat/completions` with `stream: true` returns SSE chunks. Every stream terminates with a `proxy_metadata` chunk containing compaction info, capability context, and fallback history. Errors during streaming are reported as SSE error chunks (not connection drops).

---

## Quick Start

```bash
# Install
python -m venv .venv && source .venv/bin/activate
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
| `DATABASE_URL` | `sqlite+aiosqlite:///./proxy.db` | Database connection string |
| `VALKEY_URL` | `valkey://localhost:6379` | Cache connection string |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `OPENROUTER_API_KEY` | — | OpenRouter API key |
| `GOOGLE_API_KEY` | — | Google AI API key |
| `DEEPSEEK_API_KEY` | — | DeepSeek API key |
| `GROQ_API_KEY` | — | Groq API key |
| `ZHIPU_API_KEY` | — | Zhipu API key |

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
    "anthropic": "configured",
    "openrouter": "configured",
    "google": "configured",
    "deepseek": "configured",
    "groq": "not configured",
    "zhipu": "configured"
  },
  "pseudo_models_loaded": 8
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

Includes a `proxy_metadata` field in the final streaming chunk (or in a non-streaming response) with compaction, capability, and fallback information.

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
| `ALL_MODELS_FAILED` | 503 | `call_with_fallback` | All physical models in the pseudo-model failed with `ServiceUnavailableError` or `RateLimitError` | `attempted: list[str]`, `last_error: str` |
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
pytest                          # All tests (178+ tests)
pytest tests/ -x                # Stop on first failure
pytest tests/ -v                # Verbose output
pytest tests/test_streaming.py  # Streaming-specific tests
```

---

## Data Model

| Table | Purpose | Key Columns |
|---|---|---|
| `conversations` | Per-conversation state | `pseudo_model`, `physical_model`, `total_tokens`, capability flags, `max_tools_level`, `active_snapshot_id` |
| `conversation_turns` | Individual turns | `turn_number`, `messages`, `response`, `input_tokens`, `output_tokens`, capability flags, `tool_definitions`, `thinking_blocks`, `tools_level_used` |
| `conversation_snapshots` | Compaction snapshots | `snapshot_type`, `tokens_before`, `tokens_after`, `compactor_model`, `snapshot_content`, `turn_number_at_compaction` |

All tables are auto-created on startup via `SQLModel.metadata.create_all` (no Alembic required for SQLite).
