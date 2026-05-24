# Sprint 6 — Explicit Compaction & Context Alerts ✅

> **Duration:** 1 week
> **Goal:** Conversations never become permanently unusable. Context alerts warn the user. Explicit compaction generates readable Markdown snapshots. Extremely large histories are compacted via arq.
> **Success criterion:** A 2M token conversation can be compacted with Gemini 3.5 Flash (1M context) and reactivated with a ~10K token snapshot.
> **Status:** ✅ COMPLETED — 331 tests pass, deployed to production.

---

## 1. Dependencies from Previous Sprints

| Dependency | Source | Status |
|---|---|---|
| `conversation_snapshots` table | Sprint 4 | ✅ Reused for explicit snapshots (`snapshot_type="explicit"`) |
| `conversations` with `active_snapshot_id` | Sprint 4 | ✅ Updated by POST /compact |
| `compactador` pseudo-model defined | Sprint 1 | ✅ Validated at startup, used by explicit compaction |
| `conversation_turns.turn_type` with `compaction_snapshot` | Sprint 2 | ✅ Column exists |
| Chat endpoint with capability detection | Sprint 1-5 | ✅ Context alerts injected before LLM call |
| `proxy_metadata` in responses | Sprint 1-5 | ✅ `context_alert` field added |
| Celery → **arq** (replaced) | Sprint 1 | ✅ `src/tasks/arq_app.py` — MIT, async-native, by Pydantic creator |

### 1.1 Actual files created/modified

```
src/
├── service/
│   ├── context_alert.py              # NEW — pure function for alert computation
│   └── compactor/
│       ├── prompts.py                # EXTEND — +build_explicit_compaction_prompt()
│       ├── pre_compactor.py          # (unchanged)
│       ├── continuous.py             # (unchanged)
│       └── explicit.py               # NEW — compact_conversation() + select_compactor_model()
│
├── api/
│   ├── chat.py                       # EXTEND — context_alert in streaming path
│   └── conversations.py              # EXTEND — +POST /compact, +GET /audit-log
│
├── tasks/
│   └── arq_app.py                    # NEW — arq worker (replaces Celery)
│
├── domain/
│   └── errors.py                     # EXTEND — +ContextUnusable, +HistoryTooLargeForCompactor
│
└── tests/
    ├── test_context_alerts.py        # NEW — 12 tests
    ├── test_explicit_compaction.py   # NEW — 14 tests
    ├── test_audit_log.py             # NEW — 6 tests
    └── test_sprint6_comprehensive.py # NEW — 15 HTTP integration tests
```

### 1.2 arq dependency (replaces Celery)

Add to `pyproject.toml`:
```toml
arq = ">=0.28,<0.30"
# MIT license, async-native (by Samuel Colvin — Pydantic creator)
# Uses Valkey as broker (already deployed)
# Falls back to synchronous compaction if arq unavailable
```

---

## 2. Context Alerts (`service/context_alert.py`) ✅

### 2.1 Alert thresholds

The proxy reports context usage in `proxy_metadata` on every response:

| Usage % | Alert level | Behavior |
|---|---|---|
| < 60% | `normal` | Just report `context_usage_pct` in metadata. No warning. |
| 60-80% | `moderate` | Warning: `"CONTEXT_MODERATE: consider compacting soon."` + `compaction_endpoint` URL. |
| 80-99% | `high` | Warning: `"CONTEXT_HIGH: compact recommended to avoid interruption."` + `compaction_endpoint` URL. |
| 100%+ | `unusable` | HTTP 400 `CONTEXT_UNUSABLE`. Request not forwarded to any model. Only action: compact. |

### 2.2 Implementation ✅

Located at `src/service/context_alert.py` (service layer per hexagonal architecture, not middleware).

- Pure function `get_context_alert()` returns frozen `ContextAlert` dataclass
- Alert levels: `none`, `normal`, `moderate`, `high`, `unusable`
- Returns `error_code: "CONTEXT_UNUSABLE"` at 100%+
- `context_alert` field in `proxy_metadata` on every response
- `CONTEXT_UNUSABLE` → HTTP 400 with `remediation.action: "compact"` + endpoint URL

### 2.3 Integration ✅

Integrated into BOTH paths:
- **Non-streaming**: `chat_service.py` → `process_chat_request()` before LLM call
- **Streaming**: `api/chat.py` → `_handle_streaming_with_db()` before LLM call

### 2.4 What context alerts do NOT do

- Do NOT auto-compact (even at 100%)
- Do NOT modify the prompt or messages
- Do NOT reduce context window dynamically
- Do NOT suppress the model's response
- Do NOT prevent the request from being sent (except at 100%)

---

## 3. Explicit Compaction (`service/compactor/explicit.py`) ✅

### 3.1 Endpoint ✅

```
POST /conversations/{id}/compact
```

### 3.2 Flow ✅

```
1. User invokes POST /conversations/{id}/compact
2. Proxy loads the full conversation history from DB
3. Proxy selects the compactador model with enough context window:
   - gemini/gemini-3.5-flash (1M ctx) → anthropic/claude-haiku-4-5 (200K) → zai/glm-4.5-flash (128K)
   - Uses by_context_window fallback strategy via select_compactor_model()
4. If history > 500K tokens, dispatch to arq (async) — replaces Celery
5. Compactor receives the history + structured compaction prompt
6. Compactor generates a Markdown snapshot (~8-12K tokens)
7. Snapshot stored in conversation_snapshots table with snapshot_type="explicit"
8. conversation.active_snapshot_id updated
9. Response includes: snapshot_id, tokens_reduced, preview of snapshot
```

### 3.3 Implementation notes ✅

- Located at `src/service/compactor/explicit.py`
- `compact_conversation()` — main entry point, async
- `select_compactor_model()` — by_context_window strategy
- `_run_compaction_sync()` — synchronous compaction path
- `_compact_async()` — helper for arq worker
- Reuses existing `ConversationSnapshot` model with `snapshot_type="explicit"`
- Snapshot chaining via `superseded_by` field (already existed from Sprint 4)
- Original history NEVER modified
- Compactador models now use DIRECT providers (not OpenRouter):
  - `gemini/gemini-3.5-flash` (Google direct, 1M ctx)
  - `anthropic/claude-haiku-4-5-20251001` (Anthropic direct, 200K ctx)
  - `zai/glm-4.5-flash` (Zhipu direct, 128K ctx)

---

## 4. Async Compaction via arq (`tasks/arq_app.py`) ✅

### 4.1 Why arq (replaces Celery)

Explicit compaction of a 2M token conversation involves:
1. Loading 2M tokens from DB (1-2 seconds)
2. Sending 2M tokens to the compactor model (10-30 seconds for Gemini)
3. Waiting for the compactor response (5-15 seconds)
4. Storing the snapshot (0.1 seconds)

Total: 20-60 seconds. This is too long for a synchronous HTTP request.

**Celery was replaced with arq** because:
- arq is MIT-licensed (Celery is BSD but heavier)
- arq is async-native — no `asyncio.new_event_loop()` hack needed
- Created by Samuel Colvin (same author as Pydantic)
- Uses Valkey/Redis as broker (already deployed)
- ~700 lines of code vs Celery's 50K+
- Falls back to synchronous compaction if arq is unavailable

### 4.2 arq task ✅

Located at `src/tasks/arq_app.py`:
- `compact_conversation_async()` — worker function discovered by arq by name
- `WorkerSettings` — configuration class for `arq src.tasks.arq_app.WorkerSettings`
- `create_arq_pool()` — helper for FastAPI lifespan (optional, returns None if unavailable)
- Import is lazy — wrapped in try/except so proxy starts even without arq

```python
# Usage
arq src.tasks.arq_app.WorkerSettings
```

### 4.3 What arq does NOT do in Sprint 6

- No auto-compaction scheduling
- No periodic tasks
- No task result caching beyond broker TTL
- No email/slack notifications on completion
- No task retry with exponential backoff (single attempt only)

---

## 5. Audit Log Endpoint ✅

### 5.1 GET /conversations/{id}/audit-log ✅

Returns a chronological log of all significant events in the conversation.
Located at `src/api/conversations.py`.

**Events tracked:**
- `conversation_created` — when the conversation was first created
- `pseudo_model_switched` — when the user changes pseudo-model
- `fallback_applied` — when a provider fails and fallback is used
- `compaction_explicit` / `compaction_continuous` / `compaction_external` — snapshot events
- `normalization_event` — when tools are normalized
- `degradation_event` — when images are described

### 5.2 Implementation ✅

No separate audit table needed. Events are constructed by scanning:
1. `conversation_turns` — for switches, fallbacks, event turns
2. `conversation_snapshots` — for compaction events
3. `conversation` — for creation event

```json
{
  "conversation_id": "abc-123",
  "events": [
    {"timestamp": "2026-01-15T10:00:00Z", "event_type": "conversation_created", ...},
    {"timestamp": "2026-01-15T10:10:00Z", "event_type": "pseudo_model_switched", ...},
    {"timestamp": "2026-01-15T14:30:00Z", "event_type": "compaction_explicit", ...},
    {"timestamp": "2026-01-15T14:32:00Z", "event_type": "fallback_applied", ...}
  ]
}
```

---

## 6. Tests (Sprint 6) ✅

### 6.1 test_context_alerts.py — 12 tests ✅

1. Context < 60% → normal alert level, no warning
2. Context 60-80% → moderate alert with warning message + endpoint URL
3. Context 80-99% → high alert with "Compact recommended"
4. Context exactly 100% → unusable with error_code
5. Context > 100% → unusable with remediation endpoint
6. Null context_window → "none" alert level (compactador)
7. Zero tokens → normal at 0%
8. Context percentage rounding to 1 decimal
9. Context at 59.5% → normal (below threshold)
10. Context at 60% → moderate (at threshold)
11. Context at 80% → high (at threshold)
12. ContextAlert dataclass is frozen (immutable)

### 6.2 test_explicit_compaction.py — 14 tests ✅

1. `select_compactor_model()` picks model with enough context window
2. Fallback to largest model when history exceeds all
3. Returns None when compactador pseudo-model missing
4. `POST /compact` generates snapshot with required fields
5. Snapshot stored in `conversation_snapshots` table
6. `active_snapshot_id` updated on conversation
7. Empty conversation → 400 `EMPTY_CONVERSATION`
8. Non-existent conversation → 404
9. History >500K tokens → dispatches to arq (mocked)
10. Multiple explicit compactions chain correctly via `superseded_by`
11. Snapshot contains all required sections
12. Long snapshot content truncated in preview with "..."

### 6.3 test_audit_log.py — 6 tests ✅

1. Includes `conversation_created` event
2. Includes `pseudo_model_switched` events
3. Includes `fallback_applied` events
4. Includes `compaction_explicit` events
5. Events sorted chronologically by timestamp
6. Non-existent conversation → 404

### 6.4 test_sprint6_comprehensive.py — 15 HTTP integration tests ✅

Full end-to-end HTTP tests with mocked LiteLLM + fakeredis:
- Context alerts at every threshold via `/v1/chat/completions`
- `CONTEXT_UNUSABLE` returns 400 with remediation (non-streaming)
- `CONTEXT_UNUSABLE` returns 400 with remediation (streaming)
- `POST /compact` → snapshot with correct fields
- `POST /compact` on empty conversation → 400
- `POST /compact` on non-existent → 404
- Multiple compactions chain correctly
- `GET /audit-log` includes creation, switches, fallbacks, compactions
- `GET /.../compatible-models` shows 8/8 safe
- Streaming SSE response includes `context_alert` in proxy_metadata
- `select_compactor_model` picks correct model

**Total: 45 Sprint 6 tests → all passing**

---

## 7. Acceptance Criteria — ✅ ALL COMPLETED

- [x] Context alerts appear in `proxy_metadata` at the correct thresholds
- [x] `CONTEXT_UNUSABLE` (400) returned when history exceeds all model windows
- [x] `POST /compact` generates a structured Markdown snapshot
- [x] Snapshot includes: Problem State, Technical Decisions, Code, Current Status, Technical Context, Pending Items, Tools & Capabilities, Conversation Metadata
- [x] Original history NEVER modified
- [x] Multiple explicit compactions chain correctly (`superseded_by`)
- [x] Histories >500K tokens dispatched to arq (async) — replaces Celery
- [x] `GET /conversations/{id}/audit-log` returns chronological event log
- [x] Audit log includes: creation, pseudo-model switches, fallbacks, compactions, degradations, normalizations
- [x] **45 Sprint 6 tests pass** (exceeds minimum of 16+)
- [x] **No regression** on Sprint 1-5 tests (331 total, 9 skipped integration)

---

## 8. Explicitly OUT OF SCOPE for Sprint 6

| Feature | Sprint | Progress |
|---|---|---|
| Provider cache optimization (cache_control, prompt_cache_key, CachedContent) | 7 | ⏳ |
| OpenCode integration testing | 7 | ⏳ |
| Auth middleware (Bearer token) | 8 | ❌ |
| CORS configuration | 8 | ❌ |
| Rate limiting | 8 | ❌ |
| Metrics endpoint (`GET /metrics`) | 8 | ❌ |
| HTTPS/Caddy setup | 8 | ❌ |
| README and deployment docs | 8 | ❌ |
| Real-time progress tracking for async compaction tasks | Future | ❌ |
| Scheduled/periodic compaction | Future | ❌ |
