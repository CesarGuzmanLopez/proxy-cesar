# Sprint 6 — Explicit Compaction & Context Alerts

> **Duration:** 1 week
> **Goal:** Conversations never become permanently unusable. Context alerts warn the user. Explicit compaction generates readable Markdown snapshots. Extremely large histories are compacted via Celery.
> **Success criterion:** A 2M token conversation can be compacted with Gemini 3 Flash (1M context) and reactivated with a ~10K token snapshot.

---

## 1. Dependencies from Previous Sprints

| Dependency | Source | Status |
|---|---|---|
| `conversation_snapshots` table | Sprint 4 | Created for continuous compaction |
| `conversations` with `active_snapshot_id` | Sprint 4 | Already used |
| `compactador` pseudo-model defined | Sprint 1 | Validated at startup |
| `conversation_turns.turn_type` with `compaction_snapshot` | Sprint 2 | Column exists |
| Chat endpoint with capability detection | Sprint 1-5 | Ready for alert injection |
| `proxy_metadata` in responses | Sprint 1-5 | Ready for new warning fields |
| Celery app stub | Sprint 1 | `src/tasks/celery_app.py` placeholder or new file |

### 1.1 New files/modules

```
src/
├── compactor/
│   └── explicit.py              # NEW — explicit compaction logic + POST /compact
│
├── api/
│   ├── conversations.py         # EXTEND — add compact endpoint, audit log endpoint
│   └── middleware/              # EXTEND — add context alert middleware
│       └── context_alert.py     # NEW — context usage alerts in proxy_metadata
│
├── tasks/
│   └── celery_app.py            # NEW or EXTEND — Celery for async compaction of large histories
│
└── tests/
    ├── test_explicit_compaction.py   # NEW
    └── test_context_alerts.py        # NEW
```

### 1.2 Celery dependency

Add to `pyproject.toml`:
```toml
celery = {version = ">=5.4,<5.5", extras = ["redis"]}
# Valkey is Redis-compatible for Celery broker
```

---

## 2. Context Alerts (`middleware/context_alert.py`)

### 2.1 Alert thresholds

The proxy reports context usage in `proxy_metadata` on every response:

| Usage % | Alert level | Behavior |
|---|---|---|
| < 60% | `normal` | Just report `context_usage_pct` in metadata. No warning. |
| 60-80% | `moderate` | Warning: `"CONTEXT_MODERATE: consider compacting soon."` + `compaction_endpoint` URL. |
| 80-99% | `high` | Warning: `"CONTEXT_HIGH: compact recommended to avoid interruption."` + `compaction_endpoint` URL. |
| 100%+ | `unusable` | HTTP 400 `CONTEXT_UNUSABLE`. Request not forwarded to any model. Only action: compact. |

### 2.2 Implementation

```python
def get_context_alert(
    total_tokens: int,
    context_window: int | None,
    conversation_id: str,
) -> dict:
    """
    Determine the context alert level and return alert metadata.
    Returns a dict with alert_level, warning message, and action URLs.
    """
    if context_window is None:
        return {"alert_level": "none", "context_usage_pct": None}

    pct = round((total_tokens / context_window) * 100, 1)

    if pct >= 100:
        return {
            "alert_level": "unusable",
            "context_usage_pct": pct,
            "warning": "CONTEXT_UNUSABLE: History exceeds all available model windows. Compaction is the only available action.",
            "compaction_endpoint": f"POST /conversations/{conversation_id}/compact",
        }
    elif pct >= 80:
        return {
            "alert_level": "high",
            "context_usage_pct": pct,
            "warning": f"CONTEXT_HIGH: {pct}% of context window used. Compact recommended.",
            "compaction_endpoint": f"POST /conversations/{conversation_id}/compact",
        }
    elif pct >= 60:
        return {
            "alert_level": "moderate",
            "context_usage_pct": pct,
            "warning": f"CONTEXT_MODERATE: {pct}% of context window used. Consider compacting soon.",
            "compaction_endpoint": f"POST /conversations/{conversation_id}/compact",
        }
    else:
        return {
            "alert_level": "normal",
            "context_usage_pct": pct,
        }
```

### 2.3 Integration into chat endpoint

```python
# In chat endpoint, AFTER loading conversation and BEFORE calling LiteLLM:

context_alert = get_context_alert(
    total_tokens=conv.total_tokens,
    context_window=pm.context_window,
    conversation_id=conversation_id,
)

if context_alert["alert_level"] == "unusable":
    raise HTTPException(
        status_code=400,
        detail={
            "error": "CONTEXT_UNUSABLE",
            "message": context_alert["warning"],
            "context_tokens": conv.total_tokens,
            "context_window": pm.context_window,
            "remediation": {
                "action": "compact",
                "endpoint": f"POST /conversations/{conversation_id}/compact",
                "description": "Compact the conversation history into a snapshot. Original history is preserved. The snapshot captures all critical technical context."
            }
        }
    )

# For moderate/high alert, include in proxy_metadata
proxy_metadata["context_alert"] = context_alert
```

### 2.4 What context alerts do NOT do

- Do NOT auto-compact (even at 100%)
- Do NOT modify the prompt or messages
- Do NOT reduce context window dynamically
- Do NOT suppress the model's response
- Do NOT prevent the request from being sent (except at 100%)

---

## 3. Explicit Compaction (`compactor/explicit.py`)

### 3.1 Endpoint

```
POST /conversations/{id}/compact
```

### 3.2 Flow

```
1. User invokes POST /conversations/{id}/compact
2. Proxy loads the full conversation history from DB
3. Proxy selects the compactador model with enough context window:
   - gemini-3.5-flash (1M ctx) → claude-haiku-4-5 (200K) → glm-4.5-flash (128K)
   - Uses by_context_window fallback strategy
4. If history > 500K tokens, dispatch to Celery (async)
5. Compactor receives the history + structured compaction prompt
6. Compactor generates a Markdown snapshot (~8-12K tokens)
7. Snapshot stored in conversation_snapshots table
8. conversation.active_snapshot_id updated
9. Response includes: snapshot_id, tokens_reduced, preview of snapshot
```

### 3.3 compact_conversation() function

```python
async def compact_conversation(
    conversation_id: str,
    db_session,
    config: ProxyConfig,
    celery_app=None,  # Optional Celery for async compaction
) -> dict:
    """
    Explicitly compact a conversation into a snapshot.
    Returns metadata about the compaction.
    """
    # Load conversation
    conv = await db_session.get(Conversation, uuid.UUID(conversation_id))
    if not conv:
        raise HTTPException(404, detail={"error": "CONVERSATION_NOT_FOUND"})

    # Load all turns
    result = await db_session.execute(
        select(ConversationTurn)
        .where(ConversationTurn.conversation_id == uuid.UUID(conversation_id))
        .order_by(ConversationTurn.turn_number)
    )
    turns = result.scalars().all()

    if not turns:
        raise HTTPException(400, detail={"error": "EMPTY_CONVERSATION", "message": "Nothing to compact."})

    # Reconstruct full history
    all_messages = []
    # Include existing snapshot if present
    if conv.active_snapshot_id:
        snapshot = await db_session.get(ConversationSnapshot, conv.active_snapshot_id)
        if snapshot:
            all_messages.append({
                "role": "system",
                "content": f"[Previous snapshot from turn {snapshot.turn_number_at_compaction}]\n\n{snapshot.snapshot_content}"
            })

    total_tokens = 0
    for turn in turns:
        all_messages.extend(turn.messages)
        total_tokens += turn.input_tokens + turn.output_tokens

    # Select compactador model
    compactador_pm = config.pseudo_models["compactador"]
    compactor_model = select_compactor_model(compactador_pm, total_tokens)

    if not compactor_model:
        raise HTTPException(400, detail={
            "error": "HISTORY_TOO_LARGE_FOR_COMPACTOR",
            "message": f"No compactador model has a context window large enough for {total_tokens} tokens.",
            "max_compactor_window": max(m.context_window or 0 for m in compactador_pm.physical_models),
        })

    # Dispatch to Celery if history is very large (>500K tokens)
    if total_tokens > 500_000 and celery_app:
        task = celery_app.send_task(
            "compact_conversation_async",
            args=[conversation_id, compactor_model],
        )
        return {
            "status": "processing",
            "task_id": task.id,
            "message": f"Compaction dispatched to background worker. Check status at GET /conversations/{conversation_id}.",
            "estimated_tokens": total_tokens,
            "compactor_model": compactor_model,
        }

    # Synchronous compaction for smaller histories
    compaction_prompt = build_explicit_compaction_prompt()

    compaction_messages = [
        {"role": "system", "content": compaction_prompt},
        {"role": "user", "content": json.dumps(all_messages, default=str)},
    ]

    try:
        response = await litellm.acompletion(
            model=compactor_model,
            messages=compaction_messages,
            max_tokens=12000,
            temperature=0.1,
        )
        snapshot_content = response.choices[0].message.content
        snapshot_tokens = response.usage.completion_tokens
    except Exception as e:
        raise HTTPException(502, detail={
            "error": "COMPACTION_FAILED",
            "message": f"Compactor model failed: {str(e)}",
            "compactor_model": compactor_model,
        })

    # Store snapshot
    new_snapshot = ConversationSnapshot(
        conversation_id=uuid.UUID(conversation_id),
        snapshot_type="explicit",
        tokens_before=total_tokens,
        tokens_after=snapshot_tokens,
        compactor_model=compactor_model,
        snapshot_content=snapshot_content,
        turn_number_at_compaction=len(turns),
    )
    db_session.add(new_snapshot)
    await db_session.flush()

    # Update conversation
    if conv.active_snapshot_id:
        old_snapshot = await db_session.get(ConversationSnapshot, conv.active_snapshot_id)
        if old_snapshot:
            old_snapshot.superseded_by = new_snapshot.id

    conv.active_snapshot_id = new_snapshot.id
    await db_session.commit()

    return {
        "status": "completed",
        "snapshot_id": str(new_snapshot.id),
        "tokens_before": total_tokens,
        "tokens_after": snapshot_tokens,
        "tokens_reduced_pct": round((1 - snapshot_tokens / max(total_tokens, 1)) * 100, 1),
        "compactor_model": compactor_model,
        "preview": snapshot_content[:500] + "..." if len(snapshot_content) > 500 else snapshot_content,
        "can_resume": True,
    }
```

### 3.4 Compactor model selection

```python
def select_compactor_model(compactador_pm: PseudoModel, total_tokens: int) -> str | None:
    """
    Select the compactador model with enough context window for the history.
    Uses by_context_window strategy: pick the first model whose context_window >= total_tokens.
    """
    for phys in compactador_pm.physical_models:
        if phys.context_window and phys.context_window >= total_tokens:
            return phys.model
    return None
```

### 3.5 Explicit compaction prompt

```python
def build_explicit_compaction_prompt() -> str:
    return """You are a conversation compactor. Your task is to create a comprehensive, structured snapshot of a long conversation history.

The snapshot MUST capture everything needed to continue the work without accessing the original history. The snapshot will be used as the starting context for future turns.

# Required Sections

## Problem State
- What problem or task is being worked on?
- What is the current status at the moment of compaction?

## Technical Decisions
- Every significant decision made, WITH its justification
- Why was approach A chosen over approach B?
- What constraints or tradeoffs influenced each decision?

## Code Produced
- Key code that establishes the current state
- Include file paths and context
- Don't include ALL code — only what's needed to continue
- If large files were created, summarize their structure

## Current Status
- **Resolved:** completed items
- **Unresolved:** pending items
- **In Progress at compaction:** what was actively being worked on

## Technical Context
- Environment variables in use
- Architecture decisions and patterns
- Project conventions, coding standards
- Dependencies (packages, services, APIs)
- Non-obvious constraints and assumptions

## Tools & Capabilities
- What tools were defined/used?
- Any patterns for tool usage?

## Pending Items
- Explicit next steps the user mentioned
- Implicit next steps based on what was in progress

## Conversation Metadata
- Duration/span of the conversation
- Number of turns compacted
- Pseudo-models used

Format as clean Markdown. Be precise and technical. This is for a developer to continue work — not a generic summary."""
```

### 3.6 POST /compact response format

```json
{
  "status": "completed",
  "snapshot_id": "snap-abc-123",
  "tokens_before": 1200000,
  "tokens_after": 10240,
  "tokens_reduced_pct": 99.1,
  "compactor_model": "gemini-3.5-flash",
  "preview": "# Snapshot — 2026-01-15T14:32:00Z\n\n## Problem State\nWorking on refactoring the authentication module...",
  "can_resume": true
}
```

---

## 4. Celery Async Compaction (`tasks/celery_app.py`)

### 4.1 Why Celery

Explicit compaction of a 2M token conversation involves:
1. Loading 2M tokens from DB (1-2 seconds)
2. Sending 2M tokens to the compactor model (10-30 seconds for Gemini)
3. Waiting for the compactor response (5-15 seconds)
4. Storing the snapshot (0.1 seconds)

Total: 20-60 seconds. This is too long for a synchronous HTTP request.

### 4.2 Celery task

```python
# src/tasks/celery_app.py

from celery import Celery
import os

celery_app = Celery(
    "proxy_cesar",
    broker=os.getenv("VALKEY_URL", "valkey://localhost:6379"),
    backend=os.getenv("VALKEY_URL", "valkey://localhost:6379"),
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,  # 5 minutes max for compaction
)

@celery_app.task(name="compact_conversation_async", bind=True)
def compact_conversation_async(self, conversation_id: str, compactor_model: str):
    """
    Async compaction task for very large conversations (>500K tokens).
    Runs in a Celery worker, not the FastAPI process.
    """
    import asyncio
    from src.compactor.explicit import _compact_async

    # Run the async compaction in the Celery worker's event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(_compact_async(conversation_id, compactor_model))
        return result
    except Exception as e:
        self.update_state(state="FAILURE", meta={"error": str(e)})
        raise
    finally:
        loop.close()
```

### 4.3 Starting Celery worker

```bash
celery -A src.tasks.celery_app worker --loglevel=info --concurrency=4
```

### 4.4 What Celery does NOT do in Sprint 6

- No auto-compaction scheduling
- No periodic tasks
- No task result caching beyond broker TTL
- No email/slack notifications on completion
- No task retry with exponential backoff (single attempt only)

---

## 5. Audit Log Endpoint

### 5.1 GET /conversations/{id}/audit-log

Returns a chronological log of all significant events in the conversation.

```json
{
  "conversation_id": "abc-123",
  "events": [
    {
      "timestamp": "2026-01-15T10:00:00Z",
      "event_type": "conversation_created",
      "details": {"pseudo_model": "normal", "physical_model": "qwen3-max"}
    },
    {
      "timestamp": "2026-01-15T10:05:00Z",
      "event_type": "turn_completed",
      "details": {"turn_number": 1, "input_tokens": 500, "output_tokens": 800}
    },
    {
      "timestamp": "2026-01-15T10:10:00Z",
      "event_type": "pseudo_model_switched",
      "details": {"from": "normal", "to": "tareas-avanzadas", "reason": "user_requested", "compatibility": "safe"}
    },
    {
      "timestamp": "2026-01-15T14:30:00Z",
      "event_type": "compaction_explicit",
      "details": {"tokens_before": 1200000, "tokens_after": 10240, "compactor": "gemini-3.5-flash"}
    },
    {
      "timestamp": "2026-01-15T14:32:00Z",
      "event_type": "fallback_applied",
      "details": {"failed_model": "qwen3-max", "fallback_model": "deepseek-v4-flash", "reason": "upstream_503"}
    }
  ]
}
```

### 5.2 Implementation

The audit log is NOT a separate table in Sprint 6. It is constructed by scanning:
1. `conversation_turns` (each turn is an event)
2. `conversation_snapshots` (each snapshot is an event)
3. Fallback info from `conversation_turns.fallback_applied`
4. Capability flag changes (can be derived by scanning turn fields)

```python
@router.get("/conversations/{conversation_id}/audit-log")
async def audit_log(conversation_id: str, request: Request):
    db = request.app.state.db_session_factory()

    conv = await db.get(Conversation, uuid.UUID(conversation_id))
    if not conv:
        raise HTTPException(404, detail={"error": "CONVERSATION_NOT_FOUND"})

    events = []

    # Conversation created
    events.append({
        "timestamp": conv.created_at.isoformat(),
        "event_type": "conversation_created",
        "details": {"pseudo_model": conv.pseudo_model, "physical_model": conv.physical_model},
    })

    # Turns
    turns = await db.execute(
        select(ConversationTurn)
        .where(ConversationTurn.conversation_id == uuid.UUID(conversation_id))
        .order_by(ConversationTurn.turn_number)
    )
    prev_pseudo = conv.pseudo_model
    for turn in turns.scalars().all():
        # Pseudo-model switch detection
        if turn.turn_type == "normal" and turn.pseudo_model != prev_pseudo:
            events.append({
                "timestamp": turn.created_at.isoformat(),
                "event_type": "pseudo_model_switched",
                "details": {"from": prev_pseudo, "to": turn.pseudo_model, "turn": turn.turn_number},
            })
            prev_pseudo = turn.pseudo_model

        # Fallback
        if turn.fallback_applied:
            events.append({
                "timestamp": turn.created_at.isoformat(),
                "event_type": "fallback_applied",
                "details": {"turn": turn.turn_number, "reason": turn.fallback_reason},
            })

        # Event turns
        if turn.turn_type != "normal":
            events.append({
                "timestamp": turn.created_at.isoformat(),
                "event_type": turn.turn_type,
                "details": {"turn": turn.turn_number, "model": turn.physical_model},
            })

    # Snapshots
    snapshots = await db.execute(
        select(ConversationSnapshot)
        .where(ConversationSnapshot.conversation_id == uuid.UUID(conversation_id))
        .order_by(ConversationSnapshot.created_at)
    )
    for snap in snapshots.scalars().all():
        events.append({
            "timestamp": snap.created_at.isoformat(),
            "event_type": f"compaction_{snap.snapshot_type}",
            "details": {
                "tokens_before": snap.tokens_before,
                "tokens_after": snap.tokens_after,
                "compactor": snap.compactor_model,
            },
        })

    # Sort by timestamp
    events.sort(key=lambda e: e["timestamp"])

    return {"conversation_id": conversation_id, "events": events}
```

---

## 6. Tests (Sprint 6)

### 6.1 test_explicit_compaction.py (minimum 8 tests)

1. `POST /compact` generates a snapshot
2. Snapshot stored in `conversation_snapshots`
3. `active_snapshot_id` updated on conversation
4. `GET /conversations/{id}` returns snapshot info
5. Next turn uses snapshot + new messages (not full history)
6. Compaction on empty conversation → 400 error
7. Very large history dispatched to Celery (mock)
8. Snapshot contains all required sections (Problem State, Technical Decisions, Code, etc.)

### 6.2 test_context_alerts.py (minimum 8 tests)

1. Context < 60% → normal alert level, no warning
2. Context 60-80% → moderate alert with warning message
3. Context 80-99% → high alert with warning message
4. Context ≥ 100% → unusable, 400 error `CONTEXT_UNUSABLE`
5. `proxy_metadata` includes `context_alert` field
6. `proxy_metadata` includes `compaction_endpoint` URL when warn/error
7. `CONTEXT_UNUSABLE` error includes remediation info
8. No alerts when `context_window` is null (e.g., compactador)

---

## 7. Acceptance Criteria

- [ ] Context alerts appear in `proxy_metadata` at the correct thresholds
- [ ] `CONTEXT_UNUSABLE` (400) returned when history exceeds all model windows
- [ ] `POST /compact` generates a structured Markdown snapshot
- [ ] Snapshot includes: Problem State, Technical Decisions, Code, Current Status, Technical Context, Pending Items
- [ ] Original history NEVER modified
- [ ] Multiple explicit compactions chain correctly (`superseded_by`)
- [ ] Histories >500K tokens dispatched to Celery (async)
- [ ] `GET /conversations/{id}/audit-log` returns chronological event log
- [ ] Audit log includes: creation, pseudo-model switches, fallbacks, compactions, degradations, normalizations
- [ ] All 16+ tests pass
- [ ] No regression on Sprint 1-5 tests

---

## 8. Explicitly OUT OF SCOPE for Sprint 6

| Feature | Sprint |
|---|---|
| Provider cache optimization (cache_control, prompt_cache_key, CachedContent) | 7 |
| OpenCode integration testing | 7 |
| Auth middleware (Bearer token) | 8 |
| CORS configuration | 8 |
| Rate limiting | 8 |
| Metrics endpoint (`GET /metrics`) | 8 |
| HTTPS/Caddy setup | 8 |
| README and deployment docs | 8 |
| Real-time progress tracking for async compaction tasks | Future |
| Scheduled/periodic compaction | Future |
