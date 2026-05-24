# Sprint 4 — Pre-compaction & Continuous Compaction

> **Duration:** 2 weeks
> **Status:** ✅ COMPLETE — 216 tests passing (178 Sprint 1-3 + 38 Sprint 4)
> **Goal:** Expensive models never receive enormous inputs or saturated contexts. Pre-compaction summarizes long inputs with a cheap model before the expensive one sees them. Continuous compaction keeps long-running conversations within budget by snapshotting old turns.
> **Success criterion:** 80K tokens of logs to `pensamiento-profundo-caro` → pre-compacted with `deep-flash` to ~8K → user sees exact savings. 30 turns → compaction triggers at 70% → context stays below limit.
> **Completed:** 2026-05-24

---

## 1. Dependencies from Previous Sprints

| Dependency | Source | Status |
|---|---|---|
| `pseudo_models.yaml` with `pre_compaction` and `continuous_compaction` config | Sprint 1 | Already validated |
| Input threshold check (`INPUT_EXCEEDS_THRESHOLD`) | Sprint 2 | Already implemented |
| Chat endpoint with message flow | Sprint 1-2 | Ready for compaction injection |
| DB: `conversation_turns` with `turn_type` | Sprint 2 | Already have `compaction_snapshot` type |
| DB: `conversations` with token tracking | Sprint 1 | Already tracking `total_tokens` |

### 1.1 New files/modules

```
src/
├── compactor/
│   ├── __init__.py
│   ├── pre_compactor.py         # NEW — pre-compaction logic
│   ├── continuous.py            # NEW — continuous compaction logic
│   └── prompts.py               # NEW — compaction prompts (pre and continuous)
│
└── tests/
    ├── test_pre_compaction.py       # NEW
    └── test_continuous_compaction.py # NEW
```

### 1.2 DB changes

**New table: `conversation_snapshots`**

| Column | Type | Default | Notes |
|---|---|---|---|
| `id` | UUID | PK, `gen_random_uuid()` | |
| `conversation_id` | UUID | FK → conversations.id, NOT NULL | |
| `created_at` | TIMESTAMPTZ | NOT NULL, `NOW()` | |
| `snapshot_type` | VARCHAR(32) | NOT NULL | `continuous`, `explicit` (Sprint 6), or `external` (client-side compaction detected) |
| `tokens_before` | BIGINT | NOT NULL | Tokens in the history at compaction time |
| `tokens_after` | INTEGER | NOT NULL | Tokens in the generated snapshot |
| `compactor_model` | VARCHAR(256) | NOT NULL | Physical model that generated the snapshot |
| `snapshot_content` | TEXT | NOT NULL | Markdown snapshot |
| `turn_number_at_compaction` | INTEGER | NOT NULL | Which turn triggered the compaction |
| `superseded_by` | UUID | NULLABLE | FK → conversation_snapshots.id (chain) |

**Add to `conversations`:**

| Column | Type | Default | Notes |
|---|---|---|---|
| `active_snapshot_id` | UUID | NULLABLE | FK → conversation_snapshots.id. The currently active snapshot. |

---

## 2. Pre-compaction (`compactor/pre_compactor.py`)

### 2.1 When it activates

Pre-compaction activates ONLY when ALL of:

1. `pseudo_model.pre_compaction.enabled == true` (in `pseudo_models.yaml`)
2. Estimated input tokens > `pseudo_model.pre_compaction.threshold`
3. The compactor pseudo-model (`pre_compaction.compactor`) exists and has at least one working physical model

Currently, only `pensamiento-profundo-caro` has `pre_compaction.enabled: true`.

### 2.2 Flow

```
1. Proxy detects: input 80K tokens > threshold 32K
2. Proxy calls the compactor pseudo-model (e.g., "deep-flash") with a compaction prompt
3. Compactor returns a structured summary (~8K tokens)
4. Proxy REPLACES the last user message with the summary
5. The expensive model receives the summary, not the 80K raw input
6. proxy_metadata reports pre_compaction details and estimated savings
```

### 2.3 pre_compact_input() function

```python
async def pre_compact_input(
    messages: list[dict],
    pseudo_model: PseudoModel,
    config: ProxyConfig,
    conversation_id: str,
) -> tuple[list[dict], dict]:
    """
    Pre-compact the input using the configured compactor pseudo-model.
    Returns (modified_messages, compaction_metadata).

    Only modifies the LAST user message. System messages, tool history,
    and assistant messages are passed through unchanged.
    """
    threshold = pseudo_model.pre_compaction.threshold
    target_tokens = pseudo_model.pre_compaction.target_tokens
    compactor_name = pseudo_model.pre_compaction.compactor

    # Estimate input tokens
    input_tokens = estimate_tokens(messages)

    if input_tokens <= threshold:
        return messages, {"applied": False, "reason": "below_threshold"}

    # Find the last user message to compact
    last_user_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user_idx = i
            break

    if last_user_idx is None:
        return messages, {"applied": False, "reason": "no_user_message"}

    # Get the compactor model
    compactor_pm = config.pseudo_models[compactor_name]
    compactor_model = compactor_pm.physical_models[0].model

    # Build compaction prompt
    user_message = messages[last_user_idx]
    compaction_prompt = build_pre_compaction_prompt(
        user_content=str(user_message.get("content", "")),
        target_tokens=target_tokens,
    )

    # Call compactor
    compaction_messages = [
        {"role": "user", "content": compaction_prompt}
    ]

    try:
        response = await litellm.acompletion(
            model=compactor_model,
            messages=compaction_messages,
            max_tokens=target_tokens,
        )
        summary = response.choices[0].message.content
        summary_tokens = response.usage.completion_tokens
    except Exception as e:
        # Compactor failed — pass through original input with warning
        return messages, {
            "applied": False,
            "reason": f"compactor_failed: {str(e)}",
            "warning": "Pre-compaction failed. Proceeding with original input."
        }

    # Replace the user message with the summary
    modified = list(messages)  # shallow copy
    modified[last_user_idx] = {
        "role": "user",
        "content": (
            f"[Pre-compacted input — original: {input_tokens} tokens, "
            f"compacted by {compactor_name}]\n\n{summary}"
        ),
    }

    metadata = {
        "applied": True,
        "original_input_tokens": input_tokens,
        "compacted_input_tokens": summary_tokens,
        "compactor_model": compactor_model,
        "compactor_pseudo_model": compactor_name,
        "estimated_savings_tokens": input_tokens - summary_tokens,
    }

    return modified, metadata
```

### 2.4 Pre-compaction prompt

```python
def build_pre_compaction_prompt(user_content: str, target_tokens: int) -> str:
    """
    Build the prompt for the compactor model.
    The compactor extracts relevant information, not a generic summary.
    """
    return f"""You are a pre-compactor for an expensive reasoning model.

Your job: Extract from the following text ONLY the information relevant for the user's task.
The user's request is embedded in the text below.

Rules:
1. Preserve all technical details: code snippets, error messages, log lines, file paths, version numbers
2. Preserve all constraints and requirements the user mentioned
3. Remove noise: repeated lines, irrelevant stack traces, boilerplate
4. Structure the output: use sections if the input contains multiple topics
5. Target length: approximately {target_tokens} tokens
6. DO NOT add analysis, suggestions, or commentary — just extract and organize

--- INPUT BELOW ---

{user_content}

--- END INPUT ---

Extracted content (max {target_tokens} tokens):"""
```

### 2.5 What pre-compaction does NOT do

- Does NOT compact anything other than the last user message
- Does NOT compact system messages, tool history, or assistant responses
- Does NOT activate unless explicitly configured in `pseudo_models.yaml`
- Does NOT silently fail — always reports status in `proxy_metadata`
- Does NOT compact if the compactor pseudo-model has no working models
- Does NOT modify the original message in DB (the stored messages column has the full original; the compacted version is only sent to the expensive model)

---

## 3. Continuous Compaction (`compactor/continuous.py`)

### 3.1 When it activates

Continuous compaction activates when ALL of:

1. `pseudo_model.continuous_compaction.enabled == true`
2. `conversation.total_tokens > (pseudo_model.context_window * pseudo_model.continuous_compaction.trigger_pct / 100)`
3. There are enough old turns to compact (at least 3 turns)

Currently enabled on: `pensamiento-profundo-caro` (70%), `tareas-avanzadas` (75%), `normal` (80%)

### 3.2 Flow

```
Turn 1-19: normal processing
Turn 20: context = 150K tokens (75% of 200K window) → TRIGGER
  → Identify turns to compact: turns 1-15 (oldest)
  → Send turns 1-15 + compaction prompt to deep-flash
  → Generate snapshot (~8K tokens)
  → Store snapshot in conversation_snapshots table
  → Set conversation.active_snapshot_id
  → Next turn: model receives [snapshot] + [turns 16-20] = ~48K tokens (was 150K)

Turn 21-34: no trigger (context < trigger_pct)
Turn 35: context = 95K tokens (47%) → NO TRIGGER

Turn 50: context = 165K tokens (82%) → TRIGGER
  → Compact: [old snapshot + turns 16-45] into new snapshot (~12K tokens)
  → Next turn: model receives [new snapshot] + [turns 46-50] = ~40K tokens
```

### 3.3 continuous_compact() function

```python
async def continuous_compact(
    conversation: Conversation,
    pseudo_model: PseudoModel,
    config: ProxyConfig,
    db_session,
) -> dict:
    """
    Perform continuous compaction on a conversation.
    Returns metadata about the compaction.
    """
    trigger_pct = pseudo_model.continuous_compaction.trigger_pct
    preserve_recent = pseudo_model.continuous_compaction.compact_preserve_recent
    context_window = pseudo_model.context_window

    # Load all turns
    result = await db_session.execute(
        select(ConversationTurn)
        .where(ConversationTurn.conversation_id == conversation.id)
        .order_by(ConversationTurn.turn_number)
    )
    turns = result.scalars().all()

    # Determine which turns to compact
    # Compact everything except the most recent turns (preserve_recent tokens worth)
    compact_turns = []
    preserved_turns = []
    accumulated_tokens = 0

    for turn in reversed(turns):
        turn_tokens = turn.input_tokens + turn.output_tokens
        if accumulated_tokens + turn_tokens <= preserve_recent:
            preserved_turns.insert(0, turn)
            accumulated_tokens += turn_tokens
        else:
            compact_turns.insert(0, turn)

    if len(compact_turns) < 3:
        return {"applied": False, "reason": "not_enough_turns_to_compact"}

    # Build history to compact
    # If there's an existing active snapshot, include it at the start
    history_to_compact = []
    if conversation.active_snapshot_id:
        snapshot = await db_session.get(ConversationSnapshot, conversation.active_snapshot_id)
        if snapshot:
            history_to_compact.append({
                "role": "system",
                "content": f"[Previous snapshot from turn {snapshot.turn_number_at_compaction}]\n\n{snapshot.snapshot_content}"
            })

    for turn in compact_turns:
        history_to_compact.extend(turn.messages)

    # Build compaction prompt
    compaction_prompt = build_continuous_compaction_prompt()

    # Select compactor model — use a cheap model from config
    # Reuse the pre_compaction compactor if available, otherwise use deep-flash
    compactor_name = pseudo_model.pre_compaction.compactor if pseudo_model.pre_compaction.enabled else "deep-flash"
    compactor_pm = config.pseudo_models[compactor_name]
    compactor_model = compactor_pm.physical_models[0].model

    # Call compactor
    compaction_messages = [
        {"role": "system", "content": compaction_prompt},
        {"role": "user", "content": json.dumps(history_to_compact, default=str)},
    ]

    estimated_input = estimate_tokens(history_to_compact)

    try:
        response = await litellm.acompletion(
            model=compactor_model,
            messages=compaction_messages,
            max_tokens=8000,  # Target snapshot size
        )
        snapshot_content = response.choices[0].message.content
        snapshot_tokens = response.usage.completion_tokens
    except Exception as e:
        return {"applied": False, "reason": f"compactor_failed: {str(e)}"}

    # Store snapshot
    new_snapshot = ConversationSnapshot(
        conversation_id=conversation.id,
        snapshot_type="continuous",
        tokens_before=estimated_input,
        tokens_after=snapshot_tokens,
        compactor_model=compactor_model,
        snapshot_content=snapshot_content,
        turn_number_at_compaction=len(turns),
    )
    db_session.add(new_snapshot)
    await db_session.flush()

    # Update conversation
    if conversation.active_snapshot_id:
        old_snapshot = await db_session.get(ConversationSnapshot, conversation.active_snapshot_id)
        if old_snapshot:
            old_snapshot.superseded_by = new_snapshot.id

    conversation.active_snapshot_id = new_snapshot.id
    await db_session.flush()

    return {
        "applied": True,
        "tokens_before": estimated_input,
        "tokens_after": snapshot_tokens,
        "compactor_model": compactor_model,
        "turns_compacted": len(compact_turns),
        "turns_preserved": len(preserved_turns),
        "snapshot_id": str(new_snapshot.id),
    }
```

### 3.4 Continuous compaction prompt

```python
def build_continuous_compaction_prompt() -> str:
    return """You are a conversation compactor. Your job is to create a structured snapshot of a long conversation history that preserves all critical technical context needed to continue the work.

Extract and organize the following from the conversation:

### State of the Problem
- What is the central problem or task being worked on?
- What is the current status?

### Technical Decisions Made
- Each decision with its justification (not just the outcome)
- Why was approach A chosen over approach B?
- What constraints influenced these decisions?

### Code Produced (key extracts only)
- Only the code that establishes the current state
- Don't include everything — only what's needed to continue
- Include file paths where relevant

### Current State
- Resolved: list of completed items
- Unresolved: list of pending items
- In Progress at compaction time: what was being worked on

### Technical Context
- Environment variables, architecture, dependencies
- Project conventions, coding standards mentioned
- Any non-obvious constraints or assumptions

### Pending Items
- What the user was going to do next
- Any explicit next steps mentioned

Format the output as Markdown. Be concise but complete. The goal is that someone reading this snapshot can continue the conversation without needing the original history."""
```

### 3.5 Context assembly after compaction

After continuous compaction, when building the prompt for the next turn:

```python
def assemble_context(conversation: Conversation, db_session) -> list[dict]:
    """
    Build the message array to send to the model.
    If a snapshot exists, use [snapshot] + [recent turns] instead of full history.
    """
    messages = []

    # 1. If active snapshot exists, include it as a system message
    if conversation.active_snapshot_id:
        snapshot = await db_session.get(ConversationSnapshot, conversation.active_snapshot_id)
        messages.append({
            "role": "system",
            "content": (
                f"[CONVERSATION SNAPSHOT — generated at turn {snapshot.turn_number_at_compaction} "
                f"by {snapshot.compactor_model}. Original history: {snapshot.tokens_before} tokens, "
                f"compacted to {snapshot.tokens_after} tokens.]\n\n{snapshot.snapshot_content}"
            )
        })
        # Load only turns AFTER the snapshot
        turns = await db_session.execute(
            select(ConversationTurn)
            .where(
                ConversationTurn.conversation_id == conversation.id,
                ConversationTurn.turn_number > snapshot.turn_number_at_compaction,
            )
            .order_by(ConversationTurn.turn_number)
        )
    else:
        # Load all turns
        turns = await db_session.execute(
            select(ConversationTurn)
            .where(ConversationTurn.conversation_id == conversation.id)
            .order_by(ConversationTurn.turn_number)
        )

    for turn in turns.scalars().all():
        messages.extend(turn.messages)

    return messages
```

### 3.6 What continuous compaction does NOT do

- Does NOT activate unless explicitly configured
- Does NOT modify original turns in DB
- Does NOT compact mid-turn or mid-stream
- Does NOT compact without preserving the most recent `compact_preserve_recent` tokens
- Does NOT generate snapshots without the user knowing (always reported in `proxy_metadata`)
- Does NOT chain compact across pseudo-model switches automatically

### 3.7 Interaction with OpenCode's auto-compact — the double-compaction problem

**The problem:**

OpenCode has `autoCompact: true` (default), triggering a generic summary at **95% of ContextWindow**. The proxy has continuous compaction at **70-80%**. These are independent systems that don't know about each other:

```
Turn 20: Proxy compacts turns 1-15 into snapshot. Model receives 48K.
         usage.prompt_tokens in response = 48K.
         OpenCode accumulates: session.PromptTokens += 48K.

Turn 21-30: Each turn adds ~5K prompt tokens.
         OpenCode's accumulated: 48K + (10 × 5K) = 98K. Not yet 95%.

Turn 50: OpenCode's session.PromptTokens hits 190K (95% of 200K).
         OpenCode triggers auto-compact.
         OpenCode summarizes: [proxy's snapshot + turns 16-50 + all responses].
         RESULT: double compaction — OpenCode summarized the proxy's already-compacted snapshot.
```

**Neither system knows the other compacted.** The proxy sees full message history each turn (OpenCode sends it). OpenCode accumulates `usage.prompt_tokens` from API responses. These are independent tracking systems.

### 3.7a Solution: external compaction detection + coordination

The proxy MUST detect when the client has compacted the history externally, and integrate that with the proxy's own compaction state. This makes both systems **complementary** rather than conflicting.

**Compaction layers (complementary, not conflicting):**

| Layer | Trigger | Preserves | When it fires |
|---|---|---|---|
| **Proxy continuous compaction** | 70-80% of context window | Structured snapshot: decisions, code, state, pending items | Early — manages context proactively |
| **OpenCode auto-compact** | 95% of context window (client-side) | Generic conversation summary | Late — last resort when proxy compaction wasn't enough or the conversation grew beyond proxy management |

**How they coordinate:**

1. **Proxy compacts first** (70-80%): reduces context the model sees. This means `usage.prompt_tokens` in responses is SMALLER. OpenCode's session accumulation SLOWS DOWN. The 95% trigger is delayed or never reached.

2. **If OpenCode STILL compacts** (conversation grew beyond even proxy management): the proxy detects it and incorporates OpenCode's summary into the proxy's snapshot chain. No information is lost — both compactions are tracked.

3. **The proxy NEVER re-compacts on top of an external compaction.** If the client compacted, the proxy treats the incoming message history as the new baseline and adjusts its internal token tracking accordingly.

### 3.7b External compaction detection

```python
def detect_external_compaction(
    incoming_messages: list[dict],
    conversation: Conversation,
    db_session,
) -> ExternalCompactionInfo | None:
    """
    Detect if the client (OpenCode or any other) has compacted the conversation.

    Detection signals:
    1. Message count suddenly DROPS significantly (>50% fewer messages than previous turn)
    2. First message is a system message containing summary-like content
    3. The conversation previously had many turns, now has very few messages

    Returns None if no external compaction detected.
    """
    previous_turn_count = await db_session.scalar(
        select(func.count(ConversationTurn.id))
        .where(ConversationTurn.conversation_id == conversation.id)
    )

    if previous_turn_count < 10:
        return None  # Too few turns for compaction to make sense

    incoming_msg_count = len(incoming_messages)

    # Signal 1: drastic reduction in message count
    expected_min_messages = previous_turn_count * 0.4  # At least 40% of previous
    if incoming_msg_count > expected_min_messages:
        return None  # Message count is normal — no compaction

    # Signal 2: first message looks like a summary
    first_msg = incoming_messages[0]
    is_system_or_user = first_msg.get("role") in ("system", "user")
    content = first_msg.get("content", "")
    is_long = len(str(content)) > 200  # Summaries are usually substantial text

    if not (is_system_or_user and is_long):
        return None

    # External compaction detected!
    return ExternalCompactionInfo(
        detected=True,
        incoming_message_count=incoming_msg_count,
        previous_turn_count=previous_turn_count,
        summary_preview=str(content)[:500],
    )
```

### 3.7c Handling external compaction

When external compaction is detected, the proxy:

```python
async def handle_external_compaction(
    incoming_messages: list[dict],
    conversation: Conversation,
    external_info: ExternalCompactionInfo,
    db_session,
):
    """
    When the client compacted externally (e.g., OpenCode's auto-compact):
    1. Store the client's summary as a compaction_snapshot in the proxy's DB
    2. Set it as the active snapshot
    3. Reset the proxy's token tracking to reflect the compacted state
    4. Continue processing normally — do NOT re-compact
    """
    # Store client's summary as a snapshot
    summary_content = str(incoming_messages[0].get("content", ""))
    estimated_tokens = len(summary_content) // 4  # Rough estimate

    new_snapshot = ConversationSnapshot(
        conversation_id=conversation.id,
        snapshot_type="external",           # Different from "continuous" or "explicit"
        tokens_before=conversation.total_tokens,
        tokens_after=estimated_tokens,
        compactor_model="client (external)", # Not a model we called
        snapshot_content=summary_content,
        turn_number_at_compaction=external_info.previous_turn_count,
    )
    db_session.add(new_snapshot)
    await db_session.flush()

    # Update conversation
    if conversation.active_snapshot_id:
        old = await db_session.get(ConversationSnapshot, conversation.active_snapshot_id)
        if old:
            old.superseded_by = new_snapshot.id

    conversation.active_snapshot_id = new_snapshot.id

    # Reset token tracking to reflect compacted state
    # After external compaction, the "active" context is just the summary + new messages
    new_total = estimated_tokens + sum(
        len(str(m.get("content", ""))) // 4 for m in incoming_messages[1:]
    )
    conversation.total_tokens = new_total

    await db_session.flush()

    # Report in proxy_metadata
    return {
        "external_compaction_detected": True,
        "source": "client",
        "tokens_before": external_info.previous_turn_count,  # turns before
        "tokens_after_snapshot": estimated_tokens,
        "proxy_compaction_skipped": True,  # We don't re-compact on top of client compaction
    }
```

### 3.7d Integration into chat endpoint

```python
# In chat endpoint, AFTER loading conversation, BEFORE capability detection:

# Check for external compaction (client compacted since last turn)
external_compaction = await detect_external_compaction(
    request.messages, conversation, db_session
)
if external_compaction:
    ext_meta = await handle_external_compaction(
        request.messages, conversation, external_compaction, db_session
    )
    proxy_metadata["external_compaction"] = ext_meta
    # Skip proxy's own continuous compaction check this turn
    skip_continuous_compaction = True
```

### 3.7e What this design achieves

| Scenario | What happens |
|---|---|
| Proxy compacts at 70% | Model receives [snapshot] + [recent]. OpenCode accumulates smaller prompt_tokens. 95% trigger delayed. |
| Conversation keeps growing despite proxy compaction | Proxy compacts again (chain snapshots). Eventually OpenCode's accumulated tokens hit 95%. |
| OpenCode compacts at 95% | Proxy detects external compaction, stores OpenCode's summary as a snapshot, resets token tracking. Does NOT re-compact. |
| User switches pseudo-models | Compatibility validated. Old snapshots preserved. New model gets full or compacted context. |
| OpenCode disabled auto-compact | Proxy is the sole compactor. No external compaction detection needed. |

**Key design principle:** The proxy is the **central coordinator** of all compaction. Client-side and server-side compaction are complementary layers — the proxy tracks both and ensures they never overlap destructively.

---

## 4. Integration into Chat Endpoint

### 4.1 Updated chat endpoint flow

```
After "Resolve conversation" (step 1):
  → [SAME AS SPRINT 2]

After "Determine physical model" (step 2):
  → [SAME AS SPRINT 2]

After "Check input threshold" (step 5):
  → IF threshold exceeded AND pre_compaction enabled:
       → pre_compact_input(messages, pm, config, conv_id)
       → Store compaction metadata for proxy_metadata
  → IF threshold exceeded AND pre_compaction disabled:
       → [SAME AS SPRINT 2: 400 INPUT_EXCEEDS_THRESHOLD]

After "Accumulate capabilities" (step 6):
  → IF continuous_compaction enabled:
       → Check if total_tokens > trigger_pct * context_window
       → IF triggered: continuous_compact(conversation, pm, config, db)
       → Assemble context using assemble_context()

After "Call LiteLLM" (step 7):
  → Use the potentially compacted messages and context
```

### 4.2 Updated proxy_metadata

```json
{
  "proxy_metadata": {
    "pre_compaction_applied": true,
    "pre_compaction": {
      "original_input_tokens": 80000,
      "compacted_input_tokens": 6000,
      "compactor_model": "glm-4.5-flash",
      "compactor_pseudo_model": "deep-flash",
      "savings_tokens": 74000
    },
    "continuous_compaction_applied": true,
    "continuous_compaction": {
      "tokens_before": 150000,
      "tokens_after_snapshot": 8120,
      "compactor_model": "glm-4.5-flash",
      "turns_compacted": 15,
      "turns_preserved": 5
    }
  }
}
```

---

## 5. Tests (Sprint 4)

### 5.1 test_pre_compaction.py (minimum 10 tests)

1. Input below threshold → no pre-compaction
2. Input above threshold with pre_compaction enabled → pre-compaction applied
3. Input above threshold with pre_compaction disabled → 400 error
4. Last user message replaced with summary
5. System messages and tool history NOT modified
6. Compaction metadata in response (tokens before/after)
7. Pre-compaction prompt includes all technical details
8. Compactor fails → original input used with warning
9. Compactor pseudo-model not found → error
10. Very large input (200K tokens) → compactor handles it

### 5.2 test_continuous_compaction.py (minimum 10 tests)

1. Context below trigger_pct → no compaction
2. Context above trigger_pct → continuous compaction triggered
3. Snapshot stored in `conversation_snapshots` table
4. `active_snapshot_id` updated on conversation
5. Recent turns preserved (not compacted)
6. Old turns compacted into snapshot
7. Context assembly uses snapshot + recent turns
8. Multiple compactions: second snapshot supersedes first
9. `superseded_by` chain works correctly
10. Continuous compaction NOT triggered when disabled in config

### 5.3 test_compaction_prompts.py (minimum 5 tests)

1. Pre-compaction prompt contains user content
2. Pre-compaction prompt specifies target tokens
3. Continuous compaction prompt asks for structured output
4. Continuous compaction prompt covers all required sections
5. Prompts are language-consistent (English)

---

## 6. Acceptance Criteria

- [x] 80K tokens input to `pensamiento-profundo-caro` → pre-compacted with `deep-flash` → expensive model receives ~8K tokens (unit tested)
- [x] `proxy_metadata.pre_compaction_applied: true` with before/after token counts
- [x] Pre-compaction only activates when configured
- [x] 30-turn conversation with `pensamiento-profundo-caro` triggers continuous compaction at 70% (unit tested)
- [x] Snapshot stored in DB with correct metadata
- [x] Subsequent turns use snapshot + recent turns, not full history
- [x] Multiple continuous compactions chain correctly
- [x] Original messages in DB NEVER modified by compaction
- [x] All 38 Sprint 4 tests pass (216 total)
- [x] No regression on Sprint 1-3 tests (all 178 original tests still pass)

### Known doc path deviation

The Sprint 4 spec originally placed files in `src/compactor/` but the project convention places all service modules in `src/service/`. Files were implemented in `src/service/compactor/` to maintain consistency:

| Spec path | Actual path |
|---|---|
| `src/compactor/pre_compactor.py` | `src/service/compactor/pre_compactor.py` |
| `src/compactor/continuous.py` | `src/service/compactor/continuous.py` |
| `src/compactor/prompts.py` | `src/service/compactor/prompts.py` |

---

## 7. Explicitly OUT OF SCOPE for Sprint 4

| Feature | Sprint |
|---|---|
| Explicit compaction (`POST /compact`) | 6 |
| Context alerts (60%, 80%, 100% warnings in proxy_metadata) | 6 |
| `CONTEXT_UNUSABLE` error (400 when history exceeds all models) | 6 |
| Celery-based async compaction for very large histories | 6 |
| Auto-describe images before compaction | 5 |
| Router LLM integration | 5 |
| Provider cache optimization | 7 |
| Audit log | 6 |
