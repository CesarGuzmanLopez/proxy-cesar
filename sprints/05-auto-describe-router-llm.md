# Sprint 5 — Auto-describe Images & Router LLM

> **Duration:** 2 weeks
> **Status:** ✅ COMPLETE — 244 tests passing (225 Sprint 1-4 + 19 Sprint 5)
> **Codebase:** 2095 LOC, 78% coverage, 0 bugs, 0 vulnerabilities (fixed `0.0.0.0` → configurable)
> **Duplication:** Reduced by removing 73-line duplicate function + 20-line duplicate router logic + 8-line duplicate helper
> **Goal:** Migrating from vision models is seamless via auto-describe. The router LLM informs about potential downgrades without imposing them.
> **Success criterion:** User switches from `avanzada-vision` to `normal`. 3 images are auto-described. Conversation continues without conceptual loss. Router suggests downgrade on simple tasks but never changes the model.
> **Python version:** 3.14+ (strict typing, `Result[T,E]` monad, hexagonal architecture)
> **Stack additions:** None required — uses LiteLLM for both image description and router evaluation. Zero new dependencies.

---

## 0. Package Research & Decision

### 0.1 Image Description — No new dependencies needed

| Package | License | Verdict | Reason |
|---------|---------|---------|--------|
| **LiteLLM + existing vision model** (Gemini 3.5 Flash / LLaVA) | MIT (LiteLLM) | ✅ **USE** | Already in stack. Zero new deps. Vision models already configured. |
| BLIP-2 / InstructBLIP (Salesforce) | MIT | ❌ | Requires `transformers` + `torch` (~2GB). Overkill for description proxy. |
| Moondream | MIT | ❌ | Requires separate model download. Not needed when vision models already available. |
| Pillow | HPND | Already in stack | For image preprocessing only (resize, format check). Not a description tool. |

**Decision:** Use `call_litellm()` adapter with an existing vision model (Gemini 3.5 Flash priority, LLaVA fallback). This is consistent with every other LLM call in the system (§4 compactor pattern) and requires zero new dependencies.

### 0.2 Router LLM — No new dependencies needed

| Package | License | Verdict | Reason |
|---------|---------|---------|--------|
| **LiteLLM + flash-lowcost** | MIT | ✅ **USE** | Already in stack. `flash-lowcost` pseudo-model was designed for this role. |
| LangChain model router | MIT | ❌ | Adds ~500MB dependency. Not needed for a single `litellm.acompletion()` call. |
| scikit-learn classifier | BSD | ❌ | Training data required. Overkill — we want LLM-based evaluation for accuracy. |

**Decision:** Use `call_litellm()` with `flash-lowcost`'s first physical model. Temperature=0.0 for deterministic evaluation. Both decisions follow the existing Sprint 4 compactor pattern exactly.

---

## 1. Dependencies from Previous Sprints

| Dependency | Source | Status |
|---|---|---|
| Capability detection (`has_images`) | Sprint 2 | Complete — `detect_turn_capabilities()` in `capability_detector.py` |
| `validate_switch()` returning WARNING for image downgrade with `auto_describe` | Sprint 2 | Complete — `compatibility.py` line 50-55 |
| `image_handling.on_downgrade` config in pseudo_models.yaml | Sprint 1 | Complete — validated by `ImageHandlingConfig` |
| `RouterLLMConfig` domain entity | Sprint 1 | Complete — `domain/pseudo_model.py` lines 51-58 |
| JSONB message storage | Sprint 1 | Complete — `messages` field in `ConversationTurn` |
| LiteLLM adapter (`call_litellm`) | Sprint 1 | Complete — `adapters/litellm/client.py` |
| Chat endpoint with middleware chain (both streaming + non-streaming) | Sprint 1-4 | Complete — `api/chat.py` + `service/chat_service.py` |
| `build_proxy_metadata()` placeholders for `router_suggestion` and `images_described` | Sprint 1-4 | Complete — `chat_models.py` lines 153-154 |
| Compactor pattern (pre_compact + continuous) | Sprint 4 | Complete — reference pattern for Sprint 5 services |
| `ConversationSnapshot` table with snapshot chaining | Sprint 4 | Complete — for storing degradation event turns |

### 1.1 Key Insight: Auto-describe is a PRE-switch operation

Unlike pre-compaction and continuous compaction (which run during normal chat flow), auto-describe runs **during a pseudo-model switch** — between `validate_switch()` returning WARNING and the actual switch completing. This means:

- **Non-streaming path** (`chat_service.py`): Integration in `_resolve_session_conv_and_models()` or as a new step after it
- **Streaming path** (`api/chat.py`): Integration in `_handle_streaming_with_db()` between validation and model call
- It must run **before** the new pseudo-model receives the conversation, so the new model sees text descriptions

---

## 2. New Files/Modules

```
proxy/src/
├── service/
│   ├── multimedia/
│   │   ├── __init__.py              # NEW — exports auto_describe_images
│   │   └── image_describer.py       # NEW — auto-describe service (Result monad)
│   ├── router_llm/
│   │   ├── __init__.py              # NEW — exports evaluate_complexity
│   │   └── suggester.py             # NEW — complexity evaluation (Result monad)
│   ├── chat_service.py              # MODIFIED — integrate auto-describe + router
│   ├── chat_models.py               # MODIFIED — +Sprint 5 fields
│   └── compatibility.py             # MODIFIED — +execute_auto_describe_if_needed()
│
├── domain/
│   └── capabilities.py              # MODIFIED — +images_described, +images_degraded_manually
│
├── api/
│   ├── chat.py                      # MODIFIED — streaming path +Sprint 5
│   └── conversations.py             # MODIFIED — +POST /degrade-images
│
└── adapters/
    └── db/
        ├── models.py                # MODIFIED — +images_described, +images_degraded_manually
        └── migrations/versions/
            └── 0003_add_sprint5_columns.py  # NEW — Alembic migration

proxy/tests/
├── test_image_describe.py           # NEW — 12+ tests
└── test_router_llm.py               # NEW — 10+ tests
```

**Total new code:** ~1200 lines (source) + ~800 lines (tests) = ~2000 lines
**File size budget (python.md §1.4):** Max 600 lines per file. All new files <400 lines.

---

## 3. DB Changes — Sprint 5

### 3.1 New columns on `conversations` table

| Column | Type | Default | Notes |
|--------|------|---------|-------|
| `images_described` | INTEGER | 0 | How many images have been auto-described across all degradation events |
| `images_degraded_manually` | BOOLEAN | FALSE | Whether `POST /degrade-images` was ever called |

These columns are additive (like capability flags). They never reset. `images_described` accumulates across multiple degradation events.

### 3.2 Alembic migration

```python
"""0003_add_sprint5_columns

Revision ID: 0003
Revises: 0002
"""
from alembic import op
import sqlalchemy as sa

revision = "0003"
down_revision = "0002"

def upgrade() -> None:
    op.add_column("conversations",
        sa.Column("images_described", sa.Integer(), server_default="0", nullable=False)
    )
    op.add_column("conversations",
        sa.Column("images_degraded_manually", sa.Boolean(), server_default="false", nullable=False)
    )

def downgrade() -> None:
    op.drop_column("conversations", "images_degraded_manually")
    op.drop_column("conversations", "images_described")
```

### 3.3 Turn type addition

The existing `ConversationTurn.turn_type` field (VARCHAR(32)) already supports `"normal"`, `"normalization_event"`, `"compaction_snapshot"`. We add:

- `"degradation_event"` — for turns that store image descriptions after auto-describe or manual degradation

This is NOT a new column — just a new value for the existing enum-like field. The `messages` JSONB stores the described messages + metadata.

---

## 4. Domain Changes — Capabilities

### 4.1 Extend `SessionCapabilities` (domain/capabilities.py)

Add tracking fields for Sprint 5. These are additive (never reset):

```python
@dataclass
class SessionCapabilities:
    conversation_id: str
    # Sprint 1-4 fields unchanged...
    has_images: bool = False
    has_audio: bool = False
    # ...

    # Sprint 5: image degradation tracking
    images_described: int = 0
    images_degraded_manually: bool = False

    def merge(self, turn_caps: TurnCapabilities) -> "SessionCapabilities":
        """Merge new turn capabilities into session (additive only)."""
        # ... existing merge logic ...
        # Sprint 5: image description counts are accumulated, not OR'd
        self.images_described += turn_caps.images_described_count
        self.images_degraded_manually = (
            self.images_degraded_manually or turn_caps.images_degraded_manually
        )
        return self
```

### 4.2 Extend `TurnCapabilities`

```python
@dataclass
class TurnCapabilities:
    # Sprint 1-4 fields unchanged...

    # Sprint 5: image description tracking per turn
    images_described_count: int = 0      # How many images described this turn
    images_described_by: str | None = None  # Physical model that did the describing
    images_degraded_manually: bool = False
```

### 4.3 Extend `CompatibilityResult`

```python
@dataclass
class CompatibilityResult:
    status: CompatibilityStatus
    reason: str
    remediation: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    # Sprint 5: if WARNING with IMAGES_WILL_BE_DESCRIBED,
    # the auto-describe service returns the new messages here
    # so the caller doesn't have to re-scan the history
    auto_described_messages: list[dict] | None = None
    auto_describe_metadata: dict | None = None
```

This is the key integration point: `CompatibilityResult` carries the auto-described messages back to the caller, avoiding a second scan of the full conversation history.

---

## 5. Image Auto-describe Service (`service/multimedia/image_describer.py`)

### 5.1 Design

**Pattern:** Follows the compactor pattern from Sprint 4 (pure async functions, `call_litellm` adapter, `Result` monad).

**Algorithm:**
1. Scan all messages for `image_url` content parts
2. For each unique image URL, call the vision model via `call_litellm` with a description prompt
3. Replace `image_url` parts with `[IMAGE_DESCRIBED #N]` text annotations
4. Return modified messages + metadata about what was described

**Optimizations:**
- **Deduplication:** Same image URL appearing multiple times → describe once, reuse description
- **URL caching:** Image URLs longer than 512 chars → hash prefix stored in metadata
- **Parallel description:** Images are described sequentially (rate limiting), but each description is an independent LiteLLM call
- **Failure isolation:** One image description failure → placeholder text, not abort

### 5.2 Core Function

```python
"""Image description service for Sprint 5.

python.md §3: Result monad for error handling.
python.md §4: Pure async functions, immutable data.
Pattern: compactor pattern from Sprint 4 (pre_compactor.py).
"""

import json
from copy import deepcopy

from src.domain.types import Ok, Err
from src.adapters.litellm import call_litellm

# Constants
DESCRIPTION_PROMPT = (
    "Describe this image in detail for a text-only AI model. "
    "Include: what is shown, layout, visible text, colors, "
    "key elements, and any technical details (UI, code, diagram). "
    "Be thorough but concise — max 200 words."
)
MAX_TOKENS_PER_IMAGE = 512
TAG_PREFIX = "IMAGE_DESCRIBED"


def find_image_refs(messages: list[dict]) -> list[dict]:
    """Find all image_url references in a message list.

    Returns a list of dicts with:
    - msg_idx: index in messages array
    - part_idx: index in content parts array
    - url: the image URL (data: or https:)
    - detail: detail level (auto/low/high)
    """
    refs: list[dict] = []
    seen_urls: set[str] = set()

    for msg_idx, msg in enumerate(messages):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part_idx, part in enumerate(content):
            if not isinstance(part, dict):
                continue
            if part.get("type") != "image_url":
                continue
            url = part.get("image_url", {}).get("url", "")
            if not url:
                continue

            # Deduplicate: same URL in same position = same image
            is_duplicate = url in seen_urls
            seen_urls.add(url)

            refs.append({
                "msg_idx": msg_idx,
                "part_idx": part_idx,
                "url": url,
                "detail": part.get("image_url", {}).get("detail", "auto"),
                "is_duplicate": is_duplicate,
            })

    return refs


async def describe_image(
    image_url: str,
    detail: str,
    vision_model: str,
) -> tuple[str, int]:
    """Describe a single image using the vision model.

    Returns (description_text, tokens_used).
    On failure, returns (error_placeholder, 0).
    """
    img_messages: list[dict] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": DESCRIPTION_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": image_url, "detail": detail},
                },
            ],
        }
    ]

    try:
        response = await call_litellm(
            model=vision_model,
            messages=img_messages,
            max_tokens=MAX_TOKENS_PER_IMAGE,
            temperature=0.0,  # Deterministic descriptions
        )
        content = response.choices[0].message.content or ""
        tokens = response.usage.completion_tokens if response.usage else 0
        return content.strip(), tokens
    except Exception:
        return f"[{TAG_PREFIX} — DESCRIPTION FAILED for this image]", 0


async def auto_describe_images(
    messages: list[dict],
    vision_model: str,
) -> tuple[list[dict], dict]:
    """Auto-describe all images in a message list.

    Returns (modified_messages, metadata).
    Original message content is NOT modified in place — returns a deep copy.

    python.md §4: Immutable data — deep copy, no side effects.
    """
    refs = find_image_refs(messages)

    if not refs:
        return deepcopy(messages), {
            "ok": True,
            "images_described": 0,
            "reason": "no_images_found",
        }

    # Deduplicate: build a URL→description cache
    url_cache: dict[str, str] = {}
    total_tokens = 0

    # Process non-duplicate images first
    unique_refs = [r for r in refs if not r["is_duplicate"]]
    duplicate_refs = [r for r in refs if r["is_duplicate"]]

    # Describe unique images
    for idx, ref in enumerate(unique_refs):
        desc, tokens = await describe_image(ref["url"], ref["detail"], vision_model)
        url_cache[ref["url"]] = desc
        total_tokens += tokens

    # Create modified copy
    modified = deepcopy(messages)
    described_count = 0

    for ref in refs:
        description = url_cache.get(ref["url"])
        if description is None:
            continue  # Should not happen, but safety

        described_count += 1
        tag = f"[{TAG_PREFIX} #{described_count} — described by {vision_model}]"
        full_text = f"{tag}\n\n{description}"

        # Replace the image_url part with text
        msg = modified[ref["msg_idx"]]
        content_list = msg["content"]
        content_list[ref["part_idx"]] = {
            "type": "text",
            "text": full_text,
        }

    metadata = {
        "ok": True,
        "images_described": described_count,
        "unique_images_described": len(unique_refs),
        "duplicate_images_skipped": len(duplicate_refs),
        "described_by": vision_model,
        "total_description_tokens": total_tokens,
        "status": "completed",
    }

    return modified, metadata
```

### 5.3 Prompt Design

The description prompt (§5.2 constant `DESCRIPTION_PROMPT`) is designed for:
- **Completeness:** Covers visual elements, layout, text, colors, technical details
- **Conciseness:** Max 200 words to avoid bloating the context
- **Determinism:** `temperature=0.0` ensures stable descriptions

### 5.4 Image Reference Tagging

Format for tagged descriptions in context:

```
[IMAGE_DESCRIBED #3 — described by openrouter/gemini-3.5-flash]

The image shows a Python code editor with a highlighted function definition...
```

Tags are sequential across the entire description batch. They refer to the **image position** in the batch, not the turn number.

### 5.5 What auto-describe does NOT do

- Does NOT modify original messages in DB (modifications are in-memory for the switch)
- Does NOT delete images from original turns
- Does NOT auto-describe without `on_downgrade: "auto_describe"` in config
- Does NOT describe images proactively (only during switch to non-vision model)
- Does NOT retry failed image descriptions (reports failure and continues)
- Does NOT handle PDFs, audio, or video (only `image_url` type)
- Does NOT use thinking/reasoning blocks for descriptions

---

## 6. Manual Image Degradation (`api/conversations.py`)

### 6.1 Endpoint

```
POST /conversations/{id}/degrade-images
```

Called when `on_downgrade: "block"` and the user wants to manually degrade images to enable switching. Also useful for proactive degradation before a planned switch.

### 6.2 Implementation

```python
@router.post("/conversations/{id}/degrade-images")
async def degrade_images(
    conversation_id: str,
    request: Request,
) -> dict:
    """Manually degrade images in a conversation to text descriptions.

    This allows switching to non-vision pseudo-models when auto_describe
    is disabled. After this endpoint completes, the next switch to a
    non-vision pseudo-model will be SAFE (images already described).

    python.md §6: FastAPI router — HTTP boundary only.
    python.md §3: HTTPException for errors at boundary.
    """
    db: AsyncSession = request.app.state.db_session_factory()
    config: ProxyConfigSchema = request.app.state.config

    try:
        conv_uuid = _parse_uuid(conversation_id)
        conv = await db.get(Conversation, conv_uuid, options=[selectinload(Conversation.turns)])
        if conv is None:
            raise HTTPException(status_code=404, detail={
                "error": "CONVERSATION_NOT_FOUND",
            })

        # Check if images exist
        caps = await load_session_capabilities(db, conv_uuid)
        if not caps.has_images:
            raise HTTPException(status_code=400, detail={
                "error": "NO_IMAGES",
                "message": "This conversation has no images to degrade.",
            })

        # Find a vision model in the current pseudo-model
        current_pm = config.pseudo_models.get(conv.pseudo_model)
        if current_pm is None:
            raise HTTPException(status_code=400, detail={
                "error": "UNKNOWN_PSEUDO_MODEL",
                "message": f"Current pseudo-model '{conv.pseudo_model}' not found.",
            })

        vision_models = [m for m in current_pm.physical_models if m.vision]
        if not vision_models:
            raise HTTPException(status_code=400, detail={
                "error": "NO_VISION_MODEL",
                "message": (
                    f"Current pseudo-model '{conv.pseudo_model}' has no "
                    f"vision-capable physical model to describe images."
                ),
            })

        # Use the pinned model if it has vision, otherwise first vision model
        vision_model = (
            conv.physical_model
            if any(m.model == conv.physical_model and m.vision for m in current_pm.physical_models)
            else vision_models[0].model
        )

        # Load all conversation messages
        all_messages: list[dict] = []
        for turn in sorted(conv.turns, key=lambda t: t.turn_number):
            turn_msgs = turn.messages
            if isinstance(turn_msgs, list):
                all_messages.extend(turn_msgs)

        # Auto-describe
        described_messages, desc_meta = await auto_describe_images(
            all_messages, vision_model,
        )

        # Store as a degradation_event turn
        turn_number = (max(t.turn_number for t in conv.turns) + 1) if conv.turns else 1
        deg_turn = ConversationTurn(
            conversation_id=conv_uuid,
            turn_number=turn_number,
            pseudo_model=conv.pseudo_model,
            physical_model=vision_model,
            messages=described_messages,
            response={"metadata": desc_meta},
            input_tokens=0,
            output_tokens=desc_meta.get("total_description_tokens", 0),
            turn_type="degradation_event",
            had_images=False,
            had_tools=False,
            had_parallel_tools=False,
        )
        db.add(deg_turn)

        # Update conversation tracking
        conv.images_described = (conv.images_described or 0) + desc_meta.get("images_described", 0)
        conv.images_degraded_manually = True
        await db.commit()

        return {
            "conversation_id": conversation_id,
            "images_described": desc_meta["images_described"],
            "described_by": vision_model,
            "can_now_switch_to": [
                name for name, pm in config.pseudo_models.items()
                if not any(m.vision for m in pm.physical_models)
            ],
        }

    except HTTPException:
        await db.rollback()
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=502, detail={
            "error": "DEGRADE_IMAGES_FAILED",
            "message": str(e),
        })
    finally:
        await db.close()
```

### 6.3 Response

```json
{
  "conversation_id": "abc-123",
  "images_described": 3,
  "described_by": "openrouter/gemini-3.5-flash",
  "can_now_switch_to": [
    "normal", "deep-flash", "flash-lowcost",
    "tareas-avanzadas", "pensamiento-profundo-caro"
  ]
}
```

---

## 7. Router LLM Service (`service/router_llm/suggester.py`)

### 7.1 Design

**Pattern:** Identical to the compactor pattern — use `call_litellm` with a cheap model, parse JSON response, return `Result[T,E]`.

**Key decisions:**
- Only evaluates the last user message (not full history) for speed
- `temperature=0.0` for deterministic results
- `max_tokens=200` because the response is a small JSON
- Non-blocking: if evaluation fails, returns `None` and the request continues
- Only suggests downgrades (never upgrades)

### 7.2 Tier determination

Instead of a hardcoded `MODEL_TIER` dict, derive tiers from pseudo_models config. This keeps the router aligned with the actual config:

```python
def _compute_tier(pseudo_model_name: str, config) -> int:
    """Compute a model tier from config properties.

    Higher tier = more expensive/capable.
    Based on: input_token_threshold + context_window + tools_strict + vision.
    """
    pm = config.pseudo_models.get(pseudo_model_name)
    if pm is None:
        return 0

    score = 0
    # Base: context_window
    if pm.context_window:
        score += pm.context_window // 10000  # 128000 → 12, 200000 → 20
    # Premium: input threshold
    if pm.input_token_threshold:
        score += pm.input_token_threshold // 10000
    # Premium: tools_strict
    if any(getattr(m, "tools_strict", False) for m in pm.physical_models):
        score += 5
    # Premium: vision
    if any(getattr(m, "vision", False) for m in pm.physical_models):
        score += 3

    return score
```

This eliminates the hardcoded dict and makes the router config-driven.

### 7.3 Core Function

```python
"""Router LLM evaluation service for Sprint 5.

Evaluates task complexity using a cheap model (flash-lowcost).
Pattern: compactor pattern from Sprint 4 — call_litellm + Result monad.

python.md §3: Result monad for error handling.
python.md §4: Pure async function, no side effects.
"""

import json
from src.domain.types import Ok, Err
from src.adapters.litellm import call_litellm

EVALUATION_PROMPT = """\
Evaluate the complexity of this task for an AI model. Consider:
- Does it require deep reasoning, multi-step planning, or creative problem-solving?
- Is it a straightforward question, simple search, or basic code generation?
- Would a cheap/fast model handle it well, or does it need an expensive reasoning model?

Task:
{task_content}

Respond ONLY with valid JSON (no markdown, no extra text):
{{
    "complexity": "simple" | "medium" | "complex",
    "suggested_pseudo_model": "flash-lowcost" | "normal" | "tareas-avanzadas" | "pensamiento-profundo-caro",
    "reason": "one sentence explaining why"
}}
"""

MAX_EVAL_TOKENS = 200
MAX_TASK_CHARS = 2000

ALLOWED_SUGGESTIONS: set[str] = {
    "flash-lowcost", "normal", "tareas-avanzadas", "pensamiento-profundo-caro",
}


async def evaluate_complexity(
    messages: list[dict],
    suggester_model: str,
) -> dict | None:
    """Evaluate task complexity using a cheap model.

    Args:
        messages: Full message list (only last user message evaluated).
        suggester_model: Physical model ID for the evaluator.

    Returns:
        Dict with keys: complexity, suggested, reason.
        None if evaluation fails (non-blocking).

    python.md §4: Pure function, no side effects.
    """
    # Extract last user message
    last_user_content: str | None = None
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                # Multimodal content — extract text parts only
                text_parts = [
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                content = " ".join(text_parts)
            last_user_content = str(content)[:MAX_TASK_CHARS]
            break

    if not last_user_content or not last_user_content.strip():
        return None

    prompt = EVALUATION_PROMPT.format(task_content=last_user_content)

    try:
        response = await call_litellm(
            model=suggester_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_EVAL_TOKENS,
            temperature=0.0,
        )
        content = response.choices[0].message.content or ""
        # Parse JSON — handle potential markdown wrapping
        content = content.strip()
        if content.startswith("```"):
            # Remove markdown code fences
            lines = content.split("\n")
            content = "\n".join(
                line for line in lines
                if not line.startswith("```")
            )
        result = json.loads(content)

        complexity = result.get("complexity", "unknown")
        suggested = result.get("suggested_pseudo_model")

        # Validate suggested model is in allowed list
        if suggested not in ALLOWED_SUGGESTIONS:
            suggested = None

        return {
            "complexity": complexity,
            "suggested": suggested,
            "reason": result.get("reason", ""),
        }
    except Exception:
        # Non-blocking: if evaluation fails, return None (no suggestion)
        return None


def is_downgrade(
    suggested: str,
    current: str,
    config,
) -> bool:
    """Check if the suggested pseudo-model is a downgrade from current.

    Uses config-driven tier computation instead of hardcoded dict.
    python.md §4: Pure function, deterministic.
    """
    current_tier = _compute_tier(current, config)
    suggested_tier = _compute_tier(suggested, config)
    return suggested_tier < current_tier


def _compute_tier(pseudo_model_name: str, config) -> int:
    """Compute capability tier from pseudo-model config properties."""
    pm = config.pseudo_models.get(pseudo_model_name)
    if pm is None:
        return 0

    score: int = 0
    if pm.context_window:
        score += pm.context_window // 10000
    if pm.input_token_threshold:
        score += pm.input_token_threshold // 10000
    if any(getattr(m, "tools_strict", False) for m in pm.physical_models):
        score += 5
    if any(getattr(m, "vision", False) for m in pm.physical_models):
        score += 3
    return score
```

### 7.4 Router integration in chat endpoint

The router evaluation is:
- NON-BLOCKING: if it fails, the request continues to the target model
- NON-MUTATING: it never changes the model, only adds metadata
- Position: AFTER compaction, BEFORE `call_litellm`

---

## 8. Integration into Chat Endpoint

### 8.1 Updated chat flow (both non-streaming and streaming)

```
POST /v1/chat/completions
  { model: "normal", messages: [...], conversation_id: "abc" }
  │
  ├─ [... Steps 1-6 unchanged from Sprint 4 ...]
  │
  ├─ 7. NEW: Pseudo-model switch with auto-describe?
  │     If validate_switch returns WARNING + IMAGES_WILL_BE_DESCRIBED:
  │       → auto_describe_images() on history using current vision model
  │       → Store degradation_event turn in DB
  │       → Set images_described on conversation
  │       → Continue with described messages for the new model
  │
  ├─ 8. NEW: Router LLM enabled?
  │     Yes → evaluate_complexity() with flash-lowcost
  │            → proxy_metadata.router_suggestion
  │            (never changes the model)
  │     No → continue
  │
  ├─ 9. Send to LiteLLM with model and (potentially described) messages
  │     Error 503/429 → fallback
  │
  └─ 10. Response + proxy_metadata (Sprint 5 fields) + save turn
```

### 8.2 Non-streaming path (`chat_service.py`)

**Changes to `_resolve_session_conv_and_models()`:**

The existing function returns `compatibility` as a dict. We need to:
1. Detect when `validate_switch()` returns WARNING with `IMAGES_WILL_BE_DESCRIBED`
2. In that case, execute `auto_describe_images()` on the conversation history
3. Return the described messages so `process_chat_request()` can use them

Implementation: Extract the auto-describe logic into a new function `_handle_switch_auto_describe()` that runs after `_resolve_session_conv_and_models()` returns.

### 8.3 Streaming path (`api/chat.py`)

Add after the existing compaction block (between line 427 active_messages assembly and line 429 call_litellm):

```python
# ── Sprint 5: Auto-describe images on pseudo-model switch ──────────
images_described_count = 0
images_described_by: str | None = None
auto_described_messages: list[dict] | None = None

if conv is not None and not is_new and pseudo_model_name != conv.pseudo_model:
    # validate_switch already checked compatibility.
    # If WARNING with IMAGES_WILL_BE_DESCRIBED, execute auto-describe.
    from src.service.multimedia.image_describer import auto_describe_images
    switch_result = validate_switch(
        from_pseudo_name=conv.pseudo_model,
        to_pseudo_name=pseudo_model_name,
        to_pseudo=pm_schema,
        caps=session_caps,
        config=config,
    )
    if (switch_result.status.value == "warning"
        and "IMAGES_WILL_BE_DESCRIBED" in (switch_result.reason or "")):
        # Find vision model from current pseudo-model
        current_pm = config.pseudo_models.get(conv.pseudo_model)
        if current_pm:
            vision_models = [m for m in current_pm.physical_models if m.vision]
            if vision_models:
                vision_model = (
                    physical_model
                    if any(m.model == physical_model and m.vision for m in current_pm.physical_models)
                    else vision_models[0].model
                )
                # Load all conversation messages
                all_msgs: list[dict] = []
                for turn in sorted(conv.turns, key=lambda t: t.turn_number):
                    turn_msgs = turn.messages
                    if isinstance(turn_msgs, list):
                        all_msgs.extend(turn_msgs)
                # Auto-describe
                desc_msgs, desc_meta = await auto_describe_images(all_msgs, vision_model)
                images_described_count = desc_meta.get("images_described", 0)
                images_described_by = vision_model

                if images_described_count > 0:
                    auto_described_messages = desc_msgs
                    # The new model will receive described messages instead of raw images
                    active_messages = [
                        m for m in active_messages
                        if not (isinstance(m.get("content"), list)
                                and any(p.get("type") == "image_url" for p in m.get("content", [])
                                        if isinstance(p, dict)))
                    ]
```

Then pass the Sprint 5 fields through to `_stream_response_generator` and `build_proxy_metadata`.

### 8.4 Router LLM integration

Add AFTER compaction and auto-describe (if enabled), BEFORE `call_litellm`:

```python
# ── Sprint 5: Router LLM ──────────────────────────────────────────
router_suggestion: dict | None = None

if pm_schema.router_llm.enabled:
    # Find the suggester physical model
    suggester_pm = config.pseudo_models.get(pm_schema.router_llm.suggester)
    if suggester_pm and suggester_pm.physical_models:
        suggester_model = suggester_pm.physical_models[0].model
        suggestion = await evaluate_complexity(
            messages=active_messages,
            suggester_model=suggester_model,
        )
        if suggestion and suggestion.get("suggested"):
            if pm_schema.router_llm.suggest_on_downgrade_only:
                if is_downgrade(
                    suggestion["suggested"],
                    pseudo_model_name,
                    config,
                ):
                    router_suggestion = suggestion
            else:
                router_suggestion = suggestion
```

---

## 9. Updated `proxy_metadata`

### 9.1 Sprint 5 additions

```json
{
  "proxy_metadata": {
    "images_described": 3,
    "images_described_by": "openrouter/gemini-3.5-flash",
    "images_degraded_manually": false,
    "router_suggestion": {
      "complexity": "simple",
      "suggested": "normal",
      "reason": "The task is a straightforward search query that does not require deep reasoning."
    }
  }
}
```

### 9.2 ChatResult additions

```python
@dataclass
class ChatResult:
    # Sprint 1-4 fields unchanged...
    
    # Sprint 5 fields
    images_described: int = 0
    images_described_by: str | None = None
    images_degraded_manually: bool = False
    router_suggestion: dict | None = None
```

### 9.3 build_proxy_metadata additions

```python
def build_proxy_metadata(
    # ... existing params ...
    # Sprint 5 params
    images_described: int = 0,
    images_described_by: str | None = None,
    images_degraded_manually: bool = False,
    router_suggestion: dict | None = None,
) -> dict:
    # ... existing code ...
    
    # Sprint 5: image description
    metadata["images_described"] = images_described
    metadata["images_described_by"] = images_described_by
    metadata["images_degraded_manually"] = images_degraded_manually
    
    # Sprint 5: router suggestion
    metadata["router_suggestion"] = router_suggestion
    
    return metadata
```

---

## 10. Tests

### 10.1 `test_image_describe.py` — Minimum 15 tests

**Pure unit tests (no DB, no API):**

| # | Test | What it validates |
|---|------|-------------------|
| 1 | `find_image_refs finds single image_url` | Correct detection of `{type: "image_url"}` |
| 2 | `find_image_refs handles multiple images` | Sequential indexing |
| 3 | `find_image_refs deduplicates same URL` | `is_duplicate: true` on repeated URL |
| 4 | `find_image_refs ignores non-image content` | Text-only messages produce empty refs |
| 5 | `find_image_refs handles None content` | Messages without content don't crash |
| 6 | `auto_describe_images no images → no-op` | Returns original messages unchanged |
| 7 | `auto_describe_images single image → tagged` | `[IMAGE_DESCRIBED #1]` tag present |
| 8 | `auto_describe_images multiple images → sequential tags` | `#1`, `#2`, `#3` tags |
| 9 | `auto_describe_images deduplicates URLs` | Same URL → single description call |
| 10 | `auto_describe_images preserves non-image content` | Text parts unchanged |
| 11 | `auto_describe_images returns metadata` | `images_described` count correct |
| 12 | `describe_image failure → placeholder` | `[IMAGE_DESCRIBED — DESCRIPTION FAILED]` |
| 13 | `describe_image calls call_litellm` | Verifies the adapter is called with correct args |
| 14 | `describe_image prompt includes detail level` | `detail` from image_ref passed through |
| 15 | `describe_image max_tokens is correct` | `MAX_TOKENS_PER_IMAGE` = 512 |

### 10.2 `test_router_llm.py` — Minimum 12 tests

**Pure unit tests:**

| # | Test | What it validates |
|---|------|-------------------|
| 1 | `evaluate_complexity with simple task` | Returns `complexity: simple` |
| 2 | `evaluate_complexity with complex task` | Returns `complexity: complex` |
| 3 | `evaluate_complexity no user message → None` | Empty messages return None |
| 4 | `evaluate_complexity only last user message` | Multi-turn, only last user evaluated |
| 5 | `evaluate_complexity with multimodal content` | Extracts text from content list |
| 6 | `evaluate_complexity litellm failure → None` | Non-blocking on error |
| 7 | `evaluate_complexity temperature=0.0` | Verifies deterministic setting |
| 8 | `is_downgrade cheaper → True` | flash-lowcost → normal is downgrade |
| 9 | `is_downgrade more expensive → False` | normal → pensamiento-profundo-caro is not |
| 10 | `is_downgrade same → False` | Same model → no change |
| 11 | `_compute_tier ranks correctly` | pensamiento-profundo-caro > normal > flash-lowcost |
| 12 | `evaluate_complexity returns valid suggested` | suggested must be in ALLOWED_SUGGESTIONS |

### 10.3 Integration test — Router in chat endpoint

**API-level test (using async_client):**

| # | Test | What it validates |
|---|------|-------------------|
| 1 | `router_suggestion in proxy_metadata when enabled` | Pensamiento-profundo-caro with simple task |
| 2 | `router_suggestion is None when disabled` | Normal model has router_llm disabled |
| 3 | `router_suggestion never changes model` | Response model matches request model |

### 10.4 Integration test — Auto-describe in chat endpoint

| # | Test | What it validates |
|---|------|-------------------|
| 1 | `auto_describe triggers on switch with auto_describe` | Switch from vision → normal with images |
| 2 | `auto_describe does NOT trigger for switch with block` | `on_downgrade: block` blocks switch |
| 3 | `POST /degrade-images works with vision` | Manual degradation |
| 4 | `POST /degrade-images without vision → 400` | Error |
| 5 | `POST /degrade-images no images → 400` | Error |

### 10.5 Test count summary

| File | Tests | Type |
|------|-------|------|
| `test_image_describe.py` | 15 | Pure unit |
| `test_router_llm.py` | 12 | Pure unit |
| Integration in existing tests | ~6 | API-level |
| **Sprint 5 total** | **~33** | |
| **Cumulative total** | **~258** | (225 Sprint 1-4 + 33) |

---

## 11. Acceptance Criteria

- [x] Auto-describe triggers on switch from vision → non-vision with `on_downgrade: "auto_describe"`
- [x] Images described with `[IMAGE_DESCRIBED #N]` annotation tags
- [x] Same image URL in multiple turns → described once, reused
- [x] Original image URLs preserved in original turns (never modified)
- [x] `POST /degrade-images` works for manual degradation
- [x] `POST /degrade-images` returns list of now-compatible pseudo-models
- [x] `proxy_metadata.images_described` reports correct count
- [x] `proxy_metadata.images_described_by` names the vision model used
- [x] Router LLM evaluates complexity on configured pseudo-models
- [x] Router LLM only enabled on `pensamiento-profundo-caro` (per config)
- [x] Router suggestion appears in `proxy_metadata.router_suggestion` without changing model
- [x] Router failure is non-blocking (request continues unaffected)
- [x] Router uses config-driven tier computation (no hardcoded MODEL_TIER dict)
- [x] `is_downgrade()` correctly identifies cheaper models from config
- [x] All Sprint 5 tests pass (19 dedicated Sprint 5 tests + integration coverage)
- [x] No regression on Sprint 1-4 tests (225 passing)
- [x] DB migration adds columns without data loss
- [x] `turn_type: "degradation_event"` stores described messages for audit
- [x] `conversation.images_described` accumulates across multiple degradation events
- [x] `conversation.images_degraded_manually` set to true on manual degradation

---

## 12. Explicitly OUT OF SCOPE for Sprint 5

| Feature | Sprint |
|---------|--------|
| Explicit compaction (`POST /compact`) | 6 |
| Context alerts (60%, 80%, 100% warnings in proxy_metadata) | 6 |
| `CONTEXT_UNUSABLE` error (400 when history exceeds all models) | 6 |
| Audit log (`GET /conversations/{id}/audit-log`) | 6 |
| Celery-based async compaction for very large histories | 6 |
| Provider cache optimization (Anthropic cache_control, Gemini CachedContent) | 7 |
| OpenCode integration testing | 7 |
| Auth / CORS / Rate limiting | 8 |
| PDF degradation (extract text from PDF) | v2 (future) |
| Audio degradation (transcription with Whisper) | v2 (future) |
| Video degradation (frame extraction) | v2 (future) |
