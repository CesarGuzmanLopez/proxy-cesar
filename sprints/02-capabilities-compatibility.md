# Sprint 2 — Capabilities Detection & Compatibility Validation

> **Duration:** 2 weeks  
> **Status:** ✅ COMPLETE — 178 tests passing (128 Sprint 1+2 + 50 Sprint 3)
> **Goal:** The proxy detects multimedia and tools in every message, accumulates capability flags per conversation, and blocks incompatible pseudo-model switches with descriptive errors and remediation options.
> **Success criterion:** Switching from `avanzada-vision` to `normal` with images in history → HTTP 409 with remediation options. `GET /compatible-models` is deterministic. All models properly filtered by tool compatibility.
> **Completed:** 2026-05-23

---

## 1. Dependencies from Previous Sprints

| Dependency | Source | Status |
|---|---|---|
| FastAPI app with `/v1/chat/completions` | Sprint 1 | Must be complete and passing tests |
| `pseudo_models.yaml` with all 8 pseudo-models | Sprint 1 | Already validated at startup |
| `Conversation` and `ConversationTurn` DB tables | Sprint 1 | Already created via Alembic |
| Valkey affinity (get/set/delete) | Sprint 1 | Already working |
| LiteLLM integration with all providers | Sprint 1 | Already working |
| Request schema (`ChatRequest`) | Sprint 1 | To be extended with capability detection |
| `proxy_metadata` in responses | Sprint 1 | To be extended with new fields |

### 1.1 New files/modules to create in Sprint 2

```
src/
├── middleware/
│   ├── __init__.py
│   ├── capability_detector.py    # NEW — detect images, tools, parallel tools, audio, PDF, video
│   ├── compatibility.py          # NEW — validate_switch() function
│   ├── threshold_guard.py        # NEW — check input threshold, context threshold
│   └── tool_filter.py            # NEW — filter pool by openai_tools_compatible & parallel_tools
│
├── models/
│   └── capabilities.py           # NEW — SessionCapabilities, CompatibilityResult enums
│
├── api/
│   └── conversations.py          # NEW — conversation endpoints (see §7)
│
└── tests/
    ├── test_capability_detector.py   # NEW
    ├── test_compatibility.py         # NEW — ≥20 matrix combinations
    ├── test_tool_filter.py           # NEW
    ├── test_threshold_guard.py       # NEW
    └── test_conversations_api.py     # NEW
```

### 1.2 DB schema changes (Alembic migration)

**Add to `conversations` table:**

| Column | Type | Default | Notes |
|---|---|---|---|
| `capability_has_images` | BOOLEAN | FALSE | Set to TRUE when first image is detected. Never reset. |
| `capability_has_audio` | BOOLEAN | FALSE | Set to TRUE when first audio is detected. Never reset. |
| `capability_has_pdf` | BOOLEAN | FALSE | Set to TRUE when first PDF is detected. Never reset. |
| `capability_has_video` | BOOLEAN | FALSE | Set to TRUE when first video is detected. Never reset. |
| `capability_has_tools` | BOOLEAN | FALSE | Set to TRUE when first tool call/definition appears. Never reset. |
| `capability_has_parallel_tools` | BOOLEAN | FALSE | Set to TRUE when >1 tool call in a single assistant turn. Never reset. |

**Add to `conversation_turns` table:**

| Column | Type | Default | Notes |
|---|---|---|---|
| `turn_type` | VARCHAR(32) | `'normal'` | `normal`, `compaction_snapshot`, `degradation_event` (Sprint 5), `normalization_event` (Sprint 3) |
| `had_images` | BOOLEAN | FALSE | This turn had images |
| `had_tools` | BOOLEAN | FALSE | This turn had tool definitions or tool calls |
| `had_parallel_tools` | BOOLEAN | FALSE | This turn had >1 tool call |

---

## 2. Capability Detection (`middleware/capability_detector.py`)

### 2.1 What to detect and how

The detector processes the `messages` array from the incoming request. It scans ALL messages, not just the last one — because a turn might reference images from previous turns, and tool history accumulates.

```python
from dataclasses import dataclass, field

@dataclass
class TurnCapabilities:
    """Capabilities detected in a single turn's messages."""
    has_images: bool = False
    has_audio: bool = False
    has_pdf: bool = False
    has_video: bool = False
    has_tools: bool = False               # Tool definitions present in request OR tool calls in any message
    has_parallel_tools: bool = False       # >1 tool_call in a single assistant message

@dataclass
class SessionCapabilities:
    """Accumulated capabilities across all turns of a conversation."""
    conversation_id: str
    has_images: bool = False
    has_audio: bool = False
    has_pdf: bool = False
    has_video: bool = False
    has_tools: bool = False
    has_parallel_tools: bool = False
    total_tokens: int = 0

    def merge(self, turn_caps: TurnCapabilities) -> "SessionCapabilities":
        """Merge new turn capabilities into session (additive only)."""
        self.has_images = self.has_images or turn_caps.has_images
        self.has_audio = self.has_audio or turn_caps.has_audio
        self.has_pdf = self.has_pdf or turn_caps.has_pdf
        self.has_video = self.has_video or turn_caps.has_video
        self.has_tools = self.has_tools or turn_caps.has_tools
        self.has_parallel_tools = self.has_parallel_tools or turn_caps.has_parallel_tools
        return self

    @classmethod
    def from_db_row(cls, row) -> "SessionCapabilities":
        """Load from conversations table columns."""
        return cls(
            conversation_id=str(row.id),
            has_images=row.capability_has_images,
            has_audio=row.capability_has_audio,
            has_pdf=row.capability_has_pdf,
            has_video=row.capability_has_video,
            has_tools=row.capability_has_tools,
            has_parallel_tools=row.capability_has_parallel_tools,
            total_tokens=row.total_tokens,
        )
```

### 2.2 Detection rules (exact logic)

```python
def detect_turn_capabilities(messages: list[dict], tools: list[dict] | None = None) -> TurnCapabilities:
    """
    Scan all messages in this turn to detect capabilities.
    Rules are applied deterministically — no ML, no heuristics, no fuzzy matching.
    """
    caps = TurnCapabilities()

    # 1. Tools in the request (tool definitions sent by client)
    if tools and len(tools) > 0:
        caps.has_tools = True

    for msg in messages:
        content = msg.get("content")

        # 2. Content is an array (multimodal)
        if isinstance(content, list):
            for part in content:
                part_type = part.get("type", "")

                # 2a. Images: type "image_url"
                if part_type == "image_url":
                    caps.has_images = True

                # 2b. Audio: type "input_audio"
                elif part_type == "input_audio":
                    caps.has_audio = True

                # 2c. PDF: type "file" with application/pdf mimetype
                elif part_type == "file":
                    mime = part.get("mime_type", part.get("mimetype", ""))
                    if "pdf" in mime.lower():
                        caps.has_pdf = True
                    elif any(v in mime.lower() for v in ("video/", "mp4", "webm", "avi")):
                        caps.has_video = True

                # 2d. Video: type "video_url" or file with video mime
                elif part_type in ("video_url", "video"):
                    caps.has_video = True

        # 3. Tool calls in assistant messages (already in history)
        tool_calls = msg.get("tool_calls")
        if tool_calls and isinstance(tool_calls, list) and len(tool_calls) > 0:
            caps.has_tools = True
            if len(tool_calls) > 1:
                caps.has_parallel_tools = True

        # 4. Tool results (role: "tool")
        if msg.get("role") == "tool":
            caps.has_tools = True

    return caps
```

### 2.3 Capability accumulation flow

```python
async def accumulate_capabilities(
    db_session,
    conversation_id: str,
    turn_caps: TurnCapabilities,
    existing_session: SessionCapabilities,
) -> SessionCapabilities:
    """
    Merge turn capabilities into the session.
    Update the DB row. Return updated session caps.
    """
    updated = existing_session.merge(turn_caps)

    # Update DB
    await db_session.execute(
        update(Conversation)
        .where(Conversation.id == uuid.UUID(conversation_id))
        .values(
            capability_has_images=updated.has_images,
            capability_has_audio=updated.has_audio,
            capability_has_pdf=updated.has_pdf,
            capability_has_video=updated.has_video,
            capability_has_tools=updated.has_tools,
            capability_has_parallel_tools=updated.has_parallel_tools,
        )
    )
    return updated
```

### 2.4 What capability detection does NOT do in Sprint 2

- Does NOT reject audio/PDF/video requests (those are passed through, flagged, and rejected by compatibility logic)
- Does NOT count tokens (tiktoken integration deferred to Sprint 2 — use estimate: ~4 chars = 1 token for now)
- Does NOT parse tool schemas to determine complexity (that's Sprint 3)
- Does NOT detect thinking/reasoning blocks
- Does NOT detect `tool_choice` presence

---

## 3. Compatibility Validation (`middleware/compatibility.py`)

### 3.1 CompatibilityResult

```python
from enum import Enum

class CompatibilityStatus(str, Enum):
    SAFE = "safe"
    WARNING = "warning"
    BLOCKED = "blocked"

@dataclass
class CompatibilityResult:
    status: CompatibilityStatus
    reason: str
    remediation: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)
```

### 3.2 validate_switch() — the core function

```python
def validate_switch(
    from_pseudo_name: str,
    to_pseudo_name: str,
    to_pseudo: PseudoModel,
    caps: SessionCapabilities,
    config: ProxyConfig,
) -> CompatibilityResult:
    """
    Determine if switching from one pseudo-model to another is safe.
    Returns SAFE, WARNING, or BLOCKED with reason and remediation options.

    This function IS the source of truth for compatibility. It implements
    the logic from §8.1 of the plan. Every check is deterministic.
    """
    reasons = []

    # ---- CHECK 1: Images → model without vision ----
    if caps.has_images and not any(m.vision for m in to_pseudo.physical_models):
        if to_pseudo.image_handling.on_downgrade == "auto_describe":
            reasons.append("IMAGES_WILL_BE_DESCRIBED")
            return CompatibilityResult(
                status=CompatibilityStatus.WARNING,
                reason="Images in history will be auto-described textually before migration (Sprint 5).",
                details={"images_described_by": "current_vision_model"},
            )
        else:
            return CompatibilityResult(
                status=CompatibilityStatus.BLOCKED,
                reason="Conversation contains images but destination pseudo-model lacks vision support.",
                remediation=[
                    "Enable 'auto_describe' on destination pseudo-model in pseudo_models.yaml",
                    "POST /conversations/{id}/degrade-images (available in Sprint 5)",
                ],
            )

    # ---- CHECK 2: Audio in history ----
    if caps.has_audio:
        to_has_audio = any(m.audio if hasattr(m, 'audio') else False for m in to_pseudo.physical_models)
        if not to_has_audio:
            return CompatibilityResult(
                status=CompatibilityStatus.BLOCKED,
                reason="Conversation contains audio content. No destination model supports audio. Audio degradation is not available in v1.",
                remediation=[
                    "Start a new conversation without audio content",
                    "Audio support is planned for v2",
                ],
            )

    # ---- CHECK 3: PDF in history → model without vision ----
    if caps.has_pdf and not any(m.vision for m in to_pseudo.physical_models):
        return CompatibilityResult(
            status=CompatibilityStatus.BLOCKED,
            reason="Conversation contains PDF files. Destination lacks vision support. In v1, PDFs require vision models.",
            remediation=[
                "Use a vision-capable pseudo-model (avanzada-vision, flash-vision)",
                "PDF text extraction is planned for v2",
            ],
        )

    # ---- CHECK 4: Video in history ----
    if caps.has_video:
        return CompatibilityResult(
            status=CompatibilityStatus.BLOCKED,
            reason="Video content is not supported in any pseudo-model in v1.",
            remediation=["Video support is planned for a future version"],
        )

    # ---- CHECK 5: Parallel tools → destination lacks parallel models ----
    if caps.has_parallel_tools:
        parallel_eligible = [m for m in to_pseudo.physical_models if m.parallel_tools]
        if not parallel_eligible:
            return CompatibilityResult(
                status=CompatibilityStatus.BLOCKED,
                reason="Conversation history contains parallel tool calls. No model in the destination pseudo-model supports parallel tools.",
                remediation=[
                    "POST /conversations/{id}/normalize-tools — serialize parallel calls to sequential (Sprint 3)",
                    f"Switch to a pseudo-model with parallel_tools support",
                ],
            )

    # ---- CHECK 6: Context too large for destination ----
    if to_pseudo.context_window and caps.total_tokens > to_pseudo.context_window:
        return CompatibilityResult(
            status=CompatibilityStatus.BLOCKED,
            reason=f"Accumulated context ({caps.total_tokens} tokens) exceeds destination window ({to_pseudo.context_window} tokens).",
            remediation=[
                "POST /conversations/{id}/compact — compact the conversation before switching (Sprint 6)",
                f"Switch to a pseudo-model with larger context window",
            ],
        )

    # ---- CHECK 7: Tools downgrade warning ----
    if caps.has_tools:
        to_strict_count = sum(1 for m in to_pseudo.physical_models if m.tools_strict)
        from_pm = config.pseudo_models.get(from_pseudo_name)
        from_strict_count = sum(1 for m in from_pm.physical_models if m.tools_strict) if from_pm else 0

        if to_strict_count == 0 and from_strict_count > 0:
            return CompatibilityResult(
                status=CompatibilityStatus.WARNING,
                reason="Destination pseudo-model lacks models with tools_strict support. Tool call parameter validation may be less reliable.",
                details={"from_strict_models": from_strict_count, "to_strict_models": 0},
            )

    # ---- All checks passed → SAFE ----
    return CompatibilityResult(
        status=CompatibilityStatus.SAFE,
        reason="All capabilities compatible.",
    )
```

### 3.3 When validate_switch is called

It is called in the chat endpoint (`/v1/chat/completions`) when:
1. The `conversation_id` already exists
2. The `model` field in the request differs from `conversation.pseudo_model`

**If the result is WARNING:** Continue processing, but include the warning in `proxy_metadata.warning`.

**If the result is BLOCKED:** Return HTTP 409 immediately with the full `CompatibilityResult` as the response body. Do NOT attempt to forward the request to any model.

**If the result is SAFE:** Continue normally, update conversation's `pseudo_model` and pinned `physical_model`.

### 3.4 What compatibility validation does NOT do in Sprint 2

- Does NOT auto-describe images on WARNING (that's Sprint 5)
- Does NOT normalize parallel tools (Sprint 3)
- Does NOT compact context (Sprint 6)
- Does NOT evaluate tool schema complexity to determine actual risk
- Does NOT consider router_llm suggestions
- Does NOT enforce all remediation options are actually implemented (they are informational in Sprint 2)

---

## 4. Input Threshold Guard (`middleware/threshold_guard.py`)

### 4.1 Logic

```python
def check_input_threshold(
    pseudo_model: PseudoModel,
    estimated_tokens: int,
) -> None | str:
    """
    Check if the input exceeds the pseudo-model's threshold.
    Returns None if OK, or an error code string if exceeded.
    """
    if pseudo_model.input_token_threshold is None:
        return None  # No limit (e.g., compactador)

    if estimated_tokens > pseudo_model.input_token_threshold:
        return "INPUT_EXCEEDS_THRESHOLD"
    return None
```

### 4.2 Integration in chat endpoint

```python
# In chat endpoint, BEFORE calling LiteLLM:
# Estimate input tokens (rough: 4 chars = 1 token for Sprint 2; proper tiktoken in Sprint 3)
estimated_input = estimate_tokens(messages)

threshold_error = check_input_threshold(pm, estimated_input)
if threshold_error:
    # In Sprint 4, pre_compaction would run here if enabled
    if pm.pre_compaction.enabled:
        # Will be implemented in Sprint 4
        pass
    else:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "INPUT_EXCEEDS_THRESHOLD",
                "message": f"Input ({estimated_input} tokens) exceeds threshold ({pm.input_token_threshold} tokens) for pseudo-model '{pm.display_name}'.",
                "suggestions": suggest_pseudo_models_with_higher_threshold(config, estimated_input),
            }
        )
```

### 4.3 Token estimation (temporary for Sprint 2)

```python
def estimate_tokens(messages: list[dict]) -> int:
    """
    Rough token estimation. Sprint 2 uses the 4-chars-per-token heuristic.
    Sprint 3 will replace this with tiktoken-based counting.
    """
    total_chars = 0
    for msg in messages:
        # Count content string
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part.get("text"), str):
                    total_chars += len(part["text"])
        # Count tool call arguments
        for tc in (msg.get("tool_calls") or []):
            total_chars += len(tc.get("function", {}).get("arguments", ""))

    return max(1, total_chars // 4)  # ~4 chars per token
```

### 4.4 What threshold guard does NOT do

- Does NOT count tokens precisely with tiktoken (Sprint 3)
- Does NOT pre-compact (Sprint 4)
- Does NOT check continuous compaction trigger (Sprint 4)
- Does NOT check `CONTEXT_UNUSABLE` (Sprint 6 — that's about accumulated context, not input)

---

## 5. Tool Filter (`middleware/tool_filter.py`)

### 5.1 Logic

```python
def get_eligible_models(
    pseudo_model: PseudoModel,
    session_caps: SessionCapabilities,
) -> list[PhysicalModel]:
    """
    Filter the pseudo-model's physical_models pool based on session capabilities.

    All models already have openai_tools_compatible: true (validated at startup).
    If session has parallel tools, filter to only parallel_tools: true models.
    """
    models = pseudo_model.physical_models

    if not session_caps.has_parallel_tools:
        return models  # No filtering needed

    parallel_eligible = [m for m in models if m.parallel_tools]

    if parallel_eligible:
        return parallel_eligible

    # No model supports parallel tools — return all with warning for client
    # The compatibility validator will block the switch if this happens
    return models
```

### 5.2 When tool_filter is called

Called in the chat endpoint when:
- The session has `has_tools: true` OR the current request contains tools
- The current request or session has `has_parallel_tools: true`

If the pinned physical model is NOT in the eligible list:
1. Try the first eligible model
2. Update affinity in Valkey to the new model
3. Set `proxy_metadata.tools_filter_applied: true`
4. Set `proxy_metadata.tools_filter_reason: "parallel_tools_required"`

If NO eligible models exist → return 409 `PARALLEL_TOOLS_INCOMPATIBLE` (this is also caught by `validate_switch`). The user must normalize tools first (Sprint 3).

### 5.3 What tool_filter does NOT do

- Does NOT filter by `tools_strict` (that's a preference, not a hard requirement)
- Does NOT filter by tool schema complexity
- Does NOT handle `tool_choice: "required"` enforcement (Sprint 3)
- Does NOT reorder models by tool compatibility score

---

## 6. Conversation Endpoints

### 6.1 `GET /conversations/{id}`

Returns the full state of a conversation.

```json
{
  "conversation_id": "abc-123",
  "created_at": "2026-01-15T10:00:00Z",
  "pseudo_model": "normal",
  "physical_model": "qwen3-max",
  "total_tokens": 45230,
  "turn_count": 12,
  "capabilities": {
    "has_images": false,
    "has_audio": false,
    "has_pdf": false,
    "has_video": false,
    "has_tools": true,
    "has_parallel_tools": false
  },
  "active_snapshot_id": null
}
```

### 6.2 `GET /conversations/{id}/compatible-models`

Returns ALL pseudo-models with their compatibility status given the current conversation's capabilities.

Response format:
```json
{
  "conversation_id": "abc-123",
  "current_pseudo_model": "avanzada-vision",
  "capabilities": {
    "has_images": true,
    "has_tools": true,
    "has_parallel_tools": false
  },
  "compatible_models": [
    {
      "pseudo_model": "pensamiento-profundo-caro",
      "display_name": "Pensamiento Profundo",
      "status": "blocked",
      "reason": "Destination lacks vision support and does not have auto_describe enabled.",
      "remediation": ["Set image_handling.on_downgrade: 'auto_describe' in config"]
    },
    {
      "pseudo_model": "tareas-avanzadas",
      "display_name": "Tareas Avanzadas",
      "status": "blocked",
      "reason": "Conversation contains images but destination lacks vision support.",
      "remediation": ["POST /conversations/{id}/degrade-images (Sprint 5)"]
    },
    {
      "pseudo_model": "avanzada-vision",
      "display_name": "Visión Avanzada",
      "status": "safe",
      "reason": "Current pseudo-model."
    },
    {
      "pseudo_model": "normal",
      "display_name": "Normal",
      "status": "blocked",
      "reason": "Conversation contains images but destination lacks vision support.",
      "remediation": ["POST /conversations/{id}/degrade-images (Sprint 5)"]
    }
  ]
}
```

### 6.3 `GET /conversations/{id}/tools-compatibility`

Returns tool-specific compatibility analysis.

```json
{
  "conversation_id": "abc-123",
  "tools_used": true,
  "parallel_tools_used": true,
  "pseudo_models": [
    {
      "name": "tareas-avanzadas",
      "display_name": "Tareas Avanzadas",
      "tool_support": {
        "parallel_eligible": true,
        "parallel_models": ["deepseek-v4-pro", "deepseek-v4-flash"],
        "strict_models": ["deepseek-v4-pro"],
        "non_strict_models": ["deepseek-v4-flash"],
        "blocked_models": ["minimax-m2.5"]
      }
    }
  ]
}
```

---

## 7. Incoming Content Validation (CRITICAL — prevents OpenCode silent stripping)

### 7.0 The problem this solves

Because `GET /v1/models` advertises ALL capabilities as `true` for every pseudo-model (see Sprint 1 §10.0), OpenCode sends ALL content — images, PDFs, audio, parallel tools — to every pseudo-model. This is intentional: OpenCode must never silently strip content.

The proxy is now responsible for validating that the **current pseudo-model** can actually handle the incoming content. If it can't, the proxy returns a clear, descriptive error. The user knows exactly what happened.

**This is the anti-silent-error design. Every rejection has a clear message and remediation path.**

### 7.1 validate_incoming_content()

```python
def validate_incoming_content(
    turn_caps: TurnCapabilities,
    pseudo_model: PseudoModel,
    pseudo_model_name: str,
) -> None | HTTPException:
    """
    Validate that the current pseudo-model's physical models can handle
    the incoming content. Returns None if OK, or raises HTTPException with
    a descriptive error and remediation options.

    This check runs on EVERY turn, not just on pseudo-model switches.
    It prevents silent data loss when the client sends content the model
    can't process.
    """
    physical_models = pseudo_model.physical_models

    # ---- CHECK: Images → model without vision ----
    if turn_caps.has_images:
        has_vision_model = any(m.vision for m in physical_models)
        if not has_vision_model:
            # Find pseudo-models that DO support vision
            vision_pseudos = [
                name for name, pm in config.pseudo_models.items()
                if any(m.vision for m in pm.physical_models)
            ]
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "IMAGES_NOT_SUPPORTED_BY_PSEUDO_MODEL",
                    "message": (
                        f"Pseudo-model '{pseudo_model_name}' ({pseudo_model.display_name}) "
                        f"has no vision-capable physical models. The incoming request contains images "
                        f"that cannot be processed. No content was lost — the request was rejected with this error."
                    ),
                    "remediation": [
                        f"Switch to a vision-capable pseudo-model: {vision_pseudos}",
                        "Use auto_describe to downgrade images to text (Sprint 5)",
                    ],
                    "current_pseudo_model": pseudo_model_name,
                    "vision_capable_pseudo_models": vision_pseudos,
                }
            )

    # ---- CHECK: Audio → no model supports audio in v1 ----
    if turn_caps.has_audio:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "AUDIO_NOT_SUPPORTED",
                "message": (
                    f"Pseudo-model '{pseudo_model_name}' does not support audio content. "
                    f"No pseudo-model in v1 supports audio processing. "
                    f"The request was rejected rather than silently dropping the audio."
                ),
                "remediation": [
                    "Remove audio content from the request",
                    "Audio transcription support is planned for v2",
                ],
            }
        )

    # ---- CHECK: PDF → model without vision ----
    if turn_caps.has_pdf:
        has_vision_model = any(m.vision for m in physical_models)
        if not has_vision_model:
            vision_pseudos = [
                name for name, pm in config.pseudo_models.items()
                if any(m.vision for m in pm.physical_models)
            ]
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "PDF_NOT_SUPPORTED",
                    "message": (
                        f"Pseudo-model '{pseudo_model_name}' has no vision-capable physical models. "
                        f"PDFs are treated as images in v1 and require a vision model. "
                        f"The request was rejected — no content was silently lost."
                    ),
                    "remediation": [
                        f"Switch to a vision-capable pseudo-model: {vision_pseudos}",
                        "Extract text from the PDF before sending (manual, or v2 feature)",
                    ],
                }
            )

    # ---- CHECK: Video → not supported in v1 ----
    if turn_caps.has_video:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "VIDEO_NOT_SUPPORTED",
                "message": (
                    "Video content is not supported in any pseudo-model in v1. "
                    "The request was rejected rather than silently dropping the video."
                ),
                "remediation": [
                    "Extract key frames as images and send those instead",
                    "Video frame extraction is planned for v2",
                ],
            }
        )

    # ---- CHECK: Parallel tools → model without parallel support ----
    if turn_caps.has_parallel_tools:
        has_parallel_models = any(m.parallel_tools for m in physical_models)
        if not has_parallel_models:
            parallel_pseudos = [
                name for name, pm in config.pseudo_models.items()
                if any(m.parallel_tools for m in pm.physical_models)
            ]
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "PARALLEL_TOOLS_NOT_SUPPORTED_BY_PSEUDO_MODEL",
                    "message": (
                        f"Pseudo-model '{pseudo_model_name}' has no physical models "
                        f"with parallel_tools: true. The incoming request contains parallel "
                        f"tool calls that cannot be processed."
                    ),
                    "remediation": [
                        f"Switch to a pseudo-model with parallel tool support: {parallel_pseudos}",
                        "Use POST /conversations/{id}/normalize-tools to serialize parallel calls (Sprint 3)",
                    ],
                }
            )

    return None  # All checks passed
```

### 7.2 Design rationale

| Scenario | Old behavior (without this check) | New behavior (with this check) |
|---|---|---|
| User sends image to `normal` (no vision) | OpenCode strips image silently because `GET /v1/models` says `vision: false`. User never knows. | `GET /v1/models` says `vision: true`. OpenCode sends image. Proxy returns `IMAGES_NOT_SUPPORTED_BY_PSEUDO_MODEL` with clear message and list of vision-capable pseudo-models. |
| User sends PDF to `deep-flash` (no vision) | OpenCode strips PDF silently. User wonders why model didn't see the PDF. | Proxy returns `PDF_NOT_SUPPORTED` with remediation options. |
| User sends parallel tools to `flash-lowcost` (no parallel) | OpenCode might strip tools or send them anyway. Unpredictable. | Proxy returns `PARALLEL_TOOLS_NOT_SUPPORTED_BY_PSEUDO_MODEL` with remediation. |

**Key principle:** The proxy NEVER silently drops content. It either processes it or rejects it with a clear error. The client (OpenCode) never decides what to strip — it sends everything because the proxy said it could handle it.

---

## 8. Changes to chat endpoint (Sprint 1 → Sprint 2)

The `/v1/chat/completions` endpoint from Sprint 1 must be extended with the following steps injected at the right points:

```
After "Resolve conversation" (step 1):
  → IF existing conversation AND model differs from conversation.pseudo_model:
       → call validate_switch(from, to, caps, config)
       → IF BLOCKED: return 409 immediately
       → IF WARNING: store warning for proxy_metadata

After "Determine physical model" (step 2):
  → IF session.has_parallel_tools:
       → filter pool to parallel_tools: true models
       → IF pinned model not in eligible: update affinity, set flag

Before "Call LiteLLM" (step 5):
  → detect_turn_capabilities(messages, tools)
  → validate_incoming_content(turn_caps, pm, pseudo_model_name)  ← NEW: §7
       → IF unsupported: return 400 with descriptive error
  → accumulate_capabilities(db, conv_id, turn_caps, session_caps)
  → check_input_threshold(pm, estimated_input_tokens)
  → IF exceeded AND pre_compaction disabled: return 400

After "Save turn" (step 8):
  → Save turn_capabilities to conversation_turn row (had_images, had_tools, had_parallel_tools)
  → Update conversation capability flags
```

### 8.1 Extended proxy_metadata

Add these fields to the `build_proxy_metadata()` output:

```json
{
  "proxy_metadata": {
    "physical_model": "qwen3-max",
    "pseudo_model": "normal",
    "conversation_id": "abc-123",
    "affinity_maintained": true,
    "fallback_applied": false,
    "fallback_reason": null,
    "tools_filter_applied": false,
    "tools_filter_reason": null,
    "context_tokens_total": 45000,
    "context_usage_pct": 47,
    "pseudo_model_threshold": 96000,
    "warning": null,
    "capabilities_detected": {
      "has_images": false,
      "has_tools": true
    }
  }
}
```

---

## 9. Tests (Sprint 2)

### 9.1 test_capability_detector.py (minimum 15 tests)

1. Text-only message → no capabilities
2. Single `image_url` content part → `has_images: true`
3. Multiple images → `has_images: true` (idempotent)
4. `input_audio` content part → `has_audio: true`
5. `file` with `application/pdf` mime → `has_pdf: true`
6. `file` with `video/mp4` mime → `has_video: true`
7. `video_url` type → `has_video: true`
8. Tool definitions in request → `has_tools: true`
9. Single tool_call in assistant message → `has_tools: true`, `has_parallel_tools: false`
10. Two tool_calls in assistant message → `has_tools: true`, `has_parallel_tools: true`
11. Tool result message (role: "tool") → `has_tools: true`
12. Mixed content: image + tools → both flags true
13. Empty messages array → all flags false
14. Content is null → handled gracefully (no crash)
15. Messages with non-standard fields → ignored (Pydantic extra="forbid" catches them earlier)

### 9.1b test_incoming_content_validation.py (minimum 8 tests) — NEW

These tests verify that incoming content is validated against the CURRENT pseudo-model (not just on switch):

1. Image sent to `normal` (no vision models) → 400 `IMAGES_NOT_SUPPORTED_BY_PSEUDO_MODEL`
2. Image sent to `avanzada-vision` (has vision models) → proceeds normally (200)
3. Audio sent to any pseudo-model → 400 `AUDIO_NOT_SUPPORTED`
4. PDF sent to `deep-flash` (no vision) → 400 `PDF_NOT_SUPPORTED`
5. PDF sent to `avanzada-vision` (has vision) → warning but proceeds (PDFs treated as images)
6. Video sent to any pseudo-model → 400 `VIDEO_NOT_SUPPORTED`
7. Parallel tools sent to `flash-lowcost` (no parallel models) → 400 `PARALLEL_TOOLS_NOT_SUPPORTED_BY_PSEUDO_MODEL`
8. Error responses include `remediation` array with actionable options
9. Error responses include `vision_capable_pseudo_models` or `parallel_pseudo_models` lists

### 9.2 test_compatibility.py (minimum 25 tests — covering the matrix)

Test every row of the compatibility matrix from §8.2 of the plan:

1. `normal` → `tareas-avanzadas` (no multimedia, no tools) → SAFE
2. `normal` → `tareas-avanzadas` (tools, no parallel) → SAFE
3. `normal` → `pensamiento-profundo-caro` → SAFE
4. `normal` → `deep-flash` (no multimedia, no tools) → WARNING
5. `normal` → `deep-flash` (with tools) → WARNING
6. `normal` → `flash-lowcost` (no multimedia) → WARNING
7. `normal` → `flash-lowcost` (tools, parallel) → BLOCKED
8. `tareas-avanzadas` → `normal` (no parallel tools) → SAFE
9. `tareas-avanzadas` → `normal` (parallel tools) → WARNING
10. `tareas-avanzadas` → `pensamiento-profundo-caro` → SAFE
11. `tareas-avanzadas` → `deep-flash` (with tools) → WARNING
12. `pensamiento-profundo-caro` → `tareas-avanzadas` → SAFE
13. `pensamiento-profundo-caro` → `deep-flash` (with tools) → WARNING
14. `avanzada-vision` → `flash-vision` (with images) → WARNING
15. `avanzada-vision` → `normal` (with images, auto_describe=false) → BLOCKED
16. `avanzada-vision` → `normal` (with images, auto_describe=true) → WARNING
17. `avanzada-vision` → `deep-flash` (with images) → BLOCKED
18. `flash-vision` → `avanzada-vision` (with images) → SAFE
19. `deep-flash` → `normal` (no multimedia) → SAFE
20. `deep-flash` → `normal` (with multimedia) → SAFE
21. `flash-lowcost` → `normal` → SAFE
22. Any → `compactador` → SAFE
23. Context exceeds destination window → BLOCKED (`CONTEXT_TOO_LARGE`)
24. Parallel tools → destination has parallel=false on all models → BLOCKED
25. Audio in history → destination has no audio → BLOCKED
26. Video in history → always BLOCKED
27. PDF in history → destination no vision → BLOCKED
28. Same pseudo-model → always SAFE (no switch)
29. WARNING on tools strict downgrade (strict → no strict models in destination)
30. Determinism: same inputs → same result (run twice) ✓

### 9.3 test_tool_filter.py (minimum 8 tests)

1. No parallel tools in session → all models returned
2. Parallel tools in session → only `parallel_tools: true` models returned
3. DeepSeek V4 Pro excluded from parallel-eligible for `pensamiento-profundo-caro`
4. All `tareas-avanzadas` models returned when no parallel (MiniMax included)
5. `pensamiento-profundo-caro` with parallel: only deepseek-v4-pro
6. `normal` with parallel: only deepseek-v4-flash
7. `flash-lowcost` with parallel: pool empty → returns all models (warning issued separately)
8. `deep-flash` with parallel: only deepseek-v4-flash

### 9.4 test_threshold_guard.py (minimum 6 tests)

1. Input below threshold → no error
2. Input above threshold without pre_compaction → 400 `INPUT_EXCEEDS_THRESHOLD`
3. Input above threshold with pre_compaction → passes (deferred to Sprint 4 implementation)
4. compactador (null threshold) → always passes
5. Exact threshold boundary → passes
6. Threshold +1 token → fails

### 9.5 test_conversations_api.py (minimum 8 tests)

1. `GET /conversations/{id}` returns full state with capabilities
2. Non-existent conversation → 404
3. `GET /conversations/{id}/compatible-models` returns all pseudo-models with status
4. Compatible models determinism: call twice → same result
5. Compatible models properly reflects current capabilities
6. `GET /conversations/{id}/tools-compatibility` returns tool support details
7. Tools-compatibility properly identifies parallel-eligible models
8. Conversation state updates after each turn

---

## 9. Acceptance Criteria (Sprint 2)

- [x] `capability_detector.py` detects all 6 capability types (images, audio, PDF, video, tools, parallel tools)
- [x] Capability flags accumulate in DB and are NEVER reset (additive-only `merge()`)
- [x] `validate_switch()` returns correct status for all 31 test cases from the compatibility matrix
- [x] HTTP 409 returned with descriptive error and remediation options on BLOCKED
- [x] WARNING is included in `proxy_metadata` without blocking the request
- [x] **`validate_incoming_content()` rejects unsupported content on the CURRENT pseudo-model with clear 400 errors**
- [x] **Image sent to `normal` (no vision) → 400 `IMAGES_NOT_SUPPORTED_BY_PSEUDO_MODEL` with remediation**
- [x] **Error responses include list of compatible pseudo-models (e.g., `vision_capable_pseudo_models`)**
- [x] `tool_filter.py` correctly filters pool by `parallel_tools` when `has_parallel_tools: true`
- [x] `GET /conversations/{id}/compatible-models` is deterministic and fast
- [x] `GET /conversations/{id}/tools-compatibility` returns accurate per-model tool info
- [x] Input threshold check prevents requests exceeding `input_token_threshold` (unless pre_compaction)
- [x] All **128 tests pass** (65+ Sprint 2 + all Sprint 1 tests)
- [x] No regression on Sprint 1 tests

---

## 10. Explicitly OUT OF SCOPE for Sprint 2

| Feature | Sprint |
|---|---|
| Canonical tool format storage (JSONB schema in DB) | 3 |
| Tool normalization (`POST /normalize-tools`) | 3 |
| Tool edge case handling (streaming partial, mixed, thinking blocks) | 3 |
| `tool_choice: "required"` enforcement | 3 |
| Pre-compaction (even though threshold detection is done) | 4 |
| Continuous compaction | 4 |
| Explicit compaction (`POST /compact`) | 6 |
| Image auto-describe (even though WARNING references it) | 5 |
| Router LLM evaluation | 5 |
| Provider cache optimization | 7 |
| Rate limiting | 8 |
| Auth middleware | 8 |
| Audit log | 6 |
| `POST /conversations/{id}/change-pseudo-model` (switch validation is done, but the explicit endpoint with atomic switch comes later) | 5 |

---

## 11. Sprint 3 Alignment Notes

> **Updated:** 2026-05-23 — Sprint 3 is complete.

### 11.1 Changes from Sprint 2 that affect Sprint 3 compatibility

1. **`estimate_tokens()` replaced with tiktoken** (plan-proxy.md §2)
   - `src/service/capability_detector.py` now uses `o200k_base` encoding (fallback `cl100k_base`)
   - Returns actual token counts (was ~chars/4 heuristic)
   - Test values updated in `test_capability_detector.py`

2. **`accumulate_capabilities()` now handles `max_tools_level`**
   - `SessionCapabilities.merge()` uses `max()` for `max_tools_level`
   - `SessionCapabilities` has new field `max_tools_level: int = 0`
   - DB update now includes `max_tools_level` column

3. **`TurnCapabilities` extended with Sprint 3 fields**
   - `tools_incomplete: bool = False`
   - `thinking_blocks: dict | None = None`
   - `tools_level_used: int = 0`

### 11.2 New DB columns added by Sprint 3 migration `0002`

| Table | Column | Type |
|---|---|---|
| `conversation_turns` | `tool_definitions` | JSONB |
| `conversation_turns` | `thinking_blocks` | JSONB |
| `conversation_turns` | `tools_incomplete` | BOOLEAN |
| `conversation_turns` | `tools_level_used` | INTEGER |
| `conversations` | `max_tools_level` | INTEGER |

### 11.3 New test count breakdown

| Suite | Tests | Status |
|---|---|---|
| Sprint 1 (affinity, chat, fallback, streaming, models, config) | 63 | ✅ Passing |
| Sprint 2 (capability_detector, compatibility, tool_filter, threshold_guard, conversations_api, incoming_content) | 65 | ✅ Passing |
| Sprint 3 (tools_canonical, tool_normalization, tools_edge_cases) | 50 | ✅ Passing |
| Sprint 3 e2e integration (skipped without API keys) | 9 | ⏭️ Skipped |
| **Total** | **187** | **178 ✅ + 9 ⏭️** |
