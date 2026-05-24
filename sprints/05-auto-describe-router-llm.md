# Sprint 5 — Auto-describe Images & Router LLM

> **Duration:** 2 weeks
> **Goal:** Migrating from vision models is seamless via auto-describe. The router LLM informs about potential downgrades without imposing them.
> **Success criterion:** User switches from `avanzada-vision` to `normal`. 3 images are auto-described. Conversation continues without conceptual loss. Router suggests downgrade on simple tasks but never changes the model.

---

## 1. Dependencies from Previous Sprints

| Dependency | Source | Status |
|---|---|---|
| Capability detection (has_images) | Sprint 2 | Complete |
| `validate_switch()` returning WARNING for image downgrade with `auto_describe` | Sprint 2 | Complete |
| `image_handling.on_downgrade` config in pseudo_models.yaml | Sprint 1 | Already validated |
| JSONB message storage | Sprint 1 | Ready |
| Chat endpoint with middleware chain | Sprint 1-4 | Ready for injection |

### 1.1 New files/modules

```
src/
├── multimedia/
│   ├── __init__.py
│   ├── image_describer.py       # NEW — auto-describe images
│   └── degradation.py           # NEW — POST /degrade-images endpoint
│
├── router_llm/
│   ├── __init__.py
│   └── suggester.py             # NEW — evaluate complexity, suggest downgrade
│
└── tests/
    ├── test_image_describe.py       # NEW
    └── test_router_llm.py           # NEW
```

### 1.2 DB changes

**Add to `conversations`:**

| Column | Type | Default | Notes |
|---|---|---|---|
| `images_described` | INTEGER | 0 | How many images have been auto-described |
| `images_degraded_manually` | BOOLEAN | FALSE | Whether POST /degrade-images was used |

---

## 2. Image Auto-describe (`multimedia/image_describer.py`)

### 2.1 When it activates

Auto-describe activates when ALL of:

1. `validate_switch()` returns `WARNING` with reason `IMAGES_WILL_BE_DESCRIBED`
2. The destination pseudo-model has `image_handling.on_downgrade == "auto_describe"`
3. The current pseudo-model has at least one vision-capable physical model
4. The conversation has `has_images: true`

It activates BEFORE the actual switch — during `validate_switch()` or immediately after when the chat endpoint detects the warning and decides to proceed with auto-describe.

### 2.2 Flow

```
1. User requests switch: from "avanzada-vision" to "normal"
2. validate_switch() returns WARNING: IMAGES_WILL_BE_DESCRIBED
3. Proxy identifies all images in the conversation history
4. For each image, proxy calls the current vision model with a description prompt
5. Descriptions are stored as [IMAGEN_DESCRITA #N] messages in the history
6. The conversation is marked with images_described: N
7. The switch proceeds
8. The new model receives textual descriptions instead of images
9. proxy_metadata reports: "images_described: 3, described_by: gemini-3.5-flash"
```

### 2.3 auto_describe_images() function

```python
async def auto_describe_images(
    messages: list[dict],
    vision_model: str,
) -> tuple[list[dict], dict]:
    """
    Scan messages for image_url content, describe each image using the vision model,
    and replace image content with [IMAGEN_DESCRITA #N] text.

    Returns (modified_messages, metadata).
    Original image URLs are preserved in a separate field for audit.
    """
    # Find all images
    image_refs = []
    for msg_idx, msg in enumerate(messages):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part_idx, part in enumerate(content):
            if part.get("type") == "image_url":
                image_refs.append({
                    "msg_idx": msg_idx,
                    "part_idx": part_idx,
                    "image_url": part.get("image_url", {}).get("url", ""),
                    "detail": part.get("image_url", {}).get("detail", "auto"),
                })

    if not image_refs:
        return messages, {"images_described": 0, "reason": "no_images_found"}

    described = []
    modified_messages = deepcopy(messages)
    description_prompt = (
        "Describe this image in detail for a text-only model. "
        "Include: what is shown, layout, visible text, colors, "
        "key elements, and any technical details relevant to the conversation. "
        "Be thorough but concise."
    )

    for idx, img_ref in enumerate(image_refs):
        # Build a single-image request
        img_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": description_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": img_ref["image_url"],
                            "detail": img_ref["detail"],
                        }
                    }
                ]
            }
        ]

        try:
            response = await litellm.acompletion(
                model=vision_model,
                messages=img_messages,
                max_tokens=1000,
            )
            description = response.choices[0].message.content
        except Exception as e:
            description = f"[IMAGE DESCRIPTION FAILED: {str(e)}]"

        # Replace the image_url with text description in the message
        msg = modified_messages[img_ref["msg_idx"]]
        content_list = msg["content"]

        # Replace the image_url part with a text description
        content_list[img_ref["part_idx"]] = {
            "type": "text",
            "text": f"[IMAGEN_DESCRITA #{idx + 1} — described by {vision_model}]\n\n{description}"
        }

        described.append({
            "image_index": idx + 1,
            "turn": img_ref["msg_idx"] + 1,
            "described_by": vision_model,
        })

    metadata = {
        "images_described": len(described),
        "described_by": vision_model,
        "descriptions": described,
    }

    return modified_messages, metadata
```

### 2.4 Integration into validate_switch() or chat endpoint

When `validate_switch()` returns WARNING with `IMAGES_WILL_BE_DESCRIBED`, the chat endpoint should:

```python
if result.status == CompatibilityStatus.WARNING and result.reason == "IMAGES_WILL_BE_DESCRIBED":
    # Get current vision model (the one pinned for this conversation)
    current_pm = config.pseudo_models[conversation.pseudo_model]
    vision_models = [m for m in current_pm.physical_models if m.vision]
    if vision_models:
        vision_model = existing_affinity if any(m.model == existing_affinity and m.vision for m in current_pm.physical_models) else vision_models[0].model

        # Auto-describe images in the history
        all_messages = await load_conversation_messages(db, conversation_id)
        described_messages, desc_meta = await auto_describe_images(all_messages, vision_model)

        # Store described messages as a degradation event turn
        degradation_turn = ConversationTurn(
            conversation_id=uuid.UUID(conversation_id),
            turn_number=await get_next_turn_number(db, conversation_id),
            turn_type="degradation_event",
            pseudo_model=conversation.pseudo_model,
            physical_model=vision_model,
            messages={"described_messages": described_messages, "metadata": desc_meta},
        )
        db.add(degradation_turn)

        # Update conversation flags
        conversation.images_described = desc_meta["images_described"]

        # Now proceed with the switch — the new model will receive text descriptions
        proxy_metadata["images_described"] = desc_meta["images_described"]
        proxy_metadata["images_described_by"] = vision_model
```

### 2.5 What auto-describe does NOT do

- Does NOT modify the original image messages in DB (stored in degradation_event turn)
- Does NOT delete images — they remain in original turns
- Does NOT auto-describe without `on_downgrade: "auto_describe"` in destination config
- Does NOT describe images proactively (only on switch)
- Does NOT handle PDFs or other media (only `image_url` type)
- Does NOT retry failed image descriptions (reports failure and continues)

---

## 3. Manual Image Degradation (`multimedia/degradation.py`)

### 3.1 Endpoint

```
POST /conversations/{id}/degrade-images
```

Called when `on_downgrade: "block"` and the user wants to manually degrade images to enable switching.

### 3.2 Endpoint logic

```python
@router.post("/conversations/{id}/degrade-images")
async def degrade_images(conversation_id: str, request: Request):
    """
    Manually degrade images in a conversation to text descriptions.
    This allows switching to non-vision pseudo-models when auto_describe is disabled.
    """
    db = request.app.state.db_session_factory()
    config: ProxyConfig = request.app.state.config

    conv = await db.get(Conversation, uuid.UUID(conversation_id))
    if not conv:
        raise HTTPException(404, detail={"error": "CONVERSATION_NOT_FOUND"})

    caps = await load_session_capabilities(db, conversation_id)
    if not caps.has_images:
        raise HTTPException(400, detail={"error": "NO_IMAGES", "message": "This conversation has no images to degrade."})

    # Get the current vision model
    current_pm = config.pseudo_models[conv.pseudo_model]
    vision_models = [m for m in current_pm.physical_models if m.vision]
    if not vision_models:
        raise HTTPException(400, detail={
            "error": "NO_VISION_MODEL",
            "message": "The current pseudo-model has no vision-capable physical model to describe images."
        })

    # Use the pinned model if it has vision, otherwise first vision model
    vision_model = conv.physical_model if any(m.model == conv.physical_model and m.vision for m in current_pm.physical_models) else vision_models[0].model

    # Load and describe
    all_messages = await load_conversation_messages(db, conversation_id)
    described_messages, desc_meta = await auto_describe_images(all_messages, vision_model)

    # Store
    degradation_turn = ConversationTurn(
        conversation_id=uuid.UUID(conversation_id),
        turn_number=await get_next_turn_number(db, conversation_id),
        turn_type="degradation_event",
        pseudo_model=conv.pseudo_model,
        physical_model=vision_model,
        messages={"described_messages": described_messages, "metadata": desc_meta},
    )
    db.add(degradation_turn)

    conv.images_described = desc_meta["images_described"]
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
```

### 3.3 Response

```json
{
  "conversation_id": "abc-123",
  "images_described": 3,
  "described_by": "gemini-3.5-flash",
  "can_now_switch_to": ["normal", "deep-flash", "flash-lowcost", "tareas-avanzadas", "pensamiento-profundo-caro"]
}
```

---

## 4. Router LLM (`router_llm/suggester.py`)

### 4.1 When it activates

Router LLM activates when ALL of:

1. `pseudo_model.router_llm.enabled == true`
2. `pseudo_model.router_llm.suggest_on_downgrade_only == true` (always true in current config)

Currently only enabled on: `pensamiento-profundo-caro`

### 4.2 What it does (and does NOT do)

**DOES:**
- Evaluate task complexity using a cheap model (the `suggester` pseudo-model)
- Return a suggestion in `proxy_metadata.router_suggestion`
- Inform the user "hey, this task is simple, consider using `normal` instead"

**DOES NOT:**
- Change the model automatically
- Cancel the request to the expensive model
- Block the request if the suggester fails
- Suggest upgrades (only downgrades)
- Run if the user explicitly selected the pseudo-model (it's assumed intentional)

### 4.3 evaluate_complexity() function

```python
async def evaluate_complexity(
    messages: list[dict],
    suggester_pseudo_model: str,
    config: ProxyConfig,
) -> dict | None:
    """
    Evaluate task complexity using a cheap model.
    Returns a router suggestion dict, or None if evaluation fails.
    This function is NON-BLOCKING — if it fails, the request continues normally.
    """
    suggester_pm = config.pseudo_models[suggester_pseudo_model]
    suggester_model = suggester_pm.physical_models[0].model

    # Build evaluation prompt
    # Only evaluate the last user message (not the full history)
    last_user_msg = None
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user_msg = msg.get("content", "")
            break

    if not last_user_msg:
        return None

    eval_prompt = f"""Evaluate the complexity of this task for an AI model. Consider:
- Does it require deep reasoning, multi-step planning, or creative problem-solving?
- Is it a straightforward question, simple search, or basic code generation?
- Would a cheap/fast model handle it well, or does it need an expensive reasoning model?

Task:
{str(last_user_msg)[:2000]}

Respond ONLY with valid JSON:
{{
    "complexity": "simple" | "medium" | "complex",
    "suggested_pseudo_model": "flash-lowcost" | "normal" | "tareas-avanzadas" | "pensamiento-profundo-caro",
    "reason": "one sentence explaining why"
}}"""

    try:
        response = await litellm.acompletion(
            model=suggester_model,
            messages=[{"role": "user", "content": eval_prompt}],
            max_tokens=200,
            temperature=0.0,  # Deterministic evaluation
        )
        content = response.choices[0].message.content
        result = json.loads(content)

        return {
            "complexity": result.get("complexity", "unknown"),
            "suggested": result.get("suggested_pseudo_model"),
            "reason": result.get("reason", ""),
        }
    except Exception:
        # Non-blocking: if evaluation fails, return None (no suggestion)
        return None
```

### 4.4 Integration into chat endpoint

```python
# In chat endpoint, AFTER capability detection, BEFORE calling LiteLLM:

if pm.router_llm.enabled:
    suggestion = await evaluate_complexity(
        messages=messages,
        suggester_pseudo_model=pm.router_llm.suggester,
        config=config,
    )

    if suggestion:
        current_pseudo_name = request.model
        # Only suggest if the suggested model is a downgrade (cheaper/simpler)
        if pm.router_llm.suggest_on_downgrade_only:
            if is_downgrade(suggestion["suggested"], current_pseudo_name, config):
                proxy_metadata["router_suggestion"] = suggestion
            # else: ignore — no suggestion for upgrades
        else:
            proxy_metadata["router_suggestion"] = suggestion
```

### 4.5 Downgrade determination

```python
# Model tier ordering (higher = more expensive/capable)
MODEL_TIER = {
    "flash-lowcost": 1,
    "deep-flash": 2,
    "flash-vision": 2,
    "normal": 3,
    "avanzada-vision": 4,
    "tareas-avanzadas": 5,
    "pensamiento-profundo-caro": 6,
    "compactador": 0,  # Special — never suggested
}

def is_downgrade(suggested: str, current: str, config: ProxyConfig) -> bool:
    """Check if the suggested pseudo-model is a downgrade from current."""
    suggested_tier = MODEL_TIER.get(suggested, 0)
    current_tier = MODEL_TIER.get(current, 0)
    return suggested_tier < current_tier
```

---

## 5. Updated proxy_metadata

```json
{
  "proxy_metadata": {
    "images_described": 3,
    "images_described_by": "gemini-3.5-flash",
    "router_suggestion": {
      "complexity": "simple",
      "suggested": "normal",
      "reason": "The task is a straightforward search query that does not require deep reasoning."
    }
  }
}
```

When router_suggestion is present, the user sees this and can choose to switch models for the next turn. The proxy NEVER switches automatically.

---

## 6. Tests (Sprint 5)

### 6.1 test_image_describe.py (minimum 12 tests)

1. Single image → described with `[IMAGEN_DESCRITA #1]` tag
2. Multiple images → each described with sequential tags
3. Description includes visual details
4. Original image URLs NOT lost (preserved in degradation event)
5. Non-image content preserved unchanged
6. Vision model unavailable → error handled gracefully
7. No images in conversation → no-op
8. `on_downgrade: "auto_describe"` → auto-describe triggers on switch
9. `on_downgrade: "block"` → switch blocked without auto-describe
10. `POST /degrade-images` works with vision model
11. `POST /degrade-images` without vision model → 400 error
12. `POST /degrade-images` on conversation without images → 400 error

### 6.2 test_router_llm.py (minimum 10 tests)

1. Router enabled → complexity evaluated
2. Simple task → suggests downgrade
3. Complex task → no downgrade suggestion (or suggests staying)
4. Router disabled → no evaluation performed
5. Router fails → request continues unaffected
6. `suggest_on_downgrade_only: true` → no upgrade suggestions
7. Router correctly identifies downgrade tier
8. `proxy_metadata.router_suggestion` present when applicable
9. Router suggestion NEVER changes the actual model used
10. Router uses the configured suggester pseudo-model

---

## 7. Acceptance Criteria

- [ ] Auto-describe triggers on switch from vision → non-vision with `on_downgrade: "auto_describe"`
- [ ] Images described with `[IMAGEN_DESCRITA #N]` annotation tags
- [ ] Original history preserved in degradation_event turn
- [ ] `POST /degrade-images` works for manual degradation
- [ ] Router LLM evaluates complexity on configured pseudo-models
- [ ] Router suggestion appears in `proxy_metadata` without changing the model
- [ ] Router failure is non-blocking
- [ ] All 22+ tests pass
- [ ] No regression on Sprint 1-4 tests

---

## 8. Explicitly OUT OF SCOPE for Sprint 5

| Feature | Sprint |
|---|---|
| Explicit compaction (`POST /compact`) | 6 |
| Context alerts (60%, 80%, 100%) | 6 |
| Audit log | 6 |
| Celery async tasks | 6 |
| Provider cache optimization | 7 |
| OpenCode integration testing | 7 |
| Auth / CORS / Rate limiting | 8 |
| PDF degradation (extract text) | v2 (future) |
| Audio degradation (transcription) | v2 (future) |
| Video degradation (frame extraction) | v2 (future) |
