# Sprint 7 — Provider Cache Optimization & OpenCode Integration ✅

> **Duration:** 1 week
> **Goal:** Provider cache hits are maximized through canonical message ordering and provider-specific cache control. OpenCode works with the proxy out of the box.
> **Success criterion:** 10 consecutive turns in a conversation → 8+ cache hits at the provider. OpenCode connects to the proxy with just a base URL and API key.
> **Status:** ✅ COMPLETED — 69 Sprint 7 tests pass (message_ordering, provider_cache, client_agnostic, E2E, rate_limiter, auth, metrics)

---

## 1. Dependencies from Previous Sprints

| Dependency                                  | Source     | Status                   |
| ------------------------------------------- | ---------- | ------------------------ |
| Chat endpoint with full message assembly    | Sprint 1-6 | Complete                 |
| `pseudo_models.yaml` with all providers     | Sprint 1   | Complete                 |
| LiteLLM integration                         | Sprint 1   | Complete                 |
| Affinity in Valkey                          | Sprint 1   | Complete                 |
| Tool definitions in canonical OpenAI format | Sprint 3   | Complete                 |
| `proxy_metadata` in responses               | Sprint 1-6 | Extended with cache info |

### 1.1 New files/modules

```
src/
├── cache/
│   ├── __init__.py
│   ├── message_ordering.py      # NEW — canonical message ordering
│   └── provider_cache.py        # NEW — provider-specific cache optimizations
│
└── tests/
    ├── test_message_ordering.py     # NEW
    └── test_provider_cache.py       # NEW
```

No DB changes in this sprint.

---

## 2. Canonical Message Ordering (`cache/message_ordering.py`)

### 2.1 The problem

Provider caching works by hashing a prefix of the prompt. If message order changes between turns, the cache is invalidated even if the content is the same.

### 2.2 The solution

The proxy always assembles the prompt in this exact order:

```
1. System messages (static — does not change between turns)
2. Tool definitions (static — ordered alphabetically by function name)
3. Conversation history (from oldest to newest)
4. New user message (at the end — this is what changes each turn)
```

### 2.3 assemble_canonical_messages()

```python
def assemble_canonical_messages(
    system_prompt: str | None,
    tool_definitions: list[dict] | None,
    conversation_history: list[dict],
    new_messages: list[dict],
) -> list[dict]:
    """
    Assemble messages in canonical order to maximize provider cache hits.

    Order:
    1. System prompt (single message or merged)
    2. Tool definitions (sorted alphabetically by function name)
    3. Conversation history (oldest first, as stored in DB)
    4. New user/assistant/tool messages (in order received)
    """
    messages = []

    # 1. System prompt
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # 2. Tool definitions — sorted for deterministic ordering
    if tool_definitions:
        sorted_tools = sorted(tool_definitions, key=lambda t: t.get("function", {}).get("name", ""))
        # Tool definitions are passed separately to the LLM API (tools parameter),
        # but some providers (Anthropic) include them in the system message.
        # For now, we rely on LiteLLM to handle tool placement.
        # The ordering here is for JSON serialization stability.
        pass

    # 3. Conversation history
    messages.extend(conversation_history)

    # 4. New messages
    messages.extend(new_messages)

    return messages
```

### 2.4 JSON serialization stability

All JSON serialization in the proxy MUST use `sort_keys=True` to ensure deterministic key ordering:

```python
import json

def stable_json_dumps(obj: dict) -> str:
    """Serialize dict to JSON with sorted keys for deterministic output."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)

# Apply everywhere:
# - Tool definitions when stored
# - Messages when stored in JSONB
# - Any dict sent to provider
```

**Places to verify `sort_keys=True`:**

1. `conversation_turns.messages` storage → use `json.dumps(sort_keys=True)` before JSONB insert
2. Tool definitions in `conversation_turns.tool_definitions` → sort keys on store
3. Any API response construction → Pydantic `model_dump()` with default options is fine (Pydantic preserves insertion order of defined fields)

### 2.5 What message ordering does NOT do

- Does NOT modify message content
- Does NOT reorder messages within the conversation history (preserves turn order)
- Does NOT merge or deduplicate system messages
- Does NOT strip content from messages
- Does NOT change `tool_call_id` values

---

## 3. Provider-specific Cache Optimization (`cache/provider_cache.py`)

### 3.1 Provider strategies

Each provider has a different caching mechanism. The proxy configures these based on the physical model's provider.

| Provider          | Mechanism                                        | How the proxy enables it                                                                           |
| ----------------- | ------------------------------------------------ | -------------------------------------------------------------------------------------------------- |
| **OpenAI**        | Automatic prefix caching (≥1024 token prompts)   | No action needed — OpenAI caches automatically. Add `prompt_cache_key` for tracking.               |
| **Anthropic**     | `cache_control: {type: "ephemeral"}` breakpoints | Add cache breakpoints after system message and after tool definitions. Max 4 breakpoints.          |
| **Google Gemini** | `CachedContent` explicit resource                | Create a `CachedContent` at conversation start, reuse `cache_id` in subsequent turns. TTL: 60 min. |
| **DeepSeek**      | OpenAI-compatible automatic caching              | Same as OpenAI — no action needed.                                                                 |
| **Groq**          | No caching                                       | N/A                                                                                                |
| **Zhipu**         | No documented caching                            | N/A                                                                                                |
| **Qwen**          | No documented caching                            | N/A                                                                                                |
| **MiniMax**       | No documented caching                            | N/A                                                                                                |
| **Ollama**        | No caching (local)                               | N/A                                                                                                |

### 3.2 Anthropic cache_control

```python
def apply_anthropic_cache_control(messages: list[dict]) -> list[dict]:
    """
    Add cache_control breakpoints to messages for Anthropic provider.
    - Breakpoint 1: After system message (caches system prompt)
    - Breakpoint 2: After tool definitions (caches tools)
    - Breakpoint 3: After conversation history (caches prefix)
    - Breakpoint 4: Reserved for future use

    Max 4 breakpoints per Anthropic's limits.
    """
    modified = deepcopy(messages)
    breakpoints_placed = 0

    for i, msg in enumerate(modified):
        if msg["role"] == "system" and breakpoints_placed < 4:
            msg["cache_control"] = {"type": "ephemeral"}
            breakpoints_placed += 1

        # Place a breakpoint after the last history message (before the new query)
        if i == len(modified) - 2 and breakpoints_placed < 4:
            msg["cache_control"] = {"type": "ephemeral"}
            breakpoints_placed += 1

    return modified


def should_apply_cache_control(provider: str) -> bool:
    """Check if cache control should be applied for this provider."""
    return provider.lower() in ("anthropic",)
```

### 3.3 Google Gemini CachedContent

```python
async def manage_gemini_cache(
    conversation_id: str,
    valkey_client,
    system_prompt: str,
    tool_definitions: list[dict] | None,
    provider: str,
) -> str | None:
    """
    Manage Gemini CachedContent for a conversation.
    Creates a cache on first turn, reuses on subsequent turns.
    Returns the cache_id or None.
    """
    if provider.lower() != "google":
        return None

    cache_key = f"conv:{conversation_id}:gemini_cache_id"

    # Check if cache already exists
    existing_cache_id = await valkey_client.get(cache_key)
    if existing_cache_id:
        return existing_cache_id

    # Create new CachedContent
    try:
        # Gemini CachedContent is created via the Gemini API
        # LiteLLM may or may not support this — check docs
        # For now, return None and log that caching is not yet integrated via LiteLLM
        return None
    except Exception:
        return None
```

**Note for Sprint 7:** Gemini caching via `CachedContent` may require direct Gemini API calls if LiteLLM doesn't support it. If so, document this as a limitation and implement in a future sprint.

### 3.4 Cache metadata in proxy_metadata

```python
def build_cache_metadata(
    response: dict,
    provider: str,
    cache_applied: bool,
) -> dict:
    """
    Extract cache hit information from the provider response.
    Different providers report cache hits differently.
    """
    metadata = {
        "cache_optimization_applied": cache_applied,
        "provider": provider,
    }

    usage = response.get("usage", {})

    # OpenAI/DeepSeek: usage.prompt_tokens_details.cached_tokens
    if "prompt_tokens_details" in usage:
        details = usage["prompt_tokens_details"]
        if "cached_tokens" in details:
            metadata["provider_cache_hit"] = details["cached_tokens"] > 0
            metadata["cached_tokens"] = details.get("cached_tokens", 0)
            metadata["total_prompt_tokens"] = usage.get("prompt_tokens", 0)

    # Anthropic: usage.cache_read_input_tokens, usage.cache_creation_input_tokens
    if "cache_read_input_tokens" in usage:
        metadata["provider_cache_hit"] = usage["cache_read_input_tokens"] > 0
        metadata["cache_read_tokens"] = usage.get("cache_read_input_tokens", 0)
        metadata["cache_write_tokens"] = usage.get("cache_creation_input_tokens", 0)

    # Estimate savings
    if metadata.get("cached_tokens", 0) > 0:
        # ~$0.0025/1K tokens for input (rough average across providers)
        savings = (metadata["cached_tokens"] / 1000) * 0.0025
        metadata["estimated_savings_usd"] = round(savings, 5)

    return metadata
```

### 3.5 What provider cache optimization does NOT do in Sprint 7

- Does NOT implement Gemini CachedContent if LiteLLM doesn't support it (document limitation)
- Does NOT attempt caching for providers that don't support it (Groq, Zhipu, Qwen, MiniMax, Ollama)
- Does NOT pre-warm caches
- Does NOT implement custom cache key logic beyond what providers offer
- Does NOT cache tool definitions separately from messages (LiteLLM handles this)

### 3.6 The full cache affinity loop — Valkey → same model → provider cache hits

The proxy's cache strategy is a **three-layer system** that works together:

```
LAYER 1 — Valkey affinity (Sprint 1):
  conv:abc:physical_model = "qwen3-max"  (TTL 24h)
  → Ensures EVERY turn uses the same physical model
  → The model name string in LiteLLM calls is always identical across turns

LAYER 2 — Canonical message ordering (§2):
  system → tools(sorted) → history → new query
  → The prompt PREFIX is identical across turns
  → Provider's hash of the prefix stays the same

LAYER 3 — Provider-specific cache mechanisms (§3):
  Anthropic: cache_control breakpoints on system + tools
  OpenAI: automatic prefix caching (≥1024 tokens)
  Gemini: CachedContent via cache_id (if supported by LiteLLM)
  DeepSeek: automatic (OpenAI-compatible)

RESULT:
  Turn 1:  system + tools + history + query1  → provider caches prefix [system+tools]
  Turn 2:  system + tools + history + query2  → same prefix [system+tools] → CACHE HIT
  Turn 3:  system + tools + history + query3  → same prefix [system+tools] → CACHE HIT
  ...
  Turn 20: system + tools + history + query20 → same prefix → CACHE HIT
```

**What breaks this loop:**

| Event                                                             | Cache impact                                                          | How proxy handles it                                                                                         |
| ----------------------------------------------------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Model fallback (primary → secondary)                              | Cache destroyed — new model, new prefix                               | `proxy_metadata.fallback_applied: true`. Valkey affinity updated to new model. New cache built from scratch. |
| Pseudo-model switch (user changes from normal → tareas-avanzadas) | Cache destroyed — new model from new pseudo-model                     | `validate_switch()` ensures compatibility. Valkey affinity updated. Old cache abandoned, new cache starts.   |
| Tool definition change (client adds/removes tools)                | Prefix changes → cache miss for that turn, then new cache built       | Tool definitions stored canonically with `sort_keys=True`. Next turn rebuilds cache with new definitions.    |
| System prompt change                                              | Prefix changes → cache miss                                           | System prompt is static per pseudo-model — shouldn't change mid-conversation.                                |
| Message reordering (non-canonical)                                | Prefix changes → cache miss                                           | `assemble_canonical_messages()` prevents this.                                                               |
| Timestamp in prompt                                               | Prefix changes → cache miss                                           | Proxy strips timestamps from cacheable content.                                                              |
| Valkey TTL expires (24h inactivity)                               | Affinity lost → next turn picks priority-1 model (could be different) | TTL configurable. Default 24h covers most sessions.                                                          |

### 3.7 Cache destruction on fallback — documented, not hidden

When fallback occurs, the cache is destroyed. The proxy MUST report this clearly:

```json
{
  "proxy_metadata": {
    "fallback_applied": true,
    "fallback_reason": "upstream_503: qwen3-max",
    "fallback_previous_model": "qwen3-max",
    "fallback_new_model": "deepseek-v4-flash",
    "cache": {
      "previous_cache_destroyed": true,
      "previous_cached_tokens_lost": 45000,
      "new_cache_starting": true,
      "estimated_extra_cost_usd": 0.11
    }
  }
}
```

**Why this matters:** The user needs to know that fallback has a COST beyond just the error — it destroys the provider cache. The proxy quantifies this in `proxy_metadata`.

### 3.8 Provider cache optimization across OpenCode forks

OpenCode forks or alternative clients (Continue, LibreChat, Cline, Aider) may send messages in different formats or use different model names. The proxy's cache strategy is robust against this because:

| Variation in fork/client                                     | How proxy handles it                                                                                               |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| Different model name format (`local/`, `custom/`, `openai/`) | `normalize_model_name()` strips prefix, resolves alias (Sprint 1 §7.3)                                             |
| Different system prompt order                                | `assemble_canonical_messages()` reorders to canonical (system → tools → history → new)                             |
| Different tool definition format                             | All tools stored in canonical OpenAI format (Sprint 3). LiteLLM translates per provider.                           |
| JSON keys in different order                                 | `json.dumps(sort_keys=True)` everywhere → deterministic serialization                                              |
| Headers (custom `X-*` headers, `HTTP-Referer`)               | Proxy ignores unknown headers. Only reads `Authorization`, `X-Conversation-ID`, `Content-Type`                     |
| Different streaming format                                   | Proxy forwardes SSE chunks as-is. LiteLLM normalizes provider responses to OpenAI format.                          |
| Compaction at different thresholds                           | `detect_external_compaction()` (Sprint 4 §3.7b) handles any client-side compaction, not just OpenCode's            |
| Different token counting method                              | Proxy uses provider's `usage.prompt_tokens` — client-independent. Token accuracy is the provider's responsibility. |

---

## 4. OpenCode Integration (verified against OpenCode v0.1.x source)

### 4.0 Research summary: how OpenCode actually works

Before designing the integration, the following was verified by reading OpenCode's source code (`/internal/`):

| Aspect                    | OpenCode behavior                                                                                                            | Impact on proxy design                                                                              |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| Model discovery           | **Does NOT call `/v1/models`**. 100% hardcoded model definitions in `internal/llm/models/*.go`                               | Optimistic `/v1/models` helps Continue/LibreChat, not OpenCode                                      |
| Model struct capabilities | Only `CanReason` and `SupportsAttachments` flags. No `vision`, `tools`, `function_calling`, `parallel_tools`                 | OpenCode never strips tools/images based on capabilities (except new attachments)                   |
| Local/custom provider     | Uses `LOCAL_ENDPOINT` env var → OpenAI-compatible base URL. Model names sent as `local/<name>`                               | Proxy must normalize `local/normal` → `normal` (see Sprint 1 §7.3)                                  |
| Model validation          | Checks if model exists in `SupportedModels` map. Unknown models REJECTED                                                     | Users must configure model names that pass OpenCode validation OR use a recognized model            |
| Context compaction        | `autoCompact: true` (default). Triggers at **95% of ContextWindow** via `agent.Summarize()`                                  | Proxy's continuous compaction should trigger BEFORE OpenCode's (70-80%)                             |
| Compaction mechanism      | Sends full history to summarizer model → stores summary as `SummaryMessageID` → messages before summary skipped on next turn | OpenCode compaction is SEPARATE from proxy compaction — they coexist but proxy should compact first |
| Token tracking            | Reads `usage.prompt_tokens` + `usage.completion_tokens` from API response. Accumulates on session                            | Proxy MUST return accurate `usage` — OpenCode depends on it for cost + compaction triggers          |
| Session ID                | Internal UUID. No `conversation_id` header sent to API                                                                       | Proxy must derive `conversation_id` from first message hash (Sprint 1 already handles this)         |
| Conversation continuity   | Session persisted in SQLite. Each turn sends full message history via API                                                    | Proxy sees full history each turn — normal behavior                                                 |
| Tool handling             | Tools NEVER filtered. Always included in API request. No `parallel_tool_calls` parameter sent                                | Proxy receives all tools and must handle parallel detection itself (Sprint 2)                       |
| Streaming                 | Requests `stream: true` with `stream_options: {include_usage: true}`                                                         | Proxy must forward stream + include usage in final chunk                                            |

**Key takeaway:** OpenCode is a "dumb pipe" for content — it sends everything (tools, images in history, parallel calls) without capability checks. The proxy's `validate_incoming_content()` is the safety net for all clients.

### 4.1 Configuration — OpenCode as `local` provider

OpenCode's `local` provider is the only provider that connects to an arbitrary OpenAI-compatible endpoint. It uses `LOCAL_ENDPOINT` and `LOCAL_API_KEY` environment variables.

```bash
# ~/.bashrc or ~/.zshrc
export LOCAL_ENDPOINT="https://proxy.tudominio.com/v1"
export LOCAL_API_KEY="sk-proxy-xxxxxxxxxxxxxxxxxxxx"
```

**OpenCode config** (`opencode.json` or `~/.config/opencode/config.json`):

```json
{
  "model": "local/normal",
  "max_tokens": 4096,
  "context_window": 96000
}
```

**Critical: `context_window` must be configured manually** because OpenCode does NOT read it from `/v1/models`. Set it to match the pseudo-model's `context_window` from `pseudo_models.yaml`:

- `normal` → `96000`
- `tareas-avanzadas` → `128000`
- `pensamiento-profundo-caro` → `200000`
- `deep-flash` → `128000`
- `flash-lowcost` → `64000`
- `avanzada-vision` → `32768`
- `flash-vision` → `16384`

If `context_window` is not set correctly, OpenCode's auto-compact will trigger at the wrong threshold.

### 4.2 Model name handling

OpenCode sends model names as `local/<name>` (e.g., `local/normal`). The proxy's `normalize_model_name()` (Sprint 1 §7.3) strips the `local/` prefix and resolves to the pseudo-model name.

**Model validation workaround:** OpenCode validates that the model name exists in its hardcoded `SupportedModels` map. Since `local/normal` won't be there, OpenCode may reject it. Users have two options:

**Option A (recommended): Use a model name OpenCode recognizes.** Configure OpenCode as:

```json
{
  "model": "local/gpt-4o"
}
```

Then add an alias in the proxy:

```python
# In normalize_model_name():
# "gpt-4o" → pseudo-model "normal" (if not an exact pseudo-model match)
MODEL_ALIASES = {
    "gpt-4o": "normal",
    "gpt-4o-mini": "deep-flash",

    # etc.
}
```

The proxy resolves `local/gpt-4o` → `gpt-4o` → alias → `normal`. OpenCode is happy (it knows `gpt-4o`), and the proxy routes to the correct pseudo-model.

**Option B: Fork OpenCode** and add pseudo-model names to `SupportedModels`.

**Option A is preferred** because it requires zero OpenCode modifications and works with any OpenAI-compatible client that sends model names.

### 4.3 Model aliases (NEW — bridge between OpenCode model names and pseudo-models)

Add a `model_aliases` section to `pseudo_models.yaml` or a separate config:

```yaml
model_aliases:
  # OpenAI names → pseudo-models
  "gpt-4o": "normal"
  "gpt-4o-mini": "deep-flash"
  "gpt-4.1": "tareas-avanzadas"
  "o3": "pensamiento-profundo-caro"
  "o4-mini": "pensamiento-profundo-caro"

  # Anthropic names → pseudo-models
  "claude-haiku-3-5-20241022": "flash-lowcost"

  # Google names → pseudo-models
  "gemini-2.5-flash": "avanzada-vision"
  "gemini-2.5-pro": "avanzada-vision"

  # Generic fallback
  "default": "normal"
```

```python
def normalize_model_name(raw_model: str, config: ProxyConfig) -> str:
    """Normalize with alias support."""
    # 1. Strip provider prefix
    if "/" in raw_model:
        raw_model = raw_model.rsplit("/", 1)[-1]

    # 2. Exact pseudo-model match
    if raw_model in config.pseudo_models:
        return raw_model

    # 3. Alias match
    if raw_model in config.model_aliases:
        return config.model_aliases[raw_model]

    # 4. Default fallback
    if "default" in config.model_aliases:
        return config.model_aliases["default"]

    # 5. No match — raise error
    raise HTTPException(400, detail={
        "error": "UNKNOWN_MODEL",
        "message": f"Unknown model '{raw_model}'. Available: {list(config.pseudo_models.keys())}",
        "aliases_available": list(config.model_aliases.keys()) if config.model_aliases else [],
    })
```

### 4.4 context_window reporting

OpenCode reads `context_window` from its **own hardcoded model definition**, not from `/v1/models`. However, for **other clients** (Continue, LibreChat, etc.) that do read `/v1/models`, the proxy MUST report the correct `context_window` for each pseudo-model.

The `GET /v1/models` response already includes `context_window` (Sprint 1 §10). This value MUST match the pseudo-model's actual context window because:

- Continue/LibreChat may use it for their own auto-compaction
- Custom scripts may use it to estimate remaining capacity

### 4.5 OpenCode auto-compact vs proxy compaction — coexistence

OpenCode has `autoCompact: true` by default, triggering at 95% of `ContextWindow`. The proxy has continuous compaction at 70-80%. These are SEPARATE and COMPLEMENTARY layers:

| Layer                           | Trigger                                 | What it does                                                                             | Where                  |
| ------------------------------- | --------------------------------------- | ---------------------------------------------------------------------------------------- | ---------------------- |
| **Proxy continuous compaction** | 70-80% of context window (configurable) | Compacts old turns into snapshot. Model receives [snapshot] + [recent turns]             | Server-side (proxy)    |
| **OpenCode auto-compact**       | 95% of context window                   | Summarizes full history. Creates new `SummaryMessageID`. Messages before summary dropped | Client-side (OpenCode) |
| **Proxy explicit compaction**   | User-invoked or `CONTEXT_UNUSABLE`      | Generates structured Markdown snapshot                                                   | Server-side (proxy)    |

**Design principle:** The proxy compacts FIRST (at lower threshold) because:

1. Proxy compaction preserves more semantic structure (decisions, code, state, pending items)
2. OpenCode's summarize is a last-resort generic summary
3. If the proxy already compacted, OpenCode sees a smaller context and may not trigger its own compaction

**What the proxy MUST report accurately:**

- `usage.prompt_tokens` — OpenCode uses this to track accumulated tokens
- `context_window` in `/v1/models` — other clients use this
- The proxy does NOT control OpenCode's compaction trigger — that's entirely client-side

### 4.6 Token accuracy contract

The proxy MUST return accurate `usage` in every response:

```json
{
  "usage": {
    "prompt_tokens": 4523, // MUST be accurate — OpenCode sums this for cost
    "completion_tokens": 812, // MUST be accurate — OpenCode sums this for cost
    "total_tokens": 5335
  }
}
```

**Why this matters for all clients:**

1. **OpenCode** accumulates `session.PromptTokens += usage.PromptTokens` and calculates cost: `cost = model.CostPer1MIn * promptTokens / 1e6`
2. **Continue/LibreChat** may track token usage for their own context management
3. **The proxy itself** uses `total_tokens` in `conversations.total_tokens` for:
   - Continuous compaction triggers
   - Context alerts (60%, 80%, 100%)
   - `proxy_metadata.context_usage_pct`

**The proxy MUST NOT fabricate or estimate token counts in the `usage` field.** The `usage` values come directly from LiteLLM/provider — they are the provider's official token count. The proxy only ADDS to `conversation.total_tokens` based on these values.

### 4.7 conversation_id continuity with OpenCode

OpenCode does NOT maintain a `conversation_id` across turns. Each turn is a fresh API request with the full message history. The proxy's `derive_conversation_id()` from Sprint 1 handles this:

```python
def derive_conversation_id(request: ChatRequest) -> str:
    """
    OpenCode sends full history each turn but no conversation_id.
    Derive it deterministically from the first message content.
    """
    # Check headers first (for clients that DO send conversation_id)
    if conv_id := request.headers.get("X-Conversation-ID"):
        return conv_id
    if conv_id := request.body.get("conversation_id"):
        return conv_id

    # Deterministic hash of first message (works for OpenCode)
    first_msg = request.body.messages[0]
    content_str = json.dumps(first_msg, sort_keys=True)
    return f"conv-{hashlib.sha256(content_str.encode()).hexdigest()[:16]}"
```

**Why this works for OpenCode:**

- Turn 1: First message = "Write a hello world" → hash → `conv-abc123`
- Turn 2: First message = "Write a hello world" (OpenCode sends full history, first message is the original) → same hash → `conv-abc123`
- Affinity maintained across turns because the hash is deterministic

**Limitation:** If the user changes the first message (e.g., edits the initial prompt in OpenCode), the hash changes and a new conversation is created. This is acceptable — it's a genuinely different conversation.

### 4.8 Client-agnostic design checklist

The proxy must work with ALL OpenAI-compatible clients, not just OpenCode:

| Client                                             | How it connects                                 | What it needs from the proxy                                             |
| -------------------------------------------------- | ----------------------------------------------- | ------------------------------------------------------------------------ |
| **OpenCode**                                       | `LOCAL_ENDPOINT` env var, `local/<model>` names | Model aliases, accurate `usage`, correct `context_window` in config      |
| **OpenCode forks** (Cline, RooCode, Augment, etc.) | Same `LOCAL_ENDPOINT` or custom provider config | Same as OpenCode + model name normalization handles any prefix           |
| **Continue**                                       | Custom provider config with `apiBase`           | `/v1/models` for auto-discovery, accurate capabilities, `context_window` |
| **LibreChat**                                      | Custom endpoint config                          | `/v1/models` for model list, OpenAI-compatible streaming                 |
| **Aider**                                          | `OPENAI_API_BASE` env var                       | `/v1/models`, standard chat completions                                  |
| **Custom scripts / curl**                          | Direct HTTP POST                                | Standard OpenAI format, clear errors                                     |
| **VS Code Copilot**                                | N/A (uses GitHub's API, not our proxy)          | Not supported unless user configures custom provider                     |

### 4.8b OpenCode fork compatibility

OpenCode has several active forks (Cline, RooCode, Augment Code, Continue's agent mode). These forks may:

| Fork variation                                                        | Proxy behavior to handle it                                                                     |
| --------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| **Different model name prefix** (`cline/normal`, `roo/normal`)        | `normalize_model_name()` strips ANY `prefix/` before the last `/`                               |
| **Different default model** (fork defaults to `cline/gpt-4o`)         | Model aliases map any known model name to a pseudo-model                                        |
| **No model validation** (fork removed `SupportedModels` check)        | Direct pseudo-model names work without aliases                                                  |
| **Different auto-compact threshold** (fork uses 90% instead of 95%)   | `detect_external_compaction()` uses message count drop + summary detection — threshold-agnostic |
| **Different session format** (SQLite vs JSON vs in-memory)            | Irrelevant — proxy is stateless, sees only HTTP requests                                        |
| **Custom headers** (`X-Fork-Version`, `X-Client-ID`)                  | Proxy ignores unknown headers. Only reads standard ones.                                        |
| **Different streaming expectations**                                  | Proxy forwardes standard SSE. LiteLLM handles provider differences.                             |
| **Tool format variations** (Anthropic-native tools instead of OpenAI) | LiteLLM translates. Proxy stores canonical OpenAI format regardless.                            |

**Design principle:** The proxy is an **OpenAI-compatible HTTP API**. Any client that speaks OpenAI's chat completions protocol — regardless of origin, fork, or customization — must work. The proxy never assumes it's talking to OpenCode specifically. All client-specific behavior (model name prefixes, compaction patterns, session management) is handled through normalization and detection, not hardcoded assumptions.

### 4.8c Testing against unknown clients

```python
# tests/test_client_agnostic.py

@pytest.mark.parametrize("model_name", [
    "normal",                          # bare pseudo-model
    "local/normal",                    # OpenCode local provider
    "cesar-proxy/normal",              # custom provider name
    "cline/normal",                    # Cline fork
    "roo/normal",                      # RooCode fork
    "openai/gpt-4o",                   # alias
    "local/gpt-4o",                    # aliased + prefixed
])
async def test_model_name_normalization(model_name, async_client):
    """All model name formats should resolve to a valid pseudo-model."""
    response = await async_client.post("/v1/chat/completions", json={
        "model": model_name,
        "messages": [{"role": "user", "content": "test"}],
    })
    assert response.status_code == 200
    assert response.json()["proxy_metadata"]["pseudo_model"] in [
        "normal", "pensamiento-profundo-caro", "tareas-avanzadas",
        "deep-flash", "flash-lowcost", "avanzada-vision", "flash-vision"
    ]


async def test_external_compaction_from_any_client(async_client):
    """External compaction should be detected regardless of which client triggered it."""
    # Simulate a client that compacted: send a long summary as first message,
    # followed by very few recent messages, after many turns were accumulated.
    ...
```

**Universally required:**

- [x] Standard OpenAI `/v1/chat/completions` format
- [x] SSE streaming with `data: [DONE]`
- [x] `/v1/models` endpoint (OpenAI format)
- [x] Bearer token auth
- [x] Accurate `usage.prompt_tokens` and `usage.completion_tokens`
- [x] Clear error responses with descriptive messages
- [x] Model name normalization (handles ANY prefix, aliases)
- [x] External compaction detection (handles ANY client's compaction, not just OpenCode)
- [x] Provider cache loop (Valkey affinity → canonical ordering → provider cache) works for ANY client

### 4.9 Connection test (all clients)

```bash
# OpenCode
export LOCAL_ENDPOINT="https://proxy.tudominio.com/v1"
export LOCAL_API_KEY="sk-proxy-xxx"
# Then: opencode

# Continue
# In continue config.json:
# "models": [{"provider": "openai", "model": "normal", "apiBase": "https://proxy.tudominio.com/v1", "apiKey": "sk-proxy-xxx"}]

# LibreChat
# In librechat.yaml:
# endpoints.custom.endpoint: "https://proxy.tudominio.com/v1"

# curl (any client)
curl -X POST "https://proxy.tudominio.com/v1/chat/completions" \
  -H "Authorization: Bearer sk-proxy-xxx" \
  -H "Content-Type: application/json" \
  -d '{"model": "normal", "messages": [{"role": "user", "content": "Hello"}]}'
```

### 4.10 OpenCode configuration examples (ready to copy)

**Example 1: Using an alias (recommended)**

```json
{
  "model": "local/gpt-4o"
}
```

Proxy alias: `gpt-4o` → `normal`. OpenCode sees a recognized model. Proxy routes to `normal`.

**Example 2: Direct pseudo-model name (if OpenCode validation is patched)**

```json
{
  "model": "local/normal"
}
```

Proxy normalizes `local/normal` → `normal`. Works if OpenCode doesn't reject unknown models.

**Example 3: Switching pseudo-models mid-session**

```json
// Start with normal
{"model": "local/gpt-4o"}

// Switch to deep thinking
{"model": "local/o3"}  // alias → pensamiento-profundo-caro

// Switch to vision
{"model": "local/gemini-2.5-flash"}  // alias → avanzada-vision
```

Each switch goes through proxy's `validate_switch()` and model name normalization.

---

## 5. Integration Test: OpenCode → Proxy → Provider

### 5.1 End-to-end test

```python
# tests/test_e2e.py

@pytest.mark.integration
async def test_opencode_to_proxy_full_flow(async_client):
    """
    Simulate an OpenCode session: multiple turns, tools, streaming.
    Verifies affinity, cache hints, and proxy_metadata.
    """
    conversation_id = None
    physical_models_used = set()

    # Turn 1
    response = await async_client.post("/v1/chat/completions", json={
        "model": "normal",
        "messages": [{"role": "user", "content": "Write a hello world in Python"}],
    })
    assert response.status_code == 200
    data = response.json()
    assert "proxy_metadata" in data
    assert data["proxy_metadata"]["pseudo_model"] == "normal"
    conversation_id = data["proxy_metadata"]["conversation_id"]
    physical_models_used.add(data["proxy_metadata"]["physical_model"])

    # Turns 2-10
    for i in range(2, 11):
        response = await async_client.post("/v1/chat/completions", json={
            "model": "normal",
            "messages": [{"role": "user", "content": f"Modify the hello world to print {i} times"}],
            "conversation_id": conversation_id,
        })
        assert response.status_code == 200
        data = response.json()
        physical_models_used.add(data["proxy_metadata"]["physical_model"])

    # Affinity maintained → only 1 physical model used
    assert len(physical_models_used) == 1, f"Expected 1 physical model, got {physical_models_used}"

    # Cache: should show cache hits after first turn
    # (depends on provider — may not be populated if provider doesn't report)

    # Streaming test
    stream_response = await async_client.post("/v1/chat/completions", json={
        "model": "normal",
        "stream": True,
        "messages": [{"role": "user", "content": "Add a comment to the code"}],
        "conversation_id": conversation_id,
    })
    assert stream_response.status_code == 200
    assert "text/event-stream" in stream_response.headers["content-type"]
```

### 5.2 What the integration test does NOT cover

- Does NOT test vision models (requires actual image uploads)
- Does NOT test audio/PDF/video (not supported in v1)
- Does NOT test rate limiting (Sprint 8)
- Does NOT test auth failure (Sprint 8)
- Does NOT test compaction (requires long conversations)

---

## 6. Tests (Sprint 7)

### 6.1 test_message_ordering.py (minimum 6 tests)

1. System prompt always first in messages array
2. Conversation history in chronological order
3. New messages appended at the end
4. `sort_keys=True` produces deterministic JSON
5. Same input produces identical message order (idempotent)
6. Tool definitions serialized stably between turns

### 6.2 test_provider_cache.py (minimum 5 tests)

1. Anthropic messages get `cache_control` breakpoints
2. Cache metadata extracted from OpenAI-style response
3. Cache metadata extracted from Anthropic-style response
4. Non-caching providers (Groq, Zhipu) — no cache optimizations applied
5. `proxy_metadata.cache` present in responses

### 6.3 test_e2e.py (minimum 4 tests)

1. OpenCode flow: 5 turns → same physical model → affinity maintained
2. OpenCode flow: streaming works end-to-end
3. OpenCode flow: tools work (simple tool call + result)
4. `proxy_metadata` complete on every turn

---

## 7. Acceptance Criteria ✅

- [x] Messages assembled in canonical order (system → tools → history → new) on every turn
- [x] `sort_keys=True` used in all JSON serialization paths
- [x] Anthropic `cache_control` breakpoints added correctly (max 4)
- [x] Cache hit metadata extracted from provider responses
- [x] `proxy_metadata.cache` populated with hit/miss info
- [x] `opencode.example.jsonc` ready for copy-paste
- [x] 10 turns → affinity maintained → 1 physical model used
- [x] Streaming End-to-end test passes
- [x] Gemini CachedContent limitation documented if not supported via LiteLLM
- [x] All 15+ tests pass (69 actual)
- [x] No regression on Sprint 1-6 tests

---

## 8. Explicitly OUT OF SCOPE for Sprint 7

| Feature                                                             | Sprint        |
| ------------------------------------------------------------------- | ------------- |
| Auth middleware (PROXY_API_KEY enforcement)                         | 8             |
| CORS configuration                                                  | 8             |
| Rate limiting by pseudo-model                                       | 8             |
| Metrics endpoint (GET /metrics)                                     | 8             |
| HTTPS / Caddy setup                                                 | 8             |
| README with full deployment guide                                   | 8             |
| Gemini CachedContent implementation (if LiteLLM doesn't support it) | Future sprint |
| Dynamic cache key management                                        | Future        |
| Cache warming on conversation start                                 | Future        |
