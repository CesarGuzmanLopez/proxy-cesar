# Sprint 1 — MVP: Pseudo-models + Affinity + Streaming

> **Duration:** 2 weeks
> **Goal:** The proxy works end-to-end. `"normal"` maps to a physical model. Affinity is maintained across turns. Streaming works correctly. Fallback handles upstream errors.
> **Success criterion:** 20 consecutive turns in the same conversation → always the same physical model. Streaming works with `stream: true`. Fallback works when the primary model is down.

---

## 1. Pre-requisites and Dependencies

### 1.1 What must exist before this sprint starts

| Dependency | Status | Notes |
|---|---|---|
| Python 3.12+ installed | Required | `pyproject.toml` targets >=3.12 |
| PostgreSQL 16+ running | Required | Local or Docker. Connection string in `.env` |
| Valkey 7+ running | Required | Local or Docker. Connection string in `.env` |
| LiteLLM API keys for all providers | Required | Set via env vars: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `DEEPSEEK_API_KEY`, `GROQ_API_KEY`, `ZHIPU_API_KEY` |
| Poetry or uv for dependency management | Required | Pick one. All deps pinned with exact versions |

### 1.2 What is NOT needed yet (deferred to later sprints)

- PostgreSQL schema beyond `conversations` and `conversation_turns` basic tables
- Capability detection (Sprint 2)
- Compatibility validation on pseudo-model switch (Sprint 2)
- Tool normalization / canonical format storage (Sprint 3)
- Any compaction logic (Sprint 4)
- Image handling / auto-describe (Sprint 5)
- Router LLM (Sprint 5)
- Rate limiting (Sprint 8)
- CORS (Sprint 8)
- Auth middleware (Sprint 8 — `PROXY_API_KEY` may be stubbed as a constant or env var for local dev but not enforced)
- Metrics endpoint (Sprint 8)
- HTTPS / Caddy (Sprint 8)
- Celery (Sprint 6)

### 1.3 What this sprint depends on from external systems

- **LiteLLM** must be installable via `pip` and must support all listed providers
- **Valkey** must accept `SET key value EX ttl` and `GET key` commands
- **PostgreSQL** must accept `asyncpg` connections
- Provider APIs must be reachable from the dev machine (no proxy/VPN issues)

---

## 2. Project Scaffolding

### 2.1 Directory structure to create

```
proxy/
├── pyproject.toml              # Poetry/uv project file
├── pseudo_models.yaml          # Source of truth for pseudo-models (see §3)
├── alembic.ini                 # Alembic config (migrations stub for Sprint 1)
├── .env.example                # Template for required env vars
├── README.md                   # Minimal: how to install and run
│
├── src/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app, lifespan, middleware registration
│   ├── config.py               # Load & validate pseudo_models.yaml + .env
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── pseudo_model.py     # Pydantic schemas (see §5)
│   │   └── db.py               # SQLAlchemy ORM: Conversation, Turn (bare minimum)
│   │
│   ├── cache/
│   │   ├── __init__.py
│   │   └── affinity.py         # Valkey affinity ops (see §8)
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── chat.py             # POST /v1/chat/completions (see §9)
│   │   ├── models.py           # GET /v1/models (see §10)
│   │   └── health.py           # GET /health (see §11)
│   │
│   └── db/
│       ├── __init__.py
│       ├── session.py          # Async SQLAlchemy engine + session factory
│       └── migrations/
│           └── versions/       # Alembic migration files
│
└── tests/
    ├── __init__.py
    ├── conftest.py             # Fixtures: test client, mock DB, mock Valkey
    ├── test_config.py          # Pseudo-model YAML validation
    ├── test_affinity.py        # Valkey affinity read/write
    ├── test_chat.py            # POST /v1/chat/completions
    ├── test_streaming.py       # SSE streaming
    ├── test_fallback.py        # Fallback within pseudo-model
    └── test_models_endpoint.py # GET /v1/models, GET /health
```

### 2.2 pyproject.toml — exact dependencies

```toml
[project]
name = "proxy-cesar"
version = "0.1.0"
description = "Deterministic multi-model proxy for LLMs"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115,<0.116",
    "uvicorn[standard]>=0.34,<0.35",
    "litellm>=1.55,<1.56",
    "pydantic>=2.10,<2.11",
    "pydantic-settings>=2.7,<2.8",
    "sqlalchemy[asyncio]>=2.0,<2.1",
    "asyncpg>=0.30,<0.31",
    "alembic>=1.14,<1.15",
    "valkey>=6.0,<6.1",          # Valkey Python client (redis-compatible)
    "httpx>=0.28,<0.29",
    "pyyaml>=6.0,<6.1",
    "python-dotenv>=1.0,<1.1",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3,<8.4",
    "pytest-asyncio>=0.24,<0.25",
    "pytest-mock>=3.14,<3.15",
    "httpx>=0.28,<0.29",          # For TestClient
    "fakeredis[lua]>=2.26,<2.27", # Mock Valkey in tests
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

**Important:** ALL dependencies must be MIT, BSD, or Apache 2.0 licensed. Run `pip-licenses` or equivalent on the lockfile before committing. Reject any dependency with a non-free license.

### 2.3 .env.example

```bash
# Proxy
PROXY_PORT=9110

# Database
DATABASE_URL=postgresql+asyncpg://proxy:proxy@localhost:5432/proxy_db

# Cache
VALKEY_URL=valkey://localhost:6379

# Provider API keys
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxx
OPENAI_API_KEY=sk-xxxxxxxxxxxx
GOOGLE_API_KEY=AIza-xxxxxxxxxxxx
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxx
GROQ_API_KEY=gsk_xxxxxxxxxxxx
ZHIPU_API_KEY=xxxxxxxxxxxx
```

**DO NOT commit `.env` or any file containing real API keys.** Add `.env` to `.gitignore`.

---

## 3. Pseudo-models Configuration (`pseudo_models.yaml`)

### 3.1 Exact schema (Pydantic validation at startup)

The file must define all 8 pseudo-models. The proxy MUST refuse to start if validation fails (no partial loading, no defaults, no silent skips).

```yaml
pseudo_models:
  pensamiento-profundo-caro:
    display_name: "Pensamiento Profundo"
    description: "Razonamiento de máximo nivel. Bugs imposibles, arquitectura compleja."
    input_token_threshold: 32000
    context_window: 200000
    continuous_compaction:
      enabled: true
      trigger_pct: 70
      compact_preserve_recent: 16000
    pre_compaction:
      enabled: true
      threshold: 32000
      target_tokens: 8000
      compactor: "deep-flash"
    router_llm:
      enabled: true
      suggester: "flash-lowcost"
      suggest_on_downgrade_only: true
    image_handling:
      on_downgrade: "auto_describe"
    physical_models:
      - provider: zhipu
        model: glm-5.1
        openai_tools_compatible: true
        tools_strict: false
        parallel_tools: false
        vision: false
      - provider: deepseek
        model: deepseek-v4-pro
        openai_tools_compatible: true
        tools_strict: true
        parallel_tools: true
        vision: false
    fallback_strategy: sequential

  tareas-avanzadas:
    display_name: "Tareas Avanzadas"
    description: "Desarrollo profundo y sostenido. Features largas, debugging serio."
    input_token_threshold: 64000
    context_window: 128000
    continuous_compaction:
      enabled: true
      trigger_pct: 75
      compact_preserve_recent: 32000
    pre_compaction:
      enabled: false
    router_llm:
      enabled: false
    image_handling:
      on_downgrade: "block"
    physical_models:
      - provider: deepseek
        model: deepseek-v4-pro
        openai_tools_compatible: true
        tools_strict: true
        parallel_tools: true
        vision: false
      - provider: deepseek
        model: deepseek-v4-flash
        openai_tools_compatible: true
        tools_strict: false
        parallel_tools: true
        vision: false
      - provider: minimax
        model: minimax-m2.5
        openai_tools_compatible: true
        tools_strict: false
        parallel_tools: false
        vision: false
    fallback_strategy: sequential

  avanzada-vision:
    display_name: "Visión Avanzada"
    description: "Análisis visual de alto nivel. Diseño UI, OCR complejo, diagramas."
    input_token_threshold: 32000
    context_window: 32768
    continuous_compaction:
      enabled: false
    pre_compaction:
      enabled: false
    router_llm:
      enabled: false
    image_handling:
      on_downgrade: "auto_describe"
    physical_models:
      - provider: google
        model: gemini-3.5-flash
        openai_tools_compatible: true
        tools_strict: false
        parallel_tools: false
        vision: true
      - provider: groq
        model: meta-llama/llama-4-scout-17b-16e-instruct
        openai_tools_compatible: true
        tools_strict: false
        parallel_tools: false
        vision: true
    fallback_strategy: sequential

  normal:
    display_name: "Normal"
    description: "Punto de entrada recomendado. Coding extenso, trabajo agéntico."
    input_token_threshold: 96000
    context_window: 96000
    continuous_compaction:
      enabled: true
      trigger_pct: 80
      compact_preserve_recent: 32768
    pre_compaction:
      enabled: false
    router_llm:
      enabled: false
    image_handling:
      on_downgrade: "block"
    physical_models:
      - provider: qwen
        model: qwen3-max
        openai_tools_compatible: true
        tools_strict: false
        parallel_tools: false
        vision: false
      - provider: deepseek
        model: deepseek-v4-flash
        openai_tools_compatible: true
        tools_strict: true
        parallel_tools: true
        vision: false
    fallback_strategy: sequential

  deep-flash:
    display_name: "Deep Flash"
    description: "Velocidad y costo mínimo para tareas masivas y simples."
    input_token_threshold: 128000
    context_window: 128000
    continuous_compaction:
      enabled: false
    pre_compaction:
      enabled: false
    router_llm:
      enabled: false
    image_handling:
      on_downgrade: "block"
    physical_models:
      - provider: zhipu
        model: glm-4.5-flash
        openai_tools_compatible: true
        tools_strict: false
        parallel_tools: false
        vision: false
      - provider: groq
        model: openai/gpt-oss-20b
        openai_tools_compatible: true
        tools_strict: false
        parallel_tools: false
        vision: false
      - provider: deepseek
        model: deepseek-v4-flash
        openai_tools_compatible: true
        tools_strict: true
        parallel_tools: true
        vision: false
    fallback_strategy: sequential

  flash-lowcost:
    display_name: "Flash Lowcost"
    description: "Sub-agentes baratos. Clasificación, extracción, parsing."
    input_token_threshold: 64000
    context_window: 64000
    continuous_compaction:
      enabled: false
    pre_compaction:
      enabled: false
    router_llm:
      enabled: false
    image_handling:
      on_downgrade: "block"
    physical_models:
      - provider: zhipu
        model: glm-4.5-flash
        openai_tools_compatible: true
        tools_strict: false
        parallel_tools: false
        vision: false
      - provider: qwen
        model: qwen3.5-plus
        openai_tools_compatible: true
        tools_strict: false
        parallel_tools: false
        vision: false
      - provider: ollama
        model: ollama/llama3.2
        openai_tools_compatible: true
        tools_strict: false
        parallel_tools: false
        vision: false
    fallback_strategy: sequential

  flash-vision:
    display_name: "Flash Vision"
    description: "Visión rápida y barata. OCR ligero, screenshots."
    input_token_threshold: 16000
    context_window: 16384
    continuous_compaction:
      enabled: false
    pre_compaction:
      enabled: false
    router_llm:
      enabled: false
    image_handling:
      on_downgrade: "auto_describe"
    physical_models:
      - provider: google
        model: gemini-3.5-flash
        openai_tools_compatible: true
        tools_strict: false
        parallel_tools: false
        vision: true
      - provider: ollama
        model: ollama/llava
        openai_tools_compatible: true
        tools_strict: false
        parallel_tools: false
        vision: true
    fallback_strategy: sequential

  compactador:
    display_name: "Compactador"
    description: "Operación de compactación explícita. No es modo conversacional."
    input_token_threshold: null
    context_window: null
    continuous_compaction:
      enabled: false
    pre_compaction:
      enabled: false
    router_llm:
      enabled: false
    image_handling:
      on_downgrade: "auto_describe"
    physical_models:
      - provider: google
        model: gemini-3.5-flash
        context_window: 1000000
        openai_tools_compatible: true
        vision: true
      - provider: anthropic
        model: claude-haiku-4-5-20251001
        context_window: 200000
        openai_tools_compatible: true
        vision: false
      - provider: zhipu
        model: glm-4.5-flash
        context_window: 128000
        openai_tools_compatible: true
        vision: false
    fallback_strategy: by_context_window
```

### 3.2 Validation rules at startup

The proxy MUST fail to start (non-zero exit) with a clear error message if ANY of these conditions are not met:

1. **`pseudo_models.yaml` is not parseable YAML** → `FATAL: pseudo_models.yaml is not valid YAML: <parser error>`
2. **`pseudo_models` key is missing or not a dict** → `FATAL: pseudo_models must be a mapping at root level`
3. **A pseudo-model name contains non-alphanumeric characters other than `-`** → `FATAL: Invalid pseudo-model name: '<name>'. Only alphanumeric and hyphens allowed.`
4. **`display_name` is empty or missing** → `FATAL: pseudo_model '<id>' is missing display_name`
5. **`physical_models` is empty or missing** → `FATAL: pseudo_model '<id>' has no physical_models. At least one required.`
6. **Any `physical_model.model` is empty or missing** → `FATAL: pseudo_model '<id>' physical_model at index <N> is missing 'model' field`
7. **Any `physical_model.openai_tools_compatible` is `false`** → `FATAL: pseudo_model '<id>' physical_model '<model>' has openai_tools_compatible: false. All models must be true.`
8. **`fallback_strategy` is not one of `sequential` or `by_context_window`** → `FATAL: pseudo_model '<id>' has invalid fallback_strategy: '<value>'. Must be 'sequential' or 'by_context_window'.`
9. **`image_handling.on_downgrade` is not one of `auto_describe` or `block`** → `FATAL: pseudo_model '<id>' has invalid image_handling.on_downgrade: '<value>'. Must be 'auto_describe' or 'block'.`
10. **`pre_compaction.enabled` is `true` but `pre_compaction.threshold` is missing or `null`** → `FATAL: pseudo_model '<id>' has pre_compaction enabled but threshold is missing.`
11. **`pre_compaction.enabled` is `true` but `pre_compaction.compactor` does not reference a valid pseudo-model name** → `FATAL: pre_compaction.compactor '<name>' references unknown pseudo-model.`
12. **`continuous_compaction.enabled` is `true` but `trigger_pct` is missing or not an integer 1-100** → `FATAL: continuous_compaction.trigger_pct must be 1-100.`
13. **`router_llm.enabled` is `true` but `suggester` does not reference a valid pseudo-model name** → `FATAL: router_llm.suggester '<name>' references unknown pseudo-model.`
14. **Any field not defined in the Pydantic schema is present** → `FATAL: Extra field '<field>' in pseudo_model '<id>'. Pydantic extra='forbid' is on.`

### 3.3 How `config.py` must work

```python
# src/config.py — pseudocode of exact behavior

from pathlib import Path
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings
import yaml

# --- Physical model ---
class PhysicalModel(BaseModel, extra="forbid"):
    provider: str                                    # metadata only — not used to construct model ID
    model: str                                       # EXACT LiteLLM model ID — no prefix, no transform
    openai_tools_compatible: bool = True             # MUST be True (validated post-load)
    tools_strict: bool = False
    parallel_tools: bool = False
    vision: bool = False
    context_window: int | None = None                # Optional override for compactador
    note: str | None = None

# --- Compaction config ---
class PreCompactionConfig(BaseModel, extra="forbid"):
    enabled: bool = False
    threshold: int | None = None
    target_tokens: int | None = None
    compactor: str | None = None                     # References another pseudo-model name

class ContinuousCompactionConfig(BaseModel, extra="forbid"):
    enabled: bool = False
    trigger_pct: int | None = None                    # 1-100
    compact_preserve_recent: int | None = None

# --- Router LLM config ---
class RouterLLMConfig(BaseModel, extra="forbid"):
    enabled: bool = False
    suggester: str | None = None
    suggest_on_downgrade_only: bool = True

# --- Image handling ---
class ImageHandlingConfig(BaseModel, extra="forbid"):
    on_downgrade: str = "block"  # "auto_describe" or "block"

# --- Pseudo-model ---
class PseudoModel(BaseModel, extra="forbid"):
    display_name: str
    description: str
    input_token_threshold: int | None
    context_window: int | None
    continuous_compaction: ContinuousCompactionConfig = Field(default_factory=ContinuousCompactionConfig)
    pre_compaction: PreCompactionConfig = Field(default_factory=PreCompactionConfig)
    router_llm: RouterLLMConfig = Field(default_factory=RouterLLMConfig)
    image_handling: ImageHandlingConfig = Field(default_factory=ImageHandlingConfig)
    physical_models: list[PhysicalModel]
    fallback_strategy: str = "sequential"

    @field_validator("display_name")
    @classmethod
    def non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("display_name must not be empty")
        return v

    @field_validator("fallback_strategy")
    @classmethod
    def valid_strategy(cls, v: str) -> str:
        if v not in ("sequential", "by_context_window"):
            raise ValueError(f"fallback_strategy must be 'sequential' or 'by_context_window', got '{v}'")
        return v

    @model_validator(mode="after")
    def validate_cross_dependencies(self):
        if self.pre_compaction.enabled:
            if self.pre_compaction.threshold is None:
                raise ValueError("pre_compaction.threshold is required when enabled")
            if self.pre_compaction.compactor is None:
                raise ValueError("pre_compaction.compactor is required when enabled")
        if self.continuous_compaction.enabled:
            if self.continuous_compaction.trigger_pct is None:
                raise ValueError("continuous_compaction.trigger_pct is required when enabled")
            if not (1 <= self.continuous_compaction.trigger_pct <= 100):
                raise ValueError("continuous_compaction.trigger_pct must be between 1 and 100")
        return self

# --- Top-level config ---
class ProxyConfig(BaseModel, extra="forbid"):
    pseudo_models: dict[str, PseudoModel]
    model_aliases: dict[str, str] = Field(default_factory=dict)  # {"gpt-4o": "normal", "o3": "pensamiento-profundo-caro", ...}

    @model_validator(mode="after")
    def validate_all_models_tools_compatible(self):
        for name, pm in self.pseudo_models.items():
            for i, phys in enumerate(pm.physical_models):
                if not phys.openai_tools_compatible:
                    raise ValueError(
                        f"pseudo_model '{name}' physical_model[{i}] '{phys.model}' "
                        f"has openai_tools_compatible: false. All models must be true."
                    )
        return self

    @model_validator(mode="after")
    def validate_compactor_references(self):
        """Ensure pre_compaction.compactor and router_llm.suggester reference real pseudo-models."""
        for name, pm in self.pseudo_models.items():
            if pm.pre_compaction.enabled and pm.pre_compaction.compactor:
                if pm.pre_compaction.compactor not in self.pseudo_models:
                    raise ValueError(
                        f"pseudo_model '{name}' pre_compaction.compactor '{pm.pre_compaction.compactor}' "
                        f"references unknown pseudo-model"
                    )
            if pm.router_llm.enabled and pm.router_llm.suggester:
                if pm.router_llm.suggester not in self.pseudo_models:
                    raise ValueError(
                        f"pseudo_model '{name}' router_llm.suggester '{pm.router_llm.suggester}' "
                        f"references unknown pseudo-model"
                    )
        return self


# --- Loader ---
def load_config(path: Path = Path("pseudo_models.yaml")) -> ProxyConfig:
    """Load and validate pseudo_models.yaml. Raises SystemExit(1) on any error."""
    try:
        with open(path) as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"FATAL: {path} is not valid YAML: {e}", file=sys.stderr)
        raise SystemExit(1)
    except FileNotFoundError:
        print(f"FATAL: {path} not found", file=sys.stderr)
        raise SystemExit(1)

    try:
        return ProxyConfig.model_validate(raw)
    except ValidationError as e:
        print(f"FATAL: {path} validation failed:", file=sys.stderr)
        for error in e.errors():
            loc = " -> ".join(str(p) for p in error["loc"])
            print(f"  - {loc}: {error['msg']}", file=sys.stderr)
        raise SystemExit(1)
```

### 3.4 What config.py does NOT do in Sprint 1

- Does NOT validate that models actually exist in LiteLLM (that's a runtime check on first use)
- Does NOT validate provider API keys are present (those are LiteLLM's concern)
- Does NOT load or validate rate limit config (Sprint 8)
- Does NOT load CORS config (Sprint 8)
- Does NOT handle hot-reload of config (file is loaded once at startup)

---

## 4. Database Layer (bare minimum)

### 4.1 What tables to create

**ONLY these two tables in Sprint 1:**

#### `conversations`

| Column | Type | Constraints | Notes |
|---|---|---|---|
| `id` | UUID | PK, default `gen_random_uuid()` | Conversation ID (same as `conversation_id` in API) |
| `created_at` | TIMESTAMPTZ | NOT NULL, default `NOW()` | |
| `updated_at` | TIMESTAMPTZ | NOT NULL, default `NOW()`, auto-update | |
| `pseudo_model` | VARCHAR(128) | NOT NULL | Pseudo-model name from config |
| `physical_model` | VARCHAR(256) | NOT NULL | Exact LiteLLM model ID currently pinned |
| `total_tokens` | BIGINT | NOT NULL, default 0 | Accumulated tokens across all turns |

#### `conversation_turns`

| Column | Type | Constraints | Notes |
|---|---|---|---|
| `id` | UUID | PK, default `gen_random_uuid()` | |
| `conversation_id` | UUID | FK → conversations.id, NOT NULL | |
| `turn_number` | INTEGER | NOT NULL | Sequential turn number within conversation |
| `pseudo_model` | VARCHAR(128) | NOT NULL | Pseudo-model used for this turn |
| `physical_model` | VARCHAR(256) | NOT NULL | Physical model used for this turn |
| `input_tokens` | INTEGER | NOT NULL, default 0 | |
| `output_tokens` | INTEGER | NOT NULL, default 0 | |
| `messages` | JSONB | NOT NULL | Full messages array sent to model (OpenAI format) |
| `response` | JSONB | NULLABLE | Full response from model |
| `fallback_applied` | BOOLEAN | NOT NULL, default false | |
| `fallback_reason` | VARCHAR(256) | NULLABLE | |
| `created_at` | TIMESTAMPTZ | NOT NULL, default `NOW()` | |

**Columns intentionally deferred to Sprint 2:**
- `turn_type` (normal/compaction_snapshot/degradation_event/normalization_event)
- `had_images`, `had_tools`, `had_parallel_tools`
- Any capability flags on `conversations` table
- `active_snapshot_id`
- `user_id` (no auth in Sprint 1)

### 4.2 Alembic migration

```bash
alembic init -t async alembic
# Edit alembic.ini → sqlalchemy.url = DATABASE_URL from env
# Edit alembic/env.py → use async engine, target_metadata = Base.metadata
alembic revision --autogenerate -m "initial conversations and turns"
alembic upgrade head
```

### 4.3 SQLAlchemy models

```python
# src/models/db.py

from sqlalchemy import Column, String, Integer, BigInteger, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func
import uuid
from datetime import datetime

class Base(DeclarativeBase):
    pass

class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    pseudo_model: Mapped[str] = mapped_column(String(128), nullable=False)
    physical_model: Mapped[str] = mapped_column(String(256), nullable=False)
    total_tokens: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)

    turns = relationship("ConversationTurn", back_populates="conversation", order_by="ConversationTurn.turn_number")

class ConversationTurn(Base):
    __tablename__ = "conversation_turns"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False)
    turn_number: Mapped[int] = mapped_column(Integer, nullable=False)
    pseudo_model: Mapped[str] = mapped_column(String(128), nullable=False)
    physical_model: Mapped[str] = mapped_column(String(256), nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    messages: Mapped[dict] = mapped_column(JSONB, nullable=False)
    response: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    fallback_applied: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    fallback_reason: Mapped[str | None] = mapped_column(String(256), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    conversation = relationship("Conversation", back_populates="turns")
```

### 4.4 What the DB layer does NOT do in Sprint 1

- No capability flags accumulation (Sprint 2)
- No snapshot table or snapshot logic (Sprint 4)
- No tool-related columns (Sprint 3)
- No user/auth tables (Sprint 8)
- No audit log table (Sprint 6)
- No read replicas or sharding
- No connection pooling beyond SQLAlchemy defaults
- No schema for storing pseudo-model config in DB (it lives in YAML only)

---

## 5. Environment Configuration (`.env` loading)

### 5.1 Settings class

```python
# src/config.py (continued)

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    proxy_port: int = 9110
    database_url: str = "postgresql+asyncpg://proxy:proxy@localhost:5432/proxy_db"
    valkey_url: str = "valkey://localhost:6379"

    # All provider keys (passed to LiteLLM via os.environ)
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    google_api_key: str = ""
    deepseek_api_key: str = ""
    groq_api_key: str = ""
    zhipu_api_key: str = ""

settings = Settings()
```

### 5.2 What settings.py does NOT do in Sprint 1

- No `PROXY_API_KEY` for auth (Sprint 8)
- No `CORS_ORIGINS` (Sprint 8)
- No rate limit config (Sprint 8)
- No Celery broker URL (Sprint 6)

---

## 6. LiteLLM Integration

### 6.1 How LiteLLM is configured

```python
# Called once during FastAPI lifespan startup
import litellm
import os

def setup_litellm(settings: Settings):
    """Pass all provider API keys to LiteLLM via environment variables."""
    # LiteLLM reads these from os.environ
    os.environ["ANTHROPIC_API_KEY"] = settings.anthropic_api_key
    os.environ["OPENAI_API_KEY"] = settings.openai_api_key
    os.environ["GOOGLE_API_KEY"] = settings.google_api_key
    os.environ["DEEPSEEK_API_KEY"] = settings.deepseek_api_key
    os.environ["GROQ_API_KEY"] = settings.groq_api_key
    os.environ["ZHIPU_API_KEY"] = settings.zhipu_api_key

    # Optional: suppress LiteLLM debug spam
    litellm.suppress_debug_info = True
    litellm.set_verbose = False
```

### 6.2 How the proxy calls LiteLLM

```python
async def call_litellm(
    model: str,           # EXACT physical model ID from config (e.g., "qwen3-max")
    messages: list[dict],
    stream: bool = False,
    **kwargs
) -> dict:
    """
    Call LiteLLM with the exact model ID.
    Returns the full response dict (OpenAI format).
    In Sprint 1, this is a simple passthrough — no tool translation, no compaction.
    """
    response = await litellm.acompletion(
        model=model,
        messages=messages,
        stream=stream,
        **kwargs
    )
    return response
```

### 6.3 LiteLLM error handling in Sprint 1

Only two error types are handled:

| LiteLLM error | HTTP code returned by proxy | Action |
|---|---|---|
| `litellm.exceptions.ServiceUnavailableError` (503) | Triggers fallback (next model) | Log warning, try next physical model |
| `litellm.exceptions.RateLimitError` (429) | Triggers fallback (next model) | Log warning, try next physical model |
| Any other exception | 502 Bad Gateway | Return error with message from LiteLLM |

### 6.4 What LiteLLM integration does NOT do in Sprint 1

- No chat template post-processing
- No token counting (deferred to tiktoken integration in Sprint 2)
- No cost tracking (Sprint 7)
- No prompt caching headers (Sprint 7)
- No tool format translation (handled by LiteLLM internally — the proxy just passes through in Sprint 1)

---

## 7. Pseudo-model Resolution (Core Logic)

### 7.1 How a pseudo-model name maps to a physical model

```python
def resolve_physical_model(
    pseudo_model_name: str,
    config: ProxyConfig,
    conversation_id: str | None = None,
    existing_affinity: str | None = None
) -> str:
    """
    Resolve pseudo-model name to an exact LiteLLM model ID.
    Returns the model string EXACTLY as defined in pseudo_models.yaml.

    Priority:
    1. If existing_affinity is set (conversation has a pinned model), use it if
       it's still in the pseudo-model's physical_models list.
    2. Otherwise, use the FIRST physical_model in the list (priority 1).
    """
    pm = config.pseudo_models[pseudo_model_name]

    if existing_affinity:
        # Check if the pinned model is still in the list
        for phys in pm.physical_models:
            if phys.model == existing_affinity:
                return existing_affinity
        # Pinned model no longer in config — log warning, fall through to priority 1

    return pm.physical_models[0].model
```

### 7.2 Invariants

- The `model` field from `physical_models` is used **verbatim**. No string concatenation, no prefix addition (like `f"{provider}/{model}"`), no transformation.
- The `provider` field is stored but **never used** to construct the model ID sent to LiteLLM.
- If a pseudo-model name is not found in config → immediate `500 Internal Server Error` (this is a config bug, not a user error — should have been caught at startup).

### 7.3 Model name normalization (strip provider prefixes)

**Problem:** Different clients send model names in different formats:
- **OpenCode (local provider):** `"local/normal"`, `"local/tareas-avanzadas"` (prefix `local/`)
- **OpenCode (cesar-proxy provider):** `"cesar-proxy/normal"` (user-configured prefix)
- **Continue / LibreChat / curl:** `"normal"`, `"tareas-avanzadas"` (no prefix)
- **Custom scripts:** may or may not include a prefix

**Solution:** The proxy normalizes ALL incoming model names by stripping any `provider/` prefix before resolution:

```python
def normalize_model_name(raw_model: str, config: ProxyConfig) -> str:
    """
    Normalize an incoming model name for pseudo-model resolution.

    Rules:
    1. If the name EXACTLY matches a pseudo-model name → use as-is
    2. If the name contains a '/' (provider prefix), strip everything before the last '/'
       and try the remainder as a pseudo-model name
    3. If neither matches → return the original for error handling

    Examples:
        "normal"                 → "normal"
        "local/normal"           → "normal"
        "cesar-proxy/normal"     → "normal"
        "local/pensamiento-profundo-caro" → "pensamiento-profundo-caro"
        "unknown/model"          → "model" (will fail validation if not a pseudo-model)
        "a/b/c"                  → "c" (last segment after final '/')
    """
    # Rule 1: exact match
    if raw_model in config.pseudo_models:
        return raw_model

    # Rule 2: strip provider prefix (everything before the last '/')
    if "/" in raw_model:
        # Split on LAST '/' only (handles model names that themselves contain '/')
        parts = raw_model.rsplit("/", 1)
        candidate = parts[-1]
        if candidate in config.pseudo_models:
            return candidate

    # Rule 3: no match — return original for error handling in resolve_physical_model
    return raw_model
```

**Why this is critical for OpenCode compatibility:**
OpenCode's `local` provider (the one users configure with `LOCAL_ENDPOINT` pointing to our proxy) sends model names as `local/<name>`. Without normalization, `local/normal` would fail with `UNKNOWN_PSEUDO_MODEL`. With normalization, `local/normal` → `normal` → maps correctly.

This also works for any other client that prefixes model names with a provider identifier.

### 7.4 Token counting accuracy requirement

**Why accurate token counting matters for all clients:**

1. **OpenCode** reads `usage.prompt_tokens` and `usage.completion_tokens` from the API response for:
   - Cost calculation (`model.CostPer1MIn * promptTokens / 1e6`)
   - Auto-compact trigger (95% of `ContextWindow` based on accumulated `PromptTokens`)
   - Session token tracking
2. **Continue / LibreChat** may use usage info for context window awareness
3. **The proxy itself** uses token counts for:
   - Input threshold checks (`INPUT_EXCEEDS_THRESHOLD`)
   - Continuous compaction triggers (`trigger_pct` of `context_window`)
   - Context alerts (60%, 80%, 100%)
   - `proxy_metadata.context_usage_pct`

**Accuracy requirement:** Token counts in `usage.prompt_tokens` and `usage.completion_tokens` MUST reflect the actual token usage reported by the provider (LiteLLM passes these through). The proxy MUST NOT override or fabricate these values.

**What the proxy adds:** The proxy REPORTS `context_tokens_total` (accumulated across all turns) in `proxy_metadata`, but the standard `usage` field in the OpenAI response is the per-turn token count from the provider — this MUST be accurate because clients depend on it.

---

## 8. Cache Affinity (Valkey)

### 8.1 Key schema

```
Key:   conv:{conversation_id}:physical_model
Value: exact LiteLLM model ID (e.g., "qwen3-max")
TTL:   86400 seconds (24 hours, configurable)
```

### 8.2 Operations

```python
# src/cache/affinity.py

import valkey

async def get_affinity(valkey_client: valkey.Valkey, conversation_id: str) -> str | None:
    """Get the pinned physical model for a conversation. Returns None if not set or expired."""
    key = f"conv:{conversation_id}:physical_model"
    return await valkey_client.get(key)

async def set_affinity(
    valkey_client: valkey.Valkey,
    conversation_id: str,
    physical_model: str,
    ttl_seconds: int = 86400
) -> None:
    """Pin a physical model to a conversation. TTL defaults to 24h."""
    key = f"conv:{conversation_id}:physical_model"
    await valkey_client.set(key, physical_model, ex=ttl_seconds)

async def delete_affinity(valkey_client: valkey.Valkey, conversation_id: str) -> None:
    """Remove affinity pin. Used when the user explicitly changes pseudo-models."""
    key = f"conv:{conversation_id}:physical_model"
    await valkey_client.delete(key)
```

### 8.3 Valkey client initialization

```python
# Called during FastAPI lifespan startup
import valkey.asyncio as valkey

async def setup_valkey(settings: Settings) -> valkey.Valkey:
    client = valkey.from_url(settings.valkey_url, decode_responses=True)
    # Verify connection
    await client.ping()
    return client
```

### 8.4 What affinity does NOT do in Sprint 1

- No rate limiting counters in Valkey (Sprint 8)
- No distributed locks for concurrent turn processing (Sprint 2)
- No conversation metadata beyond the physical model key
- No TTL management per pseudo-model (uses global 24h default)
- No Valkey cluster support (single instance)

---

## 9. POST /v1/chat/completions (Main Endpoint)

### 9.1 Request schema

```python
# src/models/pseudo_model.py (continued)

from pydantic import BaseModel

class Message(BaseModel, extra="forbid"):
    role: str                              # "system", "user", "assistant", "tool"
    content: str | list[dict] | None       # OpenAI format: string or multimodal content array
    name: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None

class ChatRequest(BaseModel, extra="forbid"):
    model: str                             # PSEUDO-MODEL name (e.g., "normal"), NOT a physical model
    messages: list[Message]
    conversation_id: str | None = None     # Optional. If not provided, proxy generates one.
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None
```

### 9.2 Endpoint logic (exact flow)

```python
# src/api/chat.py

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
import uuid
import json

router = APIRouter()

@router.post("/v1/chat/completions")
async def chat_completions(
    request: ChatRequest,
    fastapi_request: Request,  # For accessing app.state (config, db, valkey)
):
    app_state = fastapi_request.app.state
    config: ProxyConfig = app_state.config
    db_session = app_state.db_session_factory()
    valkey_client = app_state.valkey

    # ---- STEP 1: Resolve conversation ----
    try:
        conversation_id = request.conversation_id or str(uuid.uuid4())
        pseudo_model_name = request.model

        # Validate pseudo-model exists
        if pseudo_model_name not in config.pseudo_models:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "UNKNOWN_PSEUDO_MODEL",
                    "message": f"Unknown pseudo-model: '{pseudo_model_name}'",
                    "available": list(config.pseudo_models.keys()),
                }
            )

        # ---- STEP 2: Determine physical model ----
        existing_affinity = await get_affinity(valkey_client, conversation_id)
        physical_model = resolve_physical_model(
            pseudo_model_name, config, conversation_id, existing_affinity
        )

        # ---- STEP 3: Load or create conversation ----
        # Try to find existing conversation
        conv = await db_session.get(Conversation, conversation_id)
        is_new_conversation = conv is None

        if is_new_conversation:
            conv = Conversation(
                id=uuid.UUID(conversation_id),
                pseudo_model=pseudo_model_name,
                physical_model=physical_model,
            )
            db_session.add(conv)
            await db_session.flush()

        # ---- STEP 4: Set affinity in Valkey ----
        await set_affinity(valkey_client, conversation_id, physical_model)

        # ---- STEP 5: Prepare messages ----
        messages = [msg.model_dump(exclude_none=True) for msg in request.messages]

        # ---- STEP 6: Call LiteLLM ----
        # Use fallback logic (see §9.4)
        pm = config.pseudo_models[pseudo_model_name]
        response, fallback_info = await call_with_fallback(
            pseudo_model=pm,
            messages=messages,
            stream=request.stream,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            tools=request.tools,
            tool_choice=request.tool_choice,
        )

        # ---- STEP 7: Stream or return ----
        if request.stream:
            return StreamingResponse(
                stream_response(response, conversation_id, pseudo_model_name, physical_model, fallback_info),
                media_type="text/event-stream",
            )

        # ---- STEP 8: Non-stream: save turn, return response ----
        turn_number = await get_next_turn_number(db_session, conversation_id)

        turn = ConversationTurn(
            conversation_id=uuid.UUID(conversation_id),
            turn_number=turn_number,
            pseudo_model=pseudo_model_name,
            physical_model=physical_model,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            messages=messages,
            response=response.model_dump(),
            fallback_applied=fallback_info["applied"],
            fallback_reason=fallback_info.get("reason"),
        )
        db_session.add(turn)

        # Update conversation
        conv.physical_model = physical_model
        conv.total_tokens += (turn.input_tokens + turn.output_tokens)
        conv.updated_at = func.now()

        await db_session.commit()

        # ---- STEP 9: Build proxy_metadata ----
        metadata = build_proxy_metadata(
            pseudo_model=pseudo_model_name,
            physical_model=physical_model,
            conversation_id=conversation_id,
            context_tokens=conv.total_tokens,
            context_window=pm.context_window,
            fallback_info=fallback_info,
            affinity_maintained=not is_new_conversation and existing_affinity == physical_model,
        )

        # ---- STEP 10: Return ----
        response_dict = response.model_dump()
        response_dict["proxy_metadata"] = metadata

        # Ensure conversation_id is in the response if it was auto-generated
        if not request.conversation_id:
            response_dict["conversation_id"] = conversation_id

        return response_dict

    except HTTPException:
        raise
    except Exception as e:
        await db_session.rollback()
        raise HTTPException(status_code=502, detail={"error": "PROXY_ERROR", "message": str(e)})
    finally:
        await db_session.close()
```

### 9.3 Streaming logic

```python
async def stream_response(
    litellm_response,  # This is an async generator from litellm.acompletion(stream=True)
    conversation_id: str,
    pseudo_model: str,
    physical_model: str,
    fallback_info: dict,
):
    """SSE streaming: forward chunks, append proxy_metadata on [DONE]."""
    accumulated_content = []
    accumulated_tool_calls = []
    usage = None

    async for chunk in litellm_response:
        # Forward the chunk as-is (SSE format)
        chunk_json = chunk.model_dump_json()
        yield f"data: {chunk_json}\n\n"

        # Accumulate content for DB storage (not used in Sprint 1 beyond basic tracking)
        if chunk.choices and chunk.choices[0].delta:
            delta = chunk.choices[0].delta
            if delta.content:
                accumulated_content.append(delta.content)
        if hasattr(chunk, "usage") and chunk.usage:
            usage = chunk.usage

    # Final chunk: append proxy_metadata
    metadata = build_proxy_metadata(
        pseudo_model=pseudo_model,
        physical_model=physical_model,
        conversation_id=conversation_id,
        fallback_info=fallback_info,
    )
    final_chunk = {
        "id": f"chatcmpl-{conversation_id[:12]}",
        "object": "chat.completion.chunk",
        "choices": [{"delta": {}, "finish_reason": "stop"}],
        "proxy_metadata": metadata,
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"
```

### 9.4 Fallback logic

```python
async def call_with_fallback(
    pseudo_model: PseudoModel,
    messages: list[dict],
    stream: bool = False,
    **kwargs
) -> tuple:
    """
    Try each physical model in order. On 503/429, move to next.
    On any other error, raise immediately.
    Returns (response, fallback_info_dict).
    """
    fallback_info = {"applied": False, "reason": None, "attempted_models": []}
    last_error = None

    for phys in pseudo_model.physical_models:
        try:
            response = await litellm.acompletion(
                model=phys.model,  # EXACT string from config
                messages=messages,
                stream=stream,
                **kwargs
            )
            fallback_info["attempted_models"].append(phys.model)
            return response, fallback_info
        except (litellm.exceptions.ServiceUnavailableError, litellm.exceptions.RateLimitError) as e:
            last_error = e
            fallback_info["attempted_models"].append(phys.model)
            fallback_info["applied"] = True
            fallback_info["reason"] = f"{type(e).__name__}: {phys.model}"
            continue  # Try next model
        except Exception as e:
            # Non-retryable error — raise immediately
            raise

    # All models failed
    raise HTTPException(
        status_code=503,
        detail={
            "error": "ALL_MODELS_FAILED",
            "message": f"All models for pseudo-model '{pseudo_model.display_name}' failed.",
            "attempted": fallback_info["attempted_models"],
            "last_error": str(last_error),
        }
    )
```

### 9.5 proxy_metadata format (Sprint 1 — minimal)

```python
def build_proxy_metadata(
    pseudo_model: str,
    physical_model: str,
    conversation_id: str,
    context_tokens: int = 0,
    context_window: int | None = None,
    fallback_info: dict | None = None,
    affinity_maintained: bool = True,
) -> dict:
    metadata = {
        "physical_model": physical_model,
        "pseudo_model": pseudo_model,
        "conversation_id": conversation_id,
        "affinity_maintained": affinity_maintained,
        "fallback_applied": fallback_info["applied"] if fallback_info else False,
        "fallback_reason": fallback_info.get("reason") if fallback_info else None,
    }

    if context_window:
        metadata["context_tokens_total"] = context_tokens
        metadata["context_usage_pct"] = round((context_tokens / context_window) * 100, 1) if context_window else None

    # Fields intentionally null in Sprint 1 (filled in later sprints):
    metadata["pre_compaction_applied"] = False
    metadata["continuous_compaction_applied"] = False
    metadata["router_suggestion"] = None
    metadata["tools_filter_applied"] = False
    metadata["images_described"] = 0
    metadata["warning"] = None

    return metadata
```

### 9.6 What the chat endpoint does NOT do in Sprint 1

- No capability detection on input messages (Sprint 2)
- No pseudo-model switch validation (Sprint 2)
- No tool filtering by `openai_tools_compatible` (Sprint 2 — all models are compatible in Sprint 1 config)
- No parallel tool filtering (Sprint 2)
- No input threshold checking (`INPUT_EXCEEDS_THRESHOLD`) (Sprint 2)
- No pre-compaction (Sprint 4)
- No continuous compaction (Sprint 4)
- No router LLM evaluation (Sprint 5)
- No image auto-describe (Sprint 5)
- No canonical tool format storage (Sprint 3)
- No auth validation (Sprint 8)

---

## 10. GET /v1/models

### 10.0 Critical design decision: always advertise ALL capabilities

**Research findings (verified against OpenCode source code):**

1. **OpenCode does NOT call `/v1/models` for capability discovery.** It uses 100% hardcoded model definitions (`internal/llm/models/models.go`). The only capability flags are `CanReason` and `SupportsAttachments` — there is no `vision`, `tools`, `function_calling`, or `parallel_tools` flag.
2. **OpenCode only strips new attachments** when `SupportsAttachments=false` at `Run()` entry point. Historical images already stored in messages are NEVER filtered.
3. **OpenCode NEVER filters tools.** No tool capability flag exists in the model struct. Tools are always sent.
4. **Other OpenAI-compatible clients (Continue, LibreChat, custom scripts) DO read `/v1/models`** and may strip content based on advertised capabilities.

**Why optimistic advertising is STILL the correct approach:**

The optimistic approach (`vision: true`, `tools: true`, `parallel_tools: true`, `function_calling: true` for ALL pseudo-models) prevents silent content stripping for **all clients that DO read `/v1/models`**. OpenCode won't be affected (it never reads the endpoint), but Continue, LibreChat, and any custom scripts will send ALL content instead of silently dropping it. The proxy then validates and returns clear errors if unsupported.

**For OpenCode specifically**, the proxy integration works via the `local` provider with `LOCAL_ENDPOINT` env var — OpenCode sends all content regardless, so the proxy's `validate_incoming_content()` (Sprint 2 §7) is the safety net.

### 10.1 Response format

```json
{
  "object": "list",
  "data": [
    {
      "id": "normal",
      "object": "model",
      "created": 1700000000,
      "owned_by": "proxy-cesar",
      "display_name": "Normal",
      "description": "Punto de entrada recomendado. Coding extenso, trabajo agéntico.",
      "capabilities": {
        "vision": true,
        "tools": true,
        "parallel_tools": true,
        "streaming": true,
        "function_calling": true
      },
      "context_window": 96000,
      "input_token_threshold": 96000,
      "pricing": {
        "estimated_input_cost_per_1k": null,
        "estimated_output_cost_per_1k": null
      }
    }
  ]
}
```

**Every pseudo-model returns the same capabilities block:**
```json
"capabilities": {
    "vision": true,
    "tools": true,
    "parallel_tools": true,
    "streaming": true,
    "function_calling": true
}
```

This is true even for `normal` (which has no vision physical models), `deep-flash` (which has basic tools only), and `flash-lowcost` (which has no parallel tools). The capabilities advertised to clients are **intentionally optimistic** to prevent silent content stripping.

### 10.2 Implementation

```python
# src/api/models.py

# ALL_CAPABILITIES is a constant — every pseudo-model advertises the same optimistic capabilities
ALL_CAPABILITIES = {
    "vision": True,
    "tools": True,
    "parallel_tools": True,
    "streaming": True,
    "function_calling": True,
}

@router.get("/v1/models")
async def list_models(request: Request):
    config: ProxyConfig = request.app.state.config
    models = []
    for name, pm in config.pseudo_models.items():
        models.append({
            "id": name,
            "object": "model",
            "created": 1700000000,  # Static timestamp
            "owned_by": "proxy-cesar",
            "display_name": pm.display_name,
            "description": pm.description,
            "capabilities": ALL_CAPABILITIES,  # Always optimistic — prevents silent stripping
            "context_window": pm.context_window,
            "input_token_threshold": pm.input_token_threshold,
        })
    return {"object": "list", "data": models}
```

### 10.3 What `/v1/models` does NOT do

- Does NOT return physical model names (user doesn't need to know them)
- Does NOT return real-time pricing (Sprint 7)
- Does NOT return per-model tool levels (Sprint 3)
- Does NOT filter based on conversation state (that's `/compatible-models` in Sprint 2)
- **Does NOT advertise actual capabilities** — always optimistic. Validation happens at request time (Sprint 2), not at model listing time. This prevents OpenCode from silently stripping unsupported content.

---

## 11. GET /health

### 11.1 Response format

```json
{
  "status": "ok",
  "postgres": "connected",
  "valkey": "connected",
  "providers": {
    "anthropic": "configured",
    "openai": "configured",
    "google": "configured",
    "deepseek": "configured",
    "groq": "configured",
    "zhipu": "configured"
  },
  "uptime_seconds": 3600,
  "pseudo_models_loaded": 8
}
```

### 11.2 Implementation

```python
@router.get("/health")
async def health(request: Request):
    app_state = request.app.state
    config: ProxyConfig = app_state.config

    # Check PostgreSQL
    try:
        async with app_state.db_session_factory() as session:
            await session.execute(text("SELECT 1"))
        postgres_status = "connected"
    except Exception:
        postgres_status = "disconnected"

    # Check Valkey
    try:
        await app_state.valkey.ping()
        valkey_status = "connected"
    except Exception:
        valkey_status = "disconnected"

    # Check providers (just whether API keys are set)
    providers = {}
    for provider in ["anthropic", "openai", "google", "deepseek", "groq", "zhipu"]:
        key_env = f"{provider.upper()}_API_KEY"
        providers[provider] = "configured" if os.getenv(key_env) else "not configured"

    overall = "ok" if postgres_status == "connected" and valkey_status == "connected" else "degraded"

    return {
        "status": overall,
        "postgres": postgres_status,
        "valkey": valkey_status,
        "providers": providers,
        "pseudo_models_loaded": len(config.pseudo_models),
    }
```

### 11.3 Health endpoint rules

- **No auth required** — this endpoint is public
- Returns 200 even in degraded state (status field indicates health)
- Never returns 500 from this endpoint itself (catch all exceptions)
- Does NOT actually call provider APIs (too slow, too many tokens)

---

## 12. FastAPI App Bootstrap (`main.py`)

### 12.1 Lifespan

```python
# src/main.py

from contextlib import asynccontextmanager
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from src.config import load_config, Settings, settings
from src.cache.affinity import setup_valkey
from src.api.chat import router as chat_router
from src.api.models import router as models_router
from src.api.health import router as health_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- STARTUP ----
    print("Loading pseudo_models.yaml...")
    config = load_config()
    app.state.config = config
    print(f"Loaded {len(config.pseudo_models)} pseudo-models")

    # Database
    engine = create_async_engine(settings.database_url, echo=False)
    app.state.db_session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # Valkey
    valkey_client = await setup_valkey(settings)
    app.state.valkey = valkey_client

    # LiteLLM
    setup_litellm(settings)

    print(f"Proxy ready on port {settings.proxy_port}")

    yield  # App runs here

    # ---- SHUTDOWN ----
    await valkey_client.close()
    await engine.dispose()
    print("Proxy shut down")

app = FastAPI(title="Proxy Determinista Multi-Modelo", version="0.1.0", lifespan=lifespan)

app.include_router(chat_router)
app.include_router(models_router)
app.include_router(health_router)
```

### 12.2 Entry point

```bash
# In pyproject.toml:
# [tool.poetry.scripts]
# proxy = "src.main:main"

# Or run directly:
uvicorn src.main:app --host 0.0.0.0 --port 9110 --reload
```

---

## 13. Tests (Sprint 1)

### 13.1 test_config.py — Pseudo-model YAML validation

**Test cases (minimum 15):**

1. Valid `pseudo_models.yaml` loads without error
2. Missing file → `SystemExit(1)`
3. Invalid YAML syntax → `SystemExit(1)`
4. Missing `pseudo_models` key → `SystemExit(1)`
5. Empty `physical_models` list → validation error
6. Missing `model` field in physical model → validation error
7. `openai_tools_compatible: false` → validation error
8. `fallback_strategy: "invalid"` → validation error
9. `image_handling.on_downgrade: "invalid"` → validation error
10. `continuous_compaction.enabled: true` without `trigger_pct` → validation error
11. `pre_compaction.enabled: true` without `threshold` → validation error
12. `pre_compaction.compactor` references non-existent pseudo-model → validation error
13. `router_llm.suggester` references non-existent pseudo-model → validation error
14. Extra unknown field in pseudo-model → `extra="forbid"` catches it
15. All 8 pseudo-models loaded successfully from the production YAML

### 13.2 test_affinity.py — Valkey cache affinity

**Test cases (minimum 6):**

1. `set_affinity` writes key with correct value
2. `get_affinity` returns the written value
3. `get_affinity` returns `None` for non-existent key
4. TTL is respected (mock time or use fakeredis)
5. `delete_affinity` removes the key
6. Multiple conversations have isolated keys

### 13.3 test_chat.py — POST /v1/chat/completions

**Test cases (minimum 12):**

1. New conversation → creates conversation, sets affinity, returns response with `proxy_metadata`
2. Second turn same pseudo-model → same physical model, `affinity_maintained: true`
3. Unknown pseudo-model → 400 `UNKNOWN_PSEUDO_MODEL`
4. Auto-generated `conversation_id` when not provided
5. `conversation_id` returned in response when auto-generated
6. Response includes all expected `proxy_metadata` fields
7. Messages are forwarded correctly to LiteLLM (verify via mock)
8. Turn is saved to DB after response
9. `total_tokens` accumulates across turns
10. `conversation_id` passed via `X-Conversation-ID` header is used
11. Request with `tools` parameter is forwarded correctly (no filtering in Sprint 1)
12. Request with `stream: true` returns `text/event-stream`

### 13.4 test_streaming.py — SSE streaming

**Test cases (minimum 5):**

1. Stream response produces valid SSE format (`data: ...\n\n`)
2. Content chunks are forwarded without modification
3. Final chunk contains `proxy_metadata`
4. `[DONE]` marker is present at end of stream
5. Stream is closed cleanly on error

### 13.5 test_fallback.py — Fallback within pseudo-model

**Test cases (minimum 6):**

1. Primary model succeeds → no fallback, `fallback_applied: false`
2. Primary model returns 503 → fallback to second model, `fallback_applied: true`
3. All models return 503 → 503 `ALL_MODELS_FAILED`
4. Primary model returns 429 → fallback to second model
5. Non-retryable error (e.g., 400) → raised immediately, no fallback
6. `fallback_reason` in `proxy_metadata` explains which model failed and why

### 13.6 test_models_endpoint.py — Endpoints

**Test cases (minimum 5):**

1. `GET /v1/models` returns all 8 pseudo-models
2. Each model advertises ALL capabilities as `true` (optimistic advertising — see §10.0)
3. `GET /health` returns 200 with all services status
4. `GET /health` reflects degraded state when DB is down
5. `GET /health` does not require auth

### 13.7 conftest.py — shared fixtures

```python
# tests/conftest.py

import pytest
from httpx import ASGITransport, AsyncClient
from src.main import app
from src.config import ProxyConfig, load_config

@pytest.fixture
def valid_config() -> ProxyConfig:
    """Load the production pseudo_models.yaml for tests."""
    return load_config()

@pytest.fixture
async def async_client():
    """Async test client for FastAPI."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
```

---

## 14. Acceptance Criteria (Sprint 1)

For the sprint to be considered **DONE**, ALL of the following must pass:

- [ ] `pseudo_models.yaml` loads successfully with all 8 pseudo-models
- [ ] Proxy starts without errors; `GET /health` returns `"status": "ok"`
- [ ] `POST /v1/chat/completions` with `{"model": "normal", "messages": [...]}` returns a valid OpenAI-format response
- [ ] The response includes `proxy_metadata` with `physical_model`, `pseudo_model`, `conversation_id`, `affinity_maintained`, `fallback_applied`
- [ ] 20 consecutive turns with the same `conversation_id` and `"normal"` → same physical model (`qwen3-max`)
- [ ] Streaming works: `{"model": "normal", "stream": true, ...}` returns SSE chunks ending with `[DONE]` and `proxy_metadata` in the final chunk
- [ ] Mocked 503 on primary model → fallback to secondary model within same pseudo-model → `fallback_applied: true`
- [ ] All models return 503 → 503 error `ALL_MODELS_FAILED`
- [ ] `GET /v1/models` returns all 8 pseudo-models with ALL capabilities as `true` (optimistic, to prevent silent content stripping)
- [ ] All 40+ tests pass
- [ ] All dependencies have MIT/BSD/Apache 2.0 licenses (verified via `pip-licenses`)

---

## 15. Explicitly OUT OF SCOPE for Sprint 1

The following features are **explicitly excluded** from Sprint 1. They must NOT be implemented, planned, or stubbed in Sprint 1 code:

| Feature | Sprint |
|---|---|
| Capability detection (images, tools, parallel tools) | 2 |
| Compatibility validation on pseudo-model switch | 2 |
| Tool filtering by `openai_tools_compatible` or `parallel_tools` | 2 |
| Input threshold checking (`INPUT_EXCEEDS_THRESHOLD`) | 2 |
| Canonical tool format storage | 3 |
| `POST /normalize-tools` | 3 |
| Tool edge case handling (streaming partial, mixed content) | 3 |
| Pre-compaction | 4 |
| Continuous compaction | 4 |
| Explicit compaction (`POST /compact`) | 6 |
| Context alerts (HIGH, UNUSABLE) | 6 |
| Image auto-describe | 5 |
| Router LLM | 5 |
| Provider cache optimization (cache_control, prompt_cache_key) | 7 |
| Auth (PROXY_API_KEY) | 8 |
| CORS | 8 |
| Rate limiting | 8 |
| Celery | 6 |
| Audit log | 6 |
| Metrics endpoint | 8 |

---

## 16. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| LiteLLM doesn't support a provider in the config | Test all 6 providers during Sprint 1 setup. Remove unsupported providers from config before committing. |
| Valkey client library (`valkey`) has bugs or missing features | Fall back to `redis` library (Redis OSS is BSD-licensed and protocol-compatible with Valkey). |
| LiteLLM model IDs differ from what's in the config | Verify each model ID with `litellm.get_model_info()` or a quick test call during setup. |
| Streaming chunks are corrupted or dropped by FastAPI | Use `StreamingResponse` with `media_type="text/event-stream"` and test with `curl -N`. |
| Async session management causes connection leaks | Use `async with session` context managers everywhere. Test with `pool_size=5` and concurrent requests. |

---

## 17. Deliverables

1. **Working proxy** running on `localhost:9110`
2. **`pseudo_models.yaml`** with all 8 pseudo-models, validated at startup
3. **All 5 API endpoints** working:
   - `POST /v1/chat/completions` (streaming + non-streaming)
   - `GET /v1/models`
   - `GET /health`
4. **PostgreSQL schema** with `conversations` and `conversation_turns` tables
5. **Valkey affinity** with 24h TTL
6. **Fallback logic** within pseudo-models (503/429 → next model)
7. **`proxy_metadata`** in every response
8. **40+ passing tests**
9. **`README.md`** with:
   - How to install dependencies
   - How to set up `.env`
   - How to run the proxy
   - How to make a test request with `curl`
   - How to run tests
