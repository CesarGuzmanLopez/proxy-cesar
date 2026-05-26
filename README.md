# proxy-cesar

**Deterministic multi-model LLM proxy.** Transparent HTTP proxy between LLM clients and multiple providers. Exposes abstract **pseudo-models** that map to concrete physical models with automatic fallback, content compatibility validation, tool normalization, context compaction, context alerts, explicit compaction, audit logging, provider cache optimization, auth, rate limiting, and structured metrics.

> **Paquetería 100% libre.** MIT, BSD, Apache 2.0.
> **El proxy no decide. Valida, informa y ejecuta lo que el usuario ordena.**
> **Cuando algo no puede continuar, lo dice claramente y ofrece opciones.**
> **Cuando algo puede continuar, lo hace sin fricción.**
> **100% tipado y determinista.** La única no-determinista es la respuesta del modelo.

---

## Objetivos del Sistema

1. **Unificar múltiples proveedores LLM** detrás de una API OpenAI-compatible única
2. **Maximizar cache hits** mediante afinidad de modelo físico, orden canónico de mensajes, y optimizaciones por proveedor
3. **Validar compatibilidad** de contenido multimedia y tools al cambiar de pseudo-modelo
4. **Compactar contexto** automática y explícitamente para no saturar ventanas de modelos
5. **Informar** cada decisión del proxy en `proxy_metadata` — nunca silencio

---

## Decisiones Arquitectónicas Clave

| Decisión | Alternativa | Por qué esta gana |
|----------|-------------|-------------------|
| Error handling con `Result[T,E]` monad | Exceptions | Errores como datos — el `match/case` fuerza al caller a manejar ambos casos |
| Validación de configuración fail-fast | Runtime filtering | Si `pseudo_models.yaml` es inválido, el proxy no arranca. Nunca errores en producción |
| `openai_tools_compatible: true` obligatorio | Permitir `false` | El plan exige que todos los modelos pasen la validación de startup. Más estricto = más seguro |
| Afinidad como Protocol abstracto | Valkey directo | `AffinityPort` permite testear sin Valkey real y cambiar backend sin tocar dominio |
| Fallback `by_context_window` + chunking | Single model | Groq (131K, rápido) primario; si el historial excede, se divide en chunks con prefijo compartido (caché preservado) y cada chunk se comprime independientemente. Sin modelo caro necesario. |
| Direct model passthrough | Solo pseudo-models | Cualquier modelo `ollama/xxx`, `lmstudio/xxx` se puede llamar directo sin configuración |
| arq en vez de Celery | Celery | async-native, MIT, 700 líneas vs 50K+. Misma autora de Pydantic |
| Single API key (Bearer) | JWT/OAuth | Suficiente para v1. Multi-user es v2 |
| `proxy_metadata` en cada respuesta | Solo logging | El cliente sabe siempre qué proveedor usó, cuánto ahorró, si hubo fallback |

---

## Quick Start

```bash
cd proxy
cp .env.example .env   # Editar con API keys
.venv/bin/python -m src.main  # Puerto 9110
```

**Test rápido:**
```bash
curl http://localhost:9110/health
curl http://localhost:9110/v1/models
curl -X POST http://localhost:9110/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"normal","messages":[{"role":"user","content":"Hola"}],"stream":false}'
```

---

## Verificación Automática

```bash
./test_hard.sh   # Inicia servidor + ejecuta todas las HUs
```

Ejecuta las 17 historias de usuario definidas en `BUG_VERIFICATION_FLOW.md` (salud, chat normal, pensamiento profundo, visión, aliases, streaming, compactación, auditoría, etc.). Opciones:
- `--no-start` — solo tests, servidor ya corriendo
- `--no-kill` — no mata el servidor al terminar

---

## 10 Pseudo-Modelos en `proxy/pseudo_models.yaml`

Cada uno con límite de tokens (`input_token_threshold`), compactación continua cuando aplica, y manejo de imágenes (`on_downgrade`). Al superar el límite se activan alertas de contexto (60%/80%/100%) y se recomienda usar `/compact`.

| Pseudo-modelo | Límite | Physical models | Compactación continua | Imágenes |
|---|---|---|---|---|
| `pensamiento-profundo-caro` | **120K** | `zai/glm-5` → `deepseek/deepseek-v4-pro` → `anthropic/claude-haiku-4-5` | ✅ 70% | auto_describe |
| `tareas-avanzadas` | **200K** | `deepseek/deepseek-v4-pro` → `deepseek/deepseek-v4-flash` → `zai/glm-5` | ✅ 75% | block |
| `vision` | **120K** | `zai/glm-4.5v` | ❌ | auto_describe |
| `vision-lite` | **32K** | `zai/glm-4.6v-flash` → `groq/meta-llama/llama-4-scout` | ❌ | auto_describe |
| `normal` | **500K** | `deepseek/deepseek-v4-flash` → `zai/glm-4.5-flash` | ✅ 80% | block |
| `normal-gratis` | **200K** | `openrouter/nvidia/nemotron-120b:free` → `zai/glm-4.7-flash` → `zai/glm-4.5-flash` | ❌ | auto_describe |
| `deep-flash` | **1M** | `deepseek/deepseek-v4-flash` | ❌ | block |
| `massive-fast` | **131K** | `groq/openai/gpt-oss-20b` → `groq/qwen/qwen3-32b` | ❌ | block |
| `flash-lowcost` | **128K** | `zai/glm-4.5-flash` → `ollama/llama3.2` | ❌ | block |
| `compactador` | **20M** | `groq/openai/gpt-oss-120b` (≤131K) → `gpt-oss-20b` → `qwen3-32b` → `deepseek/deepseek-v4-flash` (1M) → `claude-haiku-4-5` (200K) → `glm-4.5-flash` (128K) | ❌ | auto_describe |

### Model Aliases

| Alias | Resuelve a |
|---|---|
| `gpt-4o` | `normal` |
| `gpt-4o-mini` | `deep-flash` |
| `gpt-4.1` | `tareas-avanzadas` |
| `o3` / `o4-mini` | `pensamiento-profundo-caro` |
| `claude-haiku-3-5-20241022` | `flash-lowcost` |
| `default` | `normal` |

---

## Modelos Locales (Passthrough Directo)

Cualquier modelo con prefijo de proveedor se puede llamar **directamente** sin estar en `pseudo_models.yaml`:

```
ollama/llama3.2      → LiteLLM → Ollama
ollama/llava         → LiteLLM → Ollama
lmstudio/my-model    → LiteLLM → LM Studio
lmstudio/0           → LiteLLM → LM Studio (multi-modelo)
local/cualquier-mo   → LiteLLM → proveedor local
```

Sin compactación continua, sin router, sin límites de threshold. Usan el sistema de conversaciones (turns, snapshots) y soportan `/compact` para compactación explícita. La respuesta se reporta tal cual la devuelve el modelo local.

---

## Provider Caching

| Proveedor | Tipo | Modelos | Ahorro |
|---|---|---|---|
| **Anthropic** | `cache_control` breakpoints | Claude Haiku 4.5 (breakpoints en system + penúltimo) | Por breakpoint |
| **DeepSeek** | Prefix automático | deepseek-v4-* | Automático, sin costo |
| **Z.ai (Zhipu)** | Prefix automático | glm-5, glm-4.5v, glm-4.5-flash, etc. | ~82% (input $0.60→$0.11) |
| **Groq** | Prefix automático | gpt-oss-20b, gpt-oss-120b | **50%** en cacheados, 2h TTL |
| **Ollama** (local) | Sin caché | llama3.2, llava, etc. | N/A (local) |

### Estrategia

- **Anthropic**: `cache_control` colocado en **content items** (no a nivel de mensaje — bug corregido). Breakpoint 1 en system message, Breakpoint 2 en penúltimo mensaje.
- **DeepSeek/Z.ai/Groq**: Prefix caching automático. El orden canónico de mensajes (`system → tools(sorted) → history → new`) maximiza el match de prefijo.
- **Chunking en compactación**: Cada chunk comparte el mismo HEAD (primeros mensajes), permitiendo cache hits en llamadas múltiples al compactor.
- **Destrucción en fallback**: Cuando cambia el modelo físico, `proxy_metadata` reporta `previous_cache_destroyed: true` con costo estimado.

```
Turno 1:  system + tools + history + query1  → provider caches prefix
Turno 2:  system + tools + history + query2  → CACHE HIT (prefix unchanged)
...
Turno n:  system + tools + history + queryN  → CACHE HIT
Fallback: cache DESTROYED → reportado en proxy_metadata
```

---

## Arquitectura

```
┌──────────────────────────────────────────────────────────────────────┐
│                        CLIENTES                                       │
│   OpenCode / Continue / LibreChat / Aider / curl                      │
│   Solo tienen: CESAR_PROXY_URL + CESAR_PROXY_KEY                      │
└─────────────────────┬────────────────────────────────────────────────┘
                      │ POST /v1/chat/completions {model, messages, key}
                      ▼
┌──────────────────────────────────────────────────────────────────────┐
│  .venv/bin/python -m src.main → uvicorn :9110                        │
│                                                                       │
│  ├── AuthMiddleware          ═══ Bearer token vs PROXY_API_KEY        │
│  ├── RateLimitMiddleware     ═══ Valkey sliding-window counter         │
│  ├── CORSMiddleware          ═══ CORS_ORIGINS                         │
│  │                                                                     │
│  └── Router → chat_completions()                                      │
│         │                                                             │
│         ├── normalize_model_name()  → alias/passthrough → pseudo      │
│         ├── detect_turn_capabilities() → has_images, has_tools        │
│         ├── validate_incoming_content() → AUDIO/PDF/VIDEO blocked     │
│         ├── load_session + validate_switch() → safe/warning/blocked   │
│         ├── get_eligible_models() → filter by parallel_tools          │
│         ├── check_input_threshold() → pre_compaction?                 │
│         ├── continuous_compact() → snapshot at trigger_pct            │
│         ├── evaluate_complexity() → Router LLM suggestion (no action) │
│         │                                                             │
│         ├── call_with_fallback()                                      │
│         │     ├── apply_anthropic_cache_control() si Anthropic        │
│         │     └── litellm.acompletion()  → 503/429 → fallback next    │
│         │                                                             │
│         └── _save_and_return() → DB + proxy_metadata                  │
│                                                                       │
│  SQLite ── conversations, turns, snapshots, capabilities               │
│  Valkey ── affinity (conv:{id}:model), rate limits                     │
│  arq    ── async compaction (>500K tokens)                             │
└──────────────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        PROVEEDORES LLM                                │
│   Z.ai (GLM) │ DeepSeek │ Anthropic │ OpenRouter │ Groq │ Ollama     │
│   ← Todos los locales: lmstudio/xxx, local/xxx también funcionan     │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Configuración

### `proxy/pseudo_models.yaml`
```yaml
pseudo_models:
  normal:
    display_name: Normal
    description: 'Punto de entrada recomendado. Límite 500K — usa /compact si lo superas.'
    input_token_threshold: 500000
    context_window: 500000
    continuous_compaction:
      enabled: true
      trigger_pct: 80
      compact_preserve_recent: 32768
    image_handling:
      on_downgrade: block
    physical_models:
      - provider: deepseek
        model: deepseek/deepseek-v4-flash
        openai_tools_compatible: true
        tools_strict: true
        parallel_tools: true
        vision: false
        context_window: 1000000
      - provider: zhipu
        model: zai/glm-4.5-flash
        openai_tools_compatible: true
        tools_strict: false
        parallel_tools: false
        vision: false
        context_window: 128000
    fallback_strategy: sequential
```

### `proxy/.env`
```bash
# Provider API keys (NUNCA salen del servidor)
ANTHROPIC_API_KEY=sk-ant-xxxx
DEEPSEEK_API_KEY=sk-xxxx
GROQ_API_KEY=gsk_xxxx
OPENROUTER_API_KEY=sk-or-xxxx
ZAI_API_KEY=xxxx
ZHIPUAI_API_KEY=xxxx

# Proxy
PROXY_PORT=9110
DATABASE_URL=sqlite+aiosqlite:///./proxy.db
VALKEY_URL=valkey://localhost:6379
PROXY_API_KEY=sk-proxy-generate-with-openssl-rand-hex-32

# KeyClaw (opcional, desactivable)
KEYCLAW_ENABLED=false
```

---

## SSL en NixOS

En NixOS, httpx/litellm leen `SSL_CERT_FILE` pero el sistema expone `NIX_SSL_CERT_FILE`.
El proxy puentea automáticamente ambas variables al arrancar. Si hay problemas SSL:

```bash
SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt setsid .venv/bin/uvicorn src.main:app --port 9110
```

---

## KeyClaw

[KeyClaw](https://keyclaw.org) es un MITM proxy opcional que filtra API keys del tráfico saliente.
Si KeyClaw está instalado y su daemon corriendo, el proxy lo usa automáticamente.
Si KeyClaw está instalado pero no funciona correctamente (SSL issues con algunos providers),
desactivar con `KEYCLAW_ENABLED=false` en `.env`.

---

## Inline Commands

Cualquier mensaje que empiece con `/` o `@` se procesa como comando inline:

| Comando | Acción |
|---|---|
| `/compact` | Normaliza tools + degrada multimedia + compacta historia |
| `/degrade` | Describe imágenes como texto |
| `/status` | Muestra estado de la conversación |
| `/help` | Lista de comandos |

Los comandos se ejecutan ANTES de llamar al LLM y devuelven respuesta textual directamente.

---

## Test Suite

```bash
cd proxy
.venv/bin/pytest                                      # ~410 tests
.venv/bin/pytest tests/test_caching.py                # 13 tests (caching)
.venv/bin/pytest tests/test_auth.py                   # 7 tests
.venv/bin/pytest -k "not streaming"                   # sin streaming mock
```

Para verificación completa de todas las historias de usuario:
```bash
./test_hard.sh                                        # 17 HUs
```

---

## Features

| Feature | Estado |
|---------|--------|
| 10 pseudo-modelos con validación startup | ✅ |
| Direct model passthrough (ollama, lmstudio, local) | ✅ |
| Afinidad de modelo físico (Valkey 24h) | ✅ |
| Streaming SSE + proxy_metadata | ✅ |
| Fallback secuencial (503/429) | ✅ |
| Detección de capabilities (imágenes, tools, parallel) | ✅ |
| Validación de compatibilidad (safe/warning/blocked) | ✅ |
| Normalización de parallel tools | ✅ |
| Pre-compactación de input largo | ✅ |
| Compactación continua (trigger_pct) | ✅ |
| Chunking con prefijo compartido para >131K | ✅ |
| Compactación explícita (POST /compact) | ✅ |
| Context alerts (60%, 80%, 100%) | ✅ |
| Audit log (GET /conversations/{id}/audit-log) | ✅ |
| arq para compactación async (>500K) | ✅ |
| Autenticación Bearer + rate limiting | ✅ |
| Orden canónico de mensajes (cache prefix) | ✅ |
| Anthropic cache_control breakpoints | ✅ |
| DeepSeek/Z.ai/Groq prefix caching | ✅ |
| Cache metadata en proxy_metadata | ✅ |
| Cache destruction metadata en fallbacks | ✅ |
| Model aliases (gpt-4o → normal, etc.) | ✅ |
| Límites de tokens por pseudo-modelo | ✅ |
| Alertas educativas (usa `/compact` si superas) | ✅ |
| Degradación de imágenes manual y automática | ✅ |
| Router LLM (sugiere downgrade, nunca impone) | ✅ |
| Inline commands (/compact, /degrade, /status) | ✅ |
| Config ejemplo para OpenCode | ✅ |
| KeyClaw graceful degradation | ✅ |
| SSL_CERT_FILE auto-setup (NixOS) | ✅ |
| Script verificación automática (test_hard.sh) | ✅ |

---

## Licencia

MIT. Todas las dependencias tienen licencia MIT/BSD/Apache 2.0. Sin Google.
