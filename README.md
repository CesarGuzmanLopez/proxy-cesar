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
| `openai_tools_compatible: true` obligatorio | Permitir `false` | El plan (plan-proxy.md §4.0) exige que todos los modelos pasen la validación de startup. Más estricto = más seguro |
| Afinidad como Protocol abstracto | Valkey directo | `AffinityPort` permite testear sin Valkey real y cambiar backend sin tocar dominio |
| Fallback `by_context_window` | Sequential fixed | El compactador elige el modelo que cubre el historial — permite compactar 1M tokens con Gemini |
| arq en vez de Celery | Celery | async-native, MIT, 700 líneas vs 50K+. Misma autora de Pydantic |
| Single API key (Bearer) | JWT/OAuth | Suficiente para v1. Multi-user es v2 |
| `proxy_metadata` en cada respuesta | Solo logging | El cliente sabe siempre qué proveedor usó, cuánto ahorró, si hubo fallback |

---

## 10 Archivos Más Importantes

### 1. `src/domain/types.py` — El Result Monad
```python
type Result[T, E] = Ok[T] | Err[E]
```
**Por qué es crítico:** Define el patrón de error handling de TODO el proyecto. Sin excepciones para errores de negocio — cada función que puede fallar retorna `Ok(valor)` o `Err(error)`. El `match/case` fuerza al caller a manejar ambos casos explícitamente. Es el pilar del determinismo. Sin este archivo, el sistema entero colapsaría a exception-driven chaos.

### 2. `src/domain/errors.py` — 11 Errores de Dominio
```python
@dataclass(frozen=True, slots=True)
class InputExceedsThreshold: estimated: int; threshold: int; pseudo_model: str
```
**Por qué es crítico:** Cada error de negocio es un tipo concreto con campos exactos. `frozen=True` = inmutables, `slots=True` = eficientes. Son datos, no excepciones. La capa API los convierte a HTTP 400/409/503 solo en el boundary. Cada error incluye `remediation` — opciones para que el usuario resuelva.

### 3. `src/service/chat_service.py` — El Orquestador (14 pasos)
```python
async def process_chat_request(...):
    # 1-3: Resolver pseudo-modelo, detectar capabilities, validar contenido
    # 4-11: Cargar sesión, verificar switch, afinidad, modelo físico
    # 12: Threshold → pre-compaction → compactación continua
    # 13: call_litellm con fallback chain
    # 14-20: Guardar turno, acumular capabilities, construir proxy_metadata
```
**Por qué es crítico:** Sigue exactamente el árbol de decisión de `plan-proxy.md §20.1`. Cada paso delega en funciones puras. NO mezcla lógica con I/O. Las decisiones (safe/warning/blocked) se toman ANTES de tocar LiteLLM. Fallback itera sobre physical_models hasta que uno funciona. 944 líneas de orquestación sin side effects no-locales.

### 4. `src/service/model_resolver.py` — Resolución + Aliases
```python
def normalize_model_name(raw_model, config):
    # 1: Exact pseudo-model match → as-is
    # 2: Strip prefix (cesar-proxy/normal → normal)
    # 3: Alias match (gpt-4o → normal)
    # 4: Default alias (unknown → normal)
```
**Por qué es crítico:** El proxy acepta nombres en múltiples formatos (OpenCode manda `local/gpt-4o`, Cline manda `cline/normal`, curl manda `normal`). Normaliza SIN transformar el model ID físico. Los aliases son bridge, no sustitutos — el modelo físico se usa exactamente como está en `pseudo_models.yaml`. La regla de oro: el cliente ve aliases, el proveedor ve modelos exactos del YAML.

### 5. `src/service/compatibility.py` — 8 Checks Deterministas
```python
def validate_switch(from_pseudo, to_pseudo, caps) -> CompatibilityResult:
    # 1-4: Imágenes, Audio, PDF, Video → modelo destino debe soportarlos
    # 5: Parallel tools → destino debe tener parallel_tools: true
    # 6: Contexto acumulado → destino debe tener context_window suficiente
    # 7-8: Tools/Upgrade → checks de capacidad
```
**Por qué es crítico:** No hay ML, no hay heurísticas. 8 checks puramente deterministas. Resultado = `Safe` (sigue sin fricción), `Warning(reason)` (sigue con advertencia), o `Blocked(reason, remediation)` (HTTP 409 con opciones). Cada BLOCKED ofrece caminos de remediación: normalize-tools, degrade-images, compact, o cambiar a otro pseudo-modelo.

### 6. `src/service/tool_filter.py` — Filtro de Pool por Tools
```python
def get_eligible_models(models, caps):
    if caps.has_parallel_tools:
        parallel = [m for m in models if m.parallel_tools]
        return parallel or models  # si ninguno soporta parallel, warning
    return models
```
**Por qué es crítico:** Todos los modelos tienen `openai_tools_compatible: true` (validado en startup). El único filtro runtime es por `parallel_tools`. Si el modelo fijado no soporta parallel y la conversación los necesita, el proxy intenta fallback al siguiente modelo del mismo pseudo-modelo antes de reportar error.

### 7. `src/service/compactor/explicit.py` — Compactación by Context Window
```python
def select_compactor_model(history_tokens, pseudo_model):
    # by_context_window: elige modelo con ventana suficiente
    # gemini-3.5-flash (1M) → claude-haiku (200K) → glm-4.5-flash (128K)
```
**Por qué es crítico:** No usa fallback secuencial fijo — selecciona dinámicamente el primer modelo cuya ventana de contexto cubre el historial. Esto permite compactar conversaciones de hasta **1M tokens** usando Gemini. Si el historial excede 500K tokens, delega a arq (async) para no bloquear HTTP. El snapshot generado es Markdown estructurado con decisiones, código, estado, y pendientes.

### 8. `src/adapters/cache/message_ordering.py` — Orden Canónico
```python
def assemble_canonical_messages(system, tools, history, new):
    # Orden: system → tools(sorted) → history → new
    # El prefijo es idéntico entre turnos → provider cachea
```
**Por qué es crítico:** El orden canónico maximiza cache hits. El prefijo `[system + tools + history]` es idéntico entre turnos. Solo la cola (`new messages`) cambia. El proveedor reconoce el prefijo y reusa el caché. `sort_keys=True` garantiza que JSON idéntico → hash idéntico. Sin esto, cada turno destruiría el caché del turno anterior.

### 9. `src/adapters/cache/provider_cache.py` — Cache por Proveedor
```python
def apply_anthropic_cache_control(messages):
    # Breakpoint 1: system message
    # Breakpoint 2: penúltimo mensaje (fin del historial)
    # Max 4 breakpoints (límite de Anthropic)

def build_cache_metadata(response, provider, cache_applied):
    # OpenAI: usage.prompt_tokens_details.cached_tokens
    # Anthropic: usage.cache_read_input_tokens
```
**Por qué es crítico:** Cada proveedor tiene un mecanismo de caché diferente. Este archivo abstrae las diferencias: Anthropic necesita `cache_control` breakpoints, OpenAI/DeepSeek tienen auto-caching, Gemini requeriría CachedContent API directa (documentado como limitación). Extrae metadata del response del proveedor y la reporta en `proxy_metadata.cache`. Si el proveedor no reporta caché, retorna `provider_cache_hit: false` sin error.

### 10. `src/auth.py` + `src/middleware/rate_limiter.py` — Capa de Producción
```python
# Auth: Bearer token, dev mode si PROXY_API_KEY vacío
# Rate limit: fixed-window por pseudo-modelo en Valkey
# Middleware order: CORS → Auth → RateLimit → handler
```
**Por qué es crítico:** Sin JWT, sin OAuth, sin dependencias externas. `AuthMiddleware` corre ANTES de `RateLimitMiddleware` para que requests no autenticados no consuman cuota. CORS es el outermost para manejar preflight primero. `PROXY_API_KEY` vacío = dev mode (sin auth). El rate limiter usa fixed-window en Valkey con keys `ratelimit:{pseudo}:{minute}` y expira automáticamente tras 2 minutos.

---

## Diagrama de Arquitectura

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
│         ├── normalize_model_name()  → alias → pseudo-model            │
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
│  Postgres ── conversations, turns, snapshots, capabilities             │
│  Valkey   ── affinity (conv:{id}:model), rate limits                  │
│  arq      ── async compaction (>500K tokens)                          │
└──────────────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        PROVEEDORES LLM                                │
│   Zhipu (GLM) │ DeepSeek │ Qwen │ Google (Gemini) │ Groq │ Ollama     │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 8 Pseudo-Modelos Definidos en `pseudo_models.yaml`

| Pseudo-modelo | Physical models (orden de prioridad) | Uso |
|---------------|--------------------------------------|-----|
| `pensamiento-profundo-caro` | zai/glm-5 → deepseek/deepseek-v4-pro | Razonamiento máximo nivel, bugs imposibles |
| `tareas-avanzadas` | deepseek/deepseek-v4-pro → deepseek-v4-flash → minimax-m2.5 | Caballo de batalla para trabajo extenso |
| `avanzada-vision` | groq/meta-llama/llama-4-scout-17b-16e-instruct → gemini-3.5-flash | Análisis visual, OCR, diagramas |
| `normal` | openrouter/qwen3-max → deepseek/deepseek-v4-flash | Punto de entrada recomendado |
| `deep-flash` | zai/glm-4.5-flash → groq/gpt-oss-20b → deepseek/deepseek-v4-flash | Tareas masivas y simples |
| `flash-lowcost` | zai/glm-4.5-flash → openrouter/qwen3.5-plus → ollama/llama3.2 | Sub-agentes baratos |
| `flash-vision` | gemini/gemini-3.5-flash → ollama/llava | Visión rápida y barata |
| `compactador` | gemini/gemini-3.5-flash (1M ctx) → claude-haiku-4-5 (200K) → glm-4.5-flash (128K) | Solo operación de compactación |

**Model Aliases** (mapean nombres de clientes a pseudo-modelos):
- `gpt-4o` → `normal`
- `o3` / `o4-mini` → `pensamiento-profundo-caro`
- `gpt-4o-mini` → `deep-flash`
- `gemini-2.5-flash` → `avanzada-vision`
- `desconocido` → `normal` (alias `default`)

---

## Configuración: API Keys y Modelos

### Dónde se definen los modelos: `proxy/pseudo_models.yaml`
```yaml
pseudo_models:
  normal:
    display_name: Normal
    physical_models:
      - provider: qwen
        model: openrouter/qwen3-max       # ← ID exacto que LiteLLM espera
        openai_tools_compatible: true     # ← TODOS deben ser true (validado startup)
        parallel_tools: false
        vision: false
      - provider: deepseek
        model: deepseek/deepseek-v4-flash
        parallel_tools: true              # ← fallback con parallel
        tools_strict: true                # ← JSON schemas garantizados
```
- El campo `model` es el identificador **exacto** que LiteLLM espera. Sin prefijos, sin transformación, sin concatenación con `provider`.
- El campo `provider` es **solo metadata**. No se usa para construir el ID.
- `openai_tools_compatible` debe ser `true` en todos los modelos. Si no, el proxy no arranca.

### Dónde se definen las API Keys: `.env` (en el servidor, NUNCA en el cliente)
```bash
# Provider API keys (NUNCA salen del servidor)
ANTHROPIC_API_KEY=sk-ant-xxxx
DEEPSEEK_API_KEY=sk-xxxx
GOOGLE_API_KEY=AIza-xxxx
GROQ_API_KEY=gsk_xxxx
ZHIPUAI_API_KEY=xxxx
ZAI_API_KEY=xxxx
OPENROUTER_API_KEY=sk-or-xxxx

# Proxy auth
PROXY_API_KEY=sk-proxy-generate-with-openssl-rand-hex-32

# CORS
CORS_ORIGINS=https://chat.guzman-lopez.com,vscode-webview://*
```

**El cliente solo maneja:**
```bash
export CESAR_PROXY_URL="https://proxy.tudominio.com/v1"
export CESAR_PROXY_KEY="sk-proxy-xxxx"  # misma que PROXY_API_KEY del servidor
```

---

## Flujo de una Petición (el árbol de decisión completo)

```
POST /v1/chat/completions { model: "gpt-4o", messages: [...], conversation_id: "abc" }
  │
  ├── 1. normalize_model_name("gpt-4o") → alias → "normal"
  │
  ├── 2. ¿Conversación nueva?
  │     Sí → seleccionar primer modelo físico de "normal" (priority 1)
  │         → Valkey: SET conv:abc:physical_model "openrouter/qwen3-max" EX 86400
  │     No  → recuperar modelo físico fijado de Valkey
  │
  ├── 3. detect_turn_capabilities() → has_images, has_tools, has_parallel_tools
  │     → Acumular flags en DB (aditivos, nunca se desactivan)
  │
  ├── 4. ¿Cambió pseudo-modelo respecto al turno anterior?
  │     Sí → validate_switch():
  │           ├── safe    → continuar
  │           ├── warning → warning en proxy_metadata
  │           └── blocked → HTTP 409 + remediation
  │
  ├── 5. ¿Hay parallel tools en el historial?
  │     Sí → get_eligible_models() → filtrar pool a parallel_tools: true
  │           → si modelo fijado no califica → intentar fallback
  │
  ├── 6. ¿Input supera umbral del pseudo-modelo?
  │     Sí + pre_compaction enabled → pre_compact_input() con modelo barato
  │     Sí + pre_compaction disabled → HTTP 400 INPUT_EXCEEDS_THRESHOLD
  │
  ├── 7. ¿Contexto acumulado > trigger_pct?
  │     Sí + continuous_compaction enabled → compactar turnos antiguos en snapshot
  │     → active_messages = [snapshot ~8K] + [turnos recientes ~40K]
  │
  ├── 8. ¿Router LLM habilitado?
  │     Sí → evaluate_complexity() con modelo barato → proxy_metadata.router_suggestion
  │     → NUNCA cambia el modelo, solo informa
  │
  ├── 9. Enviar a LiteLLM (con cache_control si Anthropic)
  │     Error 503/429 → fallback al siguiente modelo del mismo pseudo-modelo
  │     Sin fallback → HTTP 503 ALL_MODELS_FAILED
  │
  └── 10. Respuesta + proxy_metadata + actualizar DB con turno y flags
        proxy_metadata incluye:
        ├── physical_model, pseudo_model, conversation_id
        ├── context_tokens_total, context_usage_pct
        ├── affinity_maintained, fallback_applied
        ├── pre_compaction, continuous_compaction
        ├── images_described, router_suggestion
        ├── context_alert, cache (hit/miss/savings)
        └── warning, tools_filter, capabilities_detected
```

---

## Diagrama de Decisión de Compatibilidad

```
validate_switch(from, to, caps):
  │
  ├── ¿has_images y to no tiene visión?
  │     ├── to.on_downgrade == "auto_describe"  → ⚠️ WARNING (imágenes descritas)
  │     └── to.on_downgrade == "block"          → ❌ BLOCKED + remediation
  │
  ├── ¿has_audio y to no tiene audio?         → ❌ BLOCKED (no hay remedio en v1)
  │
  ├── ¿has_pdf y to no tiene visión?          → ❌ BLOCKED + sugerir extracción texto
  │
  ├── ¿has_video?                             → ❌ BLOCKED (no soportado en v1)
  │
  ├── ¿has_parallel_tools y to sin parallel_tools?
  │     → ❌ BLOCKED + ofrece POST normalize-tools
  │
  ├── ¿contexto total > context_window de to?
  │     → ❌ BLOCKED + ofrece POST /compact
  │
  ├── ¿has_tools y to no tiene openai_tools_compatible?
  │     → ❌ BLOCKED + sugerir pseudo-modelo con tools
  │
  └── Upgrade (to tiene >= capabilities que from) → ✅ SAFE (sin fricción)
```

---

## Mecanismo de Cache (3 Capas)

```
CAPA 1 — Afinidad Valkey:  conv:abc:physical_model = "openrouter/qwen3-max" (TTL 24h)
  → Cada turno usa el MISMO modelo → el proveedor reconoce el prefijo

CAPA 2 — Orden Canónico:  system → tools(sorted) → history → new
  → El prefijo [system + tools + history] es idéntico entre turnos
  → sort_keys=True en JSON serialization

CAPA 3 — Optimización por Proveedor:
  Anthropic:   cache_control breakpoints (system + history)
  OpenAI/DS:   auto-caching (no action needed)
  Gemini:      CachedContent (limitation documented)
  Groq/Zhipu:  no caching

RESULTADO:
  Turno 1:  system + tools + history + query1  → provider caches prefix
  Turno 2:  system + tools + history + query2  → CACHE HIT (prefix unchanged)
  ...
  Turno 20: system + tools + history + query20 → CACHE HIT

  Fallback:  cache DESTROYED → proxy_metadata reporta "previous_cache_destroyed: true"
```

---

## Resumen de Features Completadas

| Feature | Sprint | Estado |
|---------|--------|--------|
| 8 pseudo-modelos con validación startup | 1 | ✅ |
| Afinidad de modelo físico (Valkey 24h) | 1 | ✅ |
| Streaming SSE + proxy_metadata | 1 | ✅ |
| Fallback secuencial (503/429) | 1 | ✅ |
| Detección de capabilities (imágenes, tools, parallel tools) | 2 | ✅ |
| Flags aditivos en DB | 2 | ✅ |
| Validación de compatibilidad (safe/warning/blocked) | 2 | ✅ |
| Remediation en errores 409 | 2 | ✅ |
| Herramientas: formato canónico OpenAI | 3 | ✅ |
| Normalización de parallel tools | 3 | ✅ |
| Edge cases: streaming parcial, thinking, truncation | 3 | ✅ |
| Pre-compactación de input largo | 4 | ✅ |
| Compactación continua (trigger_pct) | 4 | ✅ |
| Snapshots con chaining (superseded_by) | 4 | ✅ |
| Auto-describe imágenes al migrar | 5 | ✅ |
| Router LLM (sugiere downgrade, nunca impone) | 5 | ✅ |
| Degradación manual de imágenes | 5 | ✅ |
| Compactación explícita (POST /compact) | 6 | ✅ |
| Context alerts (60%, 80%, 100%) | 6 | ✅ |
| Audit log (GET /audit-log) | 6 | ✅ |
| arq para compactación async (>500K) | 6 | ✅ |
| Orden canónico de mensajes | 7 | ✅ |
| Anthropic cache_control breakpoints | 7 | ✅ |
| Cache metadata en proxy_metadata | 7 | ✅ |
| Model aliases (gpt-4o → normal) | 7 | ✅ |
| OpenCode config ejemplo | 7 | ✅ |
| Auth Bearer token (PROXY_API_KEY) | 8 | ✅ |
| CORS para browsers | 8 | ✅ |
| Rate limiting por pseudo-modelo | 8 | ✅ |
| Structured JSON logging | 8 | ✅ |
| Métricas (GET /metrics) | 8 | ✅ |
| Caddyfile + systemd services | 8 | ✅ |

---

## Test Suite

```bash
cd proxy
.venv/bin/pytest                                    # 410 tests total (401 + 9 skip)
.venv/bin/pytest -k "not streaming"                 # 389 tests (sin streaming mock)
.venv/bin/pytest tests/test_message_ordering.py     # 10 tests
.venv/bin/pytest tests/test_auth.py                 # 7 tests
.venv/bin/pytest tests/test_rate_limiter.py         # 5 tests
.venv/bin/pytest tests/test_metrics.py              # 6 tests
.venv/bin/pytest tests/test_context_alerts.py       # 12 tests
.venv/bin/pytest tests/test_explicit_compaction.py  # 14 tests
```

---

## Licencia

MIT. Todas las dependencias tienen licencia MIT/BSD/Apache 2.0.
