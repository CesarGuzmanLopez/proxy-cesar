---
filename: round-1-architectural-analysis.md
date: 2026-05-27
---

# Análisis Arquitectónico Completo: proxy-cesar

## 1. PÉRDIDA DE CONTEXTO — POR QUÉ SUCEDE

### 1.1 Dualidad Streaming/No-Streaming (DUPLICACIÓN MASIVA)
**Archivos**: `api/chat.py` vs `service/chat_service.py`
- **No-streaming**: `_handle_non_streaming()` → `process_chat_request()` (chat_service.py)
- **Streaming**: `_handle_streaming_with_db()` → `_stream_response_generator()` (api/chat.py)
- Cada flujo tiene SU PROPIA implementación de:
  - Resolución de modelo (2×)
  - Detección de capacidades (2×)
  - Validación de contenido (2×)
  - Carga de conversación + turns (2×)
  - Auto-describe images (2×)
  - Router LLM (2×)
  - Resolución de physical model + afinidad (2×)
  - Tool filter (2×)
  - Persistencia de turno (2×: chat_service.py:596 vs api/chat.py:668)

**Impacto**: Cualquier bug fix o mejora en un flujo NO se aplica al otro. Esto ha causado bugs históricos (Bug 5, Bug 6, Bug 7 documentados en comentarios).

### 1.2 `build_conversation_messages()` — Reconstrucción O(n) Costosa
**Archivo**: `chat_service.py:488-522`
- Itera TODOS los turns en cada request → O(n) por request
- Reconstruye mensajes desde cero cada vez
- No hay caché de historial compilado
- Para conversaciones largas (>100 turns), esto es miles de operaciones de dict por request

### 1.3 Compactación que NO Reemplaza el Historial
- `POST /compact` crea un snapshot pero el historial completo SIGUE cargándose
- `build_conversation_messages()` ignora completamente los snapshots
- El snapshot solo se añade como mensaje de sistema adicional (compactor/explicit.py:278-286)
- **Bug**: La conversación sigue creciendo en la DB sin límite — no hay poda de turns viejos

### 1.4 Token-Limit Continuation Inconsistente
- Non-streaming: `call_with_fallback()` en chat_service.py:807-877 maneja composite response
- Streaming: `_stream_response_generator()` en api/chat.py:858-944 maneja su propia versión
- Ambos implementan el mismo concepto pero con código totalmente diferente
- Streaming acumula `accumulated_content` en memoria; non-streaming acumula `accumulated_parts`

### 1.5 Auto-Describe Secuencial (Cuello de Botella)
**Archivo**: `multimedia/image_describer.py:199-205`
- Las imágenes se describen UNA POR UNA en un loop secuencial
- `auto_describe_images()` envía cada imagen individualmente al modelo de visión
- Para 10 imágenes: 10 llamadas LLM secuenciales de ~2-5s cada una = 20-50s de latencia
- El usuario ve "pérdida de contexto" porque el proxy tarda demasiado en procesar

---

## 2. CONTAMINACIÓN DE CACHE DE PROVEEDORES

### 2.1 Canonical Message Ordering Incompleto
**Archivo**: `cache/message_ordering.py:99-144`
- `canonicalize_message_order()` ordena system messages primero y tool results después de tool_calls
- Pero solo se aplica UNA VEZ al inicio de `call_with_fallback()` (chat_service.py:761)
- **No se reordena después de auto-describe** — `messages_for_llm` se construye en `process_chat_request()` ANTES de canonicalize
- **No se reordena en el path de streaming** — `active_messages` en `_handle_streaming_with_db()` nunca pasa por canonicalize
- Esto significa que el cache del proveedor se invalida frecuentemente porque el orden de mensajes cambia entre requests

### 2.2 Cache Control Solo Anthropic
**Archivo**: `cache/provider_cache.py:20-22`
- `_PROVIDERS_WITH_CACHE_CONTROL = frozenset({"anthropic"})`
- DeepSeek, Groq, OpenAI tienen cache automático pero NO se aplican breakpoints
- `manage_gemini_cache()` es un stub vacío (línea 126-128)
- **No hay ningún proxy-side response cache** — cada request idéntico va al proveedor

### 2.3 DeepSeek Context Caching — No Optimizado
- DeepSeek usa "Context Caching on Disk" automático
- La documentación de DeepSeek dice que el cache hit requiere que el **prefijo completo** coincida
- El proxy añade [KEYVAULT:hash] placeholders que CAMBIAN entre conversaciones
- El system prompt de KeyVault (línea 95-102 de keyvault.py) se inyecta en cada request
- Cualquier variación en el system prompt rompe el cache prefix de DeepSeek
- **El proxy no aprovecha el "Common prefix detection" de DeepSeek**

### 2.4 Cache Key no Determinístico
- `stable_message_hash()` (message_ordering.py:18-31) se computa pero SOLO se usa para logging
- No hay un cache key real basado en hash de mensajes
- La afinidad (`ValkeyAffinityAdapter`) solo guarda qué physical model usó la conversación
- **No hay integración con el sistema de caching de LiteLLM** (que soporta Redis, Qdrant, S3, etc.)

### 2.5 Cache Destruction Tracking Inconsistente
- `build_cache_destruction_metadata()` (provider_cache.py:193-216) solo LOGEA la destrucción
- No previene la destrucción — solo la reporta
- Cuando hay fallback, se cambia de physical model → todo el cache prefix se pierde
- **No hay estrategia para minimizar cambios de physical model**

---

## 3. PÉRDIDA DE CABECERAS HTTP

### 3.1 KeyVault Reconstruye Response sin Headers
**Archivo**: `middleware/keyvault.py:205-223`
- `_re_inject_non_streaming()` lee el body completo del response
- Construye un NUEVO `JSONResponse` con `headers=dict(response.headers)` (línea 219)
- **Pero**: si el response original tenía un `StreamingResponse`, el re-inject usa `StreamingResponse` correctamente (línea 311-317)
- **Bug sutil**: Para non-streaming, el KeyVault middleware captura el response después de que RateLimit ya añadió headers (X-RateLimit-*), pero KeyVault las pasa correctamente

### 3.2 Forwarding Selectivo de Provider Headers
**Archivo**: `api/chat.py:272-279` y `api/chat.py:568-579`
- Solo 6 headers específicos se forwardean: `x-ratelimit-limit-requests`, `x-ratelimit-remaining-requests`, `x-ratelimit-limit-tokens`, `x-ratelimit-remaining-tokens`, `x-request-id`, `x-cache`
- El resto se meten en `X-Provider-Headers` como JSON (streaming) o NO se incluyen (non-streaming en `_handle_non_streaming()`, línea 272-279)
- **Esto es una pérdida de información significativa** para debugging de proveedores

### 3.3 Orden de Middleware — Confuso
**Archivo**: `main.py:157-182`
- El comentario dice: "1st=KeyVault(inner) → 2nd=RateLimit → 3rd=Auth(middle) → 4th=CORS(outer)"
- Pero el orden de registro es: KeyVault → RateLimit → Auth → CORS
- FastAPI usa LIFO (último registrado es el más externo)
- **Correcto**: CORS(ejecuta primero) → Auth → RateLimit → KeyVault(ejecuta último, más cerca del handler)
- **Problema**: KeyVault modifica el body y RateLimit lee el body — si el orden cambia, RateLimit leería el body ya modificado con placeholders

### 3.4 Headers de Streaming Duplicados/Perdidos
**Archivo**: `api/chat.py:568-579` (streaming) vs `api/chat.py:272-279` (non-streaming)
- Streaming: Los headers se ponen en la `StreamingResponse` (línea 581-624)
- Non-streaming: Los headers se ponen en `JSONResponse` (línea 281)
- **Bug**: Si el streaming falla antes de yield, los headers nunca llegan al cliente

---

## 4. TOOL HANDLING — 7 BUGS ESPECÍFICOS

### Bug 1: Tool Calls en Streaming Acumulan Contenido Incorrecto
**Archivo**: `api/chat.py:840-846`
```python
delta = chunk.choices[0].delta
if delta and hasattr(delta, "content") and delta.content:
    accumulated_content += delta.content
```
- Esto SOLO acumula `content`, NO `tool_calls`
- En streaming con tool_calls, `delta.tool_calls` se ignora
- `accumulated_content` solo contiene texto, nunca las llamadas a herramientas

### Bug 2: `_truncate_tool_results` Modifica los Mensajes Originales
**Archivo**: `api/chat.py:658-665`
```python
def _truncate_tool_results(messages: list[dict] | None) -> None:
    for msg in messages or []:
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            truncated = truncate_tool_result(content)
            if truncated != content:
                msg["content"] = truncated
```
- **Modifica el dict original en-place**, no una copia
- Si el mismo mensaje se reusa (ej. en retry), el tool result ya está truncado

### Bug 3: Parallel Tools Filter en Streaming — No Re-evaluado
- `_filter_eligible_models()` se llama UNA VEZ al inicio del streaming
- Si durante el streaming se detecta que el modelo no soporta parallel tools, el filtro ya pasó
- **No hay re-evaluación dinámica** de elegibilidad durante el streaming

### Bug 4: `validate_tool_call_ids` Solo en Non-Streaming
- `_process_tool_metadata()` (chat_service.py:1013-1044) valida tool call IDs
- `_build_turn_tool_metadata()` (api/chat.py:628-655) también valida
- Pero en streaming con token-limit continuation, los tool calls del primer modelo parcial se PIERDEN
- `_stream_response_generator()` no extrae tool calls de los chunks intermedios

### Bug 5: `tool_choice: "required"` no se Respeta en Fallback
- `enforce_tool_choice()` verifica si el modelo respetó `tool_choice: "required"`
- Pero si hay fallback a otro modelo, el nuevo modelo NO recibe `tool_choice`
- `call_kwargs` se pasa al siguiente modelo (api/chat.py:920) pero `tool_choice` puede no estar presente

### Bug 6: Tool Definitions No se Ordenan Consistentemente
- En `call_with_fallback()` (chat_service.py:771-773): `sort_tool_definitions(raw_tools)`
- En streaming, los tools viajan en `ctx.call_kwargs` (api/chat.py:611-619)
- **No se garantiza** que `sort_tool_definitions` se aplique en cada llamada de continuación

### Bug 7: `call_kwargs` Contaminado en Continuación Streaming
**Archivo**: `api/chat.py:910-922`
```python
cont_messages = list(ctx.messages or [])
cont_messages.append({"role": "assistant", "content": accumulated_content})
new_stream, skip_reason = await _try_physical_model(
    next_phys,
    canonicalize_message_order(cont_messages),
    True,  # stream
    ctx.call_kwargs or {},  # ← Se pasan los kwargs originales sin limpiar
    ...
)
```
- `ctx.call_kwargs` contiene los kwargs ORIGINALES incluyendo `tools` y `tool_choice`
- Si el modelo original no soporta parallel tools, el siguiente modelo tampoco recibe el hint

---

## 5. COMPARACIÓN CON ESTÁNDARES DE LA INDUSTRIA

### 5.1 LiteLLM Proxy — Cache Integrado
- LiteLLM Proxy oficial tiene caching Redis/Qdrant/S3 integrado
- Este proxy implementa su propio sistema de cache AD-HOC (ValkeyAffinityAdapter)
- **Brecha**: LiteLLM ya soporta `cache_params` con TTL, namespace, Redis Cluster, Redis Sentinel
- **Oportunidad**: Usar `litellm_cache` en vez de cache manual

### 5.2 DeepSeek Context Caching — Prefijo Determinístico
- DeepSeek cachea en disco automáticamente cuando detecta prefijos comunes
- **Recomendación DeepSeek**: Mantener system prompts consistentes entre requests
- **Problema actual**: Cada request puede tener system prompts diferentes (KeyVault + router LLM)
- **Solución**: Cachear el system prompt compilado y solo cambiar el último mensaje de usuario

### 5.3 Groq — Cache Automático de 2h
- Groq tiene cache automático con 2h TTL en modelos como GPT-OSS
- **Problema**: El proxy no reporta correctamente los cache hits de Groq
- `build_cache_metadata()` no extrae correctamente `prompt_tokens_details` para Groq

### 5.4 Anthropic — Cache Control Breakpoints
- Anthropic soporta hasta 4 breakpoints `cache_control` por request
- El proxy implementa 2 breakpoints (system message + penúltimo mensaje)
- **Brecha**: No se aprovechan los 4 breakpoints permitidos
- **Brecha**: Los breakpoints se aplican en mensajes con string content, pero Anthropic requiere content list format

### 5.5 OpenRouter — Headers de Proveedor
- OpenRouter envía `X-OpenRouter-*` headers y `X-Title-Request-*`
- El proxy solo forwardea 6 headers específicos
- **Pérdida de información**: Headers de rate limiting de OpenRouter se pierden

---

## 6. BUGS ARQUITECTÓNICOS CRÍTICOS

### 6.1 arq Pool Misconfigured — Async Compaction Roto
**Archivo**: `tasks/arq_app.py:85-97`, `main.py:124-135`
- `create_arq_pool()` no pasa `db_session_factory` ni `config` al context de arq
- `compact_conversation_async()` espera `ctx.get("db_session_factory")` y `ctx.get("config")`
- **Resultado**: Cualquier compaction >500K tokens FALLARÁ con "arq context missing"

### 6.2 Inline Migration + Alembic Split-Brain
**Archivo**: `main.py:95-110`
- main.py ejecuta ALTER TABLE para añadir columnas
- Los archivos de migración Alembic (0001-0003) usan tipos PostgreSQL específicos
- La DB es SQLite que NO soporta esos tipos
- **Resultado**: Las migraciones Alembic son inejecutables; las migraciones inline son frágiles

### 6.3 Domain Types vs Config Types Duplicados
- `domain/pseudo_model.py` tiene dataclasses de dominio PURAS
- `config/pseudo_models.py` tiene schemas Pydantic DUPLICADOS
- No hay función de conversión entre ellos
- **Resultado**: `domain/pseudo_model.py` es código muerto; el servicio importa de `config/pseudo_models` directamente

### 6.4 SSL_CERT_FILE Duplicado
- `adapters/litellm/client.py:10-25` y `main.py:13-25`
- Ambos verifican SSL_CERT_FILE al importar
- **Race condition**: Si main.py se importa primero, client.py es no-op. Si client.py se importa primero, main.py sobreescribe.
- Depende del orden de importación de Python, que es frágil

### 6.5 Env Var Inconsistente
- `settings.py:19`: `proxy_api_key`
- `auth.py:40`: `os.getenv("PROXY_API_KEY")`
- La variable de entorno `PROXY_API_KEY` mapea a `proxy_api_key` por Pydantic, pero `auth.py` no usa settings

### 6.6 Global Rate Limiter (No Per-User)
- `rate_limiter.py:108`: `ratelimit:{pseudo_model}:{minute_bucket}`
- **No hay diferenciación por usuario/IP** — un solo usuario puede agotar el límite para todos
- El modelo se extrae de los primeros 2KB del body (línea 24) — si el modelo aparece después (ej. payload muy grande con system prompt largo), no se detecta

### 6.7 Orphan Snapshot Records
- `compactor/explicit.py:152-163`: El snapshot se crea en DB ANTES de la llamada API
- Si la llamada API falla (línea 193-201), el snapshot queda huérfano
- `tokens_after=0` y `snapshot_content=""` — datos inconsistentes

---

## 7. MÉTRICAS — IN-MEMORY Y NO THREAD-SAFE

### 7.1 Race Conditions
**Archivo**: `api/metrics.py:33-76`
- `MetricsStore` es una clase simple con dicts y contadores
- Sin locks, sin atomics, sin estructuras thread-safe
- En producción con múltiples workers/threads, los contadores se corrompen

### 7.2 `record_error` Nunca se Llama
- `MetricsStore.record_error()` definido en línea 70
- **Nunca invocado** en ningún lugar del código
- Las métricas de error siempre muestran 0

### 7.3 Métricas Perdidas en Restart
- Todos los contadores son in-memory (reinician en cada deploy)
- No hay persistencia de métricas a DB o Valkey
- No hay logging estructurado de métricas para ingestión externa (Prometheus, Datadog, etc.)

---

## 8. ESTÁNDARES MODERNOS vs IMPLEMENTACIÓN ACTUAL

| Área | Estándar Moderno | Implementación Actual | Gap |
|------|------------------|----------------------|-----|
| **Streaming** | Single code path con wrapper | Duplicado completo | CRÍTICO |
| **Cache Provider** | Response cache en Redis/proxy | Solo prefix cache Anthropic | ALTO |
| **Cache Key** | Hash-based deterministic key | No implementado | ALTO |
| **Context Window** | Sliding window + summarization | Threshold fijo + compact manual | ALTO |
| **Rate Limiting** | Per-user/IP + token bucket | Per-pseudo-model global | MEDIO |
| **Headers** | Forward completo de provider headers | Selectivo (6 headers) | MEDIO |
| **Metrics** | Prometheus + structured logging | In-memory + no thread-safe | ALTO |
| **Error Handling** | Result monad consistente | Mixto (monad + HTTPException) | MEDIO |
| **Tool Calls** | Validación en ambos paths | Validación parcial | ALTO |
| **DB Migrations** | Alembic puro | Inline + Alembic split-brain | CRÍTICO |

---

## 9. PLAN DE MEJORA RECOMENDADO

### Prioridad Inmediata (Semana 1-2)
1. **Unificar path streaming/no-streaming** — Extraer lógica común a funciones shared
2. **Configurar arq pool correctamente** — Pasar db_session_factory y config
3. **Fix `record_error`** — Integrar en todos los catch de HTTPException
4. **Fix rate limiter** — Añadir per-user/IP dimension

### Prioridad Alta (Semana 3-4)
5. **Implementar proxy-side response cache** — Usar Valkey con hash-based keys
6. **Optimizar cache DeepSeek/Anthropic** — System prompt estable + breakpoints completos
7. **Forward completo de headers** — Pasar todos los provider headers al cliente
8. **Normalizar error handling** — Result monad consistente en toda la service layer

### Prioridad Media (Semana 5-6)
9. **Poda de historial en compactación** — Turns viejos a archivo/eliminación
10. **Parallel image description** — Batch de imágenes simultáneas
11. **Métricas a Prometheus/OTEL** — Reemplazar MetricsStore
12. **Unificar domain/config types** — Eliminar duplicación

### Prioridad Baja (Refinamiento continuo)
13. **Tool call validation en streaming continuation**
14. **Auto-scaling de rate limits basado en uso histórico**
15. **Dynamic tool level filtering durante streaming**
