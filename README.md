# proxy-cesar

**Proxy LLM multi-modelo determinista.** Proxy HTTP transparente entre clientes LLM y múltiples proveedores. Expone **pseudo-modelos** abstractos que mapean a modelos físicos concretos con fallback automático, validación de compatibilidad de contenido, normalización de tools, compactación explícita, alertas de contexto, auditoría, optimización de caché por proveedor, auth, rate limiting y métricas estructuradas.

> **Paquetería 100% libre.** MIT, BSD, Apache 2.0.
> **El proxy no decide. Valida, informa y ejecuta lo que el usuario ordena.**
> **Cuando algo no puede continuar, lo dice claramente y ofrece opciones.**
> **Cuando algo puede continuar, lo hace sin fricción.**
> **100% tipado y determinista.** La única no-determinista es la respuesta del modelo.

---

## Objetivos del Sistema

1. **Unificar múltiples proveedores LLM** detrás de una API OpenAI-compatible única
2. **Maximizar cache hits** mediante afinidad de modelo físico y orden canónico de mensajes
3. **Validar compatibilidad** de contenido multimedia y tools al cambiar de pseudo-modelo
4. **Delegar imágenes a tools** cuando el modelo no tiene visión pero el usuario provee tools compatibles
5. **Proteger secrets** con KeyVault middleware — intercepta API keys antes de llegar al LLM, las reemplaza con placeholders, y las reinyecta en la respuesta
6. **Informar** cada decisión del proxy en `proxy_metadata` — nunca silencio

---

## Decisiones Arquitectónicas Clave

| Decisión | Alternativa | Por qué esta gana |
|----------|-------------|-------------------|
| Error handling con `Result[T,E]` monad | Exceptions | Errores como datos — el `match/case` fuerza al caller a manejar ambos casos |
| Validación de configuración fail-fast | Runtime filtering | Si `pseudo_models.yaml` es inválido, el proxy no arranca. Nunca errores en producción |
| Sin compactación automática | Compactación continua + pre-compactación | Si se supera el umbral, error explícito. El usuario decide vía `POST /compact` |
| Delegación imagen→tool | Rechazar o degradar | Si el modelo no tiene visión pero hay tools, se transforma `image_url` a texto URL + instrucción |
| KeyVault middleware | Sanitización en servicio | Intercepta secrets en la capa más cercana al handler, nunca modifica firmas de servicios |
| Afinidad como Protocol abstracto | Valkey directo | `AffinityPort` permite testear sin Valkey real y cambiar backend sin tocar dominio |
| Fallback `by_context_window` | Single model | Múltiples modelos físicos por pseudo-modelo; si el contexto excede, se salta ese modelo |
| Direct model passthrough | Solo pseudo-models | Cualquier modelo `ollama/xxx`, `lmstudio/xxx` se puede llamar directo sin configuración |
| arq en vez de Celery | Celery | async-native, MIT, 700 líneas vs 50K+. Misma autora de Pydantic |
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

## 10 Pseudo-Modelos en `proxy/pseudo_models.yaml`

Cada uno con límite de tokens (`input_token_threshold`). Al superar el límite se retorna error explícito — el usuario debe usar `POST /conversations/{id}/compact` para compactar manualmente.

| Pseudo-modelo | Límite | Modelos físicos | Imágenes |
|---|---|---|---|
| `pensamiento-profundo-caro` | **120K** | `deepseek/deepseek-v4-pro` → `openrouter/google/gemini-3.5-flash` | auto_describe |
| `tareas-avanzadas` | **200K** | `deepseek/deepseek-v4-pro` → `deepseek/deepseek-v4-flash` | block |
| `vision` | **120K** | `groq/meta-llama/llama-4-scout-17b-16e-instruct` | auto_describe |
| `normal` | **500K** | `deepseek/deepseek-v4-flash` → `openrouter/google/gemini-3.1-flash-lite` | block |
| `normal-gratis` | **200K** | `openrouter/nvidia/nemotron-3-super-120b-a12b:free` → `groq/qwen/qwen3-32b` | auto_describe |
| `massive-fast` | **131K** | `groq/openai/gpt-oss-20b` | block |
| `flash-lowcost` | **128K** | `openrouter/google/gemini-3.1-flash-lite` | block |
| `audio` | **131K** | `groq/whisper-large-v3` → `groq/whisper-large-v3-turbo` | block |
| `imagen` | **1K** | `pruna/p-image` | block |
| `compactador` | **20M** | `groq/openai/gpt-oss-20b` (≤131K) → `deepseek/deepseek-v4-flash` (1M) | auto_describe |

### Model Aliases

| Alias | Resuelve a |
|---|---|
| `gpt-4o` | `normal` |
| `gpt-4o-mini` | `normal` |
| `gpt-4.1` | `tareas-avanzadas` |
| `o3` / `o4-mini` | `pensamiento-profundo-caro` |
| `gemini-2.5-flash` | `vision` |
| `claude-haiku-3-5-20241022` | `flash-lowcost` |
| `default` | `normal` |

---

## Modelos Locales (Passthrough Directo)

Cualquier modelo con prefijo de proveedor se puede llamar **directamente** sin estar en `pseudo_models.yaml`:

```
ollama/llama3.2      → LiteLLM → Ollama
ollama/llava         → LiteLLM → Ollama
lmstudio/my-model    → LiteLLM → LM Studio
local/cualquier-mo   → LiteLLM → proveedor local
```

Sin límites de threshold. Usan el sistema de conversaciones y soportan `POST /compact` para compactación explícita.

---

## KeyVault — Protección de Secrets

El middleware KeyVault intercepta todas las requests `POST /v1/chat/completions`:

1. **Detección**: 22 patrones de secrets (API keys OpenAI/Anthropic/GitHub/AWS, claves PEM, SSH, wallets crypto, JWT, base64 larga)
2. **Almacenamiento**: Guarda en Valkey `keyvault:{conv}:{hash}` con TTL de 1 hora
3. **Sanitización**: Reemplaza secrets por `[KEYVAULT:abc12345]` en los mensajes
4. **System prompt**: Inyecta instrucción para que el LLM use placeholders
5. **Reinyección**: Reemplaza placeholders por valores reales en la respuesta (soporta streaming y no-streaming)

El LLM **nunca ve** las keys reales. El cliente **siempre ve** las keys reales.

---

## Delegación de Imágenes a Tools

Cuando el usuario envía una imagen a un pseudo-modelo sin visión:

| Escenario | Comportamiento |
|---|---|
| Imagen + modelo sin visión + **sin tools** | `400 IMAGES_NOT_SUPPORTED_BY_PSEUDO_MODEL` |
| Imagen + modelo sin visión + **con tools compatibles** | Se reemplaza `image_url` por texto URL + instrucción para que la tool procese la imagen |

La delegación escanea las definiciones de tools buscando un parámetro de tipo `string` que pueda aceptar la URL de la imagen. Si encuentra match, transforma el mensaje en vez de rechazar.

---

## Compactación Explícita

No hay compactación automática. Si el input supera el umbral del pseudo-modelo:

```
400 INPUT_EXCEEDS_THRESHOLD — Usa POST /conversations/{id}/compact
```

El endpoint `POST /conversations/{id}/compact`:
1. Escanea el historial de la conversación
2. Si encuentra imágenes/audio, delega a un modelo con visión/audio para describirlos
3. Envía todo a un modelo compactador con prompt estructurado
4. Genera un snapshot Markdown con secciones: estado del problema, decisiones técnicas, código producido, items pendientes
5. Retorna metadata de la compactación

---

## Provider Caching

| Proveedor | Tipo | Ahorro |
|---|---|---|
| **Anthropic** | `cache_control` breakpoints | Por breakpoint |
| **DeepSeek** | Prefix automático | Automático, sin costo |
| **Groq** | Prefix automático | 50% en cacheados, 2h TTL |
| **OpenRouter** | Delega al upstream | Depende del modelo final |
| **Ollama** (local) | Sin caché | N/A |

---

## Middleware Chain (orden de ejecución)

```
CORS → Auth → RateLimit → KeyVault → Handler
```

1. **CORS** — Permite orígenes configurados
2. **Auth** — Bearer token vs `PROXY_API_KEY` (desactivado en dev si no hay key)
3. **RateLimit** — Ventana fija de 1 minuto por pseudo-modelo vía Valkey
4. **KeyVault** — Detecta secrets, los almacena, sanitiza el body, reinyecta en respuesta

---

## API Endpoints

| Método | Path | Descripción |
|---|---|---|
| `POST` | `/v1/chat/completions` | Chat completion (streaming y no-streaming) |
| `GET` | `/v1/models` | Lista pseudo-modelos disponibles |
| `GET` | `/health` | Health check |
| `GET` | `/conversations/{id}` | Obtener conversación |
| `GET` | `/conversations/{id}/compatible-models` | Modelos compatibles con la conversación |
| `GET` | `/conversations/{id}/tools-compatibility` | Compatibilidad de tools |
| `POST` | `/conversations/{id}/normalize-tools` | Normalizar tool calls |
| `POST` | `/conversations/{id}/compact` | Compactar historial explícitamente |
| `GET` | `/conversations/{id}/audit-log` | Log de eventos de la conversación |
| `GET` | `/metrics` | Métricas agregadas del proxy |

---

## proxy_metadata

Cada respuesta de chat incluye `proxy_metadata` con:

| Campo | Descripción |
|---|---|
| `physical_model` | Modelo físico que respondió |
| `pseudo_model` | Pseudo-modelo solicitado |
| `affinity_maintained` | `true` si se re-usó el mismo modelo físico |
| `fallback_applied` | `true` si hubo fallback |
| `capabilities_detected` | `has_images`, `has_tools` |
| `images_described` | Número de imágenes descritas automáticamente |
| `router_suggestion` | Sugerencia de Router LLM (si aplica) |
| `context_alert` | Alerta de contexto (niveles: moderate, high, unusable) |
| `cache` | Metadata de caché del provider |

---

## Tests

```bash
cd proxy
poetry run pytest tests/ -q --tb=short
```

**270 tests**, 0 fallos esperados (excluyendo e2e/streaming/fallback/stress).

---

## Verificación Automática

```bash
./test_hard.sh   # Inicia servidor + ejecuta todas las HUs
```

---

## Errores del Sistema

| Código | Error | Causa |
|---|---|---|
| 400 | `IMAGES_NOT_SUPPORTED_BY_PSEUDO_MODEL` | Imagen sin modelo visión ni tools compatibles |
| 400 | `AUDIO_NOT_SUPPORTED` | Audio no soportado en v1 |
| 400 | `PDF_NOT_SUPPORTED` | PDF sin modelo visión |
| 400 | `VIDEO_NOT_SUPPORTED` | Video no soportado en v1 |
| 400 | `INPUT_EXCEEDS_THRESHOLD` | Input supera límite del pseudo-modelo |
| 400 | `CONTEXT_UNUSABLE` | Contexto al 100% |
| 401 | `MISSING_AUTH` | Token inválido o faltante |
| 409 | `PSEUDO_MODEL_INCOMPATIBLE` | Switch bloqueado por capacidades |
| 413 | `CONTEXT_TOO_LARGE_FOR_ALL_MODELS` | Todos los modelos físicos excedidos |
| 429 | `RATE_LIMIT_EXCEEDED` | Límite de tasa excedido |
| 502 | `PROXY_ERROR` | Error interno del proxy |
| 503 | `ALL_MODELS_FAILED` | Todos los modelos físicos fallaron |
