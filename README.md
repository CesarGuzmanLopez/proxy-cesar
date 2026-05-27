# proxy-cesar

**Proxy LLM multi-modelo determinista.** Proxy HTTP transparente entre clientes LLM y múltiples proveedores. Expone **10 pseudo-modelos** abstractos que mapean a modelos físicos concretos con fallback automático, KeyVault (protección de secrets), Blob Vault (transformación de contenido no soportado), compactación explícita, y métricas estructuradas.

> **Paquetería 100% libre.** MIT, BSD, Apache 2.0.
> **El proxy no decide. Valida, informa y ejecuta lo que el usuario ordena.**
> **Cuando algo no puede continuar, lo dice claramente.**
> **Cuando algo puede continuar, lo hace sin fricción.**
> **100% tipado y determinista.**
> **Sin dependencia de Docker — Redis nativo en puerto 6380, SQLite como DB.**

---

## Estado en producción

**URL:** `https://chat.guzman-lopez.com`

| Componente | Estado |
|---|---|
| Proxy (puerto 9110) | ✅ Activo |
| 10 pseudo-modelos | ✅ Todos funcionales |
| Redis nativo (puerto 6380) | ✅ Conectado |
| SQLite | ✅ Conectado |
| Deploy | ✅ GitHub Actions automático |

## Proveedores

| Proveedor | Tipo | Modelos |
|---|---|---|
| **[OpenCode Go](https://opencode.ai/docs/es/go)** | Suscripción $10/mes | 8 modelos vía API OpenAI + Anthropic |
| **DeepSeek** | API directa | v4-pro, v4-flash (fallbacks) |
| **Groq** | API directa | whisper (asistente de audio, fallback visión) |
| **OpenRouter** | API directa | nemotron free (normal-gratis) |

## Documentación oficial de proveedores

- [OpenCode Go Docs](https://opencode.ai/docs/es/go) — Modelos, límites, endpoints
- [OpenCode Go API](https://opencode.ai/zen/go/v1/models) — Lista de modelos disponibles
- [LiteLLM Docs](https://docs.litellm.ai/docs) — Provider translation layer
- [LiteLLM OpenAI Compatible](https://docs.litellm.ai/docs/providers/openai_compatible) — Custom endpoint config

---

## 10 Pseudo-Modelos

| # | Pseudo-modelo | Primary | Fallback(s) | Visión | Tools | Límite | Go? |
|---|---|---|---|---|---|---|---|
| 1 | `pensamiento-profundo-caro` | qwen3.7-max (Go) | deepseek-v4-pro | No | strict | 120K | ✅ |
| 2 | `pensamiento-rapido` | qwen3.6-plus (Go) | deepseek-v4-flash | No | sí | 120K | ✅ |
| 3 | `tareas-avanzadas` | kimi-k2.6 (Go) | v4-pro → v4-flash | No | strict | 200K | ✅ |
| 4 | `codigo-preciso` | mimo-v2-pro (Go) | mimo-v2.5-pro → v4-flash | No | strict | 200K | ✅ |
| 5 | `vision` | mimo-v2-omni (Go) | llama-4-scout (Groq) | ✅ | sí | 120K | ✅ |
| 6 | `normal` | kimi-k2.5 (Go) | deepseek-v4-flash | No | strict | 500K | ✅ |
| 7 | `normal-gratis` | nemotron (OpenRouter) | qwen3-32b (Groq) | No | no | 200K | ❌ |
| 8 | `massive-fast` | minimax-m2.7 (Go) | gpt-oss-20b (Groq) | No | sí | 131K | ✅ |
| 9 | `flash-lowcost` | qwen3.5-plus (Go) | deepseek-v4-flash | No | sí | 128K | ✅ |
| 10 | `compactador` | glm-5.1 (Go) | gpt-oss-20b → v4-flash | No | sí | ∞ | ✅ |

### Model Aliases

| Alias | Resuelve a |
|---|---|
| `gpt-4o` / `gpt-4o-mini` | `normal` |
| `gpt-4.1` | `tareas-avanzadas` |
| `o3` | `pensamiento-profundo-caro` |
| `o4-mini` | `pensamiento-rapido` |
| `gemini-2.5-flash` | `vision` |
| `gemini-2.5-pro` | `codigo-preciso` |
| `claude-sonnet-4-20250514` | `codigo-preciso` |
| `claude-haiku-3-5-20241022` | `flash-lowcost` |
| `default` | `normal` |

---

## Quick Start

```bash
# 1. Clonar e instalar
cd proxy
python3.14 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Configurar
cp .env.example .env   # Editar con API keys
# Redis nativo en puerto 6380 (VALKEY_URL=valkey://localhost:6380)

# 3. Ejecutar
python -m src.main  # Puerto 9110
```

**Test local:**
```bash
curl http://localhost:9110/health
curl http://localhost:9110/v1/models
curl -X POST http://localhost:9110/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"normal","messages":[{"role":"user","content":"Hola"}]}'
```

**Test contra producción:**
```bash
curl https://chat.guzman-lopez.com/health
curl -X POST https://chat.guzman-lopez.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <tu-token>" \
  -d '{"model":"normal","messages":[{"role":"user","content":"Hola"}]}'
```

---

## Flujo de una llamada

```
POST /v1/chat/completions { model, messages, tools }
         │
    ┌────▼─────────────────────────────────────────────────┐
    │ 1. CORS → Auth → RateLimit → KeyVault middleware     │
    └──────────────────────────────────────────────────────┘
         │
    ┌────▼─────────────────────────────────────────────────┐
    │ 2. Resolver pseudo-modelo (pseudo_models.yaml)       │
    │    "normal" → kimi-k2.5 (Go primary)                 │
    └──────────────────────────────────────────────────────┘
         │
    ┌────▼─────────────────────────────────────────────────┐
    │ 3. Threshold Guard: ¿input > límite? → 400 error     │
    └──────────────────────────────────────────────────────┘
         │
    ┌────▼─────────────────────────────────────────────────┐
    │ 4. validate_incoming_content: ¿imagen/audio/pdf?      │
    │    ¿Modelo lo soporta? → No → Blob Vault              │
    └──────────────────────────────────────────────────────┘
         │
    ┌────▼─────────────────────────────────────────────────┐
    │ 5. Blob Vault (si aplica):                            │
    │    Imagen → describe con modelo visión → [BLOB:hash]  │
    │    Audio  → transcribe con whisper → [BLOB:hash]      │
    │    PDF    → extrae texto con PyMuPDF → [BLOB:hash]    │
    └──────────────────────────────────────────────────────┘
         │
    ┌────▼─────────────────────────────────────────────────┐
    │ 6. Fallback loop:                                     │
    │    for model in physical_models:                      │
    │      ¿context_window >= estimated_tokens? → skip      │
    │      call_litellm(model, api_base, api_key, msgs)     │
    │      ¿Success? → return                               │
    │      ¿429/503/401/404/400? → next                     │
    │    All failed → 502 ALL_MODELS_FAILED                 │
    └──────────────────────────────────────────────────────┘
         │
    ┌────▼─────────────────────────────────────────────────┐
    │ 7. Post-procesamiento:                                │
    │    reasoning_content → content (si content vacío)     │
    │    normalise_stream_chunk (streaming)                 │
    └──────────────────────────────────────────────────────┘
         │
         ▼
    Response JSON + proxy_metadata
```

---

## Endpoints en Go — arquitectura clave

OpenCode Go expone **dos endpoints** según el modelo:

| Tipo | Endpoint | Prefijo LiteLLM | `api_base` en config |
|---|---|---|---|
| OpenAI-compat | `/chat/completions` | `openai/` | `https://opencode.ai/zen/go/v1` |
| Anthropic-compat | `/messages` | `anthropic/` | `https://opencode.ai/zen/go` (sin `/v1`!) |

**Modelos OpenAI-compat:** `kimi-k2.5`, `kimi-k2.6`, `mimo-v2-omni`, `mimo-v2-pro`, `mimo-v2.5-pro`, `minimax-m2.7`, `minimax-m2.5`, `qwen3.5-plus`, `qwen3.6-plus`, `glm-5.1`, `glm-5`

**Modelos Anthropic-compat:** `qwen3.7-max` (único que requiere `/messages`)

> **Importante:** LiteLLM añade `/v1/messages` al `api_base` para Anthropic. Por eso NO debe incluir `/v1`.

---

## KeyVault — Protección de Secrets

El middleware KeyVault intercepta `POST /v1/chat/completions`:

1. **Detección**: 22 patrones de secrets (API keys, claves PEM, SSH, wallets crypto, JWT)
2. **Almacenamiento**: Redis `keyvault:{conv}:{hash}` con TTL 1h (en memoria si Redis no disponible)
3. **Sanitización**: Reemplaza secrets por `[KEYVAULT:abc12345]`
4. **Reinyección**: Placeholders → valores reales en respuesta (streaming + no-streaming)

El LLM **nunca ve** las keys reales. El cliente **siempre ve** las keys reales.

---

## Blob Vault — Contenido no soportado

Cuando un modelo no soporta el tipo de contenido recibido:

| Tipo | Transformación | Modelo helper |
|---|---|---|
| Imagen | Describe con modelo visión → `[BLOB:hash:image/png \| 24 KB\ndesc]` | `_find_model_with_capability("vision")` |
| Audio | Transcribe con whisper → `[BLOB:hash:audio/wav \| 50 KB\ntranscripción]` | `_find_model_with_capability("audio")` |
| PDF | Extrae texto con PyMuPDF → `[BLOB:hash:app/pdf \| 100 KB\ntexto]` | N/A (solo PyMuPDF) |
| Video | ❌ Bloqueado en `validate_incoming_content` | N/A |

Blobs almacenados en Redis (24h TTL). El proxy **nunca inspecciona** las tools del usuario.

---

## Compactación Explícita

Sin compactación automática. Si el input supera el umbral:

```
400 INPUT_EXCEEDS_THRESHOLD — Usa POST /conversations/{id}/compact
```

`POST /conversations/{id}/compact`:
1. Escanea historial → describe imágenes/audio con modelos helper
2. Compacta con modelo compactador (glm-5.1 Go)
3. Genera snapshot Markdown: estado, decisiones, código, pendientes

---

## Provider Caching

| Proveedor | Tipo | Ahorro |
|---|---|---|
| Anthropic | `cache_control` breakpoints | Por breakpoint |
| DeepSeek | Prefix automático | Automático |
| Groq | Prefix automático | 50% en cacheados, 2h TTL |
| OpenRouter | Delega al upstream | Depende del modelo final |

---

## Middleware Chain

```
CORS → Auth → RateLimit → KeyVault → Handler
```

1. **CORS** — Orígenes configurados
2. **Auth** — Bearer token vs `PROXY_API_KEY` (dev mode si no hay key)
3. **RateLimit** — Ventana fija 1 min por pseudo-modelo vía Redis
4. **KeyVault** — Detecta/almacena/sanitiza/reinyecta secrets

---

## API Endpoints

| Método | Path | Descripción |
|---|---|---|
| `POST` | `/v1/chat/completions` | Chat completion (streaming + no-streaming) |
| `GET` | `/v1/models` | Lista pseudo-modelos |
| `GET` | `/health` | Health check |
| `GET` | `/conversations/{id}` | Obtener conversación |
| `GET` | `/conversations/{id}/compatible-models` | Todos los modelos son compatibles (switch libre) |
| `GET` | `/conversations/{id}/tools-compatibility` | Compatibilidad de tools |
| `POST` | `/conversations/{id}/normalize-tools` | Normalizar parallel tool calls |
| `POST` | `/conversations/{id}/compact` | Compactar historial |
| `GET` | `/conversations/{id}/audit-log` | Log de eventos |
| `GET` | `/metrics` | Métricas agregadas |

---

## Environment

| Variable | Default | Descripción |
|---|---|---|
| `PROXY_PORT` | `9110` | Puerto HTTP |
| `PROXY_API_KEY` | — | Bearer token (vacío = dev mode) |
| `DATABASE_URL` | `sqlite+aiosqlite:///./proxy.db` | SQLite local |
| `VALKEY_URL` | `valkey://localhost:6380` | Redis nativo (no Docker) |
| `KEYCLAW_ENABLED` | `false` | KeyClaw deshabilitado |
| `OPENCODE_API_KEY` | — | API key de OpenCode Go |
| `DEEPSEEK_API_KEY` | — | Fallback DeepSeek |
| `GROQ_API_KEY` | — | Fallback Groq |
| `OPENROUTER_API_KEY` | — | OpenRouter (normal-gratis) |

---

## proxy_metadata

Cada respuesta incluye:

| Campo | Descripción |
|---|---|
| `physical_model` | Modelo físico que respondió |
| `pseudo_model` | Pseudo-modelo solicitado |
| `fallback_applied` | `true` si hubo fallback |
| `capabilities_detected` | `has_images`, `has_tools` |
| `images_described` | Imágenes descritas automáticamente |
| `context_alert` | Alerta de contexto (normal/moderate/high/unusable) |
| `cache` | Metadata de caché del provider |
| `tools_filter_applied` | `true` si se filtró por parallel tools |

---

## Tests

```bash
cd proxy
poetry run pytest tests/ -q --tb=short
```

---

## Deploy

GitHub Actions despliega a `chat.guzman-lopez.com` en cada push a `main`:

- Config: [.github/workflows/deploy.yml](.github/workflows/deploy.yml)
- Usuario del servicio: `proxy` (fijo, no dinámico)
- Redis: nativo en puerto 6380
- DB: SQLite preservada entre deploys (backup/restore automático)
- Verificación: health check post-deploy

**Secrets requeridos:** `PLATA_HOST`, `PLATA_USER`, `PLATA_SSH_KEY`, `OPENCODE_API_KEY`, `GROQ_API_KEY`, `DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY`, `PRUNA_API_KEY`, `PROXY_API_KEY`

---

## Errores del Sistema

| Código | Error | Causa |
|---|---|---|
| 400 | `INPUT_EXCEEDS_THRESHOLD` | Input supera límite del pseudo-modelo |
| 400 | `CONTEXT_UNUSABLE` | Contexto al 100%, requiere compactación |
| 400 | `PARALLEL_TOOLS_NOT_SUPPORTED` | Modelo sin parallel tools |
| 401 | `MISSING_AUTH` | Token inválido o faltante |
| 413 | `CONTEXT_TOO_LARGE_FOR_ALL_MODELS` | Todos los modelos excedidos |
| 429 | `RATE_LIMIT_EXCEEDED` | Límite de tasa |
| 502 | `ALL_MODELS_FAILED` | Todos los modelos físicos fallaron |
| 502 | `PROXY_ERROR` | Error interno del proxy |
