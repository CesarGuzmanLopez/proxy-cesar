# proxy-cesar

**Proxy LLM multi-modelo determinista.** Diez pseudo-modelos abstractos que mapean a modelos físicos concretos con fallback automático, KeyVault, Blob Vault, compactación explícita y métricas.

> **Stack:** FastAPI · SQLite · Redis nativo (6380) · OpenCode Go · Groq · OpenRouter · DeepSeek
> **Despliegue:** `chat.guzman-lopez.com` vía GitHub Actions → servidor `plata`
> **Paquetería 100% libre:** MIT, BSD, Apache 2.0

---

## Estado en producción

| Componente | Estado |
|---|---|
| Proxy (FastAPI · :9110) | ✅ Activo |
| 10 pseudo-modelos | ✅ Todos funcionales |
| Redis nativo (:6380) | ✅ Conectado |
| SQLite | ✅ Conectado |
| Caddy HTTPS | ✅ chat.guzman-lopez.com |
| Deploy CI/CD | ✅ GitHub Actions |

## Los 10 Pseudo-Modelos

| # | Pseudo-modelo | Primary (Go) | Fallback(s) | Tools | Visión | Límite |
|---|---|---|---|---|---|---|---|
| 1 | `normal` | kimi-k2.5 (256K) | deepseek-v4-flash (1M) | strict | — | 256K |
| 2 | `pensamiento-profundo-caro` | qwen3.7-max (1M) | deepseek-v4-pro (1M) | strict | — | 120K |
| 3 | `tareas-avanzadas` | kimi-k2.6 (256K) | v4-pro → v4-flash | strict | — | 256K |
| 4 | `codigo-preciso` | mimo-v2.5-pro (1M) | mimo-v2-pro → v4-flash | strict | — | 1M |
| 5 | `vision` | llama-4-scout (Groq, 131K) | mimo-v2-omni (256K) | sí | ✅ | 120K |
| 6 | `pensamiento-rapido` | qwen3.6-plus (1M) | deepseek-v4-flash (1M) | sí | — | 120K |
| 7 | `normal-gratis` | nemotron (OpenRouter, 1M) | qwen3-32b (Groq, 131K) | no | — | 131K |
| 8 | `massive-fast` | minimax-m2.7 (192K) | gpt-oss-20b (Groq, 131K) | sí | — | 131K |
| 9 | `flash-lowcost` | qwen3.5-plus (1M) | deepseek-v4-flash (1M) | sí | — | 128K |
| 10 | `compactador` | glm-5.1 (198K) | gpt-oss-20b → v4-flash | sí | — | 3M |

**Model Aliases:** `gpt-4o`/`gpt-4o-mini` → `normal` · `o3` → `pensamiento-profundo-caro` · `o4-mini` → `pensamiento-rapido` · `gemini-2.5-flash` → `vision` · `gemini-2.5-pro`/`claude-sonnet-4` → `codigo-preciso` · `claude-haiku-3-5` → `flash-lowcost` · `default` → `normal`

## Quick Start

```bash
cd proxy
python3.14 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env    # Editar API keys
python -m src.main      # :9110

curl http://localhost:9110/health
curl -X POST http://localhost:9110/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"normal","messages":[{"role":"user","content":"Hola"}]}'
```

**Contra producción:**
```bash
curl -X POST https://chat.guzman-lopez.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"model":"normal","messages":[{"role":"user","content":"Hola"}]}'
```

## Endpoints

| Método | Path | Descripción |
|---|---|---|
| `GET` | `/health` | Health check (sin auth) |
| `GET` | `/v1/models` | Lista pseudo-modelos |
| `POST` | `/v1/chat/completions` | Chat (streaming + no-streaming) |
| `GET` | `/conversations/{id}` | Estado de conversación |
| `GET` | `/conversations/{id}/compatible-models` | Compatibilidad de switch |
| `POST` | `/conversations/{id}/compact` | Compactar historial |
| `GET` | `/conversations/{id}/audit-log` | Log de eventos |
| `GET` | `/metrics` | Métricas agregadas |

## Variables de Entorno

| Variable | Default | Descripción |
|---|---|---|
| `PROXY_PORT` | `9110` | Puerto |
| `PROXY_API_KEY` | — | Bearer token (vacío = dev mode) |
| `DATABASE_URL` | `sqlite+aiosqlite:///./proxy.db` | SQLite |
| `VALKEY_URL` | `valkey://localhost:6380` | Redis nativo |
| `OPENCODE_API_KEY` | — | OpenCode Go (primary) |
| `DEEPSEEK_API_KEY` | — | Fallbacks |
| `GROQ_API_KEY` | — | Groq |
| `OPENROUTER_API_KEY` | — | OpenRouter |

## Arquitectura Clave

**Dos endpoints de OpenCode Go:**
- `openai/` → `https://opencode.ai/zen/go/v1` (OpenAI-compat, 9 modelos)
  - Usa `reasoning_effort` (low/medium/high) para control de esfuerzo
- `anthropic/` → `https://opencode.ai/zen/go` (Anthropic-compat, solo `qwen3.7-max`)
  - Usa `thinking` dict con `budget_tokens` para control de esfuerzo

**Esfuerzo de razonamiento multi-proveedor:** El proxy acepta el parámetro `thinking` del cliente y lo traduce al formato que cada proveedor entiende (`budget_tokens` para Anthropic, `reasoning_effort` para OpenAI, auto para otros). Ver `proxy/README.md`.

**Capas de procesamiento:** CORS → Auth → RateLimit → KeyVault → Blob Vault → Fallback loop → Response

**KeyVault:** Detecta 27 patrones de secrets (API keys, PEM, SSH, JWT, wallets) → los reemplaza por `[KEYVAULT:hash]` antes de llegar al LLM → los reinyecta en la respuesta.

> ⚠️ **Streaming:** Los secrets se detectan y almacenan en Valkey, y el system prompt se inyecta para que el LLM entienda los placeholders. Pero la **re-inyección en la respuesta no ocurre** — el cliente ve `[KEYVAULT:hash]` sin reemplazar. Esto es porque `BaseHTTPMiddleware` de Starlette no permite modificar el body de streaming responses. La re-inyección completa requiere bufferizar chunks SSE entre request y response, lo cual no está implementado. Para secrets reales, usa `stream: false` o implementa re-inyección del lado del cliente.

**Blob Vault:** Contenido no soportado por el modelo (imágenes en modelo sin visión) → describe con modelo helper → pasa descripción al LLM.

**Compactación:** No automática. Si el input supera el umbral → error `400 context_length_exceeded` (formato OpenAI) con sugerencia de compactar.

## Arquitectura: Sin Rehidratación de Contexto

El proxy **nunca reconstruye** el historial de conversación desde la base de datos. El cliente (opencode, Continue, etc.) envía el contexto completo en cada request. Esto es una decisión de diseño deliberada:

- ❌ **Rehidratación desde DB** causaba duplicación de mensajes: 6 mensajes del cliente se convertían en 59 tras cargar el historial = 128K tokens inflados → `ALL_MODELS_FAILED`
- ✅ **El cliente manda el contexto completo** — los clientes modernos ya gestionan su propio historial
- ✅ **DB solo para auditoría**, compactación, blobs y afinidad

### Cómo se decide qué se envía al LLM

1. Mensajes del cliente (tal cual llegan, con contexto completo)
2. Si el modelo físico no soporta el contenido (imagen sin visión, etc.) → content delegation reemplaza blobs por descripciones textuales
3. `estimate_tokens` se ejecuta **después** de la delegación (paso 2), midiendo lo que realmente se manda al LLM
4. `context_alert` compara `historial_en_DB + petición_actual` contra `context_window` del pseudo-modelo
5. Si excede → error `400 context_length_exceeded` (formato OpenAI)

### Content Delegation

Cuando el modelo seleccionado no puede procesar un tipo de contenido directamente:

| Tipo | Acción | Modelo helper |
|------|--------|---------------|
| Imagen | → descripción textual | Groq visión |
| Audio | → transcripción | Whisper |
| PDF | → extracción de texto | interno |
| DOCX | → extracción de texto | interno |

Cada descripción se **trunca** al tamaño del archivo original (`sz KB → sz × 1024 chars` como máximo), para evitar que la delegación infle el contexto más que el binario original. Mínimo 500 chars de descripción útil.

## Deploy

GitHub Actions en cada push a `main`:
1. `git clone --branch main --depth 1` en servidor `plata`
2. Backup SQLite DB
3. `chown -R proxy:proxy`
4. Restore SQLite DB
5. `systemctl restart proxy-cesar`
6. Health check (fallo → rollback)

**Secrets:** `PLATA_HOST` · `PLATA_USER` · `PLATA_SSH_KEY` · `OPENCODE_API_KEY` · `GROQ_API_KEY` · `DEEPSEEK_API_KEY` · `OPENROUTER_API_KEY` · `PRUNA_API_KEY` · `PROXY_API_KEY`

## Errores del Sistema (formato OpenAI)

Todos los errores se devuelven en formato estándar OpenAI:

```json
{"error": {"message": "...", "type": "invalid_request_error", "param": null, "code": "context_length_exceeded"}}
```

| HTTP | `code` | Causa |
|---|---|---|
| 400 | `context_length_exceeded` | Input + historial > context_window |
| 400 | `unsupported_parameters` | Parallel tools no soportados |
| 429 | `rate_limit_exceeded` | Límite de tasa por pseudo-modelo |
| 503 | `server_error` | Todos los físicos fallaron |
| 502 | `server_error` | Error interno del proxy |

## Documentación Relacionada

- [`proxy/README.md`](proxy/README.md) — Documentación técnica del proxy
- [`CLAUDE.md`](CLAUDE.md) — Guía de arquitectura para AI agents
- [`diagramas.md`](diagramas.md) — Diagramas de arquitectura (Mermaid)
- [`python.md`](python.md) — Convenciones de código Python
- [`README-plata.md`](README-plata.md) — Reglas del servidor de producción
- [`proxy/pseudo_models.yaml`](proxy/pseudo_models.yaml) — Configuración fuente de los 10 modelos
