# proxy-cesar v1.1

**Proxy LLM multi-modelo determinista.** Diez pseudo-modelos abstractos que mapean a modelos físicos concretos con fallback automático, KeyVault, Blob Vault, compactación explícita, métricas y autenticación Bearer.

> **Stack:** FastAPI · SQLite · Redis nativo (6380) · OpenCode Go · Groq · OpenRouter · DeepSeek
> **Despliegue:** `chat.guzman-lopez.com` vía GitHub Actions → servidor `plata`
> **Paquetería 100% libre:** MIT, BSD, Apache 2.0
> **Tests:** 406 tests, 73% coverage
> **Autor:** Cesar Gerardo Guzman Lopez

---

## Estado en producción

| Componente | Estado |
|---|---|
| Versión | v1.0 (`818ed11`) |
| Proxy (FastAPI · :9110) | ✅ Activo |
| 10 pseudo-modelos | ✅ Todos funcionales |
| Redis nativo (:6380) | ✅ Conectado |
| SQLite | ✅ Conectado |
| Caddy HTTPS | ✅ chat.guzman-lopez.com |
| Auth | ✅ Bearer token requerido (excepto /health) |
| Deploy CI/CD | ✅ GitHub Actions → SSH directo |

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
```

**Con auth (recomendado — si PROXY_API_KEY está seteada):**
```bash
curl -X POST http://localhost:9110/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <tu-api-key>" \
  -d '{"model":"normal","messages":[{"role":"user","content":"Hola"}]}'
```

**Sin auth (solo en dev — PROXY_API_KEY vacío):**
```bash
curl -X POST http://localhost:9110/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"normal","messages":[{"role":"user","content":"Hola"}]}'
```

**Contra producción (`chat.guzman-lopez.com`):**
La API key es obligatoria. Sin key → `401 MISSING_AUTH`.
```bash
curl -X POST https://chat.guzman-lopez.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <tu-api-key>" \
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
| `PROXY_API_KEY` | — | Bearer token (vacío = dev mode; **requerido en producción**) |
| `DATABASE_URL` | `sqlite+aiosqlite:///./proxy.db` | SQLite |
| `VALKEY_URL` | `valkey://localhost:6380` | Redis nativo |
| `OPENCODE_API_KEY` | — | OpenCode Go (primary) |
| `DEEPSEEK_API_KEY` | — | DeepSeek (fallbacks) |
| `GROQ_API_KEY` | — | Groq (visión, whisper) |
| `OPENROUTER_API_KEY` | — | OpenRouter |
| `PRUNA_API_KEY` | — | Pruna |

## Arquitectura Clave

### OpenCode Go: Endpoints y model naming

OpenCode Go expone **dos endpoints incompatibles**. La asignación de cada modelo a su endpoint es fija y no se puede cambiar:

| Endpoint | Modelos | Formato de mensajes |
|----------|---------|---------------------|
| `https://opencode.ai/zen/go/v1/chat/completions` | `kimi-k2.5`, `kimi-k2.6`, `mimo-v2.5-pro`, `deepseek-v4-pro`, `deepseek-v4-flash`, `glm-5.1` | OpenAI |
| `https://opencode.ai/zen/go/v1/messages` | `qwen3.7-max`, `qwen3.6-plus`, `qwen3.5-plus`, `minimax-m2.7` | Anthropic |

**Regla:** Nunca mezclar. Qwen y MiniMax SOLO funcionan vía endpoint Anthropic. Usar el endpoint OpenAI para estos modelos resulta en `RateLimitError` o `400 BadRequest` del proveedor.

**Model naming en pseudo_models.yaml:**
```yaml
# Correcto:
- provider: opencode-go
  model: anthropic/qwen3.5-plus      # ← prefix anthropic + endpoint anthropic
  api_base: https://opencode.ai/zen/go

- provider: opencode-go
  model: openai/kimi-k2.5             # ← prefix openai + endpoint openai
  api_base: https://opencode.ai/zen/go/v1
```

El prefijo (`openai/`, `anthropic/`) lo usa liteLLM para elegir el adaptador. liteLLM lo quita automáticamente antes de enviar al API. No hay que cambiarlo ni tocarlo.

**Si en el futuro un modelo no funciona, lo primero que hay que verificar es que el prefix coincida con el endpoint. Una regresión como la de v1.1 (que revirtió estos valores) causa que modelos Qwen/MiniMax se caigan con RateLimitError.**

**Esfuerzo de razonamiento multi-proveedor:** El proxy acepta el parámetro `thinking` del cliente y lo traduce al formato que cada proveedor entiende (`budget_tokens` para Anthropic, `reasoning_effort` para OpenAI, auto para otros). Ver `proxy/README.md`.

**Capas de procesamiento:** CORS → Auth → RateLimit → KeyVault → Blob Vault → Fallback loop → Response

**KeyVault:** Detecta 27 patrones de secrets (API keys, PEM, SSH, JWT, wallets) → los reemplaza por `[KEYVAULT:hash]` antes de llegar al LLM → los reinyecta en la respuesta.

> ⚠️ **Streaming:** Los secrets se detectan, se almacenan en Valkey y el system prompt se inyecta. Pero la **re-inyección en la respuesta no está implementada** — el cliente ve `[KEYVAULT:hash]` en lugar del valor real.
>
> **Por qué:** El middleware llama a `request.body()` para inspeccionar los mensajes. `BaseHTTPMiddleware` de Starlette cachea ese body y, si luego la request se pasa al handler (streaming), el `StreamingResponse` devuelto ya está siendo consumido por el cliente SSE. Reemplazar placeholders requeriría bufferizar todos los chunks SSE, aplicar regex, y re-yield — con el riesgo de que un placeholder quede partido entre dos chunks. No se implementó porque:
> 1. El handler devuelve un generador asíncrono (SSE), no un body completo
> 2. El middleware no puede interceptar cada chunk individual sin romper el streaming
> 3. `request.state` no se propaga al generador porque el middleware retorna antes de setearlo (línea 459-460)
>
> **Solución:** Para secrets reales usa `stream: false` o implementa re-inyección del lado del cliente.

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
1. `git fetch origin main --depth 1` + `git reset --hard origin/main` en `plata`
2. Backup SQLite DB
3. `chown -R proxy:proxy`
4. Restaurar SQLite DB
5. Escribir `.env` con secrets vía heredoc (`${{ secrets.X }}`)
6. `systemctl restart proxy-cesar`
7. Health check (fallo → rollback)

**Secrets requeridos en GitHub:**
`PLATA_HOST` · `PLATA_USER` · `PLATA_SSH_KEY` · `PLATA_SSH_PORT` ·
`PROXY_API_KEY` · `OPENCODE_API_KEY` · `DEEPSEEK_API_KEY` ·
`GROQ_API_KEY` · `OPENROUTER_API_KEY` · `PRUNA_API_KEY` · `VALKEY_URL`

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
