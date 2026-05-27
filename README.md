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
|---|---|---|---|---|---|---|
| 1 | `normal` | kimi-k2.5 | deepseek-v4-flash | strict | — | 500K |
| 2 | `pensamiento-profundo-caro` | qwen3.7-max | deepseek-v4-pro | strict | — | 120K |
| 3 | `tareas-avanzadas` | kimi-k2.6 | v4-pro → v4-flash | strict | — | 200K |
| 4 | `codigo-preciso` | mimo-v2-pro | mimo-v2.5-pro → v4-flash | strict | — | 200K |
| 5 | `vision` | mimo-v2-omni | llama-4-scout (Groq) | sí | ✅ | 120K |
| 6 | `pensamiento-rapido` | qwen3.6-plus | deepseek-v4-flash | sí | — | 120K |
| 7 | `normal-gratis` | nemotron (OpenRouter) | qwen3-32b (Groq) | no | — | 200K |
| 8 | `massive-fast` | minimax-m2.7 | gpt-oss-20b (Groq) | sí | — | 131K |
| 9 | `flash-lowcost` | qwen3.5-plus | deepseek-v4-flash | sí | — | 128K |
| 10 | `compactador` | glm-5.1 | gpt-oss-20b → v4-flash | sí | — | 3M |

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
| `KEYCLAW_ENABLED` | `false` | Deshabilitado |
| `OPENCODE_API_KEY` | — | OpenCode Go (primary) |
| `DEEPSEEK_API_KEY` | — | Fallbacks |
| `GROQ_API_KEY` | — | Groq |
| `OPENROUTER_API_KEY` | — | OpenRouter |

## Arquitectura Clave

**Dos endpoints de OpenCode Go:**
- `openai/` → `https://opencode.ai/zen/go/v1` (OpenAI-compat, 9 modelos)
- `anthropic/` → `https://opencode.ai/zen/go` (Anthropic-compat, solo `qwen3.7-max`)

**Capas de procesamiento:** CORS → Auth → RateLimit → KeyVault → Blob Vault → Fallback loop → Response

**KeyVault:** Detecta 22 patrones de secrets (API keys, PEM, SSH, JWT, wallets) → los reemplaza por `[KEYVAULT:hash]` antes de llegar al LLM → los reinyecta en la respuesta.

**Blob Vault:** Contenido no soportado por el modelo (imágenes en modelo sin visión) → describe con modelo helper → pasa descripción al LLM.

**Compactación:** No automática. Si el input supera el umbral → `400 INPUT_EXCEEDS_THRESHOLD` con sugerencia de usar `POST /conversations/{id}/compact`.

## Deploy

GitHub Actions en cada push a `main`:
1. `git clone --branch main --depth 1` en servidor `plata`
2. Backup SQLite DB
3. `chown -R proxy:proxy`
4. Restore SQLite DB
5. `systemctl restart proxy-cesar`
6. Health check (fallo → rollback)

**Secrets:** `PLATA_HOST` · `PLATA_USER` · `PLATA_SSH_KEY` · `OPENCODE_API_KEY` · `GROQ_API_KEY` · `DEEPSEEK_API_KEY` · `OPENROUTER_API_KEY` · `PRUNA_API_KEY` · `PROXY_API_KEY`

## Errores del Sistema

| Código | Error | Causa |
|---|---|---|
| 400 | `INPUT_EXCEEDS_THRESHOLD` | Input > límite del pseudo-modelo |
| 400 | `IMAGES_NOT_SUPPORTED_BY_PSEUDO_MODEL` | Imagen en modelo sin visión |
| 400 | `PARALLEL_TOOLS_NOT_SUPPORTED` | Sin parallel tools |
| 409 | `PSEUDO_MODEL_INCOMPATIBLE` | Switch bloqueado por capacidades |
| 413 | `CONTEXT_TOO_LARGE_FOR_ALL_MODELS` | Todos los físicos excedidos |
| 429 | `RATE_LIMIT_EXCEEDED` | Límite de tasa |
| 502 | `ALL_MODELS_FAILED` | Todos los físicos fallaron |
| 502 | `PROXY_ERROR` | Error interno |

## Documentación Relacionada

- [`proxy/README.md`](proxy/README.md) — Documentación técnica del proxy
- [`diagramas.md`](diagramas.md) — Diagramas de arquitectura (Mermaid)
- [`BUG_VERIFICATION_FLOW.md`](BUG_VERIFICATION_FLOW.md) — Historias de usuario + comandos de verificación
- [`README-plata.md`](README-plata.md) — Reglas del servidor de producción
- [`proxy/pseudo_models.yaml`](proxy/pseudo_models.yaml) — Configuración fuente de los 10 modelos
