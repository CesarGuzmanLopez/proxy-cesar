# Diagramas de Arquitectura — proxy-cesar v1.1

> **Advertencia:** Reflejan el estado actual (Mayo 2026). Si el código cambia, actualiza este documento.

---

## 1. Flujo de Solicitud Chat

```mermaid
sequenceDiagram
    participant C as Cliente
    participant P as Proxy (FastAPI)
    participant PM as PseudoModel Registry
    participant PH as Physical Model
    participant BV as Blob Vault
    participant LLM as LLM Provider (Go/DS/Groq/OR)

    C->>P: POST /v1/chat/completions {model, messages}
    P->>PM: resolve(pseudo_model)
    PM-->>P: physical_model + fallbacks
    P->>BV: process_content(messages, capabilities)
    Note over BV: Auto-describe images/audio/PDF → [BLOB:hash]
    BV-->>P: messages processed (blobs replaced)
    P->>PH: build_conversation_messages(DB history + new)
    Note over PH: Stable prefix for provider caching
    P->>PH: route(physical_model, processed_messages)
    Note over PH: Sequential fallback si error
    PH->>LLM: forward request (with cache_control markers)
    LLM-->>PH: response (cache_hit/miss tokens)
    PH-->>P: response
    P->>P: store in conversation history
    P-->>C: response (streaming o no)
```

---

## 2. Arquitectura de Despliegue

```mermaid
graph TB
    subgraph GitHub
        A[push a main]
        B[GitHub Actions deploy.yml]
    end

    subgraph Servidor "plata"
        C[git clone --depth 1]
        D[chown -R proxy:proxy]
        E[Backup/Restore SQLite DB]
        F[systemctl restart proxy-cesar]
        G[Health Check POST-Deploy]
    end

    subgraph Servicios
        H[Caddy reverse proxy :443]
        I[proxy-cesar.service<br/>FastAPI :9110]
        J[Redis nativo :6380<br/>systemd redis-6380]
        K[deepbde-redis Docker :6379]
        L[chemistry-apps Docker :8080, :4210]
        M[PostgreSQL 14 :5432<br/>NO usado por proxy]
    end

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G -.->|OK| H
    H -->|chat.guzman-lopez.com| I
    I --> J
    I -.->|No depende| K
    I -.->|No depende| L
    I -.->|No depende| M
```

---

## 3. Pseudo-Modelos → Physical Models (v1.1)

```mermaid
graph LR
    subgraph "Pseudo Modelos (7)"
        PROF[pensamiento-profundo-caro]
        TAREA[tareas-avanzadas]
        NORMAL[normal]
        VIS[vision]
        GRAT[normal-gratis]
        FLASH[flash]
        COMP[compactador]
    end

    subgraph "Proveedores"
        OG[OpenCode Go]
        DS[DeepSeek]
        GQ[Groq]
        OR[OpenRouter]
    end

    PROF -->|Qwen3.7 Max| OG
    TAREA -->|MiniMax M2.7| OG
    NORMAL -->|MiMo-V2.5| OG
    VIS -->|Llama 4 Scout| GQ
    GRAT -->|Nemotron free| OR
    FLASH -->|GPT-OSS 20B - temp=0.1| GQ
    COMP -->|GLM-5.1| OG

    PROF -.->|fallback| DS
    TAREA -.->|fallback| DS
    NORMAL -.->|fallback| DS
    VIS -.->|fallback MiMo V2 Omni| OG
    GRAT -.->|fallback Qwen3-32b| GQ
    FLASH -.->|fallback DS V4 Flash| DS
    COMP -.->|fallback GPT-OSS + DS Flash| GQ
    COMP -.->|fallback| DS
```

### Physical models clave (v1.1):
| Provider | Modelo | Pseudo-modelos |
|---|---|---|
| **opencode-go** | `anthropic/qwen3.7-max` | pensamiento-profundo-caro |
| | `anthropic/minimax-m2.7` | tareas-avanzadas |
| | `openai/mimo-v2.5` | **normal** |
| | `openai/mimo-v2-omni` | vision (fallback) |
| | `openai/glm-5.1` | compactador |
| **deepseek** | `deepseek/deepseek-v4-pro` | pensamiento-profundo-caro (fallback) |
| | `deepseek/deepseek-v4-flash` | tareas, normal, flash, compactador |
| **groq** | `groq/openai/gpt-oss-20b` | **flash** (primary), compactador (fallback) |
| | `groq/meta-llama/llama-4-scout-17b-16e-instruct` | vision (primary) |
| | `groq/qwen/qwen3-32b` | normal-gratis (fallback) |
| **openrouter** | `nvidia/nemotron-3-super-120b-a12b:free` | normal-gratis (primary) |

---

## 4. Capas de Procesamiento

```mermaid
graph TD
    REQ[Solicitud entrante] --> KEYV[KeyVault sanitization]
    KEYV --> BLOB[Blob Vault]
    BLOB -->|Imagen sin vision| VISION[Describir via modelo visión]
    BLOB -->|PDF| PDFEX[Extraer texto]
    BLOB -->|Audio| WHIS[Transcribir via Whisper]
    VISION --> HIST[build_conversation_messages<br/>DB history + new messages]
    PDFEX --> HIST
    WHIS --> HIST
    BLOB -->|Contenido OK| HIST
    HIST --> CACHE[apply cache_control markers<br/>si Anthropic/Go]
    CACHE --> ROUTE[Seleccionar physical model]
    ROUTE --> SystemPrompt[Inyectar system_prompt<br/>si Groq / GPT-OSS]
    SystemPrompt --> EXEC[Ejecutar contra LLM<br/>temperature/top_p forzados]
    EXEC -->|Éxito| RES[Responder]
    EXEC -->|Fallo| FALLBACK[Probar siguiente fallback]
    FALLBACK --> ROUTE
    EXEC -->|Todos fallaron| ERR[Error 502 ALL_MODELS_FAILED]
```

---

## 5. Diagrama de Paquetes (Hexagonal)

```mermaid
graph TB
    subgraph "Dominio (core)"
        DM[Domain Models<br/>PseudoModel, PhysicalModel,<br/>Conversation, Message]
        SRVC[Services<br/>chat_service.py → chat_fallback.py<br/>chat_persistence.py, chat_messages.py<br/>CompactService, PseudoModelRegistry]
        PORTS[Ports<br/>ICache, IDatabase, IAuditLog,<br/>ILLMProvider]
    end

    subgraph "Aplicación (API)"
        API[FastAPI Router<br/>chat.py → chat_streaming.py<br/>chat_stream_persistence.py<br/>conversations.py + conversation_operations.py<br/>/health /metrics]
        MIDD[Middleware<br/>KeyVault, BlobVault,<br/>AuditLog, Metrics]
    end

    subgraph "Adaptadores (Infra)"
        CACHE[Cache Adapter<br/>Valkey/Redis :6380]
        DB[Database Adapter<br/>SQLite (archivo)]
        LLM[LLM Provider Adapter<br/>OpenCode Go, OpenRouter, Groq<br/>vía HTTP]
        AUDIT[Audit Adapter<br/>Base de datos]
    end

    API --> MIDD
    MIDD --> SRVC
    SRVC --> DM
    SRVC --> PORTS
    PORTS --> CACHE
    PORTS --> DB
    PORTS --> LLM
    PORTS --> AUDIT
```

---

## 6. Flujo de Despliegue Continuo (CI/CD)

```mermaid
sequenceDiagram
    participant Dev as Desarrollador
    participant GH as GitHub
    participant GA as GitHub Actions
    participant Server as Servidor (plata)

    Dev->>GH: git push origin main
    GH->>GA: trigger deploy.yml
    GA->>Server: ssh root@plata
    Server->>Server: git clone --branch main --depth 1 /tmp/proxy-cesar
    Server->>Server: chown -R proxy:proxy /tmp/proxy-cesar
    Server->>Server: systemctl restart proxy-cesar
    Server->>Server: Health check (curl localhost:9110/health)
    Server-->>GA: Deploy result (OK/FAIL)
    GA-->>Dev: Notificación
```

> **Nota:** El deploy corre como `root`, el servicio corre como `proxy`. El `chown` post-clone es crítico. La DB se preserva entre deploys.

---

## 7. Puertos en el Servidor

| Puerto | Servicio | Propietario | Proxy-related |
|--------|----------|-------------|---------------|
| 443 | HTTPS (Caddy) | Caddy | Sí (→ :9110) |
| 9110 | proxy-cesar FastAPI | proxy | **Sí** |
| 6380 | Redis nativo | proxy | **Sí** |
| 6379 | Redis Docker (deepbde) | root | No |
| 5432 | PostgreSQL 14 | postgres | No |
| 8080 | chemistry-apps (Docker) | root | No |
| 4210 | chemistry-apps API (Docker) | root | No |
| 8000 | deepbde-backend (Docker) | root | No |
| 22 | SSH | root | No |

---

## 8. Flujo de Razonamiento (Thinking / Reasoning Effort)

```mermaid
flowchart LR
    C[Cliente<br/>thinking: 'high'] --> P[Proxy]
    P --> D{Capacidad del<br/>physical model}
    D -->|Anthropic<br/>o Go Anthropic-route| A[thinking dict<br/>budget_tokens: 16000]
    D -->|OpenAI<br/>o Go OpenAI-route| O[reasoning_effort: 'high']
    D -->|Otros| X[auto<br/>no se envía nada]
    A --> LLM[LiteLLM acompletion]
    O --> LLM
    X --> LLM
```

La normalización ocurre en `_normalise_reasoning_param()` dentro de `chat_fallback.py`:
- Cada modelo físico en la cadena de fallback se evalúa individualmente
- Go OpenAI-route models (MiMo, Kimi, GLM, DeepSeek) reciben `reasoning_effort`
- Go Anthropic-route models (Qwen, MiniMax) reciben `thinking` dict

---

## 9. Context Objects (v1.1)

| Clase | Ubicación | Parámetros | Caso de uso |
|---|---|---|---|
| `StreamingRequestContext` | `chat_models.py` | 15→1 | Setup de streaming |
| `SaveContext` | `chat_models.py` | 23→1 | Persistir turno + resultado |
| `MetadataContext` | `chat_models.py` | 22→1 | Construir `proxy_metadata` |

---

## 10. Blob Description Cache

Las descripciones de imágenes/audio/PDF generadas por el Blob Vault se
almacenan en Redis (Valkey) con clave compuesta:

```
{prefix}:{content_hash}:desc:{prompt_hash}
```

- `content_hash` — hash SHA-256 de 8 caracteres del contenido binario
- `prompt_hash` — hash SHA-256 de 8 caracteres del texto del mensaje del usuario

Esto permite que una misma imagen reciba descripciones distintas según el
contexto del prompt. Ver `src/service/tool_detector.py:396`.

---

## 11. Multi-Turn Prompt Caching (v1.1)

```mermaid
flowchart TB
    subgraph "Turno 1"
        T1[Cliente envía mensaje] --> HIST1[build_conversation_messages<br/>No hay historial DB]
        HIST1 --> LLM1[LLM: mensaje solo]
        LLM1 --> DB1[Guardar en DB]
    end

    subgraph "Turno 2"
        T2[Cliente envía nuevo mensaje] --> HIST2[build_conversation_messages<br/>DB history + nuevo mensaje]
        HIST2 --> CACHE2[⚡ cache_control markers<br/>en contenido]
        CACHE2 --> LLM2[LLM: prefix = DB history<br/>solo computa nuevo mensaje]
        LLM2 --> DB2[Guardar en DB]
    end

    subgraph "Turno 3"
        T3[...siguiente mensaje] --> HIST3[build_conversation_messages<br/>más historial DB]
        HIST3 --> CACHE3[⚡ cache_control<br/>prefijo aún más grande]
        CACHE3 --> LLM3[LLM: cache_hit=604<br/>solo computa último mensaje]
        LLM3 --> DB3[Guardar en DB]
    end
```

**Providers y mecanismos:**

| Provider | Mecanismo | Proxy envía |
|---|---|---|
| Go (Anthropic-route) | cache_control markers | ✅ |
| Go (OpenAI-route) | cache_control markers + nativo | ✅ |
| DeepSeek | Disk caching automático | Prefijo estable |
| Groq | Prefix caching >1024 tokens | Prefijo estable |
