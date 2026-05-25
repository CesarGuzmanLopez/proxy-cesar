# Plan: Proxy Determinista Multi-Modelo

> **Paquetería 100% libre.** MIT, BSD, Apache 2.0.
> **El proxy no decide. Valida, informa y ejecuta lo que el usuario ordena.**
> **Cuando algo no puede continuar, lo dice claramente y ofrece opciones.**
> **Cuando algo puede continuar, lo hace sin fricción.**
> **Código, variables, comentarios, commits y documentación técnica en inglés.**
> **100% tipado y determinista.** La única no-determinista es la respuesta del modelo.
> **Sin prefijos, sin inyección de variables, sin manipulación de strings.**
> Los nombres de modelos se usan exactamente como están en el archivo de configuración.
> Si algo no cumple, error claro — nunca silencio.

---

## Índice

1. [Visión del sistema](#1-visión-del-sistema)
2. [Stack 100% libre](#2-stack-100-libre)
3. [Arquitectura general](#3-arquitectura-general)
4. [Pseudo-modelos definitivos](#4-pseudo-modelos-definitivos)
5. [Mecanismo de afinidad de caché](#5-mecanismo-de-afinidad-de-caché)
6. [Herramientas y Function Calling](#6-herramientas-y-function-calling)
7. [Multimedia: imágenes, audio, PDF y video](#7-multimedia-imágenes-audio-pdf-y-video)
8. [Compatibilidad entre pseudo-modelos](#8-compatibilidad-entre-pseudo-modelos)
9. [Pre-compactación de entrada](#9-pre-compactación-de-entrada)
10. [Compactación continua para modelos caros](#10-compactación-continua-para-modelos-caros)
11. [Compactación explícita de conversaciones](#11-compactación-explícita-de-conversaciones)
12. [Router LLM opcional](#12-router-llm-opcional)
13. [Capa de optimización de caché de proveedor](#13-capa-de-optimización-de-caché-de-proveedor)
14. [API del proxy](#14-api-del-proxy)
15. [Integración con OpenCode](#15-integración-con-opencode)
16. [Despliegue remoto](#16-despliegue-remoto)
17. [Estructura del proyecto](#17-estructura-del-proyecto)
18. [Plan de sprints](#18-plan-de-sprints)
19. [Historias de usuario](#19-historias-de-usuario)
20. [Árboles de decisión deterministas](#20-árboles-de-decisión-deterministas)

---

## 1. Visión del sistema

### 1.1 Qué es

Un servicio HTTP que se interpone entre cualquier cliente (OpenCode, scripts, IDEs, apps) y los proveedores de LLM. Expone una API compatible con OpenAI `/v1/chat/completions`.

El usuario selecciona un **pseudo-modelo** según la intención de la tarea. El proxy:

- **Resuelve** el pseudo-modelo al modelo físico real del proveedor correspondiente
- **Mantiene afinidad** de modelo físico por conversación para maximizar cache hits
- **Valida compatibilidad** si el usuario cambia de pseudo-modelo
- **Filtra** modelos sin soporte OpenAI-compatible de tools cuando la conversación usa tools
- **Pre-compacta** entradas largas para modelos caros (configurado por pseudo-modelo)
- **Describe imágenes** al migrar de un modelo con visión a uno sin ella (configurado por pseudo-modelo)
- **Compacta continuamente** el contexto en modelos caros para no saturarlos
- **Sugiere downgrade** vía router LLM si la tarea es simple — nunca impone
- **Informa** en cada respuesta: modelo físico, uso de contexto, advertencias, ahorros

### 1.2 Qué NO hace

- No decide qué modelo usar sin que el usuario lo pida
- No compacta sin consentimiento (excepto compactación continua, que es configuración explícita del pseudo-modelo)
- No cambia de modelo automáticamente entre pseudo-modelos (solo fallback dentro del mismo)
- No descarta el historial original
- No manipula los nombres de modelos: se usan exactamente como están en `pseudo_models.yaml`
- No resume en silencio: cualquier operación sobre el contexto se notifica en `proxy_metadata`

### 1.3 Flujo de una petición

```
POST /v1/chat/completions
  { model: "normal", messages: [...], conversation_id: "abc" }
  │
  ├─ 1. ¿Conversación nueva?
  │     Sí → seleccionar modelo físico (prioridad 1 del pseudo-modelo), guardar afinidad
  │     No  → recuperar modelo físico fijado de Valkey/PostgreSQL
  │
  ├─ 2. ¿Cambió el pseudo-modelo respecto al turno anterior?
  │     Sí → validar compatibilidad con capabilities acumuladas
  │           ├─ safe      → continuar
  │           ├─ warning   → continuar con advertencia en proxy_metadata
  │           └─ blocked   → HTTP 409 con opciones de remediación
  │     No  → continuar
  │
  ├─ 3. Detectar capabilities del input actual (imágenes, tools, parallel tools)
  │     Acumular flags en DB (aditivos, nunca se desactivan)
  │
  ├─ 4. Si hay parallel tools: filtrar pool a modelos con parallel_tools: true
  │     Si el modelo fijado no califica → intentar fallback dentro del mismo pseudo-modelo
  │
  ├─ 5. ¿Input supera umbral del pseudo-modelo?
  │     Sí + pre_compaction habilitado → modelo barato resume input → modelo caro recibe resumen
  │     Sí + pre_compaction deshabilitado → error INPUT_EXCEEDS_THRESHOLD
  │     No  → continuar
  │
  ├─ 6. ¿Contexto acumulado > umbral de compactación continua?
  │     Sí + continuous_compaction habilitado → compactar turnos antiguos en snapshot
  │     No  → continuar
  │
  ├─ 7. ¿Router LLM habilitado?
  │     Sí → evaluar complejidad con modelo barato → proxy_metadata.router_suggestion
  │     (nunca cambia el modelo, solo informa)
  │
  ├─ 8. Enviar a LiteLLM con modelo físico fijado
  │     Error 503/429 → fallback al siguiente modelo del mismo pseudo-modelo
  │     Sin fallback → error ALL_MODELS_FAILED
  │
  └─ 9. Respuesta + proxy_metadata + actualizar DB con turno y flags
```

---

## 2. Stack 100% libre

| Componente           | Paquete                  | Licencia                | Propósito                                             |
| -------------------- | ------------------------ | ----------------------- | ----------------------------------------------------- |
| API Gateway          | FastAPI                  | MIT                     | Endpoints REST, async nativo, OpenAPI auto            |
| Router LLM           | LiteLLM                  | MIT                     | Abstracción multi-proveedor, fallbacks, cost tracking |
| Persistencia         | PostgreSQL + asyncpg     | PostgreSQL / Apache 2.0 | Historial, snapshots, flags de capability             |
| Caché/Afinidad       | Valkey                   | BSD                     | Afinidad de modelo físico, rate limiting              |
| Tareas asíncronas    | Celery + Valkey broker   | BSD                     | Compactación, degradación, normalización              |
| Validación           | Pydantic v2              | MIT                     | Schemas de pseudo-modelos, tools, capabilities        |
| ORM                  | SQLAlchemy 2.0 + Alembic | MIT                     | Migraciones, queries async                            |
| HTTP Client          | httpx                    | BSD                     | Llamadas a LiteLLM y proveedores                      |
| Token counting       | tiktoken                 | MIT                     | Conteo de tokens antes de enviar                      |
| Config               | PyYAML                   | MIT                     | Definición de pseudo-modelos                          |
| Procesamiento imagen | Pillow                   | HPND (BSD-like)         | Pre-procesamiento de imágenes                         |
| CORS / TLS           | Caddy (reverse proxy)    | Apache 2.0              | HTTPS en producción                                   |

**Nada de dependencias propietarias. Todo auditado y de código abierto.**

---

## 3. Arquitectura general

```mermaid
graph TD
    A[Clientes: OpenCode / Scripts / IDEs] -->|HTTP OpenAI-compatible| B[FastAPI — API Gateway]

    B --> C[Middleware: Capability Detector]
    C --> D[Middleware: Compatibility Validator]
    D --> E[Middleware: Threshold Guard]
    E --> F[Tool Filter: solo OpenAI-compatible]

    F --> G{pre_compaction?}
    G -->|Sí, input largo| H[Pre-Compactor — modelo barato]
    H --> I
    G -->|No| I{continuous_compaction?}
    I -->|Sí, contexto alto| J[Continuous Compactor — snapshot]
    J --> K
    I -->|No| K{router_llm?}
    K -->|Sí| L[Router LLM — evalúa complejidad]
    L --> M
    K -->|No| M[LiteLLM Router]

    M --> N[Proveedor A — Zhipu / GLM]
    M --> O[Proveedor B — DeepSeek]
    M --> P[Proveedor C — Anthropic]
    M --> Q[Proveedor D — Google / Gemini]
    M --> R[Proveedor E — Groq]
    M --> S[Modelos Locales — Ollama]

    B --> T[PostgreSQL]
    B --> U[Valkey]
    B --> V[Celery]

    T --> T1[Historial completo canónico]
    T --> T2[Snapshots de compactación]
    T --> T3[Capability flags]
    U --> U1[Afinidad de modelo físico]
    U --> U2[Rate limiting]
    V --> V1[Compactación async]
    V --> V2[Degradación multimedia]
    V --> V3[Normalización de tools]
```

---

## 4. Pseudo-modelos definitivos

### 4.0 Reglas de configuración

- El proxy lee `pseudo_models.yaml` al iniciar. Si hay errores de validación → **el proxy no inicia**.
- El campo `model` en `physical_models` contiene el identificador **exacto** que LiteLLM espera. Sin prefijos, sin transformación, sin concatenación con `provider`.
- El campo `provider` es **solo metadata**. No se usa para construir el ID.
- Toda la configuración se valida con Pydantic estricto (`extra = "forbid"`, tipos exactos).
- Si un modelo físico no es reconocido por LiteLLM, el error ocurre en la primera petición que lo use. El proxy reporta el error claramente — nunca silencio.
- El campo `openai_tools_compatible` debe ser `true` para todos los modelos en el config. Si es `false`, la validación de configuración falla al iniciar. Todos los modelos en el pool soportan tools.

### 4.1 `pensamiento-profundo-caro`

```yaml
pensamiento-profundo-caro:
  display_name: "Pensamiento Profundo"
  description: >
    Razonamiento de máximo nivel. Bugs imposibles, arquitectura compleja, decisiones
    con muchas variables. El usuario selecciona este modo activamente y con precaución.
    Usar para tareas puntuales difíciles de implementar, no como modelo de cabecera.
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
      note: "Fallback con tools nivel 3 y strict. Más barato que GPT/Anthropic."
  fallback_strategy: sequential
```

**Notas de diseño:**

- Prioridad: GLM 5.1 → DeepSeek V4 Pro (fallback con tools strict y parallel)
- Router LLM activo: si la tarea es simple, `proxy_metadata` sugiere bajar a `tareas-avanzadas`. El usuario decide.
- Pre-compaction activo: inputs >32K tokens se resumen con `deep-flash` antes de enviar al modelo caro
- Continuous compaction al 70%: a partir del turno donde el contexto supere 140K tokens se compacta automáticamente

### 4.2 `tareas-avanzadas`

```yaml
tareas-avanzadas:
  display_name: "Tareas Avanzadas"
  description: >
    El caballo de batalla para trabajo real extenso. Features largas, debugging
    serio donde el programador sabe que el problema es difícil, definición de
    arquitectura, refactorizaciones grandes. Largo plazo. Muchos tokens.
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
      note: "Modo thinking. Razonamiento explícito."
    - provider: minimax
      model: minimax-m2.5
      openai_tools_compatible: true # No confiable con tools
      tools_strict: false
      parallel_tools: false
      vision: false
      note: "Más barato que GPT/Anthropic."
  fallback_strategy: sequential
```

**Notas de diseño:**

- MiniMax M2.5 está en la lista pero tiene `openai_tools_compatible: true`. Cuando la conversación usa tools, el proxy lo excluye automáticamente del pool. Solo entra como fallback en conversaciones sin tools.
- DeepSeek V4 Pro soporta `strict: true` → schemas de tools garantizados.

### 4.3 `avanzada-vision`

```yaml
avanzada-vision:
  display_name: "Visión Avanzada"
  description: >
    Análisis visual de alto nivel. Diseño de interfaces, OCR complejo,
    interpretación de diagramas, maquetas, wireframes. Especialmente
    útil para trabajo de frontend a partir de imágenes de referencia.
  input_token_threshold: 32000
  context_window: 32768
  continuous_compaction:
    enabled: false # las imágenes no se compactan semánticamente
  pre_compaction:
    enabled: false
  router_llm:
    enabled: false
  image_handling:
    on_downgrade: "auto_describe"
  physical_models:
    - provider: google
      model: gemini-3.5-flash
      openai_tools_compatible: true # LiteLLM traduce
      tools_strict: false
      parallel_tools: false
      vision: true
    - provider: groq
      model: meta-llama/llama-4-scout-17b-16e-instruct
      openai_tools_compatible: true
      tools_strict: false
      parallel_tools: false
      vision: true
      note: "Alternativa rápida y barata para visión"
  fallback_strategy: sequential
```

### 4.4 `normal`

```yaml
normal:
  display_name: "Normal"
  description: >
    Punto de entrada recomendado. Coding extenso, trabajo agéntico, manipulación
    de documentos, investigaciones largas con múltiples fuentes, desarrollo de
    features de tamaño medio. Balance costo/capacidad óptimo.
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
```

### 4.5 `deep-flash`

```yaml
deep-flash:
  display_name: "Deep Flash"
  description: >
    Velocidad y costo mínimo para tareas masivas y simples. Investigaciones
    larguísimas, traducciones masivas, tareas monótonas de lectura y generación.
    Soporta hasta 10-15 sub-agentes concurrentes. Muy rápidos pero no para
    razonamiento complejo.
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
      openai_tools_compatible: true # básico pero funcional para tools simples
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
      note: "Preferir si la conversación usa tools"
  fallback_strategy: sequential
```

**Notas de diseño:**

- Todos los modelos tienen `openai_tools_compatible: true` pero con niveles de confiabilidad distintos.
- Para tools en este pseudo-modelo: el proxy informará en `proxy_metadata` que el nivel de soporte es básico (GLM y Groq) o completo (DeepSeek flash). El usuario decide si eso es suficiente para su caso.

### 4.6 `flash-lowcost`

```yaml
flash-lowcost:
  display_name: "Flash Lowcost"
  description: >
    Sub-agentes baratos. Tareas específicas y bien definidas: clasificación,
    extracción, parsing, validación de formato. Consistentes en tareas
    bien especificadas. No son los más creativos pero ejecutan con precisión.
    También funcionan como evaluadores del router LLM.
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
      openai_tools_compatible: true # local, sin soporte confiable de tools
      tools_strict: false
      parallel_tools: false
      vision: false
      note: "Fallback local."
  fallback_strategy: sequential
```

### 4.7 `flash-vision`

```yaml
flash-vision:
  display_name: "Flash Vision"
  description: >
    Visión rápida y barata. OCR ligero, screenshots, análisis visual simple.
    Compatible hacia arriba con avanzada-vision (upgrade seguro).
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
      openai_tools_compatible: true # LiteLLM traduce
      tools_strict: false
      parallel_tools: false
      vision: true
    - provider: ollama
      model: ollama/llava
      openai_tools_compatible: true
      tools_strict: false
      parallel_tools: false
      vision: true
      note: "Fallback local."
  fallback_strategy: sequential
```

### 4.8 `compactador` _(operación, no conversacional)_

```yaml
compactador:
  display_name: "Compactador"
  description: >
    Operación de compactación explícita invocada por el usuario o por el
    sistema cuando el historial supera todos los umbrales. Selecciona
    automáticamente el modelo con ventana de contexto suficiente.
    No es un modo conversacional.
  input_token_threshold: null
  context_window: null
  continuous_compaction:
    enabled: false
  pre_compaction:
    enabled: false
  router_llm:
    enabled: false
  image_handling:
    on_downgrade: "auto_describe" # Si hay imágenes en el historial, las describe
  physical_models:
    - provider: google
      model: gemini-3.5-flash
      context_window: 1000000
      openai_tools_compatible: true # no necesita tools para compactar
      vision: true # puede describir imágenes del historial
    - provider: anthropic
      model: claude-haiku-4-5-20251001 # el unico de Antrhopic que se usara en este sistema en modo normal y compactación, no usar otros modelos de Anthropic
      context_window: 200000
      openai_tools_compatible: true
      vision: false
    - provider: zhipu
      model: glm-4.5-flash
      context_window: 128000
      openai_tools_compatible: true
      vision: false
  fallback_strategy: by_context_window # selecciona el que cubre el historial dado
```

---

## 5. Mecanismo de afinidad de caché

### 5.1 El problema

Cuando una conversación usa `normal` → `qwen3-max`, el proveedor cachea el prefijo (system + tools + historial). Si en el turno 5 el proxy envía la conversación a `deepseek-v4-flash`, el caché de Qwen se destruye: DeepSeek comienza desde cero, el costo aumenta, la latencia aumenta.

### 5.2 La solución

El proxy **fija el modelo físico en el primer turno** de cada conversación y lo mantiene hasta cambio explícito del usuario.

```
Turno 1: conversación "abc", pseudo-modelo "normal"
  → proxy lee pseudo_models.yaml → primer modelo: "qwen3-max"
  → Valkey: SET conv:abc:physical_model "qwen3-max" EX 86400
  → PostgreSQL: conversation.physical_model = "qwen3-max"

Turnos 2-50: mismo pseudo-modelo
  → proxy: GET conv:abc:physical_model → "qwen3-max"
  → SIEMPRE usa "qwen3-max" (valor exacto del config, sin modificar)
  → El proveedor reutiliza el caché del prefijo

Turno 51: usuario cambia a "pensamiento-profundo-caro"
  → compatibilidad: safe
  → primer modelo del nuevo pseudo-modelo: "glm-5.1"
  → Valkey: SET conv:abc:physical_model "glm-5.1" EX 86400
  → Nuevo caché con GLM 5.1 en Zhipu

Fallback en turno 20: "qwen3-max" retorna 503
  → proxy intenta "deepseek-v4-flash" (prioridad 2)
  → Valkey: actualizar afinidad a "deepseek-v4-flash"
  → proxy_metadata: "fallback_applied: true, reason: upstream_503"
  → Notificar al usuario. El caché previo de Qwen se abandona.
```

### 5.3 Invariantes del mecanismo

- El `conversation_id` es enviado explícitamente por el cliente en cada request (body o header `X-Conversation-ID`). No hay sesiones de servidor.
- Si el cliente no envía `conversation_id`, el proxy genera uno y lo retorna en `proxy_metadata`. El cliente debe persistir ese ID para siguientes turnos.
- El proxy es **stateless a nivel HTTP**. Cualquier instancia puede atender cualquier request. El estado vive en PostgreSQL y Valkey, compartidos.
- La afinidad en Valkey tiene TTL de 24h (configurable). Pasado ese tiempo, el siguiente turno selecciona el modelo de prioridad 1 nuevamente.

---

## 6. Herramientas y Function Calling

> Esta sección es la más crítica del diseño. El function calling es la fuente de incompatibilidad más frecuente y menos obvia entre proveedores.

### 6.1 La regla — sin excepciones

> **Todo modelo que participe en una conversación con tools DEBE tener `openai_tools_compatible: true` en `pseudo_models.yaml`. Si no lo tiene, el proxy lo excluye del pool para esa conversación. No hay advertencia, no hay fallback parcial: se excluye.**

LiteLLM traduce el formato OpenAI al formato nativo del proveedor (Anthropic, Gemini, etc.) de forma transparente. El proxy no necesita conocer ni implementar ese mapeo — LiteLLM lo resuelve. El proxy solo enforces el contrato: **OpenAI format in, OpenAI format stored, LiteLLM translates per provider.**

**La regla resuelve de raíz todos los problemas de formato:** en lugar de gestionar N formatos con N comportamientos, el sistema impone un único contrato. Modelos que no lo cumplen salen del pool.

**Dentro de los modelos elegibles, se prefiere `tools_strict: true`** cuando el proveedor lo soporta (DeepSeek V4, OpenAI).

### 6.2 Por qué se eliminan ciertos modelos y no solo se advierte

Los comportamientos problemáticos que justifican la exclusión no son "menor calidad" — son fallas estructurales:

- **Ignorar tools en silencio:** el modelo recibe las definiciones y responde como si no existieran. Sin error, sin aviso. El cliente queda esperando tool calls que nunca llegan.
- **Alucinación de parámetros:** el modelo genera tool calls con campos inventados o tipos incorrectos. El cliente ejecuta código con datos malformados.
- **Streaming roto con tools:** el JSON de argumentos se fragmenta incorrectamente entre chunks. El cliente no puede deserializarlo.
- **Parallel calls corrompidas:** el modelo declara soportarlas pero mezcla los resultados o los ordena mal. El cliente empareja tool_call_ids incorrectamente.

Estos fallos no se mitigan con reintentos. Se evitan no usando esos modelos cuando hay tools.

### 6.3 Tabla de compatibilidad por modelo físico

| Modelo               | `openai_tools_compatible` | `tools_strict` | `parallel_tools` | Notas                                                                   |
| -------------------- | ------------------------- | -------------- | ---------------- | ----------------------------------------------------------------------- |
| GPT-5.4              | ✅ nativo                 | ✅             | ✅               | No usado en pseudo-modelos (costo extremo). Solo referencia de formato. |
| Gemini 3.5 Flash     | ✅ (vía LiteLLM)          | ✗              | Parcial          | LiteLLM traduce. Parallel limitado                                      |
| DeepSeek V4 Pro      | ✅ nativo                 | ✅             | ✅               | OpenAI-compatible, strict soportado                                     |
| DeepSeek V4 Flash    | ✅ nativo                 | ✗              | ✅               | Compatible, parallel soportado                                          |
| Qwen3 Max            | ✅ nativo                 | ✗              | ✗                | Compatible, parallel no confiable                                       |
| GLM-5.1              | ✅ nativo                 | ✗              | ✗                | Flagship Zhipu                                                          |
| GLM-4.5-Flash        | ✅ nativo                 | ✗              | ✗                | Básico. Solo tools simples y bien definidas                             |
| GPT-OSS 20B (Groq)   | ✅ nativo                 | ✗              | ✗                | Compatible, parallel no confiable                                       |
| MiniMax M2.5         | ✅ nativo                 | ✗              | ✗                | Básico. Schemas simples                                                 |
| Qwen3.5 Plus         | ✅ nativo                 | ✗              | ✗                | Básico. Úsese con schemas simples                                       |
| LLaVA / Ollama local | ❌                        | ✗              | ✗                | Sin soporte confiable                                                   |
| Llama3.2 local       | ❌                        | ✗              | ✗                | Sin soporte confiable                                                   |

### 6.4 Filtrado de modelos por tools

Todos los modelos en el config deben tener `openai_tools_compatible: true`. La validación de configuración rechaza cualquier modelo con `false` al iniciar. Si la conversación usa parallel tools, el pool se reduce a modelos con `parallel_tools: true`.

```python
def get_eligible_models(pseudo_model: PseudoModel, session_caps: Capabilities) -> list[PhysicalModel]:
    models = pseudo_model.physical_models
    # Todos los modelos tienen openai_tools_compatible: true (validado al cargar config)

    if session_caps.has_parallel_tools:
        parallel_eligible = [m for m in models if m.parallel_tools]
        if parallel_eligible:
            return parallel_eligible
        # Si ninguno soporta parallel, se usa la lista completa con advertencia.
        # El historial paralelo puede corromperse — el proxy lo advierte en proxy_metadata.

    return models
```

Si el modelo fijado por afinidad no soporta parallel tools y la conversación las requiere, el proxy:

1. Intenta el siguiente modelo de la lista con `parallel_tools: true`
2. Actualiza la afinidad al nuevo modelo
3. Informa en `proxy_metadata`: `{"tools_filter_applied": true, "reason": "parallel_tools_required"}`

### 6.5 Formato canónico de tools en base de datos

Todo el historial de tools se almacena en formato OpenAI canónico. LiteLLM lo traduce al formato del proveedor en cada request, y normaliza la respuesta de vuelta a OpenAI al recibirla.

**Tool definition (almacenada como fue enviada por el cliente):**

```json
{
  "type": "function",
  "function": {
    "name": "search_codebase",
    "description": "Busca en el código fuente",
    "parameters": {
      "type": "object",
      "properties": {
        "query": { "type": "string" },
        "path": { "type": "string" }
      },
      "required": ["query"]
    },
    "strict": true
  }
}
```

**Assistant turn con tool calls (formato canónico almacenado):**

```json
{
  "role": "assistant",
  "content": "Voy a buscar en el código fuente.",
  "tool_calls": [
    {
      "id": "call_abc123",
      "type": "function",
      "function": {
        "name": "search_codebase",
        "arguments": "{\"query\": \"db connection\", \"path\": \"src/\"}"
      }
    }
  ]
}
```

**Tool result (formato canónico almacenado):**

```json
{
  "role": "tool",
  "tool_call_id": "call_abc123",
  "name": "search_codebase",
  "content": "Found 3 matches in src/db/connection.ts..."
}
```

El `tool_call_id` se usa **exactamente como lo devuelve el modelo vía LiteLLM**. El proxy no lo modifica, no le agrega prefijos, no lo transforma.

### 6.6 Traducciones que LiteLLM hace transparentemente

El proxy no implementa lógica de traducción de tools. LiteLLM la hace:

| Dirección                | Qué traduce LiteLLM                                                                                                           |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| OpenAI → Anthropic       | `parameters` → `input_schema`; `tool_calls[]` → `content[{type:"tool_use"}]`; `role:"tool"` → `role:"user"` con `tool_result` |
| Anthropic → OpenAI       | `content[{type:"tool_use"}]` → `tool_calls[]`; `input` (object) → `arguments` (string JSON)                                   |
| OpenAI → Gemini          | Wrap en `functionDeclarations[]`; JSON Schema → OpenAPI 3.0                                                                   |
| Gemini → OpenAI          | `parts[{functionCall}]` → `tool_calls[]`; `functionCall.args` → `arguments`                                                   |
| OpenAI → DeepSeek / Groq | Pass-through (hablan OpenAI nativo)                                                                                           |

**El proxy no escribe ni una línea de código de traducción de tools.** Solo verifica `openai_tools_compatible` antes de enviar.

### 6.7 Edge cases en tools

| Edge case                                           | Comportamiento del proxy                                                                                                                                                    |
| --------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Streaming parcial de tool calls                     | Acumular todos los chunks antes de guardar en DB. Si el stream se interrumpe, descartar tool call incompleta y notificar con `incomplete_tool_call: true` en proxy_metadata |
| Texto + tool calls en el mismo turno                | Almacenar `content` textual y `tool_calls` como campos separados del mismo objeto assistant                                                                                 |
| Tool call con error del cliente                     | Retornar `tool_result` con `content: "ERROR: <descripción>"`. El modelo decide si reintentar                                                                                |
| Resultado de tool >8K tokens                        | Truncar con marcador `[...truncado a 8K tokens...]`. El resultado completo se guarda en log                                                                                 |
| Thinking/reasoning blocks con tools                 | OpenAI: `reasoning_content` en delta. Anthropic: `thinking` blocks. Almacenar junto con tool_calls para preservar contexto de caché                                         |
| Modelo que ignora tools (`tool_choice: "required"`) | Si el modelo responde sin tool_calls cuando se requería una, marcar el modelo como no elegible para este turno y forzar fallback. Log del evento                            |

### 6.8 Operaciones explícitas sobre tools

**`POST /conversations/{id}/normalize-tools`**

Cuando hay parallel tool calls en el historial y el usuario quiere migrar a un modelo sin soporte parallel:

```
Antes:
  assistant: tool_calls [{id:"A"}, {id:"B"}, {id:"C"}]
  tool: result_A | tool: result_B | tool: result_C

Después (serializado):
  assistant: tool_calls [{id:"A"}]
  tool: result_A
  [TOOL_SERIALIZADA: originalmente paralela en turno #5]
  assistant: tool_calls [{id:"B"}]
  tool: result_B
  [TOOL_SERIALIZADA: originalmente paralela en turno #5]
  assistant: tool_calls [{id:"C"}]
  tool: result_C
```

El historial original se conserva. La versión normalizada es la que usa el proxy para el nuevo modelo.

**`GET /conversations/{id}/tools-compatibility`**

Retorna qué pseudo-modelos son compatibles dado el historial de tools de la conversación, con el nivel de soporte disponible en cada uno.

---

## 7. Multimedia: imágenes, audio, PDF y video

### 7.1 Flags de capability por conversación

Los flags son **aditivos** (nunca se desactivan sin operación explícita) y se acumulan desde el primer turno que los activa:

```json
{
  "conversation_id": "abc",
  "capabilities": {
    "has_images": true,
    "has_audio": false,
    "has_pdf": false,
    "has_video": false,
    "has_tools": true,
    "has_parallel_tools": false,
    "tools_eligible_models": ["deepseek-v4-pro", "deepseek-v4-flash"],
    "total_tokens": 45230
  }
}
```

### 7.2 Imágenes — soporte completo en v1

**Detección:** `content[{type: "image_url"}]` en cualquier mensaje activa `has_images: true`.

**Compatibilidad:** Si `has_images: true`, solo modelos con `vision: true` pueden atender la conversación.

**Auto-describe al migrar:** Si el pseudo-modelo destino tiene `image_handling.on_downgrade: "auto_describe"`:

```
1. Proxy detecta: cambio incompatible (destino sin vision, has_images: true)
2. Proxy verifica: destino tiene on_downgrade: auto_describe
3. Proxy envía cada imagen al modelo visual actual:
   Prompt: "Describe esta imagen en detalle para un modelo sin visión.
            Incluye: qué se ve, layout, texto visible, colores, elementos clave."
4. Descripciones se insertan como mensajes system:
   [IMAGEN_DESCRITA #1 — turno 3]: "Captura de pantalla de un dashboard..."
5. Flag has_images sigue true, pero images_described: true
6. El cambio procede. El nuevo modelo recibe descripciones textuales.
7. proxy_metadata: "images_described: 3, described_by: gemini-3.5-flash"
```

Si `on_downgrade: "block"` → HTTP 409 con opciones de remediación.
Si el usuario quiere degradar manualmente → `POST /conversations/{id}/degrade-images`.

### 7.3 Audio — detectado, no procesado en v1

**Estado:** Ningún pseudo-modelo definido declara soporte de audio actualmente.

**Detección:** `content[{type: "input_audio"}]` activa `has_audio: true`.

| Escenario                         | Comportamiento en v1                                                                 |
| --------------------------------- | ------------------------------------------------------------------------------------ |
| Audio en petición                 | ❌ Error `AUDIO_NOT_SUPPORTED` — el pseudo-modelo actual no tiene capacidad de audio |
| Audio en historial, usuario migra | ❌ Bloqueado — no hay camino de degradación automático en v1                         |

**Camino futuro (v2+):** Transcripción a texto via Whisper o Gemini Audio. Operación explícita `POST /degrade-audio`.

### 7.4 PDF — detectado, tratado como imagen en v1

**Detección:** `content[{type: "file"}]` con mimetype `application/pdf` activa `has_pdf: true`.

| Escenario                                    | Comportamiento en v1                                                                                  |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| PDF en petición, modelo con visión           | ⚠️ Advertencia — el PDF se trata como imagen (el modelo "ve" las páginas). Se activa `has_pdf: true`. |
| PDF en petición, modelo sin visión           | ❌ Error `PDF_NOT_SUPPORTED` — sugerir extracción previa de texto o usar modelo con visión            |
| PDF en historial, migrar a modelo sin visión | ❌ Bloqueado (misma regla que imágenes)                                                               |

**Camino futuro (v2+):** Extracción de texto con `pdfplumber` (MIT) o `PyMuPDF`. OCR con Tesseract para PDFs escaneados. Operación `POST /extract-pdf-text`.

### 7.5 Video — rechazado en v1

**Estado:** Sin soporte en ningún pseudo-modelo definido actualmente.

**Comportamiento:** Cualquier contenido de tipo video → Error `VIDEO_NOT_SUPPORTED` con sugerencia de extraer frames clave como imágenes.

### 7.6 Tabla resumen de soporte multimedia

| Tipo     | Detección                | v1 con modelo compatible  | v1 sin modelo compatible | Camino futuro     |
| -------- | ------------------------ | ------------------------- | ------------------------ | ----------------- |
| Imágenes | `image_url`              | ✅ Completo               | Auto-describe o bloqueo  | — (ya completo)   |
| Audio    | `input_audio`            | ✅ (si modelo lo soporta) | ❌ Error                 | Transcripción ASR |
| PDF      | `file` + pdf mime        | ⚠️ Como imagen            | ❌ Error                 | Extracción texto  |
| Video    | `video` / `file` + video | ❌ Error                  | ❌ Error                 | Extracción frames |

---

## 8. Compatibilidad entre pseudo-modelos

### 8.1 Lógica de validación

```python
def validate_switch(
    from_pseudo: PseudoModel,
    to_pseudo: PseudoModel,
    caps: Capabilities
) -> CompatibilityResult:

    # 1. Imágenes → modelo sin visión
    if caps.has_images and not to_pseudo.has_vision():
        if to_pseudo.image_handling.on_downgrade == "auto_describe":
            return Warning("IMAGES_WILL_BE_DESCRIBED",
                "Las imágenes del historial serán descritas textualmente antes de migrar.")
        return Blocked("IMAGES_INCOMPATIBLE", remediation=[
            "auto_describe: true en el pseudo-modelo destino",
            "POST /conversations/{id}/degrade-images",
        ])

    # 2. Parallel tools → modelo sin soporte parallel
    if caps.has_parallel_tools:
        eligible = [m for m in to_pseudo.physical_models if m.parallel_tools]
        if not eligible:
            return Blocked("PARALLEL_TOOLS_INCOMPATIBLE", remediation=[
                "POST /conversations/{id}/normalize-tools"
            ])

    # 3. Contexto acumulado → ventana insuficiente
    if caps.total_tokens > to_pseudo.context_window:
        return Blocked("CONTEXT_TOO_LARGE", remediation=[
            "POST /conversations/{id}/compact",
        ])

    # 3. Contexto acumulado → ventana insuficiente
    if caps.total_tokens > to_pseudo.context_window:
        return Blocked("CONTEXT_TOO_LARGE", remediation=[
            "POST /conversations/{id}/compact",
        ])

    # 4. Upgrade → siempre seguro
    return Safe()
```

### 8.2 Matriz de compatibilidad de referencia

| Origen                      | Destino                     | Situación                         | Estado         | Razón                                          |
| --------------------------- | --------------------------- | --------------------------------- | -------------- | ---------------------------------------------- |
| `normal`                    | `tareas-avanzadas`          | Sin multimedia, sin tools         | ✅ Seguro      | —                                              |
| `normal`                    | `tareas-avanzadas`          | Tools, sin parallel               | ✅ Seguro      | DeepSeek en destino                            |
| `normal`                    | `pensamiento-profundo-caro` | Cualquiera                        | ✅ Seguro      | Superset de capacidades                        |
| `normal`                    | `deep-flash`                | Sin multimedia, sin tools         | ⚠️ Advertencia | Pérdida de capacidad de razonamiento           |
| `normal`                    | `deep-flash`                | Con tools                         | ⚠️ Advertencia | Tools básicos, no confiable en casos complejos |
| `normal`                    | `flash-lowcost`             | Sin multimedia                    | ⚠️ Advertencia | Pérdida significativa de capacidad             |
| `normal`                    | `flash-lowcost`             | Con tools, parallel               | ❌ Bloqueado   | Sin parallel support en destino                |
| `tareas-avanzadas`          | `normal`                    | Sin parallel tools                | ✅ Seguro      | —                                              |
| `tareas-avanzadas`          | `normal`                    | Parallel tools                    | ⚠️ Advertencia | Qwen sin parallel; DeepSeek sí                 |
| `tareas-avanzadas`          | `pensamiento-profundo-caro` | Cualquiera                        | ✅ Seguro      | —                                              |
| `tareas-avanzadas`          | `deep-flash`                | Con tools                         | ⚠️ Advertencia | Soporte básico de tools                        |
| `pensamiento-profundo-caro` | `tareas-avanzadas`          | Cualquiera                        | ✅ Seguro      | —                                              |
| `pensamiento-profundo-caro` | `deep-flash`                | Con tools                         | ⚠️ Advertencia | Tools básicos                                  |
| `avanzada-vision`           | `flash-vision`              | Con imágenes                      | ⚠️ Advertencia | Capacidad visual reducida                      |
| `avanzada-vision`           | `normal`                    | Con imágenes, auto_describe=false | ❌ Bloqueado   | Requiere degradación de imágenes               |
| `avanzada-vision`           | `normal`                    | Con imágenes, auto_describe=true  | ⚠️ Advertencia | Imágenes serán descritas automáticamente       |
| `avanzada-vision`           | `deep-flash`                | Con imágenes                      | ❌ Bloqueado   | Sin visión + sin auto_describe en destino      |
| `flash-vision`              | `avanzada-vision`           | Con imágenes                      | ✅ Seguro      | Upgrade de capacidad visual                    |
| `deep-flash`                | `normal`                    | Sin multimedia                    | ✅ Seguro      | —                                              |
| `deep-flash`                | `normal`                    | Con multimedia                    | ✅ Seguro      | Destino tiene mejor soporte                    |
| `flash-lowcost`             | `normal`                    | Cualquiera                        | ✅ Seguro      | —                                              |
| Cualquiera                  | `compactador`               | Cualquiera                        | ✅ Siempre     | Es operación, no conversación                  |

### 8.3 Regla general (resumen)

```
✅ SEGURO   → destino soporta todas las capacidades del historial
⚠️ WARN     → destino tiene menor capacidad pero el historial no se rompe
❌ BLOQUEADO → destino no puede interpretar contenido ya presente en el historial

Los bloqueos siempre incluyen opciones de remediación.
El usuario puede forzar el cambio SOLO ejecutando la operación de remediación explícita.
```

---

## 9. Pre-compactación de entrada

### 9.1 El problema

Un usuario selecciona `pensamiento-profundo-caro` y pega 80,000 tokens de logs de sistema. Enviar eso a un modelo de razonamiento profundo es costoso e innecesario: los logs son datos brutos, no necesitan razonamiento de máximo nivel para ser resumidos.

### 9.2 La solución (solo cuando está habilitado en el pseudo-modelo)

Si `pre_compaction.enabled: true` y el input supera `pre_compaction.threshold`:

```
1. Proxy detecta: input 80K tokens > threshold 32K
2. Proxy envía el input a "deep-flash" con prompt:
   "Extrae del siguiente texto solo la información relevante para la tarea del usuario.
    Responde con un resumen estructurado de máximo {target_tokens} tokens.
    Tarea del usuario: {user_prompt}"
3. El modelo barato genera resumen de ~6K tokens
4. Proxy reemplaza el input original por el resumen
5. Proxy envía el resumen al modelo caro
6. proxy_metadata:
   {
     "pre_compaction_applied": true,
     "original_input_tokens": 80000,
     "compacted_input_tokens": 6000,
     "compactor_model": "glm-4.5-flash",
     "estimated_savings_usd": 0.24
   }
```

Si `pre_compaction.enabled: false` y el input supera el umbral → Error `INPUT_EXCEEDS_THRESHOLD`. El proxy **nunca** pre-compacta sin que esté configurado explícitamente en el pseudo-modelo.

---

## 10. Compactación continua para modelos caros

### 10.1 El problema

30 turnos con `pensamiento-profundo-caro`. El contexto acumulado es 170K tokens. El modelo puede procesarlo, pero cada turno re-procesa 170K tokens de historial. El costo se multiplica con cada turno.

### 10.2 La solución (solo cuando está habilitado)

Si `continuous_compaction.enabled: true` y el contexto supera `trigger_pct` del `context_window`:

```
Turno 20: contexto = 150K tokens (75% de 200K) → dispara
  → proxy compacta turnos 1-15 en snapshot
  → contexto activo = [snapshot ~8K] + [turnos 16-20 ~40K] = ~48K tokens
  → modelo recibe 48K en lugar de 150K → ahorro ~68%
  → proxy_metadata: {continuous_compaction_applied: true, tokens_before: 150000, tokens_after: 48000}

Turno 35: contexto = 95K tokens (47%) → no dispara

Turno 50: contexto = 165K tokens (82%) → dispara
  → nuevo snapshot: compactar [snapshot_1 + turnos 16-45]
  → contexto activo = [snapshot_2 ~12K] + [turnos 46-50 ~28K] = ~40K tokens
```

### 10.3 Calidad del snapshot

El snapshot no es un resumen genérico. El prompt de compactación extrae:

```
- El problema o tarea central que se trabajaba
- Las decisiones técnicas tomadas y sus justificaciones (no solo el resultado)
- El código producido más relevante para continuar (no todo)
- El estado actual del problema: qué está resuelto, qué no, qué estaba en progreso
- Las restricciones y contexto técnico no obvios (variables de entorno, arquitectura, dependencias)
- Los puntos pendientes y lo que el usuario iba a hacer a continuación
```

---

## 11. Compactación explícita de conversaciones

### 11.1 Cuándo ocurre

El historial puede llegar a un punto donde supera la ventana de **todos** los modelos disponibles. En ese estado, la conversación es inutilizable.

```
proxy_metadata cuando context_usage_pct alcanza:
  < 60%  → solo informativo (número de tokens y porcentaje)
  60-80% → "CONTEXT_MODERATE: considera compactar pronto"
  80-99% → "CONTEXT_HIGH: warning" + compaction_endpoint
  100%+  → HTTP 400: "CONTEXT_UNUSABLE" + compaction_endpoint (única acción)
```

### 11.2 Flujo de compactación explícita

```mermaid
sequenceDiagram
    participant U as Usuario
    participant P as Proxy
    participant DB as PostgreSQL
    participant C as Compactador (Gemini 3 Flash)

    U->>P: POST /conversations/{id}/compact
    P->>DB: Leer historial completo
    DB-->>P: Historial (ej: 1.2M tokens)
    P->>P: Seleccionar modelo con ventana suficiente (by_context_window)
    P->>C: Prompt de compactación estructurado
    C-->>P: Snapshot Markdown (~8-12K tokens)
    P->>DB: Guardar snapshot (tipo: compaction_snapshot)
    P->>DB: Registrar evento: {tokens_antes: 1.2M, tokens_después: 10K}
    P-->>U: {snapshot_id, tokens_reducidos, preview_del_snapshot}
    U->>P: Siguiente petición normal
    P->>DB: Cargar snapshot + turnos posteriores al snapshot
    P->>Modelo: Contexto = [snapshot] + [nuevos turnos]
```

### 11.3 Estructura del snapshot generado

```markdown
# Snapshot de conversación — 2025-01-15T14:32:00Z

## Historial original: ~1,200,000 tokens | Pseudo-modelo: tareas-avanzadas

### Estado del problema

[Descripción del problema o tarea que se trabajaba al momento de compactar]

### Decisiones técnicas tomadas

- [Decisión con razonamiento, no solo el resultado]
- [Decisión con razonamiento]

### Código producido (extractos clave)

[Solo el código que establece el estado actual. No todo. Solo lo necesario para continuar.]

### Estado actual

- Resuelto: [lista]
- Sin resolver: [lista]
- En progreso al compactar: [descripción]

### Contexto técnico importante

[Variables de entorno, restricciones arquitectónicas, dependencias, convenciones del proyecto]

### Puntos pendientes

[Lo que el usuario iba a hacer a continuación]
```

### 11.4 Múltiples compactaciones

Una conversación puede compactarse N veces. Cada compactación genera un nuevo snapshot que reemplaza al anterior como punto de inicio activo. Los snapshots anteriores se conservan en DB con timestamp. El historial original nunca se modifica.

---

## 12. Router LLM opcional

### 12.1 Qué hace

Si `router_llm.enabled: true` en el pseudo-modelo, el proxy evalúa la complejidad de la tarea con un modelo barato antes de enviarla al modelo caro.

```
1. Proxy envía el user prompt a "flash-lowcost" con prompt:
   "Evalúa la complejidad de esta tarea para un modelo de IA.
    Responde SOLO con JSON válido:
    {
      'complexity': 'simple' | 'medium' | 'complex',
      'suggested_pseudo_model': 'flash-lowcost' | 'normal' | 'tareas-avanzadas' | 'pensamiento-profundo-caro',
      'reason': 'una frase explicando por qué'
    }"
2. Si el router sugiere un modelo más barato (downgrade):
   proxy_metadata: {
     "router_suggestion": {
       "suggested": "normal",
       "reason": "pregunta de búsqueda simple, no requiere razonamiento profundo"
     }
   }
3. El proxy NO cambia el modelo. Solo informa. El usuario decide.
```

### 12.2 Lo que nunca hace el router

- No cambia el modelo de la petición actual
- No cancela la petición al modelo caro
- No es bloqueante: si el router falla, la petición continúa al modelo original
- No sugiere upgrades, solo downgrades (`suggest_on_downgrade_only: true`)

---

## 13. Capa de optimización de caché de proveedor

### 13.1 Orden canónico de mensajes

El proxy ensambla siempre el prompt en el mismo orden para maximizar cache hits:

```
1. System prompt (estático, no cambia entre turnos)
2. Tool definitions (estáticas, ordenadas alfabéticamente por nombre de función)
3. Conversation history (del más antiguo al más reciente)
4. New user message (al final, varía en cada turno)
```

El prefijo (system + tools) es idéntico entre turnos de la misma conversación → el proveedor reutiliza el caché.

### 13.2 Estrategias por proveedor

**OpenAI:**

- `prompt_cache_key` por conversación (hash del `conversation_id + system prompt`)
- Caché automático para prompts ≥1024 tokens. TTL: 5-10 minutos de inactividad.

**Anthropic:**

- `cache_control: {type: "ephemeral"}` en system prompt y tool definitions
- Máximo 4 breakpoints de caché. Se colocan después de system y después de tools.
- Cache writes: 1.25× costo. Cache reads: 0.1× costo.

**Google Gemini:**

- `CachedContent` explícito al inicio de la conversación.
- `cache_id` reutilizado en todos los turnos siguientes.
- TTL: 60 minutos de inactividad (configurable).

### 13.3 Lo que destruye el caché (y el proxy lo evita)

| Causa de cache miss        | Cómo lo evita el proxy                             |
| -------------------------- | -------------------------------------------------- |
| Cambio de modelo físico    | Afinidad de modelo por conversación                |
| Cambio en tool definitions | Ordenamiento alfabético estable entre turnos       |
| JSON keys no estables      | `json.dumps(sort_keys=True)` en toda serialización |
| Timestamps en el prefijo   | Eliminados del contenido cacheable                 |
| Cambio de sistema prompt   | System prompt fijo por pseudo-modelo               |

### 13.4 Métricas de caché en proxy_metadata

```json
{
  "proxy_metadata": {
    "cache": {
      "provider_cache_hit": true,
      "cached_tokens": 45000,
      "cache_read_tokens": 42000,
      "cache_write_tokens": 3000,
      "estimated_savings_usd": 0.042
    }
  }
}
```

---

## 14. API del proxy

### 14.1 Endpoints

| Método | Ruta                                      | Auth | Descripción                                            |
| ------ | ----------------------------------------- | ---- | ------------------------------------------------------ |
| `POST` | `/v1/chat/completions`                    | ✅   | Endpoint principal (OpenAI-compatible + streaming SSE) |
| `GET`  | `/v1/models`                              | ✅   | Lista pseudo-modelos con capabilities                  |
| `GET`  | `/conversations/{id}`                     | ✅   | Estado completo: flags, tokens, snapshot activo        |
| `GET`  | `/conversations/{id}/compatible-models`   | ✅   | Qué pseudo-modelos son compatibles y por qué           |
| `GET`  | `/conversations/{id}/tools-compatibility` | ✅   | Nivel de tools requerido vs disponible                 |
| `POST` | `/conversations/{id}/change-pseudo-model` | ✅   | Cambiar pseudo-modelo con validación                   |
| `POST` | `/conversations/{id}/compact`             | ✅   | Compactación explícita                                 |
| `POST` | `/conversations/{id}/degrade-images`      | ✅   | Describir imágenes → texto                             |
| `POST` | `/conversations/{id}/normalize-tools`     | ✅   | Serializar parallel tool calls                         |
| `GET`  | `/conversations/{id}/audit-log`           | ✅   | Trazabilidad de decisiones del proxy                   |
| `GET`  | `/models/{pseudo_model}/tool-level`       | ✅   | Capacidades de tools de un pseudo-modelo               |
| `GET`  | `/health`                                 | ✗    | Estado: proxy, DB, Valkey, proveedores                 |
| `GET`  | `/metrics`                                | ✅   | Estadísticas: tokens, ahorros, cache hit rate          |

### 14.2 Formato de respuesta

```json
{
  "id": "chatcmpl-abc",
  "object": "chat.completion",
  "model": "normal",
  "choices": [{ "message": { "role": "assistant", "content": "..." } }],
  "usage": { "prompt_tokens": 5000, "completion_tokens": 800 },
  "proxy_metadata": {
    "physical_model": "qwen3-max",
    "pseudo_model": "normal",
    "conversation_id": "conv-abc123",
    "context_tokens_total": 45000,
    "context_usage_pct": 47,
    "pseudo_model_threshold": 96000,
    "affinity_maintained": true,
    "fallback_applied": false,
    "fallback_reason": null,
    "pre_compaction_applied": false,
    "continuous_compaction_applied": false,
    "router_suggestion": null,
    "tools_filter_applied": false,
    "images_described": 0,
    "estimated_cost_usd": 0.003,
    "session_cost_usd": 0.15,
    "cache": {
      "provider_cache_hit": true,
      "cached_tokens": 4500,
      "estimated_savings_usd": 0.004
    },
    "warning": null
  }
}
```

### 14.3 Códigos de error

| HTTP | Código                        | Significado                                                                                    |
| ---- | ----------------------------- | ---------------------------------------------------------------------------------------------- |
| 400  | `INPUT_EXCEEDS_THRESHOLD`     | Input supera umbral del pseudo-modelo. Incluye sugerencias de pseudo-modelos con mayor umbral. |
| 400  | `CONTEXT_UNUSABLE`            | Historial total > ventana de todos los modelos. Única acción: compactar.                       |
| 400  | `AUDIO_NOT_SUPPORTED`         | Audio en petición, ningún modelo del pseudo-modelo lo soporta                                  |
| 400  | `PDF_NOT_SUPPORTED`           | PDF en petición, modelo sin visión                                                             |
| 400  | `VIDEO_NOT_SUPPORTED`         | Video en petición (no soportado en v1)                                                         |
| 409  | `PSEUDO_MODEL_INCOMPATIBLE`   | Cambio de pseudo-modelo incompatible. Incluye remediation options.                             |
| 409  | `IMAGES_INCOMPATIBLE`         | Imágenes en historial, destino sin visión y sin auto_describe                                  |
| 409  | `PARALLEL_TOOLS_INCOMPATIBLE` | Historial con parallel tools, destino sin soporte. Ofrece normalize-tools.                     |
| 503  | `ALL_MODELS_FAILED`           | Todos los modelos del pseudo-modelo fallaron (503/429). Incluye razones por modelo.            |

---

## 15. Integración con OpenCode

### 15.1 Configuración mínima

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "model": "cesar-proxy/normal",
  "provider": {
    "cesar-proxy": {
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "${CESAR_PROXY_URL}",
        "apiKey": "${CESAR_PROXY_KEY}",
      },
    },
  },
}
```

```bash
# ~/.bashrc o ~/.zshrc
export CESAR_PROXY_URL="https://proxy.tudominio.com/v1"
export CESAR_PROXY_KEY="sk-proxy-xxxxxxxxxxxxxxxxxxxx"
```

El proxy expone `GET /v1/models` con la lista de pseudo-modelos y sus capabilities. OpenCode puede auto-descubrirlos.

### 15.2 Declaración manual de modelos (si el auto-descubrimiento falla)

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "model": "cesar-proxy/normal",
  "provider": {
    "cesar-proxy": {
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "${CESAR_PROXY_URL}",
        "apiKey": "${CESAR_PROXY_KEY}",
      },
      "models": {
        "pensamiento-profundo-caro": {
          "name": "pensamiento-profundo-caro",
          "capabilities": {
            "tools": true,
            "vision": false,
            "streaming": true,
            "function_calling": true,
            "parallel_tool_calls": true,
          },
        },
        "tareas-avanzadas": {
          "name": "tareas-avanzadas",
          "capabilities": {
            "tools": true,
            "vision": false,
            "streaming": true,
            "function_calling": true,
            "parallel_tool_calls": true,
          },
        },
        "normal": {
          "name": "normal",
          "capabilities": {
            "tools": true,
            "vision": false,
            "streaming": true,
            "function_calling": true,
            "parallel_tool_calls": false,
          },
        },
        "deep-flash": {
          "name": "deep-flash",
          "capabilities": {
            "tools": true,
            "vision": false,
            "streaming": true,
            "function_calling": true,
            "parallel_tool_calls": false,
          },
        },
        "flash-lowcost": {
          "name": "flash-lowcost",
          "capabilities": {
            "tools": false,
            "vision": false,
            "streaming": true,
            "function_calling": false,
            "parallel_tool_calls": false,
          },
        },
        "avanzada-vision": {
          "name": "avanzada-vision",
          "capabilities": {
            "tools": false,
            "vision": true,
            "streaming": true,
            "function_calling": false,
            "parallel_tool_calls": false,
          },
        },
        "flash-vision": {
          "name": "flash-vision",
          "capabilities": {
            "tools": false,
            "vision": true,
            "streaming": true,
            "function_calling": false,
            "parallel_tool_calls": false,
          },
        },
      },
    },
  },
}
```

### 15.3 conversation_id en OpenCode

OpenCode no envía `conversation_id` nativo. El proxy puede derivarlo del historial entrante:

```python
def derive_conversation_id(request: ChatRequest) -> str:
    # Opción A: El cliente lo envía en el header
    if conv_id := request.headers.get("X-Conversation-ID"):
        return conv_id
    # Opción B: El cliente lo envía en el body
    if conv_id := request.body.get("conversation_id"):
        return conv_id
    # Opción C: Derivar de un hash del primer mensaje (determinista)
    first_msg = request.body.messages[0]
    return f"conv-{sha256(first_msg.content.encode()).hexdigest()[:16]}"
```

La Opción C permite que aunque OpenCode no envíe un ID explícito, el mismo primer mensaje siempre derive el mismo `conversation_id`. Esto mantiene la afinidad incluso si OpenCode no lo soporta explícitamente.

### 15.4 Streaming SSE

El proxy forwardea los chunks de LiteLLM al cliente sin bufferizar:

```
OpenCode → Proxy (FastAPI) → LiteLLM → Proveedor
                │
            Streaming SSE chunk por chunk
```

- El contenido de los chunks no se modifica
- `proxy_metadata` se añade solo en el chunk final (chunk `[DONE]`)
- Si el streaming incluye tool call deltas, el proxy los acumula para almacenar el turno completo en DB, pero los forwardea sin modificar

### 15.5 Convivencia con la compactación interna de OpenCode

OpenCode tiene su propio sistema de compactación automática. No hay conflicto:

- **Compactación de OpenCode:** reduce el historial que OpenCode envía al proxy
- **Compactación del proxy:** opera sobre lo que recibe, compactando lo que el proxy envía al proveedor
- Son capas complementarias. Las respuestas del modelo llegan a OpenCode sin modificaciones de contenido.

---

## 16. Despliegue remoto

### 16.1 Topología

```
┌──────────────────────┐     HTTPS + Bearer token     ┌─────────────────────────────┐
│  CLIENTE              │ ──────────────────────────► │  SERVIDOR DEL PROXY          │
│  OpenCode / Scripts   │                              │  FastAPI :9110               │
│  (tu máquina)         │                              │  LiteLLM                     │
│                       │                              │  PostgreSQL + Valkey         │
│  Solo tiene:          │                              │                              │
│  - CESAR_PROXY_URL    │                              │  Solo el servidor tiene:     │
│  - CESAR_PROXY_KEY    │                              │  - ANTHROPIC_API_KEY         │
└──────────────────────┘                              │  - OPENAI_API_KEY            │
                                                       │  - GOOGLE_API_KEY            │
                                                       │  - DEEPSEEK_API_KEY          │
                                                       │  - GROQ_API_KEY              │
                                                       └─────────────────────────────┘
```

**Las API keys de proveedores nunca salen del servidor.** El cliente solo maneja la key del proxy.

### 16.2 Variables de entorno del servidor

```bash
# Proxy
PROXY_API_KEY=sk-proxy-xxxxxxxxxxxxxxxxxxxx
PROXY_PORT=9110
CORS_ORIGINS=https://tudominio.com,vscode-webview://...

# Base de datos
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/proxy_db
VALKEY_URL=valkey://localhost:6379

# Proveedores LLM (nunca salen del servidor)
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxx
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
GOOGLE_API_KEY=AIza-xxxxxxxxxxxxxxxxxxxx
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
ZHIPU_API_KEY=xxxxxxxxxxxxxxxxxxxx
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
```

### 16.3 HTTPS con Caddy (recomendado)

```
# /etc/caddy/Caddyfile
proxy.tudominio.com {
    reverse_proxy localhost:9110
    # Caddy obtiene y renueva certificados Let's Encrypt automáticamente
}
```

### 16.4 Rate limiting por pseudo-modelo

```python
RATE_LIMITS = {
    "pensamiento-profundo-caro": "5/minute",
    "tareas-avanzadas":          "20/minute",
    "normal":                    "60/minute",
    "deep-flash":                "120/minute",
    "flash-lowcost":             "200/minute",
    "avanzada-vision":           "10/minute",
    "flash-vision":              "30/minute",
    "compactador":               "5/minute",
}
```

### 16.5 Health check

```
GET /health → no requiere auth

200: { "status": "ok",
       "postgres": "connected",
       "valkey": "connected",
       "providers": ["anthropic", "openai", "google", "deepseek", "groq", "zhipu"] }

503: { "status": "degraded",
       "postgres": "connected",
       "valkey": "disconnected" }
```

### 16.6 Checklist de despliegue

- [ ] Proxy corriendo en servidor remoto (no localhost del cliente)
- [ ] HTTPS configurado con certificado válido (Caddy + Let's Encrypt)
- [ ] `PROXY_API_KEY` generada y distribuida al cliente
- [ ] Keys de proveedores en `.env` del servidor, nunca en el cliente
- [ ] CORS configurado para clientes browser
- [ ] `GET /health` respondiendo correctamente
- [ ] PostgreSQL y Valkey accesibles desde el proxy
- [ ] Rate limiting activo
- [ ] Logs a stdout/stderr (para docker/systemd journal)
- [ ] Cliente configurado con `baseURL` y `Authorization: Bearer <PROXY_API_KEY>`

---

## 17. Estructura del proyecto

```
proxy/
├── pyproject.toml
├── pseudo_models.yaml          # Definición de pseudo-modelos (fuente de verdad)
├── alembic.ini
├── README.md
│
├── src/
│   ├── main.py                 # FastAPI app, middleware, lifespan
│   ├── config.py               # Carga y validación de pseudo_models.yaml + .env
│   ├── auth.py                 # Validación de PROXY_API_KEY
│   │
│   ├── models/
│   │   ├── pseudo_model.py     # Pydantic schemas: PseudoModel, PhysicalModel
│   │   ├── capabilities.py     # SessionCapabilities, CompatibilityResult
│   │   └── db.py               # SQLAlchemy ORM: Conversation, Turn, Snapshot
│   │
│   ├── middleware/
│   │   ├── capability_detector.py   # Detectar has_images, has_tools, has_parallel_tools
│   │   ├── compatibility.py         # validate_switch() — matriz completa
│   │   ├── threshold_guard.py       # Verificar umbrales de input y contexto
│   │   └── tool_filter.py           # Filtrar modelos por openai_tools_compatible
│   │
│   ├── tools/
│   │   ├── canonical.py             # Almacenamiento en formato OpenAI canónico
│   │   ├── normalizer.py            # POST /normalize-tools (serializar parallel)
│   │   └── edge_cases.py            # Streaming parcial, mixed, resultados grandes
│   │
│   ├── compactor/
│   │   ├── pre_compactor.py         # Pre-compactación de input largo
│   │   ├── continuous.py            # Compactación continua por trigger_pct
│   │   └── explicit.py              # POST /compact
│   │
│   ├── multimedia/
│   │   ├── image_describer.py       # Auto-describe imágenes al migrar
│   │   ├── detector.py              # Detectar tipo de multimedia en mensajes
│   │   └── degradation.py           # POST /degrade-images, futuros degrade-audio
│   │
│   ├── cache/
│   │   ├── affinity.py              # Afinidad de modelo físico en Valkey
│   │   ├── message_ordering.py      # Orden canónico: system→tools→history→query
│   │   └── provider_cache.py        # prompt_cache_key, cache_control, CachedContent
│   │
│   ├── router_llm/
│   │   └── suggester.py             # Evaluar complejidad, sugerir downgrade
│   │
│   ├── api/
│   │   ├── chat.py                  # POST /v1/chat/completions + SSE streaming
│   │   ├── models.py                # GET /v1/models, GET /models/{id}/tool-level
│   │   └── conversations.py         # CRUD, compact, degrade, normalize, audit
│   │
│   ├── db/
│   │   ├── session.py               # Async SQLAlchemy session
│   │   └── migrations/
│   │
│   └── tasks/
│       └── celery_app.py            # Celery para ops asíncronas pesadas
│
└── tests/
    ├── test_pseudo_models.py        # Validación de config
    ├── test_compatibility.py        # Matriz completa
    ├── test_tools.py                # Tools en cada proveedor
    ├── test_tool_filter.py          # Filtrado por openai_tools_compatible
    ├── test_tool_normalization.py   # Serialización de parallel calls
    ├── test_pre_compaction.py
    ├── test_continuous_compaction.py
    ├── test_image_describe.py
    ├── test_cache.py
    ├── test_streaming.py
    └── test_e2e.py                  # OpenCode → proxy → proveedor
```

---

## 18. Plan de sprints

### Sprint 1 — MVP: pseudo-modelos + afinidad + streaming _(2 semanas)_

**Objetivo:** El proxy funciona. `"normal"` → Qwen. Afinidad mantenida. Streaming correcto.

- [ ] Proyecto FastAPI + estructura de directorios
- [ ] `pseudo_models.yaml` con 8 pseudo-modelos. Validación Pydantic estricta al inicio.
- [ ] `POST /v1/chat/completions` — mapeo pseudo-modelo → LiteLLM → proveedor
- [ ] Streaming SSE: forward de chunks sin bufferizar; `proxy_metadata` en chunk final
- [ ] Afinidad en Valkey: guardar y recuperar modelo físico por `conversation_id`
- [ ] Fallback secuencial dentro del pseudo-modelo (503/429)
- [ ] `proxy_metadata` básico: physical_model, context_usage_pct, fallback_applied
- [ ] `GET /v1/models` — lista pseudo-modelos con capabilities
- [ ] Configurar LiteLLM con todos los proveedores
- [ ] `GET /health`

**Criterio de éxito:** 20 turnos consecutivos en la misma conversación → siempre el mismo modelo físico. Streaming funciona con `stream: true`. Fallback funciona cuando el modelo primario está caído.

---

### Sprint 2 — Capabilities y compatibilidad _(2 semanas)_

**Objetivo:** El proxy detecta multimedia y tools. Bloquea cambios incompatibles.

- [ ] `CapabilityDetector`: imágenes, audio, tools, parallel tools en cada mensaje
- [ ] Acumulación de flags en PostgreSQL (aditivos)
- [ ] `tool_filter.py`: filtrar pool por `openai_tools_compatible` cuando `has_tools`
- [ ] `validate_switch()`: implementar las reglas de la sección 8.1
- [ ] `GET /conversations/{id}/compatible-models`
- [ ] `GET /conversations/{id}/tools-compatibility`
- [ ] Errores 409 con `remediation` en lenguaje natural
- [ ] Tests para ≥20 combinaciones de la matriz

**Criterio de éxito:** Cambiar de `avanzada-vision` a `normal` con imágenes → HTTP 409 con opciones. MiniMax soporta tools (nivel básico). `GET /compatible-models` es determinista.

---

### Sprint 3 — Normalización de tools y formato canónico _(2 semanas)_

**Objetivo:** Tools funcionan confiablemente. Historial portable entre pseudo-modelos.

- [ ] Almacenamiento en formato canónico OpenAI en DB
- [ ] Verificar que LiteLLM traduce correctamente: OpenAI↔Anthropic↔Gemini con tools
- [ ] Tests end-to-end de tools en cada proveedor (simple, parallel, streaming)
- [ ] `POST /normalize-tools`: serializar parallel calls a secuenciales
- [ ] Edge cases: streaming parcial, mixed content, tool errors, resultados grandes, thinking blocks
- [ ] Manejo de `tool_choice: "required"` + modelo que ignora tools → fallback forzado
- [ ] Tests de migración de historial con tools entre pseudo-modelos

**Criterio de éxito:** Un agente con parallel tools en `tareas-avanzadas` (DeepSeek V4 Pro) → `POST /normalize-tools` → migra a `normal` (Qwen3 Max)

---

### Sprint 4 — Pre-compactación y compactación continua _(2 semanas)_

**Objetivo:** Los modelos caros no reciben inputs enormes ni contextos saturados.

- [ ] `PreCompactor`: detectar input > threshold, resumir con modelo barato, reemplazar
- [ ] `ContinuousCompactor`: detectar contexto > trigger_pct, compactar snapshot
- [ ] Prompt de compactación estructurado (decisiones, código, estado, pendientes)
- [ ] Tabla `snapshots` en PostgreSQL
- [ ] Múltiples compactaciones encadenadas
- [ ] proxy_metadata: tokens_before, tokens_after, savings_usd, compactor_model
- [ ] Tests: input 100K → pre-compacted a 8K → modelo caro recibe 8K
- [ ] Tests: 30 turnos con pensamiento-profundo-caro → compactación al 70%

**Criterio de éxito:** 80K tokens de logs a `pensamiento-profundo-caro` → proxy resume con `deep-flash` a 8K → el usuario ve exactamente cuánto ahorró.

---

### Sprint 5 — Auto-describe imágenes y router LLM _(2 semanas)_

**Objetivo:** Migrar de modelos visuales es fluido. El router informa sin imponer.

- [ ] `ImageDescriber`: al cambiar con `auto_describe: true`, describir imágenes con modelo visual actual
- [ ] Anotaciones `[IMAGEN_DESCRITA #N]` en historial, preservando original
- [ ] `POST /degrade-images`: degradación manual
- [ ] `RouterLLM`: evaluar complejidad, retornar sugerencia en `proxy_metadata`
- [ ] Tests: conversación con 3 imágenes → cambiar a `normal` (auto_describe=true) → continúa
- [ ] Tests: pregunta simple en `pensamiento-profundo-caro` → router sugiere `normal`

**Criterio de éxito:** El usuario cambia de `avanzada-vision` a `normal`. Las 3 imágenes son descritas automáticamente. La conversación continúa sin pérdida conceptual. El router nunca cambia el modelo.

---

### Sprint 6 — Compactación explícita y alertas _(1 semana)_

**Objetivo:** Las conversaciones nunca quedan permanentemente inutilizables.

- [ ] `POST /compact`: compactación explícita con snapshot Markdown legible
- [ ] Alertas de contexto en proxy_metadata (60%, 80%, 100%)
- [ ] HTTP 400 `CONTEXT_UNUSABLE` cuando ningún modelo puede atender la conversación
- [ ] `GET /conversations/{id}/audit-log`
- [ ] Celery para compactaciones de historiales >500K tokens
- [ ] Tests: conversación de 1.2M tokens → compact → snapshot → retomable

**Criterio de éxito:** Una conversación de 2M tokens puede compactarse con Gemini 3 Flash (1M context) y reactivarse con un snapshot de ~10K tokens.

---

### Sprint 7 — Optimización de caché de proveedor _(1 semana)_

**Objetivo:** Los cache hits son máximos. OpenCode funciona sin configuración adicional.

- [ ] Orden canónico de mensajes (system → tools → history → query)
- [ ] `prompt_cache_key` para OpenAI
- [ ] `cache_control` breakpoints para Anthropic
- [ ] `CachedContent` para Gemini
- [ ] `json.dumps(sort_keys=True)` en toda serialización de tools
- [ ] Métricas de caché en proxy_metadata
- [ ] `opencode.json` de ejemplo listo para copiar
- [ ] Tests end-to-end: OpenCode → proxy → proveedor → respuesta con tools

**Criterio de éxito:** 10 turnos seguidos en una conversación → 8+ cache hits en el proveedor. OpenCode funciona apuntando solo al proxy con la key correspondiente.

---

### Sprint 8 — Despliegue y observabilidad _(1 semana)_

**Objetivo:** El proxy está en producción. Todo es trazable. Un nuevo usuario lo configura en <5 minutos.

- [ ] Autenticación con `PROXY_API_KEY` (Bearer token)
- [ ] CORS para clientes browser
- [ ] Rate limiting por pseudo-modelo en Valkey
- [ ] Logs estructurados JSON de cada decisión del proxy
- [ ] `GET /metrics`: tokens ahorrados, costo evitado, cache hit rate
- [ ] `Caddyfile` de ejemplo para HTTPS
- [ ] README con pasos de configuración y ejemplos de uso
- [ ] Tests de estrés: 50 conversaciones concurrentes con afinidad correcta

**Criterio de éxito:** Un nuevo usuario copia el README, configura el proxy en <5 minutos y conecta OpenCode sin modificar OpenCode.

---

**Total: 13 semanas**

---

## 19. Historias de usuario

### HU-01: Mantener continuidad en sesión larga

**Como** desarrollador trabajando en debugging de varias horas,
**quiero** que el proxy use siempre el mismo modelo físico mientras no cambie de pseudo-modelo,
**para** mantener el caché del proveedor y no pagar el costo de re-procesar el contexto.

**Criterios:** modelo físico fijado desde el turno 1; fallback notificado en proxy_metadata; afinidad persiste 24h; el modelo activo visible en cada respuesta.

---

### HU-02: Escalar sin perder contexto

**Como** usuario en `normal` que necesita más capacidad para una decisión difícil,
**quiero** escalar a `pensamiento-profundo-caro` sin perder el historial,
**para** continuar donde lo dejé.

**Criterios:** cambio sin interrupciones si compatible; historial en formato canónico traducido al formato del nuevo proveedor; cambio registrado en audit log; afinidad actualizada.

---

### HU-03: Error claro al cambio incompatible por tools

**Como** usuario con parallel tool calls en la conversación,
**quiero** recibir un error descriptivo si intento bajar a `flash-lowcost`,
**para** entender por qué y qué opciones tengo.

**Criterios:** HTTP 409 `PARALLEL_TOOLS_INCOMPATIBLE`; error explica el problema en lenguaje natural; lista pseudo-modelos compatibles; ofrece `POST /normalize-tools` como camino de remediación.

---

### HU-04: Migrar conversación visual a modelo sin visión

**Como** usuario con imágenes en la conversación que quiere cambiar a `normal`,
**quiero** que las imágenes se describan textualmente si configuro `auto_describe: true`,
**para** continuar sin perder el contexto conceptual.

**Criterios:** descripciones con tag `[IMAGEN_DESCRITA #N]`; historial original intacto; flag `has_images` sigue true; `images_described: true` en proxy_metadata; cambio procede.

---

### HU-05: Compactar antes de llegar al límite

**Como** usuario con 3 días de trabajo en la misma conversación agéntica,
**quiero** compactar el historial cuando el proxy me avise que se acerca al límite,
**para** seguir trabajando sin perder el contexto técnico.

**Criterios:** proxy informa `context_usage_pct` en cada respuesta; warning al 80%; `CONTEXT_UNUSABLE` al 100%; `POST /compact` genera snapshot Markdown legible con decisiones, código clave, estado y pendientes; peticiones siguientes usan snapshot + nuevos turnos; historial original intacto.

---

### HU-06: Recuperar conversación inutilizable

**Como** usuario cuya conversación acumuló 1.5M tokens,
**quiero** compactarla para recuperar el acceso,
**para** no perder semanas de trabajo.

**Criterios:** `CONTEXT_UNUSABLE` incluye endpoint de compactación; compactador selecciona Gemini 3 Flash (1M context) automáticamente; snapshot resultante (~10K tokens) permite retomar inmediatamente.

---

### HU-07: Sub-agentes concurrentes baratos

**Como** sistema agéntico,
**quiero** hacer peticiones concurrentes a `flash-lowcost` para múltiples sub-tareas,
**para** procesar muchas entradas simples a costo mínimo.

**Criterios:** peticiones concurrentes sin mezclar contextos; rate limiting por pseudo-modelo; cada sub-agente tiene su `conversation_id` e historial independiente; costo acumulado por sesión visible en métricas.

---

### HU-08: Saber qué cambios son posibles sin intentarlos

**Como** usuario que quiere cambiar de pseudo-modelo,
**quiero** consultar qué opciones son compatibles con mi historial actual,
**para** no intentar cambios que van a fallar.

**Criterios:** `GET /compatible-models` retorna todos los pseudo-modelos con estado (safe/warning/blocked) y razón; `GET /tools-compatibility` detalla nivel requerido vs disponible; ambas son deterministas; no modifican el estado de la conversación.

---

## 20. Árboles de decisión deterministas

### 20.1 Árbol principal por petición

```mermaid
flowchart TD
    A[Petición entrante] --> B{¿Conversación nueva?}
    B -->|Sí| C[Crear conversación\nFixar pseudo-modelo y modelo físico]
    B -->|No| D[Cargar estado desde PostgreSQL]
    D --> E{¿Snapshot activo?}
    E -->|Sí| F[Contexto = snapshot + turnos post-snapshot]
    E -->|No| G[Contexto = historial completo]
    F --> H
    G --> H{¿Cambió pseudo-modelo?}
    H -->|No| I
    H -->|Sí| J[validate_switch]
    J -->|safe| I
    J -->|warning| K[Continuar con warning en metadata]
    J -->|blocked| L[HTTP 409 con remediation]
    K --> I
    I[Detectar capabilities del input] --> M{¿has_parallel_tools?}
    M -->|Sí| N[Filtrar pool: solo parallel_tools: true]
    M -->|No| O
    N --> O{¿Input > umbral?}
    O -->|Sí + pre_compaction| P[PreCompactor]
    O -->|Sí sin pre_compaction| Q[HTTP 400 INPUT_EXCEEDS_THRESHOLD]
    O -->|No| R
    P --> R{¿Contexto > trigger_pct?}
    R -->|Sí + continuous_compaction| S[ContinuousCompactor]
    R -->|No| T
    S --> T{¿router_llm?}
    T -->|Sí| U[Evaluar complejidad → metadata.suggestion]
    T -->|No| V
    U --> V[Enviar a LiteLLM]
    V -->|Éxito| W[Normalizar respuesta + guardar en DB]
    V -->|503/429| X[Fallback dentro del pseudo-modelo]
    X -->|Disponible| V
    X -->|Sin fallback| Y[HTTP 503 ALL_MODELS_FAILED]
    W --> Z[Retornar respuesta + proxy_metadata]
```

### 20.2 Árbol de compatibilidad de tools

```mermaid
flowchart TD
    A{¿has_parallel_tools?} -->|No| B[Continuar con todos los modelos]
    A -->|Sí| C[Filtrar pool: solo parallel_tools: true]
    C --> D{¿Pool vacío?}
    D -->|Sí| E[⚠️ Warning: parallel no garantizado\nUsar pool completo]
    D -->|No| F[Usar sub-pool parallel]
    E --> G[Continuar]
    F --> G
    B --> G
```

### 20.3 Fallbacks por pseudo-modelo

```
pensamiento-profundo-caro:
  1. glm-5.1  →[503]→  2. deepseek-v4-pro  →[503]→  ERROR
  Con tools: ambos elegibles (DeepSeek con strict y parallel)
  Con parallel tools: solo deepseek-v4-pro

tareas-avanzadas:
  1. deepseek-v4-pro  →[503]→  2. deepseek-v4-flash  →[503]→  3. minimax-m2.5  →[503]→  ERROR
  Con tools: todos elegibles
  Con parallel tools: deepseek-v4-pro y deepseek-v4-flash

normal:
  1. qwen3-max  →[503]→  2. deepseek-v4-flash  →[503]→  ERROR
  Con tools: ambos elegibles
  Con parallel tools: solo deepseek-v4-flash

deep-flash:
  1. glm-4.5-flash  →[503]→  2. openai/gpt-oss-20b  →[503]→  3. deepseek-v4-flash  →[503]→  ERROR
  Con tools: todos elegibles (distintos niveles de confiabilidad)
  Con parallel tools: solo deepseek-v4-flash

flash-lowcost:
  1. glm-4.5-flash  →[503]→  2. qwen3.5-plus  →[503]→  3. ollama/llama3.2 (local)  →[503]→  ERROR
  Con tools: todos elegibles (distintos niveles de confiabilidad)
  Con parallel tools: ninguno → bloqueo o normalización requerida

avanzada-vision:
  1. gemini-3.5-flash  →[503]→  2. meta-llama/llama-4-scout-17b-16e-instruct  →[503]→  ERROR
  Con tools: ambos elegibles

flash-vision:
  1. gemini-3.5-flash  →[503]→  2. ollama/llava (local)  →[503]→  ERROR
  Con tools: todos elegibles

compactador:
  Selecciona por context_window que cubra el historial dado:
  gemini-3.5-flash (1M)  →  claude-haiku-4-5-20251001 (200K)  →  glm-4.5-flash (128K)  →  ERROR
```

> **Un fallback entre pseudo-modelos diferentes requiere siempre intervención explícita del usuario.** El proxy nunca escala ni desescala entre categorías de forma autónoma.

---

_El valor de este proxy es su predictibilidad._
_Cada decisión tiene una razón clara y registrada._
_El usuario sabe exactamente qué pasó y por qué — siempre._
