# Diagramas del Proxy LLM Multi-Modelo Determinista

## 1. Diagrama de Arquitectura de Componentes

```mermaid
graph TB
    subgraph Clientes["Clientes"]
        A1[OpenCode]
        A2[Continue]
        A3[LibreChat]
        A4[Aider]
        A5[curl / HTTP]
    end

    subgraph Proxy["Proxy César :9110"]
        direction TB
        
        subgraph Middleware["Middleware (CORS→Auth→RateLimit→KeyVault)"]
            direction LR
            M1["CORS"] --> M2["Auth<br/>Bearer Token"] --> M3["RateLimit<br/>Valkey fixed-window"] --> M4["KeyVault<br/>Detecta secrets<br/>Reinyecta en respuesta"]
        end

        subgraph API["API Layer"]
            R1["POST /v1/chat/completions"]
            R2["GET /v1/models"]
            R3["GET /health"]
            R4["GET /conversations/{id}"]
            R5["GET /conversations/{id}/compatible-models"]
            R6["GET /conversations/{id}/tools-compatibility"]
            R7["POST /conversations/{id}/normalize-tools"]
            R8["POST /conversations/{id}/compact"]
            R9["GET /conversations/{id}/audit-log"]
        end

        subgraph Service["Service Layer"]
            S1[Model Resolver]
            S2[Capability Detector]
            S3[Compatibility Validator]
            S4[Tool Filter]
            S5[Tool Detector<br/>Delegación imagen→tool]
            S6[Compactor<br/>POST /compact]
            S7[Router LLM Suggester]
        end

        subgraph Adapters["Adapters Layer"]
            A7[LiteLLM Client]
            A8[Provider Cache]
            A9[Valkey Affinity]
            A10[Local Discovery]
        end

        subgraph Config["Configuration"]
            C1[pseudo_models.yaml]
            C2[.env / Settings]
        end
    end

    subgraph KeyClaw["KeyClaw :8877"]
        KC["MITM Proxy<br/>Filtra API keys<br/>del tráfico saliente"]
    end

    subgraph Proveedores["LLM Providers"]
        P1[DeepSeek]
        P2[Groq]
        P3[OpenRouter]
        P4[Pruna]
        P5[Ollama]
        P6[LM Studio]
    end

    subgraph Almacenamiento["Storage"]
        D1[(SQLite/PostgreSQL<br/>Conversaciones)]
        D2[(Valkey<br/>Caché + Rate Limit)]
    end

    Clientes --> M1
    M3 --> API
    
    API --> Service
    Service --> Adapters
    Service --> Config
    Adapters --> KeyClaw
    KeyClaw --> Proveedores
    Adapters --> Almacenamiento
```

## 1b. Pipeline de Middleware

```mermaid
sequenceDiagram
    participant C as Cliente
    participant CO as CORS
    participant AU as Auth
    participant RL as Rate Limit
    participant KV as KeyVault
    participant API as Handler
    participant VK as Valkey
    
    C->>CO: Request
    CO->>AU: CORS ok
    AU->>RL: Auth ok
    RL->>KV: Rate ok
    
    rect rgb(240,255,240)
        Note over KV,VK: KeyVault — secret vault
        KV->>KV: detecta API keys en mensajes
        KV->>VK: guarda keyvault:conv:hash = valor
        KV->>KV: reemplaza por [KEYVAULT:hash]
        KV->>KV: inyecta system prompt
    end
    
    KV->>API: body sanitizado
    API-->>KV: response
    
    rect rgb(255,255,240)
        Note over KV,VK: Reinyección (streaming + no-streaming)
        KV->>VK: lookup keyvault:conv:hash
        KV->>KV: reemplaza [KEYVAULT:hash] → valor real
    end
    
    KV-->>RL: response
    RL-->>AU: response + headers
    AU-->>CO: response
    CO-->>C: response
```

## 1c. KeyVault — Flujo de Secrets

```mermaid
flowchart LR
    subgraph S1["1. Request"]
        R1["Usuario envía keys"] --> R2["Detecta patrones:<br/>sk-..., ghp_..., AKIA..."]
        R2 --> R3["Hash SHA256 → 8 chars"]
        R3 --> R4["Valkey: keyvault:conv:hash = valor"]
        R3 --> R5["Mensaje: reemplaza key → [KEYVAULT:hash]"]
        R5 --> R6["System prompt inyectado:<br/>explica uso de placeholders"]
    end

    subgraph S2["2. LLM"]
        L1["Recibe texto sanitizado<br/>(nunca ve keys reales)"]
        L2["Responde con placeholders:<br/>'usa [KEYVAULT:a1b2c3d4]'"]
    end

    subgraph S3["3. Response"]
        P1["Escanea respuesta"] --> P2["Encuentra [KEYVAULT:hash]"]
        P2 --> P3["Valkey lookup → valor real"]
        P3 --> P4["Reinyecta: el cliente ve la key real<br/>(soporta streaming y no-streaming)"]
    end

    S1 --> S2 --> S3
```

---

## 2. Diagrama de Flujo del Sistema (Chat Completions)

```mermaid
flowchart TB
    Start(["POST /v1/chat/completions"]) --> Auth{¿Auth habilitada?}
    Auth -->|No token| AuthFail[401 Unauthorized]
    Auth -->|Token válido| Rate{¿Rate limit excedido?}
    Rate -->|Sí| RateFail[429 Too Many Requests]
    Rate -->|No| ModelResolve[1. Resolver nombre del modelo]
    
    ModelResolve --> Normalize[nomalize_model_name]
    Normalize --> IsPseudo{¿Es pseudo-modelo<br/>conocido?}
    IsPseudo -->|Sí| Steps[Continuar flujo normal]
    IsPseudo -->|No| Passthrough[Passthrough directo<br/>modelo local/desconocido]
    
    Passthrough --> CapDetect
    Steps --> CapDetect[2. Detectar capacidades del turno]
    CapDetect --> ContentValid[3. Validar contenido entrante]
    
    ContentValid --> HasImages{¿Tiene imágenes?}
    HasImages -->|Sí, sin visión| BlobVaultImg[Blob Vault:<br/>guardar + describir + referencia]
    HasImages -->|Sí, con visión| InlineCmd
    HasImages -->|No| InlineCmd
    
    BlobVaultImg --> InlineCmd
    
    ContentValid --> HasAudio{¿Tiene audio?}
    HasAudio -->|Sí, sin audio model| BlobVaultAud[Blob Vault:<br/>guardar + describir + referencia]
    HasAudio -->|Sí, con audio model| InlineCmd
    
    BlobVaultAud --> InlineCmd
    
    ContentValid --> HasVideo{¿Tiene video?}
    HasVideo -->|Sí| BlobVaultVid[Blob Vault:<br/>guardar + metadata]
    
    BlobVaultVid --> InlineCmd
    
    InlineCmd[4. Verificar comando inline] --> IsCmd{¿status o help?}
    IsCmd -->|Sí| CmdHandled{skip_llm?}
    CmdHandled -->|Sí| CmdResponse[Responder con resultado<br/>del comando]
    CmdHandled -->|No| SessionLoad
    IsCmd -->|No| SessionLoad
    
    SessionLoad[5. Cargar sesión y conversación] --> Affinity[6. Resolver afinidad<br/>modelo físico]
    Affinity --> SwitchValid[7. Validar cambio de<br/>pseudo-modelo]
    
    SwitchValid --> SwitchResult{Resultado}
    SwitchResult -->|BLOCKED| SwitchBlock[409 PSEUDO_MODEL_INCOMPATIBLE<br/>Usa modelo con visión vía tools]
    SwitchResult -->|WARNING| StoreWarn[Guardar warning]
    SwitchResult -->|SAFE| ToolFilter
    
    StoreWarn --> ToolFilter[8. Filtrar modelos por<br/>parallel_tools]
    ToolFilter --> PhysicalResolve[9. Resolver modelo físico<br/>afinidad → primero elegible]
    
    PhysicalResolve --> IsNew{¿Es nueva<br/>conversación?}
    IsNew -->|Sí| CreateConv[Crear en DB]
    IsNew -->|No| ThresholdCheck
    CreateConv --> ThresholdCheck
    
    ThresholdCheck[10. Verificar umbral<br/>de tokens] --> IsOver{¿input > threshold?}
    IsOver -->|Sí| Threshold400[400 INPUT_EXCEEDS_THRESHOLD<br/>Usa POST /compact]
    IsOver -->|No| RouterLLM
    
    RouterLLM[11. Router LLM<br/>sugerencia de downgrade] --> IsRouter{¿Router LLM<br/>habilitado?}
    IsRouter -->|Sí| EvalComplexity[Evaluar complejidad<br/>vía LLM evaluador]
    IsRouter -->|No| LLMCall
    
    EvalComplexity --> IsSuggest{¿Sugiere downgrade?}
    IsSuggest -->|Sí| StoreSuggestion[Guardar sugerencia<br/>nunca bloquea]
    IsSuggest -->|No| LLMCall
    StoreSuggestion --> LLMCall
    
    LLMCall[12. Llamar modelo físico<br/>con fallback] --> ForEach[Por cada modelo físico<br/>en orden de prioridad]
    
    ForEach --> CheckCtx{¿input ><br/>context_window?}
    CheckCtx -->|Sí| SkipModel[Saltar - CONTEXT_SKIPPED]
    CheckCtx -->|No| CacheOpt[Optimizar caché<br/>si Anthropic/DeepSeek]
    
    CacheOpt --> LiteLLMCall[call_litellm]
    LiteLLMCall --> IsError{¿Error retryable?}
    IsError -->|Sí| Fallback[Registrar fallback<br/>→ siguiente modelo]
    IsError -->|No| Success[Respuesta exitosa]
    
    Fallback --> NextModel{¿Quedan modelos?}
    NextModel -->|Sí| CheckCtx
    NextModel -->|No| AllFailed{¿Todos saltados<br/>por contexto?}
    AllFailed -->|Sí| Context413[413 CONTEXT_TOO_LARGE]
    AllFailed -->|No| Models503[503 ALL_MODELS_FAILED]
    
    Success --> SaveTurn[13. Guardar turno en DB]
    SaveTurn --> AccumCaps[Acumular capacidades]
    AccumCaps --> Commit[Commit DB]
    Commit --> Response["14. Responder al cliente<br/>+ proxy_metadata"]
    
    Response --> StreamQ{¿Streaming?}
    StreamQ -->|Sí| SSE[Streaming SSE<br/>+ metadata final]
    StreamQ -->|No| JSON[Respuesta JSON<br/>completa]
```

---

## 3. Diagrama de Estados del Pseudo-Modelo y Modelo Físico

```mermaid
stateDiagram-v2
    [*] --> PseudoModelo
    
    state PseudoModelo {
        [*] --> Seleccionado: usuario elige modelo
        Seleccionado --> Resolviendo: normalize_model_name
        Resolviendo --> Conocido: alias o pseudo-modelo
        Resolviendo --> Passthrough: modelo local/desconocido
        Conocido --> ValidandoEntrada: validate_incoming_content
        Passthrough --> ValidandoEntrada
        ValidandoEntrada --> Compatible: contenido soportado
        ValidandoEntrada --> Delegado: imagen delegada a tool
        ValidandoEntrada --> Incompatible: 400 error explícito
        Delegado --> CargandoConversacion
        Compatible --> CargandoConversacion
        CargandoConversacion --> NuevaConversacion: no existe
        CargandoConversacion --> SwitchModelo: cambió pseudo-modelo
        CargandoConversacion --> MismoModelo: mismo pseudo-modelo
        NuevaConversacion --> EvaluandoUmbral
        MismoModelo --> EvaluandoUmbral
    }
    
    state SwitchModelo {
        [*] --> ValidandoSwitch: validate_switch
        ValidandoSwitch --> Bloqueado: 409 conflict
        ValidandoSwitch --> Advertencia: warning
        ValidandoSwitch --> Permitido: safe
        Advertencia --> EvaluandoUmbral
        Permitido --> EvaluandoUmbral
    }
    
    state EvaluandoUmbral {
        [*] --> BajoUmbral: input ≤ threshold
        [*] --> ExcedeUmbral: input > threshold
        ExcedeUmbral --> Umbral400: 400 error
    }
    
    BajoUmbral --> RouterLLM
    
    state RouterLLM {
        [*] --> Evaluando: evaluate_complexity
        Evaluando --> LLM: evaluador barato
        LLM --> Sugiriendo: respuesta parseable
        Sugiriendo --> EsDowngrade: is_downgrade?
        EsDowngrade --> SugerenciaGuardada: yes + suggest_on_downgrade_only
        EsDowngrade --> SinSugerencia: no
    }
    
    SugerenciaGuardada --> LlamandoModelo
    SinSugerencia --> LlamandoModelo
    
    state LlamandoModelo {
        [*] --> Intento1: primer modelo físico
        Intento1 --> VerificandoContexto: check context_window
        VerificandoContexto --> ContextoExcede: input > window
        VerificandoContexto --> CacheOptimizando
        ContextoExcede --> IntentoN: skip → siguiente modelo
        CacheOptimizando --> LlamandoLiteLLM
        LlamandoLiteLLM --> Exitoso: response
        LlamandoLiteLLM --> ErrorRetryable: ServiceUnavailable / RateLimit
        ErrorRetryable --> IntentoN: fallback → siguiente modelo
        IntentoN --> VerificandoContexto: más modelos?
        IntentoN --> TodosFallaron: sin más modelos
        TodosFallaron --> Contexto413: todos saltados por contexto
        TodosFallaron --> Modelos503: errores mixtos
    }
    
    Exitoso --> GuardandoTurno
    GuardandoTurno --> CommitDB
    CommitDB --> Respondiendo
    
    Respondiendo --> [*]
```

---

## 4. Diagrama de Secuencia (Flujo Completo de una Solicitud)

```mermaid
sequenceDiagram
    participant C as Cliente
    participant RL as Rate Limiter
    participant KV as KeyVault
    participant Auth as Auth
    participant API as Chat API
    participant CS as Chat Service
    participant MR as Model Resolver
    participant CD as Capability Detector
    participant CV as Compatibility Validator
    participant TD as Tool Detector
    participant TF as Tool Filter
    participant RTR as Router LLM
    participant LC as LiteLLM Client
    participant DB as DB
    participant VK as Valkey
    
    C->>KV: POST /v1/chat/completions
    KV->>KV: detecta secrets, guarda en Valkey
    KV->>KV: reemplaza por placeholders
    
    KV->>RL: body sanitizado
    RL->>VK: INCR rate limit key
    VK-->>RL: count
    RL-->>C: 429 if exceeded
    
    RL->>Auth: verify Bearer token
    Auth-->>RL: 401 if invalid
    
    RL->>API: request passes middleware
    
    API->>CS: process_chat_request(model, messages)
    
    CS->>MR: normalize_model_name(model)
    MR-->>CS: pseudo_model_name
    
    CS->>CD: detect_turn_capabilities(messages)
    CD-->>CS: turn_caps (images, audio, tools...)
    
    CS->>CV: validate_incoming_content(turn_caps, pseudo, tools)
    alt imágenes + sin visión + tools compatibles
        CV-->>CS: delegation signal
        CS->>TD: delegate_images_to_tool(messages)
        TD-->>CS: messages modificados
    else contenido no soportado
        CV-->>CS: 400 error explícito
    end
    
    CS->>DB: get or create conversation
    CS->>VK: get affinity
    VK-->>CS: pinned physical_model
    
    CS->>CV: validate_switch(from, to, caps)
    CV-->>CS: blocked/warning/safe
    
    CS->>TF: get_eligible_models(physical_models, caps)
    TF-->>CS: filtered models
    
    CS->>CS: check context alert level
    alt context ≥ 100%
        CS-->>API: 400 CONTEXT_UNUSABLE
        API-->>C: Error response
    end
    
    alt router_llm enabled
        CS->>RTR: evaluate_complexity(last_msg, suggester)
        RTR-->>CS: suggestion or None
    end
    
    loop for each physical model
        CS->>LC: call_litellm(model, messages, stream)
        alt context_window exceeded
            LC-->>CS: skip
        else success
            LC-->>CS: response
        else retryable error
            LC-->>CS: ServiceUnavailable/RateLimit
        end
    end
    
    alt all models failed
        CS-->>API: 413 or 503
        API-->>C: Error response
    end
    
    CS->>DB: save ConversationTurn
    CS->>DB: update conversation
    CS->>VK: set affinity
    
    alt streaming
        CS-->>API: StreamingResponse
        KV->>KV: reinyectar secrets en chunks SSE
        API-->>C: SSE chunks + metadata
    else non-streaming
        CS-->>API: ChatResult + proxy_metadata
        KV->>KV: reinyectar secrets en JSON
        API-->>C: JSON response
    end
```

---

## 5. Diagrama de Casos de Uso del Usuario

```mermaid
flowchart LR
    subgraph Usuarios["Actores"]
        U1["Usuario<br/>(Dev con LLM)"]
        U2["Cliente LLM<br/>(OpenCode/Continue)"]
        U3["Administrador"]
    end
    
    subgraph CasosDeUso["Casos de Uso"]
        UC1["Enviar mensaje<br/>de chat"]
        UC2["Elegir pseudo-modelo<br/>por nombre o alias"]
        UC3["Cambiar de<br/>pseudo-modelo"]
        UC4["Enviar imágenes<br/>con texto"]
        UC5["Enviar herramientas<br/>(function calling)"]
        UC6["Usar comando inline<br/>status / help"]
        UC7["Compactar historial<br/>POST /compact"]
        UC8["Ver modelos<br/>disponibles"]
        UC9["Ver estado<br/>de salud"]
        UC10["Ver log de<br/>eventos"]
    end

    subgraph Sistema["Sistema Proxy"]
        P1["Resolver modelo<br/>+ alias"]
        P2["Validar contenido<br/>+ capacidades"]
        P2b["Delegar imagen<br/>a tool"]
        P3["Router LLM<br/>sugerir downgrade"]
        P4["Fallback entre<br/>modelos físicos"]
        P5["Cache provider<br/>optimización"]
        P6["Afinnidad de<br/>conversación"]
        P7["Rate limiting"]
        P8["Auditoría de<br/>eventos"]
    end

    U1 -.->|usa| U2
    U2 --> UC1
    U2 --> UC2
    U2 --> UC3
    U2 --> UC4
    U2 --> UC5
    U2 --> UC8
    U1 --> UC6
    U1 --> UC7
    U1 --> UC9
    U1 --> UC10
    U3 --> UC9
    
    UC1 --> P1
    UC2 --> P1
    UC3 --> P2
    UC4 --> P2
    UC4 --> P2b
    UC5 --> P2
    UC6 --> P1
    UC7 --> P1
    UC10 --> P8
    
    P1 --> P6
    P2 --> P2b
    P2 --> P5
    P2 --> P6
    P3 --> P2
    P4 --> P6
    P7 -.->|aplica a| UC1
```

---

## 6. Diagrama de Eventos del Sistema

```mermaid
timeline
    title Ciclo de Vida de una Solicitud al Proxy
    section 1. Recepción
        HTTP Request : CORS + Auth + Rate Limit + KeyVault
        Chat API     : POST /v1/chat/completions
    section 2. Resolución
        Model Resolver  : normalize_model_name → pseudo-modelo
        Capability Detector  : detect_turn_capabilities
        Content Validator  : validate_incoming_content
        Tool Detector  : delegate_images_to_tool (si aplica)
    section 3. Sesión
        Conversation Loader  : get_or_create + affinity
        Switch Validator  : validate_switch
        Tool Filter  : get_eligible_models
    section 4. Router
        Router LLM  : evaluate_complexity
        LLM evaluador  : sugerir downgrade
    section 5. Ejecución
        LiteLLM Call  : attempt modelo 1
        Fallback Loop  : attempt modelo 2, 3...
        Success  : response obtenida
    section 6. Persistencia
        DB Commit  : turno + métricas
        Valkey  : actualizar afinidad
        KeyVault  : reinyectar secrets
        Stream  : SSE chunks + metadata final
```

---

## 7. Diagrama de Decisión de Estrategia de Cache

```mermaid
flowchart LR
    Provider{¿Qué proveedor?} -->|DeepSeek| AutoDC[Automatic prefix caching<br/>Cache hit: $0.0028/M]
    Provider -->|Groq| AutoG[Automatic prefix caching<br/>Cache hit: 50% descuento]
    Provider -->|Anthropic| ANC[cache_control breakpoints<br/>system + history prefix]
    Provider -->|OpenRouter| AutoOR[Delega al upstream<br/>depende del modelo final]
    Provider -->|Ollama| NoCache[Sin caché de proveedor]
```

---

## 8. Diagrama de Relación Pseudo-Modelos a Modelos Físicos

```mermaid
graph LR
    subgraph Pseudo_Models["Pseudo-Modelos"]
        PP["pensamiento-profundo-caro<br/>120K ctx"]
        TA["tareas-avanzadas<br/>200K ctx"]
        V["vision<br/>120K ctx · visión"]
        N["normal<br/>500K ctx"]
        NG["normal-gratis<br/>200K ctx"]
        MF["massive-fast<br/>131K ctx"]
        FL["flash-lowcost<br/>128K ctx"]
        AU["audio<br/>131K ctx · whisper"]
        IM["imagen<br/>text-to-image"]
        CM["compactador<br/>20M ctx · operación"]
    end

    subgraph Physical_Models["Modelos Físicos"]
        DS_P["deepseek-v4-pro<br/>$1.74/$3.48 · tools_strict"]
        DS_F["deepseek-v4-flash<br/>$0.14/$0.28 · fast"]
        GM_35["gemini-3.5-flash<br/>$0.0015/$0.009 · visión+audio"]
        GM_31["gemini-3.1-flash-lite<br/>$0.00025/$0.0015 · visión+audio"]
        L4S["llama-4-scout-17b<br/>$0.11/$0.34 · visión"]
        GPT20["gpt-oss-20b<br/>$0.075/$0.30 · 1000 t/s"]
        QW32["qwen3-32b<br/>$0.29/$0.59 · 662 t/s"]
        NT["nemotron-3-super:free<br/>gratis · 1M ctx"]
        WV3["whisper-large-v3<br/>$0.111/h · audio"]
        WV3T["whisper-large-v3-turbo<br/>$0.04/h · audio"]
        PI["p-image<br/>$0.002/img · imagen"]
    end

    PP --> DS_P
    PP --> GM_35
    TA --> DS_P
    TA --> DS_F
    V --> L4S
    N --> DS_F
    N --> GM_31
    NG --> NT
    NG --> QW32
    MF --> GPT20
    FL --> GM_31
    AU --> WV3
    AU --> WV3T
    IM --> PI
    CM --> GPT20
    CM --> DS_F
```

---

## 9. Diagrama de Flujo del Router LLM

```mermaid
flowchart TD
    Start["evaluate_complexity(messages, suggester_model)"] --> Safety{Último mensaje<br/>tiene texto?}
    Safety -->|No, solo imágenes| ReturnNone[return None]
    Safety -->|Sí| LLMEval[Evaluación LLM<br/>call_litellm con prompt<br/>temperature=0.0]
    
    LLMEval --> LLMResult{Respuesta JSON<br/>parseable?}
    LLMResult -->|Sí| ParseSug[Extraer suggested_model]
    LLMResult -->|No| ReturnNone
    
    ParseSug --> IsAllowed{Está en<br/>ALLOWED_SUGGESTIONS?}
    IsAllowed -->|Sí| CheckDowngrade
    IsAllowed -->|No| ReturnNone
    
    CheckDowngrade --> SuggestOnly{suggest_on_downgrade_only?}
    SuggestOnly -->|Sí| IsDowngrade["Es downgrade?<br/>is_downgrade(suggested, current)"]
    SuggestOnly -->|No| ReturnSuggestion[return suggested_model]
    
    IsDowngrade -->|Sí| ReturnSuggestion
    IsDowngrade -->|No| ReturnNone
    
    ReturnNone --> End[return None<br/>Request sigue sin cambios]
    ReturnSuggestion --> End2[return suggestion<br/>Nunca bloquea — solo sugiere]
```

---

## 10. Mapa de Errores del Sistema

```mermaid
graph TD
    subgraph Errores["Errores del Sistema"]
        E1["400 PARALLEL_TOOLS_NOT_SUPPORTED<br/>Tools paralelas sin soporte (único error 400 de contenido)"]
        E6["400 INPUT_EXCEEDS_THRESHOLD<br/>Input supera límite del pseudo-modelo<br/>Usa POST /compact"]
        E7["400 CONTEXT_UNUSABLE<br/>Contexto al 100%"]
        E8["401 UNAUTHORIZED<br/>Token inválido o faltante"]
        E9["409 PSEUDO_MODEL_INCOMPATIBLE<br/>Switch bloqueado por capacidades"]
        E10["413 CONTEXT_TOO_LARGE_FOR_ALL_MODELS<br/>Todos los modelos físicos excedidos"]
        E11["429 RATE_LIMIT_EXCEEDED<br/>Límite de tasa excedido"]
        E12["502 PROXY_ERROR<br/>Error interno del proxy"]
        E13["503 ALL_MODELS_FAILED<br/>Todos los modelos físicos fallaron"]
        E14["500 INTERNAL_ERROR<br/>Error interno del proxy"]
    end
    
    style E1 fill:#ffcccc
    style E2 fill:#ffcccc
    style E3 fill:#ffcccc
    style E4 fill:#ffcccc
    style E5 fill:#ffcccc
    style E6 fill:#ffcccc
    style E7 fill:#ffcccc
    style E8 fill:#ffcccc
    style E9 fill:#ffcccc
    style E10 fill:#ffcccc
    style E11 fill:#ffcccc
    style E12 fill:#ffcccc
    style E13 fill:#ffcccc
    style E14 fill:#ffcccc
```

---

## Leyenda de Colores

| Color | Significado |
|-------|------------|
| Azul | Componente del sistema |
| Verde | Flujo exitoso |
| Rojo | Error / Bloqueo |
| Amarillo | Advertencia / Decisión |
