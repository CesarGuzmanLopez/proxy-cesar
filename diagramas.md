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

    subgraph Proxy["Proxy César"]
        direction TB
        
        subgraph Middleware["Middleware Layer"]
            M1[CORS]
            M2[Auth - Bearer Token]
            M3[Rate Limiter - Valkey]
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
            R10["POST /conversations/{id}/degrade-images"]
        end

        subgraph Service["Service Layer"]
            S1[Model Resolver]
            S2[Capability Detector]
            S3[Compatibility Validator]
            S4[Tool Filter]
            S5[Image Describer]
            S6[Compactor]
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

    subgraph Proveedores["LLM Providers"]
        P1[DeepSeek]
        P2[Groq]
        P3[OpenRouter]
        P4[Pruna - Imagen]
        P5[Ollama - Local]
        P6[LM Studio - Local]
    end

    subgraph Almacenamiento["Storage"]
        D1[(SQLite/PostgreSQL<br/>Conversaciones)]
        D2[(Valkey<br/>Caché + Rate Limit)]
    end

    Clientes --> M1
    M1 --> M2
    M2 --> M3
    M3 --> API
    
    API --> Service
    Service --> Adapters
    Service --> Config
    Adapters --> Proveedores
    Adapters --> Almacenamiento
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
    IsPseudo -->|No| Passthrough[build_passthrough_pseudo_model<br/>Crear pseudo-modelo mínimo]
    
    Passthrough --> CapDetect
    Steps --> CapDetect[2. Detectar capacidades del turno]
    CapDetect --> ContentValid[3. Validar contenido entrante]
    
    ContentValid --> HasImages{¿Tiene imágenes?}
    HasImages -->|Sí, sin visión| Images400[400 IMAGES_NOT_SUPPORTED]
    HasImages -->|Sí, con visión| InlineCmd
    HasImages -->|No| InlineCmd
    
    ContentValid --> HasAudio{¿Tiene audio?}
    HasAudio -->|Sí| Audio400[400 AUDIO_NOT_SUPPORTED]
    
    ContentValid --> HasVideo{¿Tiene video?}
    HasVideo -->|Sí| Video400[400 VIDEO_NOT_SUPPORTED]
    
    ContentValid --> HasPDF{¿Tiene PDF sin visión?}
    HasPDF -->|Sí| PDF400[400 PDF_NOT_SUPPORTED]
    
    InlineCmd[4. Verificar comando inline] --> IsCmd{¿degrade, status<br/>o help?}
    IsCmd -->|Sí| CmdHandled{skip_llm?}
    CmdHandled -->|Sí| CmdResponse[Responder con resultado<br/>del comando]
    CmdHandled -->|No| SessionLoad
    IsCmd -->|No| SessionLoad
    
    SessionLoad[5. Cargar sesión y conversación] --> Affinity[6. Resolver afinidad<br/>modelo físico]
    Affinity --> SwitchValid[7. Validar cambio de<br/>pseudo-modelo]
    
    SwitchValid --> SwitchResult{Resultado}
    SwitchResult -->|BLOCKED| SwitchBlock[409 PSEUDO_MODEL_INCOMPATIBLE]
    SwitchResult -->|WARNING| StoreWarn[Guardar warning]
    SwitchResult -->|SAFE| ToolFilter
    
    StoreWarn --> ToolFilter[8. Filtrar modelos por<br/>parallel_tools]
    ToolFilter --> PhysicalResolve[9. Resolver modelo físico<br/>afinidad → primero elegible]
    
    PhysicalResolve --> IsNew{¿Es nueva<br/>conversación?}
    IsNew -->|Sí| CreateConv[Crear en DB]
    IsNew -->|No| IsSwitch{¿Cambió pseudo-modelo?}
    
    IsSwitch -->|Sí| AutoDescribe[10. Auto-describir imágenes]
    IsSwitch -->|No| ThresholdCheck
    CreateConv --> ThresholdCheck
    
    AutoDescribe --> TargetVision{¿Destino tiene<br/>visión?}
    TargetVision -->|Sí| PassThrough[Pasar imágenes<br/>sin describir]
    TargetVision -->|No, auto_describe| DescribeImages[Describir imágenes<br/>con modelo visión]
    TargetVision -->|No, block| ImageBlocked[409 BLOCKED]
    
    DescribeImages --> ThresholdCheck
    PassThrough --> ThresholdCheck
    
    ThresholdCheck[11. Verificar umbral<br/>de tokens] --> IsOver{¿input > threshold?}
    IsOver -->|Sí| Threshold400[400 INPUT_EXCEEDS_THRESHOLD]
    IsOver -->|No| PreCompact[12. Pre-compactación]
    
    PreCompact --> PreCheck{¿Habilitada y<br/>input > umbral?}
    PreCheck -->|Sí| PreCompactExec[Resumir último<br/>mensaje de usuario]
    PreCheck -->|No| ExternalDetect
    
    PreCompactExec --> ExternalDetect[13. Detectar compactación<br/>externa del cliente]
    
    ExternalDetect --> IsExternal{¿Cliente compactó<br/>historial?}
    IsExternal -->|Sí| HandleExternal[Crear snapshot external<br/>resetear total_tokens]
    IsExternal -->|No| ContinousCheck
    
    HandleExternal --> ContinousCheck[14. Compactación continua]
    
    ContinousCheck --> IsTrigger{¿total_tokens ><br/>ctx_window × trigger_pct?}
    IsTrigger -->|Sí| ContinousExec[Compactar turns antiguos<br/>→ ConversaciónSnapshot]
    IsTrigger -->|No| SnapshotCheck
    
    ContinousExec --> SnapshotCheck[15. Ensamblar contexto<br/>desde snapshot]
    
    SnapshotCheck --> HasSnapshot{¿Hay snapshot<br/>activo?}
    HasSnapshot -->|Sí| AssembleContext[Armar: snapshot<br/>+ últimos mensajes]
    HasSnapshot -->|No| ReEstimate[Usar mensajes originales]
    
    AssembleContext --> ReEstimate
    ReEstimate --> AlertCheck[16. Alerta de contexto]
    
    AlertCheck --> UsagePct{¿Porcentaje de<br/>contexto usado?}
    UsagePct -->|≥100%| Context400[400 CONTEXT_UNUSABLE]
    UsagePct -->|80-99%| HighAlert[Alta: Compact recommended]
    UsagePct -->|60-80%| ModerateAlert[Moderada: advertencia]
    UsagePct -->|<60%| RouterLLM
    
    HighAlert --> RouterLLM[17. Router LLM<br/>sugerencia de downgrade]
    ModerateAlert --> RouterLLM
    Context400 -.-> End

    RouterLLM --> IsRouter{¿Router LLM<br/>habilitado?}
    IsRouter -->|Sí| EvalComplexity[Evaluar complejidad<br/>vía LLM evaluador]
    IsRouter -->|No| LLMCall
    
    EvalComplexity --> IsSuggest{¿Sugiere downgrade?}
    IsSuggest -->|Sí| StoreSuggestion[Guardar sugerencia]
    IsSuggest -->|No| LLMCall
    StoreSuggestion --> LLMCall
    
    LLMCall[18. Llamar modelo físico<br/>con fallback] --> ForEach[Por cada modelo físico<br/>en orden de prioridad]
    
    ForEach --> CheckCtx{¿input ><br/>context_window?}
    CheckCtx -->|Sí| SkipModel[Saltar - CONTEXT_SKIPPED]
    CheckCtx -->|No| CacheOpt[Optimizar caché<br/>si Anthropic]
    
    CacheOpt --> LiteLLMCall[call_litellm]
    LiteLLMCall --> IsError{¿Error retryable?}
    IsError -->|Sí| Fallback[Registrar fallback<br/>→ siguiente modelo]
    IsError -->|No| Success[✅ Respuesta exitosa]
    
    Fallback --> NextModel{¿Quedan modelos?}
    NextModel -->|Sí| CheckCtx
    NextModel -->|No| AllFailed{¿Todos saltados<br/>por contexto?}
    AllFailed -->|Sí| Context413[413 CONTEXT_TOO_LARGE]
    AllFailed -->|No| Models503[503 ALL_MODELS_FAILED]
    
    Success --> SaveTurn[19. Guardar turno en DB]
    SaveTurn --> AccumCaps[Acumular capacidades]
    AccumCaps --> Commit[Commit DB]
    Commit --> Response["20. Responder al cliente<br/>+ proxy_metadata"]
    
    Response --> StreamQ{¿Streaming?}
    StreamQ -->|Sí| SSE[Streaming SSE<br/>+ metadata final]
    StreamQ -->|No| JSON[Respuesta JSON<br/>completa]

    subgraph Leyenda
        L1["🟢 Flujo normal"]
        L2["🔴 Error / Bloqueo"]
        L3["🟡 Advertencia"]
    end
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
        ValidandoEntrada --> Incompatible: 400 error
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
        Advertencia --> AutoDescribe: images + auto_describe
        Permitido --> AutoDescribe: images + auto_describe
        AutoDescribe --> Describiendo: modelo visión describe imágenes
        Describiendo --> TurnoDegradacion: degradation_event
        TurnoDegradacion --> EvaluandoUmbral
        AutoDescribe --> EvaluandoUmbral: sin imágenes
    }
    
    state EvaluandoUmbral {
        [*] --> BajoUmbral: input ≤ threshold
        [*] --> ExcedeUmbral: input > threshold
        ExcedeUmbral --> PreCompactando: pre_compaction enabled
        ExcedeUmbral --> Umbral400: 400 error
        PreCompactando --> BajoUmbral: compactado exitoso
        PreCompactando --> BajoUmbral: fallback sin compactar
    }
    
    state Compactacion {
        [*] --> DetectandoExterna
        DetectandoExterna --> ExternaDetectada: cliente ya compactó
        DetectandoExterna --> ContinuaCheck: sin compactación externa
        ExternaDetectada --> SnapshotExterno
        ContinuaCheck --> BajoTrigger: tokens < trigger_pct
        ContinuaCheck --> Compactando: tokens ≥ trigger_pct
        Compactando --> SnapshotInterno: snapshot creado
        SnapshotInterno --> EnsamblandoContexto
        SnapshotExterno --> EnsamblandoContexto
        BajoTrigger --> EnsamblandoContexto
        EnsamblandoContexto --> ContextoListo
    }
    
    state LlamandoModelo {
        [*] --> Intento1: primer modelo físico
        Intento1 --> VerificandoContexto: check context_window
        VerificandoContexto --> ContextoExcede: input > window
        VerificandoContexto --> CacheOptimizando
        ContextoExcede --> IntentoN: skip → siguiente modelo
        CacheOptimizando --> LlamandoLiteLLM
        LlamandoLiteLLM --> Exitoso: ✅ response
        LlamandoLiteLLM --> ErrorRetryable: ServiceUnavailable / RateLimit
        ErrorRetryable --> IntentoN: fallback → siguiente modelo
        IntentoN --> VerificandoContexto: más modelos?
        IntentoN --> TodosFallaron: sin más modelos
        TodosFallaron --> Contexto413: todos saltados por contexto
        TodosFallaron --> Modelos503: errores mixtos
    }
    
    BajoUmbral --> Compactacion
    ContextoListo --> RouterLLM
    
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
    participant Auth as Auth
    participant API as Chat API
    participant CS as Chat Service
    participant MR as Model Resolver
    participant CD as Capability Detector
    participant CV as Compatibility Validator
    participant TF as Tool Filter
    participant ID as Image Describer
    participant COMP as Compactor
    participant RTR as Router LLM
    participant LC as LiteLLM Client
    participant DB as DB
    participant VK as Valkey
    
    C->>RL: POST /v1/chat/completions
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
    
    CS->>CV: validate_incoming_content(turn_caps, pseudo)
    CV-->>CS: 400 if unsupported content
    
    CS->>DB: get or create conversation
    CS->>VK: get affinity
    VK-->>CS: pinned physical_model
    
    CS->>CV: validate_switch(from, to, caps)
    CV-->>CS: blocked/warning/safe
    
    CS->>TF: get_eligible_models(physical_models, caps)
    TF-->>CS: filtered models
    
    alt is switch and auto_describe
        CS->>ID: auto_describe_images(history, vision_model)
        ID->>LC: describe_image(url, vision_model)
        LC-->>ID: text description
        ID-->>CS: described messages + metadata
        CS->>DB: save degradation_event turn
    end
    
    CS->>COMP: _apply_compaction(messages, pseudo, config)
    
    alt pre_compaction enabled
        COMP->>LC: pre_compact_input(user_message)
        LC-->>COMP: summary
    end
    
    alt external compaction detected
        COMP->>DB: create external snapshot
    end
    
    alt continuous compaction triggered
        COMP->>LC: compact old turns
        LC-->>COMP: snapshot
        COMP->>DB: save snapshot
    end
    
    COMP-->>CS: processed messages
    
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
            Note over CS,LC: Break loop on success
        else retryable error
            LC-->>CS: ServiceUnavailable/RateLimit
            Note over CS,LC: Continue to next model
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
        API-->>C: SSE chunks + metadata
    else non-streaming
        CS-->>API: ChatResult + proxy_metadata
        API-->>C: JSON response
    end
```

---

## 5. Diagrama de Casos de Uso del Usuario

```mermaid
flowchart LR
    subgraph Usuarios["Actores"]
        U1["👤 Usuario<br/>(Dev con LLM)"]
        U2["🤖 Cliente LLM<br/>(OpenCode/Continue)"]
        U3["🔧 Administrador"]
    end
    
    subgraph CasosDeUso["Casos de Uso"]
        UC1["Enviar mensaje<br/>de chat"]
        UC2["Elegir pseudo-modelo<br/>por nombre o alias"]
        UC3["Cambiar de<br/>pseudo-modelo"]
        UC4["Enviar imágenes<br/>con texto"]
        UC5["Enviar herramientas<br/>(function calling)"]
        UC6["Enviar herramientas<br/>paralelas"]
        UC7["Streaming de<br/>respuesta"]
        UC8["Ver modelos<br/>disponibles"]
        UC9["Usar comando inline<br/>degrade / status / help"]
        UC10["Compactar historial<br/>explícitamente"]
        UC11["Degradar imágenes<br/>manualmente"]
        UC12["Ver estado<br/>de salud"]
        UC13["Ver log de<br/>eventos"]
        UC14["Normalizar<br/>tool calls"]
    end

    subgraph Sistema["Sistema Proxy"]
        P1["Resolver modelo<br/>+ alias"]
        P2["Validar contenido<br/>+ capacidades"]
        P3["Auto-describir<br/>imágenes"]
        P4["Compactación<br/>pre/continua/explicita"]
        P5["Router LLM<br/>sugerir downgrade"]
        P6["Fallback entre<br/>modelos físicos"]
        P7["Cache provider<br/>optimización"]
        P8["Afinnidad de<br/>conversación"]
        P9["Rate limiting"]
        P10["Auditoría de<br/>eventos"]
    end

    U1 -.->|usa| U2
    U2 --> UC1 & UC2 & UC3 & UC4 & UC5 & UC6 & UC7 & UC8
    U1 --> UC9 & UC10 & UC11 & UC12 & UC13 & UC14
    U3 --> UC12
    
    UC1 --> P1
    UC2 --> P1
    UC3 --> P2 & P3
    UC4 --> P2 & P3
    UC5 --> P2
    UC6 --> P2
    UC9 --> P3
    UC10 --> P4
    UC11 --> P3
    UC13 --> P10
    
    P1 --> P8
    P2 --> P7
    P2 --> P8
    P5 --> P2
    P6 --> P8
    P9 -.->|aplica a| UC1
```

---

## 6. Diagrama de Eventos del Sistema

```mermaid
timeline
    title Ciclo de Vida de una Solicitud al Proxy
    section 1. Recepción
        HTTP Request : CORS + Auth + Rate Limit
        Chat API     : POST /v1/chat/completions
    section 2. Resolución
        Model Resolver  : normalize_model_name → pseudo-modelo
        Capability Detector  : detect_turn_capabilities
        Content Validator  : validate_incoming_content
    section 3. Sesión
        Conversation Loader  : get_or_create + affinity
        Switch Validator  : validate_switch
        Tool Filter  : get_eligible_models
    section 4. Pre-procesamiento
        Image Describer  : auto_describe_images (si switch)
        Pre-Compactor  : pre_compact_input (si umbral)
        External Detector  : detect_external_compaction
        Continuous Compactor  : continuous_compact (si trigger)
        Context Assembler  : assemble_context (desde snapshot)
    section 5. Router
        Router LLM  : evaluate_complexity
         LLM evaluador  : sugerir downgrade
    section 6. Ejecución
        LiteLLM Call  : attempt modelo 1
        Fallback Loop  : attempt modelo 2, 3...
        Success  : response obtenida
    section 7. Persistencia
        DB Commit  : turno + métricas
        Valkey  : actualizar afinidad
        Stream  : SSE chunks + metadata final
```

---

## 7. Diagrama de Decisión de Estrategia de Cache

```mermaid
flowchart LR
    Provider{¿Qué proveedor?} -->|DeepSeek| AutoDC[Automatic prefix caching<br/>Cache hit: $0.0028/M]
    Provider -->|Groq| AutoG[Automatic prefix caching<br/>Cache hit: 50% descuento]
    Provider -->|OpenRouter| AutoOR[Delega al upstream<br/>depende del modelo final]
    Provider -->|Ollama| NoCache[Sin caché de proveedor]
    Provider -->|Pruna| NoCache2[Sin caché - generación<br/>de imágenes]
```

---

## 8. Diagrama de Relación Pseudo-Modelos a Modelos Físicos

```mermaid
graph LR
    subgraph Pseudo_Models["Pseudo-Modelos"]
        PP["pensamiento-profundo-caro<br/>120K ctx · auto_describe"]
        TA["tareas-avanzadas<br/>200K ctx · block"]
        V["vision<br/>120K ctx · visión"]
        N["normal<br/>500K ctx · block"]
        NG["normal-gratis<br/>200K ctx · auto_describe"]
        MF["massive-fast<br/>131K ctx · block"]
        FL["flash-lowcost<br/>128K ctx · block"]
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

## 9. Diagrama de Secuencia de Compactación

```mermaid
sequenceDiagram
    participant U as Usuario/Cliente
    participant Proxy as Proxy
    participant COMP as Compactor
    participant LC as LiteLLM
    participant DB as Database
    
    rect rgb(200, 220, 240)
        Note over U,DB: Compactación Previa (automática)
        U->>Proxy: Mensaje muy largo
        Proxy->>COMP: pre_compact_input()
        COMP->>LC: Resumir último mensaje
        LC-->>COMP: Texto resumido
        COMP-->>Proxy: Mensaje reemplazado
    end
    
    rect rgb(220, 240, 200)
        Note over U,DB: Compactación Continua (automática en umbral)
        U->>Proxy: Múltiples mensajes
        Proxy->>COMP: continuous_compact()
        COMP->>DB: Cargar todos los turns
        COMP->>COMP: turns_to_compact + preserve_recent
        
        alt Historial > contexto
            COMP->>COMP: _chunk_history() → overlap chunks
            loop por cada chunk
                COMP->>LC: Compactar chunk
                LC-->>COMP: Summary parcial
            end
        else
            COMP->>LC: Compactar historial completo
            LC-->>COMP: Snapshot estructurado
        end
        
        COMP->>DB: Crear ConversationSnapshot
        COMP-->>Proxy: tokens_after + metadata
        Proxy-->>U: Respuesta con snapshot metadata
    end
    
    rect rgb(240, 220, 200)
        Note over U,DB: Compactación Explícita (POST /compact)
        U->>Proxy: POST /conversations/{id}/compact
        
        alt total_tokens > 500K + arq disponible
            Proxy-->>U: 202 Processing (async)
            Proxy->>DB: Disparar arq worker
            DB-->>Proxy: Compactación en background
        else
            Proxy->>COMP: compact_conversation()
            COMP->>DB: Crear snapshot placeholder
            COMP->>LC: Compactar historial
            LC-->>COMP: Snapshot
            COMP->>DB: Actualizar snapshot
            COMP-->>Proxy: tokens_reduced_pct + metadata
            Proxy-->>U: 200 OK
        end
    end
```

---

## 10. Diagrama de Flujo del Router LLM

```mermaid
flowchart TD
    Start["evaluate_complexity(messages, suggester_model)"] --> Safety{¿Último mensaje<br/>tiene texto?}
    Safety -->|No, solo imágenes| ReturnNone[return None]
    Safety -->|Sí| LLMEval[Evaluación LLM<br/>call_litellm con prompt<br/>temperature=0.0]
    
    LLMEval --> LLMResult{¿Respuesta JSON<br/>parseable?}
    LLMResult -->|Sí| ParseSug[Extraer suggested_model]
    LLMResult -->|No| ReturnNone
    
    ParseSug --> IsAllowed{¿Está en<br/>ALLOWED_SUGGESTIONS?}
    IsAllowed -->|Sí| CheckDowngrade
    IsAllowed -->|No| ReturnNone
    
    CheckDowngrade --> SuggestOnly{¿suggest_on_downgrade_only?}
    SuggestOnly -->|Sí| IsDowngrade{¿Es downgrade?<br/>is_downgrade(suggested, current)}
    SuggestOnly -->|No| ReturnSuggestion[return suggested_model]
    
    IsDowngrade -->|Sí| ReturnSuggestion
    IsDowngrade -->|No| ReturnNone
    
    ReturnNone --> End[return None<br/>Request sigue sin cambios]
    ReturnSuggestion --> End2[return suggestion<br/>Nunca bloquea — solo sugiere]
```

---

## 11. Mapa de Errores del Sistema

```mermaid
graph TD
    subgraph Errores["Errores del Sistema"]
        E1["400 IMAGES_NOT_SUPPORTED<br/>Imágenes sin modelo visión"]
        E2["400 AUDIO_NOT_SUPPORTED<br/>Audio no soportado en v1"]
        E3["400 PDF_NOT_SUPPORTED<br/>PDF sin modelo visión"]
        E4["400 VIDEO_NOT_SUPPORTED<br/>Video no soportado en v1"]
        E5["400 PARALLEL_TOOLS_NOT_SUPPORTED<br/>Tools paralelas sin soporte"]
        E6["400 INPUT_EXCEEDS_THRESHOLD<br/>Input supera límite del pseudo-modelo"]
        E7["400 CONTEXT_UNUSABLE<br/>Contexto al 100%"]
        E8["401 UNAUTHORIZED<br/>Token inválido o faltante"]
        E9["409 PSEUDO_MODEL_INCOMPATIBLE<br/>Switch bloqueado por capacidades"]
        E10["413 CONTEXT_TOO_LARGE_FOR_ALL_MODELS<br/>Todos los modelos físicos excedidos"]
        E11["429 RATE_LIMIT_EXCEEDED<br/>Límite de tasa excedido"]
        E12["502 COMPACTION_FAILED<br/>Compactación falló"]
        E13["502 DEGRADE_IMAGES_FAILED<br/>Degradación de imágenes falló"]
        E14["503 ALL_MODELS_FAILED<br/>Todos los modelos físicos fallaron"]
        E15["500 INTERNAL_ERROR<br/>Error interno del proxy"]
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
    style E15 fill:#ffcccc
```

---

## Leyenda de Colores

| Color | Significado |
|-------|------------|
| 🔵 Azul | Componente del sistema |
| 🟢 Verde | Flujo exitoso |
| 🔴 Rojo | Error / Bloqueo |
| 🟡 Amarillo | Advertencia / Decisión |
| ⚪ Blanco | Actor externo |
