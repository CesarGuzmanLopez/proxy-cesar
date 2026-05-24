# Proxy Determinista Multi-Modelo — Especificación Maestra

> **Paquetería 100% libre.** MIT, BSD, Apache 2.0.
> **El proxy no decide. Valida, informa y ejecuta lo que el usuario ordena.**
> **Cuando algo no puede continuar, lo dice claramente y ofrece opciones.**
> **Cuando algo puede continuar, lo hace sin fricción.**
> **Código, variables, comentarios, commits y documentación en inglés.**
> **100% tipado y determinista.** La única no-determinista es la respuesta del modelo.
> **Sin prefijos, sin inyección de variables, sin manipulación de strings.**
> Nombres de modelos exactamente como están en el archivo de configuración.
> Si algo no cumple, error claro — nunca silencio.

---

## 1. Filosofía

El proxy resuelve la brecha entre lo que el usuario sabe (su tarea) y lo que el ecosistema LLM exige (proveedores, formatos, capacidades, costos). **Abstrae proveedores, no decisiones.**

```
1. VALIDAR compatibilidad antes de ejecutar
2. MANTENER afinidad de modelo físico durante la conversación
3. APLICAR fallbacks dentro del mismo pseudo-modelo cuando un modelo falla
4. BLOQUEAR cambios destructivos con error descriptivo y opciones de remediación
5. INFORMAR en cada respuesta: modelo físico real, uso de contexto, advertencias, ahorros
```

El proxy **no razona**, **no resume sin consentimiento**, **no decide qué modelo es mejor**. Eso es trabajo del usuario. Toda acción tiene una razón precisa y registrada.

---

## 2. Pseudo-modelos

Un pseudo-modelo es una **intención de uso** que el usuario selecciona. El proxy lo resuelve al modelo físico real del proveedor. El usuario selecciona intenciones, no proveedores.

Cada pseudo-modelo define: modelos físicos (en orden de preferencia), capacidades, umbrales, reglas de compatibilidad y comportamientos opcionales.

### 2.1 `pensamiento-profundo-caro`

**Propósito:** Razonamiento de máximo nivel. Bugs imposibles, arquitectura compleja, decisiones con muchas variables. El usuario lo selecciona activamente y con precaución. Usar para tareas puntuales difíciles, no como modelo de cabecera.

**Modelos físicos:** `glm-5.1` → `deepseek-v4-pro`

**Capacidades:** Tools (todos), parallel tools (solo deepseek-v4-pro), sin visión.

**Reglas:**

- Input >32K tokens → pre-compactado con `deep-flash` antes de enviar
- Contexto >70% → compactación continua automática (snapshot de turnos viejos)
- Router LLM activo: si la tarea es simple, sugiere bajar a `tareas-avanzadas` (nunca impone)
- Al cambiar a modelo sin visión: describir imágenes automáticamente

### 2.2 `tareas-avanzadas`

**Propósito:** Desarrollo profundo y sostenido. Features largas, debugging serio donde el programador sabe que el problema es difícil, definición de arquitectura, refactorizaciones grandes. Largo plazo, muchos tokens.

**Modelos físicos:** `deepseek-v4-pro` → `deepseek-v4-flash` (thinking) → `minimax-m2.5`

**Capacidades:** Tools (todos), parallel tools (DeepSeek, no MiniMax), sin visión.

**Reglas:**

- Contexto >75% → compactación continua automática
- Input >64K tokens → rechazado (sin pre-compaction)
- Sin router LLM (quien lo selecciona sabe que lo necesita)

### 2.3 `avanzada-vision`

**Propósito:** Análisis visual de alto nivel. Diseño de interfaces, OCR complejo, interpretación de diagramas, maquetas. Especialmente útil para frontend a partir de imágenes de referencia.

**Modelos físicos:** `gemini-3.5-flash` → `meta-llama/llama-4-scout-17b-16e-instruct`

**Capacidades:** Visión (todos), tools (todos), sin parallel tools.

**Reglas:**

- La conversación queda marcada con `has_images: true` desde el primer turno con imagen
- No se puede migrar a pseudo-modelos sin visión sin degradación explícita o auto-describe
- Sin compactación continua (las imágenes no se compactan semánticamente)

### 2.4 `normal`

**Propósito:** Punto de entrada recomendado. Coding extenso, trabajo agéntico, manipulación de documentos, investigaciones largas, desarrollo de features de tamaño medio. Balance costo/capacidad óptimo.

**Modelos físicos:** `qwen3-max` → `deepseek-v4-flash`

**Capacidades:** Tools (todos), parallel tools (solo deepseek-v4-flash), sin visión.

**Reglas:**

- Contexto >80% → compactación continua automática
- Input >96K tokens → rechazado
- Compatible hacia arriba y hacia abajo sin restricciones (salvo multimedia)

### 2.5 `deep-flash`

**Propósito:** Velocidad y costo mínimo para tareas masivas y simples. Investigaciones larguísimas, traducciones masivas, tareas monótonas de lectura y generación. Muy rápidos pero no para razonamiento complejo. Soporta 10-15 sub-agentes concurrentes.

**Modelos físicos:** `glm-4.5-flash` → `openai/gpt-oss-20b` → `deepseek-v4-flash`

**Capacidades:** Tools (todos, distintos niveles de confiabilidad), parallel tools (solo deepseek-v4-flash), sin visión.

**Reglas:**

- Sin compactación (su ventana de 128K tokens es suficiente)
- Sin router LLM
- Máximo de sub-agentes concurrentes: 15

### 2.6 `flash-lowcost`

**Propósito:** Sub-agentes baratos. Tareas específicas y bien definidas: clasificación, extracción, parsing, validación de formato. Consistentes en tareas bien explicadas. También funciona como evaluador del router LLM.

**Modelos físicos:** `glm-4.5-flash` → `qwen3.5-plus` → `ollama/llama3.2`

**Capacidades:** Tools (todos), sin parallel tools, sin visión.

**Reglas:**

- Sin compactación
- Sin router LLM
- Si el usuario lo selecciona para trabajo interactivo: advertencia, no bloqueo

### 2.7 `flash-vision`

**Propósito:** Visión rápida y barata. OCR ligero, screenshots, análisis visual simple. Compatible hacia arriba con `avanzada-vision` (upgrade seguro).

**Modelos físicos:** `gemini-3.5-flash` → `ollama/llava`

**Capacidades:** Visión (todos), tools (todos), sin parallel tools.

**Reglas:**

- Compatible hacia arriba con `avanzada-vision` (upgrade seguro)
- Conversaciones marcadas con `has_images: true`
- Auto-describe imágenes al migrar a modelo sin visión

### 2.8 `compactador` _(operación, no conversacional)_

**Propósito:** Operación de compactación explícita. Invocada por el usuario (`POST /compact`) o cuando el historial supera todos los umbrales (`CONTEXT_UNUSABLE`). Genera un snapshot estructurado del historial.

**Modelos físicos:** `gemini-3.5-flash` (1M contexto) → `claude-haiku-4-5-20251001` (200K) → `glm-4.5-flash` (128K)

**Qué hace:** Lee el historial completo → extrae decisiones, código clave, estado del problema, puntos pendientes → genera snapshot Markdown → el historial original permanece intacto.

---

## 3. Reglas fundamentales

### 3.1 Afinidad de caché

El proxy **fija el modelo físico en el primer turno** de cada conversación y lo mantiene hasta cambio explícito del usuario. Esto maximiza cache hits en el proveedor (90%+ de ahorro en tokens de entrada).

Si el modelo falla (503/429), se usa el siguiente modelo del mismo pseudo-modelo. Se notifica al usuario. El caché previo se abandona.

### 3.2 Compatibilidad al cambiar de pseudo-modelo

- **Seguro:** El destino soporta todas las capacidades del historial → cambio sin fricción
- **Advertencia:** El destino tiene menor capacidad pero el historial no se rompe → continúa con warning
- **Bloqueado:** El destino no puede interpretar contenido ya presente → error 409 con opciones de remediación

**Bloqueos típicos:**

- Hay imágenes en el historial y el destino no tiene visión ni `auto_describe`
- Hay parallel tools en el historial y el destino no tiene modelos con `parallel_tools: true`
- El contexto acumulado supera la ventana del destino

**Remediaciones disponibles:**

- `POST /degrade-images` — describir imágenes con modelo visual
- `POST /normalize-tools` — serializar parallel calls a secuenciales
- `POST /compact` — compactar historial antes de migrar

### 3.3 Tools y Function Calling

Todos los modelos en el pool deben soportar el formato OpenAI de function calling. LiteLLM traduce transparentemente al formato nativo de cada proveedor. El proxy no escribe código de traducción — solo verifica compatibilidad.

Si la conversación usa parallel tools, el pool se reduce a modelos con `parallel_tools: true`.

### 3.4 Multimedia

| Tipo     | v1                                                                                            |
| -------- | --------------------------------------------------------------------------------------------- |
| Imágenes | Soporte completo: detección, compatibilidad, auto-describe, degradación manual                |
| Audio    | Detectado y rechazado con error `AUDIO_NOT_SUPPORTED`                                         |
| PDF      | Detectado: con modelo de visión → tratado como imagen; sin visión → error `PDF_NOT_SUPPORTED` |
| Video    | Rechazado con error `VIDEO_NOT_SUPPORTED`                                                     |

### 3.5 Compactación

- **Pre-compactación:** Si el input es muy largo para un modelo caro, un modelo barato lo resume primero. Solo si el pseudo-modelo lo tiene configurado.
- **Compactación continua:** En modelos caros, cuando el contexto acumulado supera cierto umbral, los turnos viejos se compactan en un snapshot estructurado. Solo si está configurado.
- **Compactación explícita:** El usuario invoca `POST /compact` para reducir el historial. El snapshot preserva decisiones, código clave, estado del problema y pendientes.

## 4. Stack tecnológico

| Componente             | Paquete                  | Licencia                |
| ---------------------- | ------------------------ | ----------------------- |
| API                    | FastAPI                  | MIT                     |
| Router multi-proveedor | LiteLLM                  | MIT                     |
| Persistencia           | PostgreSQL + asyncpg     | PostgreSQL / Apache 2.0 |
| Caché / Afinidad       | Valkey                   | BSD                     |
| Tareas asíncronas      | Celery                   | BSD                     |
| Validación             | Pydantic v2              | MIT                     |
| ORM                    | SQLAlchemy 2.0 + Alembic | MIT                     |
| HTTP                   | httpx                    | BSD                     |
| Tokens                 | tiktoken                 | MIT                     |
| Config                 | PyYAML                   | MIT                     |
| Imágenes               | Pillow                   | HPND                    |
| HTTPS                  | Caddy                    | Apache 2.0              |

---

## 5. Plan de implementación

**13 semanas, 8 sprints:**

1. **MVP** (2 sem) — pseudo-modelos, afinidad, streaming, fallback
2. **Capabilities** (2 sem) — detección de multimedia/tools, matriz de compatibilidad
3. **Tools** (2 sem) — formato canónico, normalización, LiteLLM verificación
4. **Compactación** (2 sem) — pre-compaction, continuous compaction
5. **Imágenes + Router** (2 sem) — auto-describe, sugerencias de downgrade
6. **Compactación explícita** (1 sem) — `/compact`, alertas de contexto
7. **Caché + OpenCode** (1 sem) — optimización de caché, integración
8. **Despliegue** (1 sem) — auth, CORS, HTTPS, rate limiting, métricas, docs

---

_El valor de este proxy es su predictibilidad. Cada decisión tiene una razón clara y registrada. El usuario sabe exactamente qué pasó y por qué — siempre._
