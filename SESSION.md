# Sesión 2026-05-29 — Bug KeyVault + GPT-OSS: La Odisea del System Prompt

## Resumen

Esta sesión (29 de mayo de 2026, ~18 commits en 3 horas) documenta la
investigación y resolución de un bug crítico donde GPT-OSS 20B (vía Groq)
producía contenido **vacío** (`content=""`) al recibir ciertas combinaciones
de system prompt + placeholder KeyVault. Lo que parecía un bug de Groq resultó
ser una interacción sutil entre múltiples instrucciones del system prompt.
Groq **nunca fue el problema**.

---

## 1. El Problema

GPT-OSS 20B (Open Source en Groq) producía respuestas con `content=""` vacío
en ciertas requests. El síntoma:

- Sin KeyVault → funcionaba bien, output con contenido normal
- Con KeyVault (placeholder `[KEYVAULT:hash]` en el mensaje) → output vacío

El fallback a DeepSeek V4 Flash siempre funcionaba correctamente, pero
DeepSeek es más caro y lento (~30-60s).

## 2. La Pista Falsa: "Groq es el problema"

El primer instinto fue culpar a Groq. La comunidad dice que Groq "no soporta
system prompts largos" o "tiene problemas con tool calling". Hubo múltiples
intentos de "acomodar" el prompt para que Groq lo entendiera:

### Intentos fallidos (en orden cronológico)

| # | Cambio | Resultado |
|---|--------|-----------|
| 1 | Poner `"IF a tool is available, call it directly"` + `[KEYVAULT:hash]` | ❌ Content vacío |
| 2 | Cambiar a `"Use tools when needed"` | ❌ Content vacío |
| 3 | Quitar tool instruction, dejar solo `"[KEYVAULT:...] = masked content"` | ❌ Content vacío |
| 4 | Cambiar KeyVault prompt a `"plain text, NOT a tool call"` | ❌ Content vacío |
| 5 | Decir `"respond as if real value is there"` | ❌ Content vacío |
| 6 | **Eliminar system_prompt de Groq completamente** | ✅ Funciona, pero sin tool calling |
| 7 | Quitar KeyVault del system prompt, inyectarlo dinámicamente en mensaje | ✅ Funciona, con tool calling |

El paso **#6** nos hizo creer que "Groq no tolera ningún system prompt".
Pero el paso **#7** reveló la verdad: el problema no era Groq, era **la
interacción** entre la instrucción de tool calling Y el placeholder KeyVault
en el mismo prompt.

### El descubrimiento clave

Después de verificar **directamente contra la API de Groq** (sin proxy,
sin LiteLLM), se confirmó:

- System prompt SOLO con instrucción tool calling → ✅ funciona, llama tools
- System prompt SOLO con `[KEYVAULT:hash]` → ✅ funciona, output con contenido
- **AMBOS** en el mismo system prompt → ❌ Output vacío (`content=""`)

**Groq no era el problema.** Era la combinación de instrucciones lo que
confundía a GPT-OSS 20B específicamente. GPT-OSS es un modelo de 20B
parámetros — más pequeño, más sensible a instrucciones contradictorias o
ambiguas.

## 3. La Confusión: "System Prompt Duplicado"

Durante la investigación se generó otra confusión: **múltiples mensajes
system**. En la arquitectura del proxy:

1. El pseudo-modelo puede tener un `system_prompt` en YAML
2. El usuario puede enviar un mensaje `role: system`
3. KeyVault necesita inyectar `[KEYVAULT:...] = masked placeholder`

Si los 3 se convertían en mensajes system separados, algunos modelos
(GPT-OSS) se confundían. La solución fue **fusionar siempre** en un solo
mensaje system:

- `chat_fallback.py` (línea 192-201): el `phys.system_prompt` se fusiona con
  el primer mensaje system existente, no se agrega como nuevo
- `keyvault.py` (línea 442-451): la instrucción KeyVault se **apendea** al
  último mensaje system existente, no se inserta como nuevo
- `chat.py` (línea 147-151): para streaming, mismo approach — apendea al
  último system message

## 4. El KeyVault Streaming Buffer

Otro bug paralelo: el reemplazo de placeholders en **streaming**. Cuando
`[KEYVAULT:abc12345]` se dividía entre chunks (e.g., `"..."` + `[KEYVAULT:` +
`"abc12345]"`), el replacement fallaba porque el placeholder no era completo
en ningún chunk individual.

### Intentos de solución

1. **Cross-chunk buffer**: Acumular chunks que contienen `[` hasta que
   aparezca `]`, luego reemplazar el placeholder completo. ✅ Funciona
2. **Problema: content loss con reasoning**: GPT-OSS intercala chunks de
   `reasoning_content` y `content`. El buffer acumulaba también chunks de
   contenido real, causando **pérdida de contenido** cuando el placeholder
   se completaba en un chunk posterior mezclado con reasoning.
3. **Solución final**: Eliminar el cross-chunk buffer. Aceptar que en
   streaming, los placeholders fragmentados **no se reemplazan** (el usuario
   ve `[KEYVAULT:hash]`). En no-streaming siempre funciona correctamente.
4. **Limpieza entre fallbacks**: Agregar `ctx._keyvault_buf = ""` y
   `ctx._keyvault_pending = []` al resetear entre modelos de la cadena de
   fallback.

## 5. Lo Que Se Aprendió

### Sobre GPT-OSS y Groq

- **GPT-OSS 20B es sensible a la complejidad del system prompt.** Con
  instrucciones simples funciona perfectamente. Con instrucciones compuestas
  (tool calling + placeholder + formato) puede producir output vacío.
- **Groq no es el problema.** Es la interacción entre el modelo pequeño y
  múltiples instrucciones en el system prompt.
- **La comunidad se equivoca** cuando generaliza "Groq no soporta X". Hay que
  probar cada modelo individualmente.

### Sobre el diseño del proxy

- **Un solo system prompt.** Múltiples mensajes system confunden a modelos
  pequeños. Fusionar siempre.
- **KeyVault dinámico.** No poner la instrucción KeyVault en el system prompt
  estático del YAML. Inyectarla solo cuando hay secretos detectados. Ahorra
  tokens ~90% de las requests.
- **Streaming imperfecto.** El reemplazo de placeholders en streaming es
  inherentemente frágil con modelos que intercalan reasoning + content.
  Aceptar la limitación: non-streaming 100% correcto, streaming ~90%.
- **Model-agnostic.** Toda lógica hardcodeada por modelo (deepseek, o-series)
  se reemplazó por flags en YAML (`strip_reasoning`, `thinking`,
  `reasoning_effort`). El proxy no tiene ni un solo `if model == "deepseek"`.

### Sobre el proceso de debugging

- **Probar directamente contra la API.** El proxy añade capas de abstracción
  (LiteLLM, middleware, caché). Las pruebas directas contra Groq eliminaron
  variables y revelaron la causa raíz.
- **Un cambio a la vez.** Cada commit de esta sesión cambia UNA cosa. Cuando
  todo fallaba, poder revisar el historial mostró el patrón claro.
- **Documentar los experimentos.** El git log es la memoria del debugging.
  Cada intento fallido quedó registrado y evitable en el futuro.

## 6. La Solución Final

**Estado actual** (commit `fadee67`):

1. Todos los modelos Groq tienen system prompt con instrucciones de tool
   calling (las que la comunidad y el usuario verificaron empíricamente).
2. La instrucción KeyVault se inyecta DINÁMICAMENTE en el mensaje system
   existente, solo cuando se detecta un secreto en la request.
3. Si no hay secreto, KeyVault no toca el system prompt → GPT-OSS no se
   confunde.
4. Si hay secreto, KeyVault se apendea al último system message (nunca como
   mensaje separado).
5. En streaming, placeholders completos en un solo chunk se reemplazan.
   Fragmentados no — el usuario ve `[KEYVAULT:hash]`.

## 7. Archivos Clave Modificados

| Archivo | Cambio |
|---------|--------|
| `proxy/pseudo_models.yaml` | system_prompt en 4 modelos Groq (+40 líneas) |
| `proxy/src/middleware/keyvault.py` | _KEYVAULT_SYSTEM_PROMPT simplificado, merge en último system message, TTL 24h |
| `proxy/src/service/chat_fallback.py` | phys.system_prompt se fusiona con primer system message existente; flags YAML reemplazan hardcoding |
| `proxy/src/api/chat.py` | KeyVault streaming handler: detecta secretos, inyecta prompt dinámicamente |
| `proxy/src/api/chat_streaming.py` | Per-chunk JSON str.replace (sin buffer cross-chunk), limpieza entre fallbacks |

## 8. Commits de la Sesión (orden cronológico)

```
65fc8bf fix: KeyVault re-injection in streaming — cross-chunk placeholder buffer
82333e0 perf: keyvault stream buffer only when '[' detected
ffc44f6 fix: KeyVault TTL 1h→24h — must outlive provider cache
efc77ea fix: keyvault stream — check complete placeholder BEFORE buffer opt
d070728 debug: add chunk_delta to stream_reinject_ok log
91ad024 fix: keyvault stream also checks reasoning_content field
a725757 fix: keyvault stream — check reasoning_content + cross-chunk buffer
47fc9b4 fix: keyvault stream — defer chunks when '[' detected, flush on complete
103fbf1 fix: keyvault stream — replace chunk with '[' using real value, clear fragments
5c59100 debug: add keyvault_gen log inside generator
e9e99e8 fix: keyvault stream — replace per-field, not both content+reasoning
21f9a5d fix: track finish_reason in keyvault stream branch
af351d2 fix: keyvault stream — process ALL pending chunks, not just first
5564798 fix: keyvault stream — stop clearing after ']' closes placeholder
e50a0d3 fix: clear keyvault buffer between fallback chain models
219b0cc refactor: remove model-specific hardcoding — use YAML flags instead
394bc79 fix: KeyVault system prompt — model must use placeholder for storage too
a77ad17 fix: keyvault stream only clears field that contained '['
5282dd7 fix: merge KeyVault prompt into existing system message (not separate)
310406d fix: merge phys.system_prompt into first existing system message
4a0fbe0 fix: KeyVault prompt says [KEYVAULT:...] is plain text, NOT a tool call
ce3de95 fix: keyvault stream buffer actually clears ctx attributes (not local vars)
6bae48f fix: indentation in keyvault buffer cleanup
f766060 refactor: simplify KeyVault streaming — remove cross-chunk buffer, concise prompt
167c577 fix: KeyVault prompt — 'masked content, respond as if real value is there'
15eff7b fix: Groq prompt — 'Use tools when needed' instead of 'IF a tool is available, call it directly'
d71877c simplify: Groq + KeyVault prompts — remove tool instructions, keep only placeholder guidance
0c6c810 fix: remove KeyVault instructions from Groq system_prompts
fadee67 fix: restore Groq system_prompt for proper tool calling  ← ACTUAL
```

Total: **28 commits** en esta sesión, 18 directamente sobre KeyVault/GPT-OSS.

## 9. Próximos Pasos (No Implementados)

- Monitorear logs de `keyvault_handler_stream` para detectar falsos positivos
  (secretos detectados donde no los hay)
- Considerar reintentar streaming con KeyVault si el primer chunk del stream
  contiene un placeholder completo (caso más común)
- Evaluar si GPT-OSS 20B mejora con futuras versiones del modelo en Groq
