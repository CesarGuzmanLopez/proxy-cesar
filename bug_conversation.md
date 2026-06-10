# Bug: Conversaciones largas con PDF + imagen fallan (tool_calls sin texto)

## Resumen

En conversaciones que contienen PDFs extraídos, imágenes y múltiples turnos de interacción, el modelo `mimo-v2.5` responde únicamente con `tool_calls` (web_search) sin generar texto (`content_len=0`, `accumulated_len=0`). opencode recibe las tool_calls pero **no envía un follow-up request** con los resultados de las herramientas.

En conversaciones **nuevas y simples**, el flujo completo funciona correctamente: modelo llama tool → opencode ejecuta → envía resultados → modelo genera texto.

## Servidor y conexión

| Ítem | Detalle |
|------|---------|
| Servidor | `personal` (hostname: `my-vps`) |
| SSH | `ssh personal` |
| Proxy path | `/opt/proxy-cesar/proxy/` |
| Logs | `/var/log/proxy-cesar/proxy.log` |
| Reinicio | `ssh personal 'systemctl restart proxy-cesar'` |
| Git deploy | `ssh personal 'cd /opt/proxy-cesar && git pull origin main && systemctl restart proxy-cesar'` |
| Valkey | Puerto 6380 (iniciado manualmente con `redis-server --port 6380 --daemonize yes`) |
| Conexión Valkey | `valkey://localhost:6379` (configurado desde .env) |

## Archivos relacionados

| Archivo | Rol |
|---------|-----|
| `proxy/src/api/chat_streaming.py` | SSE streaming, ensamblado de tool_calls, logs de chunks |
| `proxy/src/api/chat_stream_persistence.py` | `_extract_tokens_from_chunks` — ensamblado de respuesta final |
| `proxy/src/service/chat_fallback.py` | `call_with_fallback` — lógica de reintentos y ban de modelos |
| `proxy/src/service/tool_detector.py` | Delegación de contenido PDF/imagen/audio, extracción de texto |
| `proxy/src/service/compatibility.py` | `validate_physical_model_content` — decide si delegar |
| `proxy/src/service/chat_service.py` | Orchestración: content delegation + build de mensajes |
| `proxy/src/adapters/cache/valkey_affinity.py` | Setup de Valkey, adapter de afinidad |
| `proxy/src/main.py` | Startup, setup de Valkey como hard requirement |
| `proxy/pseudo_models.yaml` | Config de modelos físicos |

## Problemas identificados y soluciones aplicadas

### ✅ 1. Valkey no estaba disponible (CAUSA RAÍZ #1)

**Síntoma:** `replace_base64_with_blob_refs()` retornaba mensajes sin cambios cuando `valkey=None`, dejando `type: file` en los mensajes. Xiaomi (mimo-v2.5) rechazaba con `Param Incorrect`.

**Solución:** 
- `setup_valkey` ahora conecta realmente a Valkey (antes era no-op)
- Valkey es hard requirement: el proxy no arranca si Valkey no responde
- `replace_base64_with_blob_refs` lanza error si valkey=None

### ✅ 2. `finish_reason` hardcodeado a "stop" (BLOG #2)

**Síntoma:** En la respuesta ensamblada, `finish_reason` siempre era `"stop"` aunque el modelo devolviera `"tool_calls"`. opencode recibía `finish_reason="stop"` y no ejecutaba las herramientas.

**Solución:** `_extract_tokens_from_chunks()` ahora extrae el `finish_reason` real del `last_chunk` en lugar de hardcodearlo.

**Archivo:** `proxy/src/api/chat_stream_persistence.py:452`

### ✅ 3. Texto extraído de PDF repetido 138KB en cada request (BLOG #3)

**Síntoma:** opencode reenvía el archivo original en cada request. El proxy re-extráía el PDF completo (138KB) cada vez, saturando el contexto.

**Solución:** Shorten en cache hit — cuando la descripción ya está en Valkey y es >5K chars, se devuelve una referencia corta (`[File previously extracted — 138KB stored in cache.]`).

**Archivo:** `proxy/src/service/tool_detector.py:1269-1275`

### ✅ 4. Content delegation sin Valkey fallaba (BLOG #4)

**Síntoma:** Cuando Valkey no estaba disponible, `replace_base64_with_blob_refs` retornaba mensajes sin cambios. Esto causaba que `type: file` llegara a Xiaomi.

**Solución:** Ahora requiere Valkey obligatoriamente. Si no hay Valkey, lanza `ConnectionError`.

**Archivo:** `proxy/src/service/tool_detector.py:1171-1187`

### ✅ 5. Content delegation para imágenes sin extracción (BLOG #5)

**Síntoma:** Cuando opencode envía un archivo sin `data:` URL (ej: blob URL), el archivo no se procesaba y pasaba como `type: file` raw.

**Solución:** Los archivos sin data: URL se convierten a texto placeholder.

**Archivo:** `proxy/src/service/tool_detector.py:886-900`

## Bugs NO resueltos / En investigación

### 🔴 Bug principal: tool_calls no generan follow-up en conversaciones largas

**Comportamiento observado:**

```
Conversación nueva (funciona ✅):
  Request 1: "que dia empieza el mundial"
    → model calls web_search
    → finish=tool_calls → opencode EJECUTA ✅
    → Request 2 (follow-up con tool results) → model responde con texto ✅

Conversación larga con PDF + imagen (falla ❌):
  Request: "En terminos generales que dinero recibe"
    → model calls web_search x2
    → finish=tool_calls → opencode NO envía follow-up ❌
    → No hay más requests en los logs
```

**Lo que sabemos:**
- El SSE stream envía correctamente las tool_calls (confirmado por `stream_sse_chunk` logs: `idx=0|name=web_search`, `idx=1|name=web_search`)
- El `finish_reason` es `"tool_calls"` (corregido)
- La conversación tiene 9 mensajes con un system prompt enorme (~9K tokens)
- Incluye resultados de web_search de turns anteriores en el system prompt
- Incluye imágenes (enviadas directamente, delegation=False)
- opencode NO envía follow-up request (no hay `chat_request_received` posterior)

**Hipótesis:**
1. opencode podría tener un bug procesando SSE con tool_calls cuando hay content delegado/imágenes
2. El tamaño del system prompt podría estar causando que opencode ignore las tool_calls
3. opencode podría esperar un formato diferente en la última chunk SSE

**Próximos pasos:**
1. [ ] Verificar que la última chunk SSE tiene `finish_reason="tool_calls"` y delta vacío
2. [ ] Comparar SSE output exacto entre conversación que funciona y la que falla
3. [ ] Verificar si opencode hace el follow-up request o no

## Logs de diagnóstico disponibles

```
stream_sse_chunk      → Muestra cada chunk SSE con tool_calls (index, id, name)
stream_tool_calls_assembled  → Tool_calls ensamblados al final del stream
stream_response_assembled    → Respuesta final con content_len, finish_reason, tool_calls
```

## Notas adicionales

- ValeKey corre en puerto 6380, pero la URL configurada apunta a 6379 (ambos responden)
- Los logs se limpian con: `ssh personal 'truncate -s 0 /var/log/proxy-cesar/proxy.log'`
- El proxy se despliega automáticamente via GitHub Actions para `main`
- En personal hay que hacer git pull manual después del push
- Los `.bak` logs rotados se acumulan en `/var/log/proxy-cesar/`

## Commits relacionados

```
962895e debug: log detailed tool_call chunk structure (index, id, name)
41f021c debug: log exact SSE chunk finish_reason and tool_calls
5515e18 fix: preserve finish_reason from model response instead of hardcoding 'stop'
e7a6efc fix: shorten repeated extractions for all media types (files, audio, images)
9db102c fix: shorten repeated file extractions to prevent context overflow
4a38921 feat: make Valkey a hard requirement - fail fast if unavailable
c47c136 fix: convert file parts to text even without Valkey
8ffe9f5 fix: flatten blob content to string for providers that reject list content
```

## Reproducir el bug

1. Tener una conversación larga con PDF extraído + imagen + múltiples tool_calls
2. Enviar un mensaje de texto simple ("En terminos generales que dinero recibe")
3. Observar que el modelo responde con tool_calls (web_search) y finish_reason="tool_calls"
4. opencode NO envía follow-up request con resultados
5. El usuario ve una respuesta vacía (solo "Procesando contenido...")
