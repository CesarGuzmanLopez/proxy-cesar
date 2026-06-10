# Bug: Conversaciones con PDF/imágenes no ejecutan tool_calls

## Resumen

En conversaciones que contienen PDFs extraídos, imágenes o archivos adjuntos, el modelo responde correctamente con `tool_calls` (ej: `web_search`) pero **opencode no ejecuta las herramientas**. El usuario ve la respuesta como vacía o bloqueada.

**Causa raíz:** El proxy inyectaba un chunk SSE sintético (`stream_analysis_msg` — "Analizando contenido...") ANTES de los chunks del modelo. Este chunk falso con `"delta": {"content": "..."}` **rompía el ensamblado de tool_calls** en el cliente (opencode), que no podía parsear correctamente los deltas de tool_call después de recibir un chunk de contenido no-tool_call.

## Síntomas

- Solo afecta conversaciones con PDFs, imágenes o archivos extraídos
- Conversaciones nuevas y simples funcionan correctamente
- El modelo genera tool_calls válidos (confirmado en logs)
- `finish_reason=tool_calls` se envía correctamente
- opencode NO envía follow-up con resultados de herramientas
- El problema empeora con el tiempo (más archivos = más probabilidad de fallo)

## Causa raíz

### `stream_analysis_msg` — el chunk fantasma

En `_stream_response_generator()` (chat_streaming.py:642-661), cuando la conversación tiene contenido procesado (PDFs, imágenes), el proxy inyectaba un chunk SSE ANTES de empezar el streaming del modelo:

```python
# chat_streaming.py ~line 642
if not _analysis_message_sent and current_idx == 0:
    content_counts = _count_content_types(ctx.messages or [])
    has_content = ctx.images_described > 0 or any(content_counts.values())
    if has_content:
        analysis_msg = _build_analysis_message(content_counts, ctx.images_described)
        analysis_chunk = {
            "id": f"chatcmpl-{ctx.conversation_id[:12]}",
            "object": "chat.completion.chunk",
            "choices": [{"delta": {"content": analysis_msg}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(analysis_chunk)}\n\n"
```

Este chunk tiene `"delta": {"content": "Analizando 1 imagen, 1 PDF..."}` — es un chunk de **contenido de texto**, no un delta de tool_call. Al insertarse antes de los chunks reales del modelo, el cliente (opencode) recibe:

```
data: {"delta": {"content": "Analizando contenido..."}}    ← CHUNK FANTASMA
data: {"delta": {"reasoning_content": "..."}}              ← chunks reales del modelo
data: {"delta": {"tool_calls": [{"index": 0, ...}]}}       ← tool_call chunks
data: {"finish_reason": "tool_calls"}                       ← finish
```

El cliente intenta ensamblar los tool_calls pero el chunk inicial de contenido interfiere con el parser SSE.

### ¿Por qué solo con PDFs/imágenes?

`_count_content_types()` detecta imágenes (`image_url`) y archivos PDF en los mensajes. Si encuentra alguno, `has_content=True` y se inyecta el mensaje de análisis. En conversaciones sin archivos adjuntos, `has_content=False` y no hay inyección → el flujo funciona.

### Efectos colaterales

Este bug no solo rompía tool_calls. Cualquier flujo que dependa del orden correcto de chunks SSE podía verse afectado:
- Respuestas con `reasoning_content` (thinking) podían mostrarse incorrectamente
- Clientes que esperan un formato específico de chunk podían fallar
- El chunk falso agregaba latencia innecesaria al inicio del streaming

## Solución

**Deshabilitar permanentemente la inyección de `stream_analysis_msg`.**

El mensaje de análisis era puramente cosmético ("Analizando 1 imagen, 1 PDF...") y no aportaba valor funcional. Los clientes modernos (opencode, Continue) muestran sus propios indicadores de progreso.

```python
# chat_streaming.py ~line 642
# NOTE: Analysis message injection is intentionally disabled.
# Injecting synthetic SSE chunks BEFORE the model's stream corrupts
# tool_call assembly in clients like opencode.
if False and not _analysis_message_sent and current_idx == 0:
```

**Commit:** `e7c27ec` y posteriores.

## Bugs secundarios encontrados y solucionados

Durante la investigación se encontraron y corrigieron varios bugs adicionales:

### ✅ 1. Crash en tool_call sin `id` (TypeError: 'NoneType' object is not subscriptable)
- **Síntoma:** El stream crasheaba cuando un chunk de continuación de tool_call tenía `id: null`
- **Fix:** `_id_val = tc.get("id") or ""` en vez de `tc.get('id','')[:8]`
- **Commit:** `cbb3faf`

### ✅ 2. Valkey no disponible (causa de fallos anteriores)
- **Síntoma:** `replace_base64_with_blob_refs()` retornaba mensajes sin cambios cuando `valkey=None`
- **Fix:** Valkey es hard requirement; `setup_valkey` conecta realmente
- **Commit:** `4a38921`

### ✅ 3. `finish_reason` hardcodeado a "stop"
- **Síntoma:** `_extract_tokens_from_chunks()` siempre ponía `finish_reason="stop"`
- **Fix:** Extraer `finish_reason` real del último chunk del modelo
- **Commit:** `5515e18`

### ✅ 4. Texto extraído de PDF repetido 138KB por request
- **Síntoma:** El PDF se re-extraía completo en cada request, saturando el contexto
- **Fix:** Shorten en cache hit — referencia corta `[File previously extracted — NKB stored in cache.]`
- **Commit:** `e7a6efc`, `9db102c`

### ✅ 5. `finish_reason` en metadata chunk
- **Síntoma:** El chunk de metadata siempre tenía `finish_reason="stop"` hardcodeado
- **Fix:** Usar el `finish_reason` real del modelo
- **Commit:** `4e97eed`

### ✅ 6. Null fields en chunks SSE
- **Síntoma:** `exclude_none=False` producía `"tool_calls": null` que podía confundir parsers
- **Fix:** Strip null values del delta antes de enviar
- **Commit:** `030ffa3`

## Servidor y conexión

| Ítem | Detalle |
|------|---------|
| Servidor | `personal` (hostname: `my-vps`) |
| SSH | `ssh personal` |
| Proxy path | `/opt/proxy-cesar/proxy/` |
| Logs | `/var/log/proxy-cesar/proxy.log` |
| Reinicio | `ssh personal 'sudo systemctl restart proxy-cesar'` |
| Git deploy | `ssh personal 'cd /opt/proxy-cesar && git pull origin main && sudo systemctl restart proxy-cesar'` |
| Valkey | Puerto 6380 |
| Conexión Valkey | `valkey://localhost:6379` |

## Archivos modificados

| Archivo | Cambio |
|---------|--------|
| `proxy/src/api/chat_streaming.py` | Deshabilitar `stream_analysis_msg`, fix crash null id, strip nulls, metadata finish_reason |
| `proxy/src/api/chat_stream_persistence.py` | `_extract_tokens_from_chunks` finish_reason real, metadata chunk finish_reason param |
| `proxy/src/service/tool_detector.py` | Shorten cache hits, Valkey hard requirement |
| `proxy/src/adapters/cache/valkey_affinity.py` | `setup_valkey` real connection |
| `proxy/src/main.py` | Valkey hard startup dependency |
| `proxy/src/api/chat.py` | Tool message diagnostic logging |

## Commits

```
e7c27ec fix: disable analysis message injection that corrupts tool_call SSE streaming
f316461 debug: log tool message content details (len, tc_id, preview)
030ffa3 fix: strip null values from delta in SSE chunks to prevent client misinterpretation
4e97eed fix: metadata chunk now uses actual model finish_reason instead of hardcoded stop
00c3ef0 debug: log final metadata chunk (finish_reason and size)
fed9dfb debug: log raw SSE JSON for tool_call and finish chunks
cbb3faf fix: handle null tool_call id in SSE diagnostic to prevent stream crash
962895e debug: log detailed tool_call chunk structure (index, id, name)
5515e18 fix: preserve finish_reason from model response instead of hardcoding 'stop'
e7a6efc fix: shorten repeated extractions for all media types (files, audio, images)
9db102c fix: shorten repeated file extractions to prevent context overflow
4a38921 feat: make Valkey a hard requirement - fail fast if unavailable
```
