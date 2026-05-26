# Bug Verification Flow — Historias de Usuario + Comandos Online

---

## HU-1: Chat normal (DeepSeek V4 Flash)

**Como** usuario
**Quiero** enviar mensajes con el modelo `normal`
**Para** obtener respuestas de DeepSeek V4 Flash con tools estrictas

**Criterios:**
- POST `/v1/chat/completions` con `model: "normal"` → 200, respuesta no vacía
- Usa `deepseek/deepseek-v4-flash` (fallback: `zai/glm-4.5-flash`)
- Tools strict = true, parallel tools = true
- Compactación continua activa al 80% del context window

**Comando online:**
```bash
curl -s -X POST http://localhost:9110/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"normal","messages":[{"role":"user","content":"Hola"}],"stream":false}' \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('choices',[{}])[0].get('message',{}).get('content','')[:100])"
```

---

## HU-2: Pensamiento profundo (GLM-5 + DeepSeek V4 Pro + Claude)

**Como** usuario
**Quiero** usar el modelo `pensamiento-profundo-caro`
**Para** resolver bugs imposibles y arquitectura compleja con razonamiento máximo

**Criterios:**
- POST con `model: "pensamiento-profundo-caro"` → 200
- Primary: `zai/glm-5` (200K ctx)
- Fallbacks: `deepseek/deepseek-v4-pro` (1M ctx), `anthropic/claude-haiku-4-5` (200K ctx)
- Fallback strategy: sequential (si uno falla, pasa al siguiente)
- Compactación continua activa al 70%, preserve 16K
- Router LLM activo con `flash-lowcost` como suggester (solo en downgrade)
- Imágenes: auto_describe en downgrade

**Comando online:**
```bash
curl -s -X POST http://localhost:9110/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"pensamiento-profundo-caro","messages":[{"role":"user","content":"Arquitectura de un parser LR(1)"}],"stream":false}' \
  | python3 -c "import sys,json;d=json.load(sys.stdin);c=d.get('choices',[{}])[0].get('message',{}).get('content','');print(c[:200])"
```

**Verificar router_llm (suggester flash-lowcost):**
```bash
curl -s -X POST http://localhost:9110/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"pensamiento-profundo-caro","messages":[{"role":"user","content":"Qué hora es"}],"stream":false}' \
  | python3 -c "import sys,json;d=json.load(sys.stdin);md=d.get('proxy_metadata',{});print('router_suggestion:', json.dumps(md.get('router_suggestion'), indent=2))"
```

---

## HU-3: Tareas avanzadas (DeepSeek V4 Pro + Flash)

**Como** usuario
**Quiero** usar el modelo `tareas-avanzadas`
**Para** desarrollo profundo y sostenido, features largas, debugging serio

**Criterios:**
- POST con `model: "tareas-avanzadas"` → 200
- Primary: `deepseek/deepseek-v4-pro` (1M ctx)
- Fallbacks: `deepseek/deepseek-v4-flash` (1M ctx), `zai/glm-4.5-flash` (128K ctx)
- Compactación continua al 75%, preserve 32K
- Sin router LLM, sin pre-compaction
- Imágenes: block en downgrade

**Comando online:**
```bash
curl -s -X POST http://localhost:9110/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"tareas-avanzadas","messages":[{"role":"user","content":"Implementar un microservicio en Python"}],"stream":false}' \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('choices',[{}])[0].get('message',{}).get('content','')[:100])"
```

---

## HU-4: Visión (Gemini Flash + Lite + Ollama)

**Como** usuario
**Quiero** usar el modelo `vision`
**Para** OCR, diagramas, UI, screenshots con calidad decreciente

**Criterios:**
- POST con `model: "vision"` → 200
- Primary: `gemini/gemini-3.5-flash` (1M ctx, mejor calidad)
- Fallback 1: `gemini/gemini-3.1-flash-lite` (1M ctx, barato)
- Fallback 2: `ollama/llava` (4K ctx, local gratuito)
- Sin compactación continua, sin router LLM
- Imágenes: auto_describe en downgrade
- Todos los físicos deben tener `vision: true`

**Comando online (con imagen):**
```bash
curl -s -X POST http://localhost:9110/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"vision",
    "messages":[{"role":"user","content":[
      {"type":"text","text":"Describe esta imagen"},
      {"type":"image_url","image_url":{"url":"https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/300px-PNG_transparency_demonstration_1.png","detail":"auto"}}
    ]}],
    "stream":false
  }' | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('choices',[{}])[0].get('message',{}).get('content','')[:200])"
```

---

## HU-5: Vision Lite (Z.ai + Groq)

**Como** usuario
**Quiero** usar el modelo `vision-lite`
**Para** visión rápida y gratuita sin tools

**Criterios:**
- POST con `model: "vision-lite"` → 200
- Primary: `zai/glm-4.6v-flash` (128K ctx, gratis vía Z.ai)
- Fallback: `groq/meta-llama/llama-4-scout-17b-16e-instruct` (131K ctx, visión Groq)
- Sin compactación, sin router LLM
- Todos los físicos con `vision: true`

**Comando online:**
```bash
curl -s -X POST http://localhost:9110/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"vision-lite",
    "messages":[{"role":"user","content":[
      {"type":"text","text":"Describe esta imagen"},
      {"type":"image_url","image_url":{"url":"https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/300px-PNG_transparency_demonstration_1.png","detail":"auto"}}
    ]}],
    "stream":false
  }' | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('choices',[{}])[0].get('message',{}).get('content','')[:200])"
```

---

## HU-6: Normal gratis (OpenRouter + Z.ai)

**Como** usuario
**Quiero** usar el modelo `normal-gratis`
**Para** coding y tareas generales sin costo

**Criterios:**
- POST con `model: "normal-gratis"` → 200
- Primary: `openrouter/nvidia/nemotron-3-super-120b-a12b:free` (1M ctx)
- Fallbacks: `zai/glm-4.7-flash` (203K), `openrouter/google/gemma-4-31b-it:free` (262K, con visión), `zai/glm-4.5-flash` (128K)
- Sin compactación continua, sin router LLM
- Fallback strategy: sequential
- Imágenes: auto_describe en downgrade

**Comando online:**
```bash
curl -s -X POST http://localhost:9110/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"normal-gratis","messages":[{"role":"user","content":"Hola gratis"}],"stream":false}' \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('choices',[{}])[0].get('message',{}).get('content','')[:100])"
```

---

## HU-7: Deep Flash (DeepSeek V4 Flash directo)

**Como** usuario
**Quiero** usar el modelo `deep-flash`
**Para** velocidad y costo mínimo en tareas masivas, lectura de docs gigantes, traducciones

**Criterios:**
- POST con `model: "deep-flash"` → 200
- Único físico: `deepseek/deepseek-v4-flash` (1M ctx)
- Sin compactación, sin router LLM
- Imágenes: block en downgrade

**Comando online:**
```bash
curl -s -X POST http://localhost:9110/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deep-flash","messages":[{"role":"user","content":"Traduce al inglés: Hola mundo"}],"stream":false}' \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('choices',[{}])[0].get('message',{}).get('content','')[:100])"
```

---

## HU-8: Massive Fast (Groq — GPT-OSS 20B + Qwen 32B)

**Como** usuario
**Quiero** usar el modelo `massive-fast`
**Para** texto masivo ultrarrápido vía Groq

**Criterios:**
- POST con `model: "massive-fast"` → 200
- Primary: `groq/openai/gpt-oss-20b` (131K ctx) — LiteLLM format con `openai/` intermedio
- Fallback: `groq/qwen/qwen3-32b` (131K ctx)
- Sin compactación, sin router LLM
- Tools regulares en emergencia

**Comando online:**
```bash
curl -s -X POST http://localhost:9110/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"massive-fast","messages":[{"role":"user","content":"Respuesta rápida"}],"stream":false}' \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('choices',[{}])[0].get('message',{}).get('content','')[:100])"
```

---

## HU-9: Flash Lowcost (Z.ai + Ollama)

**Como** usuario
**Quiero** usar el modelo `flash-lowcost`
**Para** sub-agentes baratos: clasificación, extracción, parsing

**Criterios:**
- POST con `model: "flash-lowcost"` → 200
- Primary: `zai/glm-4.5-flash` (128K ctx, barato)
- Fallback: `ollama/llama3.2` (128K ctx, local gratuito)
- Sin compactación, sin router LLM
- Imágenes: block en downgrade

**Comando online:**
```bash
curl -s -X POST http://localhost:9110/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"flash-lowcost","messages":[{"role":"user","content":"Clasifica: me duele la cabeza"}],"stream":false}' \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('choices',[{}])[0].get('message',{}).get('content','')[:100])"
```

---

## HU-10: Compactador explícito (POST /compact)

**Como** usuario
**Quiero** compactar una conversación explícitamente
**Para** reducir tokens usando Groq (rápido y barato) o Gemini (historias masivas)

**Criterios:**
- POST `/conversations/{id}/compact` en conversación con turns → 200, `status: completed`
- Historia ≤ 131K tokens → usa `groq/openai/gpt-oss-120b` (primary, 500+ t/s)
- Historia > 131K tokens → usa `gemini/gemini-3.5-flash` (1M ctx)
- Fallbacks disponibles: `groq/openai/gpt-oss-20b`, `groq/qwen/qwen3-32b`, `anthropic/claude-haiku-4-5`, `zai/glm-4.5-flash`
- Selección por `by_context_window` (elige el primer físico con `context_window >= total_tokens`)
- Conversación vacía → 400 `EMPTY_CONVERSATION`
- **NO** debe fallar con `greenlet_spawn` (error preexistente corregido moviendo DB ops antes del API call)
- Compactación múltiple: segundo compact genera nuevo `snapshot_id` y encadena con `superseded_by`

**Comando online — compactar conversación existente:**
```bash
CONV_ID=$(python3 -c "import uuid; print(uuid.uuid4())")
for i in 1 2 3; do
  curl -s -X POST http://localhost:9110/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"normal\",\"messages\":[{\"role\":\"user\",\"content\":\"turn $i\"}],\"conversation_id\":\"$CONV_ID\",\"stream\":false}" > /dev/null
done

curl -s -X POST "http://localhost:9110/conversations/$CONV_ID/compact" \
  -H "Content-Type: application/json" \
  -d '{}' | python3 -m json.tool
```

**Comando online — compactar conversación vacía (debe fallar):**
```bash
CONV_ID=$(python3 -c "import uuid; print(uuid.uuid4())")
curl -s -w "\nHTTP: %{http_code}" -X POST "http://localhost:9110/conversations/$CONV_ID/compact" \
  -H "Content-Type: application/json" \
  -d '{}'
```

**Comando online — doble compactación (verificar chaining):**
```bash
curl -s -X POST "http://localhost:9110/conversations/$CONV_ID/compact" \
  -H "Content-Type: application/json" \
  -d '{}' | python3 -c "import sys,json;d=json.load(sys.stdin);print(f'status={d[\"status\"]} snapshot={d[\"snapshot_id\"][:8]}')"
```

**Comando online — ver estado de la conversación (ver active_snapshot_id):**
```bash
curl -s "http://localhost:9110/conversations/$CONV_ID" | python3 -c "import sys,json;d=json.load(sys.stdin);print(f'tokens={d.get(\"total_tokens\")} turns={d.get(\"turn_count\")} snapshot={d.get(\"active_snapshot_id\",\"-\")[:8] if d.get(\"active_snapshot_id\") else \"none\"}')"
```

---

## HU-11: Compactación continua (automática al chatear)

**Como** sistema
**Quiero** compactar automáticamente cuando el contexto supera el `trigger_pct`
**Para** mantener la conversación dentro del context window sin intervención del usuario

**Criterios:**
- Se activa en pseudo-models con `continuous_compaction.enabled: true` (normal, pensamiento-profundo-caro, tareas-avanzadas)
- Se ejecuta en `_run_compaction_pipeline` antes de cada request de chat
- Si `total_tokens > context_window * trigger_pct / 100` → compacta
- Usa `groq/openai/gpt-oss-120b` como compactador primario
- Preserva los últimos `compact_preserve_recent` tokens (no se compactan)
- Almacena snapshot con `snapshot_type="continuous"`
- Actualiza `conversation.total_tokens = snapshot_tokens + preserved_tokens`
- Usa `/compact` inline command si el usuario quiere forzar compactación continua

**Comando online — forzar compactación continua vía inline command:**
```bash
CONV_ID=$(python3 -c "import uuid; print(uuid.uuid4())")
# Enviar muchos mensajes para llenar contexto
for i in $(seq 1 20); do
  curl -s -X POST http://localhost:9110/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"normal\",\"messages\":[{\"role\":\"user\",\"content\":\"Mensaje largo $i $(python3 -c \"print('x'*500)\")\"}],\"conversation_id\":\"$CONV_ID\",\"stream\":false}" > /dev/null
done

# Ver estado (tokens deben estar controlados por compactación continua)
curl -s "http://localhost:9110/conversations/$CONV_ID" \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print(f'tokens={d.get(\"total_tokens\")} turns={d.get(\"turn_count\")} snapshot={d.get(\"active_snapshot_id\",\"-\")[:8] if d.get(\"active_snapshot_id\") else \"none\"}')"
```

**Comando online — inline /compact (fuerza compactación continua desde el chat):**
```bash
curl -s -X POST http://localhost:9110/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"normal\",\"messages\":[{\"role\":\"user\",\"content\":\"/compact\"}],\"conversation_id\":\"$CONV_ID\",\"stream\":false}"
```

---

## HU-12: Degradación de imágenes online

**Como** usuario
**Quiero** degradar imágenes en una conversación
**Para** poder cambiar a un modelo sin visión sin perder contexto

**Criterios:**
- Envío de imagen a modelo `vision` → se almacena como mensaje con imagen
- POST `/conversations/{id}/degrade-images` → describe todas las imágenes y las reemplaza por texto
- Después de degradar, cambiar a `normal` debe ser seguro (no más imágenes)
- Si el pseudo-model destino tiene `on_downgrade: auto_describe` → la degradación ocurre automáticamente al cambiar
- Si el destino tiene `on_downgrade: block` → el cambio se bloquea hasta degradar manualmente

**Comando online — enviar imagen a vision:**
```bash
CONV_ID=$(python3 -c "import uuid; print(uuid.uuid4())")
curl -s -X POST http://localhost:9110/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\":\"vision\",
    \"messages\":[{\"role\":\"user\",\"content\":[
      {\"type\":\"text\",\"text\":\"Describe\"},
      {\"type\":\"image_url\",\"image_url\":{\"url\":\"https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/300px-PNG_transparency_demonstration_1.png\"}}
    ]}],
    \"conversation_id\":\"$CONV_ID\",
    \"stream\":false
  }" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('choices',[{}])[0].get('message',{}).get('content','')[:150])"
```

**Comando online — degradación manual:**
```bash
curl -s -X POST "http://localhost:9110/conversations/$CONV_ID/degrade-images" \
  -H "Content-Type: application/json" \
  -d '{}' | python3 -m json.tool
```

**Comando online — verificar compatibilidad después de degradar:**
```bash
curl -s "http://localhost:9110/conversations/$CONV_ID/compatible-models" \
  | python3 -c "import sys,json;d=json.load(sys.stdin);[print(f'{m[\"name\"]}: {m[\"status\"]}') for m in d.get('models',[])]"
```

**Comando online — cambiar a normal después de degradar (debe funcionar):**
```bash
curl -s -X POST http://localhost:9110/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"normal\",\"messages\":[{\"role\":\"user\",\"content\":\"Mensaje después de degradar\"}],\"conversation_id\":\"$CONV_ID\",\"stream\":false}" \
  | python3 -c "import sys,json;d=json.load(sys.stdin);c=d.get('choices',[{}])[0].get('message',{}).get('content','')[:100];print(c)"
```

**Comando online — enviar imagen directo a modelo sin visión (debe fallar):**
```bash
curl -s -w "\nHTTP: %{http_code}" -X POST http://localhost:9110/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"normal",
    "messages":[{"role":"user","content":[
      {"type":"text","text":"Describe"},
      {"type":"image_url","image_url":{"url":"https://example.com/img.png"}}
    ]}],
    "stream":false
  }'
```

---

## HU-13: Auditoría (GET audit-log)

**Como** usuario
**Quiero** ver el log de eventos de una conversación
**Para** saber qué compactaciones, degradaciones y cambios de modelo ocurrieron

**Criterios:**
- GET `/conversations/{id}/audit-log` → 200 con array de eventos cronológicos
- Incluye eventos de creación, cambios de pseudo-model, fallbacks, compactaciones explícitas y continuas, degradaciones de imagen

**Comando online:**
```bash
curl -s "http://localhost:9110/conversations/$CONV_ID/audit-log" \
  | python3 -c "import sys,json;d=json.load(sys.stdin);[print(f'{e.get(\"event_type\")}: {json.dumps(e.get(\"details\",{}))}') for e in d.get('events',[])]"
```

---

## HU-14: Streaming

**Como** usuario
**Quiero** recibir respuestas en streaming
**Para** ver el contenido mientras se genera

**Criterios:**
- POST con `stream: true` retorna chunks SSE
- Termina con `data: [DONE]`
- Funciona con cualquier pseudo-modelo

**Comando online:**
```bash
curl -s -X POST http://localhost:9110/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"normal","messages":[{"role":"user","content":"Cuenta hasta 5"}],"stream":true}' \
  | head -15
```

---

## HU-15: Model aliases

**Como** usuario
**Quiero** usar alias como `gpt-4o`, `claude-haiku-3-5`, etc.
**Para** migrar desde otros proveedores sin cambiar mi código

**Criterios:**
- `gpt-4o` → resuelve a `normal`
- `gpt-4o-mini` → resuelve a `deep-flash`
- `gpt-4.1` → resuelve a `tareas-avanzadas`
- `o3`, `o4-mini` → resuelven a `pensamiento-profundo-caro`
- `claude-haiku-3-5-20241022` → resuelve a `flash-lowcost`
- `gemini-2.5-flash`, `gemini-2.5-pro` → resuelven a `vision`
- `default` → resuelve a `normal`

**Comando online:**
```bash
for alias in gpt-4o gpt-4o-mini o3 claude-haiku-3-5-20241022 gemini-2.5-flash default; do
  echo -n "$alias → "
  curl -s -X POST http://localhost:9110/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$alias\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":false}" \
    | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('model','?'))"
done
```

---

## HU-16: Fallback strategy

**Como** sistema
**Quiero** que cuando el physical primario falle (timeout, rate limit, error)
**Para** pasar automáticamente al siguiente físico en la lista

**Criterios:**
- `pensamiento-profundo-caro`: sequential → GLM-5 → DeepSeek V4 Pro → Claude
- `compactador`: by_context_window → elige según tamaño de historia
- Todos los demás: sequential
- Si todos los físicos fallan → error 502 `COMPACTION_FAILED` o 5xx según el contexto

**Comando online — ver physical model real usado:**
```bash
curl -s -X POST http://localhost:9110/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"pensamiento-profundo-caro","messages":[{"role":"user","content":"test"}],"stream":false}' \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print('physical_model:', d.get('proxy_metadata',{}).get('physical_model','?'))"
```

---

## HU-17: Health check

**Como** operador
**Quiero** consultar `/health`
**Para** saber qué proveedores están configurados

**Criterios:**
- GET `/health` retorna 200
- Todos los proveedores esperados aparecen como `true` (anthropic, openrouter, google, deepseek, groq, zhipuai, zai)

**Comando online:**
```bash
curl -s http://localhost:9110/health | python3 -m json.tool
```
