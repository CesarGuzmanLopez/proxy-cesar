# Bug Verification Flow — Historias de Usuario + Comandos Online

---

## HU-1: Chat normal (DeepSeek V4 Flash)

**Como** usuario
**Quiero** enviar mensajes con el modelo `normal`
**Para** obtener respuestas de DeepSeek V4 Flash con tools estrictas

**Criterios:**
- POST `/v1/chat/completions` con `model: "normal"` → 200, respuesta no vacía
- Usa `deepseek/deepseek-v4-flash` (fallback: `openrouter/google/gemini-3.1-flash-lite`)
- Tools strict = true, parallel tools = true
- Si input supera 500K tokens → error 400 `INPUT_EXCEEDS_THRESHOLD`
- No hay compactación automática — el usuario debe usar `POST /compact`

**Comando online:**
```bash
curl -s -X POST http://localhost:9110/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"normal","messages":[{"role":"user","content":"Hola"}],"stream":false}' \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('choices',[{}])[0].get('message',{}).get('content','')[:100])"
```

**Verificar proxy_metadata:**
```bash
curl -s -X POST http://localhost:9110/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"normal","messages":[{"role":"user","content":"Hola"}],"stream":false}' \
  | python3 -c "import sys,json;d=json.load(sys.stdin);md=d.get('proxy_metadata',{});print(f'model={md.get(\"physical_model\")} affinity={md.get(\"affinity_maintained\")} fallback={md.get(\"fallback_applied\")}')"
```

---

## HU-2: Pensamiento profundo (DeepSeek V4 Pro)

**Como** usuario
**Quiero** usar el modelo `pensamiento-profundo-caro`
**Para** resolver bugs imposibles y arquitectura compleja con razonamiento máximo

**Criterios:**
- POST con `model: "pensamiento-profundo-caro"` → 200
- Primary: `deepseek/deepseek-v4-pro` (1M ctx)
- Fallbacks: `openrouter/google/gemini-3.5-flash` (1M ctx, visión+audio)
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
- Fallback: `deepseek/deepseek-v4-flash` (1M ctx)
- Sin router LLM
- Imágenes: block en downgrade

**Comando online:**
```bash
curl -s -X POST http://localhost:9110/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"tareas-avanzadas","messages":[{"role":"user","content":"Implementar un microservicio en Python"}],"stream":false}' \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('choices',[{}])[0].get('message',{}).get('content','')[:100])"
```

---

## HU-4: Visión (Llama 4 Scout via Groq)

**Como** usuario
**Quiero** usar el modelo `vision`
**Para** OCR, diagramas, UI, screenshots

**Criterios:**
- POST con `model: "vision"` → 200
- Único físico: `groq/meta-llama/llama-4-scout-17b-16e-instruct` (131K ctx, visión)
- Sin compactación, sin router LLM
- Imágenes: auto_describe en downgrade

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

## HU-5: Normal gratis (OpenRouter)

**Como** usuario
**Quiero** usar el modelo `normal-gratis`
**Para** coding y tareas generales sin costo

**Criterios:**
- POST con `model: "normal-gratis"` → 200
- Primary: `openrouter/nvidia/nemotron-3-super-120b-a12b:free` (1M ctx)
- Fallback: `groq/qwen/qwen3-32b` (131K ctx)
- Sin compactación, sin router LLM
- Imágenes: auto_describe en downgrade

**Comando online:**
```bash
curl -s -X POST http://localhost:9110/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"normal-gratis","messages":[{"role":"user","content":"Hola gratis"}],"stream":false}' \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('choices',[{}])[0].get('message',{}).get('content','')[:100])"
```

---

## HU-6: Massive Fast (Groq — GPT-OSS 20B)

**Como** usuario
**Quiero** usar el modelo `massive-fast`
**Para** texto masivo ultrarrápido vía Groq

**Criterios:**
- POST con `model: "massive-fast"` → 200
- Único físico: `groq/openai/gpt-oss-20b` (131K ctx, 1000+ t/s)
- Sin compactación, sin router LLM

**Comando online:**
```bash
curl -s -X POST http://localhost:9110/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"massive-fast","messages":[{"role":"user","content":"Respuesta rápida"}],"stream":false}' \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('choices',[{}])[0].get('message',{}).get('content','')[:100])"
```

---

## HU-7: Flash Lowcost (Gemini Flash Lite)

**Como** usuario
**Quiero** usar el modelo `flash-lowcost`
**Para** sub-agentes baratos: clasificación, extracción, parsing

**Criterios:**
- POST con `model: "flash-lowcost"` → 200
- Único físico: `openrouter/google/gemini-3.1-flash-lite` (1M ctx, $0.00025/$0.0015 por M tok)
- Sin compactación, sin router LLM

**Comando online:**
```bash
curl -s -X POST http://localhost:9110/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"flash-lowcost","messages":[{"role":"user","content":"Clasifica: me duele la cabeza"}],"stream":false}' \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('choices',[{}])[0].get('message',{}).get('content','')[:100])"
```

---

## HU-8: Audio (Whisper via Groq)

**Como** usuario
**Quiero** usar el modelo `audio`
**Para** transcripción de audio a texto

**Criterios:**
- POST con `model: "audio"` → 200
- Primary: `groq/whisper-large-v3` (131K ctx)
- Fallback: `groq/whisper-large-v3-turbo` (131K ctx)
- Sin compactación, sin router LLM

**Comando online:**
```bash
curl -s -X POST http://localhost:9110/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"audio","messages":[{"role":"user","content":"Transcribe este audio"}],"stream":false}' \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('choices',[{}])[0].get('message',{}).get('content','')[:100])"
```

---

## HU-9: Imagen (Pruna P-Image)

**Como** usuario
**Quiero** usar el modelo `imagen`
**Para** generar imágenes desde texto

**Criterios:**
- POST con `model: "imagen"` → 200
- Único físico: `pruna/p-image` (texto a imagen en <1s, $0.002/imagen)
- Sin compactación, sin router LLM

**Comando online:**
```bash
curl -s -X POST http://localhost:9110/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"imagen","messages":[{"role":"user","content":"Un gato volador"}],"stream":false}' \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('choices',[{}])[0].get('message',{}).get('content','')[:200])"
```

---

## HU-10: Compactador explícito (POST /compact)

**Como** usuario
**Quiero** compactar una conversación explícitamente
**Para** reducir tokens cuando el input supera el umbral

**Criterios:**
- POST `/conversations/{id}/compact` en conversación con turns → 200, `status: completed`
- Historia ≤ 131K tokens → usa `groq/openai/gpt-oss-20b` (primary, 1000+ t/s)
- Historia > 131K tokens → usa `deepseek/deepseek-v4-flash` (1M ctx)
- Selección por `by_context_window`
- Conversación vacía → 400 `EMPTY_CONVERSATION`
- Si hay imágenes en el historial → delega a modelo con visión para describirlas antes de compactar
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

**Comando online — compactar conversación vacía (debe fallar con 400):**
```bash
CONV_ID=$(python3 -c "import uuid; print(uuid.uuid4())")
curl -s -w "\nHTTP: %{http_code}" -X POST "http://localhost:9110/conversations/$CONV_ID/compact" \
  -H "Content-Type: application/json" \
  -d '{}'
```

---

## HU-11: Streaming SSE

**Como** usuario
**Quiero** recibir respuestas en streaming
**Para** ver el contenido mientras se genera

**Criterios:**
- POST con `stream: true` retorna chunks SSE (`data: ...\n\n`)
- Termina con `data: [DONE]`
- El último chunk antes de `[DONE]` contiene `proxy_metadata`
- Funciona con cualquier pseudo-modelo

**Comando online:**
```bash
curl -s -X POST http://localhost:9110/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"normal","messages":[{"role":"user","content":"Cuenta hasta 5"}],"stream":true}' \
  | head -15
```

**Verificar metadata final:**
```bash
curl -s -X POST http://localhost:9110/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"normal","messages":[{"role":"user","content":"Hola"}],"stream":true}' \
  | python3 -c "
import sys
for line in sys.stdin:
    line = line.strip()
    if line.startswith('data: ') and line != 'data: [DONE]':
        import json
        d = json.loads(line[6:])
        if 'proxy_metadata' in d:
            print('Metadata found:', list(d['proxy_metadata'].keys()))
"
```

---

## HU-12: Model aliases

**Como** usuario
**Quiero** usar alias como `gpt-4o`, `claude-haiku-3-5`, etc.
**Para** migrar desde otros proveedores sin cambiar mi código

**Criterios:**
- `gpt-4o` → resuelve a `normal`
- `gpt-4o-mini` → resuelve a `normal`
- `gpt-4.1` → resuelve a `tareas-avanzadas`
- `o3`, `o4-mini` → resuelven a `pensamiento-profundo-caro`
- `gemini-2.5-flash` → resuelve a `vision`
- `claude-haiku-3-5-20241022` → resuelve a `flash-lowcost`
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

## HU-13: KeyVault — Protección de Secrets

**Como** usuario
**Quiero** que mis API keys sean interceptadas antes de llegar al LLM
**Para** que el modelo nunca vea mis credenciales

**Criterios:**
- Enviar mensaje con API key (ej: `sk-...`) → proxy la detecta
- La key se reemplaza por `[KEYVAULT:hash]` en el mensaje al LLM
- La key se guarda en Valkey con TTL de 1 hora
- En la respuesta, el placeholder se reinyecta con el valor real
- Funciona tanto en streaming como no-streaming
- Se inyecta system prompt explicando el uso de placeholders

**Comando online — verificar sanitización (no-streaming):**
```bash
curl -s -X POST http://localhost:9110/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"normal","messages":[{"role":"user","content":"Mi key es sk-proj-abc123def456ghi789jkl012"}],"conversation_id":"conv-keyvault","stream":false}' \
  | python3 -c "import sys,json;d=json.load(sys.stdin);c=d.get('choices',[{}])[0].get('message',{}).get('content','')[:100];print('OK:', bool(c))"
```

**Comando online — verificar sanitización (streaming):**
```bash
curl -s -X POST http://localhost:9110/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"normal","messages":[{"role":"user","content":"Mi key es sk-proj-abc123def456ghi789jkl012"}],"conversation_id":"conv-keyvault-stream","stream":true}' \
  | tail -5
```

---

## HU-14: Contenido no soportado → Blob Vault

**Como** usuario
**Quiero** enviar una imagen/audio/PDF a un modelo que no lo soporta
**Para** que el proxy guarde el archivo como blob, genere una descripción y se la pase al modelo

**Criterios:**
- POST con imagen + modelo sin visión → 200, el modelo recibe una descripción de la imagen
- El base64 se almacena en Valkey y se reemplaza por `[BLOB:hash:mime | size | descripción]`
- El modelo puede usar sus tools si quiere procesar el blob (`GET /blobs/{hash}`)
- POST con imagen + tool + modelo sin visión → mismo comportamiento (el proxy no inspecciona tools)
- POST con imagen + `imagen` (text-to-image) → 400 error (modelo especializado)
- POST con imagen + `audio` (whisper) → 400 error (modelo especializado)

**Comando online — imagen a modelo sin visión (se blobifica):**
```bash
curl -s -X POST http://localhost:9110/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"normal",
    "messages":[{"role":"user","content":[
      {"type":"text","text":"Describe esta imagen usando la tool analyze_image"},
      {"type":"image_url","image_url":{"url":"https://example.com/img.png"}}
    ]}],
    "tools":[{"function":{"name":"analyze_image","parameters":{"type":"object","properties":{"image_url":{"type":"string"}}}}}],
    "stream":false
  }' | python3 -c "import sys,json;d=json.load(sys.stdin);print('OK:', d.get('choices',[{}])[0].get('message',{}).get('content','')[:200] if 'choices' in d else d)"
```

**Comando online — recuperar blob (tools):**
```bash
# Primero enviar un request que blobifique contenido
CONV_ID=$(python3 -c "import uuid; print(uuid.uuid4())")
curl -s -X POST http://localhost:9110/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"normal\",\"messages\":[{\"role\":\"user\",\"content\":[
    {\"type\":\"text\",\"text\":\"hola\"},
    {\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVQI12NgAAIABQABNjN9GQAAAAlwSFlzAAAWJQAAFiUBSVIk8AAAAA0lEQVQI12P4z8BQDwAEgAF/QualzQAAAABJRU5ErkJggg==\"}}
  ]}],\"conversation_id\":\"$CONV_ID\",\"stream\":false}"

# Ver el mensaje que recibió el modelo (tiene blob reference)
```

---

## HU-15: Fallback strategy

**Como** sistema
**Quiero** que cuando el physical primario falle (timeout, rate limit, error)
**Para** pasar automáticamente al siguiente físico en la lista

**Criterios:**
- `pensamiento-profundo-caro`: sequential → DeepSeek V4 Pro → Gemini 3.5 Flash
- `compactador`: by_context_window → elige según tamaño de historia
- Todos los demás: sequential
- Si todos los físicos fallan → error 503 `ALL_MODELS_FAILED` (o 413 si todos saltaron por contexto)

**Comando online — ver physical model real usado:**
```bash
curl -s -X POST http://localhost:9110/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"pensamiento-profundo-caro","messages":[{"role":"user","content":"test"}],"stream":false}' \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print('physical_model:', d.get('proxy_metadata',{}).get('physical_model','?'))"
```

---

## HU-16: Auditoría (GET audit-log)

**Como** usuario
**Quiero** ver el log de eventos de una conversación
**Para** saber qué cambios de modelo, fallbacks y compactaciones ocurrieron

**Criterios:**
- GET `/conversations/{id}/audit-log` → 200 con array de eventos cronológicos
- Incluye eventos de creación, cambios de pseudo-model, fallbacks, compactaciones explícitas

**Comando online:**
```bash
CONV_ID=$(python3 -c "import uuid; print(uuid.uuid4())")
curl -s -X POST http://localhost:9110/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"normal\",\"messages\":[{\"role\":\"user\",\"content\":\"Hola\"}],\"conversation_id\":\"$CONV_ID\",\"stream\":false}" > /dev/null

curl -s "http://localhost:9110/conversations/$CONV_ID/audit-log" \
  | python3 -c "import sys,json;d=json.load(sys.stdin);[print(f'{e.get(\"event_type\")}: {json.dumps(e.get(\"details\",{}))}') for e in d.get('events',[])]"
```

---

## HU-17: Health check + Metrics

**Como** operador
**Quiero** consultar `/health` y `/metrics`
**Para** saber qué proveedores están configurados y el estado del proxy

**Criterios:**
- GET `/health` → 200 con proveedores configurados
- GET `/metrics` → 200 con métricas de uso

**Comando online:**
```bash
curl -s http://localhost:9110/health | python3 -m json.tool
```

**Verificar métricas:**
```bash
curl -s http://localhost:9110/metrics | python3 -c "
import sys,json
d=json.load(sys.stdin)
print(f'requests: {d[\"total_requests\"]}')
print(f'compactions: {d[\"compactions\"]}')
print(f'uptime: {d[\"uptime_seconds\"]}s')
"
```

---

## HU-18: Threshold exceeded → error explícito

**Como** usuario
**Quiero** que si mi input supera el umbral del pseudo-modelo
**Para** recibir un error claro en vez de compactación silenciosa, y decidir si compacto manualmente

**Criterios:**
- POST con input > `input_token_threshold` → 400 `INPUT_EXCEEDS_THRESHOLD`
- El error incluye el número de tokens y los modelos sugeridos
- El usuario debe usar `POST /compact` para reducir el contexto

**Comando online — enviar input masivo para superar umbral:**
```bash
curl -s -w "\nHTTP: %{http_code}" -X POST http://localhost:9110/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"normal\",\"messages\":[{\"role\":\"user\",\"content\":\"$(python3 -c \"print('x'*200000)\")\"}],\"stream\":false}"
```
