# Bug Verification Flow — Historias de Usuario + Comandos Online

> **Nota:** Todos los comandos funcionan contra producción (`chat.guzman-lopez.com`) o local (`localhost:9110`).
> Ajusta la URL según corresponda. Para producción, añade `-H "Authorization: Bearer <token>"`.

---

## HU-1: Chat normal (kimi-k2.5)

**Como** usuario
**Quiero** enviar mensajes con el modelo `normal`
**Para** obtener respuestas de kimi-k2.5 con tools estrictas

**Criterios:**
- POST `/v1/chat/completions` con `model: "normal"` → 200
- Usa `openai/kimi-k2.5` (OpenCode Go, physical model)
- Fallback: `deepseek/deepseek-v4-flash`
- Tools strict = true, parallel tools = true
- Si input supera 500K tokens → error 400 `INPUT_EXCEEDS_THRESHOLD`

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

## HU-2: Pensamiento profundo (qwen3.7-max)

**Como** usuario
**Quiero** usar el modelo `pensamiento-profundo-caro`
**Para** resolver bugs imposibles y arquitectura compleja con razonamiento máximo

**Criterios:**
- POST con `model: "pensamiento-profundo-caro"` → 200
- Primary: `anthropic/qwen3.7-max` (Go, Anthropic-compat, 1M ctx)
- Fallbacks: `deepseek/deepseek-v4-pro`
- Router LLM activo con `flash-lowcost` como suggester (solo en downgrade)

**Comando online:**
```bash
curl -s -X POST http://localhost:9110/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"pensamiento-profundo-caro","messages":[{"role":"user","content":"Arquitectura de un parser LR(1)"}],"stream":false}' \
  | python3 -c "import sys,json;d=json.load(sys.stdin);c=d.get('choices',[{}])[0].get('message',{}).get('content','');print(c[:200])"
```

---

## HU-3: Tareas avanzadas (kimi-k2.6)

**Como** usuario
**Quiero** usar el modelo `tareas-avanzadas`
**Para** desarrollo profundo y sostenido, features largas, debugging serio

**Criterios:**
- POST con `model: "tareas-avanzadas"` → 200
- Primary: `openai/kimi-k2.6` (Go, 128K ctx)
- Fallback: `deepseek/deepseek-v4-pro` → `deepseek/deepseek-v4-flash`
- Sin router LLM

**Comando online:**
```bash
curl -s -X POST http://localhost:9110/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"tareas-avanzadas","messages":[{"role":"user","content":"Implementar un microservicio en Python"}],"stream":false}' \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('choices',[{}])[0].get('message',{}).get('content','')[:100])"
```

---

## HU-4: Visión (mimo-v2-omni)

**Como** usuario
**Quiero** usar el modelo `vision`
**Para** OCR, diagramas, UI, screenshots

**Criterios:**
- POST con `model: "vision"` → 200
- Primary: `openai/mimo-v2-omni` (Go, 128K ctx, visión)
- Fallback: `groq/meta-llama/llama-4-scout` (131K ctx, visión)

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

## HU-5: Normal gratis (Nemotron OpenRouter)

**Como** usuario
**Quiero** usar el modelo `normal-gratis`
**Para** coding y tareas generales sin costo

**Criterios:**
- POST con `model: "normal-gratis"` → 200
- Primary: `openrouter/nvidia/nemotron-3-super-120b-a12b:free` (1M ctx)
- Fallback: `groq/qwen/qwen3-32b` (131K ctx)

**Comando online:**
```bash
curl -s -X POST http://localhost:9110/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"normal-gratis","messages":[{"role":"user","content":"Hola gratis"}],"stream":false}' \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('choices',[{}])[0].get('message',{}).get('content','')[:100])"
```

---

## HU-6: Massive Fast (minimax-m2.7)

**Como** usuario
**Quiero** usar el modelo `massive-fast`
**Para** texto masivo ultrarrápido

**Criterios:**
- POST con `model: "massive-fast"` → 200
- Primary: `openai/minimax-m2.7` (Go, 200K ctx)
- Fallback: `groq/openai/gpt-oss-20b` (131K ctx, 1000+ t/s)

**Comando online:**
```bash
curl -s -X POST http://localhost:9110/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"massive-fast","messages":[{"role":"user","content":"Respuesta rápida"}],"stream":false}' \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('choices',[{}])[0].get('message',{}).get('content','')[:100])"
```

---

## HU-7: Flash Lowcost (qwen3.5-plus)

**Como** usuario
**Quiero** usar el modelo `flash-lowcost`
**Para** sub-agentes baratos: clasificación, extracción, parsing

**Criterios:**
- POST con `model: "flash-lowcost"` → 200
- Primary: `openai/qwen3.5-plus` (Go, 1M ctx)
- Fallback: `deepseek/deepseek-v4-flash`

**Comando online:**
```bash
curl -s -X POST http://localhost:9110/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"flash-lowcost","messages":[{"role":"user","content":"Clasifica: me duele la cabeza"}],"stream":false}' \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('choices',[{}])[0].get('message',{}).get('content','')[:100])"
```

---

## HU-8: Compactador explícito (POST /compact)

**Como** usuario
**Quiero** compactar una conversación explícitamente
**Para** reducir tokens cuando el input supera el umbral

**Criterios:**
- POST `/conversations/{id}/compact` en conversación con turns → 200, `status: completed`
- Usa `openai/glm-5.1` (primary, 128K ctx)
- Historia > umbral → fallback a `deepseek/deepseek-v4-flash` (1M ctx)
- Conversación vacía → 400 `EMPTY_CONVERSATION`

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

---

## HU-9: Streaming SSE

**Como** usuario
**Quiero** recibir respuestas en streaming
**Para** ver el contenido mientras se genera

**Criterios:**
- POST con `stream: true` retorna chunks SSE (`data: ...\n\n`)
- Termina con `data: [DONE]`
- El último chunk antes de `[DONE]` contiene `proxy_metadata`

**Comando online:**
```bash
curl -s -X POST http://localhost:9110/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"normal","messages":[{"role":"user","content":"Cuenta hasta 5"}],"stream":true}' \
  | head -15
```

---

## HU-10: Model aliases

**Como** usuario
**Quiero** usar alias como `gpt-4o`, `claude-haiku-3-5`, etc.
**Para** migrar desde otros proveedores sin cambiar mi código

**Criterios:**
- `gpt-4o` / `gpt-4o-mini` → resuelve a `normal`
- `gpt-4.1` → resuelve a `tareas-avanzadas`
- `o3` → resuelve a `pensamiento-profundo-caro`
- `o4-mini` → resuelve a `pensamiento-rapido`
- `gemini-2.5-flash` → resuelve a `vision`
- `gemini-2.5-pro` → resuelve a `codigo-preciso`
- `claude-sonnet-4-20250514` → resuelve a `codigo-preciso`
- `claude-haiku-3-5-20241022` → resuelve a `flash-lowcost`
- `default` → resuelve a `normal`

**Comando online:**
```bash
for alias in gpt-4o gpt-4o-mini o3 o4-mini claude-haiku-3-5-20241022 gemini-2.5-flash default; do
  echo -n "$alias → "
  curl -s -X POST http://localhost:9110/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$alias\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":false}" \
    | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('model','?'))"
done
```

---

## HU-11: KeyVault — Protección de Secrets

**Como** usuario
**Quiero** que mis API keys sean interceptadas antes de llegar al LLM
**Para** que el modelo nunca vea mis credenciales

**Comando online — verificar sanitización (no-streaming):**
```bash
curl -s -X POST http://localhost:9110/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"normal","messages":[{"role":"user","content":"Mi key es sk-proj-abc123def456ghi789jkl012"}],"conversation_id":"conv-keyvault","stream":false}' \
  | python3 -c "import sys,json;d=json.load(sys.stdin);c=d.get('choices',[{}])[0].get('message',{}).get('content','')[:100];print('OK:', bool(c))"
```

---

## HU-12: Contenido no soportado → Blob Vault

**Como** usuario
**Quiero** enviar una imagen/audio/PDF a un modelo que no lo soporta
**Para** que el proxy guarde el archivo como blob, genere una descripción y se la pase al modelo

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

---

## HU-13: Fallback strategy

**Como** sistema
**Quiero** que cuando el physical primario falle (timeout, rate limit, error)
**Para** pasar automáticamente al siguiente físico en la lista

**Criterios:**
- Todos los pseudo-modelos usan fallback `sequential`
- El único con `by_context_window` es `compactador`
- Si todos los físicos fallan → error 502 `ALL_MODELS_FAILED` (o 413 si todos saltaron por contexto)

**Comando online — ver physical model real usado:**
```bash
curl -s -X POST http://localhost:9110/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"pensamiento-profundo-caro","messages":[{"role":"user","content":"test"}],"stream":false}' \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print('physical_model:', d.get('proxy_metadata',{}).get('physical_model','?'))"
```

---

## HU-14: Auditoría (GET audit-log)

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

## HU-15: Health check + Metrics

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

## HU-16: Threshold exceeded → error explícito

**Comando online — enviar input masivo para superar umbral:**
```bash
curl -s -w "\nHTTP: %{http_code}" -X POST http://localhost:9110/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"normal\",\"messages\":[{\"role\":\"user\",\"content\":\"$(python3 -c \"print('x'*200000)\")\"}],\"stream\":false}"
```
