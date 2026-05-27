# Bug Verification Flow — Historias de Usuario + Comandos

> **Nota:** Usa `localhost:9110` para local, `chat.guzman-lopez.com` para producción.
> Para producción añade `-H "Authorization: Bearer <token>"`.

---

## 1. Smoke Test — Health + Models

```bash
# Health check
curl -s http://localhost:9110/health | python3 -m json.tool

# Listar modelos
curl -s http://localhost:9110/v1/models | python3 -c "import sys,json;d=json.load(sys.stdin);[print(m['id']) for m in d]"

# Métricas
curl -s http://localhost:9110/metrics | python3 -c "
import sys,json;d=json.load(sys.stdin)
print(f'requests: {d[\"total_requests\"]}')
print(f'compactions: {d[\"compactions\"]}')
print(f'uptime: {d[\"uptime_seconds\"]}s')
"
```

---

## 2. Todos los 10 Pseudo-Modelos — Verificación Rápida

```bash
for model in normal pensamiento-profundo-caro tareas-avanzadas codigo-preciso vision pensamiento-rapido normal-gratis massive-fast flash-lowcost compactador; do
  echo -n "$model → "
  curl -s -X POST http://localhost:9110/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"Hola\"}],\"stream\":false}" \
    | python3 -c "import sys,json;d=json.load(sys.stdin);md=d.get('proxy_metadata',{});print(f'physical={md.get(\"physical_model\",\"?\")} fallback={md.get(\"fallback_applied\")}')"
done
```

---

## 3. Model Aliases — Compatibilidad con Clientes Externos

**Como** usuario que migra desde otro proveedor
**Quiero** usar nombres de modelo conocidos (`gpt-4o`, `o3`, `claude-haiku-3-5`, etc.)
**Para** no tener que cambiar mi código cliente

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

## 4. Streaming SSE

**Como** usuario
**Quiero** recibir respuestas en streaming
**Para** ver el contenido mientras se genera

```bash
curl -s -X POST http://localhost:9110/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"normal","messages":[{"role":"user","content":"Cuenta hasta 5"}],"stream":true}' \
  | head -15
```

**Criterios:**
- Chunks SSE: `data: {...}\n\n`
- Termina con `data: [DONE]`
- Último chunk antes de `[DONE]` contiene `proxy_metadata`

---

## 5. KeyVault — Protección de Secrets

**Como** usuario
**Quiero** que mis API keys sean interceptadas antes de llegar al LLM
**Para** que el modelo nunca vea mis credenciales

```bash
curl -s -X POST http://localhost:9110/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"normal","messages":[{"role":"user","content":"Mi key es sk-proj-abc123def456ghi789jkl012"}],"conversation_id":"conv-keyvault","stream":false}' \
  | python3 -c "import sys,json;d=json.load(sys.stdin);c=d.get('choices',[{}])[0].get('message',{}).get('content','');print('respondió:', bool(c))"
```

**Criterios:**
- El LLM recibe el mensaje con `sk-proj-...` reemplazado por `[KEYVAULT:hash]`
- El cliente ve la respuesta con el secret reinyectado

---

## 6. Fallback Automático

**Como** sistema
**Quiero** que cuando el modelo primario falle (timeout, rate limit, error)
**Para** pasar automáticamente al siguiente físico en la lista

```bash
# Verificar qué physical model se usó realmente (proxy_metadata)
curl -s -X POST http://localhost:9110/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"normal","messages":[{"role":"user","content":"test"}],"stream":false}' \
  | python3 -c "import sys,json;d=json.load(sys.stdin);md=d.get('proxy_metadata',{});print(f'physical={md.get(\"physical_model\")} fallback={md.get(\"fallback_applied\")}')"
```

**Criterios:**
- Todos usan `sequential` fallback
- El único con `by_context_window` es `compactador`
- Si todos fallan → `502 ALL_MODELS_FAILED`

---

## 7. Visión — Envío de Imágenes

**Como** usuario
**Quiero** enviar imágenes al modelo `vision`
**Para** OCR, análisis de diagramas, UI, screenshots

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

## 8. Modelo Gratuito — Nemotron vía OpenRouter

**Como** usuario
**Quiero** usar `normal-gratis`
**Para** coding y tareas generales sin costo

```bash
curl -s -X POST http://localhost:9110/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"normal-gratis","messages":[{"role":"user","content":"Hola gratis"}],"stream":false}' \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('choices',[{}])[0].get('message',{}).get('content','')[:100])"
```

---

## 9. Compactación Explícita

**Como** usuario
**Quiero** compactar una conversación cuando el input supera el umbral
**Para** reducir tokens y continuar

```bash
# Crear conversación con varios turns
CONV_ID=$(python3 -c "import uuid; print(uuid.uuid4())")
for i in 1 2 3; do
  curl -s -X POST http://localhost:9110/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"normal\",\"messages\":[{\"role\":\"user\",\"content\":\"turn $i\"}],\"conversation_id\":\"$CONV_ID\",\"stream\":false}" > /dev/null
done

# Compactar
curl -s -X POST "http://localhost:9110/conversations/$CONV_ID/compact" \
  -H "Content-Type: application/json" \
  -d '{}' | python3 -m json.tool
```

---

## 10. Audit Log

**Como** usuario
**Quiero** ver el log de eventos de una conversación
**Para** auditar qué decisiones tomó el proxy

```bash
CONV_ID=$(python3 -c "import uuid; print(uuid.uuid4())")
curl -s -X POST http://localhost:9110/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"normal\",\"messages\":[{\"role\":\"user\",\"content\":\"Hola\"}],\"conversation_id\":\"$CONV_ID\",\"stream\":false}" > /dev/null

curl -s "http://localhost:9110/conversations/$CONV_ID/audit-log" \
  | python3 -c "import sys,json;d=json.load(sys.stdin);[print(f'{e.get(\"event_type\")}: {json.dumps(e.get(\"details\",{}))}') for e in d.get('events',[])]"
```

---

## 11. Error — Input Excede Umbral

**Como** sistema
**Quiero** devolver un error claro cuando el input supera el límite
**Para** que el usuario sepa que debe compactar

```bash
curl -s -w "\nHTTP: %{http_code}" -X POST http://localhost:9110/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"normal\",\"messages\":[{\"role\":\"user\",\"content\":\"$(python3 -c \"print('x'*200000)\")\"}],\"stream\":false}"
```

**Resultado esperado:** `HTTP: 400` con error `INPUT_EXCEEDS_THRESHOLD`

---

## 12. Restricción — Imagen en Modelo Sin Visión

**Como** sistema
**Quiero** rechazar imágenes enviadas a un modelo sin capacidad de visión
**Para** evitar erroes silenciosos del LLM

```bash
curl -s -w "\nHTTP: %{http_code}" -X POST http://localhost:9110/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"normal",
    "messages":[{"role":"user","content":[
      {"type":"text","text":"Hola"},
      {"type":"image_url","image_url":{"url":"https://example.com/img.png"}}
    ]}],
    "stream":false
  }'
```

**Resultado esperado:** `HTTP: 400` con error `IMAGES_NOT_SUPPORTED_BY_PSEUDO_MODEL`
