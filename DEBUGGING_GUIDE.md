# Guía de Debugging del Proxy

## Logging

### Configurar logs en archivo

#### Opción 1: Usar directorio estándar `/var/log/proxy-cesar`
```bash
# Crear directorio (requiere sudo)
sudo mkdir -p /var/log/proxy-cesar
sudo chmod 755 /var/log/proxy-cesar
sudo chown $USER:$USER /var/log/proxy-cesar

# Ejecutar proxy - logs se guardan automáticamente en /var/log/proxy-cesar/proxy.log
python -m src.main
```

#### Opción 2: Usar variable de entorno LOG_FILE
```bash
# Logs en ubicación personalizada
export LOG_FILE=/home/usuario/logs/proxy.log
python -m src.main

# O en una sola línea
LOG_FILE=/tmp/proxy.log python -m src.main
```

### Ver logs en tiempo real
```bash
# Opción 1: tail con formato JSON pretty-printed
tail -f /var/log/proxy-cesar/proxy.log | jq .

# Opción 2: grep para filtrar
tail -f /var/log/proxy-cesar/proxy.log | grep "stream_" | jq .

# Opción 3: buscar por conversation_id
tail -f /var/log/proxy-cesar/proxy.log | grep "conv=abc123" | jq .
```

---

## Debugging de Duplicación en Streaming

### Problema: Response llega dos veces

Este es el escenario a investigar:

```bash
# Client hace petición
curl -X POST http://localhost:9110/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "normal",
    "messages": [{"role": "user", "content": "hola"}],
    "stream": true
  }'

# Observación: ¿Se recibe response dos veces? ¿Se ve [DONE] duplicado?
```

### Pasos para depurar

#### 1. **Obtén el request_id y stream_id**
```bash
# En los logs busca
tail -f /var/log/proxy-cesar/proxy.log | jq '.| select(.message | contains("chat_request_received"))'

# Output esperado:
{
  "timestamp": "2026-05-27T19:45:30.123456+00:00",
  "level": "INFO",
  "message": "chat_request_received request_id=abc123 conv=xyz789 model=normal stream=true messages=1",
  "logger": "src.api.chat",
  ...
}
```

#### 2. **Rastrear el flujo de streaming**
```bash
# Busca por el request_id
tail -f /var/log/proxy-cesar/proxy.log | jq '. | select(.message | contains("abc123"))'

# Esperado: ves estos logs en orden
# 1. chat_request_received (request llega al proxy)
# 2. stream_gen_start (generador de streaming inicia)
# 3. stream_sending_done_marker (se envía [DONE])
# 4. stream_done_marker_sent (se confirmó [DONE])
```

#### 3. **Detectar [DONE] duplicado**
```bash
# Busca todos los [DONE] markers
tail -f /var/log/proxy-cesar/proxy.log | jq '. | select(.message | contains("done_marker"))'

# Si ves MÁS DE UNO con el mismo stream_id, hay duplicación
```

#### 4. **Ver el flujo completo de una conversación**
```bash
# Remplaza "xyz789" con el conversation_id real
conv_id="xyz789"
tail -f /var/log/proxy-cesar/proxy.log | jq '. | select(.message | contains("'$conv_id'"))'

# Salida esperada (en orden):
# [proxy_in] - Request entra al proxy
# [stream_gen_start] - Generador inicia
# [stream_sending_done_marker] - Marca [DONE]
# [stream_done_marker_sent] - Confirma envío
# [proxy_out] - Metadata de salida
```

---

## Campos útiles en los logs

| Campo | Significado | Ejemplo |
|---|---|---|
| `request_id` | ID único de la petición HTTP | `abc123` |
| `stream_id` | ID único de la sesión de streaming | `xyz789` |
| `conv` | ID corto de conversación | `a1b2c3d4` |
| `physical` | Modelo físico usado | `openai/gpt-4o-mini` |
| `message` | Evento que ocurrió | `stream_gen_start` |
| `level` | Nivel de log | `INFO`, `ERROR`, `WARNING` |
| `timestamp` | Cuándo ocurrió | ISO 8601 format |

---

## Flujo esperado: Petición Simple "hola"

```
TIME=T0   chat_request_received request_id=ABC conv=XYZ
          model=normal stream=true messages=1

T0+10ms   stream_gen_start stream_id=DEF conv=XYZ
          physical=openai/gpt-4o-mini models_available=1

T0+50ms   stream_sending_done_marker stream_id=DEF conv=XYZ
          physical=openai/gpt-4o-mini

T0+60ms   stream_done_marker_sent conv=XYZ

T0+70ms   chat_request_streaming_returned request_id=ABC conv=XYZ
```

### Si ves logs duplicados:
- Mismo `stream_id` aparece dos veces en `stream_gen_start` = BAD
- Mismo `stream_id` aparece dos veces en `stream_sending_done_marker` = BAD
- Dos `stream_id` diferentes = Normal (fallback a otro modelo)

---

## Monitoreo en Producción

### Setup para producción

```bash
# 1. Crear directorio de logs
sudo mkdir -p /var/log/proxy-cesar
sudo chown proxy-user:proxy-user /var/log/proxy-cesar

# 2. Setup logrotate para rotación automática
sudo tee /etc/logrotate.d/proxy-cesar <<EOF
/var/log/proxy-cesar/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0640 proxy-user proxy-user
}
EOF

# 3. Monitoreo con systemd
# Ver logs de la sesión actual:
journalctl -u proxy-cesar.service -f

# Ver logs de error:
journalctl -u proxy-cesar.service -p err
```

### Alertas útiles

```bash
# Alertar si hay [DONE] duplicado
tail -f /var/log/proxy-cesar/proxy.log | \
  jq '. | select(.message | contains("done_marker"))' | \
  awk '{if (prev==$0) print "DUPLICATE: " $0; prev=$0}'

# Alertar si hay errores de streaming
tail -f /var/log/proxy-cesar/proxy.log | \
  jq '. | select(.level == "ERROR" and .message | contains("stream"))'

# Alertar si una request toma más de 30 segundos
# (Se implementaría con timestamps)
```

---

## Ejemplo Real: Depuración de Issue

### Usuario reporta: "Response llega dos veces"

```bash
# 1. Obtener los logs
tail -1000 /var/log/proxy-cesar/proxy.log | \
  jq '. | select(.message | contains("stream_id"))' > stream_analysis.json

# 2. Buscar patrones sospechosos
cat stream_analysis.json | \
  jq -r '.message' | \
  sort | uniq -c | sort -rn

# 3. Output esperado (normal):
# 1 stream_gen_start
# 1 stream_sending_done_marker
# 1 stream_done_marker_sent

# 4. Output anómalo (problema):
# 2 stream_sending_done_marker  ← ¡Duplicado!
```

---

## Resumen

- **Logs van a**: `/var/log/proxy-cesar/proxy.log` (o `LOG_FILE` env var)
- **Formato**: JSON (parseable con `jq`)
- **Rastreo**: `request_id` + `stream_id` + `conv`
- **Debug**: Busca `stream_` en logs y observa flujo en orden
- **Rotación**: Automática a los 100MB

¡Con esto puedes depurar cualquier issue de streaming!
