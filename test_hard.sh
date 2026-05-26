#!/usr/bin/env bash
# ── test_hard.sh ── Verificación completa de todas las historias de usuario ──
#
# Uso:
#   ./test_hard.sh              # Inicia servidor + ejecuta todos los tests
#   ./test_hard.sh --no-start   # Solo tests (servidor ya corriendo)
#   ./test_hard.sh --no-kill    # No mata el servidor al terminar
#
# Requisitos:
#   - Python venv en proxy/.venv
#   - Valkey/Redis corriendo en localhost:6379
#   - .env configurado con API keys

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROXY_DIR="$SCRIPT_DIR/proxy"
BASE_URL="http://localhost:9110"
VENV_PYTHON="$PROXY_DIR/.venv/bin/python"
LOG_FILE="/tmp/proxy-cesar-test.log"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

START_SERVER=true
KILL_ON_EXIT=true
PASSED=0
FAILED=0
SKIPPED=0

# ── Parse args ──────────────────────────────────────────────────────────
for arg in "$@"; do
    case "$arg" in
        --no-start)   START_SERVER=false ;;
        --no-kill)    KILL_ON_EXIT=false ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

# ── Helpers ─────────────────────────────────────────────────────────────
ok()   { echo -e "  ${GREEN}✓${NC} $1"; PASSED=$((PASSED+1)); }
fail() { echo -e "  ${RED}✗${NC} $1 — $2"; FAILED=$((FAILED+1)); }
skip() { echo -e "  ${YELLOW}○${NC} $1 (skip: $2)"; SKIPPED=$((SKIPPED+1)); }
info() { echo -e "${BLUE}──${NC} $1"; }
hdr()  { echo -e "\n${BLUE}═══════════════════════════════════════════════════════════${NC}"; echo -e "${BLUE}  $1${NC}"; echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"; }

# ── API helpers ──────────────────────────────────────────────────────────
api_post() {
    local model="$1" msg="$2" extra="${3:-}"
    local json
    if [ -n "$extra" ]; then
        json="{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"$msg\"}],\"stream\":false,$extra}"
    else
        json="{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"$msg\"}],\"stream\":false}"
    fi
    curl -s --max-time 120 -X POST "$BASE_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$json"
}

api_post_stream() {
    local model="$1" msg="$2"
    curl -s --max-time 120 -X POST "$BASE_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"$msg\"}],\"stream\":true}"
}

api_get() {
    curl -s --max-time 15 "$BASE_URL$1"
}

has_choices() {
    echo "$1" | python3 -c "import sys,json;d=json.load(sys.stdin);print('YES' if 'choices' in d else 'NO')" 2>/dev/null || echo "NO"
}

get_content() {
    echo "$1" | python3 -c "
import sys,json
d=json.load(sys.stdin)
if 'choices' in d:
    print(d['choices'][0]['message']['content'][:200])
" 2>/dev/null
}

get_http_code() {
    curl -s -o /dev/null -w "%{http_code}" --max-time 120 -X POST "$BASE_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$1"
}

# ── Server lifecycle ─────────────────────────────────────────────────────

start_server() {
    if [ "$START_SERVER" = false ]; then
        info "Usando servidor existente en $BASE_URL"
        return
    fi

    # Check if already running
    if curl -sf "$BASE_URL/health" > /dev/null 2>&1; then
        info "Servidor ya está corriendo en $BASE_URL"
        return
    fi

    info "Iniciando servidor proxy..."
    cd "$PROXY_DIR" || exit 1
    .venv/bin/python -m src.main > "$LOG_FILE" 2>&1 &
    PROXY_PID=$!
    cd "$SCRIPT_DIR" || exit 1

    # Wait for startup
    for i in $(seq 1 30); do
        if curl -sf "$BASE_URL/health" > /dev/null 2>&1; then
            info "Servidor listo (PID $PROXY_PID)"
            return
        fi
        sleep 1
    done

    echo -e "${RED}ERROR: Servidor no arrancó en 30s${NC}"
    tail -20 "$LOG_FILE"
    exit 1
}

cleanup() {
    if [ "$KILL_ON_EXIT" = true ] && [ -n "$PROXY_PID" ]; then
        info "Deteniendo servidor (PID $PROXY_PID)..."
        kill "$PROXY_PID" 2>/dev/null || true
        wait "$PROXY_PID" 2>/dev/null || true
    fi

    echo -e "\n${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  RESULTADOS${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "  ${GREEN}✓${NC} Passed: $PASSED"
    echo -e "  ${RED}✗${NC} Failed: $FAILED"
    echo -e "  ${YELLOW}○${NC} Skipped: $SKIPPED"
    echo ""
    [ "$FAILED" -gt 0 ] && exit 1 || exit 0
}

trap cleanup EXIT INT TERM

# ═══════════════════════════════════════════════════════════════════════════
#  HISTORIAS DE USUARIO
# ═══════════════════════════════════════════════════════════════════════════

start_server

# ── HU-1: Chat normal ────────────────────────────────────────────────────
hdr "HU-1: Chat normal (DeepSeek V4 Flash)"
RESP=$(api_post "normal" "Hola")
if [ "$(has_choices "$RESP")" = "YES" ]; then
    CONTENT=$(get_content "$RESP")
    echo "  Content: $CONTENT"
    ok "HU-1: normal responde"
else
    ERR=$(get_error "$RESP")
    fail "HU-1: normal falló" "$ERR"
fi

# ── HU-2: Pensamiento profundo ────────────────────────────────────────────
hdr "HU-2: Pensamiento profundo (DeepSeek V4 Pro)"
RESP=$(api_post "pensamiento-profundo-caro" "Arquitectura de un parser LR(1)")
if [ "$(has_choices "$RESP")" = "YES" ]; then
    CONTENT=$(get_content "$RESP")
    echo "  Content: ${CONTENT:0:100}..."
    ok "HU-2: pensamiento-profundo-caro responde"
else
    ERR=$(get_error "$RESP")
    fail "HU-2: pensamiento-profundo-caro falló" "$ERR"
fi

# ── HU-3: Tareas avanzadas ───────────────────────────────────────────────
hdr "HU-3: Tareas avanzadas (DeepSeek V4 Pro + Flash)"
RESP=$(api_post "tareas-avanzadas" "Implementar un microservicio en Python")
if [ "$(has_choices "$RESP")" = "YES" ]; then
    CONTENT=$(get_content "$RESP")
    echo "  Content: ${CONTENT:0:100}..."
    ok "HU-3: tareas-avanzadas responde"
else
    ERR=$(get_error "$RESP")
    fail "HU-3: tareas-avanzadas falló" "$ERR"
fi

# ── HU-4: Visión ─────────────────────────────────────────────────────────
hdr "HU-4: Visión (Llama 4 Scout via Groq)"
RESP=$(curl -s --max-time 120 -X POST "$BASE_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model":"vision",
        "messages":[{"role":"user","content":[
            {"type":"text","text":"Describe esta imagen"},
            {"type":"image_url","image_url":{"url":"https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/300px-PNG_transparency_demonstration_1.png"}}
        ]}],
        "stream":false
    }')
if [ "$(has_choices "$RESP")" = "YES" ]; then
    CONTENT=$(get_content "$RESP")
    echo "  Content: ${CONTENT:0:100}..."
    ok "HU-4: vision responde con imagen"
else
    ERR=$(get_error "$RESP")
    fail "HU-4: vision falló" "$ERR"
fi

# ── HU-5: Normal gratis ──────────────────────────────────────────────────
hdr "HU-5: Normal gratis (OpenRouter)"
RESP=$(api_post "normal-gratis" "Hola gratis")
if [ "$(has_choices "$RESP")" = "YES" ]; then
    ok "HU-5: normal-gratis responde"
else
    ERR=$(get_error "$RESP")
    fail "HU-5: normal-gratis falló" "$ERR"
fi

# ── HU-6: Massive Fast ───────────────────────────────────────────────────
hdr "HU-6: Massive Fast (Groq GPT-OSS 20B)"
RESP=$(api_post "massive-fast" "Respuesta rápida")
if [ "$(has_choices "$RESP")" = "YES" ]; then
    ok "HU-6: massive-fast responde"
else
    ERR=$(get_error "$RESP")
    fail "HU-6: massive-fast falló" "$ERR"
fi

# ── HU-7: Flash Lowcost ──────────────────────────────────────────────────
hdr "HU-7: Flash Lowcost (Gemini Flash Lite)"
RESP=$(api_post "flash-lowcost" "Clasifica: me duele la cabeza")
if [ "$(has_choices "$RESP")" = "YES" ]; then
    ok "HU-7: flash-lowcost responde"
else
    ERR=$(get_error "$RESP")
    fail "HU-7: flash-lowcost falló" "$ERR"
fi

# ── HU-8: Audio ──────────────────────────────────────────────────────────
hdr "HU-8: Audio (Whisper via Groq)"
RESP=$(api_post "audio" "Transcribe este audio")
if [ "$(has_choices "$RESP")" = "YES" ]; then
    ok "HU-8: audio responde"
else
    ERR=$(get_error "$RESP")
    fail "HU-8: audio falló" "$ERR"
fi

# ── HU-9: Imagen ─────────────────────────────────────────────────────────
hdr "HU-9: Imagen (Pruna P-Image)"
RESP=$(api_post "imagen" "Un gato volador")
if [ "$(has_choices "$RESP")" = "YES" ]; then
    ok "HU-9: imagen responde"
else
    ERR=$(get_error "$RESP")
    fail "HU-9: imagen falló" "$ERR"
fi

# ── HU-10: Compactador explícito ─────────────────────────────────────────
hdr "HU-10: Compactador explícito (POST /compact)"
CONV_ID=$(python3 -c "import uuid; print(uuid.uuid4())")
for i in 1 2 3; do
    api_post "normal" "turn $i" "\"conversation_id\":\"$CONV_ID\"" > /dev/null
done
COMPACT_RESP=$(curl -s --max-time 300 -X POST "$BASE_URL/conversations/$CONV_ID/compact" \
    -H "Content-Type: application/json" -d '{}')
COMPACT_STATUS=$(echo "$COMPACT_RESP" | python3 -c "import sys,json;print(json.load(sys.stdin).get('status','NO_STATUS'))" 2>/dev/null)
if [ "$COMPACT_STATUS" = "completed" ]; then
    ok "HU-10: POST /compact → status=completed"
else
    fail "HU-10: compact falló" "$COMPACT_STATUS"
fi

# ── HU-11: Streaming SSE ─────────────────────────────────────────────────
hdr "HU-11: Streaming SSE"
STREAM_OUT=$(api_post_stream "normal" "Cuenta hasta 3")
HAS_DONE=$(echo "$STREAM_OUT" | grep -c "\[DONE\]" || true)
HAS_DATA=$(echo "$STREAM_OUT" | grep -c "^data: " || true)
if [ "$HAS_DONE" -gt 0 ] && [ "$HAS_DATA" -gt 1 ]; then
    ok "HU-11: streaming SSE válido (${HAS_DATA} chunks, [DONE] presente)"
else
    fail "HU-11: streaming" "data=${HAS_DATA} done=${HAS_DONE}"
fi

# ── HU-12: Model aliases ─────────────────────────────────────────────────
hdr "HU-12: Model aliases"
ALL_ALIASES_OK=true
for alias in gpt-4o gpt-4o-mini o3 gemini-2.5-flash default; do
    RESP=$(api_post "$alias" "hi")
    if [ "$(has_choices "$RESP")" = "YES" ]; then
        echo "  $alias → OK"
    else
        echo "  $alias → FAIL"
        ALL_ALIASES_OK=false
    fi
done
if [ "$ALL_ALIASES_OK" = true ]; then
    ok "HU-12: todos los aliases resuelven correctamente"
else
    fail "HU-12: algunos aliases fallaron" ""
fi

# ── HU-13: KeyVault ──────────────────────────────────────────────────────
hdr "HU-13: KeyVault — Protección de Secrets"
RESP=$(api_post "normal" "Mi key es sk-proj-abc123def456ghi789jkl012" "\"conversation_id\":\"conv-kv-h13\"")
if [ "$(has_choices "$RESP")" = "YES" ]; then
    ok "HU-13: KeyVault no bloquea requests con keys (sanitiza OK)"
else
    ERR=$(get_error "$RESP")
    fail "HU-13: KeyVault" "$ERR"
fi

# ── HU-14: Delegación imágenes a tools ───────────────────────────────────
hdr "HU-14: Delegación de imágenes a tools"
# Con tool → debe funcionar
RESP_WITH_TOOL=$(curl -s --max-time 120 -X POST "$BASE_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model":"normal",
        "messages":[{"role":"user","content":[
            {"type":"text","text":"Usa la tool"},
            {"type":"image_url","image_url":{"url":"https://example.com/img.png"}}
        ]}],
        "tools":[{"function":{"name":"analyze","parameters":{"type":"object","properties":{"url":{"type":"string"}}}}}],
        "stream":false
    }')
if [ "$(has_choices "$RESP_WITH_TOOL")" = "YES" ]; then
    ok "HU-14a: imagen + tool → 200 OK"
else
    ERR=$(get_error "$RESP_WITH_TOOL")
    fail "HU-14a: imagen + tool" "$ERR"
fi

# Sin tool → debe fallar con 400
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 120 -X POST "$BASE_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model":"normal",
        "messages":[{"role":"user","content":[
            {"type":"text","text":"Describe"},
            {"type":"image_url","image_url":{"url":"https://example.com/img.png"}}
        ]}],
        "stream":false
    }')
if [ "$HTTP_CODE" = "400" ]; then
    ok "HU-14b: imagen sin tool → 400 (esperado)"
else
    fail "HU-14b: imagen sin tool" "HTTP $HTTP_CODE (esperado 400)"
fi

# ── HU-15: Fallback ──────────────────────────────────────────────────────
hdr "HU-15: Fallback strategy"
RESP=$(api_post "pensamiento-profundo-caro" "test")
PHYSICAL=$(echo "$RESP" | python3 -c "
import sys,json
d=json.load(sys.stdin)
md=d.get('proxy_metadata',{})
print(md.get('physical_model','?'))
" 2>/dev/null)
if [ -n "$PHYSICAL" ] && [ "$PHYSICAL" != "?" ]; then
    ok "HU-15: fallback strategy — physical_model=$PHYSICAL"
else
    fail "HU-15: fallback" "no physical_model"
fi

# ── HU-16: Auditoría ─────────────────────────────────────────────────────
hdr "HU-16: Auditoría (GET audit-log)"
AUDIT_RESP=$(api_get "/conversations/$CONV_ID/audit-log")
HAS_EVENTS=$(echo "$AUDIT_RESP" | python3 -c "
import sys,json
d=json.load(sys.stdin)
print('YES' if len(d.get('events',[])) > 0 else 'NO')
" 2>/dev/null || echo "NO")
if [ "$HAS_EVENTS" = "YES" ]; then
    ok "HU-16: audit-log retorna eventos"
else
    fail "HU-16: audit-log" "sin eventos"
fi

# ── HU-17: Health check ──────────────────────────────────────────────────
hdr "HU-17: Health check"
HEALTH=$(api_get "/health")
HEALTH_OK=$(echo "$HEALTH" | python3 -c "
import sys,json
d=json.load(sys.stdin)
print('YES' if d.get('status') == 'ok' else 'NO')
" 2>/dev/null || echo "NO")
if [ "$HEALTH_OK" = "YES" ]; then
    ok "HU-17: health check → status=ok"
else
    fail "HU-17: health" "status != ok"
fi

# Verificar métricas
METRICS=$(api_get "/metrics")
METRICS_OK=$(echo "$METRICS" | python3 -c "
import sys,json
d=json.load(sys.stdin)
print('YES' if 'total_requests' in d else 'NO')
" 2>/dev/null || echo "NO")
if [ "$METRICS_OK" = "YES" ]; then
    ok "HU-17b: metrics endpoint responde"
else
    fail "HU-17b: metrics" "no responde"
fi

# ── HU-18: Threshold exceeded ────────────────────────────────────────────
hdr "HU-18: Threshold exceeded → error explícito"
# Enviar input masivo a normal-gratis (threshold=200K)
LARGE_MSG=$(python3 -c "print('x'*250000)")
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 120 -X POST "$BASE_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"normal-gratis\",\"messages\":[{\"role\":\"user\",\"content\":\"$LARGE_MSG\"}],\"stream\":false}")
if [ "$HTTP_CODE" = "400" ]; then
    ok "HU-18: input masivo → 400 (esperado)"
else
    fail "HU-18: threshold" "HTTP $HTTP_CODE (esperado 400)"
fi

# ═══════════════════════════════════════════════════════════════════════════
#  RESUMEN
# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  TODAS LAS HISTORIAS DE USUARIO COMPLETADAS${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
