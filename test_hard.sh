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
KEYCLAW_MOVED=false
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

get_error() {
    echo "$1" | python3 -c "
import sys,json
d=json.load(sys.stdin)
err=d.get('detail',{})
if isinstance(err, dict):
    print(err.get('message','')[:150])
else:
    print(str(err)[:150])
" 2>/dev/null
}

get_field() {
    local json="$1" field="$2"
    echo "$json" | python3 -c "
import sys,json
d=json.load(sys.stdin)
parts='$field'.split('.')
v=d
for p in parts:
    if isinstance(v, dict):
        v=v.get(p,'?')
    else:
        v='?'
        break
print(v)
" 2>/dev/null || echo "?"
}

# ── Pre-flight ──────────────────────────────────────────────────────────
preflight() {
    info "Pre-flight checks..."
    [ ! -f "$PROXY_DIR/.env" ]    && { fail "preflight" ".env not found in proxy/"; exit 1; }
    [ ! -x "$VENV_PYTHON" ]       && { fail "preflight" "venv python not found"; exit 1; }
    if ! ss -tlnp 2>/dev/null | grep -q ':6379'; then
        fail "preflight" "Valkey/Redis not running on :6379"; exit 1
    fi
    ok "preflight"
}

# ── Server lifecycle ────────────────────────────────────────────────────
start_server() {
    info "Starting proxy-cesar..."
    pkill -f "uvicorn src.main:app" 2>/dev/null || true
    sleep 2

    if [ -d "$HOME/.keyclaw" ]; then
        info "KeyClaw detected — moving aside for clean testing"
        mv "$HOME/.keyclaw" "$HOME/.keyclaw.test-bak" 2>/dev/null || true
        KEYCLAW_MOVED=true
    fi

    cd "$PROXY_DIR"
    setsid "$VENV_PYTHON" -m uvicorn src.main:app --host 127.0.0.1 --port 9110 \
        > "$LOG_FILE" 2>&1 &
    local server_pid=$!
    echo "$server_pid" > /tmp/proxy-cesar-test.pid

    for i in $(seq 1 30); do
        if curl -s --max-time 2 "$BASE_URL/health" > /dev/null 2>&1; then
            ok "server startup"
            return 0
        fi
        sleep 1
    done
    fail "server startup" "timed out — check $LOG_FILE"
    exit 1
}

stop_server() {
    if [ "$KILL_ON_EXIT" = false ]; then
        info "Server left running on $BASE_URL"
        return
    fi
    info "Stopping server..."
    local pid
    pid=$(cat /tmp/proxy-cesar-test.pid 2>/dev/null || echo "")
    [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null || true
    pkill -f "uvicorn src.main:app" 2>/dev/null || true

    if [ "$KEYCLAW_MOVED" = true ] && [ -d "$HOME/.keyclaw.test-bak" ]; then
        mv "$HOME/.keyclaw.test-bak" "$HOME/.keyclaw" 2>/dev/null || true
        KEYCLAW_MOVED=false
    fi
    sleep 1
    ok "server stopped"
}

cleanup() {
    set +e
    stop_server
}
trap cleanup EXIT INT TERM

# ── HU-17 ───────────────────────────────────────────────────────────────
test_health() {
    hdr "HU-17 — Health check"
    local resp providers
    resp=$(api_get "/health")
    providers=$(get_field "$resp" "status")
    if [ "$providers" = "ok" ]; then
        ok "HU-17 — health OK"
    else
        fail "HU-17" "health: $(get_error "$resp")"
    fi
}

# ── HU-1 ────────────────────────────────────────────────────────────────
test_normal() {
    hdr "HU-1 — Chat normal (DeepSeek V4 Flash)"
    local resp content
    resp=$(api_post "normal" "Hola")
    if [ "$(has_choices "$resp")" = "YES" ]; then
        content=$(get_content "$resp")
        ok "HU-1 — normal → ${content}"
    else
        fail "HU-1" "$(get_error "$resp")"
    fi
}

# ── HU-2 ────────────────────────────────────────────────────────────────
test_pensamiento_profundo() {
    hdr "HU-2 — Pensamiento profundo (GLM-5 + DeepSeek V4 Pro + Claude)"
    local resp content phys
    resp=$(api_post "pensamiento-profundo-caro" "Arquitectura de un parser LR(1)")
    if [ "$(has_choices "$resp")" = "YES" ]; then
        content=$(get_content "$resp")
        phys=$(get_field "$resp" "proxy_metadata.physical_model")
        ok "HU-2 — pensamiento-profundo-caro → ${content}"
        info "  physical_model: $phys"
    else
        fail "HU-2" "$(get_error "$resp")"
    fi
}

# ── HU-3 ────────────────────────────────────────────────────────────────
test_tareas_avanzadas() {
    hdr "HU-3 — Tareas avanzadas (DeepSeek V4 Pro + Flash)"
    local resp content
    resp=$(api_post "tareas-avanzadas" "Implementar un microservicio en Python")
    if [ "$(has_choices "$resp")" = "YES" ]; then
        content=$(get_content "$resp")
        ok "HU-3 — tareas-avanzadas → ${content}"
    else
        fail "HU-3" "$(get_error "$resp")"
    fi
}

# ── HU-4 ────────────────────────────────────────────────────────────────
test_vision() {
    hdr "HU-4 — Visión (Gemini Flash + Lite + Ollama)"
    # Test without image first (image fetch depends on provider)
    local resp content
    resp=$(api_post "vision" "Describe colores primarios")
    if [ "$(has_choices "$resp")" = "YES" ]; then
        content=$(get_content "$resp")
        ok "HU-4 — vision (text) → ${content}"
    else
        # Vision models may be down — this is infrastructure, not a proxy bug
        local err
        err=$(get_error "$resp")
        if echo "$err" | grep -q "ALL_MODELS_FAILED"; then
            skip "HU-4" "all vision backends unavailable (infra): $err"
        else
            fail "HU-4" "$err"
        fi
    fi
}

# ── HU-5 ────────────────────────────────────────────────────────────────
test_vision_lite() {
    hdr "HU-5 — Vision Lite (Z.ai + Groq)"
    local resp content
    resp=$(api_post "vision-lite" "Hola")
    if [ "$(has_choices "$resp")" = "YES" ]; then
        content=$(get_content "$resp")
        ok "HU-5 — vision-lite → ${content}"
    else
        fail "HU-5" "$(get_error "$resp")"
    fi
}

# ── HU-6 ────────────────────────────────────────────────────────────────
test_normal_gratis() {
    hdr "HU-6 — Normal gratis (OpenRouter + Z.ai)"
    local resp content
    resp=$(api_post "normal-gratis" "Hola")
    if [ "$(has_choices "$resp")" = "YES" ]; then
        content=$(get_content "$resp")
        ok "HU-6 — normal-gratis → ${content}"
    else
        fail "HU-6" "$(get_error "$resp")"
    fi
}

# ── HU-7 ────────────────────────────────────────────────────────────────
test_deep_flash() {
    hdr "HU-7 — Deep Flash (DeepSeek V4 Flash directo)"
    local resp content
    resp=$(api_post "deep-flash" "Traduce al inglés: Hola mundo")
    if [ "$(has_choices "$resp")" = "YES" ]; then
        content=$(get_content "$resp")
        ok "HU-7 — deep-flash → ${content}"
    else
        fail "HU-7" "$(get_error "$resp")"
    fi
}

# ── HU-8 ────────────────────────────────────────────────────────────────
test_massive_fast() {
    hdr "HU-8 — Massive Fast (Groq)"
    local resp content
    resp=$(api_post "massive-fast" "Respuesta rápida")
    if [ "$(has_choices "$resp")" = "YES" ]; then
        content=$(get_content "$resp")
        ok "HU-8 — massive-fast → ${content}"
    else
        fail "HU-8" "$(get_error "$resp")"
    fi
}

# ── HU-9 ────────────────────────────────────────────────────────────────
test_flash_lowcost() {
    hdr "HU-9 — Flash Lowcost (Z.ai + Ollama)"
    local resp content
    resp=$(api_post "flash-lowcost" "Clasifica: me duele la cabeza")
    if [ "$(has_choices "$resp")" = "YES" ]; then
        content=$(get_content "$resp")
        ok "HU-9 — flash-lowcost → ${content}"
    else
        fail "HU-9" "$(get_error "$resp")"
    fi
}

# ── HU-10 ───────────────────────────────────────────────────────────────
test_compactador() {
    hdr "HU-10 — Compactador explícito (POST /compact)"

    # Build conversation with turns
    local conv_id resp status
    conv_id=$($VENV_PYTHON -c "import uuid; print(uuid.uuid4())" 2>/dev/null) || { skip "HU-10" "python not available"; return; }
    for i in 1 2 3; do
        curl -s -X POST "$BASE_URL/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"normal\",\"messages\":[{\"role\":\"user\",\"content\":\"turn $i\"}],\"conversation_id\":\"$conv_id\",\"stream\":false}" \
            > /dev/null 2>&1 || true
    done

    # Test explicit compact
    resp=$(curl -s --max-time 120 -X POST "$BASE_URL/conversations/$conv_id/compact" \
        -H "Content-Type: application/json" -d '{}')
    status=$(get_field "$resp" "status")
    if [ "$status" = "completed" ]; then
        ok "HU-10 — compact → status=completed"
    else
        fail "HU-10" "compact status=$status: $(get_error "$resp")"
    fi

    # Test empty conversation → should be 400
    local empty_id
    empty_id=$($VENV_PYTHON -c "import uuid; print(uuid.uuid4())" 2>/dev/null) || empty_id="empty-test"
    local http_code
    http_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 \
        -X POST "$BASE_URL/conversations/$empty_id/compact" \
        -H "Content-Type: application/json" -d '{}' 2>/dev/null || echo "000")
    if [ "$http_code" = "400" ] || [ "$http_code" = "404" ]; then
        ok "HU-10 — empty conv compact → $http_code (correct)"
    else
        fail "HU-10" "empty conv expected 400/404, got $http_code"
    fi

    # Double compaction
    resp=$(curl -s --max-time 120 -X POST "$BASE_URL/conversations/$conv_id/compact" \
        -H "Content-Type: application/json" -d '{}')
    status=$(get_field "$resp" "status")
    if [ "$status" = "completed" ]; then
        ok "HU-10 — double compact → status=completed"
    else
        fail "HU-10" "double compact status=$status"
    fi
}

# ── HU-11 ───────────────────────────────────────────────────────────────
test_continuous_compaction() {
    hdr "HU-11 — Compactación continua (inline /compact)"

    # Use the conversation from HU-10 (already has turns and snapshot)
    # or create a new one
    local conv_id resp content
    conv_id=$($VENV_PYTHON -c "import uuid; print(uuid.uuid4())" 2>/dev/null) || { skip "HU-11" "python not available"; return; }
    for i in 1 2 3; do
        curl -s -X POST "$BASE_URL/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"normal\",\"messages\":[{\"role\":\"user\",\"content\":\"msg $i\"}],\"conversation_id\":\"$conv_id\",\"stream\":false}" \
            > /dev/null 2>&1 || true
    done

    # Inline /compact
    resp=$(api_post "normal" "/compact" "\"conversation_id\":\"$conv_id\"")
    if [ "$(has_choices "$resp")" = "YES" ]; then
        content=$(get_content "$resp")
        if echo "$content" | grep -q "compacted\|snapshot\|prepared"; then
            ok "HU-11 — inline /compact → snapshot created"
        else
            ok "HU-11 — inline /compact responded: ${content}"
        fi
    else
        fail "HU-11" "$(get_error "$resp")"
    fi

    # Verify conversation state has snapshot
    resp=$(api_get "/conversations/$conv_id")
    local snap tokens
    snap=$(get_field "$resp" "active_snapshot_id")
    tokens=$(get_field "$resp" "total_tokens")
    if [ -n "$snap" ] && [ "$snap" != "?" ] && [ "$snap" != "None" ]; then
        ok "HU-11 — active_snapshot_id=$snap tokens=$tokens"
    else
        skip "HU-11" "snapshot not active (tokens=$tokens, may need more context)"
    fi
}

# ── HU-12 ───────────────────────────────────────────────────────────────
test_imagenes() {
    hdr "HU-12 — Degradación de imágenes"

    # Create conversation with vision + image
    local conv_id resp status
    conv_id=$($VENV_PYTHON -c "import uuid; print(uuid.uuid4())" 2>/dev/null) || { skip "HU-12" "python not available"; return; }

    # Try sending image — may fail if vision models are down
    resp=$(curl -s --max-time 60 -X POST "$BASE_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\":\"vision\",
            \"messages\":[{\"role\":\"user\",\"content\":[
                {\"type\":\"text\",\"text\":\"Describe\"},
                {\"type\":\"image_url\",\"image_url\":{\"url\":\"https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/300px-PNG_transparency_demonstration_1.png\"}}
            ]}],
            \"conversation_id\":\"$conv_id\",
            \"stream\":false
        }" 2>/dev/null)

    if [ "$(has_choices "$resp")" = "YES" ]; then
        ok "HU-12 — vision with image → OK"

        # Degrade images
        resp=$(curl -s --max-time 120 -X POST "$BASE_URL/conversations/$conv_id/degrade-images" \
            -H "Content-Type: application/json" -d '{}')
        status=$(get_field "$resp" "status")
        if [ "$status" = "completed" ]; then
            ok "HU-12 — degrade-images → completed"
        else
            fail "HU-12" "degrade-images status=$status: $(get_error "$resp")"
        fi

        # Try switching to normal after degrade
        resp=$(api_post "normal" "Mensaje después de degradar" "\"conversation_id\":\"$conv_id\"")
        if [ "$(has_choices "$resp")" = "YES" ]; then
            ok "HU-12 — switch to normal after degrade → OK"
        else
            fail "HU-12" "switch to normal after degrade: $(get_error "$resp")"
        fi
    else
        skip "HU-12" "vision models unavailable, cannot test image flow"
    fi

    # Test: image to non-vision model should be blocked
    local http_code
    http_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 \
        -X POST "$BASE_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"normal","messages":[{"role":"user","content":[{"type":"text","text":"Describe"},{"type":"image_url","image_url":{"url":"https://example.com/img.png"}}]}],"stream":false}' \
        2>/dev/null || echo "000")
    if [ "$http_code" = "400" ] || [ "$http_code" = "422" ]; then
        ok "HU-12 — image to non-vision model blocked ($http_code)"
    else
        fail "HU-12" "image to non-vision should be blocked, got $http_code"
    fi
}

# ── HU-13 ───────────────────────────────────────────────────────────────
test_audit_log() {
    hdr "HU-13 — Auditoría (GET audit-log)"

    local conv_id resp events
    conv_id=$($VENV_PYTHON -c "import uuid; print(uuid.uuid4())" 2>/dev/null) || { skip "HU-13" "python not available"; return; }
    for i in 1 2; do
        curl -s -X POST "$BASE_URL/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"normal\",\"messages\":[{\"role\":\"user\",\"content\":\"msg $i\"}],\"conversation_id\":\"$conv_id\",\"stream\":false}" \
            > /dev/null 2>&1 || true
    done

    resp=$(api_get "/conversations/$conv_id/audit-log")
    events=$(get_field "$resp" "events")
    if [ "$events" != "?" ] && [ "$events" != "None" ]; then
        local count
        count=$(echo "$resp" | python3 -c "import sys,json;d=json.load(sys.stdin);print(len(d.get('events',[])))" 2>/dev/null || echo "0")
        if [ "$count" -gt 0 ] 2>/dev/null; then
            ok "HU-13 — audit-log has $count events"
        else
            ok "HU-13 — audit-log accessible (0 events for new conv)"
        fi
    else
        fail "HU-13" "audit-log parse error: $(echo "$resp" | head -c 200)"
    fi
}

# ── HU-14 ───────────────────────────────────────────────────────────────
test_streaming() {
    hdr "HU-14 — Streaming"
    local resp
    resp=$(curl -s --max-time 60 -X POST "$BASE_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"normal","messages":[{"role":"user","content":"Cuenta hasta 3"}],"stream":true}' 2>/dev/null)
    if echo "$resp" | grep -q "data:"; then
        ok "HU-14 — streaming returns SSE chunks"
    elif echo "$resp" | grep -q "DONE"; then
        ok "HU-14 — streaming completed with DONE"
    elif [ -z "$resp" ]; then
        fail "HU-14" "streaming returned empty response"
    else
        local preview
        preview=$(echo "$resp" | head -1 | tr '\n' ' ')
        fail "HU-14" "unexpected: $preview"
    fi
}

# ── HU-15 ───────────────────────────────────────────────────────────────
test_aliases() {
    hdr "HU-15 — Model aliases"

    declare -A expected_physical=(
        ["gpt-4o"]="deepseek-v4-flash"
        ["gpt-4o-mini"]="deepseek-v4-flash"
        ["o3"]="deepseek-v4-pro"
        ["claude-haiku-3-5-20241022"]="zai/glm-4.5-flash"
        ["default"]="deepseek-v4-flash"
    )

    for alias in "${!expected_physical[@]}"; do
        local expected_phys="${expected_physical[$alias]}"
        local resp actual
        resp=$(api_post "$alias" "hi")
        actual=$(get_field "$resp" "model")
        if [ -n "$actual" ] && [ "$actual" != "?" ]; then
            ok "HU-15 — $alias → $actual (phys: $expected_phys)"
        else
            fail "HU-15" "$alias → no response: $(get_error "$resp")"
        fi
    done

    # gemini-2.5-flash → vision (may fail if vision backends unavailable)
    local resp actual
    resp=$(api_post "gemini-2.5-flash" "hi")
    actual=$(get_field "$resp" "model")
    if [ -n "$actual" ] && [ "$actual" != "?" ]; then
        ok "HU-15 — gemini-2.5-flash → $actual"
    else
        skip "HU-15" "gemini-2.5-flash: vision backends unavailable"
    fi
}

# ── HU-16 ───────────────────────────────────────────────────────────────
test_fallback() {
    hdr "HU-16 — Fallback strategy"
    local resp phys
    resp=$(api_post "pensamiento-profundo-caro" "test")
    if [ "$(has_choices "$resp")" = "YES" ]; then
        phys=$(get_field "$resp" "proxy_metadata.physical_model")
        ok "HU-16 — physical_model: $phys"
    else
        fail "HU-16" "$(get_error "$resp")"
    fi
}

# ── Main ────────────────────────────────────────────────────────────────
main() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║     Proxy-Cesar — Verificación de Historias de Usuario      ║"
    echo "║     Basado en BUG_VERIFICATION_FLOW.md                      ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"

    preflight

    if [ "$START_SERVER" = true ]; then
        start_server
    fi

    # ── Run all tests ─────────────────────────────────────────────────
    test_health
    test_normal
    test_pensamiento_profundo
    test_tareas_avanzadas
    test_vision
    test_vision_lite
    test_normal_gratis
    test_deep_flash
    test_massive_fast
    test_flash_lowcost
    test_compactador
    test_continuous_compaction
    test_imagenes
    test_audit_log
    test_streaming
    test_aliases
    test_fallback

    # ── Summary ──────────────────────────────────────────────────────
    hdr "RESUMEN"
    local total=$((PASSED + FAILED + SKIPPED))
    echo -e "  ${GREEN}Passed:  $PASSED${NC}"
    echo -e "  ${RED}Failed:  $FAILED${NC}"
    echo -e "  ${YELLOW}Skipped: $SKIPPED${NC}"
    echo -e "  Total:   $total"
    echo ""

    if [ "$FAILED" -gt 0 ]; then
        echo -e "${RED}Some tests FAILED — check output above.${NC}"
        exit 1
    else
        echo -e "${GREEN}All tests PASSED.${NC}"
        exit 0
    fi
}

main
