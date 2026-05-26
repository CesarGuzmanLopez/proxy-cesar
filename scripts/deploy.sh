#!/usr/bin/env bash
# scripts/deploy.sh — Idempotent deployment script for proxy-cesar
#
# Runs on the target server (plata) during CI/CD.
# Designed to be safe on first run and subsequent runs.
#
# Environment variables (provided by GitHub Actions or manually):
#   PROXY_PORT          — service port (default: 9110)
#   PROXY_API_KEY       — Bearer token for API access
#   CORS_ORIGINS        — Allowed CORS origins
#   ANTHROPIC_API_KEY
#   DEEPSEEK_API_KEY
#   GOOGLE_API_KEY
#   GROQ_API_KEY
#   OPENROUTER_API_KEY
#   ZHIPUAI_API_KEY
#   ZAI_API_KEY
#   DATABASE_URL        — default: sqlite+aiosqlite:///./proxy.db
#   VALKEY_URL          — default: valkey://localhost:6379
#   KEYCLAW_VERSION     — KeyClaw version to install (default: latest)

set -euo pipefail

REPO_DIR="${REPO_DIR:-/opt/proxy-cesar}"
GIT_REPO_URL="${GIT_REPO_URL:-https://github.com/CesarGuzmanLopez/proxy-cesar.git}"
BRANCH="${BRANCH:-main}"

# ── 1. Clone or pull ──────────────────────────────────────────────────
if [ ! -d "$REPO_DIR" ]; then
    echo "[deploy] First run — cloning repository..."
    git clone --branch "$BRANCH" --depth 1 "$GIT_REPO_URL" "$REPO_DIR"
else
    echo "[deploy] Pulling latest changes..."
    cd "$REPO_DIR"
    git fetch origin "$BRANCH" --depth 1
    git reset --hard "origin/$BRANCH"
fi

cd "$REPO_DIR/proxy"

# ── 2. Python virtual environment ─────────────────────────────────────
if [ ! -d ".venv" ]; then
    echo "[deploy] Creating virtual environment..."
    python3.14 -m venv .venv --clear
fi

source .venv/bin/activate

echo "[deploy] Installing/updating dependencies..."
pip install --quiet --upgrade pip setuptools wheel
pip install --quiet ".[dev]"

# ── 3. Write .env from environment variables ──────────────────────────
cat > .env <<-EOF
# Proxy (Sprint 8)
PROXY_PORT=${PROXY_PORT:-9110}
PROXY_API_KEY=${PROXY_API_KEY:-}
CORS_ORIGINS=${CORS_ORIGINS:-https://chat.guzman-lopez.com,vscode-webview://*}

# Database
DATABASE_URL=${DATABASE_URL:-sqlite+aiosqlite:///./proxy.db}

# Cache
VALKEY_URL=${VALKEY_URL:-valkey://localhost:6379}

# Provider API keys (NEVER leave the server)
ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-}
DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY:-}
GOOGLE_API_KEY=${GOOGLE_API_KEY:-}
GROQ_API_KEY=${GROQ_API_KEY:-}
OPENROUTER_API_KEY=${OPENROUTER_API_KEY:-}
ZHIPUAI_API_KEY=${ZHIPUAI_API_KEY:-}
ZAI_API_KEY=${ZAI_API_KEY:-}
EOF

chmod 600 .env

# ── 4. KeyClaw — local MITM proxy ─────────────────────────────────────
if ! command -v keyclaw &>/dev/null; then
    echo "[deploy] Installing KeyClaw..."
    if command -v cargo &>/dev/null; then
        cargo install keyclaw
    else
        echo "[deploy] ERROR: cargo not found. Install Rust first: https://rustup.rs"
        exit 1
    fi
fi

KEYCLAW_HOME="/home/proxy/.keyclaw"
if [ ! -f "$KEYCLAW_HOME/ca.crt" ]; then
    echo "[deploy] Initialising KeyClaw..."
    keyclaw init
fi

# Create combined CA bundle (system CAs + KeyClaw CA)
if command -v update-ca-certificates &>/dev/null; then
    # Debian/Ubuntu — add to system trust store
    cp "$KEYCLAW_HOME/ca.crt" /usr/local/share/ca-certificates/keyclaw-ca.crt
    update-ca-certificates
elif [ -f /etc/ssl/cert.pem ]; then
    # Arch / combined-bundle approach
    cat /etc/ssl/cert.pem "$KEYCLAW_HOME/ca.crt" > "$KEYCLAW_HOME/combined-ca.pem"
fi

# Systemd service for KeyClaw
KEYCLAW_SERVICE_FILE="/etc/systemd/system/keyclaw.service"
if [ ! -f "$KEYCLAW_SERVICE_FILE" ]; then
    echo "[deploy] Creating KeyClaw systemd service..."
    cat > "$KEYCLAW_SERVICE_FILE" <<-EOF
[Unit]
Description=KeyClaw — Local MITM Proxy that strips secrets from LLM traffic
After=network.target
Wants=network.target

[Service]
Type=simple
User=proxy
Restart=always
RestartSec=5
ExecStart=/home/proxy/.cargo/bin/keyclaw proxy start --foreground
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
    systemctl daemon-reload
    systemctl enable keyclaw.service
    echo "[deploy] KeyClaw service created and enabled"
fi

# ── 6. Systemd service ────────────────────────────────────────────────
SERVICE_NAME="proxy-cesar"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

echo "[deploy] Updating systemd service..."
cat > "$SERVICE_FILE" <<-EOF
[Unit]
Description=Proxy Cesar — Deterministic Multi-Model LLM Proxy
After=network.target keyclaw.service
Wants=network-online.target

[Service]
Type=simple
User=proxy
WorkingDirectory=${REPO_DIR}/proxy
EnvironmentFile=${REPO_DIR}/proxy/.env
ExecStart=${REPO_DIR}/proxy/.venv/bin/uvicorn src.main:app --host 127.0.0.1 --port ${PROXY_PORT:-9110}
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable "$SERVICE_NAME" || true

# ── 7. arq worker service ─────────────────────────────────────────────
ARQ_SERVICE_NAME="proxy-cesar-arq"
ARQ_SERVICE_FILE="/etc/systemd/system/${ARQ_SERVICE_NAME}.service"

if [ ! -f "$ARQ_SERVICE_FILE" ]; then
    echo "[deploy] Creating arq worker service..."
    cat > "$ARQ_SERVICE_FILE" <<-EOF
[Unit]
Description=Proxy Cesar arq Worker (Async Compaction)
After=network-online.target valkey.service
Wants=network-online.target

[Service]
Type=simple
User=proxy
WorkingDirectory=${REPO_DIR}/proxy
EnvironmentFile=${REPO_DIR}/proxy/.env
ExecStart=${REPO_DIR}/proxy/.venv/bin/arq src.tasks.arq_app.WorkerSettings
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable "$ARQ_SERVICE_NAME" || true
    echo "[deploy] arq worker service created and enabled"
fi

# ── 8. Start KeyClaw (before proxy) ────────────────────────────────────
echo "[deploy] Starting KeyClaw..."
systemctl start keyclaw.service || true
echo "[deploy] keyclaw: $(systemctl is-active keyclaw.service)"

# ── 9. Restart services ───────────────────────────────────────────────
echo "[deploy] Restarting proxy-cesar..."
systemctl restart "$SERVICE_NAME"
echo "[deploy] proxy-cesar: $(systemctl is-active "$SERVICE_NAME")"

if systemctl is-active --quiet "$ARQ_SERVICE_NAME" 2>/dev/null; then
    systemctl restart "$ARQ_SERVICE_NAME"
    echo "[deploy] proxy-cesar-arq: $(systemctl is-active "$ARQ_SERVICE_NAME")"
fi

echo "[deploy] Deploy complete"
