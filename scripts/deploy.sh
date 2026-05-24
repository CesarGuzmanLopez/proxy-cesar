#!/usr/bin/env bash
# scripts/deploy.sh — Idempotent deployment script for proxy-cesar
#
# Runs on the target server (plata) during CI/CD.
# Designed to be safe on first run and subsequent runs.
#
# Environment variables (provided by GitHub Actions or manually):
#   PROXY_PORT          — service port (default: 9110)
#   ANTHROPIC_API_KEY
#   DEEPSEEK_API_KEY
#   GOOGLE_API_KEY
#   GROQ_API_KEY
#   OPENROUTER_API_KEY
#   ZHIPUAI_API_KEY
#   ZAI_API_KEY
#   DATABASE_URL        — default: sqlite+aiosqlite:///./proxy.db
#   VALKEY_URL          — default: valkey://localhost:6379
#   GIT_REPO_URL        — default: https://github.com/CesarGuzmanLopez/proxy-cesar.git

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
    python3 -m venv .venv
fi

source .venv/bin/activate

echo "[deploy] Installing/updating dependencies..."
pip install --quiet --upgrade pip setuptools wheel
pip install --quiet ".[dev]"

# ── 3. Write .env from environment variables ──────────────────────────
cat > .env <<-EOF
# Proxy
PROXY_PORT=${PROXY_PORT:-9110}

# Database
DATABASE_URL=${DATABASE_URL:-sqlite+aiosqlite:///./proxy.db}

# Cache
VALKEY_URL=${VALKEY_URL:-valkey://localhost:6379}

# Provider API keys
ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-}
DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY:-}
GOOGLE_API_KEY=${GOOGLE_API_KEY:-}
GROQ_API_KEY=${GROQ_API_KEY:-}
OPENROUTER_API_KEY=${OPENROUTER_API_KEY:-}
ZHIPUAI_API_KEY=${ZHIPUAI_API_KEY:-}
ZAI_API_KEY=${ZAI_API_KEY:-}
EOF

chmod 600 .env

# ── 4. Systemd service ────────────────────────────────────────────────
SERVICE_NAME="proxy-cesar"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

if [ ! -f "$SERVICE_FILE" ]; then
    echo "[deploy] Creating systemd service..."
    cat > "$SERVICE_FILE" <<-EOF
[Unit]
Description=Proxy Cesar — Deterministic Multi-Model LLM Proxy
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=proxy
WorkingDirectory=${REPO_DIR}/proxy
ExecStart=${REPO_DIR}/proxy/.venv/bin/python -m src.main
Restart=always
RestartSec=5
Environment=PROXY_PORT=${PROXY_PORT:-9110}

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable "$SERVICE_NAME"
fi

# ── 5. Restart service ────────────────────────────────────────────────
echo "[deploy] Restarting service..."
systemctl restart "$SERVICE_NAME"
echo "[deploy] Done — $(systemctl is-active "$SERVICE_NAME")"
