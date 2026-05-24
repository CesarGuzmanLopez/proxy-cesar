# proxy-cesar

**Deterministic multi-model LLM proxy.** Transparent HTTP proxy between LLM clients (Continue, LibreChat, OpenCode) and multiple providers. Exposes abstract pseudo-models that map to concrete physical models with automatic fallback, content compatibility validation, tool normalization, and context compaction.

Standard OpenAI-compatible API (`/v1/chat/completions`, `/v1/models`).

---

## Repository Structure

```
proxy-cesar/
├── proxy/                  # Python application (FastAPI + LiteLLM)
│   ├── src/                # Source code
│   ├── tests/              # Test suite (178+ tests)
│   ├── pseudo_models.yaml  # Model definitions
│   ├── pyproject.toml      # Dependencies and build config
│   └── README.md           # Full technical documentation
├── .github/workflows/      # CI/CD: auto-deploy to production
│   └── deploy.yml          # Deploy on push to main
├── scripts/
│   └── deploy.sh           # Idempotent deployment script
└── sprints/                # Design docs and sprint plans
```

---

## Production

**URL:** `https://chat.guzman-lopez.com`

Runs on server `plata` (74.208.235.42) via systemd service `proxy-cesar`. Nginx reverse-proxies with HTTPS/SSL.

### Deploy

```bash
ssh proxy@plata
cd /opt/proxy-cesar && git pull
cd proxy
source .venv/bin/activate
pip install ".[dev]"
sudo systemctl restart proxy-cesar
```

### First-time setup (new server)

```bash
# Clone
git clone --depth 1 https://github.com/CesarGuzmanLopez/proxy-cesar.git /opt/proxy-cesar

# Create venv
cd /opt/proxy-cesar/proxy
python3.14 -m venv .venv
source .venv/bin/activate
pip install ".[dev]"

# Environment
cp .env.example .env   # Fill in all API keys

# Systemd
sudo tee /etc/systemd/system/proxy-cesar.service << SERVICEEOF
[Unit]
Description=Proxy Cesar
After=network-online.target
[Service]
Type=simple
User=proxy
WorkingDirectory=/opt/proxy-cesar/proxy
ExecStart=/opt/proxy-cesar/proxy/.venv/bin/python -m src.main
Restart=always
RestartSec=5
[Install]
WantedBy=multi-user.target
SERVICEEOF

sudo systemctl daemon-reload
sudo systemctl enable --now proxy-cesar
```

### Server rules (see `/root/README-plata.md`)

- Never use `pip` outside a virtualenv (`pip.conf` enforces this)
- Never touch `/usr/lib/python3/dist-packages/` (shared system packages)
- All projects use isolated venvs
- `certbot` must remain functional for SSL renewal

---

## Quick Start (local)

```bash
cd proxy
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # Add your API keys
python -m src.main     # Starts on http://localhost:9110
```

See `proxy/README.md` for full documentation: architecture, API reference, error dictionary, data model, and development guide.

---

## Dependencies

Managed via `pyproject.toml`. Key packages:

| Package | Version | Notes |
|---|---|---|
| Python | >=3.12,<3.15 | Server runs 3.14 |
| FastAPI | >=0.115,<0.137 | HTTP framework |
| LiteLLM | >=1.83.7,<1.83.8 | Multi-provider LLM client (pinned due to Python 3.14) |
| SQLModel | >=0.0.38 | ORM |
| Valkey | >=5.0,<7.0 | Cache + affinity store |
| Pydantic | >=2.11,<3.0 | Validation (pinned at 2.12.5 by LiteLLM) |

---

## License

MIT
