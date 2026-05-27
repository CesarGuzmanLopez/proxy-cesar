"""Environment configuration via pydantic-settings.

Reads from .env file. Exact schema from sprint §5.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    proxy_host: str = "127.0.0.1"
    """Host to bind the server. Use '0.0.0.0' behind Nginx/Caddy reverse proxy."""
    proxy_port: int = 9110
    database_url: str = "sqlite+aiosqlite:///./proxy.db"
    valkey_url: str = "valkey://localhost:6379"

    # Sprint 8: Auth
    proxy_api_key: str = ""
    """Bearer token for API access. Empty = dev mode (no auth)."""

    # Sprint 8: CORS
    cors_origins: str = "http://localhost:3000"
    """Comma-separated list of allowed origins."""

    # Provider API keys — passed to LiteLLM via os.environ
    openrouter_api_key: str = ""
    deepseek_api_key: str = ""
    groq_api_key: str = ""
    pruna_api_key: str = ""
    opencode_api_key: str = ""
    """API key for OpenCode Go (opencode.ai)."""
    disabled_providers: str = ""
    """Comma-separated list of providers to disable (e.g. 'opencode-go,deepseek').
    Disabled providers are skipped during fallback — the next provider is tried.
    Set to 'opencode-go' to fall back to deepseek/groq for all pseudo-models."""
    keyclaw_enabled: bool = True
    """Set to false to disable KeyClaw secret-filtering proxy even if installed."""

    @property
    def disabled_providers_set(self) -> set[str]:
        """Parse disabled providers into a set (lowercase)."""
        if not self.disabled_providers:
            return set()
        return {
            p.strip().lower() for p in self.disabled_providers.split(",") if p.strip()
        }


settings = Settings()
