"""Environment configuration via pydantic-settings.

Reads from .env file. Exact schema from feature
"""

import logging
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    proxy_host: str = "127.0.0.1"
    """Host to bind the server. Use '0.0.0.0' behind Nginx/Caddy reverse proxy."""
    proxy_port: int = 9110
    database_url: str = "sqlite+aiosqlite:///./proxy.db"
    valkey_url: str = "valkey://localhost:6379"

    # feature Auth
    proxy_api_key: str = ""
    proxy_api_keys: str = ""
    """Bearer token for API access. Empty = dev mode (no auth)."""

    # feature CORS
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

    def validate_required_keys(self) -> None:
        """Ensure at least one provider API key is configured.

        Called on startup to fail fast if no providers are available.
        Raises ValueError if all provider keys are empty.
        """
        provider_keys = [
            ("opencode_api_key", self.opencode_api_key),
            ("deepseek_api_key", self.deepseek_api_key),
            ("groq_api_key", self.groq_api_key),
            ("openrouter_api_key", self.openrouter_api_key),
            ("pruna_api_key", self.pruna_api_key),
        ]
        available_keys = [name for name, value in provider_keys if value]
        if not available_keys:
            missing = ", ".join(name for name, _ in provider_keys)
            raise ValueError(
                f"No provider API keys configured. Set at least one of: {missing}"
            )
        logger.info("startup validated api_keys configured=%s", available_keys)


settings = Settings()
# Validate on module import
try:
    settings.validate_required_keys()
except ValueError as e:
    logger.error("settings_validation_failed error=%s", str(e))
    raise
