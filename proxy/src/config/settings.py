"""Environment configuration via pydantic-settings.

Reads from .env file. Exact schema from sprint §5.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    proxy_port: int = 9110
    database_url: str = "sqlite+aiosqlite:///./proxy.db"
    valkey_url: str = "valkey://localhost:6379"

    # Provider API keys — passed to LiteLLM via os.environ
    anthropic_api_key: str = ""
    openrouter_api_key: str = ""
    google_api_key: str = ""
    deepseek_api_key: str = ""
    groq_api_key: str = ""
    zhipuai_api_key: str = ""


settings = Settings()
