"""LiteLLM integration — setup and async completion wrapper.

python.md §5.2: Adapter pattern — wraps external library with our own interface.
analisis.md: LiteLLM handles all provider translation. Proxy never translates.
"""

import os

import litellm
from litellm.exceptions import RateLimitError, ServiceUnavailableError

from src.config.settings import Settings

# Re-export for fallback detection
__all__ = ["setup_litellm", "call_litellm", "ServiceUnavailableError", "RateLimitError"]


def setup_litellm(settings: Settings) -> None:
    """Pass all provider API keys to LiteLLM via os.environ.

    Called once during FastAPI lifespan startup.
    """
    os.environ.setdefault("ANTHROPIC_API_KEY", settings.anthropic_api_key)
    os.environ.setdefault("OPENROUTER_API_KEY", settings.openrouter_api_key)
    os.environ.setdefault("GOOGLE_API_KEY", settings.google_api_key)
    os.environ.setdefault("DEEPSEEK_API_KEY", settings.deepseek_api_key)
    os.environ.setdefault("GROQ_API_KEY", settings.groq_api_key)
    os.environ.setdefault("ZHIPU_API_KEY", settings.zhipu_api_key)

    litellm.suppress_debug_info = True
    litellm.set_verbose = False


async def call_litellm(
    model: str,
    messages: list[dict],
    stream: bool = False,
    **kwargs,
):
    """Call LiteLLM with the exact model ID from config.

    The model string is used verbatim — no prefix, no transformation.
    analisis.md §4.0: 'Sin prefijos, sin transformación, sin concatenación.'
    """
    response = await litellm.acompletion(
        model=model,
        messages=messages,
        stream=stream,
        **kwargs,
    )
    return response
