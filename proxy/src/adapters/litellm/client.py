"""LiteLLM integration — setup and async completion wrapper.

python.md §5.2: Adapter pattern — wraps external library with our own interface.
analisis.md: LiteLLM handles all provider translation. Proxy never translates.
"""

import os

import litellm
from litellm.exceptions import RateLimitError, ServiceUnavailableError

from src.config.settings import Settings

# Re-export for fallback detection
__all__ = [
    "setup_litellm",
    "call_litellm",
    "ServiceUnavailableError",
    "RateLimitError",
    "normalise_stream_chunk",
]


def normalise_stream_chunk(chunk) -> None:
    """Normalise a streaming chunk in-place so that ``content`` is always populated.

    Some providers (GLM via OpenRouter, DeepSeek in chain-of-thought mode)
    emit the assistant response in ``delta.reasoning_content`` instead of
    ``delta.content``, leaving ``delta.content`` as ``None`` for every chunk.
    This confuses downstream consumers (the chunk's ``content`` appears empty).

    When a chunk has ``reasoning_content`` but no ``content``, this function
    copies the reasoning text into ``content`` so the caller always receives
    a usable payload.

    The function is a no-op for chunks that already have ``content`` or that
    lack the ``reasoning_content`` attribute entirely.
    """
    try:
        for choice in chunk.choices:
            delta = choice.delta
            if delta is None:
                continue
            if (
                delta.content is None
                and getattr(delta, "reasoning_content", None) is not None
            ):
                delta.content = delta.reasoning_content
    except (AttributeError, TypeError, IndexError):
        pass  # Non-standard chunk format — leave untouched


def setup_litellm(settings: Settings) -> None:
    """Pass all provider API keys to LiteLLM via os.environ.

    Called once during FastAPI lifespan startup.
    """
    os.environ.setdefault("ANTHROPIC_API_KEY", settings.anthropic_api_key)
    os.environ.setdefault("OPENROUTER_API_KEY", settings.openrouter_api_key)
    os.environ.setdefault("GOOGLE_API_KEY", settings.google_api_key)
    os.environ.setdefault("DEEPSEEK_API_KEY", settings.deepseek_api_key)
    os.environ.setdefault("GROQ_API_KEY", settings.groq_api_key)
    os.environ.setdefault("ZHIPUAI_API_KEY", settings.zhipuai_api_key)
    os.environ.setdefault("ZAI_API_KEY", settings.zai_api_key)

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

    Post-processing: some providers (e.g. GLM via OpenRouter) put the response
    in ``reasoning_content`` instead of ``content`` when they run out of token
    budget (the model "thinks" long and leaves no tokens for the final answer).
    This normalisation copies ``reasoning_content`` into ``content`` so the
    caller never sees an empty assistant reply.
    """
    response = await litellm.acompletion(
        model=model,
        messages=messages,
        stream=stream,
        **kwargs,
    )

    # ── Normalise empty content (non-streaming) ────────────────────────────
    if not stream and response is not None:
        try:
            for choice in response.choices:
                msg = choice.message
                if msg and not msg.content and getattr(msg, "reasoning_content", None):
                    msg.content = msg.reasoning_content
        except (AttributeError, TypeError):
            pass  # Non-standard response format — leave as-is

    # ── Normalise streaming chunks (wrap generator) ───────────────────────
    if stream:
        # Returning the original generator wrapped so every chunk is
        # normalised on the fly.  The caller iterates over the returned
        # value transparently.
        async def _normalised_stream():
            async for chunk in response:
                normalise_stream_chunk(chunk)
                yield chunk

        return _normalised_stream()

    return response
