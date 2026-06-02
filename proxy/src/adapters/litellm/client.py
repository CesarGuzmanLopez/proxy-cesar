"""LiteLLM integration — setup and async completion wrapper.

python.md §5.2: Adapter pattern — wraps external library with our own interface.
analisis.md: LiteLLM handles all provider translation. Proxy never translates.
"""

import asyncio
import logging
import os
from pathlib import Path

_SSL_CERT_FILE = "/etc/ssl/cert.pem"
_CA_CERT_FILE = "/etc/ssl/certs/ca-certificates.crt"

logger = logging.getLogger(__name__)

# SSL_CERT_FILE is configured at startup in main.py (runs first).
# The _ensure_ssl_cert_file() function below provides a safety net for
# standalone imports where main.py's initialization hasn't run.

import litellm  # noqa: E402 — SSL_CERT_FILE must be set before import
from litellm.exceptions import RateLimitError, ServiceUnavailableError  # noqa: E402

from src.config.settings import Settings  # noqa: E402

# Re-export for fallback detection
__all__ = [
    "setup_litellm",
    "call_litellm",
    "ServiceUnavailableError",
    "RateLimitError",
    "normalise_stream_chunk",
]


def _ensure_ssl_cert_file() -> None:
    """Ensure SSL_CERT_FILE is set for httpx/LiteLLM.

    On NixOS, Python reads NIX_SSL_CERT_FILE but httpx reads SSL_CERT_FILE.
    If SSL_CERT_FILE is missing but a NixOS cert bundle exists, bridge them.
    """
    if os.environ.get("SSL_CERT_FILE"):
        return
    for candidate in (
        os.environ.get("NIX_SSL_CERT_FILE", ""),
        _CA_CERT_FILE,
        _SSL_CERT_FILE,
    ):
        if candidate and Path(candidate).exists():
            os.environ["SSL_CERT_FILE"] = candidate
            return


def _extract_response_headers(response) -> dict:
    """Extract provider response headers from a LiteLLM response object.

    Both streaming (CustomStreamWrapper) and non-streaming (ModelResponse)
    LiteLLM responses store provider response headers in
    ``_hidden_params["additional_headers"]``.  Streaming also stores raw
    headers in ``_response_headers``.

    Returns an empty dict if the response doesn't have headers (e.g. error
    responses, non-standard providers).
    """
    try:
        # CustomStreamWrapper (streaming) stores headers directly
        raw = getattr(response, "_response_headers", None)
        if raw:
            return dict(raw)
        # ModelResponse (non-streaming) stores headers in _hidden_params
        hidden = getattr(response, "_hidden_params", None)
        if hidden and isinstance(hidden, dict):
            additional = hidden.get("additional_headers")
            if additional:
                return dict(additional)
    except (AttributeError, TypeError, ValueError):
        pass
    return {}


def normalise_stream_chunk(chunk) -> None:
    """No-op: reasoning_content stays in its own field, content stays clean.

    Previously this function copied reasoning_content into content and removed it,
    which caused reasoning and response text to be mixed together in the client.
    
    Now we keep the chunk as-is: reasoning_content stays separate so clients that
    support it (opencode, Continue, etc.) display it in a thinking/expander section.
    """
    pass


def setup_litellm(settings: Settings) -> None:
    """Pass all provider API keys to LiteLLM via os.environ.

    Called once during FastAPI lifespan startup.

    ``drop_params = True`` is CRITICAL: without it, LiteLLM rejects unsupported
    parameters like ``thinking`` or ``reasoning_effort`` for providers where its
    internal detection is inaccurate (e.g. OpenCode Go custom endpoints).
    """
    litellm.drop_params = True
    _ensure_ssl_cert_file()
    os.environ.setdefault("OPENROUTER_API_KEY", settings.openrouter_api_key)
    os.environ.setdefault("DEEPSEEK_API_KEY", settings.deepseek_api_key)
    os.environ.setdefault("GROQ_API_KEY", settings.groq_api_key)
    os.environ.setdefault("PRUNA_API_KEY", settings.pruna_api_key)
    os.environ.setdefault("PRUNA_API_BASE", "https://api.pruna.ai/v1")
    os.environ.setdefault("OPENCODE_API_KEY", settings.opencode_api_key)
    os.environ.setdefault("NVIDIA_API_KEY", settings.nvidia_api_key)
    os.environ.setdefault("CEREBRAS_API_KEY", settings.cerebras_api_key)
    # Anthropic provider (anthropic/ prefix) needs ANTHROPIC_API_KEY for Go models
    if settings.opencode_api_key:
        os.environ.setdefault("ANTHROPIC_API_KEY", settings.opencode_api_key)

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

    Supports optional ``api_base`` and ``api_key`` kwargs for custom
    endpoints (e.g. OpenCode Go). When ``api_key_env`` is set on the
    physical model, the caller resolves the actual key and passes it here.

    When ``timeout`` is passed (seconds), it overrides the default
    ``DEFAULT_LLM_TIMEOUT_SECONDS``.

    Post-processing: some providers (e.g. GLM via OpenRouter) put the response
    in ``reasoning_content`` instead of ``content`` when they run out of token
    budget (the model "thinks" long and leaves no tokens for the final answer).
    This normalisation copies ``reasoning_content`` into ``content`` so the
    caller never sees an empty assistant reply.

    Timeout is managed via ``asyncio.wait_for()``.
    """
    # Extract api_base/api_key/timeout if present (passed in **kwargs)
    api_base = kwargs.pop("api_base", None)
    api_key = kwargs.pop("api_key", None)
    timeout = kwargs.pop("timeout", None)

    logger.info(
        "litellm_call model=%s stream=%s messages=%d api_base=%s",
        model,
        stream,
        len(messages),
        bool(api_base),
    )

    from src.config.constants import DEFAULT_LLM_TIMEOUT_SECONDS

    effective_timeout = timeout if timeout is not None else DEFAULT_LLM_TIMEOUT_SECONDS

    response = await asyncio.wait_for(
        litellm.acompletion(
            model=model,
            messages=messages,
            stream=stream,
            api_base=api_base,
            api_key=api_key,
            **kwargs,
        ),
        timeout=effective_timeout,
    )

    # ── Extract provider response headers ─────────────────────────────────
    # Attach to the response object so callers get headers without any API
    # surface change.  This must happen BEFORE wrapping the stream, since
    # headers live on the response object itself, not in the stream chunks.
    provider_headers = (
        _extract_response_headers(response) if response is not None else {}
    )
    try:
        response._provider_response_headers = provider_headers
    except (AttributeError, TypeError):
        pass

    return response
