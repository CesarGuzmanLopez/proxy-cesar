"""LiteLLM integration — setup and async completion wrapper.

python.md §5.2: Adapter pattern — wraps external library with our own interface.
analisis.md: LiteLLM handles all provider translation. Proxy never translates.
"""

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

_KEYCLAW_HOME = Path.home() / ".keyclaw"
_KEYCLAW_PID_FILE = _KEYCLAW_HOME / "proxy.pid"
_KEYCLAW_CA_CERT = _KEYCLAW_HOME / "ca.crt"


def _ensure_ssl_cert_file() -> None:
    """Ensure SSL_CERT_FILE is set for httpx/LiteLLM.

    On NixOS, Python reads NIX_SSL_CERT_FILE but httpx reads SSL_CERT_FILE.
    If SSL_CERT_FILE is missing but a NixOS cert bundle exists, bridge them.
    Also checks KeyClaw's combined CA bundle if available.
    """
    if os.environ.get("SSL_CERT_FILE"):
        return
    _keyclaw_combined = _KEYCLAW_HOME / "combined-ca.pem"
    for candidate in (
        os.environ.get("NIX_SSL_CERT_FILE", ""),
        str(_keyclaw_combined) if _keyclaw_combined.exists() else "",
        _CA_CERT_FILE,
        _SSL_CERT_FILE,
    ):
        if candidate and Path(candidate).exists():
            os.environ["SSL_CERT_FILE"] = candidate
            return


def _keyclaw_is_running() -> bool:
    """Check whether the KeyClaw daemon is active by inspecting its PID file."""
    try:
        pid_text = _KEYCLAW_PID_FILE.read_text().strip()
        if not pid_text.isdigit():
            return False
        # Send signal 0 to test liveness
        os.kill(int(pid_text), 0)
        return True
    except (OSError, ValueError):
        return False


def _setup_keyclaw_proxy() -> None:
    """Configure outbound HTTP/HTTPS proxy through KeyClaw if it is running.

    KeyClaw (`keyclaw proxy`) is a local MITM proxy that strips API keys and
    other secrets from LLM-bound traffic before it leaves the machine.  When
    the KeyClaw daemon is active we route all outbound provider calls through
    it for an extra layer of credential protection.

    A combined CA bundle (system CA + KeyClaw's own CA) is created so that
    httpx / certifi can verify both regular TLS certificates and KeyClaw-
    signed MITM certificates.
    """
    if not _KEYCLAW_HOME.exists():
        return

    keyclaw_ca = _KEYCLAW_CA_CERT
    combined = _KEYCLAW_HOME / "combined-ca.pem"
    system_bundle = Path(_SSL_CERT_FILE)

    if not combined.exists() and keyclaw_ca.exists() and system_bundle.exists():
        combined.write_text(system_bundle.read_text() + "\n" + keyclaw_ca.read_text())

    os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:8877")
    os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:8877")
    os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,::1,.local")

    if combined.exists():
        # Force SSL_CERT_FILE to combined bundle (system CA + KeyClaw CA)
        # so httpx clients created after this point can verify KeyClaw's MITM certs.
        os.environ["SSL_CERT_FILE"] = str(combined)
        os.environ.setdefault("REQUESTS_CA_BUNDLE", str(combined))
        os.environ.setdefault("NODE_EXTRA_CA_CERTS", str(keyclaw_ca))


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
    """
    _ensure_ssl_cert_file()
    os.environ.setdefault("OPENROUTER_API_KEY", settings.openrouter_api_key)
    os.environ.setdefault("DEEPSEEK_API_KEY", settings.deepseek_api_key)
    os.environ.setdefault("GROQ_API_KEY", settings.groq_api_key)
    os.environ.setdefault("PRUNA_API_KEY", settings.pruna_api_key)
    os.environ.setdefault("PRUNA_API_BASE", "https://api.pruna.ai/v1")
    os.environ.setdefault("OPENCODE_API_KEY", settings.opencode_api_key)
    # Anthropic provider (anthropic/ prefix) needs ANTHROPIC_API_KEY for Go models
    if settings.opencode_api_key:
        os.environ.setdefault("ANTHROPIC_API_KEY", settings.opencode_api_key)

    if not settings.keyclaw_enabled:
        litellm.suppress_debug_info = True
        litellm.set_verbose = False
        return

    if _keyclaw_is_running():
        _setup_keyclaw_proxy()
        litellm.suppress_debug_info = True
        litellm.set_verbose = False
        return

    if _KEYCLAW_HOME.exists():
        import logging

        _log = logging.getLogger("src.adapters.litellm")
        _log.warning(
            "KeyClaw is installed but the proxy daemon is not running. "
            "Starting without KeyClaw (dev mode). "
            "Outbound traffic will NOT be filtered for secrets. "
            "Start it with:  keyclaw proxy start --foreground"
        )

    # KeyClaw not installed/running — proxy starts without it (dev mode)
    litellm.suppress_debug_info = True
    litellm.set_verbose = False


async def call_litellm(
    model: str,
    messages: list[dict],
    stream: bool = False,
    timeout: float | None = None,
    **kwargs,
):
    """Call LiteLLM with the exact model ID from config.

    The model string is used verbatim — no prefix, no transformation.
    analisis.md §4.0: 'Sin prefijos, sin transformación, sin concatenación.'

    Supports optional ``api_base`` and ``api_key`` kwargs for custom
    endpoints (e.g. OpenCode Go). When ``api_key_env`` is set on the
    physical model, the caller resolves the actual key and passes it here.

    Post-processing: some providers (e.g. GLM via OpenRouter) put the response
    in ``reasoning_content`` instead of ``content`` when they run out of token
    budget (the model "thinks" long and leaves no tokens for the final answer).
    This normalisation copies ``reasoning_content`` into ``content`` so the
    caller never sees an empty assistant reply.
    """
    # Extract api_base/api_key if present (they are passed in **kwargs)
    api_base = kwargs.pop("api_base", None)
    api_key = kwargs.pop("api_key", None)

    logger.info(
        "litellm_call model=%s stream=%s messages=%d api_base=%s",
        model,
        stream,
        len(messages),
        bool(api_base),
    )

    from src.config.constants import DEFAULT_LLM_TIMEOUT_SECONDS

    effective_timeout = timeout if timeout is not None else DEFAULT_LLM_TIMEOUT_SECONDS

    response = await litellm.acompletion(
        model=model,
        messages=messages,
        stream=stream,
        api_base=api_base,
        api_key=api_key,
        timeout=effective_timeout,
        **kwargs,
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
