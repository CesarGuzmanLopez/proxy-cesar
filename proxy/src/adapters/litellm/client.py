"""LiteLLM integration — setup and async completion wrapper.

python.md §5.2: Adapter pattern — wraps external library with our own interface.
analisis.md: LiteLLM handles all provider translation. Proxy never translates.
"""

import os
from pathlib import Path

# ── SSL_CERT_FILE must be set *before* httpx/litellm are imported ────────
# On NixOS, Python reads NIX_SSL_CERT_FILE but httpx reads SSL_CERT_FILE.
# httpx caches its default SSL context at import time, so the env var must
# be present before the first `import httpx` / `import litellm`.
if not os.environ.get("SSL_CERT_FILE"):
    for candidate in (
        os.environ.get("NIX_SSL_CERT_FILE", ""),
        "/etc/ssl/certs/ca-certificates.crt",
        "/etc/ssl/cert.pem",
    ):
        if candidate and Path(candidate).exists():
            os.environ["SSL_CERT_FILE"] = candidate
            break

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

_KEYCLAW_HOME = Path.home() / ".keyclaw"
_KEYCLAW_PID_FILE = _KEYCLAW_HOME / "proxy.pid"
_KEYCLAW_CA_CERT = _KEYCLAW_HOME / "ca.crt"


def _ensure_ssl_cert_file() -> None:
    """Ensure SSL_CERT_FILE is set for httpx/LiteLLM.

    On NixOS, Python reads NIX_SSL_CERT_FILE but httpx reads SSL_CERT_FILE.
    If SSL_CERT_FILE is missing but a NixOS cert bundle exists, bridge them.
    """
    if os.environ.get("SSL_CERT_FILE"):
        return
    for candidate in (
        os.environ.get("NIX_SSL_CERT_FILE", ""),
        "/etc/ssl/certs/ca-certificates.crt",
        "/etc/ssl/cert.pem",
    ):
        if candidate and Path(candidate).exists():
            os.environ["SSL_CERT_FILE"] = candidate
            print(f"SSL_CERT_FILE set to {candidate}")
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
    except (FileNotFoundError, OSError, ValueError):
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
    system_bundle = Path("/etc/ssl/cert.pem")

    if not combined.exists() and keyclaw_ca.exists() and system_bundle.exists():
        combined.write_text(
            system_bundle.read_text() + "\n" + keyclaw_ca.read_text()
        )

    os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:8877")
    os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:8877")
    os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,::1,.local")

    if combined.exists():
        os.environ.setdefault("SSL_CERT_FILE", str(combined))
        os.environ.setdefault("REQUESTS_CA_BUNDLE", str(combined))
        os.environ.setdefault("NODE_EXTRA_CA_CERTS", str(keyclaw_ca))


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
    _ensure_ssl_cert_file()
    os.environ.setdefault("OPENROUTER_API_KEY", settings.openrouter_api_key)
    os.environ.setdefault("DEEPSEEK_API_KEY", settings.deepseek_api_key)
    os.environ.setdefault("GROQ_API_KEY", settings.groq_api_key)
    os.environ.setdefault("PRUNA_API_KEY", settings.pruna_api_key)
    os.environ.setdefault("PRUNA_API_BASE", "https://api.pruna.ai/v1")

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
