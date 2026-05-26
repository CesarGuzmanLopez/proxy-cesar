"""Sanitize sensitive data (API keys, tokens) from error messages and logs.

All provider API keys follow predictable patterns.  This module strips or
redacts them from any string before it reaches the HTTP response or logs.
"""

import re
from typing import Any

# ── Known API key patterns (extend as needed) ──────────────────────────────
_KEY_PATTERNS: list[re.Pattern] = [
    # OpenAI / LiteLLM compatible / DeepSeek: sk-...
    re.compile(r"sk-[A-Za-z0-9]{8,}"),
    # Anthropic: sk-ant-...
    re.compile(r"sk-ant-[A-Za-z0-9_-]{8,}"),
    # Google Gemini / AI Studio: AIza...
    re.compile(r"AIza[0-9A-Za-z_-]{8,}"),
    # Groq: gsk_...
    re.compile(r"gsk_[A-Za-z0-9]{8,}"),
    # OpenRouter: sk-or-v1-...
    re.compile(r"sk-or-v1-[A-Za-z0-9]{8,}"),
    # ZhipuAI / Z.ai: hex.hex format (two hex strings separated by dot)
    re.compile(r"[0-9a-f]{16,}\.[A-Za-z0-9]{8,}"),
    # Generic "api_key", "bearer", "token" or "secret" followed by value
    re.compile(r"(?i)(api[_-]?key|bearer|token|secret)\s*[:=]\s*\S{8,}"),
    # JWT format (three base64url segments separated by dots)
    re.compile(r"eyJ[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}"),
]


def sanitize(value: str) -> str:
    """Return *value* with any detected API keys redacted."""
    for pattern in _KEY_PATTERNS:
        value = pattern.sub("***REDACTED***", value)
    return value


def sanitize_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Recursively sanitize all string values in a dict."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, str):
            out[k] = sanitize(v)
        elif isinstance(v, dict):
            out[k] = sanitize_dict(v)
        elif isinstance(v, list):
            out[k] = [
                sanitize_dict(i)
                if isinstance(i, dict)
                else (sanitize(i) if isinstance(i, str) else i)
                for i in v
            ]
        else:
            out[k] = v
    return out
