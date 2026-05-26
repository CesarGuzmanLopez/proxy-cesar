"""KeyVault middleware — transparent secret detection and re-injection.

Intercepts chat completion requests and responses to:
1. Detect API keys/secrets in user messages before they reach the LLM
2. Store them in Valkey per conversation
3. Replace with [KEYVAULT:hash] placeholders
4. On response, re-inject real values where placeholders are found

The LLM never sees real secrets. The client always sees real values.
"""

import hashlib
import json
import logging
import re
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, StreamingResponse

logger = logging.getLogger(__name__)

KEYVAULT_TTL = 3600
PLACEHOLDER_PREFIX = "KEYVAULT"

_SECRET_PATTERNS: list[tuple[str, str]] = [
    # API keys
    (r"\b(sk-(?:proj-)?[A-Za-z0-9-_]{20,})\b", "openai"),
    (r"\b(sk-ant-[A-Za-z0-9-_]{20,})\b", "anthropic"),
    (r"\b(ghp_[A-Za-z0-9]{36})\b", "github"),
    (r"\b(github_pat_[A-Za-z0-9-_]{20,})\b", "github_pat"),
    (r"\b(AKIA[0-9A-Z]{16})\b", "aws_access"),
    (r"\b(glpat-[A-Za-z0-9-_]{20,})\b", "gitlab"),
    (r"\b(xox[bps]-[A-Za-z0-9-]+)\b", "slack"),
    # Environment variable assignments
    (r'\b(DEEPSEEK_API_KEY\s*=\s*["\']?)([A-Za-z0-9]{20,})(["\']?)', "deepseek_env"),
    (
        r'\b(OPENROUTER_API_KEY\s*=\s*["\']?)([A-Za-z0-9]{20,})(["\']?)',
        "openrouter_env",
    ),
    (r'\b(GROQ_API_KEY\s*=\s*["\']?)([A-Za-z0-9]{20,})(["\']?)', "groq_env"),
    (r'\b(PROXY_API_KEY\s*=\s*["\']?)([A-Za-z0-9_-]{10,})(["\']?)', "proxy_env"),
    # Private keys (PEM format, single or multi-line)
    (
        r"(-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----\s*[\s\S]*?-----END (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----)",
        "private_key_pem",
    ),
    (
        r"(-----BEGIN PGP PRIVATE KEY BLOCK-----\s*[\s\S]*?-----END PGP PRIVATE KEY BLOCK-----)",
        "pgp_private_key",
    ),
    (
        r"(-----BEGIN ENCRYPTED PRIVATE KEY-----\s*[\s\S]*?-----END ENCRYPTED PRIVATE KEY-----)",
        "encrypted_private_key",
    ),
    # Public keys
    (
        r"(ssh-(?:rsa|ed25519|dss|ecdsa-[a-z0-9-]+)\s+AAAA[A-Za-z0-9+/]+={0,2}\s*[^\n]*)",
        "ssh_public_key",
    ),
    (
        r"(-----BEGIN PUBLIC KEY-----\s*[\s\S]*?-----END PUBLIC KEY-----)",
        "public_key_pem",
    ),
    (
        r"(-----BEGIN PGP PUBLIC KEY BLOCK-----\s*[\s\S]*?-----END PGP PUBLIC KEY BLOCK-----)",
        "pgp_public_key",
    ),
    (
        r"(-----BEGIN CERTIFICATE-----\s*[\s\S]*?-----END CERTIFICATE-----)",
        "tls_certificate",
    ),
    # Crypto wallets
    (r"\b(0x[a-fA-F0-9]{64})\b", "eth_private_key"),
    (r"\b([5KL][1-9A-HJ-NP-Za-km-z]{50,51})\b", "bitcoin_wif"),
    # JWT tokens
    (
        r"\b(eyJ[a-zA-Z0-9_-]{20,}\.[a-zA-Z0-9_-]{20,}\.[a-zA-Z0-9_-]{20,})\b",
        "jwt_token",
    ),
    # Generic long base64 strings (catch-all, lower priority)
    (r"\b([A-Za-z0-9+/]{40,}={0,2})\b", "base64_long"),
]

_PLACEHOLDER_RE = re.compile(rf"\[{PLACEHOLDER_PREFIX}:([a-f0-9]{{8}})\]")
_CHAT_PATH = "/v1/chat/completions"

_KEYVAULT_SYSTEM_PROMPT = (
    "When you see placeholders like [KEYVAULT:abc12345] in user messages, "
    "they represent sensitive values (API keys, tokens) that were replaced "
    "for security. You can freely reference these placeholders in your "
    "responses — the server will automatically replace them with the "
    "real values before the user sees your response. Never try to guess "
    "or generate the real values; just use the placeholder as-is."
)


def _hash_secret(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()[:8]


def _make_placeholder(hash_val: str) -> str:
    return f"[{PLACEHOLDER_PREFIX}:{hash_val}]"


def _mask_text(text: str, secrets: dict[str, str]) -> str:
    """Find secrets in text and replace with placeholders. Returns (masked_text, updated_secrets)."""
    for pattern, kind in _SECRET_PATTERNS:
        matches = list(re.finditer(pattern, text))
        for match in reversed(matches):
            groups = match.groups()
            if not groups:
                continue

            if kind.endswith("_env"):
                prefix, secret, suffix = (
                    groups[0],
                    groups[1],
                    groups[2] if len(groups) > 2 else "",
                )
                secret_hash = _hash_secret(secret)
                secrets[secret_hash] = secret
                placeholder = _make_placeholder(secret_hash)
                text = (
                    text[: match.start()]
                    + f"{prefix}{placeholder}{suffix}"
                    + text[match.end() :]
                )
            else:
                secret = groups[0]
                secret_hash = _hash_secret(secret)
                secrets[secret_hash] = secret
                placeholder = _make_placeholder(secret_hash)
                text = text[: match.start()] + placeholder + text[match.end() :]

    return text


def _re_inject(text: str, secrets: dict[str, str]) -> str:
    """Replace [KEYVAULT:hash] with real values from secrets dict."""
    for secret_hash, real_value in secrets.items():
        placeholder = _make_placeholder(secret_hash)
        text = text.replace(placeholder, real_value)
    return text


class KeyVaultMiddleware(BaseHTTPMiddleware):
    """Transparent secret vault middleware.

    Only activates for POST /v1/chat/completions.
    Secrets are scoped per conversation_id via Valkey.
    """

    async def dispatch(self, request, call_next):
        if request.url.path != _CHAT_PATH:
            return await call_next(request)

        valkey = request.app.state.valkey
        if valkey is None:
            return await call_next(request)

        # ── Request: extract secrets, store in Valkey, mask body ──────────
        try:
            body_bytes = await request.body()
            body = json.loads(body_bytes)
        except (json.JSONDecodeError, Exception):
            return await call_next(request)

        conversation_id = body.get("conversation_id") or "anon"
        secrets: dict[str, str] = {}

        messages = body.get("messages", [])
        if isinstance(messages, list):
            for msg in messages:
                content = msg.get("content")
                if isinstance(content, str):
                    msg["content"] = _mask_text(content, secrets)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            part["text"] = _mask_text(
                                str(part.get("text", "")), secrets
                            )

        # Store secrets in Valkey
        for secret_hash, secret_value in secrets.items():
            try:
                await valkey.set(
                    f"keyvault:{conversation_id}:{secret_hash}",
                    secret_value,
                    ex=KEYVAULT_TTL,
                )
                logger.debug(
                    "keyvault_store conv=%s hash=%s", conversation_id[:12], secret_hash
                )
            except Exception:
                continue

        if secrets:
            messages.insert(0, {"role": "system", "content": _KEYVAULT_SYSTEM_PROMPT})
            logger.info(
                "keyvault_active conv=%s secrets=%d", conversation_id[:12], len(secrets)
            )

        # Reconstruct body for downstream — set cached body directly
        # instead of overriding _receive, to avoid _stream_consumed issues.
        request._body = json.dumps(body).encode()

        # ── Call handler ─────────────────────────────────────────────────
        response = await call_next(request)

        # ── Response: re-inject real values ────────────────────────────────
        if secrets:
            if isinstance(response, StreamingResponse):
                original_iterator = response.body_iterator

                async def _re_inject_stream(secrets=secrets):
                    try:
                        async for chunk in original_iterator:
                            if isinstance(chunk, bytes):
                                yield _re_inject(
                                    chunk.decode("utf-8", errors="replace"),
                                    secrets,
                                ).encode("utf-8")
                            else:
                                yield _re_inject(str(chunk), secrets)
                    except Exception:
                        pass

                return StreamingResponse(
                    content=_re_inject_stream(),
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=response.media_type,
                )
            else:
                try:
                    body_bytes_resp = b""
                    async for chunk in response.body_iterator:
                        body_bytes_resp += chunk

                    resp_json = json.loads(body_bytes_resp)

                    def _re_inject_recursive(obj: Any) -> Any:
                        if isinstance(obj, str):
                            return _re_inject(obj, secrets)
                        if isinstance(obj, dict):
                            return {k: _re_inject_recursive(v) for k, v in obj.items()}
                        if isinstance(obj, list):
                            return [_re_inject_recursive(item) for item in obj]
                        return obj

                    resp_json = _re_inject_recursive(resp_json)
                    new_body = json.dumps(resp_json).encode()

                    return JSONResponse(
                        content=json.loads(new_body),
                        status_code=response.status_code,
                        headers=dict(response.headers),
                    )
                except Exception:
                    pass  # Fall through to original response

        return response
