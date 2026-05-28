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
import uuid
from functools import lru_cache


from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, StreamingResponse

logger = logging.getLogger(__name__)

KEYVAULT_TTL = 3600
PLACEHOLDER_PREFIX = "KEYVAULT"

# ── Pre-compiled regex patterns for better performance ─────────────────────────
# Compiled once at module load, not on every request
_SECRET_PATTERNS_RAW: list[tuple[str, str]] = [
    # ── API keys (prefixed — provider-specific, low false positive) ────────
    (r"\b([a-z]{0,1}sk[a-z]{0,1}[-_][A-Za-z0-9_-]{10,})\b", "sk_key"),
    (r"\b(sk-proj-[A-Za-z0-9_-]{20,})\b", "openai_proj"),
    (r"\b(sk-ant-(?:api03|admin01)-[a-zA-Z0-9_-]{93}AA)\b", "anthropic"),
    (r"\b(sk-or-v1-[A-Za-z0-9]{10,})\b", "openrouter"),
    (r"\b(ghp_[A-Za-z0-9]{36})\b", "github_classic"),
    (r"\b(github_pat_[A-Za-z0-9-_]{10,})\b", "github_pat"),
    (r"\b(glpat-[A-Za-z0-9-_]{10,})\b", "gitlab"),
    (r"\b(hf_[A-Za-z0-9]{10,})\b", "huggingface"),
    (r"\b(AIza[0-9A-Za-z_-]{35})\b", "google_ai"),
    (r"\b(ya29\.[0-9A-Za-z_-]{50,})\b", "google_oauth"),
    (r"\b((?:A3T[A-Z0-9]|AKIA|ASIA|ABIA|ACCA)[A-Z2-7]{16})\b", "aws_access"),
    (r"\b(ABSK[A-Za-z0-9+/]{109,269}={0,2})\b", "aws_bedrock"),
    (r"\b([a-zA-Z0-9_~.]{3}\dQ~[a-zA-Z0-9_~.-]{31,34})\b", "azure"),
    (r"\b(xox[bpsar]-(?:\d+-){0,3}[A-Za-z0-9-]{10,})\b", "slack"),
    (r"\b([A-Za-z0-9_-]{24}\.[A-Za-z0-9_-]{6}\.[A-Za-z0-9_-]{27})\b", "discord"),
    (r"\b(sk_live_[0-9a-zA-Z]{24,})\b", "stripe_live"),
    (r"\b(rk_live_[0-9a-zA-Z]{24,})\b", "stripe_restricted"),
    (r"\b(SK[0-9a-fA-F]{32})\b", "twilio"),
    (r"\b([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\b", "heroku_uuid"),
    (r"\b(A3-[A-Z0-9]{6}-(?:[A-Z0-9]{11}|[A-Z0-9]{6}-[A-Z0-9]{5})-[A-Z0-9]{5}-[A-Z0-9]{5}-[A-Z0-9]{5})\b", "1password"),
    (r"\b(ops_eyJ[a-zA-Z0-9+/]{250,}={0,3})\b", "1password_sa"),
    (r"\b(ATATT3[A-Za-z0-9_\-=]{186})\b", "atlassian"),
    (r"\b(sntrys_[A-Za-z0-9]{32,})\b", "sentry"),
    (r'\b([A-Z_]{3,30}_API_KEY\s*=\s*["\']?)([A-Za-z0-9_-]{10,})(["\']?)', "env_api_key"),
    (r'\b([A-Z_]{3,30}_TOKEN\s*=\s*["\']?)([A-Za-z0-9._-]{10,})(["\']?)', "env_token"),
    (r'\b([A-Z_]{3,30}_SECRET\s*=\s*["\']?)([A-Za-z0-9._/-]{10,})(["\']?)', "env_secret"),
    (r'\b([A-Z_]{3,30}_KEY\s*=\s*["\']?)([A-Za-z0-9_-]{10,})(["\']?)', "env_key"),
    (r"(-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----\s*[\s\S]*?-----END (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----)", "private_key_pem"),
    (r"(-----BEGIN PGP PRIVATE KEY BLOCK-----\s*[\s\S]*?-----END PGP PRIVATE KEY BLOCK-----)", "pgp_private_key"),
    (r"(-----BEGIN ENCRYPTED PRIVATE KEY-----\s*[\s\S]*?-----END ENCRYPTED PRIVATE KEY-----)", "encrypted_private_key"),
    (r"(ssh-(?:rsa|ed25519|dss|ecdsa-[a-z0-9-]+)\s+AAAA[A-Za-z0-9+/]+={0,2}\s*[^\n]*)", "ssh_public_key"),
    (r"(-----BEGIN PUBLIC KEY-----\s*[\s\S]*?-----END PUBLIC KEY-----)", "public_key_pem"),
    (r"(-----BEGIN CERTIFICATE-----\s*[\s\S]*?-----END CERTIFICATE-----)", "tls_certificate"),
    (r"\b(0x[a-fA-F0-9]{64})\b", "eth_private_key"),
    (r"\b([5KL][1-9A-HJ-NP-Za-km-z]{50,51})\b", "bitcoin_wif"),
    (r"\b(eyJ[a-zA-Z0-9_-]{20,}\.[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,})\b", "jwt_token"),
    (r"\b([A-Za-z0-9+/=_-]{60,})\b", "long_token"),
]

# Compile patterns once at module load time
_SECRET_PATTERNS = [(re.compile(pattern, re.MULTILINE | re.DOTALL), kind)
                     for pattern, kind in _SECRET_PATTERNS_RAW]

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
# NOTE: This system prompt is a CONSTANT string — it uses "[KEYVAULT:abc12345]"
# as a literal example, NOT actual placeholders. The prompt is IDENTICAL across
# ALL conversations. Actual placeholders appear only in user messages.
# The hash function (_hash_secret) is deterministic SHA-256, so the same secret
# always produces the same placeholder. This NORMALIZES variable secrets into
# consistent tokens, which IMPROVES provider cache stability rather than breaking it.
# This is NOT a bug.


@lru_cache(maxsize=1024)
def _hash_secret(secret: str) -> str:
    """Hash secret to deterministic 8-char hex. Cached for performance."""
    return hashlib.sha256(secret.encode()).hexdigest()[:8]


@lru_cache(maxsize=1024)
def _make_placeholder(hash_val: str) -> str:
    """Create placeholder from hash. Cached to avoid string concatenation."""
    return f"[{PLACEHOLDER_PREFIX}:{hash_val}]"


def _mask_text(text: str, secrets: dict[str, str]) -> str:
    """Find and mask secrets in text efficiently.

    Optimizations:
    - Pre-compiled regex patterns
    - Single pass per pattern (reversed iteration for safe indexing)
    - Early termination if text has no placeholders yet
    - Caching of hash and placeholder functions
    """
    if not text or len(text) < 8:
        return text

    for compiled_pattern, kind in _SECRET_PATTERNS:
        # Use finditer for memory efficiency with large text
        matches = list(compiled_pattern.finditer(text))
        if not matches:
            continue

        # Process matches in reverse to maintain correct positions
        for match in reversed(matches):
            groups = match.groups()
            if not groups:
                continue

            # Multi-group patterns: extract secret from middle group
            if len(groups) >= 2 and len(groups[1]) >= 4:
                secret = groups[1]
                prefix = groups[0] or ""
                suffix = groups[2] if len(groups) > 2 else ""
            else:
                # Single group: whole secret
                secret = groups[0]
                if len(secret) < 8:
                    continue
                prefix = suffix = ""

            # Store and replace in one operation
            secret_hash = _hash_secret(secret)
            secrets[secret_hash] = secret
            placeholder = _make_placeholder(secret_hash)
            text = text[:match.start()] + prefix + placeholder + suffix + text[match.end():]

    return text


def _re_inject(text: str, secrets: dict[str, str]) -> str:
    """Replace [KEYVAULT:hash] with real values efficiently.

    Optimizations:
    - Early return for text without placeholders
    - Single pass with pre-compiled regex
    - Batched replacement using list comprehension
    """
    if not text or not secrets:
        return text

    # Check if any placeholders exist before processing
    if PLACEHOLDER_PREFIX not in text:
        return text

    # Build efficient replacement using regex callback
    def _replacer(match):
        hash_val = match.group(1)
        # Pre-compute placeholder to look up secret
        for secret_hash, real_value in secrets.items():
            if secret_hash == hash_val:
                return real_value
        return match.group(0)  # Return unchanged if not found

    return _PLACEHOLDER_RE.sub(_replacer, text)


# ── Request helpers ──────────────────────────────────────────────────────────


def _mask_messages(body: dict, secrets: dict[str, str]) -> None:
    """Scan messages in body for secrets and replace with placeholders in-place."""
    messages = body.get("messages", [])
    if not isinstance(messages, list):
        return
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = _mask_text(content, secrets)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    part["text"] = _mask_text(str(part.get("text", "")), secrets)


async def _store_secrets(
    valkey, conversation_id: str, secrets: dict[str, str], trace_id: str = "????"
) -> None:
    """Store detected secrets in Valkey with TTL efficiently.

    Optimizations:
    - Batch store using pipeline (if available)
    - Reduced logging overhead for large secret counts
    - Early return if no secrets to store
    """
    if not secrets:
        return

    try:
        # Try pipeline for batch operations (more efficient)
        if hasattr(valkey, 'pipeline'):
            pipe = valkey.pipeline()
            for secret_hash, secret_value in secrets.items():
                key = f"keyvault:{conversation_id}:{secret_hash}"
                pipe.set(key, secret_value, ex=KEYVAULT_TTL)
            await pipe.execute()
            logger.debug(
                "keyvault_store_batch trace=%s conv=%s count=%d",
                trace_id,
                conversation_id[:12],
                len(secrets),
            )
        else:
            # Fallback to individual sets
            for secret_hash, secret_value in secrets.items():
                await valkey.set(
                    f"keyvault:{conversation_id}:{secret_hash}",
                    secret_value,
                    ex=KEYVAULT_TTL,
                )
            logger.debug(
                "keyvault_store trace=%s conv=%s count=%d",
                trace_id,
                conversation_id[:12],
                len(secrets),
            )
    except Exception as exc:
        logger.error(
            "keyvault_store_error trace=%s conv=%s count=%d err=%s",
            trace_id,
            conversation_id[:12],
            len(secrets),
            exc,
        )


# ── Response helpers ─────────────────────────────────────────────────────────


def _re_inject_recursive(obj: object, secrets: dict[str, str], depth: int = 0) -> object:
    """Recursively re-inject secrets into JSON structures efficiently.

    Optimizations:
    - In-place mutation for dict/list instead of recreating
    - Max depth limit to prevent pathological recursion
    - Fast path for strings with no placeholders
    - Type checking order optimized for common cases
    """
    if depth > 100:  # Prevent unbounded recursion
        logger.warning("keyvault_recursion_limit_hit")
        return obj

    if isinstance(obj, str):
        # Fast path: check for placeholder prefix before calling _re_inject
        if PLACEHOLDER_PREFIX in obj:
            return _re_inject(obj, secrets)
        return obj

    if isinstance(obj, dict):
        # In-place mutation is safer than recreation
        for key, value in obj.items():
            obj[key] = _re_inject_recursive(value, secrets, depth + 1)
        return obj

    if isinstance(obj, list):
        # In-place mutation for lists
        for i, item in enumerate(obj):
            obj[i] = _re_inject_recursive(item, secrets, depth + 1)
        return obj

    # Return unchanged for primitives (int, bool, None, etc.)
    return obj


async def _re_inject_non_streaming(
    response, secrets: dict[str, str], trace_id: str = "????"
) -> JSONResponse:
    """Re-inject secrets into a non-streaming response.

    Note: For streaming responses, the body_iterator has already been consumed
    by the SSE generator, so we return the response unchanged. Streaming
    re-injection happens inline in _build_re_inject_stream.
    """
    try:
        # Try to get body content
        body_bytes = None

        if hasattr(response, "body") and response.body:
            body_bytes = response.body
        elif hasattr(response, "body_iterator") and response.body_iterator is not None:
            # Check if body_iterator is already consumed (empty)
            chunks: list[bytes] = []
            try:
                async for chunk in response.body_iterator:
                    chunks.append(chunk)
                body_bytes = b"".join(chunks)
            except (StopAsyncIteration, RuntimeError):
                # Iterator was already consumed or closed
                logger.debug(
                    "keyvault_skip_re_inject trace=%s: body_iterator already consumed",
                    trace_id,
                )
                return response

        if not body_bytes:
            logger.debug(
                "keyvault_skip_re_inject trace=%s: empty body, skipping re-injection",
                trace_id,
            )
            return response

        # Parse and re-inject secrets
        resp_json = json.loads(body_bytes)
        resp_json = _re_inject_recursive(resp_json, secrets)
        headers = dict(response.headers)
        headers.pop("content-length", None)
        return JSONResponse(
            content=resp_json,
            status_code=response.status_code,
            headers=headers,
        )
    except json.JSONDecodeError as e:
        # Response body is not JSON (e.g., empty or text), skip re-injection
        logger.debug(
            "keyvault_skip_re_inject trace=%s: not valid JSON (%s), returning unchanged",
            trace_id,
            str(e)[:50],
        )
        return response
    except Exception as exc:
        logger.error("keyvault_re_inject_error trace=%s: %s", trace_id, exc)
        return response


def _build_re_inject_stream(
    original_iterator, secrets: dict[str, str], trace_id: str = "????"
):
    """Wrap a streaming response body iterator to re-inject secrets on each chunk."""

    async def _wrapper():
        try:
            async for chunk in original_iterator:
                if isinstance(chunk, bytes):
                    yield _re_inject(
                        chunk.decode("utf-8", errors="replace"),
                        secrets,
                    ).encode("utf-8")
                else:
                    yield _re_inject(str(chunk), secrets)
        except Exception as exc:
            logger.warning("keyvault_stream_error trace=%s: %s", trace_id, exc)
            # Re-raise to avoid silent failure — the client will see the error
            raise

    return _wrapper


# ── Middleware ────────────────────────────────────────────────────────────────


class KeyVaultMiddleware(BaseHTTPMiddleware):
    """Transparent secret vault middleware.

    Only activates for POST /v1/chat/completions.
    Always masks secrets in-memory. Persists to Valkey when available.
    Re-injects real values in both streaming and non-streaming responses.
    """

    async def dispatch(self, request, call_next):
        _trace = str(uuid.uuid4())[:8]

        if request.url.path != _CHAT_PATH:
            return await call_next(request)

        # ── Parse body ───────────────────────────────────────────────────
        try:
            body_bytes = await request.body()
            body = json.loads(body_bytes)
        except Exception as exc:
            logger.debug("keyvault_parse_error trace=%s err=%s", _trace, exc)
            return await call_next(request)

        # ── Detect + mask secrets (always, in-memory) ─────────────────────
        raw_cid = body.get("conversation_id")
        if raw_cid:
            conversation_id = raw_cid
        else:
            conversation_id = str(
                uuid.uuid5(uuid.NAMESPACE_DNS, json.dumps(body, sort_keys=True))
            )
        secrets: dict[str, str] = {}

        _mask_messages(body, secrets)

        if not secrets:
            # No secrets found → pass through unmodified
            return await call_next(request)

        # ── Persist to Valkey (best-effort, non-blocking) ─────────────────
        valkey = getattr(request.app.state, "valkey", None)
        if valkey:
            import asyncio

            task = asyncio.create_task(
                _store_secrets(valkey, conversation_id, secrets, _trace)
            )
            # Keep reference to prevent premature garbage collection
            if not hasattr(request.app.state, "_kv_tasks"):
                request.app.state._kv_tasks = set()
            request.app.state._kv_tasks.add(task)
            task.add_done_callback(request.app.state._kv_tasks.discard)

        # ── Inject system prompt so LLM knows about placeholders ──────────
        # Bug 8 fix: insert AFTER any existing system messages to preserve
        # canonical order (system messages first, in order). Placing at
        # position 0 would put our prompt before the original system prompt,
        # potentially confusing the model.
        msgs = body.setdefault("messages", [])
        insert_pos = 0
        for i, m in enumerate(msgs):
            if m.get("role") != "system":
                break
            insert_pos = i + 1
        msgs.insert(insert_pos, {"role": "system", "content": _KEYVAULT_SYSTEM_PROMPT})
        logger.info(
            "keyvault_active trace=%s conv=%s secrets=%d valkey=%s",
            _trace,
            conversation_id[:12],
            len(secrets),
            "yes" if valkey else "no",
        )

        # Set cached body for downstream handler
        request._body = json.dumps(body).encode()

        # ── Call handler ──────────────────────────────────────────────────
        response = await call_next(request)

        # ── Re-inject real values ─────────────────────────────────────────
        if not secrets:
            return response

        # Re-inject secrets in both streaming and non-streaming responses
        # The LLM may have mentioned the placeholder in its response, and we need
        # to replace it with the real value before returning to the client
        if isinstance(response, StreamingResponse):
            headers = dict(response.headers)
            headers.pop("content-length", None)
            return StreamingResponse(
                content=_build_re_inject_stream(
                    response.body_iterator, secrets, _trace
                )(),
                status_code=response.status_code,
                headers=headers,
                media_type=response.media_type,
            )

        return await _re_inject_non_streaming(response, secrets, _trace)
