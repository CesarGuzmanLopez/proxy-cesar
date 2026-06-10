"""LLM call orchestration with fallback across physical models.

Handles retryable errors, context-too-large detection, token-limit continuation,
and provider-specific cache optimisations.
"""

import logging
import time
import uuid


from litellm.types.utils import ModelResponse
from litellm.exceptions import (
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    RateLimitError,
    ServiceUnavailableError,
)

from src.adapters.cache.message_ordering import (
    canonicalize_message_order,
    sort_tool_definitions,
    stable_message_hash,
)
from src.adapters.cache.provider_cache import (
    apply_anthropic_cache_control,
    should_apply_cache_control,
)
from src.adapters.litellm import call_litellm
from src.config.pseudo_models import PhysicalModelSchema, ProxyConfigSchema
from src.config.settings import settings as _global_settings
from src.domain.capabilities import TurnCapabilities
from src.domain.errors import ContextTooLargeForAllModels, AllModelsFailed
from src.domain.types import Result, Ok, Err
from src.service.capability_detector import estimate_tokens
from src.service.chat_models import FallbackInfo
from src.service.compatibility import validate_physical_model_content
from src.service.smart_fallback import SmartFallback

logger = logging.getLogger(__name__)

__all__ = [
    "call_with_fallback",
    "_try_physical_model",
    "_log_model_call_result",
    "_build_context_too_large_error",
    "_build_all_models_failed_error",
    "_normalise_reasoning_param",
    "_resolve_api_key",
]


# ── Helpers used by _try_physical_model ─────────────────────────────────────


def _strip_reasoning_from_messages(call_messages: list[dict]) -> list[dict]:
    """Remove reasoning_content from assistant messages. Returns NEW list (no muta).

    This is critical because ``call_messages`` may share dict objects with
    ``ordered_messages`` that are reused across fallback attempts (e.g. first
    model strips → second model in fallback chain sees already-stripped messages).
    """
    result: list[dict] = []
    for msg in call_messages:
        if msg.get("role") == "assistant" and "reasoning_content" in msg:
            new_msg = dict(msg)
            del new_msg["reasoning_content"]
            result.append(new_msg)
        else:
            result.append(msg)
    return result


_REASONING_MAP = {
    "low": (2048, "low"),
    "medium": (8192, "medium"),
    "high": (16000, "high"),
    "xhigh": (32000, "high"),
    "max": (64000, "high"),
}


def _map_bool_thinking(thinking: bool, provider_lower: str) -> tuple[dict | None, str | None]:
    if not thinking:
        return ({"type": "disabled"}, None) if provider_lower == "anthropic" else (None, None)
    return ({"type": "enabled"}, None) if provider_lower == "anthropic" else (None, None)


def _map_str_thinking(tl: str, provider_lower: str) -> tuple[dict | None, str | None]:
    if tl in ("disabled",):
        return ({"type": "disabled"}, None) if provider_lower == "anthropic" else (None, None)
    if tl in ("auto",):
        return None, None
    if tl in ("enabled",):
        return ({"type": "enabled"}, None) if provider_lower == "anthropic" else (None, None)
    budget, effort = _REASONING_MAP.get(tl, (None, None))
    if provider_lower == "anthropic" and budget is not None:
        return {"type": "enabled", "budget_tokens": budget}, None
    if provider_lower == "openai" and effort is not None:
        return None, effort
    return ({"type": "enabled"}, None) if provider_lower == "anthropic" else (None, None)


def _normalise_reasoning_param(
    thinking: dict | str | bool | None,
    provider: str,
) -> tuple[dict | None, str | None]:
    """Normalise ``thinking`` into the format the target provider understands.

    Returns ``(thinking_dict, reasoning_effort)`` where exactly one is set:
      - Anthropic → ``(thinking_dict, None)``  (``thinking`` dict with budget_tokens)
      - OpenAI   → ``(None, reasoning_effort_string)``  (``reasoning_effort`` param)
      - Others   → ``(None, None)``  (auto — provider decides)
    """
    if thinking is None:
        return None, None

    provider_lower = provider.lower() if provider else ""

    if isinstance(thinking, bool):
        return _map_bool_thinking(thinking, provider_lower)

    if isinstance(thinking, str):
        return _map_str_thinking(thinking.lower(), provider_lower)

    if isinstance(thinking, dict):
        return (thinking, None) if provider_lower == "anthropic" else (None, None)

    return None, None


async def _resolve_api_key(phys, conversation_id: str | None = None, affinity=None) -> str | None:
    """Resolve API key from environment, with key slot rotation.

    If multiple keys exist for the same provider (e.g. OPENCODE_API_KEY_2),
    the conversation's key slot determines which one to use.
    Fallback to the default key (_1 / no suffix) if the slot-specific key
    is not configured.
    """
    if not phys.api_key_env:
        return None
    import os

    env_var = phys.api_key_env  # e.g. "OPENCODE_API_KEY"

    # Determine key slot for this conversation
    slot = 1
    if conversation_id and affinity:
        slot = await affinity.get_key_slot(conversation_id)

    # Try slot-specific key first
    if slot > 1:
        specific = os.environ.get(f"{env_var}_{slot}")
        if specific:
            return specific

    # Fallback to default key
    return os.environ.get(env_var) or None


# ── Single model call ────────────────────────────────────────────────────────


async def _try_physical_model(
    phys: PhysicalModelSchema,
    ordered_messages: list[dict],
    stream: bool,
    kwargs: dict,
    _est_input: int,
    _trace_id: str,
    conversation_id: str | None = None,
    affinity=None,
    timeout: float | None = None,
) -> tuple[ModelResponse | dict | None, str | None]:
    """Attempt to call a single physical model.

    Args:
        phys: Physical model schema.
        ordered_messages: Canonicalized message list.
        stream: Whether to stream the response.
        kwargs: Call keyword arguments (temperature, max_tokens, etc.).
        _est_input: Estimated input token count.
        _trace_id: Trace ID for logging.
        conversation_id: Conversation ID for key resolution.
        affinity: Affinity manager for key slot resolution.
        timeout: Override the default LLM call timeout (seconds).
            If None, ``DEFAULT_LLM_TIMEOUT_SECONDS`` is used.

    Returns (response, None) on success or (None, skip_reason) if skipped.
    skip_reason is one of ``"provider_disabled"`` or ``"context_too_large"``.
    """
    if disabled := _global_settings.disabled_providers_set:
        if disabled and phys.provider and phys.provider.lower() in disabled:
            logger.info(
                "llm_skip  | trace=%s model=%s provider=%s reason=provider_disabled "
                "disabled_providers=%s",
                _trace_id,
                phys.model,
                phys.provider,
                _global_settings.disabled_providers,
            )
            return None, "provider_disabled"

    if phys.context_window is not None and _est_input > phys.context_window:
        logger.warning(
            "llm_skip  | trace=%s model=%s context_window=%d "
            "input_est=%d reason=context_too_large",
            _trace_id,
            phys.model,
            phys.context_window,
            _est_input,
        )
        return None, "context_too_large"

    logger.info(
        "llm_call  | trace=%s attempt=0 model=%s stream=%s",
        _trace_id,
        phys.model,
        stream,
    )

    call_messages = list(ordered_messages)
    if phys.system_prompt:
        # Merge into first existing system message (don't add separate one).
        # Create a shallow copy of the list and a NEW dict for the system
        # message so that mutations don't leak into subsequent fallback
        # iterations that share the original ``ordered_messages``.
        merged = False
        for i, msg in enumerate(call_messages):
            if msg.get("role") == "system":
                call_messages[i] = {
                    **msg,
                    "content": phys.system_prompt + "\n\n" + (msg.get("content") or ""),
                }
                merged = True
                break
        if not merged:
            call_messages = [{"role": "system", "content": phys.system_prompt}] + call_messages
    provider = phys.provider.lower()
    model_prefix = phys.model.split("/")[0].lower() if "/" in phys.model else provider
    cache_provider = model_prefix if model_prefix in ("anthropic",) else provider
    cache_applied = False
    if should_apply_cache_control(cache_provider):
        call_messages = apply_anthropic_cache_control(call_messages)
        cache_applied = True
        logger.info(
            "⚡ cache_control provider=%s model=%s messages=%d",
            cache_provider,
            phys.model,
            len(call_messages),
        )

    api_base = phys.api_base or None
    api_key = await _resolve_api_key(phys, conversation_id, affinity)

    # Strip reasoning_content if the model has the flag set (e.g. DeepSeek)
    if phys.strip_reasoning:
        call_messages = _strip_reasoning_from_messages(call_messages)

    raw_thinking = kwargs.get("thinking", None)

    # Determine if this model actually supports the capability
    supports_anthropic = model_prefix == "anthropic" and (phys.thinking or False)
    supports_reasoning_effort = model_prefix == "openai" and (phys.reasoning_effort or False)

    # Only apply default_thinking if the model actually supports the capability
    if supports_anthropic or supports_reasoning_effort:
        if phys.default_thinking is not None and raw_thinking is None:
            raw_thinking = phys.default_thinking
            logger.debug(
                "fallback_thinking_default model=%s default=%s",
                phys.model,
                raw_thinking,
            )

    if supports_anthropic:
        reasoning_capability = "anthropic"
    elif supports_reasoning_effort:
        reasoning_capability = "openai"
    else:
        reasoning_capability = "other"

    if reasoning_capability != "other":
        thinking_dict, reasoning_effort = _normalise_reasoning_param(
            raw_thinking, reasoning_capability
        )
        logger.debug(
            "reasoning   | trace=%s model=%s raw=%s cap=%s",
            _trace_id,
            phys.model,
            raw_thinking,
            reasoning_capability,
        )
    else:
        thinking_dict, reasoning_effort = None, None

    call_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    call_kwargs.pop("thinking", None)
    # Force physical model defaults (override client values)
    if phys.temperature is not None:
        call_kwargs["temperature"] = phys.temperature
    if phys.top_p is not None:
        call_kwargs["top_p"] = phys.top_p
    if not phys.parallel_tools:
        call_kwargs["parallel_tool_calls"] = False

    if supports_anthropic and thinking_dict is not None:
        call_kwargs["thinking"] = thinking_dict
    elif supports_reasoning_effort and reasoning_effort is not None:
        call_kwargs["reasoning_effort"] = reasoning_effort

    response = await call_litellm(
        model=phys.model,
        messages=call_messages,
        stream=stream,
        api_base=api_base,
        api_key=api_key,
        timeout=timeout,
        **call_kwargs,
    )

    if not stream:
        try:
            response._proxy_provider = provider
            response._proxy_cache_optimization_applied = cache_applied
        except (AttributeError, TypeError) as e:
            logger.warning("response_parsing_error error=%s", str(e))

    return response, None


# ── Logging helper ───────────────────────────────────────────────────────────


def _log_model_call_result(
    response: ModelResponse | dict,
    phys: PhysicalModelSchema,
    stream: bool,
    _trace_id: str,
    elapsed: float,
) -> None:
    """Log LLM call result with cache info.

    Handles both LiteLLM ``ModelResponse`` objects and plain ``dict``
    responses (e.g. composite responses from token-limit fallback).
    """
    if stream:
        logger.info(
            "llm_ok    | trace=%s model=%s elapsed=%.1fs (streaming)",
            _trace_id,
            phys.model,
            elapsed,
        )
        return
    try:
        # Extract usage info for cache diagnostics
        usage = None
        if isinstance(response, dict):
            c = (response.get("choices") or [{}])[0].get("message", {}).get(
                "content", ""
            ) or ""
            fr = (response.get("choices") or [{}])[0].get("finish_reason", "?")
            usage = response.get("usage", {})
        else:
            c = response.choices[0].message.content or ""
            fr = response.choices[0].finish_reason
            usage = response.usage  # type: ignore[attr-defined]  # justification: litellm ModelResponse has runtime attrs not in type stubs

        # Extract cache tokens from response
        cache_hit = 0
        cache_write = 0
        cache_miss = 0
        prompt_tokens = 0
        if usage:
            prompt_tokens = (
                getattr(usage, "prompt_tokens", 0)
                if not isinstance(usage, dict) else usage.get("prompt_tokens", 0)
            )
            details = (
                getattr(usage, "prompt_tokens_details", None)
                if not isinstance(usage, dict)
                else usage.get("prompt_tokens_details", {})
            )
            if details:
                cache_hit = (
                    getattr(details, "cached_tokens", 0)
                    if not isinstance(details, dict)
                    else details.get("cached_tokens", 0)
                ) or 0
            cache_read = (
                getattr(usage, "cache_read_input_tokens", 0)
                if not isinstance(usage, dict)
                else usage.get("cache_read_input_tokens", 0)
            )
            cache_write = (
                getattr(usage, "cache_creation_input_tokens", 0)
                if not isinstance(usage, dict)
                else usage.get("cache_creation_input_tokens", 0)
            )
            cache_miss = (
                getattr(usage, "prompt_cache_miss_tokens", 0)
                if not isinstance(usage, dict)
                else usage.get("prompt_cache_miss_tokens", 0)
            )
            cache_hit = cache_hit or cache_read

        cache_str = ""
        if cache_hit:
            cache_str = f" cache_hit={cache_hit}"
        if cache_write:
            cache_str += f" cache_write={cache_write}"
        if cache_miss:
            cache_str += f" cache_miss={cache_miss}"

        logger.info(
            "llm_ok    | trace=%s model=%s elapsed=%.1fs content_len=%d finish=%s prompt_tokens=%d%s",
            _trace_id,
            phys.model,
            elapsed,
            len(c),
            fr,
            prompt_tokens,
            cache_str,
        )
    except (AttributeError, IndexError, KeyError, TypeError):
        logger.warning(
            "llm_ok    | trace=%s model=%s (unexpected format)",
            _trace_id,
            phys.model,
        )


# ── Error builders ───────────────────────────────────────────────────────────


def _build_context_too_large_error(
    _est_input: int,
    pseudo_model_schema,
    _context_skipped: list[str],
    _trace_id: str,
    elapsed: float,
) -> ContextTooLargeForAllModels:
    """Build domain error when all models are skipped due to context."""
    logger.error(
        "llm_fail   | trace=%s elapsed=%.1fs reason=all_context_too_large "
        "input_est=%d models=%s",
        _trace_id,
        elapsed,
        _est_input,
        _context_skipped,
    )
    return ContextTooLargeForAllModels(
        estimated_tokens=_est_input,
        context_skipped=_context_skipped,
        pseudo_model=pseudo_model_schema.display_name or "unknown",
    )


def _build_all_models_failed_error(
    fallback_info: FallbackInfo,
    pseudo_model_schema,
    last_error: Exception | None,
) -> AllModelsFailed:
    """Build domain error when all models failed (not just skipped)."""
    return AllModelsFailed(
        pseudo_model=pseudo_model_schema.display_name or "unknown",
        attempted=fallback_info.attempted_models,
        last_error=str(last_error),
    )


# ── Fallback orchestrator helpers ────────────────────────────────────────────


def _prepare_messages(
    messages: list[dict],
    kwargs: dict,
    _trace_id: str,
) -> tuple[list[dict], dict, str]:
    """Canonicalize message order, compute hash, and sort tools."""
    ordered_messages = canonicalize_message_order(messages)
    raw_tools = kwargs.get("tools")
    _msg_hash = stable_message_hash(ordered_messages)
    logger.debug(
        "message_integrity | trace=%s hash=%s system=%s tools=%s msgs=%d",
        _trace_id,
        _msg_hash,
        any(m.get("role") == "system" for m in ordered_messages),
        bool(raw_tools),
        len(ordered_messages),
    )
    if raw_tools:
        sorted_tools = sort_tool_definitions(raw_tools)
        kwargs["tools"] = sorted_tools
        first_name = sorted_tools[0].get("function", {}).get("name", "?")
        logger.debug(
            "tools_sorted | trace=%s count=%d first=%s",
            _trace_id,
            len(sorted_tools),
            first_name,
        )
    return ordered_messages, kwargs, _msg_hash


def _handle_length_continuation(
    response,
    phys: PhysicalModelSchema,
    ordered_messages: list[dict],
    accumulated_parts: list[tuple[str, str]],
    fallback_info: FallbackInfo,
    pseudo_model_schema,
    idx: int,
    _trace_id: str,
) -> bool:
    """Check if response hit token limit and should continue with next model.

    Returns True if continuation was triggered (caller should skip to next model).
    """
    last_model = idx >= len(pseudo_model_schema.physical_models) - 1
    if not getattr(pseudo_model_schema, "continue_on_length", False) or last_model:
        return False

    try:
        finish_reason = response.choices[0].finish_reason
        if finish_reason != "length":
            return False

        partial = response.choices[0].message.content or ""
        accumulated_parts.append((phys.model, partial))
        fallback_info.applied = True
        fallback_info.reason = f"token_limit_continued: {phys.model}"
        fallback_info.attempted_models[-1] = f"{phys.model} (token_limit)"
        ordered_messages.append({"role": "assistant", "content": partial})
        logger.info(
            "llm_token_limit | trace=%s model=%s "
            "content_len=%d models_remaining=%d",
            _trace_id,
            phys.model,
            len(partial),
            len(pseudo_model_schema.physical_models) - idx - 1,
        )
        return True
    except (AttributeError, IndexError) as e:
        logger.warning("tool_call_extraction_error error=%s", str(e))
        return False


def _build_composite_response(
    accumulated_parts: list[tuple[str, str]],
    response,
    _est_input: int,
    _trace_id: str,
) -> dict:
    """Build a composite response dict from accumulated partial responses."""
    try:
        final_content = response.choices[0].message.content or ""
        finish_reason = response.choices[0].finish_reason
    except (AttributeError, IndexError):
        final_content = ""
        finish_reason = "stop"

    accumulated_parts.append((
        getattr(response, "model", "unknown"),
        final_content,
    ))
    composite_text = "".join(c for _, c in accumulated_parts)

    response_dict = {
        "id": getattr(response, "id", f"chatcmpl-{_trace_id}"),
        "object": "chat.completion",
        "model": getattr(response, "model", "unknown"),
        "choices": [
            {
                "message": {"role": "assistant", "content": composite_text},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": _est_input,
            "completion_tokens": len(composite_text),
        },
    }
    prov_h = getattr(response, "_provider_response_headers", None)
    if prov_h:
        response_dict["provider_headers"] = prov_h
    return response_dict


def _build_accumulated_only_response(
    accumulated_parts: list[tuple[str, str]],
    _est_input: int,
    _trace_id: str,
) -> dict:
    """Build composite response when all models were exhausted."""
    composite_text = "".join(c for _, c in accumulated_parts)
    last_model_name = accumulated_parts[-1][0]
    return {
        "id": f"chatcmpl-{_trace_id}",
        "object": "chat.completion",
        "model": last_model_name,
        "choices": [
            {
                "message": {"role": "assistant", "content": composite_text},
                "finish_reason": "length",
            }
        ],
        "usage": {
            "prompt_tokens": _est_input,
            "completion_tokens": len(composite_text),
        },
    }


async def _record_fallback_metric(
    smart_fallback: SmartFallback | None,
    conversation_id: str | None,
    model: str,
    elapsed: float,
    success: bool,
    error: str | None = None,
) -> None:
    """Record success/failure metric with SmartFallback."""
    if smart_fallback and conversation_id:
        await smart_fallback.record_call(
            conversation_id,
            model,
            elapsed_ms=int(elapsed * 1000),
            success=success,
            error=error,
        )


# ── Model ban system ──────────────────────────────────────────────────────────
# When a model fails 3 times within 2 minutes, it is banned for 1 hour.
# On success, the failure counter is reset.

_BAN_PREFIX = "model_ban:"
_FAIL_PREFIX = "model_fail:"
_BAN_TTL = 3600  # 1 hour ban
_FAIL_WINDOW = 120  # 2 minutes sliding window
_FAIL_THRESHOLD = 3  # 3 failures → ban


async def _is_model_banned(valkey, model_name: str) -> bool:
    """Check if a model is currently banned."""
    if valkey is None:
        return False
    try:
        return bool(await valkey.exists(f"{_BAN_PREFIX}{model_name}"))
    except Exception:
        return False


async def _record_model_failure(valkey, model_name: str) -> bool:
    """Record a model failure. Returns True if ban threshold reached."""
    if valkey is None:
        return False
    try:
        key = f"{_FAIL_PREFIX}{model_name}"
        count = await valkey.incr(key)
        # Set TTL on first failure only (sliding window from first failure)
        if count == 1:
            await valkey.expire(key, _FAIL_WINDOW)
        if count >= _FAIL_THRESHOLD:
            await valkey.setex(f"{_BAN_PREFIX}{model_name}", _BAN_TTL, "1")
            await valkey.delete(key)
            logger.warning("model_banned model=%s failures=%d ban_ttl=%ds", model_name, count, _BAN_TTL)
            return True
        logger.info("model_failure model=%s count=%d/%d window=%ds", model_name, count, _FAIL_THRESHOLD, _FAIL_WINDOW)
        return False
    except Exception as e:
        logger.warning("model_ban_error model=%s error=%s", model_name, e)
        return False


async def _clear_model_ban(valkey, model_name: str) -> None:
    """Clear ban and failure counters on successful model call."""
    if valkey is None:
        return
    try:
        ban_key = f"{_BAN_PREFIX}{model_name}"
        fail_key = f"{_FAIL_PREFIX}{model_name}"
        await valkey.delete(ban_key)
        await valkey.delete(fail_key)
    except Exception:
        pass


# ── Main fallback orchestrator ───────────────────────────────────────────────


async def call_with_fallback(
    pseudo_model_schema,
    messages: list[dict],
    stream: bool = False,
    estimated_input: int | None = None,
    start_index: int = 0,
    conversation_id: str | None = None,
    valkey_client=None,
    affinity=None,
    turn_caps: TurnCapabilities | None = None,
    config: ProxyConfigSchema | None = None,
    **kwargs,
) -> Result[tuple[ModelResponse | dict, FallbackInfo], ContextTooLargeForAllModels | AllModelsFailed]:
    """Try each physical model in order. On retryable errors, move to next.

    Retryable errors: ServiceUnavailableError (503), RateLimitError (429),
    NotFoundError (404 — e.g. model not available via provider), and
    AuthenticationError (401 — expired / invalid key for a given provider).
    Any other exception propagates immediately (the request fails fast).

    Returns Ok((response, fallback_info)) on success,
    Err(ContextTooLargeForAllModels) or Err(AllModelsFailed) on failure.
    """
    fallback_info = FallbackInfo()
    last_error: Exception | None = None
    _trace_id = str(uuid.uuid4())[:8]
    _start = time.monotonic()

    smart_fallback: SmartFallback | None = None
    if valkey_client and conversation_id:
        smart_fallback = SmartFallback(valkey_client)

    _RETRYABLE = (
        ServiceUnavailableError,
        RateLimitError,
        NotFoundError,
        AuthenticationError,
        BadRequestError,
    )

    if "thinking" not in kwargs and pseudo_model_schema.default_thinking is not None:
        kwargs["thinking"] = pseudo_model_schema.default_thinking
        logger.debug(
            "thinking_default | pseudo=%s thinking=%s",
            getattr(pseudo_model_schema, "display_name", "?"),
            pseudo_model_schema.default_thinking,
        )

    _est_input = (
        estimated_input if estimated_input is not None else await estimate_tokens(messages)
    )
    _context_skipped: list[str] = []

    ordered_messages, kwargs, _ = _prepare_messages(messages, kwargs, _trace_id)
    accumulated_parts: list[tuple[str, str]] = []

    for idx, phys in enumerate(pseudo_model_schema.physical_models):
        if idx < start_index:
            continue

        # Check if this model is banned
        if await _is_model_banned(valkey_client, phys.model):
            logger.warning(
                "llm_skip_banned | trace=%s model=%s reason=model_banned_1h",
                _trace_id,
                phys.model,
            )
            fallback_info.applied = True
            fallback_info.reason = f"model_banned: {phys.model}"
            fallback_info.attempted_models.append(f"{phys.model} (banned)")
            continue

        # Re-validate content compatibility for ALL models.
        # The primary model's content was already validated upstream, but
        # some providers (Xiaomi/MiMo) reject delegated content with
        # "Param Incorrect". Skip the primary model if it can't handle
        # the content directly — let it fall through to models that can.
        if turn_caps:
            delegation = validate_physical_model_content(turn_caps, phys)
            if delegation:
                if idx == start_index:
                    # Primary model doesn't support this content — skip to fallback
                    logger.info(
                        "content_skip_primary trace=%s model=%s reason=content_not_supported",
                        _trace_id, phys.model,
                    )
                    fallback_info.applied = True
                    fallback_info.reason = f"content_not_supported: {phys.model}"
                    fallback_info.attempted_models.append(f"{phys.model} (content_unsupported)")
                    continue
                # Fallback model: transform content for compatibility
                logger.info(
                    "content_fallback_delegation idx=%d model=%s action=%s",
                    idx, phys.model, delegation.get("action"),
                )
                if valkey_client:
                    from src.service.tool_detector import replace_base64_with_blob_refs
                    ordered_messages, _ = await replace_base64_with_blob_refs(
                        ordered_messages, conversation_id, valkey_client, config,
                    )

        try:
            response, skip_reason = await _try_physical_model(
                phys, ordered_messages, stream, kwargs, _est_input, _trace_id,
                conversation_id=conversation_id, affinity=affinity,
            )

            if response is None:
                _context_skipped.append(phys.model)
                fallback_info.applied = True
                fallback_info.reason = f"model_skipped/{skip_reason}: {phys.model}"
                fallback_info.attempted_models.append(
                    f"{phys.model} (skipped/{skip_reason})"
                )
                await _record_model_failure(valkey_client, phys.model)
                continue

            # Success — clear any previous ban/failures
            await _clear_model_ban(valkey_client, phys.model)
            fallback_info.attempted_models.append(phys.model)

            if _handle_length_continuation(
                response, phys, ordered_messages, accumulated_parts,
                fallback_info, pseudo_model_schema, idx, _trace_id
            ):
                _est_input = await estimate_tokens(ordered_messages)
                continue

            if accumulated_parts:
                response = _build_composite_response(
                    accumulated_parts, response, _est_input, _trace_id
                )

            elapsed = time.monotonic() - _start
            await _record_fallback_metric(
                smart_fallback, conversation_id, phys.model, elapsed, success=True
            )
            _log_model_call_result(response, phys, stream, _trace_id, elapsed)
            return Ok((response, fallback_info))

        except _RETRYABLE as e:
            last_error = e
            fallback_info.attempted_models.append(phys.model)
            fallback_info.applied = True
            fallback_info.reason = f"{type(e).__name__}: {phys.model}"
            elapsed = time.monotonic() - _start
            await _record_fallback_metric(
                smart_fallback, conversation_id, phys.model, elapsed,
                success=False, error=str(e)[:100]
            )
            logger.warning(
                "llm_fallback | trace=%s model=%s error=%s detail=%s elapsed=%.1fs api_base=%s",
                _trace_id,
                phys.model,
                type(e).__name__,
                str(e)[:200],
                elapsed,
                getattr(phys, "api_base", "default"),
            )
            await _record_model_failure(valkey_client, phys.model)
            continue

    elapsed = time.monotonic() - _start

    if accumulated_parts:
        response_dict = _build_accumulated_only_response(
            accumulated_parts, _est_input, _trace_id
        )
        logger.info(
            "llm_accumulated_return | trace=%s models=%d total_len=%d finish=length",
            _trace_id,
            len(accumulated_parts),
            len(response_dict["choices"][0]["message"]["content"]),
        )
        return Ok((response_dict, fallback_info))

    if len(_context_skipped) == len(pseudo_model_schema.physical_models):
        error = _build_context_too_large_error(
            _est_input, pseudo_model_schema, _context_skipped, _trace_id, elapsed
        )
        return Err(error)

    logger.error(
        "llm_fail   | trace=%s elapsed=%.1fs models=%s last_error=%s",
        _trace_id,
        elapsed,
        fallback_info.attempted_models,
        last_error,
    )
    all_error = _build_all_models_failed_error(
        fallback_info, pseudo_model_schema, last_error
    )
    return Err(all_error)
