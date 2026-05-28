"""LLM call orchestration with fallback across physical models.

Handles retryable errors, context-too-large detection, token-limit continuation,
and provider-specific cache optimisations.
"""

import logging
import re
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
from src.config.pseudo_models import PhysicalModelSchema
from src.config.settings import settings as _global_settings
from src.domain.types import Result, Ok, Err
from src.domain.errors import ContextTooLargeForAllModels, AllModelsFailed
from src.service.capability_detector import estimate_tokens
from src.service.chat_models import FallbackInfo

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


def _normalise_reasoning_param(
    thinking: dict | str | bool | None,
    provider: str,
) -> tuple[dict | None, str | None]:
    """Normalise ``thinking`` into the format the target provider understands.

    Returns ``(thinking_dict, reasoning_effort)`` where exactly one is set:
      - Anthropic → ``(thinking_dict, None)``  (``thinking`` dict with budget_tokens)
      - OpenAI   → ``(None, reasoning_effort_string)``  (``reasoning_effort`` param)
      - Others   → ``(None, None)``  (auto — provider decides)

    Accepts ``thinking`` in these forms:
      - ``None`` → auto
      - ``True`` / ``"enabled"`` → enabled with provider default budget
      - ``False`` / ``"disabled"`` → auto (disabled = don't send)
      - ``"low"``, ``"medium"``, ``"high"``, ``"xhigh"``, ``"max"`` → effort mapped per provider
      - ``"auto"`` → auto
      - ``{"type": ..., "budget_tokens": N}`` → passthrough for Anthropic, auto for others
    """
    _EFFORT_TO_BUDGET = {
        "low": 2048,
        "medium": 8192,
        "high": 16000,
        "xhigh": 32000,
        "max": 64000,
    }
    # Map extended effort strings (xhigh, max) to max OpenAI tier
    _EFFORT_TO_REASONING = {
        "low": "low",
        "medium": "medium",
        "high": "high",
        "xhigh": "high",
        "max": "high",
    }

    provider_lower = provider.lower() if provider else ""
    is_anthropic = provider_lower == "anthropic"
    is_openai = provider_lower == "openai"

    if thinking is None:
        return None, None

    if isinstance(thinking, bool):
        if not thinking:
            # Explicit disabled: Anthropic gets {"type": "disabled"}
            if is_anthropic:
                return {"type": "disabled"}, None
            return None, None  # other providers → auto
        # True / enabled
        if is_anthropic:
            return {"type": "enabled"}, None
        return None, None  # non-Anthropic → auto (no budget param to send)

    if isinstance(thinking, str):
        tl = thinking.lower()
        if tl == "disabled":
            if is_anthropic:
                return {"type": "disabled"}, None
            return None, None
        if tl in ("auto",):
            return None, None
        if tl == "enabled":
            if is_anthropic:
                return {"type": "enabled"}, None
            return None, None
        # Effort string
        if is_anthropic:
            budget = _EFFORT_TO_BUDGET.get(tl)
            if budget is not None:
                return {"type": "enabled", "budget_tokens": budget}, None
            return {"type": "enabled"}, None  # unknown string → enabled, no budget
        if is_openai:
            effort = _EFFORT_TO_REASONING.get(tl)
            if effort is not None:
                return None, effort
            return None, None  # unknown string → auto
        return None, None  # other providers → auto

    # Dict passthrough
    if isinstance(thinking, dict):
        if is_anthropic:
            return thinking, None
        return None, None  # non-Anthropic → auto

    return None, None


def _resolve_api_key(phys) -> str | None:
    """Resolve API key from environment if the physical model has api_key_env set."""
    if not phys.api_key_env:
        return None
    import os

    return os.environ.get(phys.api_key_env) or None


# ── Single model call ────────────────────────────────────────────────────────


async def _try_physical_model(
    phys: PhysicalModelSchema,
    ordered_messages: list[dict],
    stream: bool,
    kwargs: dict,
    _est_input: int,
    _trace_id: str,
) -> tuple[ModelResponse | dict, str | None]:
    """Attempt to call a single physical model.

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

    call_messages = ordered_messages
    provider = phys.provider.lower()
    model_prefix = phys.model.split("/")[0].lower() if "/" in phys.model else provider
    cache_provider = model_prefix if model_prefix in ("anthropic",) else provider
    cache_applied = False
    if should_apply_cache_control(cache_provider):
        call_messages = apply_anthropic_cache_control(ordered_messages)
        cache_applied = True
        logger.debug(
            "cache_control applied provider=%s model=%s messages=%d",
            cache_provider,
            phys.model,
            len(call_messages),
        )

    api_base = phys.api_base or None
    api_key = _resolve_api_key(phys)

    raw_thinking = kwargs.get("thinking", None)

    # Determine reasoning capability from model prefix + provider.
    # Only actual OpenAI o-series models (o1, o3, o4-mini, etc.) support
    # reasoning_effort. Models with openai/ prefix that are NOT actual
    # OpenAI models (e.g. kimi-k2.5, qwen3.6-plus) get auto.
    supports_anthropic = model_prefix == "anthropic" or provider == "anthropic"
    is_openai_reasoning_model = bool(re.search(r"/(?:o[1-9]\d*|o4-mini|o1-mini)\b", phys.model))
    supports_reasoning_effort = is_openai_reasoning_model
    if supports_anthropic:
        reasoning_capability = "anthropic"
    elif supports_reasoning_effort:
        reasoning_capability = "openai"
    else:
        reasoning_capability = "other"

    thinking_dict, reasoning_effort = _normalise_reasoning_param(raw_thinking, reasoning_capability)
    logger.debug(
        "reasoning   | trace=%s model=%s raw=%s cap=%s",
        _trace_id,
        phys.model,
        raw_thinking,
        reasoning_capability,
    )

    call_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    call_kwargs.pop("thinking", None)  # Remove raw — replaced by formatted version below

    if supports_anthropic and thinking_dict is not None:
        call_kwargs["thinking"] = thinking_dict
    elif supports_reasoning_effort and reasoning_effort is not None:
        call_kwargs["reasoning_effort"] = reasoning_effort
    # else: auto — no param sent, provider decides

    response = await call_litellm(
        model=phys.model,
        messages=call_messages,
        stream=stream,
        api_base=api_base,
        api_key=api_key,
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
    """Log LLM call result.

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
        if isinstance(response, dict):
            c = (response.get("choices") or [{}])[0].get("message", {}).get(
                "content", ""
            ) or ""
            fr = (response.get("choices") or [{}])[0].get("finish_reason", "?")
        else:
            c = response.choices[0].message.content or ""
            fr = response.choices[0].finish_reason
        logger.info(
            "llm_ok    | trace=%s model=%s elapsed=%.1fs content_len=%d finish=%s",
            _trace_id,
            phys.model,
            elapsed,
            len(c),
            fr,
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
    context_skipped_note: str,
) -> AllModelsFailed:
    """Build domain error when all models failed (not just skipped)."""
    return AllModelsFailed(
        pseudo_model=pseudo_model_schema.display_name or "unknown",
        attempted=fallback_info.attempted_models,
        last_error=str(last_error),
    )


# ── Main fallback orchestrator ───────────────────────────────────────────────


async def call_with_fallback(
    pseudo_model_schema,
    messages: list[dict],
    stream: bool = False,
    estimated_input: int | None = None,
    start_index: int = 0,
    **kwargs,
) -> tuple:
    """Try each physical model in order. On retryable errors, move to next.

    Retryable errors: ServiceUnavailableError (503), RateLimitError (429),
    NotFoundError (404 — e.g. model not available via provider), and
    AuthenticationError (401 — expired / invalid key for a given provider).
    Any other exception propagates immediately (the request fails fast).

    Sprint 7: applies provider-specific cache optimizations (Anthropic cache_control)
    and tracks cache destruction on fallback.

    Sprint 11: token-limit fallback — if a non-streaming model finishes with
    ``finish_reason="length"`` and there are more physical models available,
    the partial response is appended as an assistant message and the next
    model is called to continue (composite response built at the end).

    ``start_index`` (default 0) allows resuming from a specific position in
    the physical models list — used by streaming continuation.
    """
    fallback_info = FallbackInfo()
    last_error: Exception | None = None
    _trace_id = str(uuid.uuid4())[:8]
    _start = time.monotonic()

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
        estimated_input if estimated_input is not None else estimate_tokens(messages)
    )
    _context_skipped: list[str] = []

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

    accumulated_parts: list[tuple[str, str]] = []

    for idx, phys in enumerate(pseudo_model_schema.physical_models):
        if idx < start_index:
            continue

        try:
            response, skip_reason = await _try_physical_model(
                phys, ordered_messages, stream, kwargs, _est_input, _trace_id
            )

            if response is None:
                _context_skipped.append(phys.model)
                fallback_info.applied = True
                fallback_info.reason = f"model_skipped/{skip_reason}: {phys.model}"
                fallback_info.attempted_models.append(
                    f"{phys.model} (skipped/{skip_reason})"
                )
                continue

            fallback_info.attempted_models.append(phys.model)

            last_model = idx >= len(pseudo_model_schema.physical_models) - 1
            if (
                not stream
                and not last_model
                and getattr(pseudo_model_schema, "continue_on_length", False)
            ):
                try:
                    finish_reason = response.choices[0].finish_reason
                    if finish_reason == "length":
                        partial = response.choices[0].message.content or ""
                        accumulated_parts.append((phys.model, partial))
                        fallback_info.applied = True
                        fallback_info.reason = f"token_limit_continued: {phys.model}"
                        fallback_info.attempted_models[-1] = (
                            f"{phys.model} (token_limit)"
                        )
                        ordered_messages.append(
                            {"role": "assistant", "content": partial}
                        )
                        _est_input = estimate_tokens(ordered_messages)
                        logger.info(
                            "llm_token_limit | trace=%s model=%s "
                            "content_len=%d models_remaining=%d",
                            _trace_id,
                            phys.model,
                            len(partial),
                            len(pseudo_model_schema.physical_models) - idx - 1,
                        )
                        continue
                except (AttributeError, IndexError) as e:
                    logger.warning("tool_call_extraction_error error=%s", str(e))

            if accumulated_parts:
                try:
                    final_content = response.choices[0].message.content or ""
                    finish_reason = response.choices[0].finish_reason
                except (AttributeError, IndexError):
                    final_content = ""
                    finish_reason = "stop"
                accumulated_parts.append((phys.model, final_content))

                composite_text = "".join(c for _, c in accumulated_parts)
                response_dict = {
                    "id": getattr(response, "id", f"chatcmpl-{_trace_id}"),
                    "object": "chat.completion",
                    "model": getattr(response, "model", phys.model),
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": composite_text,
                            },
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
                response = response_dict

            elapsed = time.monotonic() - _start
            _log_model_call_result(response, phys, stream, _trace_id, elapsed)
            return response, fallback_info

        except _RETRYABLE as e:
            last_error = e
            fallback_info.attempted_models.append(phys.model)
            fallback_info.applied = True
            fallback_info.reason = f"{type(e).__name__}: {phys.model}"
            logger.warning(
                "llm_fallback | trace=%s model=%s error=%s elapsed=%.1fs",
                _trace_id,
                phys.model,
                type(e).__name__,
                time.monotonic() - _start,
            )
            continue

    elapsed = time.monotonic() - _start

    if accumulated_parts:
        composite_text = "".join(c for _, c in accumulated_parts)
        last_model_name = accumulated_parts[-1][0]
        response_dict = {
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
        logger.info(
            "llm_accumulated_return | trace=%s models=%d total_len=%d finish=length",
            _trace_id,
            len(accumulated_parts),
            len(composite_text),
        )
        return response_dict, fallback_info

    if len(_context_skipped) == len(pseudo_model_schema.physical_models):
        error = _build_context_too_large_error(
            _est_input, pseudo_model_schema, _context_skipped, _trace_id, elapsed
        )
        raise ValueError(f"ContextTooLargeForAllModels: {error}")

    context_skipped_note = ""
    if _context_skipped:
        context_skipped_note = f" ({len(_context_skipped)} skipped: context too large)"

    logger.error(
        "llm_fail   | trace=%s elapsed=%.1fs models=%s last_error=%s",
        _trace_id,
        elapsed,
        fallback_info.attempted_models,
        last_error,
    )
    error = _build_all_models_failed_error(
        fallback_info, pseudo_model_schema, last_error, context_skipped_note
    )
    raise ValueError(f"AllModelsFailed: {error}")
