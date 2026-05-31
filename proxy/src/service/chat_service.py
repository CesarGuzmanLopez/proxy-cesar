"""Chat orchestration service.

Central business logic for POST /v1/chat/completions.
Uses Result monad for domain errors, HTTPException for transport errors (routers only).
python.md §3: errors as data in domain, exceptions at boundary.

Capabilities:
- Pseudo-model resolution with affinity and fallback chains
- Capability detection, compatibility validation, threshold guards, tool filtering
- Canonical tool storage, tiktoken counting, thinking blocks, tool edge cases
- Pre-compaction, continuous compaction, external compaction detection
"""

# ── Service layer uses Result monad for domain errors ──────────────────────
# HTTPException is only raised at the router boundary, not in service logic.
# python.md §3: errors as data in domain, exceptions at boundary.

import asyncio
import logging
import time
import uuid

from sqlalchemy.orm import selectinload

from src.adapters.db.models import Conversation
from src.domain.affinity import AffinityPort
from src.domain.ports import AsyncSessionPort
from src.domain.types import Result, Ok, Err
from src.domain.errors import InputExceedsThreshold, ContextUnusable
from src.api.metrics import metrics
from src.config.pseudo_models import ProxyConfigSchema, PseudoModelSchema
from src.domain.capabilities import SessionCapabilities, TurnCapabilities
from src.service.capability_detector import (
    detect_turn_capabilities,
    estimate_tokens,
    load_session_capabilities,
)
from src.service.chat_models import ChatResult, FallbackInfo, SaveContext
from src.service.compatibility import validate_physical_model_content
from src.service.context_alert import ContextAlert, get_context_alert
from src.service.model_resolver import (
    build_passthrough_pseudo_model,
    normalize_model_name,
)
from src.service.router_llm.suggester import evaluate_complexity, is_downgrade
from src.service.threshold_guard import check_input_threshold
from src.service.tool_filter import get_eligible_models, is_pinned_model_eligible
from src.service.tool_detector import replace_base64_with_blob_refs
from src.service.pipeline_trace import PipelineTrace
from src.service.multimedia.image_processor import (
    _is_svg_data_url,
    degrade_image,
)

# ── Split-module imports ──────────────────────────────────────────────────────
from src.service.chat_fallback import call_with_fallback, _resolve_api_key
from src.service.chat_messages import build_conversation_messages, handle_auto_describe
from src.service.chat_persistence import (
    _save_and_return,
    _suggest_higher_threshold_models,
)

# ── Re-exports for backward compatibility ────────────────────────────────────
from src.adapters.cache.message_ordering import canonicalize_message_order
from src.service.chat_fallback import _normalise_reasoning_param, _try_physical_model

logger = logging.getLogger(__name__)

__all__ = [
    "process_chat_request",
    "_resolve_and_validate",
    "_parse_uuid",
    "_resolve_session_conv_and_models",
    "_check_input_threshold",
    "_apply_content_delegation",
    "_build_command_chat_result",
    "_check_context_usable",
    "_extract_cmd_name",
    "evaluate_router_suggestion",
    # Re-exports
    "call_with_fallback",
    "_try_physical_model",
    "_normalise_reasoning_param",
    "_resolve_api_key",
    "build_conversation_messages",
    "handle_auto_describe",
    "_suggest_higher_threshold_models",
    "canonicalize_message_order",
]


# ── Orchestrator ──────────────────────────────────────────────────────────────


# ── SVG image pre-processing ──────────────────────────────────────────────────


def _preprocess_svg_images(messages: list[dict]) -> list[dict]:
    """Convert SVG images to JPEG before any model sees them.

    Most vision models and providers do not support SVG format. This step
    rasterizes SVGs to JPEG (via cairosvg + PIL) so the model can process
    them regardless of whether delegation is triggered (models with vision=true
    send images directly without going through the delegation pipeline).

    Returns a new list of messages with SVG images converted to JPEG.
    """
    result: list[dict] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            result.append(msg)
            continue
        new_parts: list[dict] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = part.get("image_url", {})
                if isinstance(url, dict) and _is_svg_data_url(url.get("url", "")):
                    svg_url = url["url"]
                    jpeg_url = degrade_image(svg_url)
                    if jpeg_url.startswith("data:image/jpeg"):
                        logger.info(
                            "svg_to_jpeg url_len=%d result_len=%d",
                            len(svg_url),
                            len(jpeg_url),
                        )
                        new_parts.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": jpeg_url},
                            }
                        )
                        continue
            # Fallthrough: keep original part
            new_parts.append(part)
        if new_parts != content:
            result.append({**msg, "content": new_parts})
        else:
            result.append(msg)
    return result


async def process_chat_request(
    model: str,
    messages: list[dict],
    conversation_id: str | None,
    stream: bool,
    config: ProxyConfigSchema,
    affinity: AffinityPort,
    db: AsyncSessionPort,
    temperature: float | None = None,
    max_tokens: int | None = None,
    tools: list[dict] | None = None,
    tool_choice: str | dict | None = None,
    valkey=None,
    thinking: dict | str | bool | None = None,
    trace: PipelineTrace | None = None,
) -> ChatResult:
    """Execute the full chat completion flow.

    Each logical step is delegated to a focused helper function.
    """
    _req_id = str(uuid.uuid4())[:8]
    _t0 = time.monotonic()

    # Step 0: Preprocess SVG images → JPEG (most vision models don't support SVG)
    messages = _preprocess_svg_images(messages)
    logger.info(
        "req_start | trace=%s model=%s conv=%s messages=%d tools=%s stream=%s",
        _req_id,
        model,
        conversation_id or "new",
        len(messages),
        bool(tools),
        stream,
    )

    # Step 1-3: Resolve model and detect capabilities (no content validation yet)
    # Content validation happens AFTER physical model selection
    pseudo_model_name, pm_schema, turn_caps = _resolve_and_validate(
        model,
        messages,
        tools,
        config,
    )
    # feature track request metrics
    metrics.record_request(pseudo_model_name)
    conv_id = conversation_id or str(uuid.uuid4())
    conv_uuid = _parse_uuid(conv_id)

    # Step 4-11: Load session, check switch, load/create conversation, resolve model, set affinity
    (
        existing_affinity,
        session_caps,
        physical_model,
        provider,
        selected_phys_model,
        tools_filter,
        conv,
        is_new,
    ) = await _resolve_session_conv_and_models(
        db,
        affinity,
        conv_id,
        conv_uuid,
        pseudo_model_name,
        pm_schema,
    )

    # NEW: Validate that the SELECTED physical model can handle the incoming content
    # (this happens AFTER model selection, not before like the old logic)
    delegation = validate_physical_model_content(turn_caps, selected_phys_model)

    # Debug logging for content delegation
    logger.info(
        "content_validation trace=%s conv=%s model=%s has_images=%s model_vision=%s delegation=%s",
        _req_id,
        conversation_id[:12],
        physical_model,
        getattr(turn_caps, "has_images", False),
        getattr(selected_phys_model, "vision", False),
        bool(delegation),
    )

    # Apply image→tool delegation or blob storage for unsupported content
    # (based on the ACTUAL model's capabilities, not the pseudo-model's)
    if delegation:
        logger.info(
            "content_delegation_applying trace=%s conv=%s model=%s action=%s",
            _req_id,
            conversation_id[:12],
            physical_model,
            delegation.get("action"),
        )
    messages = await _apply_content_delegation(
        delegation, messages, conversation_id, affinity, valkey, config
    )

    # NEW: Inject guidance about blob extraction and available tools
    # This explains to the model that content was auto-extracted and it can use
    # its own specialized tools if it has them
    from src.service.tool_detector import inject_blob_extraction_guidance
    messages = inject_blob_extraction_guidance(messages)

    # feature Auto-describe images on pseudo-model switch
    auto_describe_meta: dict | None = None
    messages_for_llm: list[dict] = messages  # May be replaced by described version
    if conv is not None and not is_new and pseudo_model_name != conv.pseudo_model:
        desc_in_flight, auto_describe_meta = await handle_auto_describe(
            conv=conv,
            current_pseudo_name=conv.pseudo_model,
            config=config,
            db=db,
            pinned_physical_model=physical_model,
            in_flight_messages=messages,
        )
        if desc_in_flight is not None:
            messages_for_llm = desc_in_flight

    # ── Build messages with DB history as stable cache prefix ──────────
    # The DB has the canonical conversation history. Using it as a stable
    # prefix enables provider-side prefix caching (Anthropic cache_control,
    # DeepSeek disk cache, Go backend cache) across multi-turn conversations.
    if not is_new and conv is not None and conv.turns:
        active_messages = build_conversation_messages(conv, messages_for_llm)
        logger.debug(
            "chat_built_msgs conv=%s db_turns=%d client_msgs=%d result_msgs=%d",
            conv_id[:12], len(conv.turns), len(messages_for_llm), len(active_messages),
        )
    else:
        active_messages = messages_for_llm

    # Step 12: Check input threshold
    est_input = await estimate_tokens(messages_for_llm)
    threshold_check = _check_input_threshold(
        est_input, pm_schema, pseudo_model_name
    )
    match threshold_check:
        case Err(error=err):
            # Log and re-raise for router to handle
            logger.error(
                "req_failed | trace=%s reason=input_exceeds_threshold "
                "estimated=%d threshold=%d",
                _req_id,
                err.estimated,
                err.threshold,
            )
            raise ValueError(f"InputExceedsThreshold: {err}")
        case Ok():
            pass  # Threshold OK

    # feature Context alerts (total = history + current request)
    context_alert = get_context_alert(
        total_tokens=(conv.total_tokens if conv else 0) + est_input,
        context_window=pm_schema.context_window,
        conversation_id=conv_id,
    )
    context_check = _check_context_usable(context_alert, conv, pm_schema, conv_id)
    match context_check:
        case Err(error=err):
            logger.error(
                "req_failed | trace=%s reason=context_unusable "
                "context_tokens=%d context_window=%d",
                _req_id,
                err.context_tokens,
                err.context_window,
            )
            raise ValueError(f"ContextUnusable: {err}")
        case Ok():
            pass  # Context is usable

    # Router LLM evaluation runs in parallel with the LLM call (see below)
    # feature Router LLM — evaluate complexity (non-blocking, never changes model)

    # Log: LLM call outbound
    if trace:
        trace.llm_out(
            physical_model=physical_model,
            provider=provider,
            estimated_tokens=est_input,
        )

    # Router LLM and LLM call run in parallel — Router LLM evaluation is
    # non-blocking (suggestion is only used for logging/reporting, not model selection).
    router_task = asyncio.create_task(
        evaluate_router_suggestion(
            pm_schema=pm_schema,
            messages=active_messages,
            current_pseudo_name=pseudo_model_name,
            config=config,
        )
    )

    # Step 13: Call LiteLLM with fallback
    _t_llm_start = time.monotonic()
    response, fallback_info = await call_with_fallback(
        pseudo_model_schema=pm_schema,
        messages=active_messages,
        stream=stream,
        estimated_input=est_input,
        temperature=temperature,
        max_tokens=max_tokens,
        tools=tools,
        tool_choice=tool_choice,
        thinking=thinking,
        # FASE 2: SmartFallback parameters
        conversation_id=conv_id,
        valkey_client=valkey,
        affinity=affinity,
    )
    _t_llm_elapsed = time.monotonic() - _t_llm_start

    # Collect router suggestion result (fire-and-forget — model already chosen)
    router_suggestion: dict | None = None
    if router_task.done():
        try:
            router_suggestion = router_task.result()
        except Exception as exc:
            logger.warning("router_llm_result_error: %s", exc)
    else:
        # Router LLM still running — cancel it, model is already chosen
        router_task.cancel()
        try:
            await router_task
        except asyncio.CancelledError:
            pass  # nosonarc S7497: router_task is fire-and-forget — suppressing is intentional

    # Log: LLM response inbound
    _log_llm_response(response, physical_model, trace, _req_id)

    # feature Update physical_model from actual response when fallback occurred.
    physical_model, provider = await _update_physical_model(
        response,
        fallback_info,
        existing_affinity,
        physical_model,
        provider,
        affinity,
        conv_id,
        _req_id,
    )

    _elapsed = time.monotonic() - _t0
    logger.info(
        "req_end   | trace=%s conv=%s pseudo=%s physical=%s "
        "stream=%s fallback=%s tokens_in=%s elapsed=%.1fs",
        _req_id,
        conv_id[:12],
        pseudo_model_name,
        physical_model,
        stream,
        fallback_info.reason if fallback_info.applied else "none",
        est_input,
        _elapsed,
    )

    images_described = auto_describe_meta.get("images_described", 0) if auto_describe_meta else 0
    images_described_by = auto_describe_meta.get("described_by") if auto_describe_meta else None

    try:
        compatibility_info = _build_compatibility_info(turn_caps, pm_schema)

        return await _save_and_return(
            SaveContext(
                db=db,
                conv=conv,
                conv_uuid=conv_uuid,
                conv_id=conv_id,
                pseudo_model_name=pseudo_model_name,
                physical_model=physical_model,
                provider=provider,
                turn_caps=turn_caps,
                messages=messages,
                response=response,
                fallback_info=fallback_info,
                is_new_conversation=is_new,
                existing_affinity=existing_affinity,
                pm_schema=pm_schema,
                session_caps=session_caps,
                tools=tools,
                tool_choice=tool_choice,
                tools_filter=tools_filter,
                images_described=images_described,
                images_described_by=images_described_by,
                router_suggestion=router_suggestion,
                context_alert=context_alert,
                compatibility=compatibility_info,
            )
        )
    except Exception:
        logger.exception(
            "_save_and_return failed | pseudo=%s physical=%s conv=%s",
            pseudo_model_name,
            physical_model,
            conv_id[:12],
        )
        raise


def _log_llm_response(
    response, physical_model: str, trace: PipelineTrace | None, _req_id: str
) -> None:
    """Extract tokens/finish_reason from response and log via PipelineTrace."""
    try:
        finish_reason = None
        input_tokens = 0
        output_tokens = 0
        if hasattr(response, "choices") and response.choices:
            finish_reason = getattr(response.choices[0], "finish_reason", None)
        if hasattr(response, "usage"):
            input_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(response.usage, "completion_tokens", 0) or 0
        if trace:
            trace.llm_in(
                physical_model=physical_model,
                finish_reason=finish_reason,
                input_tokens=int(input_tokens),
                output_tokens=int(output_tokens),
            )
    except Exception as exc:
        logger.debug("llm_response_parsing_error err=%s", exc)


async def _update_physical_model(
    response,
    fallback_info: FallbackInfo,
    existing_affinity: str | None,
    physical_model: str,
    provider: str,
    affinity: AffinityPort,
    conv_id: str,
    _req_id: str,
) -> tuple[str, str]:
    """Update physical_model from actual response when fallback occurred."""
    if fallback_info.applied and existing_affinity:
        await affinity.record_failure(
            conv_id,
            existing_affinity,
            error=fallback_info.reason or "fallback_applied",
        )

    if fallback_info.applied:
        actual_model: str | None = None
        if hasattr(response, "model"):
            actual_model = response.model
        elif isinstance(response, dict):
            actual_model = response.get("model")
        if actual_model:
            logger.debug(
                "physical_model_update | trace=%s old=%s new=%s",
                _req_id,
                physical_model,
                actual_model,
            )
            physical_model = actual_model
            provider = physical_model.split("/")[0] if "/" in physical_model else provider
        elif fallback_info.attempted_models:
            last_model = fallback_info.attempted_models[-1]
            if "(" not in last_model:
                logger.debug(
                    "physical_model_update (fallback) | trace=%s old=%s new=%s",
                    _req_id,
                    physical_model,
                    last_model,
                )
                physical_model = last_model
                provider = physical_model.split("/")[0] if "/" in physical_model else provider

    await affinity.set(conv_id, physical_model)
    return physical_model, provider


def _build_compatibility_info(
    turn_caps: TurnCapabilities, pm_schema: object
) -> dict:
    """Build compatibility info for response metadata."""
    info: dict = {}
    if turn_caps.has_images and not any(m.vision for m in pm_schema.physical_models):
        info["images_delegated"] = True
    if turn_caps.has_audio and not any(m.audio for m in pm_schema.physical_models):
        info["audio_delegated"] = True
    if turn_caps.has_pdf and not any(m.vision for m in pm_schema.physical_models):
        info["pdf_delegated"] = True
    return info


# ── Step helpers ──────────────────────────────────────────────────────────────


def _resolve_and_validate(
    model: str,
    messages: list[dict],
    tools: list[dict] | None,
    config: ProxyConfigSchema,
) -> tuple[str, object, TurnCapabilities]:
    """Steps 1-3: Normalize model and detect capabilities.

    Note: Content validation is deferred until AFTER physical model selection
    to ensure we validate against the actual model that will handle the request,
    not just any model in the pseudo-model.

    Returns (resolved_model, pseudo_model_schema, turn_capabilities).
    """
    resolved = normalize_model_name(model, config)
    if resolved not in config.pseudo_models:
        pm = build_passthrough_pseudo_model(resolved)
    else:
        pm = config.pseudo_models[resolved]
    caps = detect_turn_capabilities(messages, tools)
    return resolved, pm, caps


def _parse_uuid(conv_id: str) -> uuid.UUID:
    """Parse conversation ID string to UUID."""
    try:
        return uuid.UUID(conv_id)
    except ValueError:
        return uuid.uuid5(uuid.NAMESPACE_DNS, conv_id)


async def _resolve_session_conv_and_models(
    db: AsyncSessionPort,
    affinity: AffinityPort,
    conv_id: str,
    conv_uuid: uuid.UUID,
    pseudo_model_name: str,
    pm_schema: PseudoModelSchema,
) -> tuple:
    """Steps 4-11: Load/create conversation, session, check switch, resolve physical model.

    Merged from _resolve_session_and_models + _load_or_create_conv
    to avoid loading Conversation twice (and to eagerly load .turns).
    """
    existing_affinity = await affinity.get(conv_id)

    conv = await db.get(
        Conversation, conv_uuid, options=[selectinload(Conversation.turns)]
    )
    is_new = conv is None

    session_caps = await load_session_capabilities(db, conv_uuid)

    eligible = get_eligible_models(pm_schema.physical_models, session_caps)
    tools_filter_applied = bool(
        session_caps.has_parallel_tools
        and len(eligible) < len(pm_schema.physical_models)
    )
    tools_filter_reason = "parallel_tools_required" if tools_filter_applied else None

    pinned = existing_affinity
    if (
        pinned
        and session_caps.has_parallel_tools
        and not is_pinned_model_eligible(pinned, eligible)
    ):
        pinned = None
        logger.info(
            "affinity_invalidated conv=%s reason=parallel_tools_incompatible",
            str(conv_uuid)[:12],
        )

    if pinned:
        selected_phys = next(
            (p for p in pm_schema.physical_models if p.model == pinned),
            pm_schema.physical_models[0],
        )
    elif eligible:
        selected_phys = eligible[0]
    else:
        selected_phys = pm_schema.physical_models[0]
    physical = selected_phys.model
    provider = selected_phys.provider

    if is_new:
        conv = Conversation(
            id=conv_uuid,
            pseudo_model=pseudo_model_name,
            physical_model=physical,
            total_tokens=0,
        )
        conv.turns = []
        db.add(conv)
        await db.flush()

    return (
        existing_affinity,
        session_caps,
        physical,
        provider,
        selected_phys,
        {"applied": tools_filter_applied, "reason": tools_filter_reason},
        conv,
        is_new,
    )


def _check_input_threshold(
    estimated_input: int,
    pm_schema: PseudoModelSchema,
    pseudo_model_name: str,
) -> Result[None, InputExceedsThreshold]:
    """Step 12: Check if input exceeds threshold.

    Returns Ok(None) if within threshold, Err(InputExceedsThreshold) otherwise.
    Uses domain error instead of raising HTTPException.
    """
    check = check_input_threshold(
        pseudo_model_name=pseudo_model_name,
        input_token_threshold=pm_schema.input_token_threshold,
        estimated_tokens=estimated_input,
        pre_compaction_enabled=False,
    )
    if not check.success:
        error = check.error
        return Err(error)
    return Ok(None)


# ── Content delegation ────────────────────────────────────────────────────────


async def _apply_content_delegation(
    delegation: dict | None,
    messages: list[dict],
    conversation_id: str | None,
    affinity,
    valkey,
    config,
) -> list[dict]:
    """Apply image→tool delegation or blob storage for unsupported content."""
    if not delegation:
        return messages

    return await replace_base64_with_blob_refs(
        messages,
        conversation_id,
        valkey or getattr(affinity, "_client", None),
        config,
    )


# ── Context guard ─────────────────────────────────────────────────────────────


def _check_context_usable(
    context_alert,
    conv,
    pm_schema,
    conv_id: str,
) -> Result[None, ContextUnusable]:
    """Check if conversation context is usable.

    Returns Ok(None) if context is usable, Err(ContextUnusable) otherwise.
    Uses domain error instead of raising HTTPException.
    """
    if context_alert.alert_level != "unusable":
        return Ok(None)

    error = ContextUnusable(
        conversation_id=conv_id,
        context_tokens=conv.total_tokens if conv else 0,
        context_window=pm_schema.context_window,
        warning_message=context_alert.warning,
    )
    return Err(error)


# ── Command result builder ────────────────────────────────────────────────────


def _build_command_chat_result(
    cmd_result, pseudo_model_name: str, pm_schema, conv_id: str, _req_id: str
) -> ChatResult:
    """Build a ChatResult for inline commands that bypass the LLM."""
    return ChatResult(
        conversation_id=conv_id,
        pseudo_model=pseudo_model_name,
        physical_model="(command)",
        response={
            "id": f"chatcmpl-{_req_id}",
            "object": "chat.completion",
            "model": pseudo_model_name,
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": cmd_result.response_text,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            "proxy_metadata": cmd_result.response_metadata,
        },
        fallback_info=FallbackInfo(),
        is_new_conversation=False,
        affinity_maintained=False,
        total_tokens=0,
        context_window=pm_schema.context_window
        if hasattr(pm_schema, "context_window")
        else None,
        session_caps=SessionCapabilities(conversation_id=conv_id),
        compatibility_warning=None,
        compatibility_details=None,
        tools_filter_applied=False,
        tools_filter_reason=None,
        tools_level_used=0,
        tools_incomplete=False,
        thinking_content=None,
        tool_result_truncated=False,
        pre_compaction_applied=False,
        pre_compaction_metadata=None,
        continuous_compaction_applied=False,
        continuous_compaction_metadata=None,
        external_compaction_detected=False,
        external_compaction_metadata=None,
        images_described=0,
        images_described_by=None,
        router_suggestion=None,
        context_alert=ContextAlert(alert_level="none", context_usage_pct=None),
        cache_metadata={},
    )


# ── Utility helpers ───────────────────────────────────────────────────────────


def _extract_cmd_name(messages: list[dict]) -> str:
    """Extract the command name from the last user message for logging."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str) and content.startswith("@"):
                return content.split()[0][1:]
    return "?"


# ── Shared helpers (exported for streaming path) ──────────────────────────────


async def _evaluate_router_llm(
    pm_schema,
    messages: list[dict],
    current_pseudo_name: str,
    config,
) -> dict | None:
    """Run the Router LLM evaluation (slow — makes an LLM call)."""
    suggester_pm = config.pseudo_models.get(pm_schema.router_llm.suggester)
    if not suggester_pm or not suggester_pm.physical_models:
        return None

    suggester_phys = suggester_pm.physical_models[0]
    suggester_model = suggester_phys.model
    suggestion = await evaluate_complexity(
        messages=messages,
        suggester_model=suggester_model,
        api_base=suggester_phys.api_base or None,
        api_key=_resolve_api_key(suggester_phys),
    )

    if not suggestion or not suggestion.get("suggested"):
        return None

    if pm_schema.router_llm.suggest_on_downgrade_only:
        if is_downgrade(
            suggestion["suggested"],
            current_pseudo_name,
            config,
        ):
            return suggestion
        return None
    return suggestion


async def evaluate_router_suggestion(
    pm_schema,
    messages: list[dict],
    current_pseudo_name: str,
    config,
) -> dict | None:
    """Evaluate task complexity via Router LLM — shared between paths.

    Non-blocking: if the feature is disabled or evaluation fails, returns
    ``None`` and the request continues unchanged.
    Only evaluates the **last user message** (never the full conversation).

    Args:
        pm_schema: Current pseudo-model schema.
        messages: Full message list (only last user evaluated internally).
        current_pseudo_name: Current pseudo-model name.
        config: Proxy config.

    Returns:
        Suggestion dict with ``complexity``, ``suggested``, ``reason``.
        ``None`` if disabled or evaluation fails.
    """
    if not pm_schema.router_llm.enabled:
        return None

    return await _evaluate_router_llm(pm_schema, messages, current_pseudo_name, config)
