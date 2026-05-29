"""Conversation message reconstruction and auto-describe logic.

Builds full message history from stored turns and handles automatic
image description when switching from vision to non-vision models.
"""

import logging

from src.adapters.db.models import Conversation, ConversationTurn
from src.config.pseudo_models import ProxyConfigSchema
from src.domain.ports import AsyncSessionPort
from src.service.multimedia.image_describer import auto_describe_images

logger = logging.getLogger(__name__)

__all__ = [
    "build_conversation_messages",
    "_load_messages_from_turns",
    "handle_auto_describe",
    "_resolve_auto_describe_params",
]


def _load_messages_from_turns(conv: Conversation) -> list[dict]:
    """Load all messages from conversation turns in order.

    FASE D: Filter out degradation_event turns to prevent context contamination
    from image auto-describe operations that replay full history.
    """
    all_messages: list[dict] = []
    for turn in sorted(conv.turns, key=lambda t: t.turn_number):
        # Skip degradation_event turns
        if getattr(turn, "turn_type", None) == "degradation_event":
            continue
        turn_msgs = turn.messages
        if isinstance(turn_msgs, list):
            all_messages.extend(turn_msgs)
    return all_messages


def _build_history_from_turns(conv: Conversation) -> list[dict]:
    """Reconstruct history from stored turns, skipping degradation_event turns."""
    history: list[dict] = []
    for turn in sorted(conv.turns, key=lambda t: t.turn_number):
        if getattr(turn, "turn_type", None) == "degradation_event":
            logger.debug(
                "skip_degradation_event_turn conv=%s turn=%s",
                conv.id,
                turn.turn_number,
            )
            continue

        if turn.messages:
            history.extend(turn.messages)
        if turn.response and isinstance(turn.response, dict):
            choices = turn.response.get("choices", [])
            if choices:
                msg = choices[0].get("message", {})
                assistant_entry: dict = {"role": "assistant"}
                for field in ("content", "tool_calls", "reasoning_content", "name", "thinking_blocks"):
                    val = msg.get(field)
                    if val is not None:
                        assistant_entry[field] = val
                if len(assistant_entry) > 1:
                    history.append(assistant_entry)
    return history


def _extract_system_contents(messages: list[dict]) -> set:
    """Extract system prompt contents for deduplication."""
    contents: set = set()
    for m in messages:
        if m.get("role") != "system":
            continue
        content = m.get("content")
        if isinstance(content, str):
            contents.add(("str", content))
        elif isinstance(content, list):
            try:
                contents.add(("list", tuple(str(p) for p in content)))
            except (TypeError, ValueError):
                pass
    return contents


def _deduplicate_system_prompts(history: list[dict], to_remove: set) -> list[dict]:
    """Remove system prompts whose content matches the dedup set."""
    if not to_remove:
        return history
    filtered: list[dict] = []
    for m in history:
        if m.get("role") == "system":
            content = m.get("content")
            if isinstance(content, str) and ("str", content) in to_remove:
                continue
            if isinstance(content, list):
                try:
                    if ("list", tuple(str(p) for p in content)) in to_remove:
                        continue
                except (TypeError, ValueError):
                    pass
        filtered.append(m)
    return filtered


def build_conversation_messages(
    conv: Conversation, current_messages: list[dict]
) -> list[dict]:
    """Build full conversation history by interleaving turn messages and assistant responses."""
    history = _build_history_from_turns(conv)
    current_system = _extract_system_contents(current_messages)
    history = _deduplicate_system_prompts(history, current_system)
    history.extend(current_messages)
    return history


def _resolve_auto_describe_params(
    config: ProxyConfigSchema,
    current_pseudo_name: str,
    pinned_physical_model: str,
) -> tuple[str | None, str | None, str | None, str | None, str | None]:
    """Resolve vision model for auto-describe.

    Returns:
        (vision_model, pseudo_name, skip_reason, api_base, api_key)
    """
    if current_pseudo_name is None:
        return (None, None, "no_source_model", None, None)
    current_pm = config.pseudo_models.get(current_pseudo_name)
    if current_pm is None:
        return (
            None,
            None,
            f"source_pseudo_model_not_found:{current_pseudo_name}",
            None,
            None,
        )

    vision_models = [m for m in current_pm.physical_models if m.vision]
    if not vision_models:
        return (
            None,
            None,
            f"source_{current_pseudo_name}_has_no_vision_models",
            None,
            None,
        )

    vision_phys = (
        next(
            (
                m
                for m in current_pm.physical_models
                if m.model == pinned_physical_model and m.vision
            ),
            None,
        )
        if pinned_physical_model
        else vision_models[0]
    )
    if vision_phys is None:
        vision_phys = vision_models[0]

    vision_model = vision_phys.model
    api_base = vision_phys.api_base or None
    api_key = _resolve_api_key(vision_phys)
    return (vision_model, current_pseudo_name, None, api_base, api_key)


async def handle_auto_describe(
    conv: Conversation,
    current_pseudo_name: str,
    config: ProxyConfigSchema,
    db: AsyncSessionPort,
    pinned_physical_model: str,
    in_flight_messages: list[dict] | None = None,
) -> tuple[list[dict] | None, dict | None]:
    """Execute auto-describe when switching from vision to non-vision model."""
    result = _resolve_auto_describe_params(
        config,
        current_pseudo_name,
        pinned_physical_model,
    )
    vision_model = result[0]
    skip_reason = result[2]
    api_base = result[3]
    api_key = result[4]
    if vision_model is None:
        if skip_reason:
            logger.debug("auto_describe_skipped reason=%s", skip_reason)
        return (
            None,
            None
            if not skip_reason
            else {"auto_describe_skipped": True, "skip_reason": skip_reason},
        )

    all_messages = _load_messages_from_turns(conv)
    if not all_messages:
        return (None, None)

    described_history, desc_meta = await auto_describe_images(
        all_messages,
        vision_model,
        api_base=api_base,
        api_key=api_key,
    )
    described_count = desc_meta.get("images_described", 0)

    # Process in_flight messages BEFORE early return — current turn images must be described
    described_in_flight: list[dict] | None = None
    in_flight_count = 0
    if in_flight_messages:
        desc_in_flight, in_flight_meta = await auto_describe_images(
            in_flight_messages,
            vision_model,
            api_base=api_base,
            api_key=api_key,
        )
        if desc_in_flight is not None:
            described_in_flight = desc_in_flight
            in_flight_count = in_flight_meta.get("images_described", 0)

    if described_count == 0 and in_flight_count == 0:
        return (None, desc_meta if desc_meta else None)
    # NOTE: If in_flight_count > 0, we must continue to save the degradation turn
    # even if described_count == 0, because the current turn's images were described

    turn_number = max(t.turn_number for t in conv.turns) + 1 if conv.turns else 1
    deg_turn = ConversationTurn(
        conversation_id=conv.id,
        turn_number=turn_number,
        pseudo_model=current_pseudo_name,
        physical_model=vision_model,
        input_tokens=0,
        output_tokens=desc_meta.get("total_description_tokens", 0),
        messages=described_history or all_messages,
        response={"metadata": desc_meta},
        turn_type="degradation_event",
        had_images=False,
        had_tools=False,
        had_parallel_tools=False,
    )
    db.add(deg_turn)
    conv.images_described = (
        max(conv.images_described or 0, 0) + described_count + in_flight_count
    )
    conv.capability_has_images = False

    return (described_in_flight, desc_meta)


# ── Late import to avoid circular dependency ─────────────────────────────────


def _resolve_api_key(phys) -> str | None:
    """Resolve API key from environment if the physical model has api_key_env set."""
    from src.service.chat_fallback import _resolve_api_key as _raf

    return _raf(phys)
