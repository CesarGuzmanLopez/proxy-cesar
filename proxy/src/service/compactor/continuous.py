"""Continuous compaction service for Sprint 4.

plan-proxy.md §10.2: When accumulated context exceeds trigger_pct of
context_window, old turns are compacted into a structured snapshot.

Also handles external compaction detection (§3.7):
when the client (OpenCode) compacts the history externally, the proxy
detects it and integrates it into the snapshot chain.

python.md §7: async-first.
python.md §4: pure functions for detection, async service for compaction.
"""

import json
import uuid

from sqlalchemy import func, select
from sqlmodel.ext.asyncio.session import AsyncSession

from src.adapters.db.models import Conversation, ConversationSnapshot, ConversationTurn
from src.adapters.litellm import call_litellm
from src.config.pseudo_models import PseudoModelSchema
from src.service.capability_detector import estimate_tokens
from src.service.compactor.prompts import build_continuous_compaction_prompt


# ── External Compaction Detection ─────────────────────────────────────────


class ExternalCompactionInfo:
    """Information about an externally-detected compaction."""

    def __init__(
        self,
        detected: bool,
        incoming_message_count: int = 0,
        previous_turn_count: int = 0,
        summary_preview: str = "",
    ):
        self.detected = detected
        self.incoming_message_count = incoming_message_count
        self.previous_turn_count = previous_turn_count
        self.summary_preview = summary_preview


async def detect_external_compaction(
    incoming_messages: list[dict],
    conversation: Conversation,
    db: AsyncSession,
) -> ExternalCompactionInfo | None:
    """Detect if the client (OpenCode) has compacted the conversation.

    Detection signals:
    1. Message count suddenly drops significantly (>50% fewer messages than turns)
    2. First message is a system or user message with summary-like content
    3. The conversation previously had many turns, now has very few messages

    Args:
        incoming_messages: The messages array from the current request.
        conversation: The conversation from DB.
        db: Database session.

    Returns:
        ExternalCompactionInfo if detected, None otherwise.
    """
    # Count previous turns
    result = await db.execute(
        select(func.count(ConversationTurn.id)).where(
            ConversationTurn.conversation_id == conversation.id
        )
    )
    previous_turn_count = result.scalar() or 0

    if previous_turn_count < 10:
        return None  # Too few turns for compaction to make sense

    incoming_msg_count = len(incoming_messages)

    # Signal 1: drastic reduction in message count
    expected_min_messages = previous_turn_count * 0.4  # At least 40% of previous
    if incoming_msg_count > expected_min_messages:
        return None  # Message count is normal — no compaction

    # Signal 2: first message looks like a summary
    first_msg = incoming_messages[0]
    is_system_or_user = first_msg.get("role") in ("system", "user")
    content = str(first_msg.get("content", ""))
    is_long = len(content) > 200  # Summaries are usually substantial text

    if not (is_system_or_user and is_long):
        return None

    # External compaction detected!
    return ExternalCompactionInfo(
        detected=True,
        incoming_message_count=incoming_msg_count,
        previous_turn_count=previous_turn_count,
        summary_preview=content[:500],
    )


async def handle_external_compaction(
    incoming_messages: list[dict],
    conversation: Conversation,
    external_info: ExternalCompactionInfo,
    db: AsyncSession,
) -> dict:
    """Handle external compaction detected in incoming messages.

    Stores the client's summary as a compaction_snapshot in the proxy's DB,
    sets it as the active snapshot, and resets token tracking.

    Args:
        incoming_messages: The messages array from the current request.
        conversation: The conversation from DB.
        external_info: Detection info.
        db: Database session.

    Returns:
        Metadata dict for proxy_metadata.
    """
    summary_content = str(incoming_messages[0].get("content", ""))
    estimated_tokens = len(summary_content) // 4  # Rough estimate

    new_snapshot = ConversationSnapshot(
        conversation_id=conversation.id,
        snapshot_type="external",
        tokens_before=conversation.total_tokens,
        tokens_after=estimated_tokens,
        compactor_model="client (external)",
        snapshot_content=summary_content,
        turn_number_at_compaction=external_info.previous_turn_count,
    )
    db.add(new_snapshot)
    await db.flush()

    # Chain previous snapshot if any
    if conversation.active_snapshot_id:
        old = await db.get(ConversationSnapshot, conversation.active_snapshot_id)
        if old:
            old.superseded_by = new_snapshot.id

    conversation.active_snapshot_id = new_snapshot.id

    # Reset token tracking to reflect compacted state
    new_total = estimated_tokens + sum(
        len(str(m.get("content", ""))) // 4 for m in incoming_messages[1:]
    )
    conversation.total_tokens = new_total

    await db.flush()

    return {
        "external_compaction_detected": True,
        "source": "client",
        "tokens_before_turns": external_info.previous_turn_count,
        "tokens_after_snapshot": estimated_tokens,
        "proxy_compaction_skipped": True,
    }


# ── Continuous Compaction ─────────────────────────────────────────────────


async def continuous_compact(
    conversation: Conversation,
    pseudo_model: PseudoModelSchema,
    config,
    db: AsyncSession,
) -> dict:
    """Perform continuous compaction on a conversation.

    Compacts old turns into a structured snapshot when accumulated context
    exceeds trigger_pct of the context window.

    Args:
        conversation: The conversation from DB (must have total_tokens).
        pseudo_model: The pseudo-model schema with continuous_compaction config.
        config: The proxy config.
        db: Database session.

    Returns:
        Metadata dict about the compaction.

    Metadata keys:
        - applied: bool
        - reason: str (if not applied)
        - tokens_before: int
        - tokens_after: int
        - compactor_model: str (if applied)
        - turns_compacted: int
        - turns_preserved: int
        - snapshot_id: str (if applied)
        - warning: str (if compactor failed)
    """
    trigger_pct = pseudo_model.continuous_compaction.trigger_pct
    preserve_recent = pseudo_model.continuous_compaction.compact_preserve_recent
    context_window = pseudo_model.context_window

    if not context_window or not trigger_pct:
        return {"applied": False, "reason": "no_trigger_config"}

    # Check if compaction should trigger
    trigger_threshold = context_window * trigger_pct / 100
    if conversation.total_tokens <= trigger_threshold:
        return {
            "applied": False,
            "reason": "below_trigger",
            "total_tokens": conversation.total_tokens,
            "trigger_threshold": trigger_threshold,
        }

    # Load all turns
    result = await db.execute(
        select(ConversationTurn)
        .where(ConversationTurn.conversation_id == conversation.id)
        .order_by(ConversationTurn.turn_number)
    )
    turns = result.scalars().all()

    if len(turns) < 3:
        return {"applied": False, "reason": "not_enough_turns_to_compact"}

    # Determine which turns to compact vs preserve
    compact_turns: list[ConversationTurn] = []
    preserved_turns: list[ConversationTurn] = []
    accumulated_tokens = 0

    for turn in reversed(turns):
        turn_tokens = turn.input_tokens + turn.output_tokens
        if accumulated_tokens + turn_tokens <= (preserve_recent or 0):
            preserved_turns.insert(0, turn)
            accumulated_tokens += turn_tokens
        else:
            compact_turns.insert(0, turn)

    if len(compact_turns) < 3:
        return {"applied": False, "reason": "not_enough_turns_to_compact"}

    # Build history to compact (include existing active snapshot)
    history_to_compact: list[dict] = []

    if conversation.active_snapshot_id:
        snapshot = await db.get(ConversationSnapshot, conversation.active_snapshot_id)
        if snapshot:
            history_to_compact.append({
                "role": "system",
                "content": (
                    f"[Previous snapshot from turn {snapshot.turn_number_at_compaction}]\n\n"
                    f"{snapshot.snapshot_content}"
                ),
            })

    for turn in compact_turns:
        turn_messages = turn.messages
        if isinstance(turn_messages, list):
            history_to_compact.extend(turn_messages)

    # Build compaction prompt
    compaction_prompt = build_continuous_compaction_prompt()

    # Select compactor model
    compactor_name = "deep-flash"
    if pseudo_model.pre_compaction.enabled and pseudo_model.pre_compaction.compactor:
        compactor_name = pseudo_model.pre_compaction.compactor

    compactor_pm = config.pseudo_models.get(compactor_name)
    if not compactor_pm or not compactor_pm.physical_models:
        return {
            "applied": False,
            "reason": f"compactor_not_available: {compactor_name}",
            "warning": f"Continuous compaction cannot run: compactor '{compactor_name}' is not available.",
        }

    compactor_model = compactor_pm.physical_models[0].model

    # Call compactor
    compaction_messages = [
        {"role": "system", "content": compaction_prompt},
        {"role": "user", "content": json.dumps(history_to_compact, default=str)},
    ]

    estimated_input = estimate_tokens(
        [{"role": "user", "content": json.dumps(history_to_compact, default=str)}]
    )

    try:
        response = await call_litellm(
            model=compactor_model,
            messages=compaction_messages,
            max_tokens=8000,  # Target snapshot size
        )
        response_dict = response.model_dump() if hasattr(response, "model_dump") else response
        if isinstance(response_dict, dict):
            choices = response_dict.get("choices", [])
            snapshot_content = choices[0].get("message", {}).get("content", "") if choices else ""
            usage = response_dict.get("usage", {})
            snapshot_tokens = usage.get("completion_tokens", 0) or 0
        else:
            snapshot_content = response.choices[0].message.content
            snapshot_tokens = getattr(response.usage, "completion_tokens", 0)
        if not snapshot_tokens:
            snapshot_tokens = estimate_tokens(
                [{"role": "user", "content": snapshot_content or ""}]
            )
    except Exception as exc:
        return {
            "applied": False,
            "reason": f"compactor_failed: {exc}",
            "warning": "Continuous compaction failed due to compactor error.",
        }

    # Store snapshot
    new_snapshot = ConversationSnapshot(
        conversation_id=conversation.id,
        snapshot_type="continuous",
        tokens_before=estimated_input,
        tokens_after=snapshot_tokens,
        compactor_model=compactor_model,
        snapshot_content=snapshot_content or "",
        turn_number_at_compaction=len(turns),
    )
    db.add(new_snapshot)
    await db.flush()

    # Chain with previous snapshot
    if conversation.active_snapshot_id:
        old_snapshot = await db.get(ConversationSnapshot, conversation.active_snapshot_id)
        if old_snapshot:
            old_snapshot.superseded_by = new_snapshot.id

    conversation.active_snapshot_id = new_snapshot.id
    await db.flush()

    return {
        "applied": True,
        "tokens_before": estimated_input,
        "tokens_after": snapshot_tokens,
        "compactor_model": compactor_model,
        "turns_compacted": len(compact_turns),
        "turns_preserved": len(preserved_turns),
        "snapshot_id": str(new_snapshot.id),
        "snapshot_type": "continuous",
    }


async def assemble_context(
    conversation: Conversation,
    db: AsyncSession,
) -> list[dict]:
    """Build the message array to send to the model.

    If an active snapshot exists, use [snapshot] + [recent turns] instead of
    full history.

    Args:
        conversation: The conversation from DB.
        db: Database session.

    Returns:
        List of message dicts ready for the LLM request.
    """
    messages: list[dict] = []

    # If active snapshot exists, include it as a system message
    if conversation.active_snapshot_id:
        snapshot = await db.get(ConversationSnapshot, conversation.active_snapshot_id)
        if snapshot:
            messages.append({
                "role": "system",
                "content": (
                    f"[CONVERSATION SNAPSHOT — generated at turn "
                    f"{snapshot.turn_number_at_compaction} "
                    f"by {snapshot.compactor_model}. "
                    f"Original history: {snapshot.tokens_before} tokens, "
                    f"compacted to {snapshot.tokens_after} tokens.]\n\n"
                    f"{snapshot.snapshot_content}"
                ),
            })

            # Load only turns AFTER the snapshot
            result = await db.execute(
                select(ConversationTurn)
                .where(
                    ConversationTurn.conversation_id == conversation.id,
                    ConversationTurn.turn_number > snapshot.turn_number_at_compaction,
                )
                .order_by(ConversationTurn.turn_number)
            )
            recent_turns = result.scalars().all()
        else:
            # Snapshot reference broken — fall back to full history
            result = await db.execute(
                select(ConversationTurn)
                .where(ConversationTurn.conversation_id == conversation.id)
                .order_by(ConversationTurn.turn_number)
            )
            recent_turns = result.scalars().all()
    else:
        # Load all turns
        result = await db.execute(
            select(ConversationTurn)
            .where(ConversationTurn.conversation_id == conversation.id)
            .order_by(ConversationTurn.turn_number)
        )
        recent_turns = result.scalars().all()

    for turn in recent_turns:
        turn_messages = turn.messages
        if isinstance(turn_messages, list):
            messages.extend(turn_messages)

    return messages
