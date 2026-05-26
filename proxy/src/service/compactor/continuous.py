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


def _select_turns_to_compact(
    turns: list[ConversationTurn],
    preserve_recent: int | None,
) -> tuple[list[ConversationTurn], list[ConversationTurn]]:
    """Select which turns to compact and which to preserve.

    Walks turns from most recent backwards, preserving up to preserve_recent tokens.
    Returns (compact_turns, preserved_turns).
    """
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

    return compact_turns, preserved_turns


async def _build_compaction_history(
    db: AsyncSession,
    conversation: Conversation,
    compact_turns: list[ConversationTurn],
) -> list[dict]:
    """Build the history list to send to the compactor.

    Includes existing active snapshot if present, plus all messages from compact turns.
    """
    history: list[dict] = []

    if conversation.active_snapshot_id:
        snapshot = await db.get(ConversationSnapshot, conversation.active_snapshot_id)
        if snapshot:
            history.append(
                {
                    "role": "system",
                    "content": (
                        f"[Previous snapshot from turn {snapshot.turn_number_at_compaction}]\n\n"
                        f"{snapshot.snapshot_content}"
                    ),
                }
            )

    for turn in compact_turns:
        turn_messages = turn.messages
        if isinstance(turn_messages, list):
            history.extend(turn_messages)

    # Strip image_url parts — the compactor is text-only and cannot
    # process images.  Raw JSON blobs in the history would confuse it.
    history = _strip_images_from_messages(history)

    return history


def _strip_images_from_messages(messages: list[dict]) -> list[dict]:
    """Replace ``image_url`` content parts with a text placeholder.

    The compactor LLM (usually Groq) is text-only.  If the history contains
    ``image_url`` parts from vision-model turns they'd arrive as opaque JSON.
    """
    result: list[dict] = []
    image_counter: int = 0
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            result.append(msg)
            continue

        new_parts: list[dict] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                image_counter += 1
                new_parts.append(
                    {
                        "type": "text",
                        "text": (
                            f"[Image #{image_counter} omitted — "
                            f"user shared an image at this point]"
                        ),
                    }
                )
            else:
                new_parts.append(part)

        result.append({**msg, "content": new_parts})

    return result


def _resolve_compactor_model(
    pseudo_model: PseudoModelSchema,
    config,
    estimated_input: int = 0,
) -> tuple[str | None, int | None, str | None]:
    """Resolve the compactor physical model name.

    Uses by_context_window strategy: picks the first model whose
    context_window >= estimated_input. Falls back to the model with
    the largest context window.  This lets Groq models (fast, 131K) handle
    most compactions while larger models (Gemini 1M) handle big histories.

    For very large histories, we divide-and-conquer: split into chunks
    that fit within the cheap model's context and compress each separately,
    then concatenate results.  This avoids ever needing an expensive model.

    Returns (compactor_model, context_window, error_reason).
    If error_reason is set, compactor_model and context_window are None.
    """
    compactor_name = "deep-flash"
    if pseudo_model.pre_compaction.enabled and pseudo_model.pre_compaction.compactor:
        compactor_name = pseudo_model.pre_compaction.compactor

    compactor_pm = config.pseudo_models.get(compactor_name)
    if not compactor_pm or not compactor_pm.physical_models:
        return None, None, f"compactor_not_available: {compactor_name}"

    # by_context_window: prefer fast models with enough room
    for phys in compactor_pm.physical_models:
        cw = phys.context_window
        if cw is None:
            return phys.model, cw, None
        if isinstance(cw, (int, float)) and cw >= estimated_input:
            return phys.model, cw, None

    # Fall back to largest context window
    def _cw(m):
        cw = getattr(m, 'context_window', 0)
        return cw if isinstance(cw, (int, float)) else 0
    largest = max(compactor_pm.physical_models, key=_cw)
    cw = getattr(largest, 'context_window', None)
    return getattr(largest, 'model', None), cw, None


def _chunk_history(
    history: list[dict],
    compaction_prompt: str,
    context_window: int,
    output_buffer: int = 8000,
) -> list[list[dict]]:
    """Split history into overlapping chunks with shared prefix.

    Each chunk includes the **first messages** (common context/prefix) plus
    a unique segment of the middle/end.  All chunks start with the same
    prefix, so providers (Anthropic, DeepSeek) reuse their prompt cache
    across chunk calls — saving cost and latency.

    Chunks are compressed independently, then concatenated into one snapshot.
    This lets us use fast Groq compressions for any size history.
    """
    prompt_tokens = estimate_tokens(
        [{"role": "system", "content": compaction_prompt}]
    )
    available = context_window - prompt_tokens - output_buffer
    if available <= 0:
        return [history]

    # Reserve space for the first messages (global context) + overlap
    # We keep at least the first ~10% of messages or 5, whichever is larger
    context_count = max(5, len(history) // 10)
    head = history[:context_count]
    body = history[context_count:]

    if not body:
        return [history]

    head_str = json.dumps(head, default=str)
    head_tokens = estimate_tokens([{"role": "user", "content": head_str}])
    leftover = available - head_tokens

    chunks: list[list[dict]] = []
    current: list[dict] = []
    current_tokens = 0

    for msg in body:
        msg_str = json.dumps(msg, default=str)
        msg_tokens = estimate_tokens([{"role": "user", "content": msg_str}])

        if current_tokens + msg_tokens > leftover and current:
            # Prepend head for context on each chunk
            chunks.append([*head, *current])
            current = [msg]
            current_tokens = msg_tokens
        else:
            current.append(msg)
            current_tokens += msg_tokens

    if current:
        chunks.append([*head, *current])

    return chunks or [history]


async def _store_compaction_snapshot(
    db: AsyncSession,
    conversation: Conversation,
    compactor_model: str,
    snapshot_content: str,
    snapshot_tokens: int,
    estimated_input: int,
    total_turns: int,
) -> str:
    """Store a new snapshot in DB, chain with previous, update active_snapshot_id."""
    new_snapshot = ConversationSnapshot(
        conversation_id=conversation.id,
        snapshot_type="continuous",
        tokens_before=estimated_input,
        tokens_after=snapshot_tokens,
        compactor_model=compactor_model,
        snapshot_content=snapshot_content or "",
        turn_number_at_compaction=total_turns,
    )
    db.add(new_snapshot)
    await db.flush()

    if conversation.active_snapshot_id:
        old_snapshot = await db.get(
            ConversationSnapshot, conversation.active_snapshot_id
        )
        if old_snapshot:
            old_snapshot.superseded_by = new_snapshot.id

    conversation.active_snapshot_id = new_snapshot.id
    await db.flush()

    return str(new_snapshot.id)


async def continuous_compact(
    conversation: Conversation,
    pseudo_model: PseudoModelSchema,
    config,
    db: AsyncSession,
) -> dict:
    """Perform continuous compaction on a conversation.

    Compacts old turns into a structured snapshot when accumulated context
    exceeds trigger_pct of the context window.
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
    compact_turns, preserved_turns = _select_turns_to_compact(turns, preserve_recent)

    if len(compact_turns) < 3:
        return {"applied": False, "reason": "not_enough_turns_to_compact"}

    # Build history to compact
    history_to_compact = await _build_compaction_history(
        db, conversation, compact_turns
    )

    estimated_input = estimate_tokens(
        [{"role": "user", "content": json.dumps(history_to_compact, default=str)}]
    )

    # Resolve compactor model (use by_context_window: Groq for fast cheap compressions)
    compaction_prompt = build_continuous_compaction_prompt()
    compactor_model, compactor_ctx, error_reason = _resolve_compactor_model(
        pseudo_model, config, estimated_input,
    )
    if error_reason:
        return {
            "applied": False,
            "reason": error_reason,
            "warning": "Continuous compaction cannot run: compactor not available.",
        }

    # If history is too large for one call, split into chunks (divide & conquer).
    # Each chunk includes the first messages (shared cache prefix) + its segment.
    # Chunks are independent, so the compactor cache prefix hits on every chunk.
    try:
        if compactor_ctx and estimated_input > compactor_ctx - 8000:
            chunks = _chunk_history(
                history_to_compact, compaction_prompt, compactor_ctx
            )
            snapshot_parts: list[str] = []
            total_snapshot_tokens = 0

            for chunk in chunks:
                chunk_messages = [
                    {"role": "system", "content": compaction_prompt},
                    {"role": "user", "content": json.dumps(chunk, default=str)},
                ]
                chunk_resp = await call_litellm(
                    model=compactor_model,
                    messages=chunk_messages,
                    max_tokens=8000,
                )
                chunk_dict = (
                    chunk_resp.model_dump()
                    if hasattr(chunk_resp, "model_dump")
                    else chunk_resp
                )
                if isinstance(chunk_dict, dict):
                    choices = chunk_dict.get("choices", [])
                    content = (
                        choices[0].get("message", {}).get("content", "")
                        if choices else ""
                    )
                    usage = chunk_dict.get("usage", {})
                    tok = usage.get("completion_tokens", 0) or 0
                else:
                    content = chunk_resp.choices[0].message.content
                    tok = getattr(chunk_resp.usage, "completion_tokens", 0)
                if not tok:
                    tok = estimate_tokens(
                        [{"role": "user", "content": content or ""}]
                    )
                if content:
                    snapshot_parts.append(content)
                total_snapshot_tokens += tok

            snapshot_content = "\n\n".join(snapshot_parts)
            snapshot_tokens = total_snapshot_tokens
        else:
            # Single call — fits in context
            compaction_messages = [
                {"role": "system", "content": compaction_prompt},
                {"role": "user", "content": json.dumps(history_to_compact, default=str)},
            ]
            response = await call_litellm(
                model=compactor_model,
                messages=compaction_messages,
                max_tokens=8000,
            )
            response_dict = (
                response.model_dump() if hasattr(response, "model_dump") else response
            )
            if isinstance(response_dict, dict):
                choices = response_dict.get("choices", [])
                snapshot_content = (
                    choices[0].get("message", {}).get("content", "") if choices else ""
                )
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
    snapshot_id = await _store_compaction_snapshot(
        db=db,
        conversation=conversation,
        compactor_model=compactor_model,
        snapshot_content=snapshot_content,
        snapshot_tokens=snapshot_tokens,
        estimated_input=estimated_input,
        total_turns=len(turns),
    )

    # Reset total_tokens to reflect compacted state:
    # snapshot tokens + preserved recent turns
    preserved_tokens = sum(
        t.input_tokens + t.output_tokens for t in preserved_turns
    )
    conversation.total_tokens = snapshot_tokens + preserved_tokens

    return {
        "applied": True,
        "tokens_before": estimated_input,
        "tokens_after": snapshot_tokens,
        "compactor_model": compactor_model,
        "turns_compacted": len(compact_turns),
        "turns_preserved": len(preserved_turns),
        "snapshot_id": snapshot_id,
        "snapshot_type": "continuous",
    }


async def assemble_context(
    conversation: Conversation,
    db: AsyncSession,
) -> list[dict]:
    """Build the message array to send to the model.

    If an active snapshot exists, use [snapshot] + [recent turns] instead of
    full history.  Uses ``conversation.turns`` (must be eagerly loaded) to
    avoid redundant DB queries.

    ``image_url`` parts are stripped from historic turns — they belong to past
    turns and are not needed for the current request.  If images were relevant
    they are already captured in the snapshot or in recent-turn descriptions.

    Args:
        conversation: The conversation from DB (turns must be eagerly loaded).
        db: Database session.

    Returns:
        List of message dicts ready for the LLM request.
    """
    messages: list[dict] = []

    # If active snapshot exists, include it as a system message
    if conversation.active_snapshot_id:
        snapshot = await db.get(ConversationSnapshot, conversation.active_snapshot_id)
        if snapshot:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        f"[CONVERSATION SNAPSHOT — generated at turn "
                        f"{snapshot.turn_number_at_compaction} "
                        f"by {snapshot.compactor_model}. "
                        f"Original history: {snapshot.tokens_before} tokens, "
                        f"compacted to {snapshot.tokens_after} tokens.]\n\n"
                        f"{snapshot.snapshot_content}"
                    ),
                }
            )

            # Filter + sort in-memory instead of re-querying DB
            recent_turns = sorted(
                (t for t in (conversation.turns or [])
                 if t.turn_number > snapshot.turn_number_at_compaction),
                key=lambda t: t.turn_number,
            )
        else:
            # Snapshot reference broken — fall back to all turns
            recent_turns = sorted(
                (conversation.turns or []),
                key=lambda t: t.turn_number,
            )
    else:
        # Use eagerly loaded turns directly
        recent_turns = sorted(
            (conversation.turns or []),
            key=lambda t: t.turn_number,
        )

    for turn in recent_turns:
        turn_messages = turn.messages
        if isinstance(turn_messages, list):
            messages.extend(turn_messages)

    # Strip image_url parts — historic images are not needed for the current
    # request.  If they were relevant they are in the snapshot or already
    # described by auto-describe.  Raw image_url blobs confuse text-only models.
    messages = _strip_images_from_messages(messages)

    return messages
