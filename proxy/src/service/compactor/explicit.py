"""Explicit compaction service for Sprint 6.

plan-proxy.md §11: POST /conversations/{id}/compact.
Reuses existing ConversationSnapshot table with snapshot_type="explicit".
For histories >500K tokens, dispatches to arq for async processing.

python.md §7: async-first.
python.md §3: HTTPException at boundary, Result monad in service layer.
"""

import json
import uuid

from fastapi import HTTPException
from sqlalchemy import select
from sqlmodel.ext.asyncio.session import AsyncSession

from src.adapters.db.models import Conversation, ConversationSnapshot, ConversationTurn
from src.adapters.litellm import call_litellm
from src.service.capability_detector import estimate_tokens
from src.service.compactor.prompts import build_explicit_compaction_prompt

_MIN_RETENTION = 0.05


def _compaction_max_tokens(estimated_input: int, default: int = 12000) -> int:
    """Ensure compaction output is at least 5% of input."""
    return max(default, int(estimated_input * _MIN_RETENTION))


# ── Compactor model selection ─────────────────────────────────────────────


def select_compactor_model(config, total_tokens: int):
    """Select the compactador model with enough context window for the history.

    Uses by_context_window strategy: picks the first physical model whose
    context_window >= total_tokens. Falls back to the largest available.

    Args:
        config: Proxy config with pseudo-model definitions.
        total_tokens: Total tokens in the conversation history.

    Returns:
        Physical model dict/object with ``.model``, ``.context_window``,
        ``.api_base``, ``.api_key_env``, or ``None`` if no model available.
    """
    compactador_pm = config.pseudo_models.get("compactador")
    if not compactador_pm or not compactador_pm.physical_models:
        return None

    # Try to find a model with enough context window (skip audio capability models)
    for phys in compactador_pm.physical_models:
        if getattr(phys, "audio", False):
            continue  # Skip whisper — not a chat model
        if phys.context_window and phys.context_window >= total_tokens:
            return phys

    # Fall back to the model with the largest context window (skip audio models)
    candidates = [m for m in compactador_pm.physical_models if not getattr(m, "audio", False)]
    if not candidates:
        return None
    largest = max(
        candidates,
        key=lambda m: m.context_window or 0,
    )
    return largest


# ── Internal helpers ──────────────────────────────────────────────────────


def _parse_uuid(value: str) -> uuid.UUID:
    """Parse a UUID string, falling back to UUID5 for non-UUID strings."""
    try:
        return uuid.UUID(value)
    except ValueError:
        return uuid.uuid5(uuid.NAMESPACE_DNS, value)


def _build_compaction_history(turns: list[ConversationTurn]) -> list[dict]:
    """Build the full message history from conversation turns."""
    all_messages: list[dict] = []
    for turn in turns:
        turn_msgs = turn.messages
        if isinstance(turn_msgs, list):
            all_messages.extend(turn_msgs)
        elif isinstance(turn_msgs, dict) and "messages" in turn_msgs:
            all_messages.extend(turn_msgs["messages"])
    return all_messages


def _parse_compactor_response(response) -> tuple[str, int]:
    """Extract snapshot content and token count from compactor API response."""
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
    return snapshot_content or "", snapshot_tokens


async def _run_compaction_sync(
    conversation_id: str,
    compactor_model: str,
    all_messages: list[dict],
    total_tokens: int,
    db: AsyncSession,
    config,
    conv: Conversation | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
) -> dict:
    """Run compaction synchronously and store the snapshot.

    All DB operations happen *before* the compactor API call to avoid
    losing the SQLAlchemy greenlet context (known issue when calling
    async HTTP clients and then re-entering the async session).
    """
    conv_uuid = _parse_uuid(conversation_id)
    compaction_prompt = build_explicit_compaction_prompt()

    # ── DB operations BEFORE the API call ──────────────────────────────
    # Count turns
    result = await db.execute(
        select(ConversationTurn).where(ConversationTurn.conversation_id == conv_uuid)
    )
    all_turns = result.scalars().all()
    turn_count = len(all_turns)

    # Load conversation if not passed in
    if conv is None:
        conv = await db.get(Conversation, conv_uuid)

    # Create snapshot (added + flushed before the API call)
    new_snapshot = ConversationSnapshot(
        conversation_id=conv_uuid,
        snapshot_type="explicit",
        tokens_before=total_tokens,
        tokens_after=0,  # placeholder
        compactor_model=compactor_model,
        snapshot_content="",  # placeholder
        turn_number_at_compaction=turn_count,
    )
    db.add(new_snapshot)
    await db.flush()

    # Chain with previous snapshot
    if conv and conv.active_snapshot_id:
        old = await db.get(ConversationSnapshot, conv.active_snapshot_id)
        if old:
            old.superseded_by = new_snapshot.id

    if conv:
        conv.active_snapshot_id = new_snapshot.id

    # ── Describe images using any available vision model ──────────────
    all_messages = await _prepare_multimedia_for_compaction(all_messages, config)

    # ── Call compactor model API ───────────────────────────────────────
    compaction_messages = [
        {"role": "system", "content": compaction_prompt},
        {"role": "user", "content": json.dumps(all_messages, default=str)},
    ]

    try:
        response = await call_litellm(
            model=compactor_model,
            messages=compaction_messages,
            api_base=api_base,
            api_key=api_key,
            max_tokens=_compaction_max_tokens(total_tokens),
            temperature=0.1,
        )
        snapshot_content, snapshot_tokens = _parse_compactor_response(response)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "COMPACTION_FAILED",
                "message": f"Compactor model failed: {exc}",
                "compactor_model": compactor_model,
            },
        )

    # ── Update snapshot with API results ───────────────────────────────
    new_snapshot.tokens_after = snapshot_tokens
    new_snapshot.snapshot_content = snapshot_content or ""
    await db.commit()

    return {
        "status": "completed",
        "snapshot_id": str(new_snapshot.id),
        "tokens_before": total_tokens,
        "tokens_after": snapshot_tokens,
        "tokens_reduced_pct": round(
            (1 - snapshot_tokens / max(total_tokens, 1)) * 100, 1
        ),
        "compactor_model": compactor_model,
        "preview": (snapshot_content[:500] + "...")
        if len(snapshot_content or "") > 500
        else (snapshot_content or ""),
        "can_resume": True,
    }


# ── Main entry point ──────────────────────────────────────────────────────


async def compact_conversation(
    conversation_id: str,
    db: AsyncSession,
    config,
    arq_pool=None,
) -> dict:
    """Explicitly compact a conversation into a structured snapshot.

    Args:
        conversation_id: UUID string of the conversation.
        db: Async DB session.
        config: Proxy config with pseudo-model definitions.
        arq_pool: Optional arq Redis pool for async dispatch (>500K tokens).

    Returns:
        Dict with compaction result metadata.

    Raises:
        HTTPException: 404 (not found), 400 (empty/no compactor),
                       502 (compactor model failed).
    """
    conv_uuid = _parse_uuid(conversation_id)
    conv = await db.get(Conversation, conv_uuid)
    if not conv:
        raise HTTPException(
            status_code=404,
            detail={"error": "CONVERSATION_NOT_FOUND"},
        )

    # Load all turns
    result = await db.execute(
        select(ConversationTurn)
        .where(ConversationTurn.conversation_id == conv_uuid)
        .order_by(ConversationTurn.turn_number)
    )
    turns = result.scalars().all()

    if not turns:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "EMPTY_CONVERSATION",
                "message": "Nothing to compact.",
            },
        )

    # Reconstruct full history
    all_messages: list[dict] = []
    if conv.active_snapshot_id:
        snapshot = await db.get(ConversationSnapshot, conv.active_snapshot_id)
        if snapshot:
            all_messages.append(
                {
                    "role": "system",
                    "content": (
                        f"[Previous snapshot from turn {snapshot.turn_number_at_compaction}]\n\n"
                        f"{snapshot.snapshot_content}"
                    ),
                }
            )

    all_messages.extend(_build_compaction_history(turns))
    total_tokens = conv.total_tokens

    # Select compactador model
    compactor_phys = select_compactor_model(config, total_tokens)
    if not compactor_phys:
        compactador_pm = config.pseudo_models.get("compactador")
        max_window = 0
        if compactador_pm and compactador_pm.physical_models:
            max_window = max(
                (m.context_window or 0) for m in compactador_pm.physical_models
                if not getattr(m, "audio", False)
            )
        raise HTTPException(
            status_code=400,
            detail={
                "error": "HISTORY_TOO_LARGE_FOR_COMPACTOR",
                "message": (
                    f"No compactador model has a context window large enough "
                    f"for {total_tokens} tokens."
                ),
                "max_compactor_window": max_window,
            },
        )

    compactor_model = compactor_phys.model
    api_base = compactor_phys.api_base or None
    api_key = _resolve_api_key(compactor_phys)

    # Dispatch to arq if history > 500K tokens and pool available
    if total_tokens > 500_000 and arq_pool is not None:
        job = await arq_pool.enqueue_job(
            "compact_conversation_async",
            conversation_id,
            compactor_model,
            api_base,
            api_key,
        )
        return {
            "status": "processing",
            "task_id": job.job_id,
            "message": (
                f"Compaction dispatched to background worker. "
                f"Check status at GET /conversations/{conversation_id}."
            ),
            "estimated_tokens": total_tokens,
            "compactor_model": compactor_model,
        }

    # Synchronous compaction for smaller histories
    return await _run_compaction_sync(
        conversation_id=conversation_id,
        compactor_model=compactor_model,
        api_base=api_base,
        api_key=api_key,
        all_messages=all_messages,
        total_tokens=total_tokens,
        db=db,
        config=config,
        conv=conv,
    )


# ── Async helper for arq worker ──────────────────────────────────────────


async def _compact_async(
    conversation_id: str,
    compactor_model: str,
    db_session_factory,
    config,
    api_base: str | None = None,
    api_key: str | None = None,
) -> dict:
    """Async compaction helper called by the arq worker.

    Creates its own DB session since it runs in a separate process.
    """
    db = db_session_factory()
    try:
        conv_uuid = _parse_uuid(conversation_id)
        conv = await db.get(Conversation, conv_uuid)
        if not conv:
            return {"status": "failed", "error": "CONVERSATION_NOT_FOUND"}

        result = await db.execute(
            select(ConversationTurn)
            .where(ConversationTurn.conversation_id == conv_uuid)
            .order_by(ConversationTurn.turn_number)
        )
        turns = result.scalars().all()

        all_messages: list[dict] = []
        if conv.active_snapshot_id:
            snapshot = await db.get(ConversationSnapshot, conv.active_snapshot_id)
            if snapshot:
                all_messages.append(
                    {
                        "role": "system",
                        "content": (
                            f"[Previous snapshot from turn {snapshot.turn_number_at_compaction}]\n\n"
                            f"{snapshot.snapshot_content}"
                        ),
                    }
                )

        all_messages.extend(_build_compaction_history(turns))
        total_tokens = conv.total_tokens

        return await _run_compaction_sync(
            conversation_id=conversation_id,
            compactor_model=compactor_model,
            all_messages=all_messages,
            total_tokens=total_tokens,
            db=db,
            config=config,
            conv=conv,
            api_base=api_base,
            api_key=api_key,
        )
    finally:
        await db.close()


def _history_has_images(history: list[dict]) -> bool:
    """Check if any message in history contains an image_url content part."""
    for msg in history:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    return True
    return False


def _find_vision_model(config):
    """Find any configured physical model with vision capability.

    Returns the full ``PhysicalModelSchema`` or ``None``.
    """
    for pm in config.pseudo_models.values():
        for phys in pm.physical_models:
            if getattr(phys, "vision", False):
                return phys
    return None


def _resolve_api_key(phys) -> str | None:
    """Resolve API key from environment if the physical model has api_key_env set."""
    if not phys or not phys.api_key_env:
        return None
    import os
    return os.environ.get(phys.api_key_env) or None


async def _prepare_multimedia_for_compaction(history: list[dict], config) -> list[dict]:
    """Describe images using any available vision model before compaction."""
    if not _history_has_images(history):
        return history

    vision_phys = _find_vision_model(config)
    if vision_phys is None:
        return history

    vision_model = vision_phys.model
    api_base = vision_phys.api_base or None
    api_key = _resolve_api_key(vision_phys)

    from src.service.multimedia.image_describer import auto_describe_images

    try:
        described, _meta = await auto_describe_images(
            history, vision_model,
            api_base=api_base, api_key=api_key,
        )
        return described
    except Exception:
        return history
