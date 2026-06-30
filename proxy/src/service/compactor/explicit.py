"""Explicit compaction service for feature

plan-proxy.md §11: POST /conversations/{id}/compact.
Reuses existing ConversationSnapshot table with snapshot_type="explicit".
For histories >500K tokens, dispatches to arq for async processing.

python.md §7: async-first.
python.md §3: errors as data in domain, exceptions at boundary.

FASE 3: CompactionOrchestrator with per-conversation mutex to prevent race conditions.
"""

import asyncio
import json
import logging
import uuid

from collections.abc import Sequence
from sqlalchemy import select

from src.adapters.db.models import Conversation, ConversationSnapshot, ConversationTurn
from src.adapters.litellm import call_litellm
from src.domain.errors import (
    ConversationNotFound,
    EmptyConversation,
    HistoryTooLargeForCompactor,
    CompactionFailed,
)
from src.domain.ports import AsyncSessionPort
from src.domain.types import Result, Ok, Err
from src.service.capability_detector import estimate_tokens
from src.service.compactor.prompts import build_explicit_compaction_prompt

logger = logging.getLogger(__name__)

_MIN_RETENTION = 0.05


# ── Compaction Orchestrator (FASE 3) ──────────────────────────────────────


class CompactionOrchestrator:
    """Single coordinator for all compaction strategies.

    Prevents race conditions by ensuring only one compaction per conversation
    at a time using asyncio.Lock.
    """

    def __init__(self):
        self._locks: dict[uuid.UUID | str, asyncio.Lock] = {}

    async def try_compact(
        self,
        conversation_id: str,
        db: AsyncSessionPort,
        config,
        trigger: str = "explicit",  # "explicit", "pre", "continuous"
        arq_pool=None,
        valkey=None,
    ) -> tuple[bool, Result[dict, ConversationNotFound | EmptyConversation | HistoryTooLargeForCompactor | CompactionFailed]]:
        """Try to acquire lock and start compaction.

        Args:
            conversation_id: Conversation ID to compact
            db: Async DB session
            config: Proxy config
            trigger: What triggered the compaction
            arq_pool: Optional arq pool for large histories
            valkey: Optional Valkey client for images

        Returns:
            (success: bool, result: Result[dict, ...]) where:
            - success=True: compaction started/completed, result is Ok(dict) or Err(...)
            - success=False: compaction already in progress, result is empty dict
        """
        conv_uuid = _parse_uuid(conversation_id)

        # Get or create lock for this conversation
        lock = self._locks.setdefault(conv_uuid, asyncio.Lock())

        # Non-blocking acquire: check if already locked
        if lock.locked():
            logger.info(
                "compact_already_in_progress conv=%s trigger=%s",
                str(conv_uuid)[:12],
                trigger,
            )
            return False, Ok({})

        await lock.acquire()

        try:
            # Lock acquired - proceed with compaction
            result = await compact_conversation(
                conversation_id=conversation_id,
                db=db,
                config=config,
                arq_pool=arq_pool,
                valkey=valkey,
            )
            return True, result
        finally:
            lock.release()


def _compaction_max_tokens(estimated_input: int, default: int = 12000) -> int:
    """Ensure compaction output is at least 5% of input."""
    return max(default, int(estimated_input * _MIN_RETENTION))


# ── Compactor model selection ─────────────────────────────────────────────


def select_compactor_model(config, total_tokens: int):
    """Select a compactor model with enough context window for the history.

    Looks for a "compactador" pseudo-model in config. Returns None if not found.

    Args:
        config: Proxy config with pseudo-model definitions.
        total_tokens: Total tokens in the conversation history.

    Returns:
        Physical model dict/object with ``.model``, ``.context_window``,
        ``.api_base``, ``.api_key_env``, or ``None`` if no model available.
    """
    compactor_pm = config.pseudo_models.get("compactador")
    if not compactor_pm or not compactor_pm.physical_models:
        return None

    # Try to find a model with enough context window (skip audio models)
    for phys in compactor_pm.physical_models:
        if getattr(phys, "audio", False):
            continue
        if phys.context_window and phys.context_window >= total_tokens:
            return phys

    # Fall back to the model with the largest context window (skip audio models)
    candidates = [
        m for m in compactor_pm.physical_models if not getattr(m, "audio", False)
    ]
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


def _estimate_json_size(messages: list[dict]) -> int:
    """Rough estimate of JSON string length for a list of messages."""
    # Each message has at least ~30 chars overhead for {"role":"x","content":"..."}
    return sum(30 + len(str(msg.get("content", ""))) for msg in messages)


def _split_into_chunks(messages: list[dict], max_chars: int) -> list[list[dict]]:
    """Split messages into chunks that fit within max_chars of JSON.

    Each chunk is compacted separately. Assembles messages one by one
    until the estimated JSON size exceeds max_chars, then starts a new chunk.
    """
    if not messages:
        return []

    chunks: list[list[dict]] = []
    current: list[dict] = []

    for msg in messages:
        # Estimate what the JSON would look like with this message added
        test_chunk = current + [msg]
        est_size = _estimate_json_size(test_chunk)

        if current and est_size > max_chars:
            chunks.append(current)
            current = [msg]
        else:
            current.append(msg)

    if current:
        chunks.append(current)

    return chunks


def _build_compaction_history(turns: Sequence[ConversationTurn]) -> list[dict]:
    """Build the full message history from conversation turns.

    Each ConversationTurn stores ALL messages cumulatively (the full history
    up to that turn).  Iterating through all turns creates O(N^2) duplication.
    Only the LAST turn's messages are needed — they already contain the
    complete history.
    """
    if not turns:
        return []
    last_turn = turns[-1]
    turn_msgs = last_turn.messages
    if isinstance(turn_msgs, list):
        return list(turn_msgs)
    if isinstance(turn_msgs, dict) and "messages" in turn_msgs:
        return list(turn_msgs["messages"])
    return []


async def _parse_compactor_response(response) -> tuple[str, int]:
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

    # Strip <think> blocks — the compactor does not need reasoning
    if snapshot_content and "<think>" in snapshot_content:
        import re
        snapshot_content = re.sub(r"<think>.*?</think>", "", snapshot_content, flags=re.DOTALL).strip()

    if not snapshot_tokens:
        snapshot_tokens = await estimate_tokens(
            [{"role": "user", "content": snapshot_content or ""}]
        )
    return snapshot_content or "", snapshot_tokens


async def _run_compaction_sync(
    conversation_id: str,
    compactor_model: str,
    all_messages: list[dict],
    total_tokens: int,
    db: AsyncSessionPort,
    config,
    conv: Conversation | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    *,
    turn_count: int = 0,
    valkey=None,
    trace_id: str | None = None,
) -> Result[dict, CompactionFailed]:
    """Run compaction synchronously and store the snapshot.

    Returns Ok(dict) with compaction metadata on success,
    Err(CompactionFailed) on failure.
    """
    _trace = trace_id or str(uuid.uuid4())[:8]
    conv_uuid = _parse_uuid(conversation_id)
    compaction_prompt = build_explicit_compaction_prompt()

    # ── DB operations BEFORE the API call ──────────────────────────────
    if turn_count == 0:
        result = await db.execute(
            select(ConversationTurn).where(
                ConversationTurn.conversation_id == conv_uuid  # type: ignore[arg-type]  # justification: ORM column comparison: SQLModel Field() types don't expose InstrumentedAttribute; mypy sees bool, runtime returns BinaryExpression
            )
        )
        all_turns = result.scalars().all()
        turn_count = len(all_turns)

    # Load conversation if not passed in
    if conv is None:
        conv = await db.get(Conversation, conv_uuid)

    # ── Estimate safe JSON size budget ───────────────────────────────────
    # JSON serialization overhead is ~2-3x message tokens. We chunk when
    # the JSON string exceeds 80K chars (~20K tokens) to stay well within
    # the compactor's context window for both input AND output.
    _MAX_JSON_CHUNK_CHARS = 80000

    # ── Describe images using any available vision model ──────────────
    all_messages = await _prepare_multimedia_for_compaction(
        all_messages, config, valkey
    )

    # ── Split messages into chunks if needed ───────────────────────────
    chunks = _split_into_chunks(all_messages, _MAX_JSON_CHUNK_CHARS)

    chunk_results: list[str] = []
    for idx, chunk in enumerate(chunks):
        chunk_label = f"## Block {idx + 1}\n" if len(chunks) > 1 else ""

        compaction_messages = [
            {"role": "system", "content": compaction_prompt},
            {"role": "user", "content": chunk_label + json.dumps(chunk, default=str)},
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
            chunk_content, _ = await _parse_compactor_response(response)
            if chunk_content:
                chunk_results.append(chunk_content)
        except Exception as exc:
            logger.error(
                "compaction_chunk_failed trace=%s conv=%s chunk=%d/%d compactor=%s: %s",
                _trace,
                conversation_id[:12],
                idx + 1,
                len(chunks),
                compactor_model,
                exc,
            )

    if not chunk_results:
        return Err(CompactionFailed(
            conversation_id=conversation_id,
            compactor_model=compactor_model,
            reason="All compaction chunks failed",
        ))

    snapshot_content = "\n\n".join(chunk_results)
    snapshot_tokens = await estimate_tokens(
        [{"role": "user", "content": snapshot_content}]
    )

    # ── Create snapshot ONLY after API call succeeds ───────────────────
    new_snapshot = ConversationSnapshot(
        conversation_id=conv_uuid,
        snapshot_type="explicit",
        tokens_before=total_tokens,
        tokens_after=snapshot_tokens,
        compactor_model=compactor_model,
        snapshot_content=snapshot_content or "",
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

    await db.commit()

    logger.info(
        "compaction_done trace=%s conv=%s tokens_before=%d tokens_after=%d compactor=%s",
        _trace,
        conversation_id[:12],
        total_tokens,
        snapshot_tokens,
        compactor_model,
    )

    return Ok({
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
    })


# ── Main entry point ──────────────────────────────────────────────────────


async def compact_conversation(
    conversation_id: str,
    db: AsyncSessionPort,
    config,
    arq_pool=None,
    valkey=None,
) -> Result[dict, ConversationNotFound | EmptyConversation | HistoryTooLargeForCompactor | CompactionFailed]:
    """Explicitly compact a conversation into a structured snapshot.

    Args:
        conversation_id: UUID string of the conversation.
        db: Async DB session.
        config: Proxy config with pseudo-model definitions.
        arq_pool: Optional arq Redis pool for async dispatch (>500K tokens).
        valkey: Optional Valkey client for image description cache.

    Returns:
        Ok(dict) with compaction result metadata on success,
        Err(...) for domain errors.
    """
    conv_uuid = _parse_uuid(conversation_id)
    conv = await db.get(Conversation, conv_uuid)
    if not conv:
        # Return domain error
        return Err(ConversationNotFound(conversation_id=conversation_id))

    _trace = str(uuid.uuid4())[:8]
    logger.info(
        "compaction_start trace=%s conv=%s",
        _trace,
        conversation_id[:12],
    )

    # Load all turns
    result = await db.execute(
        select(ConversationTurn)
        .where(ConversationTurn.conversation_id == conv_uuid)  # type: ignore[arg-type]  # justification: ORM column comparison: SQLModel Field() types don't expose InstrumentedAttribute; mypy sees bool/int, runtime returns BinaryExpression
        .order_by(ConversationTurn.turn_number)  # type: ignore[arg-type]  # justification: ORM column comparison: SQLModel Field() types don't expose InstrumentedAttribute; mypy sees bool/int, runtime returns BinaryExpression
    )
    turns = result.scalars().all()  # type: ignore[union-attr]  # justification: ScalarResult.all() returns Sequence[object]; actual type depends on query entity

    if not turns:
        # Return domain error
        return Err(EmptyConversation(conversation_id=conversation_id))

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

    all_messages.extend(_build_compaction_history(turns))  # type: ignore[arg-type]  # justification: ScalarResult.all() returns Sequence[object]; turns are ConversationTurn at runtime
    total_tokens = conv.total_tokens

    # Select compactor model (requires a "compactador" pseudo-model in config)
    compactor_phys = select_compactor_model(config, total_tokens)
    if not compactor_phys:
        return Err(HistoryTooLargeForCompactor(
            total_tokens=total_tokens,
            max_compactor_window=0,
        ))

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
        return Ok({
            "status": "processing",
            "task_id": job.job_id,
            "message": (
                f"Compaction dispatched to background worker. "
                f"Check status at GET /conversations/{conversation_id}."
            ),
            "estimated_tokens": total_tokens,
            "compactor_model": compactor_model,
        })

    # Synchronous compaction for smaller histories
    return await _run_compaction_sync(  # type: ignore[return-value]  # CompactionFailed ⊂ union
        conversation_id=conversation_id,
        compactor_model=compactor_model,
        api_base=api_base,
        api_key=api_key,
        all_messages=all_messages,
        total_tokens=total_tokens,
        db=db,
        config=config,
        conv=conv,
        turn_count=len(turns),
        valkey=valkey,
        trace_id=_trace,
    )


# ── Async helper for arq worker ──────────────────────────────────────────


async def _compact_async(
    conversation_id: str,
    compactor_model: str,
    db_session_factory,
    config,
    api_base: str | None = None,
    api_key: str | None = None,
    valkey=None,
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
            .where(ConversationTurn.conversation_id == conv_uuid)  # type: ignore[arg-type]  # justification: ORM column comparison: SQLModel Field() types don't expose InstrumentedAttribute; mypy sees bool/int, runtime returns BinaryExpression
            .order_by(ConversationTurn.turn_number)  # type: ignore[arg-type]  # justification: ORM column comparison: SQLModel Field() types don't expose InstrumentedAttribute; mypy sees bool/int, runtime returns BinaryExpression
        )
        turns = result.scalars().all()  # type: ignore[union-attr]  # justification: ScalarResult.all() returns Sequence[object]; actual type depends on query entity

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

        all_messages.extend(_build_compaction_history(turns))  # type: ignore[arg-type]  # justification: ScalarResult.all() returns Sequence[object]; turns are ConversationTurn at runtime
        total_tokens = conv.total_tokens

        compaction_result = await _run_compaction_sync(
            conversation_id=conversation_id,
            compactor_model=compactor_model,
            all_messages=all_messages,
            total_tokens=total_tokens,
            db=db,
            config=config,
            conv=conv,
            api_base=api_base,
            api_key=api_key,
            turn_count=len(turns),
            valkey=valkey,
        )
        # Handle Result type
        if isinstance(compaction_result, Err):
            return {"status": "failed", "error": str(compaction_result.error)}
        return compaction_result.value
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


async def _prepare_multimedia_for_compaction(
    history: list[dict],
    config,
    valkey=None,
) -> list[dict]:
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
            history,
            vision_model,
            api_base=api_base,
            api_key=api_key,
            valkey=valkey,
        )
        return described
    except Exception:
        return history
