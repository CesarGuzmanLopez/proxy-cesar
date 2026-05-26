"""Conversation state, compatibility, compact, audit, and tool normalization endpoints.

Sprint 2 §6: GET /conversations/{id}, GET /compatible-models, GET /tools-compatibility.
Sprint 3 §4.3: POST /conversations/{id}/normalize-tools.
Sprint 6 §3: POST /conversations/{id}/compact, GET /conversations/{id}/audit-log.
"""

import uuid

from fastapi import APIRouter, HTTPException, Request
from starlette.responses import JSONResponse
from sqlalchemy import func, select
from sqlmodel.ext.asyncio.session import AsyncSession

from src.adapters.db.models import Conversation, ConversationSnapshot, ConversationTurn
from src.schemas.tools import NormalizeToolsRequest, NormalizeToolsResponse
from src.service.capability_detector import load_session_capabilities
from src.service.compatibility import validate_switch
from src.service.compactor.explicit import compact_conversation
from src.service.tools_normalizer import generate_preview, normalize_history

router = APIRouter()


@router.get(
    "/conversations/{conversation_id}",
    responses={404: {"description": "Conversation not found"}},
)
async def get_conversation(
    conversation_id: str,
    fastapi_request: Request,
):
    """Return full state of a conversation with capabilities.

    GET /conversations/{id}
    """
    db: AsyncSession = fastapi_request.app.state.db_session_factory()

    try:
        try:
            conv_uuid = uuid.UUID(conversation_id)
        except ValueError:
            conv_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, conversation_id)

        conv = await db.get(Conversation, conv_uuid)
        if conv is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "CONVERSATION_NOT_FOUND",
                    "message": f"Conversation '{conversation_id}' not found.",
                },
            )

        # Count turns (avoid lazy load of conv.turns)
        result = await db.execute(
            select(func.count(ConversationTurn.id)).where(
                ConversationTurn.conversation_id == conv_uuid
            )
        )
        turn_count = result.scalar() or 0

        # Load capabilities safely (handle missing fields gracefully)
        has_images = getattr(conv, "capability_has_images", False)
        has_audio = getattr(conv, "capability_has_audio", False)
        has_pdf = getattr(conv, "capability_has_pdf", False)
        has_video = getattr(conv, "capability_has_video", False)
        has_tools = getattr(conv, "capability_has_tools", False)
        has_parallel = getattr(conv, "capability_has_parallel_tools", False)
        max_tools_level = getattr(conv, "max_tools_level", 0)

        return {
            "conversation_id": conversation_id,
            "created_at": conv.created_at.isoformat() if conv.created_at else None,
            "pseudo_model": conv.pseudo_model,
            "physical_model": conv.physical_model,
            "total_tokens": conv.total_tokens,
            "turn_count": turn_count,
            "capabilities": {
                "has_images": has_images,
                "has_audio": has_audio,
                "has_pdf": has_pdf,
                "has_video": has_video,
                "has_tools": has_tools,
                "has_parallel_tools": has_parallel,
            },
            "max_tools_level": max_tools_level,
            "active_snapshot_id": str(conv.active_snapshot_id)
            if conv.active_snapshot_id
            else None,
        }
    finally:
        await db.close()


@router.get(
    "/conversations/{conversation_id}/compatible-models",
    responses={404: {"description": "Conversation not found"}},
)
async def get_compatible_models(
    conversation_id: str,
    fastapi_request: Request,
):
    """Return ALL pseudo-models with their compatibility status.

    GET /conversations/{id}/compatible-models
    """
    config = fastapi_request.app.state.config
    db: AsyncSession = fastapi_request.app.state.db_session_factory()

    try:
        try:
            conv_uuid = uuid.UUID(conversation_id)
        except ValueError:
            conv_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, conversation_id)

        conv = await db.get(Conversation, conv_uuid)
        if conv is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "CONVERSATION_NOT_FOUND",
                    "message": f"Conversation '{conversation_id}' not found.",
                },
            )

        # Build session capabilities from DB
        caps = await load_session_capabilities(db, conv_uuid, conv.total_tokens)

        # Evaluate every pseudo-model
        compatible_models = []
        current_pseudo = conv.pseudo_model

        for name, pm in config.pseudo_models.items():
            if name == current_pseudo:
                compatible_models.append(
                    {
                        "pseudo_model": name,
                        "display_name": pm.display_name,
                        "status": "safe",
                        "reason": "Current pseudo-model.",
                    }
                )
                continue

            result = validate_switch(
                from_pseudo_name=current_pseudo,
                to_pseudo_name=name,
                to_pseudo=pm,
                caps=caps,
                config=config,
            )

            compatible_models.append(
                {
                    "pseudo_model": name,
                    "display_name": pm.display_name,
                    "status": result.status.value,
                    "reason": result.reason,
                    "remediation": result.remediation if result.remediation else None,
                }
            )

        return {
            "conversation_id": conversation_id,
            "current_pseudo_model": current_pseudo,
            "capabilities": {
                "has_images": caps.has_images,
                "has_tools": caps.has_tools,
                "has_parallel_tools": caps.has_parallel_tools,
            },
            "compatible_models": compatible_models,
        }
    finally:
        await db.close()


@router.get(
    "/conversations/{conversation_id}/tools-compatibility",
    responses={404: {"description": "Conversation not found"}},
)
async def get_tools_compatibility(
    conversation_id: str,
    fastapi_request: Request,
):
    """Return tool-specific compatibility analysis per pseudo-model.

    GET /conversations/{id}/tools-compatibility
    """
    config = fastapi_request.app.state.config
    db: AsyncSession = fastapi_request.app.state.db_session_factory()

    try:
        try:
            conv_uuid = uuid.UUID(conversation_id)
        except ValueError:
            conv_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, conversation_id)

        conv = await db.get(Conversation, conv_uuid)
        if conv is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "CONVERSATION_NOT_FOUND",
                    "message": f"Conversation '{conversation_id}' not found.",
                },
            )

        # Check if conversation has tools
        has_tools = getattr(conv, "capability_has_tools", False)
        has_parallel = getattr(conv, "capability_has_parallel_tools", False)

        pseudo_models_list = []
        for name, pm in config.pseudo_models.items():
            parallel_eligible = [
                m.model for m in pm.physical_models if m.parallel_tools
            ]
            strict_models = [m.model for m in pm.physical_models if m.tools_strict]
            non_strict = [m.model for m in pm.physical_models if not m.tools_strict]
            blocked = [m.model for m in pm.physical_models if not m.parallel_tools]

            pseudo_models_list.append(
                {
                    "name": name,
                    "display_name": pm.display_name,
                    "tool_support": {
                        "parallel_eligible": len(parallel_eligible) > 0,
                        "parallel_models": parallel_eligible,
                        "strict_models": strict_models,
                        "non_strict_models": non_strict,
                        "blocked_models": (blocked if has_parallel else []),
                    },
                }
            )

        return {
            "conversation_id": conversation_id,
            "tools_used": has_tools,
            "parallel_tools_used": has_parallel,
            "pseudo_models": pseudo_models_list,
        }
    finally:
        await db.close()


@router.post(
    "/conversations/{conversation_id}/normalize-tools",
    responses={
        400: {"description": "No parallel tools to normalize"},
        404: {"description": "Conversation not found"},
        500: {"description": "Normalization failed"},
    },
)
async def normalize_tools(
    conversation_id: str,
    request: NormalizeToolsRequest,
    fastapi_request: Request,
):
    """Serialize parallel tool calls in the conversation history.

    plan-proxy.md §6.8: POST /conversations/{id}/normalize-tools.
    The original history is preserved. A normalization_event turn is inserted.
    After this, the conversation can switch to pseudo-models without
    parallel tool support.

    If dry_run is true, returns the preview without modifying the conversation.
    """
    db: AsyncSession = fastapi_request.app.state.db_session_factory()

    try:
        try:
            conv_uuid = uuid.UUID(conversation_id)
        except ValueError:
            conv_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, conversation_id)

        conv = await db.get(Conversation, conv_uuid)
        if conv is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "CONVERSATION_NOT_FOUND",
                    "message": f"Conversation '{conversation_id}' not found.",
                },
            )

        # Check if conversation has parallel tools
        has_parallel = getattr(conv, "capability_has_parallel_tools", False)
        if not has_parallel:
            # Check individual turns for parallel calls (capability might not be set)
            pass  # Continue anyway — normalize_history will handle empty case

        # Load all turns in order
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
                    "error": "NO_TURNS",
                    "message": "Conversation has no turns to normalize.",
                },
            )

        # Reconstruct full message history
        all_messages: list[dict] = []
        for turn in turns:
            turn_messages = turn.messages
            if isinstance(turn_messages, dict) and "messages" in turn_messages:
                all_messages.extend(turn_messages["messages"])
            elif isinstance(turn_messages, dict):
                all_messages.append(turn_messages)
            elif isinstance(turn_messages, list):
                all_messages.extend(turn_messages)

        # Normalize
        normalized_messages, meta = normalize_history(all_messages)

        if meta.turns_serialized == 0:
            return {
                "conversation_id": conversation_id,
                "normalized_turns": 0,
                "parallel_calls_serialized": 0,
                "turns_affected": [],
                "original_history_preserved": True,
                "normalization_event_id": None,
                "preview": "No parallel tool calls found. Nothing to normalize.",
                "message": "This conversation has no parallel tool calls to normalize.",
            }

        # Generate preview
        preview = generate_preview(all_messages, meta)

        if request.dry_run:
            return NormalizeToolsResponse(
                conversation_id=conversation_id,
                normalized_turns=meta.turns_serialized,
                parallel_calls_serialized=meta.parallel_calls_serialized,
                turns_affected=meta.affected_turns,
                original_history_preserved=True,
                normalization_event_id=None,
                preview=preview,
            )

        # Create normalization event turn
        norm_turn = ConversationTurn(
            conversation_id=conv_uuid,
            turn_number=len(turns) + 1,
            turn_type="normalization_event",
            pseudo_model=conv.pseudo_model,
            physical_model=conv.physical_model,
            messages={
                "normalized_history": normalized_messages,
                "metadata": {
                    "turns_serialized": meta.turns_serialized,
                    "parallel_calls_serialized": meta.parallel_calls_serialized,
                    "affected_turns": meta.affected_turns,
                },
            },
        )
        db.add(norm_turn)
        await db.commit()
        await db.refresh(norm_turn)

        return NormalizeToolsResponse(
            conversation_id=conversation_id,
            normalized_turns=meta.turns_serialized,
            parallel_calls_serialized=meta.parallel_calls_serialized,
            turns_affected=meta.affected_turns,
            original_history_preserved=True,
            normalization_event_id=str(norm_turn.id),
            preview=preview,
        )

    except HTTPException:
        await db.rollback()
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=500,
            detail={
                "error": "NORMALIZATION_FAILED",
                "message": f"Failed to normalize tools: {e}",
            },
        ) from e
    finally:
        await db.close()


# ── Sprint 6: POST /conversations/{id}/compact ──────────────────────────────


@router.post(
    "/conversations/{conversation_id}/compact",
    responses={
        400: {"description": "Empty conversation or history too large for compactor"},
        404: {"description": "Conversation not found"},
        502: {"description": "Compactor model failed"},
    },
)
async def compact_conversation_endpoint(
    conversation_id: str,
    fastapi_request: Request,
) -> dict:
    """Explicitly compact a conversation into a structured Markdown snapshot.

    POST /conversations/{id}/compact
    plan-proxy.md §11: Compactación explícita de conversaciones.
    For histories >500K tokens, dispatches to arq background worker.

    Returns:
        Dict with snapshot_id, tokens_before/after, preview, status.
    """
    config = fastapi_request.app.state.config
    arq_pool = getattr(fastapi_request.app.state, "arq_pool", None)

    db = fastapi_request.app.state.db_session_factory()
    try:
        return await compact_conversation(
            conversation_id=conversation_id,
            db=db,
            config=config,
            arq_pool=arq_pool,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail={"error": "COMPACTION_FAILED", "message": str(e)},
        ) from e
    finally:
        await db.close()


# ── Sprint 6: GET /conversations/{id}/audit-log ─────────────────────────────


@router.get(
    "/conversations/{conversation_id}/audit-log",
    responses={404: {"description": "Conversation not found"}},
)
async def audit_log(
    conversation_id: str,
    fastapi_request: Request,
) -> dict:
    """Return chronological event log for a conversation.

    GET /conversations/{id}/audit-log
    Constructed by scanning conversation_turns + conversation_snapshots.
    No separate audit table needed — Sprint 6 §5.

    Events include: conversation_created, pseudo_model_switched,
    fallback_applied, compaction_explicit, compaction_continuous,
    normalization_event, degradation_event.

    Returns:
        Dict with conversation_id and ordered events list.
    """
    db: AsyncSession = fastapi_request.app.state.db_session_factory()

    try:
        conv_uuid = _parse_uuid(conversation_id)
        conv = await db.get(Conversation, conv_uuid)
        if not conv:
            raise HTTPException(
                status_code=404,
                detail={"error": "CONVERSATION_NOT_FOUND"},
            )

        events: list[dict] = []

        # Conversation created
        events.append(
            {
                "timestamp": conv.created_at.isoformat(),
                "event_type": "conversation_created",
                "details": {
                    "pseudo_model": conv.pseudo_model,
                    "physical_model": conv.physical_model,
                },
            }
        )

        # Turns: detect pseudo-model switches, fallbacks, event types
        turns_result = await db.execute(
            select(ConversationTurn)
            .where(ConversationTurn.conversation_id == conv_uuid)
            .order_by(ConversationTurn.turn_number)
        )
        prev_pseudo = conv.pseudo_model
        for turn in turns_result.scalars().all():
            # Pseudo-model switch detection
            if turn.turn_type == "normal" and turn.pseudo_model != prev_pseudo:
                events.append(
                    {
                        "timestamp": turn.created_at.isoformat(),
                        "event_type": "pseudo_model_switched",
                        "details": {
                            "from": prev_pseudo,
                            "to": turn.pseudo_model,
                            "turn": turn.turn_number,
                        },
                    }
                )
                prev_pseudo = turn.pseudo_model

            # Fallback
            if turn.fallback_applied:
                events.append(
                    {
                        "timestamp": turn.created_at.isoformat(),
                        "event_type": "fallback_applied",
                        "details": {
                            "turn": turn.turn_number,
                            "reason": turn.fallback_reason,
                        },
                    }
                )

            # Non-normal event turns
            if turn.turn_type != "normal":
                events.append(
                    {
                        "timestamp": turn.created_at.isoformat(),
                        "event_type": turn.turn_type,
                        "details": {"turn": turn.turn_number},
                    }
                )

        # Snapshots (explicit + continuous + external)
        snap_result = await db.execute(
            select(ConversationSnapshot)
            .where(ConversationSnapshot.conversation_id == conv_uuid)
            .order_by(ConversationSnapshot.created_at)
        )
        for snap in snap_result.scalars().all():
            events.append(
                {
                    "timestamp": snap.created_at.isoformat(),
                    "event_type": f"compaction_{snap.snapshot_type}",
                    "details": {
                        "tokens_before": snap.tokens_before,
                        "tokens_after": snap.tokens_after,
                        "compactor": snap.compactor_model,
                    },
                }
            )

        events.sort(key=lambda e: e["timestamp"])

        return {"conversation_id": conversation_id, "events": events}

    except HTTPException:
        await db.rollback()
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=500,
            detail={"error": "AUDIT_LOG_FAILED", "message": str(e)},
        ) from e
    finally:
        await db.close()


# ── Blob retrieval ────────────────────────────────────────────────────────────


@router.get("/blobs/{blob_hash}")
async def get_blob(blob_hash: str, request: Request):
    """Retrieve a stored base64 blob by hash.

    Blobs are stored by the content transformation when the user sends
    base64-encoded content (images, audio, files) to a model that can't
    process them. Tools and sub-models can fetch the real base64 data
    from this endpoint when they need to process the content.
    """
    valkey = request.app.state.valkey
    if valkey is None:
        raise HTTPException(status_code=503, detail={"error": "BLOB_STORE_UNAVAILABLE"})

    try:
        cursor = 0
        pattern = f"blob:*:{blob_hash}"
        scanned = 0
        while scanned < 10:  # limit scan iterations
            cursor, keys = await valkey.scan(cursor=cursor, match=pattern, count=100)
            for key in keys:
                data = await valkey.get(key)
                if data:
                    return JSONResponse(content={"data": data})
            scanned += 1
            if cursor == 0:
                break
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail={"error": "BLOB_STORE_ERROR", "message": str(exc)},
        ) from exc

    raise HTTPException(status_code=404, detail={"error": "BLOB_NOT_FOUND"})


# ── Shared helper ──────────────────────────────────────────────────────────────


def _parse_uuid(value: str) -> uuid.UUID:
    """Parse a UUID string, falling back to UUID5 for non-UUID strings."""
    try:
        return uuid.UUID(value)
    except ValueError:
        return uuid.uuid5(uuid.NAMESPACE_DNS, value)
