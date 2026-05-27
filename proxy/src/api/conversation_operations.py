"""Compact, audit log, and blob retrieval endpoints.

Sprint 6: POST /conversations/{id}/compact, GET /conversations/{id}/audit-log, GET /blobs/{hash}.
Sprint 5: POST /conversations/{id}/degrade-images.
"""

import uuid

from fastapi import APIRouter, HTTPException, Request
from starlette.responses import JSONResponse
from sqlalchemy import select
from sqlmodel.ext.asyncio.session import AsyncSession

from src.adapters.db.models import Conversation, ConversationSnapshot, ConversationTurn
from src.service.capability_detector import load_session_capabilities
from src.service.compactor.explicit import compact_conversation

router = APIRouter()


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
    valkey = getattr(fastapi_request.app.state, "valkey", None)

    db = fastapi_request.app.state.db_session_factory()
    try:
        return await compact_conversation(
            conversation_id=conversation_id,
            db=db,
            config=config,
            arq_pool=arq_pool,
            valkey=valkey,
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


@router.post(
    "/conversations/{conversation_id}/degrade-images",
    responses={
        200: {"description": "Images described and degraded"},
        400: {"description": "No images to degrade"},
        404: {"description": "Conversation not found"},
    },
)
async def degrade_images(
    conversation_id: str,
    fastapi_request: Request,
) -> dict:
    """Manually describe all images in a conversation to enable switching.

    POST /conversations/{id}/degrade-images
    Sprint 5: Manual degradation of images so the conversation can switch
    to pseudo-models without vision support.

    Returns:
        Dict with images_described count, described_by, and can_now_switch_to list.
    """
    # Late imports so patches work correctly
    from src.service.multimedia.image_describer import auto_describe_images
    from src.service.chat_messages import _load_messages_from_turns

    db: AsyncSession = fastapi_request.app.state.db_session_factory()
    config = fastapi_request.app.state.config

    try:
        try:
            conv_uuid = uuid.UUID(conversation_id)
        except ValueError:
            conv_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, conversation_id)

        conv = await db.get(Conversation, conv_uuid)
        if conv is None:
            raise HTTPException(
                status_code=404,
                detail={"error": "CONVERSATION_NOT_FOUND"},
            )

        session_caps = await load_session_capabilities(db, conv_uuid)
        if not session_caps.has_images:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "NO_IMAGES",
                    "message": "This conversation has no images to degrade.",
                },
            )

        # Collect all messages from turns
        all_messages = _load_messages_from_turns(conv)

        # Find a vision model from the current pseudo model
        current_pm = config.pseudo_models.get(conv.pseudo_model)
        vision_model = None
        api_base = None
        api_key = None
        if current_pm:
            vision_models = [m for m in current_pm.physical_models if getattr(m, "vision", False)]
            if vision_models:
                vision_phys = vision_models[0]
                vision_model = vision_phys.model
                api_base = getattr(vision_phys, "api_base", None) or None
                from src.service.chat_fallback import _resolve_api_key as _resolve_key
                api_key = _resolve_key(vision_phys)

        if not vision_model:
            vision_model = "groq/meta-llama/llama-4-scout-17b-16e-instruct"

        described, desc_meta = await auto_describe_images(
            all_messages,
            vision_model,
            api_base=api_base,
            api_key=api_key,
        )

        described_count = desc_meta.get("images_described", 0)
        described_by = desc_meta.get("described_by") or vision_model

        # Update conversation state
        conv.capability_has_images = False
        conv.images_described = (conv.images_described or 0) + described_count
        conv.images_degraded_manually = True
        await db.commit()

        # Build list of models that were previously blocked and can now be used
        can_now_switch_to = [
            name
            for name, pm in config.pseudo_models.items()
            if name != conv.pseudo_model
            and not any(getattr(m, "vision", False) for m in pm.physical_models)
        ]

        return {
            "images_described": described_count,
            "described_by": described_by,
            "status": "completed",
            "can_now_switch_to": can_now_switch_to,
        }
    except HTTPException:
        await db.rollback()
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=500,
            detail={"error": "DEGRADE_FAILED", "message": str(e)},
        ) from e
    finally:
        await db.close()


@router.get(
    "/conversations/{conversation_id}/audit-log",
    responses={
        404: {"description": "Conversation not found"},
        500: {"description": "Audit log retrieval failed"},
    },
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

        turns_result = await db.execute(
            select(ConversationTurn)
            .where(ConversationTurn.conversation_id == conv_uuid)
            .order_by(ConversationTurn.turn_number)
        )
        prev_pseudo = conv.pseudo_model
        for turn in turns_result.scalars().all():
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

            if turn.turn_type != "normal":
                events.append(
                    {
                        "timestamp": turn.created_at.isoformat(),
                        "event_type": turn.turn_type,
                        "details": {"turn": turn.turn_number},
                    }
                )

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


@router.get(
    "/blobs/{blob_hash}",
    responses={
        200: {"description": "Base64 blob data"},
        404: {"description": "Blob not found"},
        503: {"description": "Blob store unavailable or error"},
    },
)
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
        while scanned < 10:
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


def _parse_uuid(value: str) -> uuid.UUID:
    """Parse a UUID string, falling back to UUID5 for non-UUID strings."""
    try:
        return uuid.UUID(value)
    except ValueError:
        return uuid.uuid5(uuid.NAMESPACE_DNS, value)
