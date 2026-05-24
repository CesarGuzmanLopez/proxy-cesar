"""Conversation state, compatibility, and tool normalization endpoints.

Sprint 2 §6: GET /conversations/{id}, GET /compatible-models, GET /tools-compatibility.
Sprint 3 §4.3: POST /conversations/{id}/normalize-tools.
Sprint 5 §6: POST /conversations/{id}/degrade-images.
"""

import uuid

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload
from sqlmodel.ext.asyncio.session import AsyncSession

from src.adapters.db.models import Conversation, ConversationTurn
from src.config.pseudo_models import ProxyConfigSchema
from src.schemas.tools import NormalizeToolsRequest, NormalizeToolsResponse
from src.service.capability_detector import load_session_capabilities
from src.service.compatibility import validate_switch
from src.service.multimedia.image_describer import auto_describe_images
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
            select(func.count(ConversationTurn.id))
            .where(ConversationTurn.conversation_id == conv_uuid)
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
            "active_snapshot_id": str(conv.active_snapshot_id) if conv.active_snapshot_id else None,
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
                compatible_models.append({
                    "pseudo_model": name,
                    "display_name": pm.display_name,
                    "status": "safe",
                    "reason": "Current pseudo-model.",
                })
                continue

            result = validate_switch(
                from_pseudo_name=current_pseudo,
                to_pseudo_name=name,
                to_pseudo=pm,
                caps=caps,
                config=config,
            )

            compatible_models.append({
                "pseudo_model": name,
                "display_name": pm.display_name,
                "status": result.status.value,
                "reason": result.reason,
                "remediation": result.remediation if result.remediation else None,
            })

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
            parallel_eligible = [m.model for m in pm.physical_models if m.parallel_tools]
            strict_models = [m.model for m in pm.physical_models if m.tools_strict]
            non_strict = [m.model for m in pm.physical_models if not m.tools_strict]
            blocked = [m.model for m in pm.physical_models if not m.parallel_tools]

            pseudo_models_list.append({
                "name": name,
                "display_name": pm.display_name,
                "tool_support": {
                    "parallel_eligible": len(parallel_eligible) > 0,
                    "parallel_models": parallel_eligible,
                    "strict_models": strict_models,
                    "non_strict_models": non_strict,
                    "blocked_models": (
                        blocked if has_parallel else []
                    ),
                },
            })

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


# ── Sprint 5: POST /conversations/{id}/degrade-images ─────────────────────────


@router.post(
    "/conversations/{conversation_id}/degrade-images",
    responses={
        404: {"description": "Conversation not found"},
        400: {"description": "No images to degrade or no vision model available"},
        502: {"description": "Image degradation failed"},
    },
)
async def degrade_images(
    conversation_id: str,
    fastapi_request: Request,
) -> dict:
    """Manually degrade images in a conversation to text descriptions.

    After this endpoint completes, the conversation's images are described
    textually. A subsequent switch to a non-vision pseudo-model will be SAFE
    (images already described).

    python.md §6: FastAPI router — HTTP boundary only.
    python.md §3: HTTPException for errors at boundary.
    """
    db: AsyncSession = fastapi_request.app.state.db_session_factory()
    config: ProxyConfigSchema = fastapi_request.app.state.config

    try:
        conv_uuid = _parse_uuid(conversation_id)
        conv = await db.get(
            Conversation, conv_uuid,
            options=[selectinload(Conversation.turns)],
        )
        if conv is None:
            raise HTTPException(status_code=404, detail={
                "error": "CONVERSATION_NOT_FOUND",
            })

        # Check if images exist
        caps = await load_session_capabilities(db, conv_uuid)
        if not caps.has_images:
            raise HTTPException(status_code=400, detail={
                "error": "NO_IMAGES",
                "message": "This conversation has no images to degrade.",
            })

        # Find a vision model in the current pseudo-model
        current_pm = config.pseudo_models.get(conv.pseudo_model)
        if current_pm is None:
            raise HTTPException(status_code=400, detail={
                "error": "UNKNOWN_PSEUDO_MODEL",
                "message": f"Current pseudo-model '{conv.pseudo_model}' not found.",
            })

        vision_models = [m for m in current_pm.physical_models if m.vision]
        if not vision_models:
            raise HTTPException(status_code=400, detail={
                "error": "NO_VISION_MODEL",
                "message": (
                    f"Current pseudo-model '{conv.pseudo_model}' has no "
                    f"vision-capable physical model to describe images."
                ),
            })

        # Use pinned model if it has vision, otherwise first vision model
        vision_model: str = (
            conv.physical_model
            if any(
                m.model == conv.physical_model and m.vision
                for m in current_pm.physical_models
            )
            else vision_models[0].model
        )

        # Load all conversation messages from turns
        all_messages: list[dict] = []
        for turn in sorted(conv.turns, key=lambda t: t.turn_number):
            turn_msgs = turn.messages
            if isinstance(turn_msgs, list):
                all_messages.extend(turn_msgs)

        # Auto-describe
        described_messages, desc_meta = await auto_describe_images(
            all_messages, vision_model,
        )

        described_count = desc_meta.get("images_described", 0)
        if described_count == 0:
            return {
                "conversation_id": conversation_id,
                "images_described": 0,
                "described_by": vision_model,
                "message": "No images found to degrade.",
            }

        # Store as a degradation_event turn
        turn_number: int = (
            max(t.turn_number for t in conv.turns) + 1
        ) if conv.turns else 1

        deg_turn = ConversationTurn(
            conversation_id=conv_uuid,
            turn_number=turn_number,
            pseudo_model=conv.pseudo_model,
            physical_model=vision_model,
            messages=described_messages,
            response={"metadata": desc_meta},
            input_tokens=0,
            output_tokens=desc_meta.get("total_description_tokens", 0),
            turn_type="degradation_event",
            had_images=False,
            had_tools=False,
            had_parallel_tools=False,
        )
        db.add(deg_turn)

        # Update conversation tracking
        conv.images_described = (conv.images_described or 0) + described_count
        conv.images_degraded_manually = True
        await db.commit()

        return {
            "conversation_id": conversation_id,
            "images_described": described_count,
            "described_by": vision_model,
            "can_now_switch_to": [
                name
                for name, pm in config.pseudo_models.items()
                if not any(m.vision for m in pm.physical_models)
            ],
        }

    except HTTPException:
        await db.rollback()
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=502,
            detail={
                "error": "DEGRADE_IMAGES_FAILED",
                "message": str(e),
            },
        ) from e
    finally:
        await db.close()


# ── Shared helper ──────────────────────────────────────────────────────────────


def _parse_uuid(value: str) -> uuid.UUID:
    """Parse a UUID string, falling back to UUID5 for non-UUID strings."""
    try:
        return uuid.UUID(value)
    except ValueError:
        return uuid.uuid5(uuid.NAMESPACE_DNS, value)
