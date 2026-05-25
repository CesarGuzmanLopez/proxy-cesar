"""Tests for the audit log endpoint.

Sprint 6 §5: GET /conversations/{id}/audit-log.
Minimum 4 tests per sprint spec.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.api.conversations import audit_log


@pytest.fixture
def mock_db():
    """Returns a mock AsyncSession."""
    db = AsyncMock()
    db.get = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    db.close = AsyncMock()
    scalar_result = MagicMock()
    scalar_result.scalar = MagicMock(return_value=0)
    scalar_result.scalars = MagicMock(
        return_value=MagicMock(all=MagicMock(return_value=[]))
    )
    db.execute = AsyncMock(return_value=scalar_result)
    return db


def _make_request(db_session_factory, config):
    """Create a mock FastAPI request with given state."""
    request = AsyncMock()
    request.app.state.db_session_factory = db_session_factory
    request.app.state.config = config
    return request


# ── Tests ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_audit_log_creation_event():
    """Audit log includes conversation_created event."""
    db = AsyncMock()
    conv_id = uuid.uuid4()
    conv = MagicMock()
    conv.id = conv_id
    conv.created_at.isoformat.return_value = "2026-05-24T10:00:00"
    conv.pseudo_model = "normal"
    conv.physical_model = "qwen3-max"
    conv.active_snapshot_id = None
    db.get = AsyncMock(return_value=conv)

    scalar_result = MagicMock()
    scalar_result.scalars = MagicMock(
        return_value=MagicMock(all=MagicMock(return_value=[]))
    )
    db.execute = AsyncMock(return_value=scalar_result)

    db_session_factory = MagicMock(return_value=db)
    request = _make_request(db_session_factory, MagicMock())

    result = await audit_log(str(conv_id), request)

    assert result["conversation_id"] == str(conv_id)
    assert len(result["events"]) >= 1
    assert result["events"][0]["event_type"] == "conversation_created"
    assert result["events"][0]["details"]["pseudo_model"] == "normal"


@pytest.mark.asyncio
async def test_audit_log_pseudo_model_switch():
    """Audit log includes pseudo_model_switched events when pseudo-model changes."""
    db = AsyncMock()
    conv_id = uuid.uuid4()
    conv = MagicMock()
    conv.id = conv_id
    conv.created_at.isoformat.return_value = "2026-05-24T10:00:00"
    conv.pseudo_model = "normal"
    conv.physical_model = "qwen3-max"
    conv.active_snapshot_id = None
    db.get = AsyncMock(return_value=conv)

    # Mock a turn with pseudo-model switch
    turn = MagicMock()
    turn.turn_type = "normal"
    turn.pseudo_model = "tareas-avanzadas"
    turn.fallback_applied = False
    turn.turn_number = 5
    turn.created_at.isoformat.return_value = "2026-05-24T10:30:00"

    scalar_result = MagicMock()
    scalar_result.scalars = MagicMock(
        return_value=MagicMock(all=MagicMock(return_value=[turn]))
    )
    db.execute = AsyncMock(return_value=scalar_result)

    db_session_factory = MagicMock(return_value=db)
    request = _make_request(db_session_factory, MagicMock())

    result = await audit_log(str(conv_id), request)

    events = result["events"]
    switch_events = [e for e in events if e["event_type"] == "pseudo_model_switched"]
    assert len(switch_events) == 1
    assert switch_events[0]["details"]["from"] == "normal"
    assert switch_events[0]["details"]["to"] == "tareas-avanzadas"


@pytest.mark.asyncio
async def test_audit_log_fallback_event():
    """Audit log includes fallback_applied events."""
    db = AsyncMock()
    conv_id = uuid.uuid4()
    conv = MagicMock()
    conv.id = conv_id
    conv.created_at.isoformat.return_value = "2026-05-24T10:00:00"
    conv.pseudo_model = "normal"
    conv.physical_model = "qwen3-max"
    conv.active_snapshot_id = None
    db.get = AsyncMock(return_value=conv)

    # Mock a turn with fallback
    turn = MagicMock()
    turn.turn_type = "normal"
    turn.pseudo_model = "normal"
    turn.fallback_applied = True
    turn.fallback_reason = "ServiceUnavailableError: qwen3-max"
    turn.turn_number = 3
    turn.created_at.isoformat.return_value = "2026-05-24T10:15:00"

    scalar_result = MagicMock()
    scalar_result.scalars = MagicMock(
        return_value=MagicMock(all=MagicMock(return_value=[turn]))
    )
    db.execute = AsyncMock(return_value=scalar_result)

    db_session_factory = MagicMock(return_value=db)
    request = _make_request(db_session_factory, MagicMock())

    result = await audit_log(str(conv_id), request)

    fallback_events = [
        e for e in result["events"] if e["event_type"] == "fallback_applied"
    ]
    assert len(fallback_events) == 1
    assert (
        fallback_events[0]["details"]["reason"] == "ServiceUnavailableError: qwen3-max"
    )


@pytest.mark.asyncio
async def test_audit_log_compaction_event():
    """Audit log includes compaction_explicit events."""
    from src.adapters.db.models import ConversationSnapshot

    db = AsyncMock()
    conv_id = uuid.uuid4()
    conv = MagicMock()
    conv.id = conv_id
    conv.created_at.isoformat.return_value = "2026-05-24T10:00:00"
    conv.pseudo_model = "normal"
    conv.physical_model = "qwen3-max"
    conv.active_snapshot_id = uuid.uuid4()
    db.get = AsyncMock(return_value=conv)

    # Mock a snapshot
    snap = MagicMock(spec=ConversationSnapshot)
    snap.created_at.isoformat.return_value = "2026-05-24T14:00:00"
    snap.snapshot_type = "explicit"
    snap.tokens_before = 1200000
    snap.tokens_after = 10240
    snap.compactor_model = "gemini-3.5-flash"

    scalar_result_turns = MagicMock()
    scalar_result_turns.scalars = MagicMock(
        return_value=MagicMock(all=MagicMock(return_value=[]))
    )
    db.execute = AsyncMock(return_value=scalar_result_turns)

    scalar_result_snaps = MagicMock()
    scalar_result_snaps.scalars = MagicMock(
        return_value=MagicMock(all=MagicMock(return_value=[snap]))
    )
    # Return snaps on second call, turns on first
    db.execute = AsyncMock(side_effect=[scalar_result_turns, scalar_result_snaps])

    db_session_factory = MagicMock(return_value=db)
    request = _make_request(db_session_factory, MagicMock())

    result = await audit_log(str(conv_id), request)

    compaction_events = [
        e for e in result["events"] if e["event_type"] == "compaction_explicit"
    ]
    assert len(compaction_events) == 1
    assert compaction_events[0]["details"]["tokens_before"] == 1200000
    assert compaction_events[0]["details"]["tokens_after"] == 10240
    assert compaction_events[0]["details"]["compactor"] == "gemini-3.5-flash"


@pytest.mark.asyncio
async def test_audit_log_events_chronological():
    """Audit log events are sorted chronologically by timestamp."""
    db = AsyncMock()
    conv_id = uuid.uuid4()
    conv = MagicMock()
    conv.id = conv_id
    conv.created_at.isoformat.return_value = "2026-05-24T10:00:00"
    conv.pseudo_model = "normal"
    conv.physical_model = "qwen3-max"
    conv.active_snapshot_id = None
    db.get = AsyncMock(return_value=conv)

    # Create events with out-of-order creation
    turn2 = MagicMock()
    turn2.turn_type = "normal"
    turn2.pseudo_model = "normal"
    turn2.fallback_applied = True
    turn2.fallback_reason = "RateLimitError"
    turn2.turn_number = 2
    turn2.created_at.isoformat.return_value = "2026-05-24T10:05:00"

    turn1 = MagicMock()
    turn1.turn_type = "normal"
    turn1.pseudo_model = "tareas-avanzadas"
    turn1.fallback_applied = False
    turn1.turn_number = 1
    turn1.created_at.isoformat.return_value = "2026-05-24T10:02:00"

    scalar_result = MagicMock()
    scalar_result.scalars = MagicMock(
        return_value=MagicMock(all=MagicMock(return_value=[turn1, turn2]))
    )
    db.execute = AsyncMock(return_value=scalar_result)

    db_session_factory = MagicMock(return_value=db)
    request = _make_request(db_session_factory, MagicMock())

    result = await audit_log(str(conv_id), request)

    timestamps = [e["timestamp"] for e in result["events"]]
    assert timestamps == sorted(timestamps)


@pytest.mark.asyncio
async def test_audit_log_conversation_not_found():
    """Audit log on non-existent conversation returns 404."""
    db = AsyncMock()
    db.get = AsyncMock(return_value=None)
    db_session_factory = MagicMock(return_value=db)
    request = _make_request(db_session_factory, MagicMock())

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        await audit_log(str(uuid.uuid4()), request)
    assert exc_info.value.status_code == 404
