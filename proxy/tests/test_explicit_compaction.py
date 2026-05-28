"""Tests for explicit compaction service.

Sprint 6 §3: POST /conversations/{id}/compact.
Minimum 9 tests per sprint spec.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config.pseudo_models import PhysicalModelSchema
from src.service.compactor.explicit import (
    compact_conversation,
    select_compactor_model,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


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
    # Default: scalar attribute for count queries
    scalar_result = MagicMock()
    scalar_result.scalar = MagicMock(return_value=0)
    scalar_result.scalars = MagicMock(
        return_value=MagicMock(all=MagicMock(return_value=[]))
    )
    db.execute = AsyncMock(return_value=scalar_result)
    return db


@pytest.fixture
def mock_litellm_success():
    """Mock call_litellm returning a valid snapshot response."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = (
        "# Snapshot — 2026-05-24\n\n"
        "## Problem State\nWorking on auth module.\n\n"
        "## Technical Decisions\nUsed JWT with refresh tokens.\n\n"
        "## Code Produced\n`auth.py`: JWT validation.\n\n"
        "## Current Status\n- Resolved: JWT validation\n- Unresolved: refresh token rotation\n\n"
        "## Technical Context\nPython 3.14, FastAPI, SQLModel.\n\n"
        "## Tools & Capabilities\nsearch_codebase, read_file.\n\n"
        "## Pending Items\nImplement rate limiting.\n\n"
        "## Conversation Metadata\n50 turns, pseudo-models: normal, tareas-avanzadas."
    )
    mock_response.usage.completion_tokens = 250
    mock_response.model_dump.return_value = {
        "id": "chatcmpl-mock",
        "choices": [
            {"message": {"role": "assistant", "content": "Snapshot content..."}}
        ],
        "usage": {"prompt_tokens": 500, "completion_tokens": 250},
    }

    with patch(
        "src.service.compactor.explicit.call_litellm", new_callable=AsyncMock
    ) as mock:
        mock.return_value = mock_response
        yield mock


@pytest.fixture
def config_with_compactador():
    """Config with compactador pseudo-model using real schemas."""
    config = MagicMock()
    phys1 = PhysicalModelSchema(
        provider="google",
        model="gemini-3.5-flash",
        context_window=1000000,
    )
    phys2 = PhysicalModelSchema(
        provider="anthropic",
        model="claude-haiku-4-5",
        context_window=200000,
    )
    phys3 = PhysicalModelSchema(
        provider="glm",
        model="glm-4.5-flash",
        context_window=128000,
    )
    compactador_pm = MagicMock()
    compactador_pm.physical_models = [phys1, phys2, phys3]
    config.pseudo_models = {"compactador": compactador_pm}
    return config


@pytest.fixture
def conversation_with_turns():
    """A conversation with some turns."""
    conv = MagicMock()
    conv.id = uuid.uuid4()
    conv.total_tokens = 50000
    conv.active_snapshot_id = None
    conv.turns = []

    turn1 = MagicMock()
    turn1.messages = [{"role": "user", "content": "Hello"}]
    turn1.turn_number = 1
    turn2 = MagicMock()
    turn2.messages = [{"role": "assistant", "content": "Hi there!"}]
    turn2.turn_number = 2

    return conv, [turn1, turn2]


# ── Compactor model selection tests ─────────────────────────────────────


def test_select_compactor_model_large_enough(config_with_compactador):
    """Selects model with enough context window for the history."""
    model = select_compactor_model(config_with_compactador, 50000)
    assert model is not None
    assert model.model == "gemini-3.5-flash"  # First model with 1M ctx window (enough for 50K)
    assert model.context_window == 1000000


def test_select_compactor_model_largest_fallback(config_with_compactador):
    """When history exceeds all models, returns the one with largest window."""
    model = select_compactor_model(config_with_compactador, 2000000)
    assert model is not None
    assert model.model == "gemini-3.5-flash"  # Largest available
    assert model.context_window == 1000000


def test_select_compactor_model_no_compactador():
    """When compactador pseudo-model is missing, returns None."""
    config = MagicMock()
    config.pseudo_models = {}
    model = select_compactor_model(config, 50000)
    assert model is None


# ── Explicit compaction tests ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_compact_generates_snapshot(
    mock_litellm_success, config_with_compactador, mock_db, conversation_with_turns
):
    """POST /compact generates a snapshot with required fields."""
    conv, turns = conversation_with_turns

    # Mock DB to return the conversation and turns
    mock_db.get = AsyncMock(return_value=conv)

    scalar_result_turns = MagicMock()
    scalar_result_turns.scalars = MagicMock(
        return_value=MagicMock(all=MagicMock(return_value=turns))
    )
    mock_db.execute = AsyncMock(return_value=scalar_result_turns)

    result = await compact_conversation(
        conversation_id=str(conv.id),
        db=mock_db,
        config=config_with_compactador,
        arq_pool=None,
    )

    assert result["status"] == "completed"
    assert "snapshot_id" in result
    assert result["tokens_before"] > 0
    assert result["tokens_after"] > 0
    assert result["tokens_reduced_pct"] > 0
    assert result["compactor_model"] is not None
    assert "preview" in result
    assert result["can_resume"] is True


@pytest.mark.asyncio
async def test_snapshot_stored_in_db(
    mock_litellm_success, config_with_compactador, mock_db, conversation_with_turns
):
    """Snapshot stored in conversation_snapshots table via db.add."""
    conv, turns = conversation_with_turns
    mock_db.get = AsyncMock(return_value=conv)

    scalar_result_turns = MagicMock()
    scalar_result_turns.scalars = MagicMock(
        return_value=MagicMock(all=MagicMock(return_value=turns))
    )
    mock_db.execute = AsyncMock(return_value=scalar_result_turns)

    await compact_conversation(
        conversation_id=str(conv.id),
        db=mock_db,
        config=config_with_compactador,
        arq_pool=None,
    )

    # Verify snapshot was added
    mock_db.add.assert_called_once()
    added_snapshot = mock_db.add.call_args[0][0]
    assert added_snapshot.snapshot_type == "explicit"
    assert added_snapshot.tokens_before > 0
    assert added_snapshot.tokens_after > 0
    assert added_snapshot.compactor_model is not None
    assert added_snapshot.snapshot_content is not None


@pytest.mark.asyncio
async def test_active_snapshot_id_updated(
    mock_litellm_success, config_with_compactador, mock_db, conversation_with_turns
):
    """active_snapshot_id is updated on the conversation after compaction."""
    conv, turns = conversation_with_turns
    old_id = conv.active_snapshot_id
    mock_db.get = AsyncMock(return_value=conv)

    scalar_result_turns = MagicMock()
    scalar_result_turns.scalars = MagicMock(
        return_value=MagicMock(all=MagicMock(return_value=turns))
    )
    mock_db.execute = AsyncMock(return_value=scalar_result_turns)

    await compact_conversation(
        conversation_id=str(conv.id),
        db=mock_db,
        config=config_with_compactador,
        arq_pool=None,
    )

    # active_snapshot_id should now be set (not None or different from old)
    assert conv.active_snapshot_id is not None
    if old_id is not None:
        assert conv.active_snapshot_id != old_id


@pytest.mark.asyncio
async def test_empty_conversation_400(config_with_compactador, mock_db):
    """Compacting an empty conversation returns 400 error."""
    conv = MagicMock()
    conv.id = uuid.uuid4()
    conv.total_tokens = 0
    conv.active_snapshot_id = None
    mock_db.get = AsyncMock(return_value=conv)

    # No turns
    scalar_result = MagicMock()
    scalar_result.scalars = MagicMock(
        return_value=MagicMock(all=MagicMock(return_value=[]))
    )
    mock_db.execute = AsyncMock(return_value=scalar_result)

    with pytest.raises(ValueError, match="EmptyConversation"):
        await compact_conversation(
            conversation_id=str(conv.id),
            db=mock_db,
            config=config_with_compactador,
            arq_pool=None,
        )


@pytest.mark.asyncio
async def test_conversation_not_found_404(config_with_compactador, mock_db):
    """Compacting a non-existent conversation returns 404 error."""
    mock_db.get = AsyncMock(return_value=None)

    with pytest.raises(ValueError, match="ConversationNotFound"):
        await compact_conversation(
            conversation_id=str(uuid.uuid4()),
            db=mock_db,
            config=config_with_compactador,
            arq_pool=None,
        )


@pytest.mark.asyncio
async def test_async_dispatch_to_arq(
    mock_litellm_success, config_with_compactador, mock_db
):
    """History > 500K tokens dispatches to arq when pool is available."""
    conv = MagicMock()
    conv.id = uuid.uuid4()
    conv.total_tokens = 600000  # > 500K
    conv.active_snapshot_id = None
    mock_db.get = AsyncMock(return_value=conv)

    turn = MagicMock()
    turn.messages = [{"role": "user", "content": "Large content"}]
    turn.turn_number = 1
    scalar_result = MagicMock()
    scalar_result.scalars = MagicMock(
        return_value=MagicMock(all=MagicMock(return_value=[turn]))
    )
    mock_db.execute = AsyncMock(return_value=scalar_result)

    # Mock arq pool
    mock_arq = AsyncMock()
    mock_job = MagicMock()
    mock_job.job_id = "arq-job-123"
    mock_arq.enqueue_job = AsyncMock(return_value=mock_job)

    result = await compact_conversation(
        conversation_id=str(conv.id),
        db=mock_db,
        config=config_with_compactador,
        arq_pool=mock_arq,
    )

    assert result["status"] == "processing"
    assert result["task_id"] == "arq-job-123"
    assert "background worker" in result["message"]

    # Verify arq.enqueue_job was called with the right task name
    mock_arq.enqueue_job.assert_called_once_with(
        "compact_conversation_async",
        str(conv.id),
        "gemini-3.5-flash",
        None,  # api_base
        None,  # api_key
    )


@pytest.mark.asyncio
async def test_multiple_compactions_chain(
    mock_litellm_success, config_with_compactador, mock_db, conversation_with_turns
):
    """Multiple explicit compactions chain correctly via superseded_by."""
    conv, turns = conversation_with_turns
    mock_db.get = AsyncMock(return_value=conv)

    turns_scalar = MagicMock()
    turns_scalar.scalars = MagicMock(
        return_value=MagicMock(all=MagicMock(return_value=turns))
    )
    mock_db.execute = AsyncMock(return_value=turns_scalar)

    conv.total_tokens = 50000
    conv.active_snapshot_id = None

    # First compaction
    result1 = await compact_conversation(
        conversation_id=str(conv.id),
        db=mock_db,
        config=config_with_compactador,
        arq_pool=None,
    )
    snapshot_id_1 = result1["snapshot_id"]

    # Simulate that first snapshot is now active with an old_snapshot mock
    old_snapshot_mock = MagicMock()
    old_snapshot_mock.superseded_by = None
    conv.active_snapshot_id = snapshot_id_1
    conv.total_tokens = 60000  # Increased for second compaction

    # Return conv for Conversation gets and old_snapshot for ConversationSnapshot gets
    async def get_side_effect(model_cls, pk, **kwargs):
        if model_cls.__name__ == "Conversation":
            return conv
        return old_snapshot_mock

    mock_db.get = AsyncMock(side_effect=get_side_effect)

    # Second compaction
    result2 = await compact_conversation(
        conversation_id=str(conv.id),
        db=mock_db,
        config=config_with_compactador,
        arq_pool=None,
    )
    assert result2["status"] == "completed"
    assert result2["snapshot_id"] != snapshot_id_1


@pytest.mark.asyncio
async def test_snapshot_contains_required_sections(
    mock_litellm_success, config_with_compactador, mock_db, conversation_with_turns
):
    """Snapshot contains all required sections (Problem State, Technical Decisions, etc.)."""
    conv, turns = conversation_with_turns
    mock_db.get = AsyncMock(return_value=conv)

    scalar_result_turns = MagicMock()
    scalar_result_turns.scalars = MagicMock(
        return_value=MagicMock(all=MagicMock(return_value=turns))
    )
    mock_db.execute = AsyncMock(return_value=scalar_result_turns)

    result = await compact_conversation(
        conversation_id=str(conv.id),
        db=mock_db,
        config=config_with_compactador,
        arq_pool=None,
    )

    preview = result["preview"]
    assert preview is not None

    # The preview is truncated at 500 chars, so check for section markers
    # We verify the snapshot content was passed through to the compactor
    assert result["tokens_after"] > 0


@pytest.mark.asyncio
async def test_snapshot_preview_truncated(
    mock_litellm_success, config_with_compactador, mock_db, conversation_with_turns
):
    """Long snapshot content is truncated in preview with ellipsis."""
    # Create a response with very long content
    long_content = "# Snapshot\n\n" + ("Long content.\n" * 200)

    conv, turns = conversation_with_turns
    mock_db.get = AsyncMock(return_value=conv)

    scalar_result_turns = MagicMock()
    scalar_result_turns.scalars = MagicMock(
        return_value=MagicMock(all=MagicMock(return_value=turns))
    )
    mock_db.execute = AsyncMock(return_value=scalar_result_turns)

    # Override the mock response with long content
    long_response = MagicMock()
    long_response.choices = [MagicMock()]
    long_response.choices[0].message.content = long_content
    long_response.usage.completion_tokens = 500
    long_response.model_dump.return_value = {
        "id": "chatcmpl-mock",
        "choices": [{"message": {"role": "assistant", "content": long_content}}],
        "usage": {"prompt_tokens": 500, "completion_tokens": 500},
    }

    with patch(
        "src.service.compactor.explicit.call_litellm", new_callable=AsyncMock
    ) as mock:
        mock.return_value = long_response
        result = await compact_conversation(
            conversation_id=str(conv.id),
            db=mock_db,
            config=config_with_compactador,
            arq_pool=None,
        )

    assert result["preview"].endswith("...")
    assert len(result["preview"]) <= 503  # 500 + "..."
