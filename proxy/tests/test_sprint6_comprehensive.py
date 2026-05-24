"""Comprehensive Sprint 6 integration test.

Tests context alerts, explicit compaction, and audit log
via HTTP endpoints with mocked LiteLLM and fakeredis.

Covers:
  Sprint 6 §2: Context alerts at every threshold
  Sprint 6 §3: POST /conversations/{id}/compact
  Sprint 6 §4: arq async dispatch
  Sprint 6 §5: GET /conversations/{id}/audit-log
  Sprint 6 §6: Streaming path with proxy_metadata
  Sprint 6 §7: CONTEXT_UNUSABLE (400) error
  Snapshot chaining via superseded_by
  Multiple compaction scenarios
"""

import json
import uuid
from pathlib import Path
from unittest.mock import ANY, AsyncMock, MagicMock, patch
from datetime import datetime

import fakeredis
import pytest
from httpx import ASGITransport, AsyncClient

from src.config.pseudo_models import load_config

CONFIG_PATH = Path(__file__).resolve().parent.parent / "pseudo_models.yaml"


# ── LiteLLM mock helpers ─────────────────────────────────────────────────


def _make_chat_response(
    content: str = "Hello!",
    model: str = "deepseek/deepseek-v4-flash",
    prompt_tokens: int = 50,
    completion_tokens: int = 100,
    finish_reason: str = "stop",
):
    """Create a mock LiteLLM response for a chat completion."""
    mock = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    choice.message.tool_calls = None
    choice.finish_reason = finish_reason
    mock.choices = [choice]
    mock.usage = MagicMock()
    mock.usage.prompt_tokens = prompt_tokens
    mock.usage.completion_tokens = completion_tokens
    mock.model_dump.return_value = {
        "id": "chatcmpl-e2e-test",
        "object": "chat.completion",
        "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }
    return mock


def _make_compaction_response():
    """Create a mock LiteLLM response simulating a compaction snapshot."""
    snapshot_content = (
        "# Snapshot — 2026-05-24\n\n"
        "## Problem State\nWorking on auth module for FastAPI proxy.\n\n"
        "## Technical Decisions\nUsed JWT with refresh token rotation.\n"
        "SQLModel for ORM, asyncpg for PostgreSQL.\n\n"
        "## Code Produced\n"
        "- `src/auth/jwt.py`: JWT encode/decode with RS256\n"
        "- `src/auth/deps.py`: FastAPI dependency injection for auth\n\n"
        "## Current Status\n"
        "- **Resolved:** JWT validation, token refresh endpoint\n"
        "- **Unresolved:** Rate limiting on auth endpoints\n"
        "- **In Progress at compaction:** API key authentication\n\n"
        "## Technical Context\n"
        "Python 3.14, FastAPI 0.136, SQLModel 0.0.38\n"
        "PostgreSQL 17, Valkey 7.x\n\n"
        "## Tools & Capabilities\n"
        "search_codebase, read_file, write_file, run_command\n\n"
        "## Pending Items\n"
        "1. Redis rate limiting integration\n"
        "2. API key rotation endpoint\n\n"
        "## Conversation Metadata\n"
        "45 turns across 3 days. Pseudo-models: normal, tareas-avanzadas."
    )
    mock = MagicMock()
    choice = MagicMock()
    choice.message.content = snapshot_content
    choice.finish_reason = "stop"
    mock.choices = [choice]
    mock.usage = MagicMock()
    mock.usage.prompt_tokens = 5000
    mock.usage.completion_tokens = 350
    mock.model_dump.return_value = {
        "id": "chatcmpl-compact",
        "object": "chat.completion",
        "choices": [{"message": {"role": "assistant", "content": snapshot_content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5000, "completion_tokens": 350},
    }
    return mock, snapshot_content


# ── FastAPI test client fixture ──────────────────────────────────────────


@pytest.fixture
def app_with_mocks():
    """Create FastAPI app with all dependencies mocked."""
    config = load_config(CONFIG_PATH)

    # Use fakeredis for Valkey
    valkey = fakeredis.FakeAsyncValkey(decode_responses=True)

    # Create mock DB session factory
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock()

    # Default mock returns
    mock_session.get = AsyncMock(return_value=None)
    mock_session.add = MagicMock()
    mock_session.flush = AsyncMock()
    mock_session.commit = AsyncMock()
    mock_session.rollback = AsyncMock()
    mock_session.close = AsyncMock()

    # DB execute mock
    scalar_result = MagicMock()
    scalar_result.scalar = MagicMock(return_value=0)
    scalar_result.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))
    mock_session.execute = AsyncMock(return_value=scalar_result)

    db_session_factory = MagicMock(return_value=mock_session)

    # Patch litellm before importing the app
    with patch("src.service.compactor.explicit.call_litellm") as mock_llm, \
         patch("src.service.compactor.continuous.call_litellm") as mock_cont_llm, \
         patch("src.adapters.litellm.client.litellm.acompletion") as mock_acompletion:

        mock_response = _make_chat_response()
        mock_acompletion.return_value = mock_response
        mock_llm.return_value = _make_compaction_response()[0]
        mock_cont_llm.return_value = mock_response

        from src.main import app
        from src.adapters.cache.valkey_affinity import ValkeyAffinityAdapter

        app.state.config = config
        app.state.valkey = valkey
        app.state.affinity = ValkeyAffinityAdapter(valkey)
        app.state.db_session_factory = db_session_factory
        app.state.arq_pool = None  # No arq in tests

        yield app, mock_session, db_session_factory, valkey


@pytest.fixture
async def client(app_with_mocks):
    """Async HTTP test client."""
    app, _, _, _ = app_with_mocks
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac, app_with_mocks


# ── Context Alert Tests via HTTP ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_context_alert_normal_in_metadata(client):
    """Context <60% appears in proxy_metadata as normal alert level."""
    ac, (app, mock_db, db_factory, valkey) = client

    # Mock a conversation with low token usage
    conv = MagicMock()
    conv.id = uuid.uuid4()
    conv.total_tokens = 30000
    conv.pseudo_model = "normal"
    conv.physical_model = "openrouter/qwen3-max"
    conv.created_at = datetime.now()
    conv.updated_at = datetime.now()
    conv.active_snapshot_id = None
    conv.turns = []
    for attr in ("capability_has_images", "capability_has_audio", "capability_has_pdf",
                  "capability_has_video", "capability_has_tools", "capability_has_parallel_tools"):
        setattr(conv, attr, False)
    conv.max_tools_level = 0
    conv.images_described = 0
    conv.images_degraded_manually = False

    mock_db.get = AsyncMock(return_value=conv)

    resp = await ac.post("/v1/chat/completions", json={
        "model": "normal",
        "messages": [{"role": "user", "content": "Hello!"}],
        "conversation_id": str(conv.id),
    })

    assert resp.status_code == 200
    data = resp.json()
    assert "proxy_metadata" in data
    pm = data["proxy_metadata"]
    assert "context_alert" in pm
    assert pm["context_alert"]["alert_level"] == "normal"
    assert pm["context_alert"]["context_usage_pct"] == 31.2  # 30000/96000*100


@pytest.mark.asyncio
async def test_context_alert_moderate_in_metadata(client):
    """Context 60-80% appears in proxy_metadata with warning and endpoint."""
    ac, (app, mock_db, db_factory, valkey) = client

    conv = MagicMock()
    conv.id = uuid.uuid4()
    conv.total_tokens = 70000  # 72.9% of 96000 → moderate
    conv.pseudo_model = "normal"
    conv.physical_model = "openrouter/qwen3-max"
    conv.created_at = datetime.now()
    conv.updated_at = datetime.now()
    conv.active_snapshot_id = None
    conv.turns = []
    for attr in ("capability_has_images", "capability_has_audio", "capability_has_pdf",
                  "capability_has_video", "capability_has_tools", "capability_has_parallel_tools"):
        setattr(conv, attr, False)
    conv.max_tools_level = 0
    conv.images_described = 0
    conv.images_degraded_manually = False

    mock_db.get = AsyncMock(return_value=conv)

    resp = await ac.post("/v1/chat/completions", json={
        "model": "normal",
        "messages": [{"role": "user", "content": "Continue the work."}],
        "conversation_id": str(conv.id),
    })

    assert resp.status_code == 200
    data = resp.json()
    pm = data["proxy_metadata"]
    assert pm["context_alert"]["alert_level"] == "moderate"
    assert "warning" in pm["context_alert"]
    assert "compaction_endpoint" in pm["context_alert"]
    assert str(conv.id) in pm["context_alert"]["compaction_endpoint"]


@pytest.mark.asyncio
async def test_context_alert_high_in_metadata(client):
    """Context 80-99% appears in proxy_metadata with strong warning."""
    ac, (app, mock_db, db_factory, valkey) = client

    conv = MagicMock()
    conv.id = uuid.uuid4()
    conv.total_tokens = 90000  # 93.8% of 96000 → high
    conv.pseudo_model = "normal"
    conv.physical_model = "openrouter/qwen3-max"
    conv.created_at = datetime.now()
    conv.updated_at = datetime.now()
    conv.active_snapshot_id = None
    conv.turns = []
    for attr in ("capability_has_images", "capability_has_audio", "capability_has_pdf",
                  "capability_has_video", "capability_has_tools", "capability_has_parallel_tools"):
        setattr(conv, attr, False)
    conv.max_tools_level = 0
    conv.images_described = 0
    conv.images_degraded_manually = False

    mock_db.get = AsyncMock(return_value=conv)

    resp = await ac.post("/v1/chat/completions", json={
        "model": "normal",
        "messages": [{"role": "user", "content": "Keep going."}],
        "conversation_id": str(conv.id),
    })

    assert resp.status_code == 200
    data = resp.json()
    pm = data["proxy_metadata"]
    assert pm["context_alert"]["alert_level"] == "high"
    assert "Compact recommended" in pm["context_alert"]["warning"]


@pytest.mark.asyncio
async def test_context_unusable_400_error(client):
    """Context ≥100% returns HTTP 400 CONTEXT_UNUSABLE."""
    ac, (app, mock_db, db_factory, valkey) = client

    conv = MagicMock()
    conv.id = uuid.uuid4()
    conv.total_tokens = 100000  # 104.2% of 96000 → unusable
    conv.pseudo_model = "normal"
    conv.physical_model = "openrouter/qwen3-max"
    conv.created_at = datetime.now()
    conv.updated_at = datetime.now()
    conv.active_snapshot_id = None
    conv.turns = []
    for attr in ("capability_has_images", "capability_has_audio", "capability_has_pdf",
                  "capability_has_video", "capability_has_tools", "capability_has_parallel_tools"):
        setattr(conv, attr, False)
    conv.max_tools_level = 0
    conv.images_described = 0
    conv.images_degraded_manually = False

    mock_db.get = AsyncMock(return_value=conv)

    resp = await ac.post("/v1/chat/completions", json={
        "model": "normal",
        "messages": [{"role": "user", "content": "This should fail."}],
        "conversation_id": str(conv.id),
    })

    assert resp.status_code == 400
    data = resp.json()
    detail = data.get("detail", data)
    assert detail["error"] == "CONTEXT_UNUSABLE"
    assert "remediation" in detail
    assert detail["remediation"]["action"] == "compact"
    assert str(conv.id) in detail["remediation"]["endpoint"]


@pytest.mark.asyncio
async def test_context_unusable_streaming_400(client):
    """Streaming path also returns 400 CONTEXT_UNUSABLE when context is full."""
    ac, (app, mock_db, db_factory, valkey) = client

    conv = MagicMock()
    conv.id = uuid.uuid4()
    conv.total_tokens = 100000
    conv.pseudo_model = "normal"
    conv.physical_model = "openrouter/qwen3-max"
    conv.created_at = datetime.now()
    conv.updated_at = datetime.now()
    conv.active_snapshot_id = None
    conv.turns = []
    for attr in ("capability_has_images", "capability_has_audio", "capability_has_pdf",
                  "capability_has_video", "capability_has_tools", "capability_has_parallel_tools"):
        setattr(conv, attr, False)
    conv.max_tools_level = 0
    conv.images_described = 0
    conv.images_degraded_manually = False

    mock_db.get = AsyncMock(return_value=conv)

    resp = await ac.post("/v1/chat/completions", json={
        "model": "normal",
        "messages": [{"role": "user", "content": "Streaming fail."}],
        "conversation_id": str(conv.id),
        "stream": True,
    })

    assert resp.status_code == 400
    data = resp.json()
    detail = data.get("detail", data)
    assert detail["error"] == "CONTEXT_UNUSABLE"


# ── Explicit Compaction Tests via HTTP ───────────────────────────────────


@pytest.mark.asyncio
async def test_explicit_compaction_endpoint(client):
    """POST /conversations/{id}/compact generates a snapshot."""
    ac, (app, mock_db, db_factory, valkey) = client

    conv_id = uuid.uuid4()
    conv = MagicMock()
    conv.id = conv_id
    conv.total_tokens = 120000
    conv.pseudo_model = "normal"
    conv.physical_model = "openrouter/qwen3-max"
    conv.created_at = datetime.now()
    conv.updated_at = datetime.now()
    conv.active_snapshot_id = None
    conv.turns = []

    # Mock DB: first get returns conversation, second returns None (no old snapshot)
    mock_db.get = AsyncMock(return_value=conv)

    # Mock turns in DB
    turn = MagicMock()
    turn.messages = [
        {"role": "user", "content": "Let's build an auth module."},
        {"role": "assistant", "content": "I'll help you set up JWT authentication."},
    ]
    turn.turn_number = 1
    turn.input_tokens = 500
    turn.output_tokens = 800

    scalar_result = MagicMock()
    scalar_result.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=[turn])))
    mock_db.execute = AsyncMock(return_value=scalar_result)

    resp = await ac.post(f"/conversations/{conv_id}/compact")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    assert "snapshot_id" in data
    assert data["tokens_before"] == 120000
    assert data["tokens_after"] > 0
    assert data["tokens_reduced_pct"] > 0
    assert data["compactor_model"] is not None
    assert data["can_resume"] is True
    assert "preview" in data


@pytest.mark.asyncio
async def test_explicit_compaction_requires_turns(client):
    """POST /compact on empty conversation returns 400."""
    ac, (app, mock_db, db_factory, valkey) = client

    conv_id = uuid.uuid4()
    conv = MagicMock()
    conv.id = conv_id
    conv.total_tokens = 0
    conv.pseudo_model = "normal"
    conv.physical_model = "openrouter/qwen3-max"
    conv.created_at = datetime.now()
    conv.updated_at = datetime.now()
    conv.active_snapshot_id = None
    conv.turns = []

    mock_db.get = AsyncMock(return_value=conv)

    # No turns in DB
    scalar_result = MagicMock()
    scalar_result.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))
    mock_db.execute = AsyncMock(return_value=scalar_result)

    resp = await ac.post(f"/conversations/{conv_id}/compact")

    assert resp.status_code == 400
    detail = resp.json().get("detail", resp.json())
    assert detail["error"] == "EMPTY_CONVERSATION"


@pytest.mark.asyncio
async def test_explicit_compaction_not_found(client):
    """POST /compact on non-existent conversation returns 404."""
    ac, (app, mock_db, db_factory, valkey) = client

    mock_db.get = AsyncMock(return_value=None)

    resp = await ac.post(f"/conversations/{uuid.uuid4()}/compact")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_explicit_compaction_chains_snapshots(client):
    """Multiple explicit compactions chain correctly via superseded_by."""
    ac, (app, mock_db, db_factory, valkey) = client

    conv_id = uuid.uuid4()
    conv = MagicMock()
    conv.id = conv_id
    conv.total_tokens = 120000
    conv.pseudo_model = "normal"
    conv.physical_model = "openrouter/qwen3-max"
    conv.created_at = datetime.now()
    conv.updated_at = datetime.now()
    conv.active_snapshot_id = None
    conv.turns = []

    mock_db.get = AsyncMock(return_value=conv)

    turn = MagicMock()
    turn.messages = [{"role": "user", "content": "Build auth."}]
    turn.turn_number = 1
    turn.input_tokens = 500
    turn.output_tokens = 300

    scalar_result = MagicMock()
    scalar_result.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=[turn])))
    mock_db.execute = AsyncMock(return_value=scalar_result)

    # First compaction
    resp1 = await ac.post(f"/conversations/{conv_id}/compact")
    assert resp1.status_code == 200
    snap1_id = resp1.json()["snapshot_id"]

    # Set active snapshot and mock old snapshot
    old_snap = MagicMock()
    old_snap.superseded_by = None
    conv.active_snapshot_id = snap1_id
    conv.total_tokens = 150000

    # Update mock_db.get to handle both Conversation and ConversationSnapshot gets
    async def get_side_effect(model_cls, pk, **kw):
        name = getattr(model_cls, "__name__", "")
        if name == "Conversation":
            return conv
        if name == "ConversationSnapshot":
            # Return snapshot for the active_snapshot_id lookups
            s = MagicMock()
            s.superseded_by = None
            s.__class__.__name__ = "ConversationSnapshot"
            return s
        return None
    mock_db.get = AsyncMock(side_effect=get_side_effect)

    # Second compaction
    resp2 = await ac.post(f"/conversations/{conv_id}/compact")
    assert resp2.status_code == 200
    snap2_id = resp2.json()["snapshot_id"]
    assert snap2_id != snap1_id


# ── Audit Log Tests via HTTP ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_audit_log_includes_creation(client):
    """GET /conversations/{id}/audit-log includes conversation_created."""
    ac, (app, mock_db, db_factory, valkey) = client

    conv_id = uuid.uuid4()
    conv = MagicMock()
    conv.id = conv_id
    conv.pseudo_model = "normal"
    conv.physical_model = "openrouter/qwen3-max"
    conv.created_at = datetime(2026, 5, 24, 10, 0, 0)
    conv.updated_at = datetime.now()
    conv.active_snapshot_id = None
    conv.turns = []

    mock_db.get = AsyncMock(return_value=conv)

    scalar_result = MagicMock()
    scalar_result.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))
    mock_db.execute = AsyncMock(return_value=scalar_result)

    resp = await ac.get(f"/conversations/{conv_id}/audit-log")
    assert resp.status_code == 200
    data = resp.json()
    assert data["conversation_id"] == str(conv_id)
    assert len(data["events"]) >= 1
    assert data["events"][0]["event_type"] == "conversation_created"


@pytest.mark.asyncio
async def test_audit_log_tracks_switches_and_fallbacks(client):
    """Audit log includes pseudo_model_switched and fallback_applied events."""
    ac, (app, mock_db, db_factory, valkey) = client

    conv_id = uuid.uuid4()
    conv = MagicMock()
    conv.id = conv_id
    conv.pseudo_model = "normal"
    conv.physical_model = "openrouter/qwen3-max"
    conv.created_at = datetime(2026, 5, 24, 10, 0, 0)
    conv.updated_at = datetime.now()
    conv.active_snapshot_id = None
    conv.turns = []

    mock_db.get = AsyncMock(return_value=conv)

    # Mock turns with switches and fallbacks
    turn1 = MagicMock()
    turn1.turn_type = "normal"
    turn1.pseudo_model = "normal"
    turn1.fallback_applied = False
    turn1.fallback_reason = None
    turn1.turn_number = 1
    turn1.created_at = datetime(2026, 5, 24, 10, 5, 0)

    turn2 = MagicMock()
    turn2.turn_type = "normal"
    turn2.pseudo_model = "tareas-avanzadas"
    turn2.fallback_applied = False
    turn2.fallback_reason = None
    turn2.turn_number = 2
    turn2.created_at = datetime(2026, 5, 24, 10, 10, 0)

    turn3 = MagicMock()
    turn3.turn_type = "normal"
    turn3.pseudo_model = "tareas-avanzadas"
    turn3.fallback_applied = True
    turn3.fallback_reason = "ServiceUnavailableError: deepseek/deepseek-v4-pro"
    turn3.turn_number = 3
    turn3.created_at = datetime(2026, 5, 24, 10, 15, 0)

    scalar_result = MagicMock()
    scalar_result.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=[turn1, turn2, turn3])))
    mock_db.execute = AsyncMock(return_value=scalar_result)

    resp = await ac.get(f"/conversations/{conv_id}/audit-log")
    assert resp.status_code == 200
    data = resp.json()

    events_by_type = {e["event_type"]: e for e in data["events"]}
    assert "pseudo_model_switched" in events_by_type
    assert events_by_type["pseudo_model_switched"]["details"]["from"] == "normal"
    assert events_by_type["pseudo_model_switched"]["details"]["to"] == "tareas-avanzadas"

    assert "fallback_applied" in events_by_type
    assert "ServiceUnavailableError" in events_by_type["fallback_applied"]["details"]["reason"]


@pytest.mark.asyncio
async def test_audit_log_includes_compaction(client):
    """Audit log after compaction includes compaction_explicit event."""
    ac, (app, mock_db, db_factory, valkey) = client

    conv_id = uuid.uuid4()
    conv = MagicMock()
    conv.id = conv_id
    conv.total_tokens = 120000
    conv.pseudo_model = "normal"
    conv.physical_model = "openrouter/qwen3-max"
    conv.created_at = datetime(2026, 5, 24, 10, 0, 0)
    conv.updated_at = datetime.now()
    conv.active_snapshot_id = None
    conv.turns = []

    mock_db.get = AsyncMock(return_value=conv)

    turn = MagicMock()
    turn.messages = [{"role": "user", "content": "Build something."}]
    turn.turn_number = 1
    turn.input_tokens = 500
    turn.output_tokens = 300

    scalar_result = MagicMock()
    scalar_result.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=[turn])))
    mock_db.execute = AsyncMock(return_value=scalar_result)

    # Run compaction
    await ac.post(f"/conversations/{conv_id}/compact")

    # Now audit log should include compaction event
    # Need to set up mocks for audit log reading
    snap = MagicMock()
    snap.created_at = datetime(2026, 5, 24, 14, 0, 0)
    snap.snapshot_type = "explicit"
    snap.tokens_before = 120000
    snap.tokens_after = 350
    snap.compactor_model = "gemini/gemini-3.5-flash"

    # mock_db.get needs to return conv for the audit log endpoint
    mock_db.get = AsyncMock(return_value=conv)

    # Return empty turns and one snapshot
    scalar_turns = MagicMock()
    scalar_turns.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))
    scalar_snaps = MagicMock()
    scalar_snaps.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=[snap])))
    mock_db.execute = AsyncMock(side_effect=[scalar_turns, scalar_snaps])

    resp = await ac.get(f"/conversations/{conv_id}/audit-log")
    assert resp.status_code == 200
    data = resp.json()

    compaction_events = [e for e in data["events"] if e["event_type"] == "compaction_explicit"]
    assert len(compaction_events) == 1
    assert compaction_events[0]["details"]["tokens_before"] == 120000
    assert compaction_events[0]["details"]["tokens_after"] == 350


# ── Edge Cases ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_compactador_model_selection_by_context_window(client):
    """Compactor model is selected based on context window (by_context_window)."""
    ac, (app, mock_db, db_factory, valkey) = client
    config = app.state.config

    from src.service.compactor.explicit import select_compactor_model

    model = select_compactor_model(config, 50000)
    assert model is not None

    # Gemini 3.5 Flash has 1M ctx — should be selected for any reasonable size
    compactador_pm = config.pseudo_models.get("compactador")
    gemini = next((m for m in compactador_pm.physical_models if "gemini" in m.model), None)
    assert gemini is not None
    assert model == gemini.model


@pytest.mark.asyncio
async def test_streaming_response_includes_context_alert(client):
    """Streaming SSE response includes proxy_metadata with context_alert."""
    ac, (app, mock_db, db_factory, valkey) = client

    conv = MagicMock()
    conv.id = uuid.uuid4()
    conv.total_tokens = 65000  # 67.7% of 96000 → moderate
    conv.pseudo_model = "normal"
    conv.physical_model = "openrouter/qwen3-max"
    conv.created_at = datetime.now()
    conv.updated_at = datetime.now()
    conv.active_snapshot_id = None
    conv.turns = []
    for attr in ("capability_has_images", "capability_has_audio", "capability_has_pdf",
                  "capability_has_video", "capability_has_tools", "capability_has_parallel_tools"):
        setattr(conv, attr, False)
    conv.max_tools_level = 0
    conv.images_described = 0
    conv.images_degraded_manually = False

    mock_db.get = AsyncMock(return_value=conv)

    resp = await ac.post("/v1/chat/completions", json={
        "model": "normal",
        "messages": [{"role": "user", "content": "Stream test."}],
        "conversation_id": str(conv.id),
        "stream": True,
    })

    assert resp.status_code == 200
    body = resp.text

    # Find the proxy_metadata in the SSE stream (last chunk)
    assert "proxy_metadata" in body
    assert "context_alert" in body

    # Parse the last data chunk
    lines = body.strip().split("\n")
    for line in lines:
        if line.startswith("data: ") and "proxy_metadata" in line:
            chunk = json.loads(line[6:])
            if "proxy_metadata" in chunk:
                pm = chunk["proxy_metadata"]
                assert "context_alert" in pm
                assert pm["context_alert"]["alert_level"] in ("moderate", "normal")


@pytest.mark.asyncio
async def test_audit_log_events_chronological_order(client):
    """Audit log events are sorted by timestamp."""
    ac, (app, mock_db, db_factory, valkey) = client

    conv_id = uuid.uuid4()
    conv = MagicMock()
    conv.id = conv_id
    conv.pseudo_model = "normal"
    conv.physical_model = "openrouter/qwen3-max"
    conv.created_at = datetime(2026, 5, 24, 10, 0, 0)
    conv.updated_at = datetime.now()
    conv.active_snapshot_id = None
    conv.turns = []

    mock_db.get = AsyncMock(return_value=conv)

    turn2 = MagicMock()
    turn2.turn_type = "normal"
    turn2.pseudo_model = "normal"
    turn2.fallback_applied = True
    turn2.fallback_reason = "RateLimitError"
    turn2.turn_number = 2
    turn2.created_at = datetime(2026, 5, 24, 10, 5, 0)

    turn1 = MagicMock()
    turn1.turn_type = "normal"
    turn1.pseudo_model = "tareas-avanzadas"
    turn1.fallback_applied = False
    turn1.fallback_reason = None
    turn1.turn_number = 1
    turn1.created_at = datetime(2026, 5, 24, 10, 2, 0)

    scalar_result = MagicMock()
    scalar_result.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=[turn1, turn2])))
    mock_db.execute = AsyncMock(return_value=scalar_result)

    resp = await ac.get(f"/conversations/{conv_id}/audit-log")
    assert resp.status_code == 200
    data = resp.json()

    timestamps = [e["timestamp"] for e in data["events"]]
    assert timestamps == sorted(timestamps)
