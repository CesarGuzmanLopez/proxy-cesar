"""Shared test fixtures.

python.md: pytest-asyncio, fakeredis for Valkey mocking.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import fakeredis
import pytest
from httpx import ASGITransport, AsyncClient

from src.config.pseudo_models import ProxyConfigSchema, load_config

# Path to the real pseudo_models.yaml
CONFIG_PATH = Path(__file__).resolve().parent.parent / "pseudo_models.yaml"


@pytest.fixture
def valid_config() -> ProxyConfigSchema:
    """Load the production pseudo_models.yaml for tests."""
    return load_config(CONFIG_PATH)


@pytest.fixture
def mock_valkey():
    """Create a fake Valkey client for testing."""
    client = fakeredis.FakeAsyncValkey(decode_responses=True)
    return client


@pytest.fixture
def mock_litellm():
    """Patch litellm.acompletion with a mock returning a valid response."""
    mock_response = MagicMock()
    mock_response.choices = []
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 50
    mock_response.usage.completion_tokens = 100
    mock_response.model_dump.return_value = {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "choices": [{"message": {"role": "assistant", "content": "Hello!"}}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 100},
    }

    with patch(
        "src.adapters.litellm.client.litellm.acompletion", new_callable=AsyncMock
    ) as mock:
        mock.return_value = mock_response
        yield mock


class _ConversationStore:
    """In-memory store for conversations during e2e tests.

    Allows the mock DB to return previously created conversations
    on subsequent ``get()`` calls, making multi-turn tests work.
    """

    def __init__(self):
        self._store: dict = {}

    def put(self, conv_uuid, conversation):
        self._store[conv_uuid] = conversation

    def get(self, conv_uuid):
        return self._store.get(conv_uuid)


@pytest.fixture
async def async_client(mock_valkey):
    """Async test client with mocked dependencies."""
    from src.main import app
    from src.adapters.cache.valkey_affinity import ValkeyAffinityAdapter
    from src.adapters.db.models import Conversation

    # Override app state with test config
    config = load_config(CONFIG_PATH)
    app.state.config = config
    app.state.valkey = mock_valkey
    app.state.affinity = ValkeyAffinityAdapter(mock_valkey)

    # In-memory conversation store for multi-turn tests
    conv_store = _ConversationStore()

    # Mock DB session factory
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock()

    async def _mock_get(model_cls, ident, **kwargs):
        if model_cls is Conversation:
            return conv_store.get(ident)
        return None

    mock_session.get = AsyncMock(side_effect=_mock_get)

    def _mock_add(obj):
        if isinstance(obj, Conversation):
            conv_store.put(obj.id, obj)

    mock_session.add = MagicMock(side_effect=_mock_add)
    mock_session.flush = AsyncMock()
    mock_session.commit = AsyncMock()
    mock_session.rollback = AsyncMock()
    mock_session.close = AsyncMock()

    # Mock execute for SELECT queries (used by audit_log, compatible-models, etc.)
    mock_result = AsyncMock()
    mock_result.scalars = MagicMock(return_value=AsyncMock())
    mock_result.scalars.return_value.all = MagicMock(return_value=[])
    mock_result.scalar = MagicMock(return_value=0)
    mock_session.execute = AsyncMock(return_value=mock_result)

    app.state.db_session_factory = MagicMock(return_value=mock_session)

    # Compaction Orchestrator for FASE 3
    from src.service.compactor.explicit import CompactionOrchestrator

    app.state.compaction_orchestrator = CompactionOrchestrator()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
