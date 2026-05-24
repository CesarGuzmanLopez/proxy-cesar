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

    with patch("src.adapters.litellm.client.litellm.acompletion",
               new_callable=AsyncMock) as mock:
        mock.return_value = mock_response
        yield mock


@pytest.fixture
async def async_client(mock_valkey):
    """Async test client with mocked dependencies."""
    from src.main import app
    from src.config.settings import settings
    from src.adapters.cache.valkey_affinity import ValkeyAffinityAdapter

    # Override app state with test config
    config = load_config(CONFIG_PATH)
    app.state.config = config
    app.state.valkey = mock_valkey
    app.state.affinity = ValkeyAffinityAdapter(mock_valkey)

    # Mock DB session factory
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock()
    mock_session.get = AsyncMock(return_value=None)
    mock_session.add = MagicMock()
    mock_session.flush = AsyncMock()
    mock_session.commit = AsyncMock()
    mock_session.rollback = AsyncMock()
    mock_session.close = AsyncMock()
    app.state.db_session_factory = MagicMock(return_value=mock_session)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
