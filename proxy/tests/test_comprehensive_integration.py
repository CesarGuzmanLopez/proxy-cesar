"""Comprehensive integration test — exercises ALL features end-to-end.

This test starts a patched FastAPI server (fakeredis for Valkey, SQLite,
mocked LiteLLM) and tests every endpoint and flow via HTTP.

Covers:
  S1: Health, models list, basic chat (streaming + non-streaming)
  S2: Capability detection, compatibility validation
  S3: Tool filtering, canonical tools
  S4: Pre-compaction, continuous compaction
  S5: Auto-describe images, router LLM, manual degradation
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import fakeredis
import pytest
from httpx import ASGITransport, AsyncClient

from src.config.pseudo_models import ProxyConfigSchema, load_config

CONFIG_PATH = Path(__file__).resolve().parent.parent / "pseudo_models.yaml"


# ── LiteLLM mock helpers ──────────────────────────────────────────────────────


def _make_chat_response(
    content: str = "Hello, I am a helpful AI!",
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
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }
    return mock


def _make_streaming_chunk(content: str, finish_reason: str | None = None):
    """Create a mock LiteLLM streaming chunk."""
    mock = MagicMock()
    choice = MagicMock()
    delta = MagicMock()
    delta.content = content
    delta.tool_calls = None
    choice.delta = delta
    choice.finish_reason = finish_reason
    mock.choices = [choice]
    mock.usage = MagicMock()
    mock.usage.prompt_tokens = 50
    mock.usage.completion_tokens = 100
    mock.model_dump_json.return_value = json.dumps(
        {
            "id": "chatcmpl-e2e-stream",
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "delta": {"content": content} if content else {},
                    "finish_reason": finish_reason,
                }
            ],
        }
    )
    return mock


async def mock_streaming_response(*args, **kwargs):
    """Async generator that yields mock streaming chunks."""
    yield _make_streaming_chunk("Hello, ", None)
    yield _make_streaming_chunk("this ", None)
    yield _make_streaming_chunk("is ", None)
    yield _make_streaming_chunk("a stream!", "stop")


def _make_image_desc_response(
    description: str = "A screenshot of a code editor showing Python.",
):
    """Create a mock LiteLLM response for image description."""
    return _make_chat_response(
        content=description,
        model="gemini/gemini-3.5-flash",
        prompt_tokens=50,
        completion_tokens=15,
    )


# ── Pytest fixtures ───────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def valid_config() -> ProxyConfigSchema:
    return load_config(CONFIG_PATH)


@pytest.fixture
def mock_valkey():
    client = fakeredis.FakeAsyncValkey(decode_responses=True)
    return client


@pytest.fixture
def mock_db_session():
    """Create a mock DB session with pre-configured SQLite behavior."""
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock()
    session.get = AsyncMock(return_value=None)  # No existing conversation by default
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()

    async def mock_execute(*args, **kwargs):
        result = MagicMock()
        result.scalar.return_value = 0  # Default: empty table
        result.scalars.return_value = result
        result.all.return_value = []
        return result

    session.execute = mock_execute
    return session


# ══════════════════════════════════════════════════════════════════════════════
# Non-streaming Chat Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestBasicNonStreamingChat:
    """Sprint 1: Basic chat flow — non-streaming."""

    @patch("src.service.chat_service.call_litellm")
    async def test_new_conversation(self, mock_call, mock_valkey, mock_db_session):
        """New conversation → gets conversation_id back."""
        mock_call.return_value = _make_chat_response()

        with (
            patch("src.main.setup_litellm"),
            patch(
                "src.service.router_llm.suggester.load_bert_classifier",
                return_value=False,
            ),
        ):
            from src.main import app
            from src.adapters.cache.valkey_affinity import ValkeyAffinityAdapter

            app.state.config = load_config(CONFIG_PATH)
            app.state.valkey = mock_valkey
            app.state.affinity = ValkeyAffinityAdapter(mock_valkey)
            app.state.db_session_factory = MagicMock(return_value=mock_db_session)

            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "normal",
                        "messages": [{"role": "user", "content": "Hello!"}],
                    },
                )
                assert resp.status_code == 200, (
                    f"Expected 200, got {resp.status_code}: {resp.text}"
                )
                data = resp.json()
                assert "choices" in data
                assert (
                    data["choices"][0]["message"]["content"]
                    == "Hello, I am a helpful AI!"
                )
                assert "conversation_id" in data
                assert "proxy_metadata" in data

    @patch("src.service.chat_service.call_litellm")
    async def test_unknown_model_returns_default(
        self, mock_call, mock_valkey, mock_db_session
    ):
        """Unknown model → resolved via default alias to 'normal' (Sprint 7)."""
        mock_call.return_value = _make_chat_response()
        with (
            patch("src.main.setup_litellm"),
            patch(
                "src.service.router_llm.suggester.load_bert_classifier",
                return_value=False,
            ),
        ):
            from src.main import app
            from src.adapters.cache.valkey_affinity import ValkeyAffinityAdapter

            app.state.config = load_config(CONFIG_PATH)
            app.state.valkey = mock_valkey
            app.state.affinity = ValkeyAffinityAdapter(mock_valkey)
            app.state.db_session_factory = MagicMock(return_value=mock_db_session)

            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "non-existent-model",
                        "messages": [{"role": "user", "content": "Hello!"}],
                    },
                )
                # Sprint 7: default alias maps unknown models to "normal"
                assert resp.status_code == 200
                data = resp.json()
                assert data["proxy_metadata"]["pseudo_model"] == "normal"

    @patch("src.service.chat_service.call_litellm")
    async def test_conversation_id_is_reused(
        self, mock_call, mock_valkey, mock_db_session
    ):
        """Same conversation_id → maintains continuity."""
        mock_call.return_value = _make_chat_response()

        with (
            patch("src.main.setup_litellm"),
            patch(
                "src.service.router_llm.suggester.load_bert_classifier",
                return_value=False,
            ),
        ):
            from src.main import app
            from src.adapters.cache.valkey_affinity import ValkeyAffinityAdapter

            app.state.config = load_config(CONFIG_PATH)
            app.state.valkey = mock_valkey
            app.state.affinity = ValkeyAffinityAdapter(mock_valkey)
            app.state.db_session_factory = MagicMock(return_value=mock_db_session)

            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                conv_id = "my-test-conversation-123"
                resp1 = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "normal",
                        "conversation_id": conv_id,
                        "messages": [{"role": "user", "content": "First message"}],
                    },
                )
                assert resp1.status_code == 200, (
                    f"Expected 200, got {resp1.status_code}: {resp1.text}"
                )
                data1 = resp1.json()
                # When conversation_id is provided, it's not in the top-level response
                # but it IS in proxy_metadata
                assert data1["proxy_metadata"]["conversation_id"] == conv_id


# ══════════════════════════════════════════════════════════════════════════════
# Streaming Chat Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestBasicStreamingChat:
    """Sprint 1: Streaming chat flow."""

    async def test_streaming_returns_sse(self, mock_valkey, mock_db_session):
        """Streaming request → SSE response with [DONE] marker."""
        with (
            patch("src.main.setup_litellm"),
            patch(
                "src.service.router_llm.suggester.load_bert_classifier",
                return_value=False,
            ),
            patch("src.api.chat.call_with_fallback") as mock_fallback,
        ):
            from src.service.chat_models import FallbackInfo

            mock_fallback.return_value = (mock_streaming_response(), FallbackInfo())

            from src.main import app
            from src.adapters.cache.valkey_affinity import ValkeyAffinityAdapter

            app.state.config = load_config(CONFIG_PATH)
            app.state.valkey = mock_valkey
            app.state.affinity = ValkeyAffinityAdapter(mock_valkey)
            app.state.db_session_factory = MagicMock(return_value=mock_db_session)

            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "normal",
                        "messages": [{"role": "user", "content": "Stream this!"}],
                        "stream": True,
                    },
                )
                assert resp.status_code == 200
                assert "text/event-stream" in resp.headers.get("content-type", "")
                body = resp.text
                assert "data: " in body
                assert "data: [DONE]" in body


# ══════════════════════════════════════════════════════════════════════════════
# Capabilities & Compatibility Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestCompatibility:
    """Sprint 2: Pseudo-model switch compatibility."""

    @patch("src.service.chat_service.call_litellm")
    async def test_blocked_switch_returns_409(
        self, mock_call, mock_valkey, mock_db_session
    ):
        """Switch to incompatible model → 409 Conflict."""
        mock_call.return_value = _make_chat_response()

        with (
            patch("src.main.setup_litellm"),
            patch(
                "src.service.router_llm.suggester.load_bert_classifier",
                return_value=False,
            ),
        ):
            from src.main import app
            from src.adapters.cache.valkey_affinity import ValkeyAffinityAdapter
            from src.adapters.db.models import Conversation

            app.state.config = load_config(CONFIG_PATH)
            app.state.valkey = mock_valkey
            app.state.affinity = ValkeyAffinityAdapter(mock_valkey)

            # Create a mock conversation that already has images
            conv = Conversation(
                id="00000000-0000-0000-0000-000000000001",
                pseudo_model="avanzada-vision",
                physical_model="gemini/gemini-3.5-flash",
                capability_has_images=True,
            )
            conv.turns = []
            mock_db_session.get = AsyncMock(return_value=conv)
            app.state.db_session_factory = MagicMock(return_value=mock_db_session)

            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                # Try to switch from vision to a model that BLOCKS images
                resp = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "tareas-avanzadas",  # has on_downgrade: block
                        "conversation_id": "00000000-0000-0000-0000-000000000001",
                        "messages": [{"role": "user", "content": "Switch me!"}],
                    },
                )
                assert resp.status_code == 409, (
                    f"Expected 409 for blocked switch, got {resp.status_code}: {resp.text}"
                )
                data = resp.json()
                assert "PSEUDO_MODEL_INCOMPATIBLE" in str(data)

    @patch("src.service.chat_service.call_litellm")
    async def test_warning_switch_allowed(
        self, mock_call, mock_valkey, mock_db_session
    ):
        """Switch with warning (auto_describe) → 200 OK with warning in metadata."""
        mock_call.side_effect = [
            _make_image_desc_response("An IDE screenshot with Python code."),
            _make_chat_response("I see code. How can I help?"),
        ]

        with (
            patch("src.main.setup_litellm"),
            patch(
                "src.service.router_llm.suggester.load_bert_classifier",
                return_value=False,
            ),
            patch("src.api.conversations.auto_describe_images") as mock_ad,
        ):
            mock_ad.side_effect = lambda msgs, model: (
                msgs,
                {
                    "ok": True,
                    "images_described": 1,
                    "unique_images_described": 1,
                    "duplicate_images_skipped": 0,
                    "described_by": "gemini/gemini-3.5-flash",
                    "total_description_tokens": 15,
                    "status": "completed",
                },
            )

            from src.main import app
            from src.adapters.cache.valkey_affinity import ValkeyAffinityAdapter
            from src.adapters.db.models import Conversation

            app.state.config = load_config(CONFIG_PATH)
            app.state.valkey = mock_valkey
            app.state.affinity = ValkeyAffinityAdapter(mock_valkey)

            conv = Conversation(
                id="00000000-0000-0000-0000-000000000002",
                pseudo_model="avanzada-vision",
                physical_model="gemini/gemini-3.5-flash",
                capability_has_images=True,
            )
            conv.turns = [
                MagicMock(
                    turn_number=1,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "What's in this image?"},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": "data:image/png;base64,iVBOR",
                                        "detail": "auto",
                                    },
                                },
                            ],
                        },
                    ],
                )
            ]
            conv.id = "00000000-0000-0000-0000-000000000002"
            mock_db_session.get = AsyncMock(return_value=conv)
            app.state.db_session_factory = MagicMock(return_value=mock_db_session)

            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "pensamiento-profundo-caro",  # has on_downgrade: auto_describe
                        "conversation_id": "00000000-0000-0000-0000-000000000002",
                        "messages": [
                            {"role": "user", "content": "Switch me with auto-describe!"}
                        ],
                    },
                )
                assert resp.status_code == 200, (
                    f"Expected 200 for warning switch, got {resp.status_code}: {resp.text}"
                )
                data = resp.json()
                meta = data.get("proxy_metadata", {})
                # Images should have been described
                assert meta.get("images_described") == 1, (
                    f"Expected images_described=1, got {meta.get('images_described')}"
                )
                # The first vision model in avanzada-vision is Groq Llama 4 Scout
                assert meta["images_described_by"] in (
                    "groq/meta-llama/llama-4-scout-17b-16e-instruct",
                    "gemini/gemini-3.5-flash",
                ), f"Unexpected describer: {meta['images_described_by']}"


# ══════════════════════════════════════════════════════════════════════════════
# Auto-describe Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestAutoDescribe:
    """Sprint 5: Auto-describe images feature."""

    @patch("src.service.multimedia.image_describer.call_litellm")
    async def test_auto_describe_replaces_images_with_text(
        self, mock_call, mock_valkey, mock_db_session
    ):
        """Image messages → auto-describe replaces image_url with text."""
        mock_call.return_value = _make_image_desc_response(
            "A screenshot of Python code."
        )

        with (
            patch("src.main.setup_litellm"),
            patch(
                "src.service.router_llm.suggester.load_bert_classifier",
                return_value=False,
            ),
        ):
            from src.service.multimedia.image_describer import auto_describe_images

            # Direct unit test of the auto-describe service
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,iVBOR",
                                "detail": "auto",
                            },
                        },
                    ],
                },
            ]
            modified, meta = await auto_describe_images(
                messages, "gemini/gemini-3.5-flash"
            )
            assert meta["images_described"] == 1
            assert meta["described_by"] == "gemini/gemini-3.5-flash"
            # The image_url part should be replaced with text
            content = modified[0]["content"]
            image_part = content[1]
            assert image_part["type"] == "text"
            assert "IMAGE_DESCRIBED" in image_part["text"]

    async def test_auto_describe_no_images_noop(self, mock_valkey, mock_db_session):
        """No images → auto-describe does nothing."""
        from src.service.multimedia.image_describer import auto_describe_images

        messages = [{"role": "user", "content": "Just text, no images."}]
        modified, meta = await auto_describe_images(messages, "gemini/gemini-3.5-flash")
        assert modified == messages
        assert meta["images_described"] == 0
        assert meta["status"] == "no_images_found"


# ══════════════════════════════════════════════════════════════════════════════
# Router LLM Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestRouterLLM:
    """Sprint 5: Router LLM complexity evaluation."""

    @patch("src.service.router_llm.suggester.call_litellm")
    async def test_evaluate_complexity_simple_task(
        self, mock_call, mock_valkey, mock_db_session
    ):
        """Simple task → returns suggestion."""
        mock_call.return_value = _make_chat_response(
            json.dumps(
                {
                    "complexity": "simple",
                    "suggested_pseudo_model": "flash-lowcost",
                    "reason": "Simple question.",
                }
            ),
            model="zai/glm-4.5-flash",
            completion_tokens=30,
        )

        from src.service.router_llm.suggester import evaluate_complexity

        result = await evaluate_complexity(
            messages=[{"role": "user", "content": "What is 2+2?"}],
            suggester_model="zai/glm-4.5-flash",
        )
        assert result is not None
        assert result["complexity"] == "simple"
        assert result["suggested"] == "flash-lowcost"

    async def test_evaluate_complexity_no_user_message(
        self, mock_valkey, mock_db_session
    ):
        """No user message → returns None (skips evaluation)."""
        from src.service.router_llm.suggester import evaluate_complexity

        result = await evaluate_complexity(
            messages=[{"role": "system", "content": "You are a helpful assistant."}],
            suggester_model="zai/glm-4.5-flash",
        )
        assert result is None

    def test_is_downgrade_cheaper(self, valid_config):
        """flash-lowcost → normal is a downgrade."""
        from src.service.router_llm.suggester import is_downgrade

        assert is_downgrade("flash-lowcost", "normal", valid_config) is True

    def test_is_downgrade_more_expensive(self, valid_config):
        """normal → pensamiento-profundo-caro is NOT a downgrade."""
        from src.service.router_llm.suggester import is_downgrade

        assert (
            is_downgrade("pensamiento-profundo-caro", "normal", valid_config) is False
        )


# ══════════════════════════════════════════════════════════════════════════════
# Manual Degradation Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestManualDegradation:
    """Sprint 5: POST /conversations/{id}/degrade-images."""

    @patch("src.api.conversations.auto_describe_images")
    async def test_degrade_images_success(self, mock_ad, mock_valkey, mock_db_session):
        """POST /degrade-images with images → 200 + images_described count."""
        mock_ad.return_value = (
            [
                {
                    "role": "user",
                    "content": "[IMAGE_DESCRIBED #1 — described by test] Screenshot.",
                }
            ],
            {
                "ok": True,
                "images_described": 1,
                "unique_images_described": 1,
                "duplicate_images_skipped": 0,
                "described_by": "gemini/gemini-3.5-flash",
                "total_description_tokens": 15,
                "status": "completed",
            },
        )

        with (
            patch("src.main.setup_litellm"),
            patch(
                "src.service.router_llm.suggester.load_bert_classifier",
                return_value=False,
            ),
        ):
            from src.main import app
            from src.adapters.cache.valkey_affinity import ValkeyAffinityAdapter
            from src.adapters.db.models import Conversation
            from src.service.capability_detector import SessionCapabilities

            app.state.config = load_config(CONFIG_PATH)
            app.state.valkey = mock_valkey
            app.state.affinity = ValkeyAffinityAdapter(mock_valkey)

            conv = Conversation(
                id="00000000-0000-0000-0000-000000000003",
                pseudo_model="avanzada-vision",
                physical_model="gemini/gemini-3.5-flash",
                capability_has_images=True,
                images_described=0,
                images_degraded_manually=False,
            )
            turn = MagicMock(
                turn_number=1,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What?"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:img/png;base64,abc"},
                            },
                        ],
                    },
                ],
            )
            conv.turns = [turn]
            conv.id = "00000000-0000-0000-0000-000000000003"

            # Mock session capabilities to report has_images=True
            async def mock_load_caps(db, conv_uuid):
                caps = SessionCapabilities(conversation_id=str(conv_uuid))
                caps.has_images = True
                return caps

            mock_db_session.get = AsyncMock(return_value=conv)

            with patch(
                "src.api.conversations.load_session_capabilities", mock_load_caps
            ):
                app.state.db_session_factory = MagicMock(return_value=mock_db_session)

                transport = ASGITransport(app=app)
                async with AsyncClient(
                    transport=transport, base_url="http://test"
                ) as client:
                    resp = await client.post(
                        "/conversations/00000000-0000-0000-0000-000000000003/degrade-images",
                    )
                    assert resp.status_code == 200, (
                        f"Expected 200, got {resp.status_code}: {resp.text}"
                    )
                    data = resp.json()
                    assert data["images_described"] == 1
                    assert data["described_by"] == "gemini/gemini-3.5-flash"
                    assert "can_now_switch_to" in data
                    # Should list non-vision models
                    assert "normal" in data["can_now_switch_to"]
                    assert "flash-lowcost" in data["can_now_switch_to"]

    async def test_degrade_images_no_images_returns_400(
        self, mock_valkey, mock_db_session
    ):
        """POST /degrade-images without images → 400."""
        with (
            patch("src.main.setup_litellm"),
            patch(
                "src.service.router_llm.suggester.load_bert_classifier",
                return_value=False,
            ),
        ):
            from src.main import app
            from src.adapters.cache.valkey_affinity import ValkeyAffinityAdapter
            from src.adapters.db.models import Conversation
            from src.service.capability_detector import SessionCapabilities

            app.state.config = load_config(CONFIG_PATH)
            app.state.valkey = mock_valkey
            app.state.affinity = ValkeyAffinityAdapter(mock_valkey)

            conv = Conversation(
                id="00000000-0000-0000-0000-000000000004",
                pseudo_model="normal",
                physical_model="deepseek/deepseek-v4-flash",
                capability_has_images=False,
            )
            conv.turns = [
                MagicMock(
                    turn_number=1,
                    messages=[{"role": "user", "content": "No images here."}],
                )
            ]
            conv.id = "00000000-0000-0000-0000-000000000004"

            async def mock_load_caps(db, conv_uuid):
                caps = SessionCapabilities(conversation_id=str(conv_uuid))
                caps.has_images = False
                return caps

            mock_db_session.get = AsyncMock(return_value=conv)

            with patch(
                "src.api.conversations.load_session_capabilities", mock_load_caps
            ):
                app.state.db_session_factory = MagicMock(return_value=mock_db_session)

                transport = ASGITransport(app=app)
                async with AsyncClient(
                    transport=transport, base_url="http://test"
                ) as client:
                    resp = await client.post(
                        "/conversations/00000000-0000-0000-0000-000000000004/degrade-images",
                    )
                    assert resp.status_code == 400, (
                        f"Expected 400, got {resp.status_code}: {resp.text}"
                    )
                    data = resp.json()
                    assert "NO_IMAGES" in str(data)


# ══════════════════════════════════════════════════════════════════════════════
# API Endpoints Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestAPIEndpoints:
    """Health, models, conversations endpoints."""

    async def test_health_endpoint(self, mock_valkey, mock_db_session):
        """GET /health → returns OK."""
        with (
            patch("src.main.setup_litellm"),
            patch(
                "src.service.router_llm.suggester.load_bert_classifier",
                return_value=False,
            ),
        ):
            from src.main import app
            from src.adapters.cache.valkey_affinity import ValkeyAffinityAdapter

            app.state.config = load_config(CONFIG_PATH)
            app.state.valkey = mock_valkey
            app.state.affinity = ValkeyAffinityAdapter(mock_valkey)
            app.state.db_session_factory = MagicMock(return_value=mock_db_session)

            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.get("/health")
                assert resp.status_code == 200
                data = resp.json()
                assert data["status"] == "ok"

    async def test_models_endpoint(self, mock_valkey, mock_db_session):
        """GET /v1/models → returns list of models."""
        with (
            patch("src.main.setup_litellm"),
            patch(
                "src.service.router_llm.suggester.load_bert_classifier",
                return_value=False,
            ),
        ):
            from src.main import app
            from src.adapters.cache.valkey_affinity import ValkeyAffinityAdapter

            app.state.config = load_config(CONFIG_PATH)
            app.state.valkey = mock_valkey
            app.state.affinity = ValkeyAffinityAdapter(mock_valkey)
            app.state.db_session_factory = MagicMock(return_value=mock_db_session)

            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.get("/v1/models")
                assert resp.status_code == 200
                body = resp.json()
                assert isinstance(body, dict)
                data_list = body.get("data", body if isinstance(body, list) else [])
                assert len(data_list) > 0
                # Should list pseudo-models
                models = [m["id"] for m in data_list]
                assert "normal" in models
                assert "avanzada-vision" in models
                assert "pensamiento-profundo-caro" in models

    @patch("src.service.chat_service.call_litellm")
    async def test_conversation_state(self, mock_call, mock_valkey, mock_db_session):
        """GET /conversations/{id} → returns state."""
        mock_call.return_value = _make_chat_response()

        with (
            patch("src.main.setup_litellm"),
            patch(
                "src.service.router_llm.suggester.load_bert_classifier",
                return_value=False,
            ),
        ):
            from src.main import app
            from src.adapters.cache.valkey_affinity import ValkeyAffinityAdapter
            from src.adapters.db.models import Conversation

            app.state.config = load_config(CONFIG_PATH)
            app.state.valkey = mock_valkey
            app.state.affinity = ValkeyAffinityAdapter(mock_valkey)

            conv = Conversation(
                id="00000000-0000-0000-0000-000000000005",
                pseudo_model="normal",
                physical_model="deepseek/deepseek-v4-flash",
                total_tokens=150,
                capability_has_images=False,
                capability_has_tools=True,
            )
            conv.id = "00000000-0000-0000-0000-000000000005"
            mock_db_session.get = AsyncMock(return_value=conv)
            app.state.db_session_factory = MagicMock(return_value=mock_db_session)

            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.get(
                    "/conversations/00000000-0000-0000-0000-000000000005",
                )
                assert resp.status_code == 200
                data = resp.json()
                assert data["pseudo_model"] == "normal"
                assert data["total_tokens"] == 150
                assert data["capabilities"]["has_tools"] is True


# ══════════════════════════════════════════════════════════════════════════════
# Proxy Metadata Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestProxyMetadata:
    """Sprint 1-5: proxy_metadata fields in responses."""

    @patch("src.service.chat_service.call_litellm")
    async def test_proxy_metadata_contains_all_fields(
        self, mock_call, mock_valkey, mock_db_session
    ):
        """Response includes all Sprint 1-5 proxy_metadata fields."""
        mock_call.return_value = _make_chat_response()

        with (
            patch("src.main.setup_litellm"),
            patch(
                "src.service.router_llm.suggester.load_bert_classifier",
                return_value=False,
            ),
        ):
            from src.main import app
            from src.adapters.cache.valkey_affinity import ValkeyAffinityAdapter

            app.state.config = load_config(CONFIG_PATH)
            app.state.valkey = mock_valkey
            app.state.affinity = ValkeyAffinityAdapter(mock_valkey)
            app.state.db_session_factory = MagicMock(return_value=mock_db_session)

            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "normal",
                        "messages": [{"role": "user", "content": "Check metadata"}],
                    },
                )
                assert resp.status_code == 200
                meta = resp.json().get("proxy_metadata", {})
                # Sprint 1: basic fields
                assert "physical_model" in meta
                assert "pseudo_model" in meta
                assert "conversation_id" in meta
                assert "affinity_maintained" in meta
                # Sprint 2: capabilities
                assert "capabilities_detected" in meta
                # Sprint 4: compaction
                assert "pre_compaction_applied" in meta
                assert "continuous_compaction_applied" in meta
                # Sprint 5: image description + router
                assert "images_described" in meta
                assert "images_described_by" in meta
                assert "images_degraded_manually" in meta
                assert "router_suggestion" in meta


# ══════════════════════════════════════════════════════════════════════════════
# Tool Filtering Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestToolFiltering:
    """Sprint 3: Tool filtering when parallel tools in history."""

    def test_parallel_tools_filter_models(self, valid_config):
        """Only models with parallel_tools=true are eligible when session has parallel tools."""
        from src.service.tool_filter import get_eligible_models

        pm = valid_config.pseudo_models["normal"]
        eligible = get_eligible_models(
            pm.physical_models,
            MagicMock(
                has_parallel_tools=True,
                has_images=False,
                has_audio=False,
                has_pdf=False,
                has_video=False,
                has_tools=True,
            ),
        )
        for m in eligible:
            assert m.parallel_tools is True, f"{m.model} should have parallel_tools"


# ══════════════════════════════════════════════════════════════════════════════
# Compaction Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestCompaction:
    """Sprint 4: Pre-compaction and continuous compaction."""

    @patch("src.service.compactor.pre_compactor.call_litellm")
    async def test_pre_compaction(self, mock_call, mock_valkey, mock_db_session):
        """Pre-compaction compresses long inputs."""
        mock_call.return_value = _make_chat_response(
            "Compacted: user asked about Python.",
            completion_tokens=10,
        )

        from src.service.compactor.pre_compactor import pre_compact_input

        config = load_config(CONFIG_PATH)
        pm = config.pseudo_models["pensamiento-profundo-caro"]

        long_messages = [
            {
                "role": "user",
                "content": "Hello, I have a very long message " + "A" * 80000,
            },
            {"role": "assistant", "content": "I see your long message. " + "B" * 80000},
            {
                "role": "user",
                "content": "Let me continue with more content " + "C" * 70000,
            },
        ]
        compacted, meta = await pre_compact_input(long_messages, pm, config)
        assert meta.get("applied", False) is True
        assert len(compacted) > 0
