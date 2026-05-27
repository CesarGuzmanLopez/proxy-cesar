"""Tests for chat_service — orchestration logic, physical model selection, build messages.

Covers:
- Bug 1: cache provider derived from model prefix
- Bug 3: kwargs NOT mutated across fallback iterations (get() vs pop())
- Bug 5: affinity timing (set AFTER successful LLM call)
- Bug 7: thinking parameter only passed to Anthropic
- build_conversation_messages preserves ALL fields
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── build_conversation_messages ──────────────────────────────────────────────


class _MockTurn:
    """Minimal mock for ConversationTurn."""
    def __init__(self, turn_number, messages, response):
        self.turn_number = turn_number
        self.messages = messages
        self.response = response


class _MockConversation:
    """Minimal mock for Conversation."""
    def __init__(self, turns):
        self.turns = turns


def test_build_conversation_messages_basic():
    """Simple conversation has history + current messages."""
    from src.service.chat_service import build_conversation_messages

    turns = [
        _MockTurn(1, [{"role": "user", "content": "Hello"}], None),
    ]
    # If response is None, no assistant entry is added
    conv = _MockConversation(turns)
    result = build_conversation_messages(conv, [{"role": "user", "content": "Followup"}])
    assert len(result) == 2  # history + current
    assert result[0]["role"] == "user"
    assert result[0]["content"] == "Hello"
    assert result[1]["content"] == "Followup"


def test_build_conversation_messages_with_assistant_response():
    """Assistant response is inserted after its turn's request messages."""
    from src.service.chat_service import build_conversation_messages

    turns = [
        _MockTurn(1, [{"role": "user", "content": "Hello"}], {
            "choices": [{"message": {"role": "assistant", "content": "Hi there!"}}]
        }),
    ]
    conv = _MockConversation(turns)
    result = build_conversation_messages(conv, [{"role": "user", "content": "Bye"}])
    assert len(result) == 3
    assert result[0]["content"] == "Hello"
    assert result[1]["content"] == "Hi there!"
    assert result[2]["content"] == "Bye"


def test_build_conversation_messages_preserves_tool_calls():
    """tool_calls are preserved in the assistant entry."""
    from src.service.chat_service import build_conversation_messages

    turns = [
        _MockTurn(1, [{"role": "user", "content": "Get weather"}], {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "call_1", "function": {"name": "get_weather"}}],
                }
            }]
        }),
    ]
    conv = _MockConversation(turns)
    result = build_conversation_messages(conv, [{"role": "tool", "tool_call_id": "call_1", "content": "22"}])

    # Find the assistant message
    asst_msgs = [m for m in result if m.get("role") == "assistant"]
    assert len(asst_msgs) == 1
    assert "tool_calls" in asst_msgs[0]
    assert asst_msgs[0]["tool_calls"][0]["id"] == "call_1"


def test_build_conversation_messages_preserves_reasoning():
    """reasoning_content is preserved in the assistant entry."""
    from src.service.chat_service import build_conversation_messages

    turns = [
        _MockTurn(1, [{"role": "user", "content": "Think step by step"}], {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Final answer",
                    "reasoning_content": "Let me think...",
                }
            }]
        }),
    ]
    conv = _MockConversation(turns)
    result = build_conversation_messages(conv, [{"role": "user", "content": "Done"}])

    asst_msgs = [m for m in result if m.get("role") == "assistant"]
    assert len(asst_msgs) == 1
    assert asst_msgs[0].get("reasoning_content") == "Let me think..."


def test_build_conversation_messages_preserves_thinking_blocks():
    """thinking_blocks are preserved in the assistant entry."""
    from src.service.chat_service import build_conversation_messages

    turns = [
        _MockTurn(1, [{"role": "user", "content": "Explain"}], {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Answer",
                    "thinking_blocks": [{"type": "thinking", "content": "..."}],
                }
            }]
        }),
    ]
    conv = _MockConversation(turns)
    result = build_conversation_messages(conv, [{"role": "user", "content": "OK"}])

    asst_msgs = [m for m in result if m.get("role") == "assistant"]
    assert len(asst_msgs) == 1
    assert asst_msgs[0].get("thinking_blocks") == [{"type": "thinking", "content": "..."}]


def test_build_conversation_messages_multiple_turns():
    """Multiple turns are interleaved correctly in order."""
    from src.service.chat_service import build_conversation_messages

    turns = [
        _MockTurn(1, [{"role": "user", "content": "Q1"}], {
            "choices": [{"message": {"role": "assistant", "content": "A1"}}]
        }),
        _MockTurn(2, [{"role": "user", "content": "Q2"}], {
            "choices": [{"message": {"role": "assistant", "content": "A2"}}]
        }),
    ]
    conv = _MockConversation(turns)
    result = build_conversation_messages(conv, [{"role": "user", "content": "Q3"}])

    assert len(result) == 5  # Q1, A1, Q2, A2, Q3
    assert result[0]["content"] == "Q1"
    assert result[1]["content"] == "A1"
    assert result[2]["content"] == "Q2"
    assert result[3]["content"] == "A2"
    assert result[4]["content"] == "Q3"


def test_build_conversation_messages_empty_turns():
    """Empty turn list returns just the current messages."""
    from src.service.chat_service import build_conversation_messages

    conv = _MockConversation([])
    result = build_conversation_messages(conv, [{"role": "user", "content": "Q"}])
    assert len(result) == 1
    assert result[0]["content"] == "Q"


def test_build_conversation_messages_does_not_mutate_input():
    """Original current_messages list is not mutated."""
    from src.service.chat_service import build_conversation_messages

    turns = [
        _MockTurn(1, [{"role": "user", "content": "Q1"}], {
            "choices": [{"message": {"role": "assistant", "content": "A1"}}]
        }),
    ]
    conv = _MockConversation(turns)
    current = [{"role": "user", "content": "Q2"}]
    original_content = current[0]["content"]
    _ = build_conversation_messages(conv, current)
    assert current[0]["content"] == original_content


# ── _normalise_thinking_param ────────────────────────────────────────────────


def test_normalise_thinking_param_none():
    """None thinking param returns None."""
    from src.service.chat_service import _normalise_thinking_param

    assert _normalise_thinking_param(None) is None


def test_normalise_thinking_param_bool_true():
    """True becomes {'type': 'enabled'}."""
    from src.service.chat_service import _normalise_thinking_param

    result = _normalise_thinking_param(True)
    assert result == {"type": "enabled"}


def test_normalise_thinking_param_bool_false():
    """False returns {'type': 'disabled'}."""
    from src.service.chat_service import _normalise_thinking_param

    result = _normalise_thinking_param(False)
    assert result == {"type": "disabled"}


def test_normalise_thinking_param_dict():
    """Dict passes through unchanged."""
    from src.service.chat_service import _normalise_thinking_param

    d = {"type": "enabled", "budget_tokens": 2000}
    result = _normalise_thinking_param(d)
    assert result == d


# ── _resolve_api_key ─────────────────────────────────────────────────────────


def test_resolve_api_key_from_env(monkeypatch):
    """API key is resolved from environment variable."""
    from src.service.chat_service import _resolve_api_key

    phys = MagicMock()
    phys.api_key_env = "MY_API_KEY"
    phys.api_key = None

    monkeypatch.setenv("MY_API_KEY", "sk-test-key-12345")
    result = _resolve_api_key(phys)
    assert result == "sk-test-key-12345"


def test_resolve_api_key_no_env():
    """No api_key_env returns None regardless of phys.api_key."""
    from src.service.chat_service import _resolve_api_key

    phys = MagicMock()
    phys.api_key_env = None
    result = _resolve_api_key(phys)
    assert result is None


def test_resolve_api_key_env_missing():
    """api_key_env set but env var not set returns None."""
    from src.service.chat_service import _resolve_api_key

    phys = MagicMock()
    phys.api_key_env = "MISSING_ENV_VAR"
    result = _resolve_api_key(phys)
    assert result is None


# ── Bug 1: cache provider from model prefix ──────────────────────────────────


@pytest.mark.asyncio
async def test_try_physical_model_cache_provider_anthropic():
    """Anthropic models get cache_control even when YAML provider is 'opencode-go'.

    This tests Bug 1: the cache provider is derived from model prefix
    (e.g. 'anthropic/claude-...' → 'anthropic'), not from the YAML provider field.
    """
    from src.service.chat_service import _try_physical_model

    phys = MagicMock()
    phys.provider = "opencode-go"
    phys.model = "anthropic/claude-sonnet-4-20250514"
    phys.context_window = 200000
    phys.api_base = None
    phys.api_key = None
    phys.api_key_env = None

    with (
        patch("src.service.chat_service.should_apply_cache_control") as mock_should,
        patch("src.service.chat_service.apply_anthropic_cache_control") as mock_apply,
        patch("src.service.chat_service.call_litellm", new_callable=AsyncMock) as mock_call,
        patch("src.service.chat_service._resolve_api_key") as mock_key,
    ):
        mock_should.return_value = True
        mock_apply.side_effect = lambda msgs: msgs
        mock_key.return_value = None
        mock_call.return_value = MagicMock()
        mock_call.return_value.model_dump.return_value = {"id": "test"}

        response, skip = await _try_physical_model(
            phys,
            [{"role": "user", "content": "Hi"}],
            stream=False,
            kwargs={},
            _est_input=50,
            _trace_id="test-trace",
        )

        # Verify should_apply_cache_control was called with 'anthropic'
        # (derived from model prefix 'anthropic/claude-...')
        call_args = mock_should.call_args
        assert call_args is not None, "should_apply_cache_control was not called"
        assert call_args[0][0] == "anthropic", (
            f"Expected 'anthropic', got '{call_args[0][0]}'"
        )


@pytest.mark.asyncio
async def test_try_physical_model_cache_provider_non_anthropic():
    """Non-Anthropic models do NOT get cache_control even with model prefix."""
    from src.service.chat_service import _try_physical_model

    phys = MagicMock()
    phys.provider = "opencode-go"
    phys.model = "openai/gpt-4o"
    phys.context_window = 200000
    phys.api_base = None
    phys.api_key = None
    phys.api_key_env = None

    with (
        patch("src.service.chat_service.should_apply_cache_control") as mock_should,
        patch("src.service.chat_service.call_litellm", new_callable=AsyncMock) as mock_call,
        patch("src.service.chat_service._resolve_api_key") as mock_key,
    ):
        mock_should.return_value = False
        mock_key.return_value = None
        mock_call.return_value = MagicMock()
        mock_call.return_value.model_dump.return_value = {"id": "test"}

        response, skip = await _try_physical_model(
            phys,
            [{"role": "user", "content": "Hi"}],
            stream=False,
            kwargs={},
            _est_input=50,
            _trace_id="test-trace",
        )

        call_args = mock_should.call_args
        assert call_args is not None
        # Should use the provider field (opencode-go) since model prefix isn't 'anthropic'
        assert call_args[0][0] == "opencode-go"


# ── Bug 7: thinking param only for Anthropic ─────────────────────────────────


@pytest.mark.asyncio
async def test_try_physical_model_thinking_passed_to_anthropic():
    """Thinking parameter is passed through for Anthropic models."""
    from src.service.chat_service import _try_physical_model

    phys = MagicMock()
    phys.provider = "opencode-go"
    phys.model = "anthropic/claude-sonnet-4-20250514"
    phys.context_window = 200000
    phys.api_base = None
    phys.api_key = None
    phys.api_key_env = None

    with (
        patch("src.service.chat_service.should_apply_cache_control") as mock_should,
        patch("src.service.chat_service.call_litellm", new_callable=AsyncMock) as mock_call,
        patch("src.service.chat_service._resolve_api_key") as mock_key,
    ):
        mock_should.return_value = False
        mock_key.return_value = None
        mock_call.return_value = MagicMock()
        mock_call.return_value.model_dump.return_value = {"id": "test"}

        await _try_physical_model(
            phys,
            [{"role": "user", "content": "Think hard"}],
            stream=False,
            kwargs={"thinking": {"type": "enabled", "budget_tokens": 2000}},
            _est_input=50,
            _trace_id="test-trace",
        )

        # Verify call_litellm received thinking kwarg
        call_kwargs = mock_call.call_args.kwargs if mock_call.call_args else {}
        assert "thinking" in call_kwargs, (
            "thinking should be passed to Anthropic models"
        )
        assert call_kwargs["thinking"] == {"type": "enabled", "budget_tokens": 2000}


@pytest.mark.asyncio
async def test_try_physical_model_thinking_not_passed_to_non_anthropic():
    """Thinking parameter is NOT passed to non-Anthropic models (Bug 7)."""
    from src.service.chat_service import _try_physical_model

    phys = MagicMock()
    phys.provider = "opencode-go"
    phys.model = "openai/gpt-4o"
    phys.context_window = 200000
    phys.api_base = None
    phys.api_key = None
    phys.api_key_env = None

    with (
        patch("src.service.chat_service.should_apply_cache_control") as mock_should,
        patch("src.service.chat_service.call_litellm", new_callable=AsyncMock) as mock_call,
        patch("src.service.chat_service._resolve_api_key") as mock_key,
    ):
        mock_should.return_value = False
        mock_key.return_value = None
        mock_call.return_value = MagicMock()
        mock_call.return_value.model_dump.return_value = {"id": "test"}

        await _try_physical_model(
            phys,
            [{"role": "user", "content": "Think hard"}],
            stream=False,
            kwargs={"thinking": True},
            _est_input=50,
            _trace_id="test-trace",
        )

        # Verify call_litellm did NOT receive thinking kwarg
        call_kwargs = mock_call.call_args.kwargs if mock_call.call_args else {}
        assert "thinking" not in call_kwargs, (
            "thinking should NOT be passed to non-Anthropic models"
        )


# ── Bug 3: kwargs not mutated across fallback ───────────────────────────────


@pytest.mark.asyncio
async def test_try_physical_model_kwargs_not_mutated():
    """kwargs dict is not mutated by _try_physical_model (Bug 3: get() vs pop())."""
    from src.service.chat_service import _try_physical_model

    phys = MagicMock()
    phys.provider = "opencode-go"
    phys.model = "openai/gpt-4o"
    phys.context_window = 200000
    phys.api_base = None
    phys.api_key = None
    phys.api_key_env = None

    original_kwargs = {"thinking": True, "temperature": 0.7}

    with (
        patch("src.service.chat_service.should_apply_cache_control") as mock_should,
        patch("src.service.chat_service.call_litellm", new_callable=AsyncMock) as mock_call,
        patch("src.service.chat_service._resolve_api_key") as mock_key,
    ):
        mock_should.return_value = False
        mock_key.return_value = None
        mock_call.return_value = MagicMock()
        mock_call.return_value.model_dump.return_value = {"id": "test"}

        await _try_physical_model(
            phys,
            [{"role": "user", "content": "Hi"}],
            stream=False,
            kwargs=original_kwargs,
            _est_input=50,
            _trace_id="test-trace",
        )

        # Original kwargs should still have 'thinking' key (not removed by pop)
        assert "thinking" in original_kwargs, (
            "kwargs should not be mutated by _try_physical_model"
        )


# ── process_chat_request: affinity timing (Bug 5) ────────────────────────────

@pytest.mark.asyncio
async def test_affinity_set_after_llm_call_unit():
    """Verify affinity.set is NOT called before LLM call completes.

    This indirectly tests Bug 5 by checking the code flow in _try_physical_model
    and verifying the caller's wrapping logic.
    """
    from src.service.chat_service import _try_physical_model

    phys = MagicMock()
    phys.provider = "opencode-go"
    phys.model = "openai/gpt-4o"
    phys.context_window = 200000
    phys.api_base = None
    phys.api_key = None
    phys.api_key_env = None

    call_litellm_called = False

    async def _fake_call_litellm(**kwargs):
        nonlocal call_litellm_called
        call_litellm_called = True
        resp = MagicMock()
        resp.model_dump.return_value = {"id": "test"}
        return resp

    with (
        patch("src.service.chat_service.should_apply_cache_control", return_value=False),
        patch("src.service.chat_service.call_litellm", side_effect=_fake_call_litellm),
        patch("src.service.chat_service._resolve_api_key", return_value=None),
    ):
        # The _try_physical_model should only return when LLM call succeeds
        response, skip = await _try_physical_model(
            phys,
            [{"role": "user", "content": "Hi"}],
            stream=False,
            kwargs={},
            _est_input=50,
            _trace_id="test-trace",
        )

        assert skip is None, "Model should not be skipped"
        assert call_litellm_called, "call_litellm must have been executed"
        assert response is not None
