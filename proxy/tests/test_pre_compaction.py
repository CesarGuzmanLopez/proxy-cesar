"""Tests for pre-compaction service.

Sprint 4 §5.1 — minimum 10 tests.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.service.compactor.pre_compactor import _extract_text_content, pre_compact_input

# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_content(approx_chars: int) -> str:
    """Generate content of approximately the given character length."""
    base = "Technical log entry: error=timeout module=auth_service file=src/db/connection.ts:42 traceback=RuntimeError "
    repeats = max(1, approx_chars // len(base))
    return base * repeats


# ── Constants ────────────────────────────────────────────────────────────────

# Threshold is 32000 tokens. Estimate ~4 chars/token with tiktoken.
# 800K chars should comfortably exceed 32K tokens (~200K tokens estimated)
CONTENT_ABOVE_THRESHOLD = _make_content(800_000)

# Small content stays well below threshold
CONTENT_BELOW_THRESHOLD = "Hello, short message."

# Long system message (no user message) to test no_user_message path
CONTENT_SYSTEM_LONG = _make_content(800_000)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_litellm_success():
    """Mock call_litellm returning a valid summary response."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Extracted relevant information."
    mock_response.usage.completion_tokens = 50

    mock_response.model_dump.return_value = {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Extracted relevant information.",
                }
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
    }

    with patch(
        "src.service.compactor.pre_compactor.call_litellm", new_callable=AsyncMock
    ) as mock:
        mock.return_value = mock_response
        yield mock


@pytest.fixture
def mock_litellm_failure():
    """Mock call_litellm raising an exception."""
    with patch(
        "src.service.compactor.pre_compactor.call_litellm", new_callable=AsyncMock
    ) as mock:
        mock.side_effect = RuntimeError("API unavailable")
        yield mock


@pytest.fixture
def config_with_compactor():
    """Minimal config with pre-compaction enabled."""
    config = MagicMock()
    compactor_pm = MagicMock()
    phys = MagicMock()
    phys.model = "glm-4.5-flash"
    compactor_pm.physical_models = [phys]
    config.pseudo_models = {"deep-flash": compactor_pm}
    return config


@pytest.fixture
def pseudo_model_with_pre():
    """Pseudo-model schema with pre_compaction enabled."""
    pm = MagicMock()
    pm.pre_compaction.enabled = True
    pm.pre_compaction.threshold = 32000
    pm.pre_compaction.target_tokens = 8000
    pm.pre_compaction.compactor = "deep-flash"
    pm.context_window = 200000
    return pm


# ── Pre-compaction Tests ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_input_below_threshold_no_compaction(
    mock_litellm_success, config_with_compactor, pseudo_model_with_pre
):
    """Input below threshold → no pre-compaction applied."""
    messages = [{"role": "user", "content": CONTENT_BELOW_THRESHOLD}]
    modified, meta = await pre_compact_input(
        messages, pseudo_model_with_pre, config_with_compactor
    )
    assert meta["applied"] is False
    assert meta["reason"] == "below_threshold"
    assert modified is messages  # Same object, no copy


@pytest.mark.asyncio
async def test_input_above_threshold_triggers_compaction(
    mock_litellm_success, config_with_compactor, pseudo_model_with_pre
):
    """Input above threshold → pre-compaction applied."""
    messages = [{"role": "user", "content": CONTENT_ABOVE_THRESHOLD}]
    modified, meta = await pre_compact_input(
        messages, pseudo_model_with_pre, config_with_compactor
    )
    assert meta["applied"] is True
    assert "original_input_tokens" in meta
    assert "compacted_input_tokens" in meta
    assert meta["compactor_model"] == "glm-4.5-flash"
    assert meta["compactor_pseudo_model"] == "deep-flash"
    assert "savings_tokens" in meta


@pytest.mark.asyncio
async def test_last_user_message_replaced_with_summary(
    mock_litellm_success, config_with_compactor, pseudo_model_with_pre
):
    """Last user message is replaced with compacted summary."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "First question?"},
        {"role": "assistant", "content": "First answer."},
        {"role": "user", "content": CONTENT_ABOVE_THRESHOLD},
    ]
    modified, meta = await pre_compact_input(
        messages, pseudo_model_with_pre, config_with_compactor
    )
    assert meta["applied"] is True
    assert "[Pre-compacted input" in modified[-1]["content"]
    assert modified[-1]["role"] == "user"
    # System, first user, and assistant messages are unchanged
    assert modified[0]["content"] == "You are a helpful assistant."
    assert modified[1]["content"] == "First question?"
    assert modified[2]["content"] == "First answer."


@pytest.mark.asyncio
async def test_system_and_tool_messages_not_modified(
    mock_litellm_success, config_with_compactor, pseudo_model_with_pre
):
    """System messages and tool history are NOT modified."""
    messages = [
        {"role": "system", "content": "Original system prompt."},
        {"role": "user", "content": CONTENT_ABOVE_THRESHOLD},
        {
            "role": "assistant",
            "content": "Response with tool_calls",
            "tool_calls": [{"id": "call_1"}],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "Tool result"},
    ]
    modified, meta = await pre_compact_input(
        messages, pseudo_model_with_pre, config_with_compactor
    )
    assert meta["applied"] is True
    # Only the last user message (index 1) should be modified
    assert modified[0]["role"] == "system"
    assert modified[0]["content"] == "Original system prompt."
    assert modified[2]["role"] == "assistant"
    assert modified[2]["tool_calls"] == [{"id": "call_1"}]
    assert modified[3]["role"] == "tool"


@pytest.mark.asyncio
async def test_compaction_metadata_in_response(
    mock_litellm_success, config_with_compactor, pseudo_model_with_pre
):
    """Response includes all compaction metadata fields."""
    messages = [{"role": "user", "content": CONTENT_ABOVE_THRESHOLD}]
    _, meta = await pre_compact_input(
        messages, pseudo_model_with_pre, config_with_compactor
    )
    assert meta["applied"] is True
    assert meta["original_input_tokens"] > 0
    assert meta["compacted_input_tokens"] > 0
    assert meta["savings_tokens"] > 0
    assert meta["compactor_model"] == "glm-4.5-flash"
    assert meta["compactor_pseudo_model"] == "deep-flash"


@pytest.mark.asyncio
async def test_prompt_includes_technical_details(
    mock_litellm_success, config_with_compactor, pseudo_model_with_pre
):
    """Pre-compaction prompt includes all technical details from the input."""
    messages = [{"role": "user", "content": CONTENT_ABOVE_THRESHOLD}]
    with patch(
        "src.service.compactor.pre_compactor.build_pre_compaction_prompt"
    ) as mock_build:
        mock_build.return_value = "mock prompt"
        await pre_compact_input(messages, pseudo_model_with_pre, config_with_compactor)
        # Verify the prompt was built with the actual user content
        mock_build.assert_called_once()
        call_user_content = mock_build.call_args[1]["user_content"]
        assert "Technical log entry" in call_user_content


@pytest.mark.asyncio
async def test_compactor_failure_uses_original_input(
    mock_litellm_failure, config_with_compactor, pseudo_model_with_pre
):
    """Compactor fails → original input used with warning."""
    messages = [{"role": "user", "content": CONTENT_ABOVE_THRESHOLD}]
    modified, meta = await pre_compact_input(
        messages, pseudo_model_with_pre, config_with_compactor
    )
    assert meta["applied"] is False
    assert "compactor_failed" in meta["reason"]
    assert "warning" in meta
    assert modified is messages  # Original, unmodified


@pytest.mark.asyncio
async def test_compactor_pseudo_model_not_found(config_with_compactor):
    """Compactor pseudo-model not found → error with warning."""
    pm = MagicMock()
    pm.pre_compaction.enabled = True
    pm.pre_compaction.threshold = 1
    pm.pre_compaction.target_tokens = 8000
    pm.pre_compaction.compactor = "nonexistent-model"

    config = MagicMock()
    config.pseudo_models = {}  # Empty: no compactor model available

    messages = [{"role": "user", "content": "Any message will exceed threshold of 1."}]
    modified, meta = await pre_compact_input(messages, pm, config)
    assert meta["applied"] is False
    assert "compactor_pseudo_model_not_found" in meta["reason"]
    assert "warning" in meta
    assert modified is messages


@pytest.mark.asyncio
async def test_no_compactor_physical_models(config_with_compactor):
    """Compactor pseudo-model has no physical models → warning."""
    pm = MagicMock()
    pm.pre_compaction.enabled = True
    pm.pre_compaction.threshold = 1
    pm.pre_compaction.target_tokens = 8000
    pm.pre_compaction.compactor = "deep-flash"

    config = MagicMock()
    compactor_pm = MagicMock()
    compactor_pm.physical_models = []  # No physical models!
    config.pseudo_models = {"deep-flash": compactor_pm}

    messages = [{"role": "user", "content": "Any message will exceed threshold of 1."}]
    modified, meta = await pre_compact_input(messages, pm, config)
    assert meta["applied"] is False
    assert "compactor_no_physical_models" in meta["reason"]
    assert "warning" in meta
    assert modified is messages


@pytest.mark.asyncio
async def test_no_user_message_no_compaction(
    mock_litellm_success, config_with_compactor, pseudo_model_with_pre
):
    """No user message in input → no compaction applied."""
    # threshold is 32000, and we exceed it with a system message (no user role)
    messages = [{"role": "system", "content": CONTENT_SYSTEM_LONG}]
    modified, meta = await pre_compact_input(
        messages, pseudo_model_with_pre, config_with_compactor
    )
    assert meta["applied"] is False
    assert meta["reason"] == "no_user_message"
    assert modified is messages


@pytest.mark.asyncio
async def test_very_large_input_handled(
    mock_litellm_success, config_with_compactor, pseudo_model_with_pre
):
    """Very large input (200K tokens) → compactor handles it without error."""
    # 2M chars → well beyond any threshold
    very_large = _make_content(2_000_000)
    messages = [{"role": "user", "content": very_large}]
    modified, meta = await pre_compact_input(
        messages, pseudo_model_with_pre, config_with_compactor
    )
    # Should either apply compaction or fail gracefully
    assert "applied" in meta
    if meta["applied"]:
        assert meta["original_input_tokens"] > 0
        assert meta["compacted_input_tokens"] > 0
    else:
        # If skipping, there should be a reason
        assert "reason" in meta


def test_extract_text_content_string():
    """_extract_text_content handles plain string content."""
    msg = {"role": "user", "content": "Hello world"}
    assert _extract_text_content(msg) == "Hello world"


def test_extract_text_content_list():
    """_extract_text_content handles multimodal content list."""
    msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image:"},
            {"type": "image_url", "image_url": {"url": "data:image/..."}},
        ],
    }
    result = _extract_text_content(msg)
    assert "Describe this image:" in result
    assert "data:image" not in result  # Image URLs are excluded
