"""Sprint 5 — Router LLM evaluation service tests.

Pure unit tests (no DB, no API). Tests the suggester module:
- evaluate_complexity() — 6 tests
- is_downgrade() — 3 tests
- _compute_tier() — 1 test
- _extract_last_user_content() — 2 tests
Total: 12 tests

python.md §4: Pure functions tested deterministically.
python.md §3: Result monad — errors returned, not raised.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.service.router_llm.suggester import (
    ALLOWED_SUGGESTIONS,
    MAX_TASK_CHARS,
    _compute_tier,
    _extract_last_user_content,
    evaluate_complexity,
    is_downgrade,
)
from src.config.pseudo_models import load_config

# Load production config for tier tests
_CONFIG = load_config("pseudo_models.yaml")


def _make_simple_eval_response(complexity: str = "simple", suggested: str = "normal", reason: str = "Simple task.") -> MagicMock:
    """Create a mock LiteLLM response for a simple task evaluation."""
    payload = {
        "complexity": complexity,
        "suggested_pseudo_model": suggested,
        "reason": reason,
    }
    mock = MagicMock()
    mock.choices = [MagicMock()]
    mock.choices[0].message.content = json.dumps(payload)
    mock.usage.completion_tokens = 30
    return mock


# ── evaluate_complexity tests ─────────────────────────────────────────────────


class TestEvaluateComplexity:
    """6 tests for evaluate_complexity()."""

    @patch("src.service.router_llm.suggester.call_litellm")
    async def test_simple_task_returns_suggestion(self, mock_call):
        """Simple task → returns complexity: simple with suggested model."""
        mock_call.return_value = _make_simple_eval_response("simple", "normal", "Simple question.")
        result = await evaluate_complexity(
            messages=[{"role": "user", "content": "What is 2+2?"}],
            suggester_model="zai/glm-4.5-flash",
        )
        assert result is not None
        assert result["complexity"] == "simple"
        assert result["suggested"] == "normal"
        assert result["source"] == "llm"

    @patch("src.service.router_llm.suggester.call_litellm")
    async def test_no_user_message_returns_none(self, mock_call):
        """No user message → None (skip evaluation)."""
        result = await evaluate_complexity(
            messages=[{"role": "system", "content": "You are a helper."}],
            suggester_model="zai/glm-4.5-flash",
        )
        assert result is None
        mock_call.assert_not_called()

    @patch("src.service.router_llm.suggester.call_litellm")
    async def test_only_last_user_message_evaluated(self, mock_call):
        """Multi-turn → only last user message evaluated."""
        mock_call.return_value = _make_simple_eval_response("simple", "normal", "Simple task.")
        messages = [
            {"role": "user", "content": "First message about architecture."},
            {"role": "assistant", "content": "Let me help with that."},
            {"role": "user", "content": "What is the weather?"},
        ]
        result = await evaluate_complexity(
            messages=messages,
            suggester_model="zai/glm-4.5-flash",
        )
        assert result is not None
        # Verify the prompt included "weather" not "architecture"
        call_args, call_kwargs = mock_call.call_args
        prompt_text = call_kwargs["messages"][0]["content"]
        assert "weather" in prompt_text
        assert "architecture" not in prompt_text

    @patch("src.service.router_llm.suggester.call_litellm")
    async def test_multimodal_content_extracts_text(self, mock_call):
        """Multimodal message → extracts text parts only, ignores images."""
        mock_call.return_value = _make_simple_eval_response("simple", "normal", "Text extraction task.")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this diagram."},
                    {"type": "image_url", "image_url": {"url": "https://example.com/diag.png"}},
                ],
            }
        ]
        result = await evaluate_complexity(
            messages=messages,
            suggester_model="zai/glm-4.5-flash",
        )
        assert result is not None
        # The prompt should contain the text part, not the image
        call_args, call_kwargs = mock_call.call_args
        prompt_text = call_kwargs["messages"][0]["content"]
        assert "Describe this diagram" in prompt_text
        assert "image_url" not in prompt_text

    @patch("src.service.router_llm.suggester.call_litellm")
    async def test_image_only_message_returns_none(self, mock_call):
        """Image-only message (no text) → None (skip evaluation)."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
                ],
            }
        ]
        result = await evaluate_complexity(
            messages=messages,
            suggester_model="zai/glm-4.5-flash",
        )
        assert result is None
        mock_call.assert_not_called()

    @patch("src.service.router_llm.suggester.call_litellm")
    async def test_litellm_failure_returns_none(self, mock_call):
        """LiteLLM failure → None (non-blocking)."""
        mock_call.side_effect = ConnectionError("API unavailable")
        result = await evaluate_complexity(
            messages=[{"role": "user", "content": "Hello"}],
            suggester_model="zai/glm-4.5-flash",
        )
        assert result is None


# ── is_downgrade tests ────────────────────────────────────────────────────────


class TestIsDowngrade:
    """3 tests for is_downgrade()."""

    def test_cheaper_model_is_downgrade(self):
        """flash-lowcost → normal is a downgrade."""
        assert is_downgrade("flash-lowcost", "normal", _CONFIG) is True

    def test_more_expensive_not_downgrade(self):
        """Switching TO a more expensive model is NOT a downgrade.
        Suggested (pensamiento) costs more than current (normal) → False.
        """
        assert is_downgrade("pensamiento-profundo-caro", "normal", _CONFIG) is False

    def test_same_model_not_downgrade(self):
        """Same model → not a downgrade."""
        assert is_downgrade("normal", "normal", _CONFIG) is False


# ── _compute_tier tests ────────────────────────────────────────────────────────


class TestComputeTier:
    """1 test for _compute_tier()."""

    def test_ranks_correctly(self):
        """pensamiento-profundo-caro > normal > flash-lowcost."""
        deep_tier = _compute_tier("pensamiento-profundo-caro", _CONFIG)
        normal_tier = _compute_tier("normal", _CONFIG)
        flash_tier = _compute_tier("flash-lowcost", _CONFIG)
        deep_v4_tier = _compute_tier("deep-flash", _CONFIG)

        assert deep_tier > normal_tier, (
            f"pensamiento-profundo-caro ({deep_tier}) should be > normal ({normal_tier})"
        )
        assert normal_tier > flash_tier, (
            f"normal ({normal_tier}) should be > flash-lowcost ({flash_tier})"
        )
        assert deep_v4_tier >= normal_tier or deep_v4_tier < flash_tier, (
            f"deep-flash ({deep_v4_tier}) tier check"
        )


# ── _extract_last_user_content tests ──────────────────────────────────────────


class TestExtractLastUserContent:
    """2 tests for _extract_last_user_content()."""

    def test_extracts_last_user_text(self):
        """Returns last user message text, capped at MAX_TASK_CHARS."""
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "response"},
            {"role": "user", "content": "last message here"},
        ]
        result = _extract_last_user_content(messages)
        assert result == "last message here"

    def test_image_only_returns_none(self):
        """Image-only message returns None."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
                ],
            }
        ]
        result = _extract_last_user_content(messages)
        assert result is None


# ── ALLOWED_SUGGESTIONS invariant ──────────────────────────────────────────────


class TestAllowedSuggestions:
    """1 test for ALLOWED_SUGGESTIONS invariant."""

    def test_all_suggestions_are_valid_pseudo_models(self):
        """All entries in ALLOWED_SUGGESTIONS exist in config."""
        for name in ALLOWED_SUGGESTIONS:
            assert name in _CONFIG.pseudo_models, (
                f"ALLOWED_SUGGESTIONS contains '{name}' which is not in pseudo_models.yaml"
            )
