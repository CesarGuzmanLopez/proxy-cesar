"""Feature— Router LLM evaluation service tests.

Pure unit tests (no DB, no API). Tests the suggester module:
- evaluate_complexity() — 6 tests
- is_downgrade() — 3 tests
- _compute_tier() — 1 test
- _extract_last_user_content() — 2 tests
- evaluate_router_suggestion() — 2 tests
Total: 14 tests

python.md §4: Pure functions tested deterministically.
python.md §3: Result monad — errors returned, not raised.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.service.chat_service import evaluate_router_suggestion
from src.service.router_llm.suggester import (
    ALLOWED_SUGGESTIONS,
    _compute_tier,
    _extract_last_user_content,
    evaluate_complexity,
    is_downgrade,
)
from src.config.pseudo_models import load_config

# Load production config for tier tests
_CONFIG = load_config("pseudo_models.yaml")


def _make_simple_eval_response(
    complexity: str = "simple", suggested: str = "normal", reason: str = "Simple task."
) -> MagicMock:
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
        mock_call.return_value = _make_simple_eval_response(
            "simple", "normal", "Simple question."
        )
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
        mock_call.return_value = _make_simple_eval_response(
            "simple", "normal", "Simple task."
        )
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
        mock_call.return_value = _make_simple_eval_response(
            "simple", "normal", "Text extraction task."
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this diagram."},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/diag.png"},
                    },
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
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/img.png"},
                    },
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
        """normal-gratis → normal is a downgrade."""
        assert is_downgrade("normal-gratis", "normal", _CONFIG) is True

    def test_more_expensive_not_downgrade(self):
        """Switching TO a more expensive model is NOT a downgrade.
        Suggested (normal) costs more than current (tareas-avanzadas) → False.
        """
        assert is_downgrade("normal", "tareas-avanzadas", _CONFIG) is False

    def test_same_model_not_downgrade(self):
        """Same model → not a downgrade."""
        assert is_downgrade("normal", "normal", _CONFIG) is False


# ── _compute_tier tests ────────────────────────────────────────────────────────


class TestComputeTier:
    """2 tests for _compute_tier()."""

    def test_ranks_correctly(self):
        """Tiers reflect real config: context_window + tools_strict + vision."""
        deep_tier = _compute_tier("pensamiento-profundo-caro", _CONFIG)
        normal_tier = _compute_tier("normal", _CONFIG)
        flash_tier = _compute_tier("flash", _CONFIG)
        vision_tier = _compute_tier("vision", _CONFIG)

        # All have real context_windows now, so tier ordering is config-driven
        assert deep_tier == 57, f"kimi-k2.5 + deepseek-v4-pro = {deep_tier}"
        assert flash_tier == 24, f"gpt-oss-120b + deepseek-v4-flash = {flash_tier}"
        assert normal_tier == 206, f"mimo-v2.5 + deepseek-v4-flash = {normal_tier}"
        assert vision_tier == 29, f"llama-4-scout + mimo-v2-omni = {vision_tier}"

    def test_unknown_model_returns_zero(self):
        """Non-existent model name → returns 0."""
        assert _compute_tier("non_existent_model", _CONFIG) == 0


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
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/img.png"},
                    },
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


# ── evaluate_router_suggestion tests ───────────────────────────────────────────


class TestEvaluateRouterSuggestion:
    """2 tests for evaluate_router_suggestion()."""

    @pytest.mark.asyncio
    async def test_evaluate_router_suggestion_disabled(self):
        """router_llm disabled → returns None."""
        pm_schema = MagicMock()
        pm_schema.router_llm.enabled = False

        result = await evaluate_router_suggestion(
            pm_schema=pm_schema,
            messages=[{"role": "user", "content": "Hello"}],
            current_pseudo_name="normal",
            config=MagicMock(),
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_evaluate_router_suggestion_no_suggester(self):
        """Suggester model not found → returns None."""
        pm_schema = MagicMock()
        pm_schema.router_llm.enabled = True
        pm_schema.router_llm.suggester = "nonexistent-suggester"

        config = MagicMock()
        config.pseudo_models.get.return_value = None

        result = await evaluate_router_suggestion(
            pm_schema=pm_schema,
            messages=[{"role": "user", "content": "Hello"}],
            current_pseudo_name="normal",
            config=config,
        )

        assert result is None
        config.pseudo_models.get.assert_called_once_with("nonexistent-suggester")
