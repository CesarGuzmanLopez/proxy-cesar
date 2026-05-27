"""Sprint 5 — Auto-describe images service tests.

Pure unit tests (no DB, no API). Tests the image_describer module:
- find_image_refs() — 5 tests
- describe_image() — 3 tests
- auto_describe_images() — 7 tests
Total: 15 tests

python.md §4: Pure functions tested deterministically.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.service.multimedia.image_describer import (
    MAX_TOKENS_PER_IMAGE,
    TAG_PREFIX,
    auto_describe_images,
    describe_image,
    find_image_refs,
)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_image_msg(
    url: str = "https://example.com/img.png", detail: str = "auto"
) -> dict:
    """Create a message with a single image_url content part."""
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is in this image?"},
            {"type": "image_url", "image_url": {"url": url, "detail": detail}},
        ],
    }


def _make_text_msg(text: str = "Hello") -> dict:
    """Create a plain text message."""
    return {"role": "user", "content": text}


def _make_mock_litellm_individual(description: str = "A screenshot."):
    """Mock for individual describe_image() calls — returns plain string."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = description
    mock_response.usage.completion_tokens = 15
    return mock_response


def _make_mock_litellm(description: str = "A screenshot of a code editor.", count: int = 1):
    """Mock for batch auto_describe_images() calls — returns JSON array.

    The batch parser expects a JSON array of strings as the response content.
    """
    import json as _json
    descriptions = [f"{description}" if i == 0 else f"{description} ({i+1})" for i in range(count)]
    json_response = _json.dumps(descriptions)
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json_response
    mock_response.usage.completion_tokens = 15 * count
    return mock_response


# ── find_image_refs tests ──────────────────────────────────────────────────────


class TestFindImageRefs:
    """5 tests for find_image_refs()."""

    def test_finds_single_image_url(self):
        """Single image_url → found with correct metadata."""
        messages = [_make_image_msg("https://example.com/test.png")]
        refs = find_image_refs(messages)
        assert len(refs) == 1
        assert refs[0]["url"] == "https://example.com/test.png"
        assert refs[0]["detail"] == "auto"
        assert refs[0]["msg_idx"] == 0
        assert refs[0]["part_idx"] == 1  # Second content part

    def test_handles_multiple_images(self):
        """Multiple images → sequential indexing."""
        messages = [
            _make_image_msg("https://example.com/a.png"),
            _make_image_msg("https://example.com/b.png"),
        ]
        refs = find_image_refs(messages)
        assert len(refs) == 2
        assert refs[0]["url"] == "https://example.com/a.png"
        assert refs[1]["url"] == "https://example.com/b.png"

    def test_deduplicates_same_url(self):
        """Same URL in multiple messages → second is duplicate."""
        messages = [
            _make_image_msg("https://example.com/same.png"),
            _make_image_msg("https://example.com/same.png"),
        ]
        refs = find_image_refs(messages)
        assert len(refs) == 2
        assert refs[0]["is_duplicate"] is False
        assert refs[1]["is_duplicate"] is True

    def test_ignores_text_only_messages(self):
        """Text-only messages → empty refs."""
        messages = [_make_text_msg("Just text"), _make_text_msg("More text")]
        refs = find_image_refs(messages)
        assert len(refs) == 0

    def test_handles_messages_without_content(self):
        """Messages with None content → no crash."""
        messages = [{"role": "user"}, {"role": "assistant", "content": None}]
        refs = find_image_refs(messages)
        assert len(refs) == 0


# ── describe_image tests ──────────────────────────────────────────────────────


class TestDescribeImage:
    """3 tests for describe_image()."""

    @patch("src.service.multimedia.image_describer.call_litellm")
    async def test_calls_litellm_with_correct_args(self, mock_call):
        """describe_image calls call_litellm with correct vision model."""
        mock_call.return_value = _make_mock_litellm_individual("A screenshot.")
        desc, tokens = await describe_image(
            image_url="https://example.com/img.png",
            detail="high",
            vision_model="gemini/gemini-3.5-flash",
        )
        mock_call.assert_awaited_once()
        args, kwargs = mock_call.await_args
        assert kwargs["model"] == "gemini/gemini-3.5-flash"
        assert kwargs["max_tokens"] == MAX_TOKENS_PER_IMAGE
        assert kwargs["temperature"] == 0.0
        # Check the image_url is in the messages
        msg = kwargs["messages"][0]
        assert msg["role"] == "user"
        content = msg["content"]
        assert any(
            p.get("type") == "image_url"
            and p.get("image_url", {}).get("url") == "https://example.com/img.png"
            and p.get("image_url", {}).get("detail") == "high"
            for p in content
        )

    @patch("src.service.multimedia.image_describer.call_litellm")
    async def test_returns_description_and_tokens(self, mock_call):
        """Returns (description_text, tokens_used)."""
        mock_call.return_value = _make_mock_litellm_individual("A diagram of architecture.")
        desc, tokens = await describe_image(
            image_url="https://example.com/diag.png",
            detail="auto",
            vision_model="gemini/gemini-3.5-flash",
        )
        assert desc == "A diagram of architecture."
        assert tokens == 15

    @patch("src.service.multimedia.image_describer.call_litellm")
    async def test_failure_returns_placeholder(self, mock_call):
        """On LiteLLM failure → placeholder text, 0 tokens."""
        mock_call.side_effect = ConnectionError("API unavailable")
        desc, tokens = await describe_image(
            image_url="https://example.com/fail.png",
            detail="auto",
            vision_model="gemini/gemini-3.5-flash",
        )
        assert TAG_PREFIX in desc
        assert "FAILED" in desc
        assert tokens == 0


# ── auto_describe_images tests ────────────────────────────────────────────────


class TestAutoDescribeImages:
    """7 tests for auto_describe_images()."""

    @patch("src.service.multimedia.image_describer.call_litellm")
    async def test_no_images_returns_original(self, mock_call):
        """No images → returns original messages unchanged."""
        messages = [_make_text_msg("Hello")]
        modified, meta = await auto_describe_images(
            messages,
            "gemini/gemini-3.5-flash",
        )
        assert modified == messages
        assert meta["images_described"] == 0
        assert meta["status"] == "no_images_found"
        mock_call.assert_not_called()

    @patch("src.service.multimedia.image_describer.call_litellm")
    async def test_single_image_tagged(self, mock_call):
        """Single image → [IMAGE_DESCRIBED #1] tag present."""
        mock_call.return_value = _make_mock_litellm("A code screenshot.")
        messages = [_make_image_msg("https://example.com/code.png")]
        modified, meta = await auto_describe_images(
            messages,
            "gemini/gemini-3.5-flash",
        )
        assert meta["images_described"] == 1
        content = modified[0]["content"]
        tag_text = content[1]["text"]  # Replaced image part
        assert f"[{TAG_PREFIX} #1" in tag_text
        assert "A code screenshot." in tag_text

    @patch("src.service.multimedia.image_describer.call_litellm")
    async def test_multiple_images_sequential_tags(self, mock_call):
        """Multiple images → #1, #2, #3 sequential tags."""
        mock_call.return_value = _make_mock_litellm("Description.", count=3)
        messages = [
            _make_image_msg("https://example.com/a.png"),
            _make_image_msg("https://example.com/b.png"),
            _make_image_msg("https://example.com/c.png"),
        ]
        modified, meta = await auto_describe_images(
            messages,
            "gemini/gemini-3.5-flash",
        )
        assert meta["images_described"] == 3
        assert meta["unique_images_described"] == 3
        mock_call.assert_awaited_once()  # Single batch call
        # Each modified message should have the tag
        for msg in modified:
            content = msg["content"]
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    if TAG_PREFIX in part["text"]:
                        break
            else:
                pytest.fail("No IMAGE_DESCRIBED tag found in modified message")

    @patch("src.service.multimedia.image_describer.call_litellm")
    async def test_deduplicates_urls(self, mock_call):
        """Same URL → described once, metadata shows skipped duplicate."""
        mock_call.return_value = _make_mock_litellm("Description.", count=1)
        messages = [
            _make_image_msg("https://example.com/same.png"),
            _make_image_msg("https://example.com/same.png"),
        ]
        modified, meta = await auto_describe_images(
            messages,
            "gemini/gemini-3.5-flash",
        )
        assert meta["images_described"] == 2  # Both instances tagged
        assert meta["unique_images_described"] == 1  # Only 1 unique
        assert meta["duplicate_images_skipped"] == 1
        mock_call.assert_awaited_once()  # One batch call

    @patch("src.service.multimedia.image_describer.call_litellm")
    async def test_preserves_non_image_content(self, mock_call):
        """Non-image text parts preserved unchanged."""
        mock_call.return_value = _make_mock_litellm("Description.")
        messages = [_make_image_msg("https://example.com/img.png")]
        modified, meta = await auto_describe_images(
            messages,
            "gemini/gemini-3.5-flash",
        )
        # The first content part (text) should be unchanged
        first_part = modified[0]["content"][0]
        assert first_part["type"] == "text"
        assert first_part["text"] == "What is in this image?"

    @patch("src.service.multimedia.image_describer.call_litellm")
    async def test_returns_correct_metadata(self, mock_call):
        """Metadata has correct count, model, and token info."""
        mock_call.return_value = _make_mock_litellm("Description text.")
        messages = [_make_image_msg("https://example.com/img.png")]
        modified, meta = await auto_describe_images(
            messages,
            "gemini/gemini-3.5-flash",
        )
        assert meta["images_described"] == 1
        assert meta["described_by"] == "gemini/gemini-3.5-flash"
        assert meta["total_description_tokens"] == 15
        assert meta["ok"] is True

    @patch("src.service.multimedia.image_describer.call_litellm")
    async def test_handles_detail_level(self, mock_call):
        """detail parameter is passed through to the vision model."""
        mock_call.return_value = _make_mock_litellm("Description.", count=1)
        messages = [_make_image_msg("https://example.com/img.png", detail="high")]
        modified, meta = await auto_describe_images(
            messages,
            "gemini/gemini-3.5-flash",
        )
        assert meta["images_described"] == 1
        mock_call.assert_awaited_once()
        _, kwargs = mock_call.call_args
        msg = kwargs["messages"][0]
        content = msg["content"]
        image_part = next(p for p in content if p.get("type") == "image_url")
        assert image_part["image_url"]["detail"] == "high"
