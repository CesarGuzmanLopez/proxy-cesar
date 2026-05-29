"""Tests for tool_detector — content delegation, blob processing, image description.

Covers:
- _classify_content_parts: image/audio/file detection
- _build_blob_output: metadata, truncation, filename
- replace_base64_with_blob_refs: full flow
- inject_blob_extraction_guidance: system message injection
- _describe_image_batch: prompt exhaustiveness, max_tokens

python.md §4: Pure functions tested deterministically.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.service.tool_detector import (
    _build_blob_output,
    _classify_content_parts,
    inject_blob_extraction_guidance,
    replace_base64_with_blob_refs,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_image_url_msg(url: str = "data:image/png;base64,iVBORw0KGgo=") -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is this?"},
            {"type": "image_url", "image_url": {"url": url, "detail": "auto"}},
        ],
    }


def _make_base64_image_msg() -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": "Check this"},
            {
                "type": "image",
                "image": "data:image/png;base64,iVBORw0KGgo=" + "A" * 100,
            },
        ],
    }


def _make_text_with_data_url_msg() -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": "data:image/jpeg;base64,/9j/4AAQ="},
        ],
    }


def _make_mixed_msg() -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": "Here are two images"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo=AAAA"}},
            {"type": "text", "text": "And another"},
            {
                "type": "image",
                "image": "data:image/png;base64,iVBORw0KGgo=" + "B" * 100,
            },
        ],
    }


# ── _classify_content_parts ───────────────────────────────────────────────────


class TestClassifyContentParts:
    """10 tests for _classify_content_parts()."""

    def test_image_url_standard_format(self):
        msg = _make_image_url_msg()
        user_text, images, audios, files, others = _classify_content_parts(
            msg["content"]
        )
        assert len(images) == 1
        assert images[0][0]  # hash not empty
        assert images[0][2] == "image/png"  # mime
        assert audios == []
        assert files == []
        assert len(others) == 1  # text part

    def test_image_base64_format(self):
        msg = _make_base64_image_msg()
        user_text, images, audios, files, others = _classify_content_parts(
            msg["content"]
        )
        assert len(images) == 1
        assert images[0][2] == "image/png"
        assert audios == []
        assert files == []

    def test_text_with_data_url(self):
        msg = _make_text_with_data_url_msg()
        user_text, images, audios, files, others = _classify_content_parts(
            msg["content"]
        )
        assert len(images) == 1
        assert images[0][2] == "image/jpeg"
        assert len(others) == 0

    def test_multiple_images(self):
        msg = _make_mixed_msg()
        user_text, images, audios, files, others = _classify_content_parts(
            msg["content"]
        )
        assert len(images) == 2
        assert audios == []
        assert files == []
        assert len(others) == 2  # two text parts

    def test_empty_content(self):
        user_text, images, audios, files, others = _classify_content_parts([])
        assert images == []
        assert audios == []
        assert files == []
        assert others == []

    def test_audio_format(self):
        content = [
            {
                "type": "input_audio",
                "input_audio": {"data": "data:audio/wav;base64,UklGRiQ="},
            }
        ]
        user_text, images, audios, files, others = _classify_content_parts(content)
        assert images == []
        assert len(audios) == 1
        assert files == []

    def test_file_format(self):
        content = [
            {
                "type": "file",
                "file": {"data": "data:application/pdf;base64,JVBERi0="},
            }
        ]
        user_text, images, audios, files, others = _classify_content_parts(content)
        assert images == []
        assert audios == []
        assert len(files) == 1
        assert files[0][2] == "application/pdf"

    def test_tuple_has_filename_field(self):
        msg = _make_base64_image_msg()
        user_text, images, audios, files, others = _classify_content_parts(
            msg["content"]
        )
        assert len(images) == 1
        # The info tuple is (h, raw, mime, sz, filename) — filename is always empty for data URLs
        assert len(images[0]) == 5
        assert images[0][4] == ""

    def test_user_text_extracted(self):
        msg = _make_image_url_msg()
        user_text, images, audios, files, others = _classify_content_parts(
            msg["content"]
        )
        assert user_text == "What is this?"


# ── _build_blob_output ────────────────────────────────────────────────────────


class TestBuildBlobOutput:
    """7 tests for _build_blob_output()."""

    def test_image_blob_with_description(self):
        images = [("abc123", "raw", "image/png", "10", "screenshot.png")]
        descs = ["A login screen with username and password fields."]
        out = _build_blob_output([], images, descs, [], [], [], [])
        assert len(out) == 1
        text = str(out[0]["text"])
        assert "File extracted: image" in text
        assert "10 KB" in text
        assert "Vision model" in text
        assert "A login screen" in text

    def test_audio_blob_with_transcription(self):
        audios = [("def456", "raw", "audio/wav", "50", "recording.wav")]
        aresults = ["Hello world"]
        out = _build_blob_output([], [], [], audios, aresults, [], [])
        assert len(out) == 1
        text = str(out[0]["text"])
        assert "File extracted: audio" in text
        assert "Hello world" in text

    def test_pdf_blob_with_text(self):
        files = [("ghi789", "raw", "application/pdf", "200", "report.pdf")]
        fresults = ["[PDF: 5 pages, 200 KB.\n\nThis is the extracted text.]"]
        out = _build_blob_output([], [], [], [], [], files, fresults)
        assert len(out) == 1
        text = str(out[0]["text"])
        assert "File extracted: document" in text
        assert "PyMuPDF" in text

    def test_empty_description_warning(self):
        images = [("abc123", "raw", "image/png", "10", "")]
        descs = [""]
        out = _build_blob_output([], images, descs, [], [], [], [])
        text = str(out[0]["text"])
        assert "Warning: Content extraction failed" in text

    def test_multiple_blobs(self):
        images = [
            ("img1", "raw", "image/png", "5", "a.png"),
            ("img2", "raw", "image/jpeg", "8", "b.jpg"),
        ]
        descs = ["First image", "Second image"]
        out = _build_blob_output([], images, descs, [], [], [], [])
        assert len(out) == 2
        assert "First image" in str(out[0]["text"])
        assert "Second image" in str(out[1]["text"])

    def test_truncation_respects_file_size(self):
        # 1 KB file -> max ~624 chars
        long_desc = "D" * 2000
        images = [("hash", "raw", "image/png", "1", "small.png")]
        descs = [long_desc]
        out = _build_blob_output([], images, descs, [], [], [], [])
        text = str(out[0]["text"])
        extracted = text.split("\n", 1)[1] if "\n" in text else text
        assert len(extracted) <= 750

    def test_others_preserved(self):
        others = [{"type": "text", "text": "Regular text"}]
        out = _build_blob_output(others, [], [], [], [], [], [])
        assert len(out) == 1
        assert out[0] == others[0]


# ── replace_base64_with_blob_refs ─────────────────────────────────────────────


class TestReplaceBase64WithBlobRefs:
    """4 async tests for replace_base64_with_blob_refs()."""

    @pytest.mark.asyncio
    async def test_no_images_returns_original(self):
        messages: list[dict[str, object]] = [{"role": "user", "content": "Just text"}]
        result = await replace_base64_with_blob_refs(
            messages, conversation_id="test", valkey=MagicMock(), config=None
        )
        assert result == messages

    @pytest.mark.asyncio
    async def test_single_image_replaced(self):
        valkey = AsyncMock()
        valkey.exists = AsyncMock(return_value=False)
        valkey.set = AsyncMock()
        valkey.get = AsyncMock(return_value=None)  # No cache
        messages = [_make_base64_image_msg()]

        with patch(
            "src.service.tool_detector._describe_image_batch",
            new_callable=AsyncMock,
        ) as mock_describe:
            mock_describe.return_value = ["A screenshot"]
            result = await replace_base64_with_blob_refs(
                messages, conversation_id="test", valkey=valkey, config=MagicMock()
            )

        assert len(result) == 1
        content = result[0]["content"]
        assert isinstance(content, list)
        # The original text is preserved + image replaced with text blob
        text_parts = [p for p in content if p.get("type") == "text"]
        assert len(text_parts) == 2  # original text + blob
        assert "A screenshot" in str(text_parts[1]["text"])

    @pytest.mark.asyncio
    async def test_valkey_none_returns_original(self):
        messages = [_make_base64_image_msg()]
        result = await replace_base64_with_blob_refs(
            messages, conversation_id="test", valkey=None, config=None
        )
        assert result == messages

    @pytest.mark.asyncio
    async def test_non_user_messages_unchanged(self):
        valkey = AsyncMock()
        messages: list[dict] = [
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
        ]
        result = await replace_base64_with_blob_refs(
            messages, conversation_id="test", valkey=valkey, config=None
        )
        assert result[0] == messages[0]


# ── inject_blob_extraction_guidance ───────────────────────────────────────────


class TestInjectBlobExtractionGuidance:
    """3 tests for inject_blob_extraction_guidance()."""

    def test_no_blobs_returns_unchanged(self):
        messages = [{"role": "user", "content": "Hello"}]
        result = inject_blob_extraction_guidance(messages)
        assert result == messages

    def test_with_blobs_adds_system_message(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "[File extracted: image\n"

                        ),
                    }
                ],
            }
        ]
        result = inject_blob_extraction_guidance(messages)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "File Content Extraction" in str(result[0]["content"])

    def test_existing_system_no_duplicate(self):
        messages: list[dict] = [
            {"role": "system", "content": "You are helpful."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "[File extracted: image\n"
                            "  source: abc.png"
                        ),
                    }
                ],
            },
        ]
        result = inject_blob_extraction_guidance(messages)
        assert len(result) == 2  # No extra system message added


# ── _describe_image_batch ─────────────────────────────────────────────────────


class TestDescribeImageBatch:
    """3 async tests for _describe_image_batch prompt and parameters."""

    @pytest.mark.asyncio
    async def test_prompt_contains_exhaustive_instructions(self):
        with patch(
            "src.adapters.litellm.call_litellm", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = MagicMock()
            mock_call.return_value.model_dump.return_value = {
                "choices": [{"message": {"content": '["desc1"]'}}]
            }
            from src.service.tool_detector import _describe_image_batch

            # Create a proper PhysicalModel mock with all required attributes
            phys = MagicMock()
            phys.model = "groq/llama-3.2-90b-vision"
            phys.has_vision = True
            phys.api_base = "https://api.groq.com"
            phys.api_key_env = "GROQ_API_KEY"
            phys.context_window = 128000

            config = MagicMock()
            pm = MagicMock()
            pm.physical_models = [phys]
            pm.fallback_chain = []
            pm.input_token_threshold = 10000
            pm.capabilities = MagicMock()
            pm.capabilities.has_vision = True
            pm.capabilities.max_parallel_tools = 0
            pm.capabilities.supports_audio = False
            pm.capabilities.supports_pdf = False
            pm.capabilities.supports_images = True
            pm.context_window = 128000
            config.pseudo_models = {"vision_test": pm}

            await _describe_image_batch(
                [("hash1", "data:image/png;base64,iVBORw0KGgo=")],
                "What is this?",
                config,
            )

            assert mock_call.called
            args, kwargs = mock_call.call_args
            messages = kwargs["messages"]
            system_msg = messages[0]["content"]
            assert "exhaustively" in system_msg
            assert "Extract ALL visible text" in system_msg
            assert "crossed out" in system_msg
            assert "strikethrough" in system_msg

    @pytest.mark.asyncio
    async def test_max_tokens_is_2048_per_image(self):
        with patch(
            "src.adapters.litellm.call_litellm", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = MagicMock()
            mock_call.return_value.model_dump.return_value = {
                "choices": [{"message": {"content": '["desc1", "desc2"]'}}]
            }
            from src.service.tool_detector import _describe_image_batch

            phys = MagicMock()
            phys.model = "groq/llama-3.2-90b-vision"
            phys.has_vision = True
            phys.api_base = "https://api.groq.com"
            phys.api_key_env = "GROQ_API_KEY"
            phys.context_window = 128000

            config = MagicMock()
            pm = MagicMock()
            pm.physical_models = [phys]
            pm.fallback_chain = []
            pm.input_token_threshold = 10000
            pm.capabilities = MagicMock()
            pm.capabilities.has_vision = True
            pm.capabilities.max_parallel_tools = 0
            pm.capabilities.supports_audio = False
            pm.capabilities.supports_pdf = False
            pm.capabilities.supports_images = True
            pm.context_window = 128000
            config.pseudo_models = {"vision_test": pm}

            await _describe_image_batch(
                [
                    ("h1", "data:image/png;base64,iVBORw0KGgo="),
                    ("h2", "data:image/png;base64,iVBORw0KGgo="),
                ],
                "What are these?",
                config,
            )

            assert mock_call.called
            args, kwargs = mock_call.call_args
            assert kwargs["max_tokens"] == 4096  # 2048 * 2 images

    @pytest.mark.asyncio
    async def test_returns_empty_on_no_vision_model(self):
        config = MagicMock()
        config.pseudo_models = {}
        from src.service.tool_detector import _describe_image_batch

        result = await _describe_image_batch(
            [("h1", "data:image/png;base64,abc")],
            "test",
            config,
        )
        assert result == [""]
