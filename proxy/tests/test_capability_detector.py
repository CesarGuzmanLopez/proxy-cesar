"""Tests for capability detection logic.

"""
import pytest

from src.domain.capabilities import TurnCapabilities
from src.service.capability_detector import detect_turn_capabilities, estimate_tokens


def test_text_only_message_no_capabilities():
    """Text-only message → all flags false."""
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    caps = detect_turn_capabilities(messages)
    assert caps == TurnCapabilities()


def test_single_image_url():
    """Single image_url → has_images: true."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/img.jpg"},
                },
            ],
        }
    ]
    caps = detect_turn_capabilities(messages)
    assert caps.has_images is True
    assert caps.has_audio is False
    assert caps.has_tools is False


def test_multiple_images_idempotent():
    """Multiple images → has_images: true (idempotent)."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/1.jpg"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/2.jpg"},
                },
            ],
        }
    ]
    caps = detect_turn_capabilities(messages)
    assert caps.has_images is True


def test_input_audio():
    """input_audio content part → has_audio: true."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Transcribe this:"},
                {
                    "type": "input_audio",
                    "input_audio": {"data": "...", "format": "wav"},
                },
            ],
        }
    ]
    caps = detect_turn_capabilities(messages)
    assert caps.has_audio is True


def test_file_pdf_mime():
    """file with application/pdf mime → has_pdf: true."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "file",
                    "mime_type": "application/pdf",
                    "file_data": "...",
                }
            ],
        }
    ]
    caps = detect_turn_capabilities(messages)
    assert caps.has_pdf is True


def test_file_video_mime():
    """file with video/mp4 mime → has_video: true."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "file",
                    "mime_type": "video/mp4",
                    "file_data": "...",
                }
            ],
        }
    ]
    caps = detect_turn_capabilities(messages)
    assert caps.has_video is True


def test_video_url_type():
    """video_url type → has_video: true."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video_url",
                    "video_url": {"url": "https://example.com/vid.mp4"},
                }
            ],
        }
    ]
    caps = detect_turn_capabilities(messages)
    assert caps.has_video is True


def test_tool_definitions_in_request():
    """Tool definitions in request → has_tools: true."""
    messages = [{"role": "user", "content": "Search the codebase."}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_codebase",
                "description": "Search code",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            },
        }
    ]
    caps = detect_turn_capabilities(messages, tools)
    assert caps.has_tools is True
    assert caps.has_parallel_tools is False


def test_single_tool_call_in_assistant():
    """Single tool_call → has_tools: true, has_parallel_tools: false."""
    messages = [
        {
            "role": "assistant",
            "content": "Let me search.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                }
            ],
        }
    ]
    caps = detect_turn_capabilities(messages)
    assert caps.has_tools is True
    assert caps.has_parallel_tools is False


def test_two_tool_calls_parallel():
    """Two tool_calls → has_tools: true, has_parallel_tools: true."""
    messages = [
        {
            "role": "assistant",
            "content": "Let me search and read.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                },
            ],
        }
    ]
    caps = detect_turn_capabilities(messages)
    assert caps.has_tools is True
    assert caps.has_parallel_tools is True


def test_tool_result_message():
    """Tool result (role: 'tool') → has_tools: true."""
    messages = [
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "Found 3 matches.",
        }
    ]
    caps = detect_turn_capabilities(messages)
    assert caps.has_tools is True


def test_mixed_image_and_tools():
    """Mixed content: image + tools → both flags true."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/img.jpg"},
                },
            ],
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "analyze", "arguments": "{}"},
                },
            ],
        },
    ]
    caps = detect_turn_capabilities(messages)
    assert caps.has_images is True
    assert caps.has_tools is True


def test_empty_messages():
    """Empty messages array → all flags false."""
    caps = detect_turn_capabilities([])
    assert caps == TurnCapabilities()


def test_null_content_handled():
    """Content is null → handled gracefully (no crash)."""
    messages = [{"role": "user", "content": None}]
    caps = detect_turn_capabilities(messages)
    assert caps == TurnCapabilities()


def test_non_standard_fields_ignored():
    """Messages with non-standard fields → ignored."""
    messages = [
        {
            "role": "user",
            "content": "Hello",
            "extra_field": "ignored",
        }
    ]
    caps = detect_turn_capabilities(messages)
    assert caps == TurnCapabilities()


def test_alternate_mime_key():
    """file with mimetype (not mime_type) → still detected."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "file",
                    "mimetype": "application/pdf",
                    "file_data": "...",
                }
            ],
        }
    ]
    caps = detect_turn_capabilities(messages)
    assert caps.has_pdf is True


@pytest.mark.asyncio
async def test_estimate_tokens_empty():
    """Empty messages → at least 1 token."""
    assert await estimate_tokens([]) == 1


@pytest.mark.asyncio
async def test_estimate_tokens_text():
    """Text message counted by tiktoken (includes 4-token overhead)."""
    messages = [{"role": "user", "content": "Hello world"}]
    # tiktoken o200k_base: "Hello"=1 + " world"=1 + 4 overhead = 6
    assert await estimate_tokens(messages) == 6


@pytest.mark.asyncio
async def test_estimate_tokens_multimodal():
    """Multimodal content counts text parts via tiktoken."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/img.jpg"},
                },
            ],
        }
    ]
    # tiktoken o200k_base: text=4 + 4 overhead = 8
    assert await estimate_tokens(messages) == 8
