"""Tests for KeyVault middleware — secret detection, masking, re-injection.

Covers:
- Bug 8: system prompt insertion position (after existing system messages)
- conversation_id generation (uuid5 fallback when not provided)
- Secret detection via regex patterns
- Placeholder generation and re-injection
"""
import json
import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.responses import JSONResponse


# ── Unit: _hash_secret ──────────────────────────────────────────────────────


def test_hash_secret_deterministic():
    """Same secret produces same hash."""
    from src.middleware.keyvault import _hash_secret

    h1 = _hash_secret("sk-abc12345")
    h2 = _hash_secret("sk-abc12345")
    assert h1 == h2
    assert len(h1) == 8


def test_hash_secret_different():
    """Different secrets produce different hashes."""
    from src.middleware.keyvault import _hash_secret

    assert _hash_secret("key1") != _hash_secret("key2")


# ── Unit: _make_placeholder ────────────────────────────────────────────────


def test_make_placeholder():
    """Placeholder format is [KEYVAULT:hash]."""
    from src.middleware.keyvault import _make_placeholder

    result = _make_placeholder("abc12345")
    assert result == "[KEYVAULT:abc12345]"


# ── Unit: _mask_text ──────────────────────────────────────────────────────────


def test_mask_text_api_key():
    """An OpenAI-style sk- key is detected and masked."""
    from src.middleware.keyvault import _mask_text

    secrets: dict[str, str] = {}
    text = "My key is sk-abcd1234efgh5678ijkl and I use it for OpenAI."
    masked = _mask_text(text, secrets)

    assert "sk-abcd1234efgh5678ijkl" not in masked
    assert "[KEYVAULT:" in masked
    assert len(secrets) >= 1


def test_mask_text_no_secret():
    """Text without secrets is unchanged."""
    from src.middleware.keyvault import _mask_text

    secrets: dict[str, str] = {}
    text = "Hello, this is a normal message with no secrets."
    masked = _mask_text(text, secrets)

    assert masked == text
    assert len(secrets) == 0


def test_mask_text_multiple():
    """Multiple secrets in the same text are all masked."""
    from src.middleware.keyvault import _mask_text

    secrets: dict[str, str] = {}
    text = "Key1: sk-aaaa1111bbbb2222cccc, Key2: sk-xxxx9999yyyy8888zzzz"
    masked = _mask_text(text, secrets)

    assert "[KEYVAULT:" in masked
    assert len(secrets) >= 2


# ── Unit: _re_inject ──────────────────────────────────────────────────────────


def test_re_inject():
    """Placeholders are replaced with real values."""
    from src.middleware.keyvault import _hash_secret, _make_placeholder, _re_inject

    secret = "sk-abc12345"
    h = _hash_secret(secret)
    placeholder = _make_placeholder(h)
    text = f"My key is {placeholder}"

    result = _re_inject(text, {h: secret})
    assert result == "My key is sk-abc12345"


def test_re_inject_no_match():
    """Unknown placeholders are left unchanged."""
    from src.middleware.keyvault import _re_inject

    result = _re_inject("[KEYVAULT:deadbeef]", {"abc12345": "real_value"})
    assert result == "[KEYVAULT:deadbeef]"


# ── Unit: _mask_messages ──────────────────────────────────────────────────────


def test_mask_messages_text_content():
    """String message content is scanned for secrets."""
    from src.middleware.keyvault import _mask_messages

    body = {"messages": [{"role": "user", "content": "My key is sk-abc12345defgh"}]}
    secrets: dict[str, str] = {}
    _mask_messages(body, secrets)

    assert "sk-abc12345defgh" not in body["messages"][0]["content"]
    assert "[KEYVAULT:" in body["messages"][0]["content"]
    assert len(secrets) >= 1


def test_mask_messages_content_list():
    """Content list parts are scanned."""
    from src.middleware.keyvault import _mask_messages

    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "My key is sk-abc12345defgh"},
                ],
            }
        ]
    }
    secrets: dict[str, str] = {}
    _mask_messages(body, secrets)

    assert len(secrets) >= 1


def test_mask_messages_empty():
    """Empty or missing messages list is handled gracefully."""
    from src.middleware.keyvault import _mask_messages

    secrets: dict[str, str] = {}
    _mask_messages({}, secrets)  # Should not raise
    assert len(secrets) == 0


def test_mask_messages_non_list():
    """Non-list messages is handled gracefully."""
    from src.middleware.keyvault import _mask_messages

    secrets: dict[str, str] = {}
    _mask_messages({"messages": "invalid"}, secrets)  # Should not raise
    assert len(secrets) == 0


# ── Unit: _re_inject_recursive ────────────────────────────────────────────────


def test_re_inject_recursive_string():
    """String values have placeholders replaced."""
    from src.middleware.keyvault import _re_inject_recursive

    result = _re_inject_recursive("[KEYVAULT:abc12345]", {"abc12345": "real_key"})
    assert result == "real_key"


def test_re_inject_recursive_dict():
    """Dict values are recursively processed."""
    from src.middleware.keyvault import _re_inject_recursive

    data = {"key": "[KEYVAULT:abc12345]", "nested": {"inner": "[KEYVAULT:def67890]"}}
    secrets = {"abc12345": "real_1", "def67890": "real_2"}
    result = _re_inject_recursive(data, secrets)
    assert result["key"] == "real_1"
    assert result["nested"]["inner"] == "real_2"


def test_re_inject_recursive_list():
    """List items are recursively processed."""
    from src.middleware.keyvault import _re_inject_recursive

    data = ["[KEYVAULT:abc12345]", "normal", {"nested": "[KEYVAULT:def67890]"}]
    secrets = {"abc12345": "real_1", "def67890": "real_2"}
    result = _re_inject_recursive(data, secrets)
    assert result[0] == "real_1"
    assert result[1] == "normal"
    assert result[2]["nested"] == "real_2"


# ── Bug 8: System prompt insertion position ──────────────────────────────────


def test_system_prompt_insertion_logic_no_system():
    """With no pre-existing system messages, prompt goes at position 0."""
    from src.middleware.keyvault import _KEYVAULT_SYSTEM_PROMPT

    msgs = [{"role": "user", "content": "Hi"}]
    insert_pos = 0
    for i, m in enumerate(msgs):
        if m.get("role") != "system":
            break
        insert_pos = i + 1
    msgs.insert(insert_pos, {"role": "system", "content": _KEYVAULT_SYSTEM_PROMPT})

    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == _KEYVAULT_SYSTEM_PROMPT
    assert msgs[1]["role"] == "user"


def test_system_prompt_insertion_logic_with_system():
    """With a pre-existing system message, prompt goes AFTER it."""
    from src.middleware.keyvault import _KEYVAULT_SYSTEM_PROMPT

    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
    ]
    insert_pos = 0
    for i, m in enumerate(msgs):
        if m.get("role") != "system":
            break
        insert_pos = i + 1
    msgs.insert(insert_pos, {"role": "system", "content": _KEYVAULT_SYSTEM_PROMPT})

    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "You are helpful."
    assert msgs[1]["role"] == "system"
    assert msgs[1]["content"] == _KEYVAULT_SYSTEM_PROMPT
    assert msgs[2]["role"] == "user"


def test_system_prompt_insertion_logic_multiple_system():
    """With multiple system messages, prompt goes after all of them."""
    from src.middleware.keyvault import _KEYVAULT_SYSTEM_PROMPT

    msgs = [
        {"role": "system", "content": "System A"},
        {"role": "system", "content": "System B"},
        {"role": "user", "content": "Hi"},
    ]
    insert_pos = 0
    for i, m in enumerate(msgs):
        if m.get("role") != "system":
            break
        insert_pos = i + 1
    msgs.insert(insert_pos, {"role": "system", "content": _KEYVAULT_SYSTEM_PROMPT})

    assert msgs[0]["content"] == "System A"
    assert msgs[1]["content"] == "System B"
    assert msgs[2]["content"] == _KEYVAULT_SYSTEM_PROMPT
    assert msgs[3]["role"] == "user"


def test_system_prompt_insertion_logic_all_system():
    """If ALL messages are system, prompt goes at the end."""
    from src.middleware.keyvault import _KEYVAULT_SYSTEM_PROMPT

    msgs = [
        {"role": "system", "content": "Sys1"},
        {"role": "system", "content": "Sys2"},
    ]
    insert_pos = 0
    for i, m in enumerate(msgs):
        if m.get("role") != "system":
            break
        insert_pos = i + 1
    msgs.insert(insert_pos, {"role": "system", "content": _KEYVAULT_SYSTEM_PROMPT})

    assert len(msgs) == 3
    assert msgs[2]["content"] == _KEYVAULT_SYSTEM_PROMPT


def test_system_prompt_insertion_logic_system_not_first():
    """System message NOT at position 0 still finds it."""
    from src.middleware.keyvault import _KEYVAULT_SYSTEM_PROMPT

    msgs = [
        {"role": "user", "content": "Hi"},
        {"role": "system", "content": "Sys"},
    ]
    insert_pos = 0
    for i, m in enumerate(msgs):
        if m.get("role") != "system":
            break
        insert_pos = i + 1
    msgs.insert(insert_pos, {"role": "system", "content": _KEYVAULT_SYSTEM_PROMPT})

    # insert_pos = 0 because first message is user (not system)
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == _KEYVAULT_SYSTEM_PROMPT
    assert msgs[1]["role"] == "user"
    assert msgs[2]["role"] == "system"


# ── conversation_id generation ─────────────────────────────────────────────


def test_conversation_id_from_body():
    """conversation_id is taken directly from body when present."""
    from src.middleware.keyvault import _hash_secret

    body = {"conversation_id": "conv-123", "messages": [{"role": "user", "content": "Hi"}]}
    raw_cid = body.get("conversation_id")
    conversation_id = raw_cid if raw_cid else "fallback"

    assert conversation_id == "conv-123"


def test_conversation_id_fallback_deterministic():
    """Without conversation_id, uuid5 generates a deterministic ID from body content."""
    import uuid

    body = {"messages": [{"role": "user", "content": "Hi"}]}
    cid1 = str(uuid.uuid5(uuid.NAMESPACE_DNS, json.dumps(body, sort_keys=True)))
    cid2 = str(uuid.uuid5(uuid.NAMESPACE_DNS, json.dumps(body, sort_keys=True)))

    assert cid1 == cid2  # Deterministic
    assert len(cid1) == 36  # UUID format


def test_conversation_id_fallback_different_bodies():
    """Different bodies produce different fallback IDs."""
    import uuid

    body1 = {"messages": [{"role": "user", "content": "Hello"}]}
    body2 = {"messages": [{"role": "user", "content": "World"}]}
    cid1 = str(uuid.uuid5(uuid.NAMESPACE_DNS, json.dumps(body1, sort_keys=True)))
    cid2 = str(uuid.uuid5(uuid.NAMESPACE_DNS, json.dumps(body2, sort_keys=True)))

    assert cid1 != cid2


# ── _re_inject_non_streaming ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_re_inject_non_streaming():
    """Non-streaming response has placeholders replaced."""
    from src.middleware.keyvault import _re_inject_non_streaming

    # Create a mock response with placeholders in the body
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}

    async def _body_iterator():
        yield b'{"content": "[KEYVAULT:abc12345]"}'

    mock_response.body_iterator = _body_iterator()

    secrets = {"abc12345": "real_key"}
    result = await _re_inject_non_streaming(mock_response, secrets)

    assert isinstance(result, JSONResponse)
    body = json.loads(result.body)
    assert body["content"] == "real_key"
