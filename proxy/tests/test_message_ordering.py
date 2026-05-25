"""Tests for canonical message ordering (Sprint 7 §2).

Verifies that messages are assembled in canonical order for cache hit maximization.
"""

from src.adapters.cache.message_ordering import (
    assemble_canonical_messages,
    sort_tool_definitions,
    stable_json_dumps,
)


def test_system_prompt_first():
    """System prompt appears first in the assembled messages."""
    messages, _ = assemble_canonical_messages(
        system_prompt="You are a helpful assistant.",
        tool_definitions=None,
        conversation_history=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ],
        new_messages=[{"role": "user", "content": "How are you?"}],
    )
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a helpful assistant."


def test_no_system_prompt():
    """When no system prompt, messages start with history."""
    messages, _ = assemble_canonical_messages(
        system_prompt=None,
        tool_definitions=None,
        conversation_history=[{"role": "user", "content": "Hello"}],
        new_messages=[{"role": "user", "content": "Bye"}],
    )
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello"


def test_conversation_history_chronological_order():
    """History messages appear in the order they were provided (oldest first)."""
    history = [
        {"role": "user", "content": "Turn 1"},
        {"role": "assistant", "content": "Response 1"},
        {"role": "user", "content": "Turn 2"},
    ]
    messages, _ = assemble_canonical_messages(
        system_prompt="sys",
        tool_definitions=None,
        conversation_history=history,
        new_messages=[{"role": "user", "content": "Turn 3"}],
    )
    # History occupies positions 1, 2, 3 (after system at 0)
    assert messages[1]["content"] == "Turn 1"
    assert messages[2]["content"] == "Response 1"
    assert messages[3]["content"] == "Turn 2"


def test_new_messages_appended_at_end():
    """New messages come after history at the tail."""
    messages, _ = assemble_canonical_messages(
        system_prompt=None,
        tool_definitions=None,
        conversation_history=[{"role": "user", "content": "Old"}],
        new_messages=[{"role": "user", "content": "New"}],
    )
    assert messages[-1]["content"] == "New"


def test_idempotent_output():
    """Same input produces identical message order every time."""
    args = dict(
        system_prompt="sys",
        tool_definitions=[
            {"type": "function", "function": {"name": "search", "parameters": {}}},
            {"type": "function", "function": {"name": "fetch", "parameters": {}}},
        ],
        conversation_history=[{"role": "user", "content": "Q"}],
        new_messages=[{"role": "user", "content": "Followup"}],
    )
    msgs1, tools1 = assemble_canonical_messages(**args)
    msgs2, tools2 = assemble_canonical_messages(**args)

    assert len(msgs1) == len(msgs2)
    for a, b in zip(msgs1, msgs2):
        assert a == b
    assert tools1 == tools2


def test_tool_definitions_sorted_alphabetically():
    """Tool definitions are sorted by function name for stable prefix."""
    tools = [
        {"type": "function", "function": {"name": "zebra", "parameters": {}}},
        {"type": "function", "function": {"name": "alpha", "parameters": {}}},
        {"type": "function", "function": {"name": "beta", "parameters": {}}},
    ]
    _, sorted_tools = assemble_canonical_messages(
        system_prompt=None,
        tool_definitions=tools,
        conversation_history=[],
        new_messages=[],
    )
    names = [t["function"]["name"] for t in sorted_tools]
    assert names == ["alpha", "beta", "zebra"]


def test_sort_tool_definitions_idempotent():
    """Sorting already-sorted tools produces the same result."""
    tools = [
        {"type": "function", "function": {"name": "a", "parameters": {}}},
        {"type": "function", "function": {"name": "b", "parameters": {}}},
    ]
    once = sort_tool_definitions(tools)
    twice = sort_tool_definitions(once)
    assert once == twice


def test_stable_json_dumps_sort_keys():
    """stable_json_dumps uses sort_keys=True for deterministic output."""
    obj = {"b": 2, "a": 1}
    result = stable_json_dumps(obj)
    assert result == '{"a": 1, "b": 2}'


def test_messages_not_modified_by_reference():
    """Original messages are deep-copied, not modified."""
    original_system = "sys"
    original_history = [{"role": "user", "content": "Q"}]
    original_new = [{"role": "user", "content": "N"}]

    msgs, _ = assemble_canonical_messages(
        system_prompt=original_system,
        tool_definitions=None,
        conversation_history=original_history,
        new_messages=original_new,
    )
    # Mutate the returned messages
    msgs[0]["content"] = "changed"
    # Original history should be unchanged
    assert original_history[0]["content"] == "Q"
    assert original_new[0]["content"] == "N"


def test_empty_all():
    """Assembling with no system, no tools, no history, no new returns empty."""
    msgs, tools = assemble_canonical_messages(
        system_prompt=None,
        tool_definitions=None,
        conversation_history=[],
        new_messages=[],
    )
    assert msgs == []
    assert tools is None
