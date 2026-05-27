"""Tests for canonical message ordering (Sprint 7 §2).

Verifies that messages are assembled in canonical order for cache hit maximization.
"""

from src.adapters.cache.message_ordering import (
    assemble_canonical_messages,
    canonicalize_message_order,
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


# ── canonicalize_message_order (Bug 9: tool result reordering) ──────────


def test_canonicalize_system_first():
    """System messages are always moved to the front."""
    result = canonicalize_message_order([
        {"role": "user", "content": "Hi"},
        {"role": "system", "content": "You are helpful."},
        {"role": "assistant", "content": "Hello!"},
    ])
    assert result[0]["role"] == "system"
    assert result[0]["content"] == "You are helpful."
    assert len(result) == 3


def test_canonicalize_multiple_systems():
    """Multiple system messages maintain relative order at front."""
    result = canonicalize_message_order([
        {"role": "user", "content": "Hi"},
        {"role": "system", "content": "System A"},
        {"role": "assistant", "content": "Hello"},
        {"role": "system", "content": "System B"},
    ])
    assert result[0]["role"] == "system"
    assert result[0]["content"] == "System A"
    assert result[1]["role"] == "system"
    assert result[1]["content"] == "System B"
    assert len(result) == 4


def test_canonicalize_no_change_already_system_first():
    """When system is already first, order is preserved."""
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "A"},
        {"role": "assistant", "content": "B"},
    ]
    result = canonicalize_message_order(msgs)
    assert result == msgs


def test_canonicalize_tool_result_follows_tool_call():
    """Tool result follows its corresponding tool_call message when it comes after."""
    result = canonicalize_message_order([
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "What's the weather?"},
        {"role": "assistant", "tool_calls": [
            {"id": "call_1", "function": {"name": "get_weather"}},
        ], "content": None},
        {"role": "tool", "tool_call_id": "call_1", "content": '{"temp": 22}'},
    ])
    # Find positions
    positions = {m["role"]: i for i, m in enumerate(result)}
    assert positions["tool"] > positions["assistant"], (
        "Tool result should come after its tool_call message"
    )


def test_canonicalize_tool_result_multiple_calls():
    """Multiple tool results follow their respective tool_call messages."""
    result = canonicalize_message_order([
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Get weather and news"},
        {"role": "assistant", "tool_calls": [
            {"id": "call_1", "function": {"name": "get_weather"}},
            {"id": "call_2", "function": {"name": "get_news"}},
        ], "content": None},
        {"role": "tool", "tool_call_id": "call_1", "content": '{"temp": 22}'},
        {"role": "tool", "tool_call_id": "call_2", "content": '{"headline": "N/A"}'},
    ])
    # Find positions
    asst_idx = next(i for i, m in enumerate(result) if m["role"] == "assistant")
    tool_indices = [i for i, m in enumerate(result) if m["role"] == "tool"]
    for ti in tool_indices:
        assert ti > asst_idx, (
            f"Tool result at {ti} should follow tool_call at {asst_idx}"
        )


def test_canonicalize_tool_result_no_matching_call():
    """Tool result without matching tool_call stays in place (unchanged)."""
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Hi"},
        {"role": "tool", "tool_call_id": "orphan", "content": "data"},
    ]
    result = canonicalize_message_order(msgs)
    # Should still have 3 messages in same order
    assert [m["role"] for m in result] == ["system", "user", "tool"]


def test_canonicalize_preserves_original_order_without_tools():
    """Without tool messages, non-system messages preserve original order."""
    msgs = [
        {"role": "user", "content": "Third"},
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "First"},
        {"role": "assistant", "content": "Second"},
    ]
    result = canonicalize_message_order(msgs)
    # After system at front, remaining messages should be: user(Third), user(First), assistant(Second)
    assert result[1]["content"] == "Third"
    assert result[2]["content"] == "First"
    assert result[3]["content"] == "Second"
