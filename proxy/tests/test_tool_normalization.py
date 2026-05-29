"""Tests for tool normalization (tools_normalizer.py).

"""

from src.service.tools_normalizer import generate_preview, normalize_history


def _make_parallel_turn(n_calls: int, turn_num: int = 5):
    """Build a parallel tool call turn with n_calls."""
    msg = {
        "role": "assistant",
        "content": "Let me search for that.",
        "tool_calls": [],
    }
    results = []
    for i in range(n_calls):
        cid = f"call_{turn_num}_{i}"
        msg["tool_calls"].append(
            {
                "id": cid,
                "type": "function",
                "function": {"name": "search", "arguments": "{}"},
            }
        )
        results.append(
            {
                "role": "tool",
                "tool_call_id": cid,
                "content": f"Result {i}",
            }
        )
    return msg, results


# ---------------------------------------------------------------------------
# normalize_history
# ---------------------------------------------------------------------------


def test_single_parallel_turn_serialized():
    """Single parallel turn (3 calls) → 3 assistant+tool pairs."""
    msg, results = _make_parallel_turn(3)
    messages = [{"role": "user", "content": "Search"}, msg, *results]
    normalized, meta = normalize_history(messages)
    # Original: 1 user + 1 assistant + 3 tool = 5
    # After: 1 user + (3 assistant + 3 tool + 2 annotation) = 1 + 8 = 9
    # (3 serialized assistant messages + 3 tool results + 2 annotations)
    assert meta.turns_serialized == 1
    assert meta.parallel_calls_serialized == 3
    assert len(normalized) == 9  # 1 user + 3*(assistant+tool) + 2 annotations


def test_multiple_parallel_turns():
    """Multiple parallel turns → all serialized correctly."""
    msg1, res1 = _make_parallel_turn(2, turn_num=1)
    msg2, res2 = _make_parallel_turn(3, turn_num=2)
    messages = [
        {"role": "user", "content": "First"},
        msg1,
        *res1,
        {"role": "user", "content": "Second"},
        msg2,
        *res2,
    ]
    normalized, meta = normalize_history(messages)
    assert meta.turns_serialized == 2
    assert meta.parallel_calls_serialized == 5
    # msg1 at index 1 → turn 2, msg2 at index 5 → turn 6 (1-indexed)
    assert 2 in meta.affected_turns
    assert 6 in meta.affected_turns
    assert len(meta.affected_turns) == 2


def test_annotation_messages_inserted():
    """Annotation messages inserted with correct content."""
    msg, results = _make_parallel_turn(2)
    messages = [{"role": "user", "content": "Hi"}, msg, *results]
    normalized, meta = normalize_history(messages)
    # Find system annotations
    annotations = [m for m in normalized if m.get("role") == "system"]
    assert len(annotations) == 1  # n-1 = 1 for 2 calls
    assert "[TOOL_SERIALIZED:" in annotations[0]["content"]
    assert "call 1 of 2" in annotations[0]["content"]


def test_original_history_not_modified():
    """Original history is not modified (deep copy)."""
    msg, results = _make_parallel_turn(2)
    messages = [{"role": "user", "content": "Hi"}, msg, *results]
    original_len = len(messages)
    normalized, meta = normalize_history(messages)
    assert len(messages) == original_len  # Original unchanged
    assert len(normalized) > original_len  # Normalized is larger
    # Verify original structure intact
    assert len(messages[1]["tool_calls"]) == 2


def test_no_parallel_tools_no_change():
    """No parallel tools → returns same history."""
    messages = [
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant",
            "content": "Hi",
            "tool_calls": [{"id": "call_1", "function": {"arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "Done"},
    ]
    normalized, meta = normalize_history(messages)
    assert meta.turns_serialized == 0
    assert meta.parallel_calls_serialized == 0
    assert len(normalized) == len(messages)


def test_mixed_parallel_and_sequential():
    """Mixed parallel and sequential turns → only parallel modified."""
    seq_msg = {
        "role": "assistant",
        "content": "Single call",
        "tool_calls": [{"id": "call_seq", "function": {"arguments": "{}"}}],
    }
    seq_result = {"role": "tool", "tool_call_id": "call_seq", "content": "Done"}
    par_msg, par_results = _make_parallel_turn(2)
    messages = [
        {"role": "user", "content": "A"},
        seq_msg,
        seq_result,
        {"role": "user", "content": "B"},
        par_msg,
        *par_results,
    ]
    normalized, meta = normalize_history(messages)
    assert meta.turns_serialized == 1
    assert meta.parallel_calls_serialized == 2
    # Count assistant messages
    assistants = [m for m in normalized if m.get("role") == "assistant"]
    # 1 sequential + 2 serialized parallel = 3
    assert len(assistants) == 3


def test_tool_call_ids_preserved():
    """Tool call IDs preserved through serialization."""
    msg, results = _make_parallel_turn(2)
    messages = [{"role": "user", "content": "Hi"}, msg, *results]
    normalized, meta = normalize_history(messages)
    # Each serialized assistant should have one tool_call with original ID
    for m in normalized:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            assert len(m["tool_calls"]) == 1
            assert m["tool_calls"][0]["id"] in ("call_5_0", "call_5_1")


def test_content_only_on_first_serialized():
    """Content text only on first serialized message of each group."""
    msg, results = _make_parallel_turn(3)
    messages = [{"role": "user", "content": "Hi"}, msg, *results]
    normalized, meta = normalize_history(messages)
    serialized = [
        m for m in normalized if m.get("role") == "assistant" and m.get("tool_calls")
    ]
    assert serialized[0]["content"] == "Let me search for that."
    assert serialized[1]["content"] is None
    assert serialized[2]["content"] is None


def test_empty_history():
    """Empty history → no error."""
    normalized, meta = normalize_history([])
    assert normalized == []
    assert meta.turns_serialized == 0


def test_many_parallel_calls():
    """Turn with 10 parallel calls → all serialized correctly."""
    msg, results = _make_parallel_turn(10)
    messages = [{"role": "user", "content": "Go"}, msg, *results]
    normalized, meta = normalize_history(messages)
    assert meta.turns_serialized == 1
    assert meta.parallel_calls_serialized == 10
    # 1 user + 10*(assistant+tool) + 9 annotations = 1 + 20 + 9 = 30
    assert len(normalized) == 30


def test_tool_result_preserved_after_serialization():
    """Tool results preserved exactly (content unchanged)."""
    msg, results = _make_parallel_turn(2)
    messages = [{"role": "user", "content": "Hi"}, msg, *results]
    normalized, meta = normalize_history(messages)
    tool_msgs = [m for m in normalized if m.get("role") == "tool"]
    assert len(tool_msgs) == 2
    assert tool_msgs[0]["content"] == "Result 0"
    assert tool_msgs[1]["content"] == "Result 1"


# ---------------------------------------------------------------------------
# generate_preview
# ---------------------------------------------------------------------------


def test_generate_preview_normal():
    """generate_preview returns readable description."""
    msg, results = _make_parallel_turn(3)
    messages = [{"role": "user", "content": "Hi"}, msg, *results]
    _, meta = normalize_history(messages)
    preview = generate_preview(messages, meta)
    assert "Turn" in preview
    assert "parallel" in preview
    assert "3" in preview


def test_generate_preview_no_parallel():
    """generate_preview with no parallel turns."""
    messages = [{"role": "user", "content": "Hi"}]
    _, meta = normalize_history(messages)
    preview = generate_preview(messages, meta)
    assert "Nothing to normalize" in preview
