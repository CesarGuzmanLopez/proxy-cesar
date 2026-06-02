"""Canonical message ordering for deterministic provider cache hits.

# Feature: assemble messages in identical order across turns so the provider
recognizes the unchanged prefix and reuses its cache.

Order:
  1. System messages (static — never change between turns)
  2. Tool definitions (sorted alphabetically by function name for stability)
  3. Conversation history (oldest → newest, as stored in DB)
  4. New user/assistant/tool messages (appended at the end)
"""

import hashlib
import json
from copy import deepcopy


def stable_message_hash(messages: list[dict]) -> str:
    """Compute a deterministic SHA-256 hash of the message list.

    Uses stable_json_dumps so the same logical messages always produce the
    same hash, regardless of dict key ordering or cosmetic differences.

    Useful for:
    - Verifying message integrity between call_with_fallback and _save_and_return
    - Tracking whether cacheable prefixes changed between turns
    - Debugging context corruption
    """
    return hashlib.sha256(stable_json_dumps_bytes(messages)).hexdigest()[:16]


def stable_json_dumps(obj: dict | list) -> str:
    """Serialize to JSON with sorted keys for deterministic output.

    MUST be used everywhere the proxy serializes content that becomes part
    of the provider's cacheable prefix:
      - Tool definitions stored in DB
      - Messages stored in JSONB columns
      - Any dict sent to the provider (via LiteLLM)
    """
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


def stable_json_dumps_bytes(obj: dict | list) -> bytes:
    """Serialize to JSON bytes with sorted keys."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")


def sort_tool_definitions(tools: list[dict]) -> list[dict]:
    """Sort tool definitions alphabetically by function name for stable prefix."""
    return sorted(
        deepcopy(tools),
        key=lambda t: t.get("function", {}).get("name", ""),
    )


# NOTE: Tools are sorted ONCE inside call_with_fallback() (chat_service.py:771-773)
# and the sorted result is preserved in call_kwargs through the entire continuation
# chain. In streaming continuation (api/chat.py:920), ctx.call_kwargs already contains
# the ALREADY-SORTED tools list, so no re-sorting is needed.
# This is NOT a bug.


def canonicalize_message_order(messages: list[dict]) -> list[dict]:
    """Reorder a flat message list into canonical order for provider cache hits.

    Canonical order:
      1. System message(s) — always first (static per pseudo-model)
      2. Remaining messages — in their original order, with tool results
         positioned immediately after the assistant message that contains
         the matching tool_calls.

    The purpose is to ensure the system prompt is always at position 0,
    making the cacheable prefix (system + early history) identical across
    turns. Without canonicalization, a client that places the system message
    at the end (or anywhere other than the start) would break the provider's
    prefix cache on every turn.

    This is a NO-OP (no change) for clients that already send system first.

    Args:
        messages: Flat list of OpenAI-format message dicts.

    Returns:
        New list with system messages moved to the front, and tool results
        reordered to follow their corresponding tool_call messages.
    """
    system_msgs = [m for m in messages if m.get("role") == "system"]
    other_msgs = [m for m in messages if m.get("role") != "system"]

    # Bug 9 fix: ensure tool results follow their corresponding tool_call
    # messages. Without this, interleaved tool results break the provider's
    # cache prefix on re-ordered message lists.
    reordered: list[dict] = []
    pending_tool_ids: set[str] = set()
    for msg in other_msgs:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            reordered.append(msg)
            tool_ids = {
                tc.get("id", "")
                for tc in msg["tool_calls"]
                if isinstance(tc, dict) and tc.get("id")
            }
            pending_tool_ids.update(tool_ids)
        elif msg.get("role") == "tool" and msg.get("tool_call_id") in pending_tool_ids:
            reordered.append(msg)
            pending_tool_ids.discard(msg["tool_call_id"])
        else:
            reordered.append(msg)
    return deepcopy(system_msgs + reordered)
