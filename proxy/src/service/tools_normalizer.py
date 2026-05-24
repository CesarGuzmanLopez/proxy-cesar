"""Tool normalization: serialize parallel tool calls to sequential.

plan-proxy.md §6.8: POST /conversations/{id}/normalize-tools.
Converts parallel tool calls into sequential calls so the conversation
can be migrated to a pseudo-model without parallel_tools support.

python.md §4: pure functions, declarative style, immutable data.
The original history is NEVER modified in-place — always a deep copy.
"""

import copy
from dataclasses import dataclass, field


@dataclass
class NormalizationMetadata:
    """Metadata about a normalization operation."""
    turns_serialized: int = 0
    parallel_calls_serialized: int = 0
    affected_turns: list[int] = field(default_factory=list)


def normalize_history(
    messages: list[dict],
) -> tuple[list[dict], NormalizationMetadata]:
    """Convert parallel tool calls to sequential tool calls in message history.

    plan-proxy.md §6.8: Each parallel tool call is split into its own
    assistant+tool pair. Annotations are inserted to mark the original
    parallel call structure.

    Args:
        messages: Full message history (list of OpenAI-format message dicts).

    Returns:
        Tuple of (normalized_messages, NormalizationMetadata).
        Original messages are NEVER modified — a deep copy is returned.

    Rules:
        1. Only modify messages that have >1 tool_call in an assistant message
        2. Keep original tool_call IDs intact
        3. Insert [TOOL_SERIALIZED] annotation messages between serialized groups
        4. Preserve all non-tool content (system, user, text responses)
        5. Original history is NEVER modified in-place — return a deep copy
    """
    normalized: list[dict] = []
    meta = NormalizationMetadata()

    # Build a set of tool_call_ids that belong to parallel groups
    parallel_call_ids: set[str] = set()
    for msg in messages:
        if msg.get("role") == "assistant" and len(msg.get("tool_calls", [])) > 1:
            for tc in msg["tool_calls"]:
                parallel_call_ids.add(tc.get("id", ""))

    skip_tool_results: set[str] = set()

    for i, msg in enumerate(copy.deepcopy(messages)):
        tool_calls = msg.get("tool_calls", [])

        if msg.get("role") == "assistant" and len(tool_calls) > 1:
            # This turn has parallel tool calls — serialize them
            turn_number = i + 1
            meta.turns_serialized += 1
            meta.affected_turns.append(turn_number)
            meta.parallel_calls_serialized += len(tool_calls)

            for idx, tc in enumerate(tool_calls):
                serialized_msg: dict = {
                    "role": "assistant",
                    "content": msg.get("content") if idx == 0 else None,
                    "tool_calls": [tc],
                }
                normalized.append(serialized_msg)

                # Find the corresponding tool result
                tc_id = tc.get("id", "")
                for j in range(i + 1, len(messages)):
                    result_msg = messages[j]
                    if (
                        result_msg.get("role") == "tool"
                        and result_msg.get("tool_call_id") == tc_id
                    ):
                        normalized.append(copy.deepcopy(result_msg))
                        skip_tool_results.add(tc_id)
                        break

                # Insert annotation (except after the last call)
                if idx < len(tool_calls) - 1:
                    normalized.append({
                        "role": "system",
                        "content": (
                            f"[TOOL_SERIALIZED: originally parallel in "
                            f"turn #{turn_number}, call {idx + 1} of "
                            f"{len(tool_calls)}]"
                        ),
                    })

        elif (
            msg.get("role") == "tool"
            and msg.get("tool_call_id", "") in skip_tool_results
        ):
            # This tool result was already placed after its serialized call
            continue

        elif (
            msg.get("role") == "tool"
            and msg.get("tool_call_id", "") in parallel_call_ids
        ):
            # This is a tool result for a parallel call we already handled
            continue

        else:
            # Non-tool message or non-parallel message — pass through
            normalized.append(msg)

    return normalized, meta


def generate_preview(messages: list[dict], meta: NormalizationMetadata) -> str:
    """Generate a human-readable preview of the normalization result.

    Args:
        messages: Original messages (before normalization).
        meta: NormalizationMetadata from the normalization operation.

    Returns:
        A human-readable string describing what was normalized.
    """
    if not meta.affected_turns:
        return "No parallel tool calls found. Nothing to normalize."

    preview_parts: list[str] = []
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant" and len(msg.get("tool_calls", [])) > 1:
            turn_num = i + 1
            n_calls = len(msg["tool_calls"])
            preview_parts.append(
                f"Turn {turn_num}: {n_calls} parallel calls "
                f"→ {n_calls} sequential calls."
            )

    return " ".join(preview_parts)
