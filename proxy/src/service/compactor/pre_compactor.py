"""Pre-compaction service for Sprint 4.

plan-proxy.md §9.2: When input exceeds threshold, a cheap compactor model
summarizes it before the expensive model sees it.

python.md §7: async-first, uses call_litellm for HTTP calls.
python.md §4: pure functions where possible.
"""

import json

from src.adapters.litellm import call_litellm
from src.config.pseudo_models import PseudoModelSchema
from src.service.capability_detector import estimate_tokens
from src.service.compactor.prompts import build_pre_compaction_prompt


async def pre_compact_input(
    messages: list[dict],
    pseudo_model: PseudoModelSchema,
    config,
) -> tuple[list[dict], dict]:
    """Pre-compact the input using the configured compactor pseudo-model.

    Only modifies the LAST user message. System messages, tool history,
    and assistant messages are passed through unchanged.

    Args:
        messages: The original messages array from the request.
        pseudo_model: The target pseudo-model (with pre_compaction config).
        config: The proxy config (to resolve compactor pseudo-model).

    Returns:
        Tuple of (modified_messages, compaction_metadata).
        If no compaction needed or compactor fails, returns original messages
        with metadata explaining why.

    Metadata keys:
        - applied: bool
        - reason: str (if not applied)
        - original_input_tokens: int
        - compacted_input_tokens: int (if applied)
        - compactor_model: str (if applied)
        - compactor_pseudo_model: str (if applied)
        - savings_tokens: int (if applied)
        - warning: str (if compactor failed)
    """
    threshold = pseudo_model.pre_compaction.threshold
    target_tokens = pseudo_model.pre_compaction.target_tokens
    compactor_name = pseudo_model.pre_compaction.compactor

    # Estimate input tokens
    input_tokens = estimate_tokens(messages)

    if input_tokens <= (threshold or 0):
        return messages, {
            "applied": False,
            "reason": "below_threshold",
            "original_input_tokens": input_tokens,
        }

    # Find the last user message to compact
    last_user_idx = _find_last_user_message(messages)
    if last_user_idx is None:
        return messages, {
            "applied": False,
            "reason": "no_user_message",
            "original_input_tokens": input_tokens,
        }

    # Resolve compactor model
    compactor_pm = config.pseudo_models.get(compactor_name)
    if compactor_pm is None:
        return messages, {
            "applied": False,
            "reason": f"compactor_pseudo_model_not_found: {compactor_name}",
            "warning": (
                f"Pre-compaction configured with compactor '{compactor_name}' "
                f"but that pseudo-model does not exist. Proceeding with original input."
            ),
            "original_input_tokens": input_tokens,
        }

    if not compactor_pm.physical_models:
        return messages, {
            "applied": False,
            "reason": "compactor_no_physical_models",
            "warning": f"Compactor '{compactor_name}' has no physical models. Proceeding with original input.",
            "original_input_tokens": input_tokens,
        }

    compactor_model = compactor_pm.physical_models[0].model

    # Build compaction prompt
    user_message = messages[last_user_idx]
    user_content = _extract_text_content(user_message)
    compaction_prompt = build_pre_compaction_prompt(
        user_content=user_content,
        target_tokens=target_tokens or 8000,
    )

    # Call compactor
    compaction_messages = [{"role": "user", "content": compaction_prompt}]

    try:
        response = await call_litellm(
            model=compactor_model,
            messages=compaction_messages,
            max_tokens=target_tokens or 8000,
        )
        response_dict = response.model_dump() if hasattr(response, "model_dump") else response
        if isinstance(response_dict, dict):
            choices = response_dict.get("choices", [])
            if choices:
                summary = choices[0].get("message", {}).get("content", "")
            else:
                summary = ""
        else:
            summary = response.choices[0].message.content
        summary_tokens = 0
        if isinstance(response_dict, dict):
            usage = response_dict.get("usage", {})
            summary_tokens = usage.get("completion_tokens", 0) or 0
        if not summary_tokens:
            summary_tokens = estimate_tokens([{"role": "user", "content": summary or ""}])
    except Exception as exc:
        # Compactor failed — pass through original input with warning
        return messages, {
            "applied": False,
            "reason": f"compactor_failed: {exc}",
            "warning": (
                "Pre-compaction failed due to compactor error. "
                "Proceeding with original input."
            ),
            "original_input_tokens": input_tokens,
            "compactor_model": compactor_model,
        }

    # Replace the user message with the summary
    modified = list(messages)
    modified[last_user_idx] = {
        "role": "user",
        "content": (
            f"[Pre-compacted input — original: {input_tokens} tokens, "
            f"compacted by {compactor_name}]\n\n{summary}"
        ),
    }

    metadata = {
        "applied": True,
        "original_input_tokens": input_tokens,
        "compacted_input_tokens": summary_tokens,
        "compactor_model": compactor_model,
        "compactor_pseudo_model": compactor_name,
        "savings_tokens": input_tokens - summary_tokens,
    }

    return modified, metadata


def _find_last_user_message(messages: list[dict]) -> int | None:
    """Find the index of the last user message in the messages array."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            return i
    return None


def _extract_text_content(message: dict) -> str:
    """Extract text content from a message, handling both string and array content."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text", "")
                if text:
                    parts.append(text)
        return "\n".join(parts)
    return str(content)
