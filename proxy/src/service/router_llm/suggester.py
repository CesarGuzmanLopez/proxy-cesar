"""Router LLM — task complexity evaluation service.

# Feature: Evaluates whether a task is simple enough to be handled by a
cheaper pseudo-model. Always non-blocking — if evaluation fails, the
request continues to the original model unchanged.

Uses a cheap LLM (configurable per pseudo-model) for evaluation.
Temperature=0.0 for deterministic results.

Safety: NEVER evaluates image-only messages (cheap models lack vision).
If the last user message contains only images with no text, returns ``None``.

python.md §3: ``Result`` monad pattern — errors are returned, not raised.
python.md §4: Pure functions, deterministic, no side effects.
"""

import asyncio
import json
import logging

from src.adapters.litellm import call_litellm

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

EVALUATION_PROMPT: str = """\
Evaluate the complexity of this task for an AI model. Consider:
- Does it require deep reasoning, multi-step planning, or creative problem-solving?
- Is it a straightforward question, simple search, or basic code generation?
- Would a cheap/fast model handle it well, or does it need an expensive reasoning model?

Task:
{task_content}

Respond ONLY with valid JSON (no markdown, no extra text):
{{
    "complexity": "simple" | "medium" | "complex",
    "suggested_pseudo_model": "flash" | "normal" | "tareas-avanzadas" | "pensamiento-profundo-caro",
    "reason": "one sentence explaining why"
}}
"""

MAX_EVAL_TOKENS: int = 200
"""Maximum tokens for the evaluation response (small JSON)."""

MAX_TASK_CHARS: int = 2000
"""Truncate task content to this many characters before evaluation."""

ALLOWED_SUGGESTIONS: set[str] = {
    "pensamiento-profundo-caro",
    "tareas-avanzadas",
    "normal",
    "vision",
    "normal-gratis",
    "flash",
    "compactador",
}
"""All 7 pseudo-models can be suggested by the router."""


# ── Public API ─────────────────────────────────────────────────────────────────


async def evaluate_complexity(
    messages: list[dict],
    suggester_model: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
) -> dict | None:
    """Evaluate task complexity using a cheap LLM evaluator.

    Only evaluates the **last user message** (not full history) for speed.
    This is a NON-BLOCKING advisory — failures return ``None`` and the
    request continues to the target model unchanged.

    **Image safety:** If the last user message contains only images (no text),
    evaluation is skipped entirely — cheap models don't support vision and
    classifying images without context is meaningless.

    Args:
        messages: Full message list (only last user message used).
        suggester_model: Physical model ID for the evaluator
            (e.g. ``zai/glm-4.5-flash``). If ``None``, returns ``None``.
        api_base: Custom API base URL (e.g. for OpenCode Go models).
        api_key: Custom API key (resolved from api_key_env).

    Returns:
        Dict with keys ``complexity``, ``suggested``, ``reason``.
        ``None`` if evaluation fails or no user message found.
    """
    if suggester_model is None:
        return None

    last_user_content: str | None = _extract_last_user_content(messages)
    if not last_user_content:
        return None

    prompt = EVALUATION_PROMPT.format(task_content=last_user_content)

    try:
        response = await asyncio.wait_for(
            call_litellm(
                model=suggester_model,
                messages=[{"role": "user", "content": prompt}],
                api_base=api_base,
                api_key=api_key,
                max_tokens=MAX_EVAL_TOKENS,
                temperature=0.0,
            ),
            timeout=10.0,  # Fast timeout — router is advisory only
        )
        content: str = response.choices[0].message.content or ""
        result: dict = _parse_evaluation_response(content)

        if not result:
            return None

        suggested = result.get("suggested_pseudo_model")
        if suggested not in ALLOWED_SUGGESTIONS:
            suggested = None

        return {
            "complexity": result.get("complexity", "unknown"),
            "suggested": suggested,
            "reason": result.get("reason", ""),
            "source": "llm",
        }
    except Exception as exc:
        logger.warning("router_llm_evaluation_error: %s", exc)
        return None


def is_downgrade(
    suggested: str,
    current: str,
    config,
) -> bool:
    """Check if the suggested pseudo-model is a downgrade from current.

    Uses config-driven tier computation instead of a hardcoded dict.
    A model is a "downgrade" if its computed tier is lower than the
    current model's tier.

    Args:
        suggested: Suggested pseudo-model name.
        current: Current pseudo-model name.
        config: ``ProxyConfigSchema`` with all pseudo-model definitions.

    Returns:
        ``True`` if suggested is strictly less capable than current.
    """
    current_tier: int = _compute_tier(current, config)
    suggested_tier: int = _compute_tier(suggested, config)
    return suggested_tier < current_tier


# ── Internal helpers ───────────────────────────────────────────────────────────


def _extract_last_user_content(messages: list[dict]) -> str | None:
    """Extract the text content from the last user message.

    Handles both plain text (``str``) and multimodal (``list[dict]``) content.

    **Image safety:** If the last user message contains only image parts
    with no text parts, returns ``None`` so the router skips evaluation.
    This prevents sending image data to non-vision evaluator models.

    Returns:
        Truncated text content (max ``MAX_TASK_CHARS`` chars).
        ``None`` if no user message or no text content found.
    """
    last_user_msg = _find_last_user_message(messages)
    if last_user_msg is None:
        return None

    content = last_user_msg.get("content")

    if isinstance(content, str):
        return _extract_text_from_string(content)

    if isinstance(content, list):
        return _extract_text_from_multimodal(content)

    if content is not None:
        return str(content)[:MAX_TASK_CHARS]

    return None


def _find_last_user_message(messages: list[dict]) -> dict | None:
    """Find the last message with role='user'."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg
    return None


def _extract_text_from_string(content: str) -> str | None:
    """Extract text from a plain string content."""
    text = content.strip()
    if not text:
        return None
    return text[:MAX_TASK_CHARS]


def _extract_text_from_multimodal(content: list) -> str | None:
    """Extract text parts from a multimodal content list.

    Returns ``None`` if the message contains only images (no text).
    """
    text_parts: list[str] = []

    for p in content:
        if not isinstance(p, dict):
            continue
        if p.get("type") == "text":
            text = str(p.get("text", "")).strip()
            if text:
                text_parts.append(text)

    if not text_parts:
        return None

    return " ".join(text_parts)[:MAX_TASK_CHARS]


def _parse_evaluation_response(content: str) -> dict | None:
    """Parse JSON from the evaluator model response.

    Handles potential markdown code fences around the JSON.
    Returns ``None`` if parsing fails.
    """
    cleaned: str = content.strip()

    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        )
        cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def _compute_tier(pseudo_model_name: str, config) -> int:
    """Compute a capability tier from pseudo-model config properties.

    Higher tier = more expensive/capable model.
    Factors:
    - ``context_window`` (base score)
    - ``input_token_threshold`` (scale)
    - ``tools_strict`` models (premium +5)
    - ``vision`` models (premium +3)

    This is deterministic and config-driven — no hardcoded tiers.
    """
    pm = config.pseudo_models.get(pseudo_model_name)
    if pm is None:
        return 0

    score: int = 0

    if pm.context_window:
        score += pm.context_window // 10_000

    if pm.input_token_threshold:
        score += pm.input_token_threshold // 10_000

    if any(getattr(m, "tools_strict", False) for m in pm.physical_models):
        score += 5

    if any(getattr(m, "vision", False) for m in pm.physical_models):
        score += 3

    return score
