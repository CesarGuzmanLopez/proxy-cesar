"""Router LLM — task complexity evaluation service.

Sprint 5: Evaluates whether a task is simple enough to be handled by a
cheaper pseudo-model. Always non-blocking — if evaluation fails, the
request continues to the original model unchanged.

Two evaluation strategies (configurable):
1. **LLM-based** (default): Uses ``call_litellm`` with a cheap model.
   Temperature=0.0 for deterministic evaluation.
2. **BERT classifier** (optional): Loads a lightweight ONNX model at startup
   for millisecond-latency evaluation with zero API cost.

Safety: NEVER evaluates image-only messages (cheap models don't support vision).
If the last user message contains only images with no text, returns ``None``.

python.md §3: ``Result`` monad pattern — errors are returned, not raised.
python.md §4: Pure functions, deterministic, no side effects.
"""

import json
import logging
import os
from pathlib import Path

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
    "suggested_pseudo_model": "flash-lowcost" | "normal" | "tareas-avanzadas" | "pensamiento-profundo-caro",
    "reason": "one sentence explaining why"
}}
"""
"""Prompt for the evaluator model. Asks for structured JSON output only."""

MAX_EVAL_TOKENS: int = 200
"""Maximum tokens for the evaluation response (small JSON)."""

MAX_TASK_CHARS: int = 2000
"""Truncate task content to this many characters before evaluation."""

ALLOWED_SUGGESTIONS: set[str] = {
    "flash-lowcost",
    "normal",
    "tareas-avanzadas",
    "pensamiento-profundo-caro",
}
"""Only these pseudo-models can be suggested by the router."""

BERT_MODEL_PATH: str = os.getenv(
    "ROUTER_BERT_MODEL_PATH",
    str(Path(__file__).resolve().parent / "models" / "router-classifier.onnx"),
)
"""Path to an ONNX BERT classifier for fast local evaluation.
Set env var ``ROUTER_BERT_MODEL_PATH`` to enable BERT-based routing.
If unset or the file is missing, falls back to LLM-based evaluation.
"""


# ── BERT classifier (loaded at startup, optional) ──────────────────────────────

_bert_session = None
"""Optional ONNX Runtime session loaded at startup.
``None`` if BERT routing is not configured.
"""

_bert_labels: list[str] = ["simple", "medium", "complex"]
"""Classification labels matching the BERT model output."""


def load_bert_classifier(model_path: str = BERT_MODEL_PATH) -> bool:
    """Load BERT ONNX classifier at startup.

    Called once during FastAPI lifespan startup (``main.py``).
    If the model file doesn't exist, silently falls back to LLM-based routing.

    Args:
        model_path: Path to the ONNX model file.

    Returns:
        ``True`` if the BERT model was loaded successfully.
        ``False`` if the file is missing or loading failed (fallback to LLM).
    """
    global _bert_session

    model_file = Path(model_path)
    if not model_file.exists():
        logger.info(
            "BERT router model not found at %s. "
            "Falling back to LLM-based routing. "
            "Set ROUTER_BERT_MODEL_PATH to enable.",
            model_path,
        )
        return False

    try:
        import onnxruntime as ort  # type: ignore[import-untyped]

        _bert_session = ort.InferenceSession(
            str(model_file),
            providers=["CPUExecutionProvider"],
        )
        logger.info(
            "BERT router model loaded from %s. "
            "Using CPUExecutionProvider for inference.",
            model_path,
        )
        return True
    except ImportError:
        logger.warning(
            "onnxruntime not installed. "
            "Install with: pip install onnxruntime. "
            "Falling back to LLM-based routing."
        )
        return False
    except Exception as exc:
        logger.error(
            "Failed to load BERT router model from %s: %s. "
            "Falling back to LLM-based routing.",
            model_path,
            exc,
        )
        return False


def _classify_with_bert(text: str) -> dict | None:
    """Classify task complexity using the loaded BERT model.

    Returns the same dict format as ``evaluate_complexity()``.
    Returns ``None`` if BERT model is not loaded or inference fails.
    """
    if _bert_session is None:
        return None

    try:
        # Tokenize and run inference
        # The model expects: input_ids, attention_mask, token_type_ids
        # We use a simple whitespace tokenizer as a fast proxy.
        # For production, use a proper tokenizer (see note below).
        tokens = text.lower().split()[:128]
        input_ids = [[1] + [hash(t) % 30000 for t in tokens] + [2]]
        attention_mask = [[1] * len(input_ids[0])]

        # Pad to 128
        pad_len = 128 - len(input_ids[0])
        if pad_len > 0:
            input_ids[0] += [0] * pad_len
            attention_mask[0] += [0] * pad_len

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        outputs = _bert_session.run(None, inputs)
        scores = outputs[0][0]

        import numpy as np  # type: ignore[import-untyped]

        predicted_idx = int(np.argmax(scores))
        confidence = float(np.max(scores))

        if confidence < 0.5:
            return None  # Low confidence — defer to LLM

        complexity = _bert_labels[predicted_idx]

        # Map complexity to suggested model
        suggested_map = {
            "simple": "flash-lowcost",
            "medium": "normal",
            "complex": None,  # No suggestion for complex tasks
        }

        return {
            "complexity": complexity,
            "suggested": suggested_map.get(complexity),
            "reason": f"BERT classifier confidence: {confidence:.2f}",
            "source": "bert",
        }
    except Exception as exc:
        logger.debug("BERT classification failed: %s", exc)
        return None


# ── Public API ─────────────────────────────────────────────────────────────────


async def evaluate_complexity(
    messages: list[dict],
    suggester_model: str | None = None,
) -> dict | None:
    """Evaluate task complexity — BERT first, LLM fallback.

    Strategy:
    1. If BERT classifier is loaded → try fast local classification
    2. If BERT is unavailable or returns low confidence → use LLM
    3. If both fail → return ``None`` (non-blocking)

    Only evaluates the **last user message** (not full history) for speed.
    This is a NON-BLOCKING advisory — failures return ``None`` and the
    request continues to the target model unchanged.

    **Image safety:** If the last user message contains only images (no text),
    evaluation is skipped entirely — cheap models don't support vision and
    classifying images without context is meaningless.

    Args:
        messages: Full message list (only last user message used).
        suggester_model: Physical model ID for LLM fallback
            (e.g. ``zai/glm-4.5-flash``). Can be ``None`` if using BERT only.

    Returns:
        Dict with keys ``complexity``, ``suggested``, ``reason``.
        ``None`` if evaluation fails or no user message found.
    """
    # Extract last user message content
    last_user_content: str | None = _extract_last_user_content(messages)
    if not last_user_content:
        # SAFETY: No text content found (e.g., image-only message).
        # Cheap evaluator models don't support vision, so skip evaluation.
        return None

    # Strategy 1: Try BERT classifier (fast local, zero cost)
    bert_result = _classify_with_bert(last_user_content)
    if bert_result is not None:
        return bert_result

    # Strategy 2: Try LLM evaluation (requires suggester_model)
    if suggester_model is None:
        return None

    prompt = EVALUATION_PROMPT.format(task_content=last_user_content)

    try:
        response = await call_litellm(
            model=suggester_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_EVAL_TOKENS,
            temperature=0.0,  # Deterministic
        )
        content: str = response.choices[0].message.content or ""
        result: dict = _parse_evaluation_response(content)

        if not result:
            return None

        # Validate suggested model is in allowed list
        suggested = result.get("suggested_pseudo_model")
        if suggested not in ALLOWED_SUGGESTIONS:
            suggested = None

        return {
            "complexity": result.get("complexity", "unknown"),
            "suggested": suggested,
            "reason": result.get("reason", ""),
            "source": "llm",
        }
    except Exception:
        # Non-blocking: failure → no suggestion
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
        # SAFETY: Message contains only images, no text.
        # Non-vision evaluator models cannot process this.
        return None

    return " ".join(text_parts)[:MAX_TASK_CHARS]


def _parse_evaluation_response(content: str) -> dict | None:
    """Parse JSON from the evaluator model response.

    Handles potential markdown code fences around the JSON.
    Returns ``None`` if parsing fails.
    """
    cleaned: str = content.strip()

    # Remove markdown code fences if present
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
