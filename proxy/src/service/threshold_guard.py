"""Input threshold guard.

Checks if the input token estimate exceeds the pseudo-model's threshold.
If pre_compaction is enabled, the check is informational (actual compaction
is deferred to feature).
"""

from src.domain.errors import InputExceedsThreshold
from src.domain.types import Err, Ok, Result


def check_input_threshold(
    pseudo_model_name: str,
    input_token_threshold: int | None,
    estimated_tokens: int,
    pre_compaction_enabled: bool = False,
) -> Result[None, InputExceedsThreshold]:
    """Check if input exceeds the pseudo-model's threshold.

    Args:
        pseudo_model_name: Name of the pseudo-model.
        input_token_threshold: Maximum input tokens allowed (None = no limit).
        estimated_tokens: Estimated token count for the input.
        pre_compaction_enabled: If True, excess is handled later (feature).

    Returns:
        Ok(None) if within threshold or pre_compaction handles it.
        Err(InputExceedsThreshold) if exceeded and no pre_compaction.
    """
    if input_token_threshold is None:
        # No limit (e.g., compactador)
        return Ok(None)

    if estimated_tokens > input_token_threshold:
        if pre_compaction_enabled:
            # Pre-compaction will handle it (feature)
            return Ok(None)
        return Err(
            InputExceedsThreshold(
                estimated=estimated_tokens,
                threshold=input_token_threshold,
                pseudo_model=pseudo_model_name,
            )
        )

    return Ok(None)
