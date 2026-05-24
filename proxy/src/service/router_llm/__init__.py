"""Router LLM evaluation service.

Sprint 5: Optional task complexity evaluation for expensive pseudo-models.
Suggests downgrade when a task is simple enough for a cheaper model.
"""

from src.service.router_llm.suggester import evaluate_complexity, is_downgrade

__all__ = [
    "evaluate_complexity",
    "is_downgrade",
]
