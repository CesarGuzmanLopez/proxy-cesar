"""Smart/adaptive fallback with per-model metrics tracking.

FASE 2: Instead of always trying models in order, score each model based on:
- Recent success rate (higher is better)
- Recent error count in last 1h (skip if >3)
- Average latency (lower is better)

Metrics stored in Valkey with 1h TTL, per conversation.
"""

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import valkey.asyncio as valkey_async

logger = logging.getLogger(__name__)


class SmartFallback:
    """Adaptive fallback orchestrator with per-model scoring."""

    def __init__(self, valkey_client: "valkey_async.Valkey") -> None:
        self._client = valkey_client

    async def record_call(
        self,
        conversation_id: str,
        model: str,
        elapsed_ms: int,
        success: bool,
        error: str | None = None,
    ) -> None:
        """Record a call attempt for this model in this conversation.

        Updates metrics: call count, success count, error count, latency.
        """
        key = f"smart_fallback_metrics:{conversation_id}:{model}"

        try:
            metrics_json = await self._client.get(key)
            if metrics_json:
                metrics = json.loads(metrics_json)
            else:
                metrics = {
                    "calls": 0,
                    "successes": 0,
                    "errors_1h": 0,
                    "total_latency_ms": 0,
                    "last_error": None,
                }

            # Update counters
            metrics["calls"] = metrics.get("calls", 0) + 1
            if success:
                metrics["successes"] = metrics.get("successes", 0) + 1
            else:
                metrics["errors_1h"] = metrics.get("errors_1h", 0) + 1
                if error:
                    metrics["last_error"] = error[:100]

            metrics["total_latency_ms"] = (
                metrics.get("total_latency_ms", 0) + elapsed_ms
            )

            # Calculate derived metrics
            metrics["success_rate"] = (
                metrics["successes"] / metrics["calls"]
                if metrics["calls"] > 0
                else 0.5  # Default to 0.5 if no calls yet
            )
            metrics["avg_latency_ms"] = (
                metrics["total_latency_ms"] / metrics["calls"]
                if metrics["calls"] > 0
                else 0
            )

            # Store with 1h TTL
            await self._client.set(
                key,
                json.dumps(metrics),
                ex=3600,  # 1 hour
            )
        except Exception as e:
            logger.warning(
                "smart_fallback_record_error model=%s error=%s",
                model,
                str(e)[:100],
            )
