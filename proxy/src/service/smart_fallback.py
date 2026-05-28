"""Smart/adaptive fallback with per-model metrics tracking.

FASE 2: Instead of always trying models in order, score each model based on:
- Recent success rate (higher is better)
- Recent error count in last 1h (skip if >3)
- Average latency (lower is better)

Metrics stored in Valkey with 1h TTL, per conversation.
"""

import json
import logging
import time
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import valkey.asyncio as valkey_async
    from src.config.pseudo_models import PhysicalModelSchema

logger = logging.getLogger(__name__)


class SmartFallback:
    """Adaptive fallback orchestrator with per-model scoring."""

    def __init__(self, valkey_client: "valkey_async.Valkey") -> None:
        self._client = valkey_client

    async def choose_model(
        self,
        physical_models: list["PhysicalModelSchema"],
        conversation_id: str,
    ) -> "PhysicalModelSchema | None":
        """Choose next model considering recent failures and latency.

        Score = success_rate - recency_penalty - latency_score
        Skip if recent_errors > 3

        Returns the best-scored model, or None if all are bad.
        """
        if not physical_models:
            return None

        scored_models = []

        for phys in physical_models:
            metrics = await self._get_metrics(conversation_id, phys.model)

            # Skip if too many recent errors
            if metrics["errors_1h"] > 3:
                logger.debug(
                    "smart_fallback_skip conv=%s model=%s errors=%d",
                    str(conversation_id)[:12],
                    phys.model,
                    metrics["errors_1h"],
                )
                continue

            # Calculate score
            success_rate = metrics["success_rate"]
            recency_penalty = metrics["errors_1h"] * 0.1  # Penalize recent errors
            latency_score = metrics["avg_latency_ms"] / 1000.0 if metrics["avg_latency_ms"] > 0 else 0
            score = success_rate - recency_penalty - latency_score

            scored_models.append((score, phys))

        if not scored_models:
            # All models are bad, return first as fallback
            logger.warning(
                "smart_fallback_all_bad conv=%s models=%d",
                str(conversation_id)[:12],
                len(physical_models),
            )
            return physical_models[0]

        # Sort by score (highest first)
        scored_models.sort(key=lambda x: x[0], reverse=True)
        best_score, best_model = scored_models[0]

        logger.debug(
            "smart_fallback_choose conv=%s model=%s score=%.2f",
            str(conversation_id)[:12],
            best_model.model,
            best_score,
        )

        return best_model

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

            metrics["total_latency_ms"] = metrics.get("total_latency_ms", 0) + elapsed_ms

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

    async def _get_metrics(
        self,
        conversation_id: str,
        model: str,
    ) -> dict:
        """Get current metrics for a model in a conversation.

        Returns dict with: calls, successes, errors_1h, success_rate, avg_latency_ms
        """
        key = f"smart_fallback_metrics:{conversation_id}:{model}"

        try:
            metrics_json = await self._client.get(key)
            if metrics_json:
                return json.loads(metrics_json)
        except Exception as e:
            logger.warning(
                "smart_fallback_get_metrics_error model=%s error=%s",
                model,
                str(e)[:100],
            )

        # Default metrics for unknown/new models
        return {
            "calls": 0,
            "successes": 0,
            "errors_1h": 0,
            "total_latency_ms": 0,
            "success_rate": 0.5,
            "avg_latency_ms": 0,
            "last_error": None,
        }

    async def reset_metrics(
        self,
        conversation_id: str,
        model: str,
    ) -> None:
        """Reset metrics for a specific model (e.g., after deployment)."""
        key = f"smart_fallback_metrics:{conversation_id}:{model}"
        await self._client.delete(key)
