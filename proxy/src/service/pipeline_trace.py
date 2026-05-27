"""Pipeline tracing: structured logging for request lifecycle.

Tracks a request through 4 checkpoints:
1. proxy_in: request arrives at the proxy
2. llm_out: request leaves proxy → LLM provider
3. llm_in: response arrives from LLM provider
4. proxy_out: response leaves proxy → client

All logs include a trace_id to correlate a single request flow.
"""

import logging
import time
import uuid
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PipelineTrace:
    """Trace a request through the proxy pipeline.

    Generates a unique trace_id and logs at 4 checkpoints with
    consistent structured fields.
    """

    trace_id: str
    conversation_id: str
    pseudo_model: str
    start_time: float

    @classmethod
    def create(cls, conversation_id: str, pseudo_model: str) -> "PipelineTrace":
        """Create a new trace for this request."""
        return cls(
            trace_id=str(uuid.uuid4())[:8],
            conversation_id=conversation_id,
            pseudo_model=pseudo_model,
            start_time=time.time(),
        )

    def proxy_in(
        self,
        messages_count: int,
        tools_count: int,
        stream: bool,
    ) -> None:
        """Log: request arrives at the proxy endpoint."""
        logger.info(
            "proxy_in",
            extra={
                "trace_id": self.trace_id,
                "conversation_id": self.conversation_id,
                "pseudo_model": self.pseudo_model,
                "messages": messages_count,
                "tools": tools_count,
                "stream": stream,
            },
        )

    def llm_out(
        self,
        physical_model: str,
        provider: str | None,
        estimated_tokens: int,
    ) -> None:
        """Log: request leaves proxy → LLM provider."""
        logger.info(
            "llm_out",
            extra={
                "trace_id": self.trace_id,
                "conversation_id": self.conversation_id,
                "physical_model": physical_model,
                "provider": provider or "unknown",
                "estimated_tokens": estimated_tokens,
            },
        )

    def llm_in(
        self,
        physical_model: str,
        finish_reason: str | None,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Log: response arrives from LLM provider."""
        elapsed_ms = int((time.time() - self.start_time) * 1000)
        logger.info(
            "llm_in",
            extra={
                "trace_id": self.trace_id,
                "conversation_id": self.conversation_id,
                "physical_model": physical_model,
                "finish_reason": finish_reason or "unknown",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "elapsed_ms": elapsed_ms,
            },
        )

    def proxy_out(
        self,
        http_status: int,
        stream: bool,
        details: dict | None = None,
    ) -> None:
        """Log: response leaves proxy → client."""
        elapsed_ms = int((time.time() - self.start_time) * 1000)
        extra = {
            "trace_id": self.trace_id,
            "conversation_id": self.conversation_id,
            "http_status": http_status,
            "stream": stream,
            "elapsed_ms": elapsed_ms,
        }
        if details:
            extra.update(details)
        logger.info("proxy_out", extra=extra)

    def llm_error(
        self,
        physical_model: str,
        error: str,
    ) -> None:
        """Log: error during LLM call."""
        elapsed_ms = int((time.time() - self.start_time) * 1000)
        logger.error(
            "llm_error",
            extra={
                "trace_id": self.trace_id,
                "conversation_id": self.conversation_id,
                "physical_model": physical_model,
                "error": error,
                "elapsed_ms": elapsed_ms,
            },
        )

    def persistence_error(
        self,
        error: str,
    ) -> None:
        """Log: error during turn persistence."""
        logger.error(
            "persistence_error",
            extra={
                "trace_id": self.trace_id,
                "conversation_id": self.conversation_id,
                "error": error,
            },
        )
