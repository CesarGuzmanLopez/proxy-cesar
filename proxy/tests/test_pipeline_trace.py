"""Tests for pipeline_trace.py — request lifecycle logging."""

import logging
import time

import pytest

from src.service.pipeline_trace import PipelineTrace


@pytest.fixture
def trace():
    """Create a fresh trace for each test."""
    return PipelineTrace.create(
        conversation_id="test-conv-123",
        pseudo_model="normal",
    )


def test_trace_creation():
    """PipelineTrace.create generates unique trace_id and initializes fields."""
    trace1 = PipelineTrace.create("conv1", "normal")
    trace2 = PipelineTrace.create("conv1", "normal")

    assert trace1.trace_id != trace2.trace_id
    assert len(trace1.trace_id) == 8
    assert trace1.conversation_id == "conv1"
    assert trace1.pseudo_model == "normal"
    assert trace1.start_time > 0


def test_proxy_in_log(trace, caplog):
    """proxy_in logs request arrival with correct fields."""
    with caplog.at_level(logging.INFO):
        trace.proxy_in(messages_count=3, tools_count=2, stream=True)

    record = caplog.records[0]
    assert record.name == "src.service.pipeline_trace"
    assert record.getMessage() == "proxy_in"
    assert record.trace_id == trace.trace_id
    assert record.messages == 3
    assert record.tools == 2
    assert record.stream is True


def test_llm_out_log(trace, caplog):
    """llm_out logs request leaving proxy with model details."""
    with caplog.at_level(logging.INFO):
        trace.llm_out(
            physical_model="openai/kimi-k2.5",
            provider="openai",
            estimated_tokens=1024,
        )

    record = caplog.records[0]
    assert record.getMessage() == "llm_out"
    assert record.physical_model == "openai/kimi-k2.5"
    assert record.provider == "openai"
    assert record.estimated_tokens == 1024


def test_llm_in_log(trace, caplog):
    """llm_in logs response from LLM with token and timing info."""
    with caplog.at_level(logging.INFO):
        trace.llm_in(
            physical_model="openai/kimi-k2.5",
            finish_reason="stop",
            input_tokens=100,
            output_tokens=50,
        )

    record = caplog.records[0]
    assert record.getMessage() == "llm_in"
    assert record.finish_reason == "stop"
    assert record.input_tokens == 100
    assert record.output_tokens == 50
    assert hasattr(record, "elapsed_ms")
    assert record.elapsed_ms >= 0


def test_proxy_out_log_streaming(trace, caplog):
    """proxy_out logs streaming response with status code."""
    with caplog.at_level(logging.INFO):
        trace.proxy_out(http_status=200, stream=True, details={"chunks": 42})

    record = caplog.records[0]
    assert record.getMessage() == "proxy_out"
    assert record.http_status == 200
    assert record.stream is True
    assert record.chunks == 42


def test_proxy_out_log_non_streaming(trace, caplog):
    """proxy_out logs non-streaming response with content length."""
    with caplog.at_level(logging.INFO):
        trace.proxy_out(
            http_status=200,
            stream=False,
            details={"content_len": 1024},
        )

    record = caplog.records[0]
    assert record.stream is False
    assert record.content_len == 1024


def test_proxy_out_error(trace, caplog):
    """proxy_out logs error responses correctly."""
    with caplog.at_level(logging.INFO):
        trace.proxy_out(http_status=502, stream=False)

    record = caplog.records[0]
    assert record.http_status == 502


def test_llm_error_log(trace, caplog):
    """llm_error logs provider errors with elapsed time."""
    with caplog.at_level(logging.ERROR):
        trace.llm_error(
            physical_model="openai/kimi-k2.5",
            error="Connection timeout after 30s",
        )

    record = caplog.records[0]
    assert record.levelname == "ERROR"
    assert record.getMessage() == "llm_error"
    assert record.error == "Connection timeout after 30s"
    assert hasattr(record, "elapsed_ms")


def test_persistence_error_log(trace, caplog):
    """persistence_error logs database failures."""
    with caplog.at_level(logging.ERROR):
        trace.persistence_error(error="Unique constraint violation on turn_number")

    record = caplog.records[0]
    assert record.levelname == "ERROR"
    assert record.getMessage() == "persistence_error"
    assert record.error == "Unique constraint violation on turn_number"
    assert record.trace_id == trace.trace_id


def test_elapsed_time_increases(trace, caplog):
    """Elapsed time increases across calls."""
    with caplog.at_level(logging.INFO):
        trace.proxy_in(messages_count=1, tools_count=0, stream=False)
        time.sleep(0.01)  # 10ms
        trace.proxy_out(http_status=200, stream=False)

    logs = caplog.records
    in_elapsed = getattr(logs[0], "elapsed_ms", 0)
    out_elapsed = getattr(logs[1], "elapsed_ms", 0)

    # proxy_out should have more elapsed time than proxy_in
    assert out_elapsed > in_elapsed
    assert out_elapsed >= 10  # At least 10ms


def test_full_pipeline_trace(caplog):
    """Full request pipeline: in → out → in → out."""
    with caplog.at_level(logging.INFO):
        trace = PipelineTrace.create("test-conv", "normal")
        trace.proxy_in(messages_count=2, tools_count=1, stream=True)
        trace.llm_out(
            physical_model="openai/kimi-k2.5",
            provider="openai",
            estimated_tokens=500,
        )
        trace.llm_in(
            physical_model="openai/kimi-k2.5",
            finish_reason="tool_calls",
            input_tokens=450,
            output_tokens=120,
        )
        trace.proxy_out(http_status=200, stream=True, details={"chunks": 15})

    assert len(caplog.records) == 4
    assert caplog.records[0].getMessage() == "proxy_in"
    assert caplog.records[1].getMessage() == "llm_out"
    assert caplog.records[2].getMessage() == "llm_in"
    assert caplog.records[3].getMessage() == "proxy_out"

    # All logs share the same trace_id
    trace_id = caplog.records[0].trace_id
    for record in caplog.records:
        assert record.trace_id == trace_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
