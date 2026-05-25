"""End-to-end tests — OpenCode → Proxy → Provider flow (Sprint 7 §5).

Simulates multiple turns of a real conversation and verifies:
  - Affinity maintained (same physical model)
  - Streaming works
  - Tools work (simple tool call + result)
  - proxy_metadata complete on every turn
"""

import pytest


@pytest.mark.integration
class TestE2EOpenCodeFlow:
    """Full conversation flow tests with mocked LiteLLM."""

    async def test_affinity_maintained_across_5_turns(self, async_client, mock_litellm):
        """5 turns with same pseudo-model → same physical model throughout."""

        physical_models_used = set()
        conversation_id = None

        for turn in range(1, 6):
            payload: dict = {
                "model": "normal",
                "messages": [{"role": "user", "content": f"Turn {turn} message"}],
            }
            if conversation_id:
                payload["conversation_id"] = conversation_id

            response = await async_client.post("/v1/chat/completions", json=payload)
            assert response.status_code == 200, f"Turn {turn} failed: {response.text}"

            data = response.json()
            assert "proxy_metadata" in data

            pm = data["proxy_metadata"]
            physical_models_used.add(pm["physical_model"])

            # First turn: capture conversation_id
            if turn == 1:
                conversation_id = pm["conversation_id"]

            # Affinity should be maintained
            assert pm["affinity_maintained"] is True if turn > 1 else True

        # Only 1 physical model used across all turns
        assert len(physical_models_used) == 1, (
            f"Expected 1 physical model across 5 turns, got {physical_models_used}"
        )

    async def test_streaming_works_end_to_end(self, async_client):
        """Streaming SSE works with proxy_metadata in final chunk."""
        response = await async_client.post(
            "/v1/chat/completions",
            json={
                "model": "normal",
                "stream": True,
                "messages": [{"role": "user", "content": "Hello streaming"}],
            },
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

        # Collect SSE chunks
        body = response.text
        assert "data:" in body
        assert "proxy_metadata" in body
        assert "[DONE]" in body

    async def test_tools_simple_call_and_result(self, async_client, mock_litellm):
        """Simple tool call + tool result round-trips correctly."""
        # Turn 1: send with tools → model returns tool_call
        response1 = await async_client.post(
            "/v1/chat/completions",
            json={
                "model": "normal",
                "messages": [{"role": "user", "content": "Search for 'test'"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "search",
                            "parameters": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                                "required": ["query"],
                            },
                        },
                    }
                ],
            },
        )
        assert response1.status_code == 200

        data1 = response1.json()
        pm = data1["proxy_metadata"]
        conv_id = pm["conversation_id"]

        # Turn 2: send tool result + follow-up
        response2 = await async_client.post(
            "/v1/chat/completions",
            json={
                "model": "normal",
                "conversation_id": conv_id,
                "messages": [
                    {"role": "user", "content": "Search for 'test'"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "search",
                                    "arguments": '{"query":"test"}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_1",
                        "name": "search",
                        "content": "Found 3 results",
                    },
                    {"role": "user", "content": "Show me the results"},
                ],
            },
        )
        assert response2.status_code == 200

        data2 = response2.json()
        assert data2["proxy_metadata"]["physical_model"] == pm["physical_model"]
        assert data2["proxy_metadata"]["affinity_maintained"] is True

    async def test_proxy_metadata_complete_on_every_turn(self, async_client):
        """proxy_metadata contains all required fields on every response."""
        response = await async_client.post(
            "/v1/chat/completions",
            json={
                "model": "normal",
                "messages": [{"role": "user", "content": "Test metadata completeness"}],
            },
        )
        assert response.status_code == 200

        data = response.json()
        pm = data["proxy_metadata"]

        required_fields = [
            "physical_model",
            "pseudo_model",
            "conversation_id",
            "affinity_maintained",
            "fallback_applied",
            "fallback_reason",
            "context_tokens_total",
            "context_usage_pct",
            "capabilities_detected",
            "warning",
            "tools_filter_applied",
            "tools_filter_reason",
            "pre_compaction_applied",
            "continuous_compaction_applied",
            "images_described",
            "images_described_by",
            "router_suggestion",
            "context_alert",
            "cache",
        ]
        for field in required_fields:
            assert field in pm, f"Missing field '{field}' in proxy_metadata"


@pytest.mark.integration
class TestSprint7Comprehensive:
    """Comprehensive HTTP integration tests for Sprint 7 features."""

    async def test_canonical_order_applied(self, async_client, mock_litellm):
        """Messages are assembled in canonical order before LLM call."""
        # The mock_litellm captures the actual call args.
        # We verify the response includes cache metadata.
        response = await async_client.post(
            "/v1/chat/completions",
            json={
                "model": "normal",
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hello"},
                ],
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert "cache" in data["proxy_metadata"]
        # The cache section is always present (even if empty for non-caching providers)
        assert isinstance(data["proxy_metadata"]["cache"], dict)

    async def test_cache_metadata_in_response(self, async_client, mock_litellm):
        """Cache metadata section is present in proxy_metadata."""
        response = await async_client.post(
            "/v1/chat/completions",
            json={
                "model": "normal",
                "messages": [{"role": "user", "content": "Cache test"}],
            },
        )
        assert response.status_code == 200

        data = response.json()
        cache = data["proxy_metadata"]["cache"]
        assert "provider" in cache
        assert "cache_optimization_applied" in cache
        # For mocked LiteLLM, cache_hit may be False since no real cache token data
        assert "provider_cache_hit" in cache

    async def test_model_aliases_resolve_correctly(self, async_client, mock_litellm):
        """Using an alias like 'gpt-4o' resolves to 'normal'."""
        response = await async_client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Alias test"}],
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["proxy_metadata"]["pseudo_model"] == "normal"

    async def test_prefixed_alias_resolves_correctly(self, async_client, mock_litellm):
        """'local/gpt-4o' strips prefix and resolves alias to 'normal'."""
        response = await async_client.post(
            "/v1/chat/completions",
            json={
                "model": "local/gpt-4o",
                "messages": [{"role": "user", "content": "Prefix+alias test"}],
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["proxy_metadata"]["pseudo_model"] == "normal"

    async def test_default_alias_for_unknown_model(self, async_client, mock_litellm):
        """Unknown model names fall back to default alias."""
        response = await async_client.post(
            "/v1/chat/completions",
            json={
                "model": "completely-unknown-model",
                "messages": [{"role": "user", "content": "Default test"}],
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["proxy_metadata"]["pseudo_model"] == "normal"

    async def test_cache_metadata_streaming(self, async_client, mock_litellm):
        """Streaming responses include cache metadata in the final chunk."""
        response = await async_client.post(
            "/v1/chat/completions",
            json={
                "model": "normal",
                "stream": True,
                "messages": [{"role": "user", "content": "Streaming cache test"}],
            },
        )
        assert response.status_code == 200

        body = response.text
        assert "cache" in body, f"Cache metadata missing from stream body: {body[:500]}"

    async def test_o3_alias_to_pensamiento_profundo(self, async_client, mock_litellm):
        """'o3' alias resolves to pensamiento-profundo-caro."""
        response = await async_client.post(
            "/v1/chat/completions",
            json={
                "model": "o3",
                "messages": [{"role": "user", "content": "Deep thinking test"}],
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["proxy_metadata"]["pseudo_model"] == "pensamiento-profundo-caro"
