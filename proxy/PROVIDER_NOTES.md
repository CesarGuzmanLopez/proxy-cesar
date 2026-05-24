# Provider Notes — LiteLLM Tool Translation Verification

> **Last updated:** 2026-05-23
> **Sprint 3 §3:** LiteLLM translation verification for all provider-model combinations.

## Verification Matrix

| Provider | Model | Simple tool | Complex schema | Parallel calls | Streaming + tools | Strict mode | Status |
|---|---|---|---|---|---|---|---|
| DeepSeek | `deepseek-v4-pro` | ✅ | ✅ | ✅ | ✅ | ✅ | ⚪ Not tested |
| DeepSeek | `deepseek-v4-flash` | ✅ | ✅ | ✅ | ✅ | ❌ (not supported) | ⚪ Not tested |
| Google | `gemini-3.5-flash` | ✅ | ✅ | ❌ (partial) | ✅ | ❌ | ⚪ Not tested |
| Zhipu | `glm-5.1` | ✅ | ✅ | ❌ | ✅ | ❌ | ⚪ Not tested |
| Zhipu | `glm-4.5-flash` | ✅ | ⚠️ simple only | ❌ | ✅ | ❌ | ⚪ Not tested |
| Qwen | `qwen3-max` | ✅ | ✅ | ❌ | ✅ | ❌ | ⚪ Not tested |
| Qwen | `qwen3.5-plus` | ✅ | ⚠️ simple only | ❌ | ✅ | ❌ | ⚪ Not tested |
| Groq | `openai/gpt-oss-20b` | ✅ | ✅ | ❌ | ✅ | ❌ | ⚪ Not tested |
| MiniMax | `minimax-m2.5` | ✅ | ⚠️ simple only | ❌ | ✅ | ❌ | ⚪ Not tested |
| Anthropic | `claude-haiku-4-5` | ✅ | ✅ | ❌ | ✅ | ❌ | ⚪ Not tested |
| Ollama | `ollama/llama3.2` | ✅ | ⚠️ simple only | ❌ | ⚠️ | ❌ | ⚪ Not tested |
| Ollama | `ollama/llava` | ✅ | ⚠️ simple only | ❌ | ⚠️ | ❌ | ⚪ Not tested |

**Legend:** ✅ Supported · ❌ Not supported · ⚠️ Limited · ⚪ Pending verification

## How to Run Integration Tests

```bash
# Requires API keys in .env for all providers
.venv/bin/python -m pytest tests/test_tools_e2e.py -v --run-integration
```

Expected cost per full run: ~$1-2 USD.

## Known Issues & Workarounds

### Pending: LiteLLM model name mapping

The `pseudo_models.yaml` uses short names (e.g., `deepseek-v4-pro`, `gemini-3.5-flash`).
LiteLLM requires specific model strings (e.g., `deepseek/deepseek-chat`, `gemini/gemini-2.0-flash`).
The correct LiteLLM model strings must be verified and documented here during integration testing.

### Pending: Streaming + tool tests

Streaming + tools are not yet tested in `test_tools_e2e.py`. Add tests after basic tool
call round-trips are verified.

### Reserved: Zhipu parallel tools

Zhipu GLM models have reported inconsistencies with parallel tool calls in LiteLLM.
If tests fail, flag `parallel_tools: false` in `pseudo_models.yaml` for these models.

## Provider-Specific Notes

Will be filled in as integration tests are run. Each failure should document:
1. Exact error message
2. Whether the issue is in LiteLLM or the provider
3. Any workaround or config change applied
