---
filename: llm-proxy-libraries-summary.md
date: 2026-05-24
---

# Python Libraries Research for FastAPI LLM Proxy

## 1. LLM Context Compaction & Alerts

**No mature standalone library** exists specifically for LLM context window monitoring with alerts.

Closest options:

| Library | License | Notes |
|---------|---------|-------|
| `litellm` (BerriAI) | MIT | Proxy layer with token counting, cost tracking, rate limiting. 1.86.0. Very active. Has built-in token tracking but no "context compaction" feature. Its proxy includes usage monitoring. |
| `langchain` | MIT | Has `token_counter` utilities and callbacks for tracking token usage across providers. Callback system can be used to build alerts. Heavy dependency. |
| `guardrails-ai` | Apache 2.0 | Structured output validation and guardrails. Could be extended for context alerts but not its primary focus. |
| `instructor` | MIT | Pydantic-based structured extraction. Has retry/validation logic. Not directly for context monitoring. |

**Recommendation**: Implement from scratch using `tiktoken` + SDK-level token reporting. ~100 lines of Python. Each major provider SDK (OpenAI, Anthropic, Google) returns usage stats in responses. No specialized library needed.

## 2. Celery Alternatives

| Library | License | Redis Required | Async | Stars | Verdict |
|---------|---------|---------------|-------|-------|---------|
| **arq** (v0.28) | **MIT** | Yes | Native asyncio | ~3k | **Best fit**. Built by Samuel Colvin (Pydantic). Native FastAPI/asyncio compatibility. Small (~700 lines). Redis-backed. |
| **huey** (v3.0) | **MIT** | Optional (SQLite/fs/redis) | Partial (synchronous worker) | ~5k | **Lighter but sync**. Good if you don't want Redis. No native asyncio. |
| **dramatiq** (v2.1) | **LGPLv3+** | Optional (Redis/RabbitMQ) | No | ~4k | **Disqualified** - LGPL license incompatible with your requirements. |
| **taskiq** (v0.12) | **MIT** | Optional (Redis/RabbitMQ/ZMQ) | Full async | ~1.5k | Good async-native option. Has OpenTelemetry/metrics extras. Less mature (Alpha). |

**Recommendation**: **arq** is the clear winner. It's asyncio-native, MIT, lightweight, from the creator of Pydantic (which you already use). Taskiq is a secondary option if you want multiple broker backends.

## 3. Markdown Structure Validation

No existing library validates "this markdown must contain sections X, Y, Z". Options:

| Approach | Library | License | Notes |
|----------|---------|---------|-------|
| Parse + check headings | `markdown-it-py` v4.2 | MIT | Python port of markdown-it. Parse to token stream, then check heading hierarchy with ~20 lines of code. |
| Parse + check structure | Python `markdown` (stdlib) | BSD | Python-Markdown. Parse MD, then use `toc` extension to extract heading structure. |
| AST traversal | `mdformat` / `markdown-it-py` | MIT | Parse to AST, traverse nodes to validate required sections exist. |
| Pydantic validation | Write custom | MIT | Use `markdown-it-py` + Pydantic model for expected structure. |

**Recommendation**: Use `markdown-it-py` (MIT, active, well-maintained) to parse markdown to tokens, then validate section headings programmatically. ~30 lines of code. Not worth a separate library.

## 4. Token Counting Beyond tiktoken

| Library/Approach | Models | License | Notes |
|-----------------|--------|---------|-------|
| **tiktoken** v0.13 | OpenAI models | MIT | BPE tokenizer. Official from OpenAI. Only first-party knowledge. |
| **anthropic SDK** v0.104 | Claude models | MIT | Uses Anthropic's own tokenizer internally. Can count via API (`usage` in response) or use `anthropic.Tokenizer` directly. |
| **google-generativeai** v0.8 | Gemini models | Apache 2.0 | `count_tokens()` method available. |
| **transformers** (AutoTokenizer) | Any model (DeepSeek, Llama, etc.) | Apache 2.0 | Can load any tokenizer from HuggingFace. Universal but heavier (downloads model files). |
| **litellm** | 100+ providers | MIT | Unified token counting with `litellm.token_counter()`. Wraps tiktoken + provider SDKs. |

**Recommendation**: For a proxy, **litellm** gives you the most model coverage. Alternatively, the `anthropic` SDK has a `client.count_tokens()` method. For DeepSeek, you can use `transformers` or simply use OpenAI's tiktoken (DeepSeek uses same tokenizer as GPT). Gemini uses SentencePiece which is included in the `google-generativeai` SDK.

## 5. Monitoring/Metrics for FastAPI

| Library | License | Notes |
|---------|---------|-------|
| **prometheus-fastapi-instrumentator** v7.1 | ISC | Auto-instruments FastAPI with request count, duration, histograms. ISC ≈ MIT. Very mature. |
| **prometheus-client** v0.25 | Apache 2.0 | Low-level Prometheus client. Build custom metrics. |
| **opentelemetry-api** + **opentelemetry-instrumentation-fastapi** | Apache 2.0 | Distributed tracing, metrics, logging. More complex but industry standard. |

**Recommendation**: Use **prometheus-fastapi-instrumentator** (ISC license, equivalent to MIT) for auto-instrumentation, plus **prometheus-client** for custom metrics (context usage rates, compaction events, token counts by model). This combination is mature and battle-tested.
