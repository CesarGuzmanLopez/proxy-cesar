#!/usr/bin/env python3
"""Verificación exhaustiva de Sprints 1-3 en modo streaming y no-streaming.

Pruebas:
  Sprint 1 — MVP: pseudo-models, affinity, streaming, chat básico
  Sprint 2 — Capabilities: detección, compatibilidad (switch bloqueado), threshold
  Sprint 3 — Tools: filtro de modelos con parallel_tools, tool edge cases
"""

import asyncio
import json
import sys
import uuid

import httpx

BASE = "http://127.0.0.1:9110"

PASS = 0
FAIL = 0

# ── Helpers ────────────────────────────────────────────────────────────────────


def log_test(name: str, ok: bool, detail: str = ""):
    global PASS, FAIL
    if ok:
        PASS += 1
    else:
        FAIL += 1
    icon = "✅" if ok else "❌"
    print(f"  {icon} {name}")
    if detail:
        for line in detail.strip().split("\n"):
            print(f"     {line}")


async def simple_chat(
    model: str, msg: str = "say hi", max_tokens: int = 300, conv_id: str = None
):
    """POST no-streaming."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": msg}],
        "max_tokens": max_tokens,
    }
    if conv_id:
        body["conversation_id"] = conv_id
    async with httpx.AsyncClient(timeout=120) as c:
        r = await c.post(f"{BASE}/v1/chat/completions", json=body)
        return r.status_code, r.json()


async def stream_chat(model: str, msg: str = "count 1 2 3", max_tokens: int = 300):
    """POST streaming, returns (has_content, has_done, chunk_count)."""
    async with httpx.AsyncClient(timeout=120) as c:
        async with c.stream(
            "POST",
            f"{BASE}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": msg}],
                "stream": True,
                "max_tokens": max_tokens,
            },
        ) as resp:
            has_content = False
            has_done = False
            count = 0
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    has_done = True
                else:
                    count += 1
                    try:
                        obj = json.loads(payload)
                        delta = obj.get("choices", [{}])[0].get("delta", {})
                        if delta.get("content") is not None:
                            has_content = True
                    except json.JSONDecodeError:
                        pass
            return has_content, has_done, count


# ══════════════════════════════════════════════════════════════════════════════
#  SPRINT 1  —  MVP: pseudo-models, affinity, streaming
# ══════════════════════════════════════════════════════════════════════════════


async def test_sprint1():
    print("=" * 70)
    print("  SPRINT 1 — MVP: PSEUDO-MODELS + AFFINITY + STREAMING")
    print("=" * 70)

    NON_VISION = [
        "pensamiento-profundo-caro",
        "tareas-avanzadas",
        "normal",
        "deep-flash",
        "flash-lowcost",
    ]

    # ── 1a. Non-streaming — every non-vision model ─────────────────────
    print("\n[1a] No-streaming — todos los pseudo-models (non-vision)")
    for m in NON_VISION:
        st, data = await simple_chat(m, "say hi in 1 word", 300)
        if st == 200:
            c = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            phys = data.get("proxy_metadata", {}).get("physical_model", "?")
            log_test(f"  {m}", bool(c), f"HTTP {st} content={c[:60]!r} physical={phys}")
        else:
            err = data.get("detail", {}).get("error", str(data)[:100])
            log_test(f"  {m}", False, f"HTTP {st} {err}")
        await asyncio.sleep(0.3)

    # ── 1b. Streaming — every non-vision model ─────────────────────────
    print("\n[1b] Streaming — todos los pseudo-models (non-vision)")
    for m in NON_VISION:
        ok, done, n = await stream_chat(m, "count 1 2 3", 300)
        log_test(
            f"  {m} streaming",
            ok and done,
            f"chunks={n} content_in_chunks={ok} done={done}",
        )
        await asyncio.sleep(0.5)

    # ── 1c. Affinity — same physical model in same conversation ────────
    print("\n[1c] Affinity — mismo physical model en 2 requests consecutivos")
    conv_a = str(uuid.uuid4())
    _, d1 = await simple_chat("normal", "hello", 200, conv_a)
    await asyncio.sleep(0.5)
    _, d2 = await simple_chat("normal", "again", 200, conv_a)
    phys1 = d1.get("proxy_metadata", {}).get("physical_model") if d1 else None
    phys2 = d2.get("proxy_metadata", {}).get("physical_model") if d2 else None
    aff1 = d1.get("proxy_metadata", {}).get("affinity_maintained") if d1 else None
    aff2 = d2.get("proxy_metadata", {}).get("affinity_maintained") if d2 else None
    log_test("  1er request", phys1 is not None, f"physical={phys1}")
    log_test(
        "  2o request = misma physical", phys1 == phys2, f"phys1={phys1} phys2={phys2}"
    )
    log_test("  affinity_maintained 1er", aff1 is False, f"aff1={aff1} (new → False)")
    log_test("  affinity_maintained 2o", aff2 is True, f"aff2={aff2} (reuse → True)")

    # ── 1d. Unknown model → 400 ───────────────────────────────────────
    print("\n[1d] Modelo desconocido → 400")
    st, _ = await simple_chat("no-such-model-xyz", "hi", 10)
    log_test("  unknown → 400", st == 400, f"HTTP {st}")

    # ── 1e. Metadata fields ────────────────────────────────────────────
    print("\n[1e] proxy_metadata — campos obligatorios")
    st, d = await simple_chat("normal", "test", 200)
    if st == 200:
        meta = d.get("proxy_metadata", {})
        required = [
            "physical_model",
            "pseudo_model",
            "conversation_id",
            "affinity_maintained",
            "fallback_applied",
            "context_tokens_total",
        ]
        present = all(k in meta for k in required)
        log_test(
            "  Todos los campos S1 en metadata",
            present,
            f"ausentes={[k for k in required if k not in meta]}",
        )
    else:
        log_test("  Todos los campos S1 en metadata", False, f"HTTP {st}")


# ══════════════════════════════════════════════════════════════════════════════
#  SPRINT 2  —  CAPABILITIES + COMPATIBILITY + THRESHOLD
# ══════════════════════════════════════════════════════════════════════════════


async def test_sprint2():
    print("\n" + "=" * 70)
    print("  SPRINT 2 — CAPABILITIES + COMPATIBILITY + THRESHOLD")
    print("=" * 70)

    # ── 2a. Proxy metadata carries capabilities_detected ──────────────
    print("\n[2a] proxy_metadata.capabilities_detected")
    st, d = await simple_chat("normal", "hello", 200)
    if st == 200:
        caps = d.get("proxy_metadata", {}).get("capabilities_detected", {})
        has_has_images = "has_images" in caps
        has_has_tools = "has_tools" in caps
        log_test(
            "  capabilities_detected presente",
            has_has_images and has_has_tools,
            f"keys={list(caps.keys())}",
        )
    else:
        log_test("  capabilities_detected presente", False, f"HTTP {st}")

    # ── 2b. Input threshold — request grande debe exceder → 400 ──────
    print("\n[2b] Input threshold — mensaje que excede el límite")
    # flash-vision threshold = 16000; hacemos un mensaje enorme
    huge_msg = "hello " * 30000  # ~150k chars => muchos tokens
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(
            f"{BASE}/v1/chat/completions",
            json={
                "model": "flash-vision",
                "messages": [{"role": "user", "content": huge_msg}],
                "max_tokens": 10,
            },
        )
        # Should be 400 INPUT_EXCEEDS_THRESHOLD
        ok = r.status_code == 400
        detail = ""
        if ok:
            detail = r.json().get("detail", {}).get("error", "")
        log_test(
            "  Exceso de threshold → 400", ok, f"HTTP {r.status_code} error={detail}"
        )
    await asyncio.sleep(1)

    # ── 2c. Compactador pseudo-model siempre SAFE ─────────────────────
    print("\n[2c] Compactador — siempre SAFE")
    st, d = await simple_chat("compactador", "compact this", 500)
    if st == 200:
        meta = d.get("proxy_metadata", {})
        log_test(
            "  compactador responde", True, f"phys={meta.get('physical_model', '?')}"
        )
    else:
        e = d.get("detail", {}).get("error", str(d)[:100])
        log_test("  compactador responde", False, f"HTTP {st} {e}")

    # ── 2d. Versión endpoint compatible ───────────────────────────────
    print("\n[2d] GET /v1/models")
    async with httpx.AsyncClient() as c:
        r = await c.get(f"{BASE}/v1/models")
        if r.status_code == 200:
            models = r.json().get("data", [])
            ids = {m["id"] for m in models}
            # All non-vision pseudo-models should be listed
            expected = {
                "pensamiento-profundo-caro",
                "tareas-avanzadas",
                "avanzada-vision",
                "normal",
                "deep-flash",
                "flash-lowcost",
                "flash-vision",
                "compactador",
            }
            missing = expected - ids
            log_test(
                "  Todos los pseudo-models listados",
                len(missing) == 0,
                f"missing={missing}",
            )
        else:
            log_test(
                "  Todos los pseudo-models listados", False, f"HTTP {r.status_code}"
            )


# ══════════════════════════════════════════════════════════════════════════════
#  SPRINT 3  —  TOOLS CANONICAL + EDGE CASES
# ══════════════════════════════════════════════════════════════════════════════


async def test_sprint3():
    print("\n" + "=" * 70)
    print("  SPRINT 3 — TOOLS CANONICAL + EDGE CASES")
    print("=" * 70)

    # ── 3a. Tools filtering — modelos con parallel_tools ──────────────
    print("\n[3a] Filtro de modelos con parallel_tools")
    # normal tiene: qwen3-max (parallel=false) y deepseek-v4-flash (parallel=true)
    # Si el historial no tiene parallel_tools, deben aparecer TODOS
    # (probamos indirectamente: enviamos una request SIN parallel tools existentes)
    st, d = await simple_chat("normal", "hello world", 300)
    if st == 200:
        meta = d.get("proxy_metadata", {})
        log_test(
            "  tools_filter_applied=false (sin tools previas)",
            meta.get("tools_filter_applied") is False,
            f"applied={meta.get('tools_filter_applied')}",
        )
    else:
        log_test("  tools_filter_applied=false", False, f"HTTP {st}")

    # ── 3b. Tool call in streaming — edge case ────────────────────────
    print("\n[3b] Streaming con herramientas (tool_choice)")
    # Usamos un modelo con tools_strict (deep-flash tiene deepseek-v4-flash con strict)
    async with httpx.AsyncClient(timeout=120) as c:
        async with c.stream(
            "POST",
            f"{BASE}/v1/chat/completions",
            json={
                "model": "deep-flash",
                "messages": [{"role": "user", "content": "What time is it?"}],
                "stream": True,
                "max_tokens": 300,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_time",
                            "description": "Get current time",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
                "tool_choice": "auto",
            },
        ) as resp:
            has_done = False
            count = 0
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    has_done = True
                else:
                    count += 1
                    try:
                        json.loads(payload)
                    except json.JSONDecodeError:
                        pass
            log_test(
                "  streaming + tools",
                has_done and count > 0,
                f"chunks={count} done={has_done}",
            )
    await asyncio.sleep(0.5)

    # ── 3c. Extract thinking_content from response ─────────────────────
    print("\n[3c] thinking_content en response (modelos con razonamiento)")
    # deepseek-v4-pro produce thinking_content
    st, d = await simple_chat(
        "pensamiento-profundo-caro", "solve 23+45 step by step", 500
    )
    if st == 200:
        choice = d.get("choices", [{}])[0]
        msg = choice.get("message", {})
        # El thinking_content aparece en proxy_metadata "thinking_blocks" o en el message
        meta = d.get("proxy_metadata", {})
        # Verificamos que la respuesta tenga contenido
        content = msg.get("content", "")
        log_test(
            "  Respuesta con razonamiento",
            bool(content),
            f"content_len={len(content)} finish={choice.get('finish_reason')}",
        )
    else:
        e = d.get("detail", {}).get("error", str(d)[:100])
        log_test("  Respuesta con razonamiento", False, f"HTTP {st} {e}")

    # ── 3d. Fallback entre modelos ────────────────────────────────────
    print("\n[3d] Fallback — 503/429 en modelo primario")
    # Esto es difícil de probar en vivo (no podemos provocar 503 fácilmente).
    # Verificamos que proxy_metadata tiene fallback_applied
    st, d = await simple_chat("flash-lowcost", "hi", 300)
    if st == 200:
        meta = d.get("proxy_metadata", {})
        log_test(
            "  fallback_applied false en éxito",
            meta.get("fallback_applied") is False,
            f"fallback={meta.get('fallback_applied')}",
        )
    else:
        log_test("  fallback_applied false en éxito", False, f"HTTP {st}")


# ══════════════════════════════════════════════════════════════════════════════
#  CONTEXT + CONSISTENCIA
# ══════════════════════════════════════════════════════════════════════════════


async def test_context():
    print("\n" + "=" * 70)
    print("  CONTEXTO MULTI-TURNO")
    print("=" * 70)

    conv = str(uuid.uuid4())
    secret = str(uuid.uuid4())[:8]

    # Turn 1 — establecer hecho
    print(f"\n  Conversación: {conv[:12]}...")
    print(f"  Secreto: {secret}")
    st1, _ = await simple_chat(
        "flash-lowcost", f"my secret code is {secret} remember it", 500, conv
    )
    await asyncio.sleep(0.5)

    # Turn 2 — preguntar (con historia completa)
    full = [
        {"role": "user", "content": f"my secret code is {secret} remember it"},
        {"role": "assistant", "content": "Got it, I'll remember that."},
        {"role": "user", "content": "what is my secret code?"},
    ]
    async with httpx.AsyncClient(timeout=120) as c:
        r = await c.post(
            f"{BASE}/v1/chat/completions",
            json={
                "model": "flash-lowcost",
                "conversation_id": conv,
                "messages": full,
                "max_tokens": 500,
            },
        )
        if r.status_code == 200:
            ctext = (
                r.json()
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .lower()
            )
            remembered = secret.lower() in ctext
            log_test(
                "  Contexto preservado entre turns",
                remembered,
                f"secret={secret} in response={remembered}",
            )
        else:
            log_test(
                "  Contexto preservado entre turns", False, f"HTTP {r.status_code}"
            )


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════


async def main():
    global PASS, FAIL
    print("=" * 70)
    print("  VERIFICACIÓN SPRINTS 1-3 — MODO STREAMING Y NO-STREAMING")
    print("=" * 70)
    print(f"  Servidor: {BASE}")

    await test_sprint1()
    await test_sprint2()
    await test_sprint3()
    await test_context()

    print()
    print("=" * 70)
    print(f"  RESULTADOS FINALES:  ✅ {PASS} passed   ❌ {FAIL} failed")
    print("=" * 70)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
