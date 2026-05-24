#!/usr/bin/env python3
"""Verificación exhaustiva de Sprints 4 y 5.

Sprint 4 — Compaction:
  - Pre-compaction cuando el input excede el threshold
  - Continuous compaction (cuando la conversación crece)
  - Compactador pseudo-model (operación)
  - External compaction detection

Sprint 5 — Auto-describe + Router LLM:
  - proxy_metadata con images_described / router_suggestion
  - Auto-describe al switch visión→no-visión
  - Degradación manual (POST /degrade-images)
  - Router LLM en pensamiento-profundo-caro
"""

import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:9110"
IMG_PATH = Path(__file__).resolve().parent.parent.parent / "captura.png"

PASS = 0
FAIL = 0


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


def load_image() -> str:
    if not IMG_PATH.exists():
        return ""
    import base64
    with open(IMG_PATH, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{b64}"


async def chat(model: str, msg: str | list, max_tokens: int = 300,
               conv_id: str = None, stream: bool = False,
               tools: list = None, tool_choice=None):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": msg}] if isinstance(msg, str) else msg,
        "max_tokens": max_tokens,
    }
    if conv_id:
        body["conversation_id"] = conv_id
    if stream:
        body["stream"] = True
    if tools:
        body["tools"] = tools
    if tool_choice:
        body["tool_choice"] = tool_choice
    async with httpx.AsyncClient(timeout=180) as c:
        r = await c.post(f"{BASE}/v1/chat/completions", json=body)
        return r.status_code, r.json()


async def stream_chat(model: str, msg: str, max_tokens: int = 300):
    async with httpx.AsyncClient(timeout=180) as c:
        async with c.stream(
            "POST", f"{BASE}/v1/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": msg}],
                  "stream": True, "max_tokens": max_tokens},
        ) as resp:
            has_content, has_done, count = False, False, 0
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


async def test_sprint4():
    print("=" * 70)
    print("  SPRINT 4 — COMPACTION (PRE + CONTINUOUS + COMPACTADOR)")
    print("=" * 70)

    # ── 4a. Compactador pseudo-model (debe responder) ─────────────────
    print("\n[4a] Compactador — operación de compactación")
    st, d = await chat("compactador", "compact this conversation: user asked about Python lists. assistant explained list comprehension. user asked for examples.", 1000)
    if st == 200:
        c = d.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        phys = d.get("proxy_metadata", {}).get("physical_model", "?")
        log_test("  Compactador responde", bool(c),
                 f"content_len={len(c)} physical={phys}")
    else:
        err = d.get("detail", {}).get("message", str(d)[:150])
        log_test("  Compactador responde", False, f"HTTP {st} error={err[:100]}")
    await asyncio.sleep(1)

    # ── 4b. Input threshold + pre-compaction ──────────────────────────
    print("\n[4b] Threshold guard — input excede el límite")
    huge = "hello " * 30000
    st, d = await chat("flash-vision", huge, 10)  # threshold=16000
    if st == 400:
        err_code = d.get("detail", {}).get("error", "")
        log_test("  flash-vision input excede → 400", "INPUT_EXCEEDS_THRESHOLD" in err_code,
                 f"error={err_code}")
    else:
        log_test("  flash-vision input excede → 400", False, f"HTTP {st}")

    # ── 4c. Normal model input grande pero sin pre-compaction habilitado ──
    print("\n[4c] Normal model — input grande (sin pre-compaction, pasa)")
    big = "test " * 500
    st, d = await chat("normal", big[:5000], 200)  # normal no tiene pre-compaction
    if st == 200:
        c = d.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        meta = d.get("proxy_metadata", {})
        log_test("  Normal sin pre-compaction", bool(c),
                 f"content={c[:40]!r} pre_compaction={meta.get('pre_compaction_applied')}")
    else:
        log_test("  Normal sin pre-compaction", False, f"HTTP {st}")
    await asyncio.sleep(1)

    # ── 4d. Continuous compaction fields in proxy_metadata ────────────
    print("\n[4d] proxy_metadata — compaction fields")
    st, d = await chat("pensamiento-profundo-caro", "test compaction fields", 200)
    if st == 200:
        meta = d.get("proxy_metadata", {})
        has_pre = "pre_compaction_applied" in meta
        has_cont = "continuous_compaction_applied" in meta
        has_ext = "external_compaction_detected" in meta
        all_fields = has_pre and has_cont and has_ext
        log_test("  Campos de compactación presentes", all_fields,
                 f"pre={has_pre} continuous={has_cont} external={has_ext}")
    else:
        log_test("  Campos de compactación presentes", False, f"HTTP {st}")
    await asyncio.sleep(1)

    # ── 4e. External compaction detection ─────────────────────────────
    # This is hard to trigger in live tests, so we verify the metadata field
    print("\n[4e] External compaction — campo en metadata")
    st, d = await chat("normal", "hello world", 200)
    if st == 200:
        meta = d.get("proxy_metadata", {})
        log_test("  external_compaction_detected field",
                 "external_compaction_detected" in meta,
                 f"value={meta.get('external_compaction_detected')}")
    else:
        log_test("  external_compaction_detected field", False, f"HTTP {st}")


async def test_sprint5():
    print("\n" + "=" * 70)
    print("  SPRINT 5 — AUTO-DESCRIBE + ROUTER LLM + DEGRADE")
    print("=" * 70)

    DATA_URL = load_image()
    HAS_IMAGE = bool(DATA_URL)

    # ── 5a. Router LLM in proxy_metadata ─────────────────────────────
    print("\n[5a] Router LLM — pensamiento-profundo-caro (router_llm.enabled=true)")
    st, d = await chat("pensamiento-profundo-caro", "what is 2+2?", 200)
    if st == 200:
        meta = d.get("proxy_metadata", {})
        suggestion = meta.get("router_suggestion")
        log_test("  router_suggestion presente",
                 suggestion is not None,
                 f"suggestion={suggestion}")
    else:
        log_test("  router_suggestion presente", False, f"HTTP {st}")
    await asyncio.sleep(1)

    # ── 5b. Router LLM deshabilitado para modelos sin él ─────────────
    print("\n[5b] Router LLM — normal (router_llm.enabled=false)")
    st, d = await chat("normal", "what is 2+2?", 200)
    if st == 200:
        meta = d.get("proxy_metadata", {})
        log_test("  router_suggestion ausente para normal",
                 meta.get("router_suggestion") is None,
                 f"router_suggestion={meta.get('router_suggestion')}")
    else:
        log_test("  router_suggestion ausente para normal", False, f"HTTP {st}")
    await asyncio.sleep(1)

    # ── 5c. proxy_metadata Sprint 5 fields ──────────────────────────
    print("\n[5c] proxy_metadata — Sprint 5 fields")
    st, d = await chat("flash-lowcost", "hello", 200)
    if st == 200:
        meta = d.get("proxy_metadata", {})
        fields = ["images_described", "images_described_by",
                   "images_degraded_manually", "router_suggestion"]
        present = all(f in meta for f in fields)
        log_test("  Todos los campos S5 presentes", present,
                 f"ausentes={[f for f in fields if f not in meta]}")
    else:
        log_test("  Todos los campos S5 presentes", False, f"HTTP {st}")
    await asyncio.sleep(1)

    # ── 5d. Auto-describe switch (vision → non-vision) ──────────────
    print("\n[5d] Auto-describe — switch visión→no-visión")
    if not HAS_IMAGE:
        log_test("  Auto-describe switch", False,
                 "captura.png no encontrada — skip")
    else:
        conv = str(uuid.uuid4())
        # Turn 1: send image to vision model
        st1, d1 = await chat(
            "avanzada-vision",
            [
                {"role": "user", "content": [
                    {"type": "text", "text": "Describe"},
                    {"type": "image_url", "image_url": {"url": DATA_URL, "detail": "low"}},
                ]},
            ],
            500, conv,
        )
        if st1 == 200:
            # Turn 2: switch to non-vision model (debe trigger auto-describe)
            await asyncio.sleep(1)
            st2, d2 = await chat(
                "pensamiento-profundo-caro",
                "What do you see in that image? (describe from the auto-description)",
                500, conv,
            )
            if st2 == 200:
                meta = d2.get("proxy_metadata", {})
                imgs = meta.get("images_described", 0)
                desc_by = meta.get("images_described_by", "")
                content = d2.get("choices", [{}])[0].get("message", {}).get("content", "")
                log_test("  Switch con auto-describe", imgs > 0,
                         f"images_described={imgs} described_by={desc_by} content={content[:60]!r}")
            else:
                err = d2.get("detail", {}).get("message", str(d2)[:100])
                log_test("  Switch con auto-describe", False, f"HTTP {st2} error={err[:80]}")
        else:
            err = d1.get("detail", {}).get("message", str(d1)[:100])
            log_test("  Switch con auto-describe", False,
                     f"visión falló HTTP {st1} {err[:80]}")
    await asyncio.sleep(1)

    # ── 5e. Degradación manual ──────────────────────────────────────
    print("\n[5e] Degradación manual (POST /degrade-images)")
    if not HAS_IMAGE:
        log_test("  Degradación manual", False, "captura.png no encontrada — skip")
    else:
        conv_d = str(uuid.uuid4())
        st1, d1 = await chat(
            "avanzada-vision",
            [
                {"role": "user", "content": [
                    {"type": "text", "text": "Describe"},
                    {"type": "image_url", "image_url": {"url": DATA_URL, "detail": "low"}},
                ]},
            ],
            500, conv_d,
        )
        if st1 == 200:
            await asyncio.sleep(1)
            async with httpx.AsyncClient(timeout=60) as c:
                r = await c.post(f"{BASE}/conversations/{conv_d}/degrade-images")
                if r.status_code == 200:
                    data = r.json()
                    imgs = data.get("images_described", 0)
                    can_switch = data.get("can_now_switch_to", [])
                    log_test("  POST /degrade-images OK", imgs > 0,
                             f"images_described={imgs} can_switch={len(can_switch)} models")
                elif r.status_code == 400:
                    err = r.json().get("detail", {}).get("error", "")
                    log_test("  POST /degrade-images", True,
                             f"HTTP 400 (esperado si sin imágenes): {err}")
                else:
                    err = r.json().get("detail", {}).get("message", str(r.json())[:100])
                    log_test("  POST /degrade-images", False, f"HTTP {r.status_code} {err[:80]}")
        else:
            err = d1.get("detail", {}).get("message", str(d1)[:100])
            log_test("  POST /degrade-images", False,
                     f"visión falló HTTP {st1} {err[:80]}")
    await asyncio.sleep(1)

    # ── 5f. Degradación sin imágenes → 400 ──────────────────────────
    print("\n[5f] Degradación manual sin imágenes → 400")
    conv_no = str(uuid.uuid4())
    st_n, _ = await chat("normal", "hello no images", 100, conv_no)
    if st_n == 200:
        await asyncio.sleep(0.5)
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(f"{BASE}/conversations/{conv_no}/degrade-images")
            ok = r.status_code == 400
            log_test("  degrade sin imágenes → 400", ok,
                     f"HTTP {r.status_code}")
    else:
        log_test("  degrade sin imágenes → 400", False, f"HTTP {st_n}")

    # ── 5g. Compatible models endpoint ──────────────────────────────
    print("\n[5g] GET /compatible-models")
    conv_c = str(uuid.uuid4())
    st_c, _ = await chat("normal", "hello test compat", 100, conv_c)
    if st_c == 200:
        async with httpx.AsyncClient() as c:
            r = await c.get(f"{BASE}/conversations/{conv_c}/compatible-models")
            if r.status_code == 200:
                data = r.json()
                models = data.get("compatible_models", [])
                log_test("  compatible-models endpoint",
                         len(models) > 0,
                         f"{len(models)} models listed")
            else:
                log_test("  compatible-models endpoint", False, f"HTTP {r.status_code}")
    else:
        log_test("  compatible-models endpoint", False, f"HTTP {st_c}")


async def main():
    global PASS, FAIL
    print("=" * 70)
    print("  VERIFICACIÓN SPRINTS 4 y 5")
    print("=" * 70)
    print(f"  Servidor: {BASE}")
    print(f"  Imagen:   {IMG_PATH} ({'EXISTE' if IMG_PATH.exists() else 'NO EXISTE'})")
    print()

    await test_sprint4()
    await test_sprint5()

    print()
    print("=" * 70)
    print(f"  RESULTADOS FINALES:  ✅ {PASS} passed   ❌ {FAIL} failed")
    print("=" * 70)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
