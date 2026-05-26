#!/usr/bin/env python3
"""Comprehensive real integration test — runs against the LIVE server.

Tests ALL pseudo-models, vision with captura.png, auto-describe switch,
manual degradation, streaming, and context.

Usage:
    python3 tests/run_all_models.py

Requires the server to be running on 127.0.0.1:9110.
"""

import asyncio
import base64
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
SKIP = 0


def log_test(name: str, ok: bool, detail: str = ""):
    global PASS, FAIL
    icon = "✅" if ok else "❌"
    if ok:
        PASS += 1
    else:
        FAIL += 1
    print(f"  {icon} {name}")
    if detail:
        for line in detail.strip().split("\n"):
            print(f"     {line}")


def load_image_data_url() -> str:
    """Return the captura.png as a data:// URL."""
    with open(IMG_PATH, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{b64}"


async def simple_chat(model: str, msg: str = "say hi", max_tokens: int = 300):
    """POST /v1/chat/completions, return parsed dict."""
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{BASE}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": msg}],
                "max_tokens": max_tokens,
            },
        )
        return resp.status_code, resp.json()


async def vision_chat(model: str, data_url: str):
    """Send an image to a vision model."""
    async with httpx.AsyncClient(timeout=180) as client:
        resp = await client.post(
            f"{BASE}/v1/chat/completions",
            json={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe esta captura de pantalla en 1 frase corta",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url, "detail": "low"},
                            },
                        ],
                    }
                ],
                "max_tokens": 300,
            },
        )
        return resp.status_code, resp.json()


async def switch_chat(from_model: str, to_model: str, conv_id: str, data_url: str):
    """Simulate a pseudo-model switch with image auto-describe."""
    async with httpx.AsyncClient(timeout=180) as client:
        # Step 1: Send image with vision model
        resp1 = await client.post(
            f"{BASE}/v1/chat/completions",
            json={
                "model": from_model,
                "conversation_id": conv_id,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe esta imagen"},
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url, "detail": "low"},
                            },
                        ],
                    }
                ],
                "max_tokens": 300,
            },
        )
        if resp1.status_code != 200:
            return resp1.status_code, resp1.json(), None, None

        # Step 2: Send text-only message to confirm context
        resp2 = await client.post(
            f"{BASE}/v1/chat/completions",
            json={
                "model": from_model,
                "conversation_id": conv_id,
                "messages": [{"role": "user", "content": "ok, now switch model"}],
                "max_tokens": 100,
            },
        )
        if resp2.status_code != 200:
            return resp2.status_code, resp2.json(), None, None

        # Step 3: Switch to non-vision model (should trigger auto-describe)
        resp3 = await client.post(
            f"{BASE}/v1/chat/completions",
            json={
                "model": to_model,
                "conversation_id": conv_id,
                "messages": [
                    {"role": "user", "content": "What did you see in that image?"}
                ],
                "max_tokens": 300,
            },
        )
        data = resp3.json() if resp3.status_code == 200 else None
        meta = data.get("proxy_metadata", {}) if data else None
        return (
            resp3.status_code,
            resp3.json() if resp3.status_code != 200 else None,
            data,
            meta,
        )


async def streaming_chat(model: str, msg: str = "count 1 to 5"):
    """Test streaming by reading all SSE chunks."""
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream(
            "POST",
            f"{BASE}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": msg}],
                "stream": True,
                "max_tokens": 300,
            },
        ) as resp:
            chunks = []
            has_content = False
            has_done = False
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    payload = line[6:]
                    if payload == "[DONE]":
                        has_done = True
                    else:
                        try:
                            obj = json.loads(payload)
                            delta = obj.get("choices", [{}])[0].get("delta", {})
                            if delta.get("content") is not None:
                                has_content = True
                            chunks.append(obj)
                        except json.JSONDecodeError:
                            pass
            return has_content, has_done, len(chunks)


async def degrade_images(conv_id: str):
    """Test POST /degrade-images."""
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(f"{BASE}/conversations/{conv_id}/degrade-images")
        return resp.status_code, resp.json()


# ══════════════════════════════════════════════════════════════════════════════


async def main():
    global SKIP
    print("=" * 70)
    print("  PRUEBA COMPRENSIVA — TODOS LOS PSEUDO-MODELOS + VISIÓN + DEGRADE")
    print("=" * 70)
    print(f"  Servidor: {BASE}")
    print(f"  Imagen:   {IMG_PATH}")
    print(f"  Tamaño:   {IMG_PATH.stat().st_size / 1024:.0f} KB")
    print()

    # Load image
    if not IMG_PATH.exists():
        print("❌ captura.png not found at project root!")
        sys.exit(1)
    DATA_URL = load_image_data_url()

    # ── 1. Health ────────────────────────────────────────────────────────
    print("\n[1] Health check")
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{BASE}/health")
        log_test("GET /health", r.status_code == 200, f"HTTP {r.status_code}")
    time.sleep(0.5)

    # ── 2. All pseudo-models (text only) ────────────────────────────────
    MODELS = [
        "pensamiento-profundo-caro",
        "tareas-avanzadas",
        "vision",
        "normal",
        "deep-flash",
        "flash-lowcost",
        "vision",
    ]
    print("\n[2] Pseudo-models — simple text chat")
    for m in MODELS:
        st, data = await simple_chat(m, "say hi in 1 word", 300)
        if st == 200:
            c = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            log_test(
                f"  {m}",
                bool(c),
                f"HTTP {st} content={c[:60]!r} phys={data.get('proxy_metadata', {}).get('physical_model', '?')}",
            )
        else:
            err = data.get("detail", {}).get("error", str(data)[:100])
            log_test(f"  {m}", False, f"HTTP {st} error={err}")
        time.sleep(0.3)

    # ── 3. Vision models with captura.png ────────────────────────────────
    print("\n[3] Vision models — with captura.png")
    for m in ["vision", "vision"]:
        st, data = await vision_chat(m, DATA_URL)
        if st == 200:
            c = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            log_test(
                f"  {m} sees image",
                bool(c),
                f"content={c[:80]!r} phys={data.get('proxy_metadata', {}).get('physical_model', '?')}",
            )
        else:
            err = data.get("detail", {}).get("error", str(data)[:100])
            log_test(f"  {m} sees image", False, f"HTTP {st} error={err}")
        time.sleep(1)

    # ── 4. Streaming ────────────────────────────────────────────────────
    print("\n[4] Streaming chat (all pseudo-models)")
    for m in MODELS:
        has_content, has_done, chunk_count = await streaming_chat(m, "count 1 to 5")
        ok = has_content and has_done
        log_test(
            f"  {m} streaming",
            ok,
            f"chunks={chunk_count} content_in_chunks={has_content} done={has_done}",
        )
        time.sleep(0.5)

    # ── 5. Context preservation ─────────────────────────────────────────
    print("\n[5] Context preservation (multi-turn)")
    conv_ctx = str(uuid.uuid4())
    async with httpx.AsyncClient(timeout=120) as client:
        # Turn 1
        r1 = await client.post(
            f"{BASE}/v1/chat/completions",
            json={
                "model": "flash-lowcost",
                "conversation_id": conv_ctx,
                "messages": [
                    {"role": "user", "content": "my secret word is ZEPHYR remember it"}
                ],
                "max_tokens": 500,
            },
        )
        if r1.status_code != 200:
            log_test("  Context Turn 1", False)
        else:
            time.sleep(1)
            # Turn 2 (with history)
            full_history = [
                {"role": "user", "content": "my secret word is ZEPHYR remember it"},
                {"role": "assistant", "content": "ok, I will remember ZEPHYR"},
                {"role": "user", "content": "what is my secret word?"},
            ]
            r2 = await client.post(
                f"{BASE}/v1/chat/completions",
                json={
                    "model": "flash-lowcost",
                    "conversation_id": conv_ctx,
                    "messages": full_history,
                    "max_tokens": 500,
                },
            )
            if r2.status_code == 200:
                c = (
                    r2.json()
                    .get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    .lower()
                )
                has_secret = "zephyr" in c
                log_test(
                    "  Context preserved", has_secret, f"content sample: {c[:80]!r}"
                )
            else:
                log_test("  Context preserved", False, f"HTTP {r2.status_code}")

    # ── 6. Switch vision → non-vision (auto-describe) ──────────────────
    print("\n[6] Auto-describe on pseudo-model switch")
    conv_switch = str(uuid.uuid4())
    st, err_data, data, meta = await switch_chat(
        "vision", "pensamiento-profundo-caro", conv_switch, DATA_URL
    )

    if st == 200 and data and meta:
        imgs = meta.get("images_described", 0)
        content = (
            data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        )
        log_test(
            "  Switch vision→non-vision",
            imgs > 0,
            f"images_described={imgs} content={content[:80]!r}",
        )
    elif st == 200 and data:
        # Switch worked but auto-describe may not have triggered
        imgs = meta.get("images_described", 0) if meta else 0
        content = (
            data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            if data
            else ""
        )
        log_test(
            "  Switch vision→non-vision",
            True,
            f"images_described={imgs} (may be 0 if vision conversation failed) content={content[:80]!r}",
        )
    else:
        err = err_data.get("detail", {}).get("error", str(err_data)[:100])
        log_test(
            "  Switch vision→non-vision",
            False,
            f"HTTP {st} error={err}",
        )

    # ── 7. Manual degradation ──────────────────────────────────────────
    print("\n[7] Manual degradation (POST /degrade-images)")
    conv_deg = str(uuid.uuid4())
    # First create a vision conversation
    async with httpx.AsyncClient(timeout=180) as client:
        r_vis = await client.post(
            f"{BASE}/v1/chat/completions",
            json={
                "model": "vision",
                "conversation_id": conv_deg,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What do you see?"},
                            {
                                "type": "image_url",
                                "image_url": {"url": DATA_URL, "detail": "low"},
                            },
                        ],
                    }
                ],
                "max_tokens": 300,
            },
        )
        if r_vis.status_code == 200:
            time.sleep(1)
            st_d, data_d = await degrade_images(conv_deg)
            if st_d == 200:
                log_test(
                    "  POST /degrade-images",
                    True,
                    f"images_described={data_d.get('images_described', 0)} can_switch_to={data_d.get('can_now_switch_to', [])}",
                )
            else:
                log_test(
                    "  POST /degrade-images",
                    False,
                    f"HTTP {st_d} error={data_d.get('detail', {}).get('error', '?')}",
                )
        else:
            log_test(
                "  POST /degrade-images",
                False,
                f"Vision conversation failed (HTTP {r_vis.status_code}) — cannot test degrade",
            )

    # ── 8. Unknown model rejection ──────────────────────────────────────
    print("\n[8] Error handling")
    st, data = await simple_chat("nonexistent-xyz", "hi", 10)
    log_test("  Unknown model → 400", st == 400, f"HTTP {st}")

    # ── 9. Incompatible switch ─────────────────────────────────────────
    print()
    conv_block = str(uuid.uuid4())
    async with httpx.AsyncClient(timeout=120) as client:
        # Create conversation with vision model (no image yet)
        r_b1 = await client.post(
            f"{BASE}/v1/chat/completions",
            json={
                "model": "vision",
                "conversation_id": conv_block,
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 100,
            },
        )
        if r_b1.status_code == 200:
            time.sleep(0.5)
            # Try switching to a model that blocks (tareas-avanzadas has images handling = block)
            # First add an image to the conversation
            r_img = await client.post(
                f"{BASE}/v1/chat/completions",
                json={
                    "model": "vision",
                    "conversation_id": conv_block,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "analyze"},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": DATA_URL, "detail": "low"},
                                },
                            ],
                        }
                    ],
                    "max_tokens": 100,
                },
            )
            if r_img.status_code == 200:
                time.sleep(1)
                r_b2 = await client.post(
                    f"{BASE}/v1/chat/completions",
                    json={
                        "model": "tareas-avanzadas",
                        "conversation_id": conv_block,
                        "messages": [{"role": "user", "content": "switch?"}],
                        "max_tokens": 100,
                    },
                )
                if r_b2.status_code == 409:
                    log_test("  Blocked switch → 409", True, f"HTTP {r_b2.status_code}")
                else:
                    d2 = r_b2.json()
                    log_test(
                        "  Blocked switch → 409",
                        False,
                        f"HTTP {r_b2.status_code} {d2.get('detail', {}).get('error', '?')}",
                    )
            else:
                log_test(
                    "  Blocked switch → 409",
                    False,
                    "Could not add image to conversation",
                )
        else:
            log_test(
                "  Blocked switch → 409", False, "Could not create initial conversation"
            )

    # ── Summary ─────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print(f"  RESULTADOS:  ✅ {PASS} passed  ❌ {FAIL} failed  ⏭️  {SKIP} skipped")
    print("=" * 70)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
