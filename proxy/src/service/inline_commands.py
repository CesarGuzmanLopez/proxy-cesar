"""Inline command handler — Sprint 9.

Commands the user types directly in the chat message to perform operations
without leaving the conversation or using curl.

Commands:
  @compact      — Normalize tools + degrade multimedia + compact history.
                  One-shot prep for switching pseudo-models.
  @degrade      — Describe all multimedia (images, PDFs, video frames) as text.
  @status       — Show conversation state (tokens, model, capabilities).
  @help         — List all commands.

Design:
  - Commands start with "@" (easy to type, unlikely as LLM content).
  - The proxy checks for commands BEFORE any LLM processing.
  - If a command is detected, it's handled inline and a text response
    is returned instead of forwarding to the LLM.
  - Commands are idempotent where possible.
"""

import re
import uuid

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from sqlmodel.ext.asyncio.session import AsyncSession

from src.adapters.db.models import Conversation, ConversationSnapshot, ConversationTurn
from src.service.capability_detector import load_session_capabilities
from src.service.compactor.explicit import compact_conversation
from src.service.multimedia.image_describer import auto_describe_images
from src.service.tools_normalizer import generate_preview, normalize_history

# ── Command regex ───────────────────────────────────────────────────────────

_COMMAND_RE = re.compile(r"^@(\w[\w-]*)\s*(.*)", re.IGNORECASE)

_VALID_COMMANDS = frozenset({"compact", "degrade", "status", "help"})


# ── Result type ─────────────────────────────────────────────────────────────


class InlineCommandResult:
    """Result of handling an inline command."""

    def __init__(
        self,
        handled: bool = False,
        response_text: str = "",
        response_metadata: dict | None = None,
        skip_llm: bool = False,
    ):
        self.handled = handled
        self.response_text = response_text
        self.response_metadata = response_metadata or {}
        self.skip_llm = skip_llm


# ── Public entry point ─────────────────────────────────────────────────────


async def handle_inline_command(
    messages: list[dict],
    conversation_id: str | None,
    config,
    db: AsyncSession,
    arq_pool=None,
) -> InlineCommandResult:
    """Check if the last user message contains an inline command.

    Must be called BEFORE any LLM processing. If a command is detected,
    the caller should return the result instead of calling the LLM.
    """
    last_msg = _find_last_user_message(messages)
    if last_msg is None:
        return InlineCommandResult()

    content = last_msg.get("content", "")
    if not isinstance(content, str):
        return InlineCommandResult()

    match = _COMMAND_RE.match(content.strip())
    if not match:
        return InlineCommandResult()

    command = match.group(1).lower()
    args = match.group(2).strip()

    # Resolve aliases
    command = _resolve_aliases(command) or command

    if command not in _VALID_COMMANDS:
        return InlineCommandResult()

    if not conversation_id and command in ("compact", "degrade"):
        return InlineCommandResult(
            handled=True,
            response_text=(
                "⚠️ **No active conversation.**\n\n"
                "Start a conversation first, then use:\n"
                "- `@compact` — prepare conversation for model switch\n"
                "- `@degrade` — describe multimedia as text\n"
                "- `@status` — show conversation state"
            ),
            skip_llm=True,
        )

    match command:
        case "compact":
            return await _handle_compact(conversation_id, config, db, arq_pool, args)
        case "degrade":
            return await _handle_degrade(conversation_id, config, db)
        case "status":
            return await _handle_status(conversation_id, db, config)
        case "help":
            return _handle_help()
    return InlineCommandResult()


# ── @compact — one-shot prep for model switch ──────────────────────────────


async def _handle_compact(
    conversation_id: str | None,
    config,
    db: AsyncSession,
    arq_pool=None,
    args: str = "",
) -> InlineCommandResult:
    """@compact: normalize tools + degrade multimedia + compact history.

    This is the one-shot command to prepare a conversation for switching
    to a different pseudo-model. It runs all three operations in sequence
    so the resulting conversation is maximally portable.
    """
    if not conversation_id:
        return _no_conv_result()

    steps: list[str] = []
    total_meta: dict = {}

    conv_uuid = _parse_uuid(conversation_id)
    conv = await db.get(Conversation, conv_uuid)
    if not conv:
        return InlineCommandResult(
            handled=True,
            response_text="⚠️ Conversation not found.",
            skip_llm=True,
        )

    caps = await load_session_capabilities(db, conv_uuid, conv.total_tokens)

    # ── Step 1: Normalize parallel tools ─────────────────────────────────
    if caps.has_parallel_tools:
        try:
            norm_result = await _do_normalize(conv_uuid, db)
            if norm_result:
                steps.append(f"✅ **Tools normalised:** {norm_result['turns_serialized']} turn(s) serialised")
                total_meta["normalize"] = norm_result
        except Exception as e:
            steps.append(f"⚠️ Tool normalisation skipped: {e}")

    # ── Step 2: Degrade multimedia ───────────────────────────────────────
    if caps.has_images:
        try:
            deg_result = await _do_degrade(conv, conv_uuid, config, db)
            if deg_result and deg_result.get("images_described", 0) > 0:
                steps.append(
                    f"🖼️ **Multimedia degraded:** {deg_result['images_described']} "
                    f"item(s) described by {deg_result.get('described_by', '?')}"
                )
                total_meta["degrade"] = deg_result
        except Exception as e:
            steps.append(f"⚠️ Degradation skipped: {e}")

    # ── Step 3: Compact history ──────────────────────────────────────────
    dry_run = "--dry" in args
    if not dry_run:
        try:
            comp_result = await compact_conversation(
                conversation_id=conversation_id,
                db=db,
                config=config,
                arq_pool=arq_pool,
            )
            if comp_result.get("status") == "processing":
                steps.append(
                    f"📦 **Compaction dispatched** — task `{comp_result.get('task_id', '?')}`"
                )
            else:
                steps.append(
                    f"📦 **History compacted:** "
                    f"{comp_result.get('tokens_before', 0):,} → "
                    f"{comp_result.get('tokens_after', 0):,} tokens "
                    f"(−{comp_result.get('tokens_reduced_pct', 0)}%)"
                )
            total_meta["compact"] = comp_result
        except HTTPException as e:
            detail = e.detail if isinstance(e.detail, dict) else {"error": str(e)}
            steps.append(f"⚠️ Compaction not needed: {detail.get('error', '')}")

    if not steps:
        steps.append("ℹ️ Nothing to do — conversation is already clean.")

    preview = ""
    if total_meta.get("compact") and total_meta["compact"].get("preview"):
        preview = (
            "\n\n**Snapshot preview:**\n```\n"
            + total_meta["compact"]["preview"][:600]
            + "\n```"
        )

    return InlineCommandResult(
        handled=True,
        response_text="## 🧹 Conversation prepared for model switch\n\n" + "\n".join(steps) + preview,
        response_metadata=total_meta,
        skip_llm=True,
    )


# ── @degrade — describe multimedia as text ────────────────────────────────


async def _handle_degrade(
    conversation_id: str | None,
    config,
    db: AsyncSession,
) -> InlineCommandResult:
    """@degrade: describe all multimedia (images, PDFs, etc.) as text."""
    if not conversation_id:
        return _no_conv_result()

    conv_uuid = _parse_uuid(conversation_id)
    conv = await db.get(Conversation, conv_uuid, options=[selectinload(Conversation.turns)])
    if not conv:
        return InlineCommandResult(
            handled=True,
            response_text="⚠️ Conversation not found.",
            skip_llm=True,
        )

    caps = await load_session_capabilities(db, conv_uuid, conv.total_tokens)

    if not caps.has_images:
        return InlineCommandResult(
            handled=True,
            response_text="✅ No multimedia found. Nothing to degrade.",
            skip_llm=True,
        )

    # Find a vision model
    current_pm = config.pseudo_models.get(conv.pseudo_model)
    if not current_pm:
        return InlineCommandResult(
            handled=True,
            response_text=f"⚠️ Unknown pseudo-model: '{conv.pseudo_model}'.",
            skip_llm=True,
        )

    vision_models = [m for m in current_pm.physical_models if m.vision]
    if not vision_models:
        # Try to find ANY vision pseudo-model as fallback
        for name, pm in config.pseudo_models.items():
            vm = [m for m in pm.physical_models if m.vision]
            if vm:
                vision_models = vm
                break

    if not vision_models:
        return InlineCommandResult(
            handled=True,
            response_text=(
                "⚠️ No vision-capable model available. "
                "Make sure at least one pseudo-model has `vision: true` models."
            ),
            skip_llm=True,
        )

    vision_model = (
        conv.physical_model
        if any(m.model == conv.physical_model and m.vision for m in current_pm.physical_models)
        else vision_models[0].model
    )

    # Load all messages from turns
    all_messages: list[dict] = []
    for turn in sorted(conv.turns, key=lambda t: t.turn_number):
        turn_msgs = turn.messages
        if isinstance(turn_msgs, list):
            all_messages.extend(turn_msgs)

    result = await auto_describe_images(all_messages, vision_model)
    described_messages, desc_meta = result

    described_count = desc_meta.get("images_described", 0)
    if described_count == 0:
        return InlineCommandResult(
            handled=True,
            response_text="ℹ️ No multimedia items found to degrade.",
            skip_llm=True,
        )

    # Store as degradation_event turn
    turn_number = (max(t.turn_number for t in conv.turns) + 1) if conv.turns else 1
    deg_turn = ConversationTurn(
        conversation_id=conv_uuid,
        turn_number=turn_number,
        pseudo_model=conv.pseudo_model,
        physical_model=vision_model,
        messages=described_messages,
        response={"metadata": desc_meta},
        input_tokens=0,
        output_tokens=desc_meta.get("total_description_tokens", 0),
        turn_type="degradation_event",
        had_images=False,
        had_tools=False,
        had_parallel_tools=False,
    )
    db.add(deg_turn)
    conv.images_described = (conv.images_described or 0) + described_count
    conv.images_degraded_manually = True
    await db.commit()

    return InlineCommandResult(
        handled=True,
        response_text=(
            f"🖼️ **Multimedia degraded to text!**\n\n"
            f"- Items described: {described_count}\n"
            f"- Unique items: {desc_meta.get('unique_images_described', 0)}\n"
            f"- Described by: `{vision_model}`\n"
            f"- Description tokens: {desc_meta.get('total_description_tokens', 0)}\n\n"
            f"Now you can switch to a pseudo-model without vision support.\n"
            f"The descriptions are preserved in the conversation history."
        ),
        response_metadata=desc_meta,
        skip_llm=True,
    )


# ── @status — conversation state ───────────────────────────────────────────


async def _handle_status(
    conversation_id: str | None,
    db: AsyncSession,
    config,
) -> InlineCommandResult:
    """@status: show current conversation state."""
    if not conversation_id:
        return InlineCommandResult(
            handled=True,
            response_text="📊 **No active conversation.**\n\nStart a chat first, then use `@status`.",
            skip_llm=True,
        )

    conv_uuid = _parse_uuid(conversation_id)
    conv = await db.get(Conversation, conv_uuid)
    if not conv:
        return InlineCommandResult(
            handled=True,
            response_text="⚠️ Conversation not found.",
            skip_llm=True,
        )

    caps = await load_session_capabilities(db, conv_uuid, conv.total_tokens)

    result = await db.execute(
        select(ConversationTurn).where(ConversationTurn.conversation_id == conv_uuid)
    )
    turns = result.scalars().all()
    turn_count = len(turns)

    status_lines = [
        f"📊 **Conversation Status**\n",
        f"- **ID:** `{conversation_id[:16]}...`",
        f"- **Pseudo-modelo:** `{conv.pseudo_model}`",
        f"- **Modelo físico:** `{conv.physical_model}`",
        f"- **Turnos:** {turn_count}",
        f"- **Tokens totales:** {conv.total_tokens:,}",
    ]

    if hasattr(conv, 'context_window') and conv.context_window:
        pct = round((conv.total_tokens / conv.context_window) * 100, 1)
        status_lines.append(f"- **Contexto usado:** {pct}% de {conv.context_window:,}")

    cap_flags = []
    if caps.has_images:
        cap_flags.append("🖼️ imágenes")
    if caps.has_tools:
        cap_flags.append("🔧 tools")
    if caps.has_parallel_tools:
        cap_flags.append("⚡ parallel")
    if caps.has_pdf:
        cap_flags.append("📄 PDF")
    if caps.images_described:
        cap_flags.append(f"📝 {caps.images_described} descritos")
    if cap_flags:
        status_lines.append(f"- **Capacidades:** {' | '.join(cap_flags)}")

    if conv.active_snapshot_id:
        status_lines.append(f"- **Snapshot:** ✅ (`{str(conv.active_snapshot_id)[:8]}...`)")
    else:
        status_lines.append("- **Snapshot:** ❌")

    status_lines.extend([
        "",
        "**Commands:**",
        "- `@compact` — prepare for model switch (normalize + degrade + compact)",
        "- `@degrade` — describe multimedia as text",
        "- `@status` — this screen",
    ])

    return InlineCommandResult(
        handled=True,
        response_text="\n".join(status_lines),
        response_metadata={
            "pseudo_model": conv.pseudo_model,
            "physical_model": conv.physical_model,
            "turn_count": turn_count,
            "total_tokens": conv.total_tokens,
        },
        skip_llm=True,
    )


# ── @help ──────────────────────────────────────────────────────────────────


def _handle_help() -> InlineCommandResult:
    return InlineCommandResult(
        handled=True,
        response_text=(
            "📚 **Inline Commands — Help**\n\n"
            "Type any of these as your chat message:\n\n"
            "### `@compact`\n"
            "**One-shot prep for switching models.** Runs three steps:\n"
            "1. Normalize parallel tools → sequential\n"
            "2. Describe multimedia (images, PDFs) as text\n"
            "3. Compact conversation history into a snapshot\n\n"
            "After `@compact`, the conversation is maximally portable.\n\n"
            "### `@degrade`\n"
            "Describe all multimedia items as text.\n"
            "Use before switching to a non-vision pseudo-model.\n\n"
            "### `@status`\n"
            "Show current conversation state: model, tokens, capabilities.\n\n"
            "---\n"
            "*Commands work on the current conversation only.*"
        ),
        skip_llm=True,
    )


# ── Internal helpers ───────────────────────────────────────────────────────


async def _do_normalize(conv_uuid: uuid.UUID, db: AsyncSession) -> dict | None:
    """Run tool normalisation and return result metadata."""
    result = await db.execute(
        select(ConversationTurn)
        .where(ConversationTurn.conversation_id == conv_uuid)
        .order_by(ConversationTurn.turn_number)
    )
    turns = result.scalars().all()
    if not turns:
        return None

    all_messages: list[dict] = []
    for turn in turns:
        tm = turn.messages
        if isinstance(tm, list):
            all_messages.extend(tm)
        elif isinstance(tm, dict):
            msgs = tm.get("messages", tm)
            if isinstance(msgs, list):
                all_messages.extend(msgs)

    normalized, meta = normalize_history(all_messages)
    if meta.turns_serialized == 0:
        return None

    conv = await db.get(Conversation, conv_uuid)
    norm_turn = ConversationTurn(
        conversation_id=conv_uuid,
        turn_number=(max(t.turn_number for t in turns) + 1),
        turn_type="normalization_event",
        pseudo_model=conv.pseudo_model if conv else "?",
        physical_model=conv.physical_model if conv else "?",
        messages={"normalized_history": normalized, "metadata": vars(meta)},
    )
    db.add(norm_turn)
    await db.flush()
    return {
        "turns_serialized": meta.turns_serialized,
        "parallel_calls_serialized": meta.parallel_calls_serialized,
        "affected_turns": meta.affected_turns,
    }


async def _do_degrade(
    conv: Conversation,
    conv_uuid: uuid.UUID,
    config,
    db: AsyncSession,
) -> dict | None:
    """Run multimedia degradation and return result metadata."""
    current_pm = config.pseudo_models.get(conv.pseudo_model)
    if not current_pm:
        return None

    vision_models = [m for m in current_pm.physical_models if m.vision]
    if not vision_models:
        return None

    vision_model = (
        conv.physical_model
        if any(m.model == conv.physical_model and m.vision for m in current_pm.physical_models)
        else vision_models[0].model
    )

    conv = await db.get(Conversation, conv_uuid, options=[selectinload(Conversation.turns)])
    all_messages: list[dict] = []
    for turn in sorted(conv.turns, key=lambda t: t.turn_number):
        tm = turn.messages
        if isinstance(tm, list):
            all_messages.extend(tm)

    _, desc_meta = await auto_describe_images(all_messages, vision_model)
    return desc_meta


def _find_last_user_message(messages: list[dict]) -> dict | None:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg
    return None


def _parse_uuid(value: str) -> uuid.UUID:
    try:
        return uuid.UUID(value)
    except ValueError:
        return uuid.uuid5(uuid.NAMESPACE_DNS, value)


def _no_conv_result() -> InlineCommandResult:
    return InlineCommandResult(
        handled=True,
        response_text="⚠️ **No active conversation.**\n\nStart a conversation first.",
        skip_llm=True,
    )


_ALIASES: dict[str, str] = {
    "compactar": "compact",
    "comprime": "compact",
    "comprimir": "compact",
    "prepare": "compact",
    "preparar": "compact",
    "degradar": "degrade",
    "describir": "degrade",
    "describe": "degrade",
    "estado": "status",
    "info": "status",
    "ayuda": "help",
    "comandos": "help",
}


def _resolve_aliases(command: str) -> str | None:
    return _ALIASES.get(command)
