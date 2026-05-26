"""Inline command handler — Sprint 9.

Commands the user types directly in the chat message.

Commands:
  /status       — Show conversation state (tokens, model, capabilities).
  /help         — List all commands.

Design:
  - The proxy checks for commands BEFORE any LLM processing.
  - If a command is detected, it's handled inline.
  - No degradation, no compaction — those are handled by tools/OpenCode.
"""

import re
import uuid

from sqlalchemy import select
from sqlmodel.ext.asyncio.session import AsyncSession

from src.adapters.db.models import Conversation, ConversationTurn
from src.service.capability_detector import load_session_capabilities

_COMMAND_RE = re.compile(r"^[/@]?(\b(?:status|help)\b)\s*(.*)", re.IGNORECASE)

_VALID_COMMANDS = frozenset({"status", "help"})


class InlineCommandResult:
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


async def handle_inline_command(
    messages: list[dict],
    conversation_id: str | None,
    config,
    db: AsyncSession,
) -> InlineCommandResult:
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
    command = _resolve_aliases(command) or command

    if command not in _VALID_COMMANDS:
        return InlineCommandResult()

    match command:
        case "status":
            return await _handle_status(conversation_id, db, config)
        case "help":
            return _handle_help()
    return InlineCommandResult()


async def _handle_status(
    conversation_id: str | None,
    db: AsyncSession,
    config,
) -> InlineCommandResult:
    if not conversation_id:
        return InlineCommandResult(
            handled=True,
            response_text="📊 **No active conversation.**\n\nStart a chat first.",
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
    turn_count = len(result.scalars().all())

    status_lines = [
        "📊 **Conversation Status**\n",
        f"- **ID:** `{conversation_id[:16]}...`",
        f"- **Pseudo-modelo:** `{conv.pseudo_model}`",
        f"- **Modelo físico:** `{conv.physical_model}`",
        f"- **Turnos:** {turn_count}",
        f"- **Tokens totales:** {conv.total_tokens:,}",
    ]

    if hasattr(conv, "context_window") and conv.context_window:
        pct = round((conv.total_tokens / conv.context_window) * 100, 1)
        status_lines.append(f"- **Contexto usado:** {pct}% de {conv.context_window:,}")

    cap_flags = []
    if caps.has_images:
        cap_flags.append("🖼️ imágenes")
    if caps.has_tools:
        cap_flags.append("🔧 tools")
    if caps.has_parallel_tools:
        cap_flags.append("⚡ parallel")
    if cap_flags:
        status_lines.append(f"- **Capacidades:** {' | '.join(cap_flags)}")

    if conv.active_snapshot_id:
        status_lines.append(
            f"- **Snapshot:** ✅ (`{str(conv.active_snapshot_id)[:8]}...`)"
        )
    else:
        status_lines.append("- **Snapshot:** ❌")

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


def _handle_help() -> InlineCommandResult:
    return InlineCommandResult(
        handled=True,
        response_text=(
            "📚 **Inline Commands**\n\n"
            "### `/status`\n"
            "Show current conversation state: model, tokens, capabilities.\n\n"
            "---\n"
            "Compaction: handled natively by OpenCode.\n"
            "Image degradation: use a vision model as a tool."
        ),
        skip_llm=True,
    )


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


_ALIASES: dict[str, str] = {
    "estado": "status",
    "info": "status",
    "ayuda": "help",
    "comandos": "help",
}


def _resolve_aliases(command: str) -> str | None:
    return _ALIASES.get(command)
