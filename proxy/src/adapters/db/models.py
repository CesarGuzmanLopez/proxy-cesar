"""SQLModel ORM models — Sprint 1 + Sprint 2 + Sprint 3 + Sprint 4.

python.md §6.2: SQLModel combines SQLAlchemy + Pydantic.
Sprint 1: conversations + conversation_turns (basic).
Sprint 2: +capability_* columns, turn_type, had_* flags.
Sprint 3: +tool_definitions, thinking_blocks, tools_incomplete,
          tools_level_used, max_tools_level.
Sprint 4: +conversation_snapshots table, +active_snapshot_id on conversations.
"""

import uuid
from datetime import datetime


from sqlalchemy import Column, DateTime, Text, func
from sqlalchemy import Uuid as SA_Uuid
from sqlalchemy import JSON as SA_JSON
from sqlmodel import Field, Relationship, SQLModel

# Reusable SQL default expressions
_SERVER_NOW = "now()"


class ConversationBase(SQLModel):
    pseudo_model: str = Field(max_length=128)
    physical_model: str = Field(max_length=256)
    total_tokens: int = Field(default=0, ge=0)

    # Sprint 2: Capability flags (additive, never reset)
    capability_has_images: bool = Field(default=False)
    capability_has_audio: bool = Field(default=False)
    capability_has_pdf: bool = Field(default=False)
    capability_has_video: bool = Field(default=False)
    capability_has_tools: bool = Field(default=False)
    capability_has_parallel_tools: bool = Field(default=False)

    # Sprint 3: max tool complexity level used in this conversation
    max_tools_level: int = Field(default=0, ge=0)

    # Sprint 5: image degradation tracking
    images_described: int = Field(default=0, ge=0)
    images_degraded_manually: bool = Field(default=False)


class Conversation(ConversationBase, table=True):
    __tablename__ = "conversations"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4, primary_key=True, sa_type=SA_Uuid()
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        sa_column=Column(DateTime(), server_default=_SERVER_NOW),
    )
    updated_at: datetime = Field(
        default_factory=datetime.now,
        sa_column=Column(DateTime(), server_default=_SERVER_NOW, onupdate=func.now()),
    )

    # Sprint 4: active snapshot for compacted conversations
    active_snapshot_id: uuid.UUID | None = Field(
        default=None, sa_type=SA_Uuid(), foreign_key="conversation_snapshots.id"
    )

    # FASE 3: Optimistic locking for future multi-writer scenarios
    version: int = Field(default=0, ge=0)

    turns: list["ConversationTurn"] = Relationship(back_populates="conversation")
    snapshots: list["ConversationSnapshot"] = Relationship(
        back_populates="conversation",
        sa_relationship_kwargs={
            "foreign_keys": "ConversationSnapshot.conversation_id",
            "order_by": "ConversationSnapshot.turn_number_at_compaction",
        },
    )


class ConversationTurnBase(SQLModel):
    conversation_id: uuid.UUID = Field(
        foreign_key="conversations.id", sa_type=SA_Uuid()
    )
    turn_number: int
    pseudo_model: str = Field(max_length=128)
    physical_model: str = Field(max_length=256)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    messages: list | dict = Field(default_factory=list, sa_type=SA_JSON)
    response: dict | None = Field(default=None, sa_type=SA_JSON)
    fallback_applied: bool = False
    fallback_reason: str | None = Field(default=None, max_length=256)

    # Sprint 2: Turn type and capability flags
    turn_type: str = Field(default="normal", max_length=32)
    had_images: bool = Field(default=False)
    had_tools: bool = Field(default=False)
    had_parallel_tools: bool = Field(default=False)

    # Sprint 3: Tool canonical storage columns
    tool_definitions: dict | None = Field(default=None, sa_type=SA_JSON)
    thinking_blocks: dict | None = Field(default=None, sa_type=SA_JSON)
    tools_incomplete: bool = Field(default=False)
    tools_level_used: int = Field(default=0, ge=0)


class ConversationTurn(ConversationTurnBase, table=True):
    __tablename__ = "conversation_turns"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4, primary_key=True, sa_type=SA_Uuid()
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        sa_column=Column(DateTime(), server_default=_SERVER_NOW),
    )

    conversation: Conversation = Relationship(back_populates="turns")


class ConversationSnapshotBase(SQLModel):
    """Snapshot of a compacted conversation.

    plan-proxy.md §11.3: Stores a Markdown snapshot preserving decisions,
    code, state, and pending items. Original history is never modified.

    Sprint 4: used by continuous compaction and (later) explicit compaction.
    """

    conversation_id: uuid.UUID = Field(
        foreign_key="conversations.id", sa_type=SA_Uuid()
    )
    snapshot_type: str = Field(max_length=32)
    """'continuous', 'explicit' (Sprint 6), or 'external' (client-side)."""

    tokens_before: int = Field(ge=0)
    """Tokens in the history at compaction time."""

    tokens_after: int = Field(ge=0)
    """Tokens in the generated snapshot."""

    compactor_model: str = Field(max_length=256)
    """Physical model that generated the snapshot."""

    snapshot_content: str = Field(sa_type=Text)
    """Markdown snapshot content."""

    turn_number_at_compaction: int = Field(ge=0)
    """Which turn triggered the compaction."""


class ConversationSnapshot(ConversationSnapshotBase, table=True):
    __tablename__ = "conversation_snapshots"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4, primary_key=True, sa_type=SA_Uuid()
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        sa_column=Column(DateTime(), server_default=_SERVER_NOW),
    )

    superseded_by: uuid.UUID | None = Field(
        default=None,
        sa_type=SA_Uuid(),
        foreign_key="conversation_snapshots.id",
    )

    conversation: Conversation = Relationship(
        back_populates="snapshots",
        sa_relationship_kwargs={
            "foreign_keys": "ConversationSnapshot.conversation_id",
        },
    )
