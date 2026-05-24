"""SQLModel ORM models — Sprint 1 + Sprint 2 + Sprint 3.

python.md §6.2: SQLModel combines SQLAlchemy + Pydantic.
Sprint 1: conversations + conversation_turns (basic).
Sprint 2: +capability_* columns, turn_type, had_* flags.
Sprint 3: +tool_definitions, thinking_blocks, tools_incomplete,
          tools_level_used, max_tools_level.
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import Column, DateTime, Integer, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlmodel import Field, Relationship, SQLModel


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


class Conversation(ConversationBase, table=True):
    __tablename__ = "conversations"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4, primary_key=True, sa_type=UUID(as_uuid=True)
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        sa_column=Column(DateTime(timezone=True), server_default="now()"),
    )
    updated_at: datetime = Field(
        default_factory=datetime.now,
        sa_column=Column(
            DateTime(timezone=True), server_default="now()", onupdate="now()"
        ),
    )

    turns: list["ConversationTurn"] = Relationship(back_populates="conversation")


class ConversationTurnBase(SQLModel):
    conversation_id: uuid.UUID = Field(
        foreign_key="conversations.id", sa_type=UUID(as_uuid=True)
    )
    turn_number: int
    pseudo_model: str = Field(max_length=128)
    physical_model: str = Field(max_length=256)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    messages: dict = Field(default={}, sa_type=JSONB)
    response: Optional[dict] = Field(default=None, sa_type=JSONB)
    fallback_applied: bool = False
    fallback_reason: Optional[str] = Field(default=None, max_length=256)

    # Sprint 2: Turn type and capability flags
    turn_type: str = Field(default="normal", max_length=32)
    had_images: bool = Field(default=False)
    had_tools: bool = Field(default=False)
    had_parallel_tools: bool = Field(default=False)

    # Sprint 3: Tool canonical storage columns
    tool_definitions: Optional[dict] = Field(default=None, sa_type=JSONB)
    thinking_blocks: Optional[dict] = Field(default=None, sa_type=JSONB)
    tools_incomplete: bool = Field(default=False)
    tools_level_used: int = Field(default=0, ge=0)


class ConversationTurn(ConversationTurnBase, table=True):
    __tablename__ = "conversation_turns"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4, primary_key=True, sa_type=UUID(as_uuid=True)
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        sa_column=Column(DateTime(timezone=True), server_default="now()"),
    )

    conversation: Conversation = Relationship(back_populates="turns")
