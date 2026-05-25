"""Initial migration: conversations + conversation_turns with Sprint 1 & Sprint 2 columns.

Sprint 1: basic schema (id, pseudo_model, physical_model, tokens, timestamps).
Sprint 2: +capability_* columns, turn_type, had_* flags.

Revision ID: 0001
Revises: None
Create Date: 2026-05-23
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- conversations ---
    op.create_table(
        "conversations",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("pseudo_model", sa.String(128), nullable=False),
        sa.Column("physical_model", sa.String(256), nullable=False),
        sa.Column(
            "total_tokens", sa.BigInteger(), server_default=sa.text("0"), nullable=False
        ),
        # Sprint 2: capability flags
        sa.Column(
            "capability_has_images",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column(
            "capability_has_audio",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column(
            "capability_has_pdf",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column(
            "capability_has_video",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column(
            "capability_has_tools",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column(
            "capability_has_parallel_tools",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
    )

    # --- conversation_turns ---
    op.create_table(
        "conversation_turns",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "conversation_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("conversations.id"),
            nullable=False,
        ),
        sa.Column("turn_number", sa.Integer(), nullable=False),
        sa.Column("pseudo_model", sa.String(128), nullable=False),
        sa.Column("physical_model", sa.String(256), nullable=False),
        sa.Column(
            "input_tokens", sa.Integer(), server_default=sa.text("0"), nullable=False
        ),
        sa.Column(
            "output_tokens", sa.Integer(), server_default=sa.text("0"), nullable=False
        ),
        sa.Column("messages", postgresql.JSONB(), nullable=False),
        sa.Column("response", postgresql.JSONB(), nullable=True),
        sa.Column(
            "fallback_applied",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column("fallback_reason", sa.String(256), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        # Sprint 2: turn type and capability flags
        sa.Column(
            "turn_type",
            sa.String(32),
            server_default=sa.text("'normal'"),
            nullable=False,
        ),
        sa.Column(
            "had_images", sa.Boolean(), server_default=sa.text("false"), nullable=False
        ),
        sa.Column(
            "had_tools", sa.Boolean(), server_default=sa.text("false"), nullable=False
        ),
        sa.Column(
            "had_parallel_tools",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
    )

    op.create_index(
        "idx_turns_conversation_id", "conversation_turns", ["conversation_id"]
    )


def downgrade() -> None:
    op.drop_table("conversation_turns")
    op.drop_table("conversations")
