"""Sprint 3: Add tool canonical storage columns.

Adds to conversation_turns:
- tool_definitions: JSONB — tool definitions in canonical OpenAI format
- thinking_blocks: JSONB — thinking/reasoning content from models
- tools_incomplete: BOOLEAN — TRUE if a tool call was interrupted mid-stream
- tools_level_used: INTEGER — max tool complexity level in this turn

Adds to conversations:
- max_tools_level: INTEGER — highest tools_level ever used

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-23
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- conversations ---
    op.add_column(
        "conversations",
        sa.Column("max_tools_level", sa.Integer(), server_default=sa.text("0"), nullable=False),
    )

    # --- conversation_turns ---
    op.add_column(
        "conversation_turns",
        sa.Column("tool_definitions", postgresql.JSONB(), nullable=True),
    )
    op.add_column(
        "conversation_turns",
        sa.Column("thinking_blocks", postgresql.JSONB(), nullable=True),
    )
    op.add_column(
        "conversation_turns",
        sa.Column("tools_incomplete", sa.Boolean(), server_default=sa.text("false"), nullable=False),
    )
    op.add_column(
        "conversation_turns",
        sa.Column("tools_level_used", sa.Integer(), server_default=sa.text("0"), nullable=False),
    )


def downgrade() -> None:
    op.drop_column("conversation_turns", "tools_level_used")
    op.drop_column("conversation_turns", "tools_incomplete")
    op.drop_column("conversation_turns", "thinking_blocks")
    op.drop_column("conversation_turns", "tool_definitions")
    op.drop_column("conversations", "max_tools_level")
