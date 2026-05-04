"""chatbot_leads and chatbot_conversations

Revision ID: 001_chatbot
Revises:
Create Date: 2026-05-04

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "001_chatbot"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "chatbot_leads",
        sa.Column(
            "lead_id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("psid", sa.String(255), nullable=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("phone_number", sa.String(64), nullable=False),
        sa.Column("email", sa.String(255), nullable=True),
        sa.Column("lead_type", sa.String(16), nullable=False),
        sa.Column("item_of_interest", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("lead_id", name="pk_chatbot_leads"),
        sa.CheckConstraint(
            "lead_type IN ('hard', 'soft')",
            name="ck_chatbot_leads_lead_type",
        ),
    )
    op.create_table(
        "chatbot_conversations",
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("lead_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "conversation",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
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
        sa.ForeignKeyConstraint(
            ["lead_id"],
            ["chatbot_leads.lead_id"],
            name="fk_chatbot_conversations_lead_id_chatbot_leads",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("session_id", name="pk_chatbot_conversations"),
    )
    op.create_index(
        "ix_chatbot_conversations_lead_id",
        "chatbot_conversations",
        ["lead_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_chatbot_conversations_lead_id",
        table_name="chatbot_conversations",
    )
    op.drop_table("chatbot_conversations")
    op.drop_table("chatbot_leads")
