"""Persist complete chatbot state and idempotent turns."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20260703_0002"
down_revision = "20260525_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("chatbot_conversations", sa.Column("state_snapshot", postgresql.JSONB(), nullable=True))
    op.add_column("chatbot_conversations", sa.Column("state_schema_version", sa.Integer(), server_default="1", nullable=False))
    op.add_column("chatbot_conversations", sa.Column("state_version", sa.Integer(), server_default="0", nullable=False))
    op.add_column("chatbot_conversations", sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True))
    op.create_table(
        "chatbot_turns",
        sa.Column("session_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("chatbot_conversations.session_id", ondelete="CASCADE"), primary_key=True),
        sa.Column("turn_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("request_message", sa.Text(), nullable=False),
        sa.Column("response", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_table(
        "chatbot_outbox",
        sa.Column("event_id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("turn_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("event_key", sa.String(128), nullable=False),
        sa.Column("event_type", sa.String(64), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column("status", sa.String(16), server_default="pending", nullable=False),
        sa.Column("attempt_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("session_id", "turn_id", "event_key", name="uq_chatbot_outbox_event"),
    )
    op.create_index("ix_chatbot_outbox_session_id", "chatbot_outbox", ["session_id"])


def downgrade() -> None:
    op.drop_index("ix_chatbot_outbox_session_id", table_name="chatbot_outbox")
    op.drop_table("chatbot_outbox")
    op.drop_table("chatbot_turns")
    op.drop_column("chatbot_conversations", "closed_at")
    op.drop_column("chatbot_conversations", "state_version")
    op.drop_column("chatbot_conversations", "state_schema_version")
    op.drop_column("chatbot_conversations", "state_snapshot")
