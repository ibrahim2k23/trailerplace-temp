"""SQLAlchemy ORM models for chatbot persistence (Alembic metadata source)."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class ChatbotLead(Base):
    __tablename__ = "chatbot_leads"
    __table_args__ = (
        CheckConstraint("lead_type IN ('hard', 'soft')", name="ck_chatbot_leads_lead_type"),
    )

    lead_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),
    )
    psid: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    phone_number: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    email: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    lead_type: Mapped[str] = mapped_column(String(16), nullable=False)
    contact_status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="missing_contact",
        server_default="missing_contact",
    )
    item_of_interest: Mapped[str] = mapped_column(Text, nullable=False)

    conversations: Mapped[list["ChatbotConversation"]] = relationship(
        back_populates="lead",
    )


class ChatbotConversation(Base):
    __tablename__ = "chatbot_conversations"

    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    lead_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chatbot_leads.lead_id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    conversation: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    lead: Mapped["ChatbotLead"] = relationship(back_populates="conversations")
