"""SQLAlchemy ORM models for chatbot persistence."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, func
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
    psid: Mapped[str | None] = mapped_column(String(255), nullable=True)
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    phone_number: Mapped[str | None] = mapped_column(String(64), nullable=True)
    email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    lead_type: Mapped[str] = mapped_column(String(16), nullable=False)
    contact_status: Mapped[str] = mapped_column(String(32), nullable=False, default="missing_contact", server_default="missing_contact")
    item_of_interest: Mapped[str] = mapped_column(Text, nullable=False)

    conversations: Mapped[list["ChatbotConversation"]] = relationship(back_populates="lead")


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
    state_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    state_schema_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    state_version: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    lead: Mapped[ChatbotLead] = relationship(back_populates="conversations")
    # The unit of work orders inserts by RELATIONSHIP, not by the raw foreign key. Without
    # this, a session's FIRST turn — the one flush where the conversation row and its turn
    # row are both new — could emit the chatbot_turns INSERT first and violate
    # chatbot_turns_session_id_fkey. It worked most of the time purely by luck of ordering.
    turns: Mapped[list["ChatbotTurn"]] = relationship(back_populates="conversation", cascade="all, delete-orphan")


class ChatbotTurn(Base):
    __tablename__ = "chatbot_turns"

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chatbot_conversations.session_id", ondelete="CASCADE"),
        primary_key=True,
    )
    turn_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    request_message: Mapped[str] = mapped_column(Text, nullable=False)
    response: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    conversation: Mapped[ChatbotConversation] = relationship(back_populates="turns")


class ChatbotOutbox(Base):
    __tablename__ = "chatbot_outbox"
    __table_args__ = (UniqueConstraint("session_id", "turn_id", "event_key", name="uq_chatbot_outbox_event"),)

    event_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    turn_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    event_key: Mapped[str] = mapped_column(String(128), nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending", server_default="pending")
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
