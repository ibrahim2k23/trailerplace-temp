from __future__ import annotations

import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from dotenv import load_dotenv
from sqlalchemy import select

from src.db_models import ChatbotConversation, ChatbotLead
from src import db

load_dotenv()

logger = logging.getLogger(__name__)
_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="chat_persist")


def persistence_enabled() -> bool:
    flag = (os.getenv("TRAILERPLACE_PERSIST_CHATS") or "1").strip().lower()
    return flag not in {"0", "false", "no", "off"} and db.database_enabled()


def _as_uuid(value: str | uuid.UUID) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


def _session():
    return db.get_session_factory()()


def ensure_persistence_schema() -> None:
    if not persistence_enabled():
        return
    try:
        db.ensure_schema()
    except Exception:
        logger.exception("Failed to prepare chatbot persistence schema")


def create_or_get_soft_lead(
    *,
    session_id: str,
    full_name: str,
    email: Optional[str],
    phone: str,
    item_of_interest: str = "Trailer inquiry",
) -> Optional[str]:
    if not persistence_enabled():
        return None
    ensure_persistence_schema()
    with _session() as session:
        existing = session.execute(
            select(ChatbotLead).where(ChatbotLead.psid == session_id)
        ).scalar_one_or_none()
        if existing:
            existing.name = full_name
            existing.phone_number = phone
            existing.email = email or None
            if item_of_interest:
                existing.item_of_interest = item_of_interest
            session.commit()
            return str(existing.lead_id)

        lead = ChatbotLead(
            lead_id=uuid.uuid4(),
            psid=session_id,
            name=full_name,
            phone_number=phone,
            email=email or None,
            lead_type="soft",
            item_of_interest=item_of_interest,
        )
        session.add(lead)
        session.commit()
        return str(lead.lead_id)


def update_lead_item_of_interest(session_id: str, item_of_interest: str) -> None:
    if not persistence_enabled() or not item_of_interest:
        return
    with _session() as session:
        lead = session.execute(
            select(ChatbotLead).where(ChatbotLead.psid == session_id)
        ).scalar_one_or_none()
        if not lead:
            return
        lead.item_of_interest = item_of_interest
        session.commit()


def upsert_conversation(
    *,
    session_id: str,
    lead_id: Optional[str],
    conversation: list[dict[str, Any]],
) -> None:
    if not persistence_enabled() or not lead_id:
        return
    ensure_persistence_schema()
    sid = _as_uuid(session_id)
    lid = _as_uuid(lead_id)
    with _session() as session:
        row = session.get(ChatbotConversation, sid)
        if row:
            row.conversation = conversation
        else:
            row = ChatbotConversation(
                session_id=sid,
                lead_id=lid,
                conversation=conversation,
            )
            session.add(row)
        session.commit()


def enqueue_upsert_conversation(
    *,
    session_id: str,
    lead_id: Optional[str],
    conversation: list[dict[str, Any]],
) -> None:
    if not persistence_enabled() or not lead_id:
        return

    def _run() -> None:
        try:
            upsert_conversation(
                session_id=session_id,
                lead_id=lead_id,
                conversation=conversation,
            )
        except Exception:
            logger.exception("Background conversation persistence failed")

    _pool.submit(_run)


def enqueue_save_user_feedback(
    session_id: str,
    turn_idx: int,
    text: str,
    timestamp_iso: str,
) -> None:
    """Compatibility helper used by app.py.

    Feedback is stored inside the matching conversation turn when persistence is enabled.
    """
    if not persistence_enabled():
        return

    def _run() -> None:
        try:
            sid = _as_uuid(session_id)
            with _session() as session:
                row = session.get(ChatbotConversation, sid)
                if not row or not isinstance(row.conversation, list):
                    return
                conversation = list(row.conversation)
                if 0 <= turn_idx < len(conversation):
                    conversation[turn_idx] = {
                        **conversation[turn_idx],
                        "user_feedback": text,
                        "feedback_at": timestamp_iso,
                    }
                    row.conversation = conversation
                    session.commit()
        except Exception:
            logger.exception("Background feedback persistence failed")

    _pool.submit(_run)
