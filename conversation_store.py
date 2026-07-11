from __future__ import annotations

import logging
import os
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from dotenv import load_dotenv
from sqlalchemy import func, select

from src.db_models import ChatbotConversation, ChatbotLead, ChatbotOutbox, ChatbotTurn
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


def _merge_existing_feedback(
    existing: list[dict[str, Any]] | None,
    incoming: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Preserve feedback saved directly to JSONB when rebuilding conversation turns."""
    if not isinstance(existing, list):
        return incoming

    merged: list[dict[str, Any]] = []
    for idx, turn in enumerate(incoming):
        next_turn = dict(turn)
        existing_turn = existing[idx] if idx < len(existing) and isinstance(existing[idx], dict) else {}
        if "feedback" not in next_turn:
            existing_feedback = existing_turn.get("feedback")
            if existing_feedback in (None, ""):
                existing_feedback = existing_turn.get("user_feedback")
            if existing_feedback not in (None, ""):
                next_turn["feedback"] = existing_feedback
        merged.append(next_turn)
    return merged


def _messages_to_conversation(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    conversation: list[dict[str, Any]] = []
    for idx in range(0, len(messages or []), 2):
        user_msg = messages[idx] if idx < len(messages or []) else None
        assistant_msg = messages[idx + 1] if idx + 1 < len(messages or []) else None
        turn = {
            "user": user_msg.get("content") if isinstance(user_msg, dict) else None,
            "chatbot": assistant_msg.get("content") if isinstance(assistant_msg, dict) else None,
        }
        if isinstance(assistant_msg, dict):
            for key in ("trailer_category", "metadata_filters_collected"):
                if assistant_msg.get(key) not in (None, "", [], {}):
                    turn[key] = assistant_msg[key]
            feedback = assistant_msg.get("feedback")
            if feedback in (None, ""):
                feedback = assistant_msg.get("user_feedback")
            if feedback not in (None, ""):
                turn["feedback"] = feedback
        conversation.append(turn)
    return conversation


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
    full_name: Optional[str] = None,
    email: Optional[str] = None,
    phone: Optional[str] = None,
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
            if full_name:
                existing.name = full_name
            if phone:
                existing.phone_number = phone
            if email is not None:
                existing.email = email or None
            existing.contact_status = (
                "contact_available"
                if (existing.phone_number or existing.email)
                else "missing_contact"
            )
            if item_of_interest:
                existing.item_of_interest = item_of_interest
            session.commit()
            return str(existing.lead_id)

        contact_status = "contact_available" if (phone or email) else "missing_contact"
        lead = ChatbotLead(
            lead_id=uuid.uuid4(),
            psid=session_id,
            name=full_name or None,
            phone_number=phone or None,
            email=email or None,
            lead_type="soft",
            contact_status=contact_status,
            item_of_interest=item_of_interest,
        )
        session.add(lead)
        session.commit()
        return str(lead.lead_id)


def update_lead_contact(
    *,
    session_id: str,
    full_name: Optional[str] = None,
    email: Optional[str] = None,
    phone: Optional[str] = None,
) -> Optional[str]:
    if not persistence_enabled():
        return None
    ensure_persistence_schema()
    with _session() as session:
        lead = session.execute(
            select(ChatbotLead).where(ChatbotLead.psid == session_id)
        ).scalar_one_or_none()
        if not lead:
            lead = ChatbotLead(
                lead_id=uuid.uuid4(),
                psid=session_id,
                name=full_name or None,
                phone_number=phone or None,
                email=email or None,
                lead_type="soft",
                contact_status="contact_available" if (phone or email) else "missing_contact",
                item_of_interest="Trailer inquiry",
            )
            session.add(lead)
        else:
            if full_name:
                lead.name = full_name
            if email is not None:
                lead.email = email or None
            if phone:
                lead.phone_number = phone
            lead.contact_status = (
                "contact_available" if (lead.phone_number or lead.email) else "missing_contact"
            )
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


def promote_lead_to_hard(session_id: str) -> None:
    if not persistence_enabled() or not session_id:
        return
    with _session() as session:
        lead = session.execute(
            select(ChatbotLead).where(ChatbotLead.psid == session_id)
        ).scalar_one_or_none()
        if not lead:
            return
        lead.lead_type = "hard"
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
            row.conversation = _merge_existing_feedback(row.conversation, conversation)
        else:
            row = ChatbotConversation(
                session_id=sid,
                lead_id=lid,
                conversation=conversation,
            )
            session.add(row)
        session.commit()


def persist_messages_snapshot(
    *,
    session_id: str,
    lead_id: Optional[str],
    messages: list[dict[str, Any]],
) -> None:
    if not persistence_enabled() or not lead_id or not session_id:
        return
    upsert_conversation(
        session_id=session_id,
        lead_id=lead_id,
        conversation=_messages_to_conversation(messages or []),
    )


def get_conversation(session_id: str) -> list[dict[str, Any]]:
    if not persistence_enabled() or not session_id:
        return []
    ensure_persistence_schema()
    try:
        sid = _as_uuid(session_id)
    except Exception:
        return []
    with _session() as session:
        row = session.get(ChatbotConversation, sid)
        if not row or not isinstance(row.conversation, list):
            return []
        return list(row.conversation)


@contextmanager
def durable_turn(
    session_id: str,
    turn_id: uuid.UUID,
    request_message: str,
):
    """Serialize one session's turns and expose its transaction to the service."""
    ensure_persistence_schema()
    sid = _as_uuid(session_id)
    with _session() as session:
        # Works for both existing and not-yet-created conversation rows.
        session.execute(select(func.pg_advisory_xact_lock(func.hashtext(str(sid)))))
        row = session.get(ChatbotConversation, sid)
        receipt = session.get(ChatbotTurn, (sid, turn_id))
        if receipt and receipt.request_message != request_message:
            raise ValueError("turn_id was already used with a different message")
        try:
            yield session, row, receipt
            session.commit()
        except Exception:
            session.rollback()
            raise


def restore_session(session_id: str) -> dict[str, Any]:
    if not persistence_enabled():
        return {"exists": False}
    ensure_persistence_schema()
    try:
        sid = _as_uuid(session_id)
    except Exception:
        return {"exists": False}
    with _session() as session:
        row = session.get(ChatbotConversation, sid)
        if not row:
            return {"exists": False}
        snapshot = dict(row.state_snapshot or {})
        messages = snapshot.get("messages")
        if not isinstance(messages, list):
            messages = []
            for turn in row.conversation or []:
                if turn.get("user") is not None:
                    messages.append({"role": "user", "content": turn.get("user")})
                if turn.get("chatbot") is not None:
                    messages.append({"role": "assistant", "content": turn.get("chatbot")})
        return {
            "exists": True,
            "closed": row.closed_at is not None,
            "state_version": row.state_version,
            "messages": messages,
            "sales_phase": snapshot.get("sales_phase", "main"),
            "customer_full_name": snapshot.get("customer_full_name"),
            "customer_email": snapshot.get("customer_email"),
            "customer_phone": snapshot.get("customer_phone"),
            "contact_status": snapshot.get("contact_status"),
        }


def close_session(session_id: str) -> None:
    if not persistence_enabled():
        return
    try:
        sid = _as_uuid(session_id)
    except Exception:
        return
    with _session() as session:
        row = session.get(ChatbotConversation, sid)
        if row and row.closed_at is None:
            row.closed_at = datetime.now(timezone.utc)
            session.commit()


def deliver_pending_outbox(limit: int = 10) -> None:
    """Best-effort at-least-once delivery; failed rows remain retryable."""
    if not persistence_enabled():
        return
    from src.chatbot.tools import email_tools

    handlers = {
        "interested_listing": email_tools.send_interested_listing_email,
        "non_sales_faq": email_tools.send_non_sales_faq_email,
        "escalation_alert": email_tools.send_escalation_alert_email,
        "results_shown": email_tools.send_trailer_results_shown_email,
    }
    with _session() as session:
        rows = list(session.execute(
            select(ChatbotOutbox)
            .where(ChatbotOutbox.status.in_(("pending", "failed")))
            .order_by(ChatbotOutbox.created_at)
            .with_for_update(skip_locked=True)
            .limit(limit)
        ).scalars())
        for row in rows:
            row.status = "processing"
            row.attempt_count += 1
        session.commit()

    for row in rows:
        try:
            handlers[row.event_type](**row.payload)
            status, error = "sent", None
        except Exception as exc:
            logger.exception("outbox_delivery_failed | event_id=%s", row.event_id)
            status, error = "failed", str(exc)[:2000]
        with _session() as session:
            current = session.get(ChatbotOutbox, row.event_id)
            if current:
                current.status = status
                current.last_error = error
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
                        "feedback": text or None,
                    }
                    row.conversation = conversation
                    session.commit()
        except Exception:
            logger.exception("Background feedback persistence failed")

    _pool.submit(_run)
