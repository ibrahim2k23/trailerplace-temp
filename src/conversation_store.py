from __future__ import annotations

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Callable

from sqlalchemy import func, select

from src import db
from src.config import settings
from src.db_models import ChatbotConversation, ChatbotLead, ChatbotOutbox, ChatbotTurn

logger = logging.getLogger(__name__)
_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="chat_persist")
_OUTBOX_HANDLERS: dict[str, Callable[..., Any]] = {}


def persistence_enabled() -> bool:
    return settings.trailerplace_persist_chats and db.database_enabled()


def register_outbox_handler(event_type: str, handler: Callable[..., Any]) -> None:
    _OUTBOX_HANDLERS[event_type] = handler


# All outbox event types carry a rendered {subject, body} payload, so every one
# dispatches to the same sender (M7 step 3). Wired at app startup.
_DEFAULT_OUTBOX_EVENT_TYPES = (
    "interested_listing",
    "non_sales_faq",
    "escalation_alert",
    "team_request",
    "results_shown",
    "unanswered_question",
)


def register_default_outbox_handlers() -> None:
    from src.tools import email_sender

    def _handler(*, subject: str, body: str) -> None:
        if not email_sender.send_email(subject, body):
            raise RuntimeError("email_sender.send_email returned False")

    for event_type in _DEFAULT_OUTBOX_EVENT_TYPES:
        register_outbox_handler(event_type, _handler)


def _as_uuid(value: str | uuid.UUID) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


def _session():
    return db.get_session_factory()()


def _merge_existing_feedback(existing: list[dict[str, Any]] | None, incoming: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
    full_name: str | None = None,
    email: str | None = None,
    phone: str | None = None,
    item_of_interest: str = "Trailer inquiry",
) -> str | None:
    if not persistence_enabled():
        return None
    ensure_persistence_schema()
    with _session() as session:
        lead = session.execute(select(ChatbotLead).where(ChatbotLead.psid == session_id)).scalar_one_or_none()
        if not lead:
            lead = ChatbotLead(
                lead_id=uuid.uuid4(),
                psid=session_id,
                name=full_name or None,
                phone_number=phone or None,
                email=email or None,
                lead_type="soft",
                contact_status="contact_available" if (phone or email) else "missing_contact",
                item_of_interest=item_of_interest,
            )
            session.add(lead)
        else:
            if full_name:
                lead.name = full_name
            if phone:
                lead.phone_number = phone
            if email is not None:
                lead.email = email or None
            if item_of_interest:
                lead.item_of_interest = item_of_interest
            lead.contact_status = "contact_available" if (lead.phone_number or lead.email) else "missing_contact"
        session.commit()
        return str(lead.lead_id)


def update_lead_contact(*, session_id: str, full_name: str | None = None, email: str | None = None, phone: str | None = None) -> str | None:
    if not persistence_enabled():
        return None
    ensure_persistence_schema()
    with _session() as session:
        lead = session.execute(select(ChatbotLead).where(ChatbotLead.psid == session_id)).scalar_one_or_none()
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
            lead.contact_status = "contact_available" if (lead.phone_number or lead.email) else "missing_contact"
        session.commit()
        return str(lead.lead_id)


def update_lead_item_of_interest(session_id: str, item_of_interest: str, *, session=None) -> None:
    if not persistence_enabled() or not item_of_interest:
        return
    # A caller inside durable_turn passes its own session so the update lands in the
    # same transaction (M7 step 3); otherwise open a short-lived session and commit.
    if session is not None:
        lead = session.execute(select(ChatbotLead).where(ChatbotLead.psid == session_id)).scalar_one_or_none()
        if lead:
            lead.item_of_interest = item_of_interest
        return
    with _session() as own:
        lead = own.execute(select(ChatbotLead).where(ChatbotLead.psid == session_id)).scalar_one_or_none()
        if lead:
            lead.item_of_interest = item_of_interest
            own.commit()


def promote_lead_to_hard(session_id: str, *, session=None) -> None:
    if not persistence_enabled() or not session_id:
        return
    # Same-transaction upgrade when the route hands us its durable session; the lead
    # is never downgraded (Locked Decision: lead hardness).
    if session is not None:
        lead = session.execute(select(ChatbotLead).where(ChatbotLead.psid == session_id)).scalar_one_or_none()
        if lead:
            lead.lead_type = "hard"
        return
    with _session() as own:
        lead = own.execute(select(ChatbotLead).where(ChatbotLead.psid == session_id)).scalar_one_or_none()
        if lead:
            lead.lead_type = "hard"
            own.commit()


def upsert_conversation(*, session_id: str, lead_id: str | None, conversation: list[dict[str, Any]]) -> None:
    if not persistence_enabled() or not lead_id:
        return
    ensure_persistence_schema()
    sid = _as_uuid(session_id)
    with _session() as session:
        row = session.get(ChatbotConversation, sid)
        if row:
            row.conversation = _merge_existing_feedback(row.conversation, conversation)
        else:
            session.add(ChatbotConversation(session_id=sid, lead_id=_as_uuid(lead_id), conversation=conversation))
        session.commit()


def persist_messages_snapshot(*, session_id: str, lead_id: str | None, messages: list[dict[str, Any]]) -> None:
    upsert_conversation(session_id=session_id, lead_id=lead_id, conversation=_messages_to_conversation(messages or []))


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
        return list(row.conversation) if row and isinstance(row.conversation, list) else []


@contextmanager
def durable_turn(session_id: str, turn_id: str | uuid.UUID, request_message: str):
    ensure_persistence_schema()
    sid = _as_uuid(session_id)
    tid = _as_uuid(turn_id)
    with _session() as session:
        session.execute(select(func.pg_advisory_xact_lock(func.hashtext(str(sid)))))
        row = session.get(ChatbotConversation, sid)
        receipt = session.get(ChatbotTurn, (sid, tid))
        if receipt and receipt.request_message != request_message:
            raise ValueError("turn_id was already used with a different message")
        try:
            yield session, row, receipt
            session.commit()
        except Exception:
            session.rollback()
            raise


def enqueue_outbox_event(session, *, session_id: str | uuid.UUID, turn_id: str | uuid.UUID, event_key: str, event_type: str, payload: dict[str, Any]) -> ChatbotOutbox:
    event = ChatbotOutbox(
        session_id=_as_uuid(session_id),
        turn_id=_as_uuid(turn_id),
        event_key=event_key,
        event_type=event_type,
        payload=payload,
    )
    session.add(event)
    return event


def _apply_feedback_to_messages(
    messages: list[dict[str, Any]], conversation: list[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    """Map conversation[i]['feedback'] back onto the i-th assistant message.

    The simplified turn list holds exactly one entry per user/assistant pair, which is
    the same invariant that makes app.py's `turn_idx = i // 2` correct. Without this,
    feedback saved before a browser refresh looks lost in the UI even though it is in
    the database.
    """
    restored = [dict(message) for message in messages]
    for message in restored:
        message.setdefault("listings", None)
        message.setdefault("user_feedback", None)
    for idx, turn in enumerate(conversation or []):
        feedback = turn.get("feedback") if isinstance(turn, dict) else None
        if feedback in (None, ""):
            continue
        assistant_idx = idx * 2 + 1
        if assistant_idx < len(restored) and restored[assistant_idx].get("role") == "assistant":
            restored[assistant_idx]["user_feedback"] = feedback
    return restored


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
        messages = _apply_feedback_to_messages(messages, row.conversation)
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
    ensure_persistence_schema()
    try:
        sid = _as_uuid(session_id)
    except Exception:
        return
    with _session() as session:
        row = session.get(ChatbotConversation, sid)
        if row and row.closed_at is None:
            row.closed_at = datetime.now(timezone.utc)
            session.commit()


def deliver_pending_outbox_async(limit: int = 10) -> None:
    """Fire-and-forget outbox drain, submitted to the background persistence pool.

    Email delivery (Graph OAuth token + sendMail, or SMTP) has no bearing on the
    reply the user is waiting on, but `deliver_pending_outbox` used to run
    synchronously inside `/chat` after the durable-turn commit. A slow or failing
    network path to login.microsoftonline.com (a TLS handshake reset, a stalled
    connection before its 30s timeout) then became latency on every chat turn that
    happened to be the one to trigger a drain — up to ~60s in the worst case. The
    outbox is already designed for exactly this: a failed row just stays retryable
    and the next drain (triggered by the next turn, on any session) picks it up.
    Running it off the request thread makes that the *only* behavior a slow send
    has, instead of also blocking the response.
    """
    if not persistence_enabled():
        return
    _pool.submit(deliver_pending_outbox, limit)


def deliver_pending_outbox(limit: int = 10) -> None:
    if not persistence_enabled():
        return
    with _session() as session:
        rows = list(
            session.execute(
                select(ChatbotOutbox)
                .where(ChatbotOutbox.status.in_(("pending", "failed")))
                .order_by(ChatbotOutbox.created_at)
                .with_for_update(skip_locked=True)
                .limit(limit)
            ).scalars()
        )
        for row in rows:
            row.status = "processing"
            row.attempt_count += 1
            row.claimed_at = datetime.now(timezone.utc)
        session.commit()

    for row in rows:
        try:
            handler = _OUTBOX_HANDLERS.get(row.event_type)
            if handler is None:
                raise RuntimeError(f"No outbox handler registered for {row.event_type}")
            handler(**row.payload)
            status, error = "sent", None
            logger.info(
                "TOOL outbox: event_id=%s event_type=%s session=%s -> sent (attempt %d)",
                row.event_id, row.event_type, row.session_id, row.attempt_count,
            )
        except Exception as exc:
            logger.exception(
                "outbox_delivery_failed | event_id=%s event_type=%s session=%s attempt=%d",
                row.event_id, row.event_type, row.session_id, row.attempt_count,
            )
            status, error = "failed", str(exc)[:2000]
        with _session() as session:
            current = session.get(ChatbotOutbox, row.event_id)
            if current:
                current.status = status
                current.last_error = error
                session.commit()


def enqueue_upsert_conversation(*, session_id: str, lead_id: str | None, conversation: list[dict[str, Any]]) -> None:
    if not persistence_enabled() or not lead_id:
        return

    def _run() -> None:
        try:
            upsert_conversation(session_id=session_id, lead_id=lead_id, conversation=conversation)
        except Exception:
            logger.exception("Background conversation persistence failed")

    _pool.submit(_run)


def enqueue_save_user_feedback(session_id: str, turn_idx: int, text: str, timestamp_iso: str) -> None:
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
                    conversation[turn_idx] = {**conversation[turn_idx], "feedback": text or None}
                    row.conversation = conversation
                    session.commit()
        except Exception:
            logger.exception("Background feedback persistence failed")

    _pool.submit(_run)
