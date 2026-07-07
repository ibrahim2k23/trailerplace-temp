from __future__ import annotations

import logging
import os
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar

from src.email_sender import send_faq_email_sync, send_ticket_notification

logger = logging.getLogger(__name__)

# The durable path returns early via _queue_email; only the in-memory path
# reaches the synchronous SMTP senders below, which can block the request for
# up to the SMTP timeout (~30s). Dispatch those sends to a small background pool
# so the turn returns immediately. Delivery is best-effort (like the durable
# outbox); failures are logged, not surfaced to the customer.
_EMAIL_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="email-send")


def _send_async(fn, **kwargs) -> None:
    def _run() -> None:
        try:
            fn(**kwargs)
        except Exception:
            logger.exception("background_email_send_failed | fn=%s", getattr(fn, "__name__", fn))

    # Tests (and any caller needing deterministic delivery) set EMAIL_SEND_SYNC=1
    # to run the send inline instead of on the background pool.
    if os.getenv("EMAIL_SEND_SYNC") == "1":
        _run()
        return
    _EMAIL_EXECUTOR.submit(_run)
from src.conversation_store import (
    get_conversation,
    promote_lead_to_hard,
    update_lead_item_of_interest,
)

FAQ_CATEGORY_LABELS = {
    "contact_human": "Customer asked to contact a person.",
    "financing": "Customer asked about financing.",
    "trade_in": "Customer asked about trade-in.",
    "service_parts": "Customer asked about service or parts.",
    "store_info": "Customer asked about store information.",
}
FAQ_ITEM_OF_INTEREST_LABELS = {
    "contact_human": "Wants to talk to a sales representative",
    "financing": "Finance Query",
    "trade_in": "Trade-In Query",
    "service_parts": "Spare Parts Query",
    "store_info": "Store Information Query",
}
TRAILER_RESULTS_SHOWN_SUBJECT = "Trailer Results Shown to User"
TRAILER_RESULTS_SHOWN_DESCRIPTION = "Trailer results were shown to the user."
_EMAIL_EVENTS: ContextVar[list[dict] | None] = ContextVar("chatbot_email_events", default=None)


@contextmanager
def capture_email_events():
    events: list[dict] = []
    token = _EMAIL_EVENTS.set(events)
    try:
        yield events
    finally:
        _EMAIL_EVENTS.reset(token)


def _queue_email(event_type: str, payload: dict) -> dict | None:
    events = _EMAIL_EVENTS.get()
    if events is None:
        return None
    events.append({"event_type": event_type, "payload": payload})
    return {"status": "queued", "body_preview": ""}


def _conversation_transcript_from_db(session_id: str) -> str:
    turns = get_conversation(session_id)
    if not turns:
        return ""
    lines: list[str] = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        user_text = str(turn.get("user") or "").strip()
        chatbot_text = str(turn.get("chatbot") or "").strip()
        if user_text:
            lines.append("User:")
            lines.append(user_text)
            lines.append("")
        if chatbot_text:
            lines.append("Chatbot:")
            lines.append(chatbot_text)
            lines.append("")
    return "\n".join(lines).strip()


def _append_conversation(summary_text: str, session_id: str) -> str:
    transcript = _conversation_transcript_from_db(session_id)
    if not transcript:
        return summary_text
    return f"{summary_text}\n\nConversation:\n{transcript}"


def send_interested_listing_email(
    *,
    session_id: str,
    full_name: str,
    email: Optional[str],
    phone: str,
    item_name: str,
) -> dict:
    """Tool 2: notify the business that the customer is interested in a listing."""
    queued = _queue_email("interested_listing", dict(
        session_id=session_id, full_name=full_name, email=email, phone=phone, item_name=item_name
    ))
    if queued:
        return queued
    item = (item_name or "").strip()
    details = _append_conversation("", session_id).strip()
    _send_async(send_ticket_notification,
        full_name=full_name,
        email=email,
        phone=phone,
        item_name=item,
        details=details,
    )
    promote_lead_to_hard(session_id)
    update_lead_item_of_interest(session_id, item)
    return {
        "status": "sent",
        "body_preview": (
            f"Full Name: {full_name}\n"
            f"Email: {(email or '').strip() or 'Not provided'}\n"
            f"Phone Number: {phone}\n\n"
            f'The user is interested in "{item}"'
            + (f"\n\n{details}" if details else "")
        ),
    }


def send_non_sales_faq_email(
    *,
    session_id: str = "",
    full_name: str,
    email: Optional[str],
    phone: str,
    faq_category: str,
    summary: str | None = None,
    user_message: str | None = None,
    context_summary: str | None = None,
) -> dict:
    """Tool 3: notify the business about contact, financing, trade-in, service, or store info."""
    queued = _queue_email("non_sales_faq", dict(
        session_id=session_id, full_name=full_name, email=email, phone=phone,
        faq_category=faq_category, summary=summary, user_message=user_message,
        context_summary=context_summary,
    ))
    if queued:
        return queued
    category = (faq_category or "contact_human").strip().lower()
    if category not in FAQ_CATEGORY_LABELS:
        category = "contact_human"
    summary_line = (summary or FAQ_CATEGORY_LABELS[category]).strip()
    if not summary_line.startswith(f"[{category}]"):
        summary_line = f"[{category}] {summary_line}"
    summary_line = _append_conversation(summary_line, session_id)
    _send_async(send_faq_email_sync,
        full_name=full_name,
        email=email,
        phone=phone,
        subject=f"FAQ inquiry: {category}",
        summary_line=summary_line,
    )
    promote_lead_to_hard(session_id)
    update_lead_item_of_interest(
        session_id,
        FAQ_ITEM_OF_INTEREST_LABELS.get(category, "Wants to talk to a sales representative"),
    )
    return {
        "status": "sent",
        "body_preview": (
            f"Name: {full_name}\n"
            f"Email: {(email or '').strip() or 'Not provided'}\n"
            f"Phone Number: {phone}\n\n"
            f"{summary_line}"
        ),
    }


def send_escalation_alert_email(
    *,
    session_id: str = "",
    full_name: str,
    email: Optional[str],
    phone: str,
    summary: str,
    user_message: str | None = None,
    context_summary: str | None = None,
) -> dict:
    """Tool 4: alert the business about an unsupported customer-requested action."""
    queued = _queue_email("escalation_alert", dict(
        session_id=session_id, full_name=full_name, email=email, phone=phone,
        summary=summary, user_message=user_message, context_summary=context_summary,
    ))
    if queued:
        return queued
    summary_line = (summary or "Customer requested an action the chatbot cannot complete.").strip()
    if not summary_line.startswith("[Escalation Alert]"):
        summary_line = f"[Escalation Alert] {summary_line}"
    summary_line = _append_conversation(summary_line, session_id)
    _send_async(send_faq_email_sync,
        full_name=full_name,
        email=email,
        phone=phone,
        subject="Escalation Alert",
        summary_line=summary_line,
    )
    return {
        "status": "sent",
        "body_preview": (
            f"Name: {full_name}\n"
            f"Email: {(email or '').strip() or 'Not provided'}\n"
            f"Phone Number: {phone}\n\n"
            f"{summary_line}"
        ),
    }


def send_trailer_results_shown_email(
    *,
    session_id: str,
    full_name: str,
    email: Optional[str],
    phone: str,
) -> dict:
    """Silently notify the business after a non-empty trailer result batch is shown."""
    queued = _queue_email("results_shown", dict(
        session_id=session_id, full_name=full_name, email=email, phone=phone
    ))
    if queued:
        return queued
    summary_line = _append_conversation(TRAILER_RESULTS_SHOWN_DESCRIPTION, session_id)
    _send_async(send_faq_email_sync,
        full_name=full_name,
        email=email,
        phone=phone,
        subject=TRAILER_RESULTS_SHOWN_SUBJECT,
        summary_line=summary_line,
    )
    return {
        "status": "sent",
        "body_preview": (
            f"Name: {full_name}\n"
            f"Email: {(email or '').strip() or 'Not provided'}\n"
            f"Phone Number: {phone}\n\n"
            f"{summary_line}"
        ),
    }
