from __future__ import annotations

from typing import Optional

from src.email_sender import send_faq_email_sync, send_ticket_notification
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
    item = (item_name or "").strip()
    details = _append_conversation("", session_id).strip()
    send_ticket_notification(
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
    category = (faq_category or "contact_human").strip().lower()
    if category not in FAQ_CATEGORY_LABELS:
        category = "contact_human"
    summary_line = (summary or FAQ_CATEGORY_LABELS[category]).strip()
    if not summary_line.startswith(f"[{category}]"):
        summary_line = f"[{category}] {summary_line}"
    summary_line = _append_conversation(summary_line, session_id)
    send_faq_email_sync(
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
    summary_line = (summary or "Customer requested an action the chatbot cannot complete.").strip()
    if not summary_line.startswith("[Escalation Alert]"):
        summary_line = f"[Escalation Alert] {summary_line}"
    summary_line = _append_conversation(summary_line, session_id)
    send_faq_email_sync(
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
    summary_line = _append_conversation(TRAILER_RESULTS_SHOWN_DESCRIPTION, session_id)
    send_faq_email_sync(
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
