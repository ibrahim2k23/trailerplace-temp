from __future__ import annotations

from typing import Optional

from src.email_sender import send_faq_email_sync, send_ticket_notification
from src.conversation_store import promote_lead_to_hard, update_lead_item_of_interest

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
    send_ticket_notification(
        full_name=full_name,
        email=email,
        phone=phone,
        item_name=item,
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
) -> dict:
    """Tool 3: notify the business about contact, financing, trade-in, service, or store info."""
    category = (faq_category or "contact_human").strip().lower()
    if category not in FAQ_CATEGORY_LABELS:
        category = "contact_human"
    summary_line = (summary or FAQ_CATEGORY_LABELS[category]).strip()
    if not summary_line.startswith(f"[{category}]"):
        summary_line = f"[{category}] {summary_line}"
    message_line = (user_message or "").strip()
    if message_line:
        summary_line = f"{summary_line}\n\nLast user message: {message_line}"
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
    message_line = (user_message or "").strip()
    if message_line:
        summary_line = f"{summary_line}\n\nLast user message: {message_line}"
    context_line = (context_summary or "").strip()
    if context_line:
        summary_line = f"{summary_line}\n\nContext: {context_line}"
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
