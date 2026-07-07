from __future__ import annotations

import json
import logging
import os
import re
from copy import deepcopy
from difflib import SequenceMatcher
from threading import RLock
from typing import Any, Literal, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.chatbot.categories import resolve_category_from_text
from src.chatbot.graph import build_chatbot_graph
from src.chatbot.llm import make_llm
from src.chatbot.email_reply import compose_email_tool_reply
from src.chatbot.inventory_matcher import search_trailers
from src.chatbot.inventory_matcher import should_attempt_chat_lookup
from src.chatbot.inventory_matcher import is_potential_direct_inventory_lookup
from src.chatbot.inventory_matcher import validated_direct_inventory_extraction
from src.chatbot.make_resolver import resolve_make_from_text
from src.chatbot.prompts import (
    TRAILERPLACE_ACTION_SECTION,
    TRAILERPLACE_KNOWLEDGE_SECTION,
    TRAILERPLACE_PERSONA_SECTION,
    escalation_actions,
)
from src.chatbot.tools.email_tools import (
    capture_email_events,
    send_escalation_alert_email,
    send_interested_listing_email,
    send_non_sales_faq_email,
    send_trailer_results_shown_email,
)
from src.conversation_store import (
    close_session,
    create_or_get_soft_lead,
    deliver_pending_outbox,
    durable_turn,
    enqueue_upsert_conversation,
    persistence_enabled,
    persist_messages_snapshot,
    update_lead_contact,
)
from src.db_models import ChatbotConversation, ChatbotOutbox, ChatbotTurn
from src.models import ChatRequest, ChatResponse

load_dotenv()

logger = logging.getLogger(__name__)

_lock = RLock()
_sessions: dict[str, dict[str, Any]] = {}
_CHAT_MODEL = "gpt-4o-mini"
_GREETING_RE = re.compile(r"^\s*(hi|hello|hey|good\s+(morning|afternoon|evening))[\s!.]*$", re.I)
_GENERAL_INTENT_RE = re.compile(
    r"\b("
    r"trailer|haul|hauling|buy|looking|want|need|search|show|more|"
    r"recommend|recommendation|option|options|best|proceed|continue|"
    r"gooseneck|bumper\s*[- ]?\s*pull|hitch|payload|capacity|"
    r"financ(?:e|ing)|trade(?:-|\s)?in|service|parts|human|contact|"
    r"store|hours|location|interested"
    r")\b",
    re.I,
)


def _create_or_get_conversation(db_session: Any, *, session_id: str, lead_id: str, conversation: list[dict[str, Any]]) -> ChatbotConversation:
    """Atomically create a conversation, or return the row created elsewhere."""
    db_session.execute(
        pg_insert(ChatbotConversation)
        .values(session_id=session_id, lead_id=lead_id, conversation=conversation)
        .on_conflict_do_nothing(index_elements=[ChatbotConversation.session_id])
    )
    return db_session.get(ChatbotConversation, session_id)
_METADATA_UPDATE_RE = re.compile(
    r"\b(?:gooseneck|bumper\s*[- ]?\s*pull|hitch|payload|capacity|under|below|max|budget|"
    r"length|long|width|wide|make\s+it|change\s+it|black|white|gray|grey|silver|red|blue|"
    r"green|yellow|orange|tan)\b"
    r"|\$\s*\d"
    r"|\b\d+(?:\.\d+)?\s*(?:ft|feet|foot|'|lbs?|pounds?|#)\b"
    r"|\b\d+(?:\.\d+)?\s*[xX]\s*\d+(?:\.\d+)?\b",
    re.I,
)
_CONFUSION_ESCALATION_REPLY = (
    "I've forwarded your request to our sales department, and they will reach out to you soon."
)
_ESCALATION_SENT_REPLY = (
    "I've sent your query to our team, and they'll reach out to you soon. "
    "In the meantime, I can keep helping you narrow down the right trailer."
)
_CONFUSION_REPEAT_THRESHOLD = 2
_RESULT_NAV_CONFUSION_REPEAT_THRESHOLD = 3
_CONFUSION_HISTORY_WINDOW = 6
_CONFUSION_SCORE_THRESHOLD = 85
_CONFUSION_SIMILARITY_THRESHOLD = 0.86
_CONTACT_ONLY_ACK = (
    "Thanks for sharing your contact details. I've saved them. "
    "How can I help you today?"
)
_CONTACT_CONTINUE_ACK = "Thanks for sharing your contact details. I've saved them."


class ContactExtraction(BaseModel):
    full_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    name_confidence: Literal["none", "low", "medium", "high"] = "none"
    name_evidence: str = ""


class ConfusionDetectionDecision(BaseModel):
    similar_repeat_count: int = 0
    confused: bool = False
    confusion_score: int = 0
    is_answer_to_assistant_question: bool = False


class ContactPromptReplyDecision(BaseModel):
    action: Literal[
        "answer_contact_question",
        "acknowledge_contact_details",
        "acknowledge_and_continue",
        "decline_contact_details",
        "resume_saved_request",
        "resume_saved_request_with_update",
        "route_latest_request",
    ] = "resume_saved_request"
    remaining_message: Optional[str] = None
    reason: str = ""

class ContactPolicyDecision(BaseModel):
    violates_contact_policy: bool = False
    reason: str = ""


class ContactPolicyRewrite(BaseModel):
    assistant_text: str = ""


class InitialMessagePreservationDecision(BaseModel):
    has_meaningful_non_contact_intent: bool = False
    reason: str = ""


class RoutingDecision(BaseModel):
    # Single merged main-phase router (E5): replaces the catalogue-overview,
    # unsupported-business-action, and ROUTE/SMALLTALK gates.
    route: Literal["escalation", "catalogue_overview", "listings_followup", "other"] = "other"
    reason: str = ""


class ListingReferenceDecision(BaseModel):
    is_listing_selection: bool = False
    has_explicit_listing_reference: bool = False
    reference_intent: Literal["interest", "details", "none"] = "none"
    selected_index: Optional[int] = None
    selected_title: Optional[str] = None
    selected_url: Optional[str] = None
    confidence: Literal["low", "medium", "high"] = "low"
    reason: str = ""


class ListingReferenceIntentDecision(BaseModel):
    has_explicit_listing_reference: bool = False
    reference_kind: Literal["ordinal", "identifier", "demonstrative", "none"] = "none"
    reference_intent: Literal["interest", "details", "none"] = "none"
    confidence: Literal["low", "medium", "high"] = "low"
    reason: str = ""


def _model_dump(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _should_save_initial_message_for_resume(message: str) -> bool:
    # D1: Deterministic replacement for what used to be an LLM call on every first
    # turn. Preserve any non-empty opening message so the customer's request is
    # never lost behind the contact prompt; only a bare greeting (no other intent)
    # is not worth resuming. This removes one hot-path LLM call and the failure
    # mode where a transient outage dropped the saved request.
    text = (message or "").strip()
    if not text:
        return False
    if _GREETING_RE.match(text):
        return False
    return True


def _format_recent_message_transcript(messages: list[dict[str, Any]], limit: int = 4) -> str:
    lines: list[str] = []
    for item in (messages or [])[-limit:]:
        role = str(item.get("role") or "").strip().lower()
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        speaker = "User" if role == "user" else "Chatbot" if role == "assistant" else role.title() or "Message"
        lines.append(f"{speaker}:")
        lines.append(content)
        lines.append("")
    return "\n".join(lines).strip()


def _email_context_summary(session: dict[str, Any]) -> str:
    parts: list[str] = []
    category = str(session.get("trailer_category") or "").strip()
    if category:
        parts.append(f"Category: {category}")
    slots = session.get("slots_collected") or {}
    if slots:
        parts.append(f"Slots: {json.dumps(slots, ensure_ascii=True, default=str)}")
    metadata = session.get("metadata_filters_collected") or {}
    if metadata:
        parts.append(f"Metadata filters: {json.dumps(metadata, ensure_ascii=True, default=str)}")
    transcript = _format_recent_message_transcript(session.get("messages") or [], limit=4)
    if transcript:
        parts.append(f"Recent conversation:\n{transcript}")
    return " | ".join(parts)[:1800]


def _new_session(session_id: str) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "sales_phase": "main",
        "customer_full_name": None,
        "customer_full_name_confidence": "none",
        "customer_email": None,
        "customer_phone": None,
        "lead_id": None,
        "contact_status": "missing_contact",
        "messages": [],
        "active_search_request_text": "",
        "trailer_category": None,
        "pending_category_change": None,
        "pending_category_suggestion": None,
        "category_needs_clarification": False,
        "category_clarification_key": None,
        "slots_collected": {},
        "slots_skipped": [],
        "metadata_filters_collected": {},
        "defaulted_metadata_filters": [],
        "requested_non_metadata_features": [],
        "make_category_options": [],
        "awaiting_slot": None,
        "pending_questions": [],
        "asked_questions": [],
        "already_shown_listing_urls": [],
        "last_listings": [],
        "tool_events": [],
        "has_shown_search_results": False,
        "active_category_cycle_id": 1,
        "confusion_escalated": False,
        "confusion_signal_count": 0,
        "initial_contact_request_asked": False,
        "initial_contact_followup_asked": False,
        "awaiting_initial_contact_reply": False,
        "pending_contact_action": None,
        "pending_contact_actions": [],
        "active_question_text": None,
        "active_question_tracker": None,
        "active_question_unanswered_count": 0,
        "active_question_attempts": {},
        "pending_initial_user_message": None,
        "pending_results_notification": False,
    }


def _persist_email_transcript_snapshot(session: dict[str, Any]) -> None:
    persist_messages_snapshot(
        session_id=str(session.get("session_id") or ""),
        lead_id=session.get("lead_id"),
        messages=session.get("messages") or [],
    )


def _send_results_shown_notification(session: dict[str, Any]) -> None:
    """Send the internal results notification without affecting the customer response."""
    if not _has_contact(session):
        session["pending_results_notification"] = True
        return
    session["pending_results_notification"] = False
    try:
        _persist_email_transcript_snapshot(session)
        result = send_trailer_results_shown_email(
            session_id=session.get("session_id") or "",
            full_name=session.get("customer_full_name") or "",
            email=session.get("customer_email"),
            phone=session.get("customer_phone") or "",
        )
        logger.info(
            "trailer_results_shown_email_sent | session_id=%s | result=%s",
            session.get("session_id"),
            json.dumps(result, default=str),
        )
    except Exception:
        logger.exception(
            "trailer_results_shown_email_failed | session_id=%s",
            session.get("session_id"),
        )


def _send_pending_results_notification_if_ready(session: dict[str, Any]) -> None:
    if session.get("pending_results_notification") and _has_contact(session):
        _send_results_shown_notification(session)


def _get_session(session_id: str) -> dict[str, Any]:
    with _lock:
        if session_id not in _sessions:
            _sessions[session_id] = _new_session(session_id)
        return _sessions[session_id]


def reset_session(session_id: str) -> None:
    with _lock:
        _sessions.pop(session_id, None)
    close_session(session_id)


# Note: some contact classifiers pin gpt-4o-mini directly (model=_CHAT_MODEL) and
# deliberately do not consult OPENAI_MODEL; others resolve OPENAI_MODEL first.
# Both behaviors are preserved exactly through make_llm.
def _contact_llm():
    return make_llm(model=_CHAT_MODEL, structured_output=ContactExtraction)


def _confusion_llm():
    return make_llm(model=_CHAT_MODEL, structured_output=ConfusionDetectionDecision)


def _contact_prompt_reply_llm():
    return make_llm(structured_output=ContactPromptReplyDecision)


def _contact_prompt_bridge_llm():
    # temperature 0: this reply has hard content constraints (must not ask a
    # trailer question, must not repeat the saved request); determinism keeps it
    # on-policy and testable.
    return make_llm()


def _contact_policy_validator_llm():
    return make_llm(model=_CHAT_MODEL, structured_output=ContactPolicyDecision)


def _contact_policy_rewriter_llm():
    return make_llm(model=_CHAT_MODEL, structured_output=ContactPolicyRewrite)


def _initial_message_preservation_llm():
    return make_llm(model=_CHAT_MODEL, structured_output=InitialMessagePreservationDecision)


def _routing_llm():
    return make_llm(structured_output=RoutingDecision)


def _regex_contact(message: str) -> dict[str, Optional[str]]:
    email_match = re.search(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", message)
    phone_match = re.search(r"(?:\+?\d[\d\s().-]{7,}\d)", message)
    name = None
    name_match = re.search(
        r"\b(?:i am|i'm|im|my name is|name is)\s+([A-Za-z][A-Za-z .'-]{1,60})",
        message,
        re.IGNORECASE,
    )
    if name_match:
        name = re.split(
            r"\s+(?:and\s+)?(?:my\s+)?(?:email|phone|number)\b|\s+(?:call|reach)\s+me\b",
            name_match.group(1),
            flags=re.I,
        )[0]
        name = re.sub(r"\s+\band\b$", "", name, flags=re.I)
        name = re.sub(r"[,.;]+$", "", name).strip()
        if re.search(
            r"\b(looking|searching|trying|need|want|interested|shopping|browsing|asking|checking)\b",
            name,
            re.I,
        ):
            name = None
    return {
        "full_name": name,
        "email": email_match.group(0).strip() if email_match else None,
        "phone": phone_match.group(0).strip() if phone_match else None,
    }


_BAD_NAME_WORDS = {
    "comfortable",
    "sharing",
    "details",
    "detail",
    "why",
    "need",
    "looking",
    "interested",
    "trailer",
    "trailers",
    "option",
    "options",
    "search",
    "find",
    "want",
    "share",
    "contact",
    "email",
    "phone",
    "number",
}


def _is_plausible_person_name(name: str) -> bool:
    text = re.sub(r"\s+", " ", str(name or "").strip())
    if not text:
        return False
    if len(text) > 60 or "?" in text:
        return False
    if re.search(r"[\d@/:\\]|https?://|www\.", text, re.I):
        return False
    words = text.split()
    if not 1 <= len(words) <= 4:
        return False
    normalized_words = {re.sub(r"[^a-z]", "", word.lower()) for word in words}
    if normalized_words & _BAD_NAME_WORDS:
        return False
    if re.search(r"\b(i|me|my|you|your|we|our|they|them|such|could|would|please|tell)\b", text, re.I):
        return False
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z .'-]*", text))


def _is_explicit_name_statement(message: str) -> bool:
    return bool(
        re.search(
            r"\b(?:my name is|name is|i am|i'm|im|this is|call me|folks call me)\s+[A-Za-z]",
            message or "",
            re.I,
        )
        or re.search(r"\b[A-Za-z][A-Za-z .'-]{1,50}\s+here\b", message or "", re.I)
    )


def _should_accept_extracted_name(
    name: Any,
    confidence: str,
    message: str,
    existing_name: Any = None,
) -> bool:
    clean_name = re.sub(r"\s+", " ", str(name or "").strip())
    confidence = str(confidence or "none").strip().lower()
    if confidence not in {"medium", "high"}:
        return False
    if not _is_plausible_person_name(clean_name):
        return False
    if not existing_name:
        return True
    return confidence == "high" and _is_explicit_name_statement(message)


def _extract_contact(message: str, current: dict[str, Any]) -> dict[str, Any]:
    fallback = _regex_contact(message)
    try:
        result = _contact_llm().invoke(
            [
                SystemMessage(
                    content=("""
                        Extract customer contact details from the current user message only.

Return:

* full_name: string or null
* email: string or null
* phone: string or null
* name_confidence: "high", "medium", or "none"

Rules:

* Return null for missing fields.
* Do not guess or use previous context.
* Extract email and phone when clearly present.
* Extract a name only if the user clearly gives a personal name or nickname.
* Do not extract names from email usernames, greetings, trailer requests, trailer brands, product names, locations, objections, or questions.
* Use the current known values to interpret a contact-only reply. If exactly one contact field is
  missing and the message is a plausible standalone value for that field, extract it.
* A standalone alphabetic personal name can answer a request for the missing name.
* A standalone phone number can answer a request for the missing phone/contact method.
* A standalone email address can answer a request for the missing email/contact method.
* Do not treat "yes", "no", trailer categories, brands, stock numbers, weights, dimensions, or
  ordinary shopping replies as contact details.

Name confidence:

* Use "high" when the user explicitly identifies themselves.
  Examples: "my name is Ibrahim", "I am Ibrahim", "I'm Ibrahim", "this is Ibrahim", "Ibrahim here", "call me Tex", "mera naam Ibrahim hai".
* Use "medium" when a plausible standalone name or nickname appears with a phone/email.
  Also use "medium" for a plausible standalone name when name is the missing requested field.
  Examples: "Ibrahim", "Ibrahim 03304388550", "Ibrahim, ibrahim@example.com", "Ali Khan - 03304388550".
* Use "none" when no clear name is given.
* Known name=null, email="mk@gmail.com"; message "ibrahim" -> full_name="Ibrahim", name_confidence="medium".
* Known phone=null; message "03304388550" -> phone="03304388550".
* Known email=null; message "ibrahim@example.com" -> email="ibrahim@example.com".

Examples:

* "Ibrahim 03304388550" → full_name="Ibrahim", phone="03304388550", name_confidence="medium"
* "My name is Ibrahim, 03304388550" → full_name="Ibrahim", phone="03304388550", name_confidence="high"
* "I need a trailer, 03304388550" → full_name=null, phone="03304388550", name_confidence="none"
* "I'm looking for car haulers, [ibrahim@example.com](mailto:ibrahim@example.com)" → full_name=null, email="[ibrahim@example.com](mailto:ibrahim@example.com)", name_confidence="none"
* "Do you have Big Tex trailers? 03304388550" → full_name=null, phone="03304388550", name_confidence="none"

When unsure, do not extract a name.
"""
                    )
                ),
                HumanMessage(
                    content=(
                        f"Current known values: name={current.get('customer_full_name')!r}, "
                        f"email={current.get('customer_email')!r}, phone={current.get('customer_phone')!r}\n"
                        f"Message: {message}"
                    )
                ),
            ]
        )
        data = _model_dump(result)
    except Exception:
        logger.exception("Contact extraction LLM failed; using regex fallback")
        return fallback

    extracted = {
        "full_name": data.get("full_name"),
        "email": data.get("email"),
        "phone": data.get("phone"),
        "name_confidence": data.get("name_confidence") or "none",
        "name_evidence": data.get("name_evidence") or "",
    }
    if not _should_accept_extracted_name(
        extracted.get("full_name"),
        str(extracted.get("name_confidence") or "none"),
        message,
        current.get("customer_full_name"),
    ):
        extracted["full_name"] = None
    if any(extracted.get(key) for key in ("full_name", "email", "phone")):
        return extracted
    fallback_confidence = "high" if fallback.get("full_name") and _is_explicit_name_statement(message) else "none"
    if fallback.get("full_name") and not _should_accept_extracted_name(
        fallback.get("full_name"),
        fallback_confidence,
        message,
        current.get("customer_full_name"),
    ):
        fallback["full_name"] = None
    return {
        "full_name": fallback.get("full_name"),
        "email": fallback.get("email"),
        "phone": fallback.get("phone"),
        "name_confidence": fallback_confidence if fallback.get("full_name") else "none",
        "name_evidence": "",
    }


def _has_contact(session: dict[str, Any]) -> bool:
    return bool(
        session.get("customer_full_name")
        and (session.get("customer_phone") or session.get("customer_email"))
    )


def _has_full_initial_details(session: dict[str, Any]) -> bool:
    return _has_contact(session)


def _sync_contact_status(session: dict[str, Any]) -> None:
    if _has_contact(session):
        session["contact_status"] = "contact_available"
    elif session.get("contact_status") != "contact_declined":
        session["contact_status"] = "missing_contact"


def _contact_request_allowed(session: dict[str, Any]) -> bool:
    if _has_contact(session):
        return False
    return bool(session.get("pending_contact_action") or session.get("pending_contact_actions"))


def _enforce_contact_response_policy(
    session: dict[str, Any], assistant_text: str, active_question: str | None = None
) -> str:
    if session.get("contact_status") != "contact_declined" or _contact_request_allowed(session):
        return assistant_text
    email_address = r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"
    us_phone = (
        r"(?<!\d)(?:\+?1[\s.-]?)?"
        r"(?:\([2-9]\d{2}\)|[2-9]\d{2})[\s.-]?"
        r"\d{3}[\s.-]?\d{4}(?!\d)"
    )
    contact_request = (
        r"\b(?:provide|share|send|give|enter|confirm|need|collect|request|ask\s+for)\b"
        r"[^.!?\n]{0,80}\b(?:e-?mail(?:\s+address)?|phone(?:\s+number)?|"
        r"contact\s+(?:details?|information|info)|callback\s+(?:details?|number))\b"
    )
    # The dealership answering with its OWN phone/site (e.g. a "how do I contact
    # you" FAQ) is not soliciting the declined customer's contact — strip those
    # from the probe so they do not trip the guard.
    probe = str(assistant_text or "")
    for own in ("979-532-1486", "979.532.1486", "9795321486", "trailerplace.com"):
        probe = probe.replace(own, " ")
    if not re.search(
        rf"(?:{email_address})|(?:{us_phone})|(?:{contact_request})",
        probe,
        flags=re.IGNORECASE,
    ):
        return assistant_text
    logger.warning("contact_response_policy_regex_blocked")
    return str(active_question or "What type of trailer are you looking for?").strip()


def _ensure_lead(session: dict[str, Any]) -> None:
    if session.get("lead_id"):
        return
    try:
        session["lead_id"] = create_or_get_soft_lead(
            session_id=session["session_id"],
            full_name=session.get("customer_full_name"),
            email=session.get("customer_email"),
            phone=session.get("customer_phone"),
            item_of_interest="Trailer inquiry",
        )
    except Exception:
        logger.exception("Soft lead creation failed")


def _persist_contact(session: dict[str, Any]) -> None:
    _sync_contact_status(session)
    try:
        lead_id = update_lead_contact(
            session_id=session["session_id"],
            full_name=session.get("customer_full_name"),
            email=session.get("customer_email"),
            phone=session.get("customer_phone"),
        )
        if lead_id:
            session["lead_id"] = lead_id
    except Exception:
        logger.exception("Lead contact update failed")


def _apply_contact_from_request_and_message(session: dict[str, Any], request: ChatRequest) -> bool:
    changed = False
    was_sufficient = _has_contact(session)
    if request.customer_full_name and (
        not session.get("customer_full_name")
        or _should_accept_extracted_name(
            request.customer_full_name,
            "high",
            request.message,
            session.get("customer_full_name"),
        )
    ):
        session["customer_full_name"] = request.customer_full_name
        session["customer_full_name_confidence"] = "high"
        changed = True
    if request.customer_email is not None and request.customer_email and not session.get("customer_email") and not was_sufficient:
        session["customer_email"] = request.customer_email or None
        changed = True
    if request.customer_phone and not session.get("customer_phone") and not was_sufficient:
        session["customer_phone"] = request.customer_phone
        changed = True

    should_extract_from_message = (
        not _has_contact(session)
        or (
            str(session.get("customer_full_name_confidence") or "none") != "high"
            and _is_explicit_name_statement(request.message)
        )
    )
    if not was_sufficient and should_extract_from_message:
        contact = _extract_contact(request.message, session)
        name = contact.get("full_name")
        if name and _should_accept_extracted_name(
            name,
            str(contact.get("name_confidence") or "none"),
            request.message,
            session.get("customer_full_name"),
        ):
            if name != session.get("customer_full_name"):
                session["customer_full_name"] = name
                session["customer_full_name_confidence"] = str(contact.get("name_confidence") or "medium")
                changed = True
        for source, target in (
            ("email", "customer_email"),
            ("phone", "customer_phone"),
        ):
            value = contact.get(source)
            if value and not session.get(target):
                session[target] = value
                changed = True
    _sync_contact_status(session)
    if changed:
        _persist_contact(session)
    return changed


def _initial_contact_request_text(session: dict[str, Any]) -> str:
    missing: list[str] = []
    if not session.get("customer_full_name"):
        missing.append("name")
    if not (session.get("customer_email") or session.get("customer_phone")):
        missing.append("email or phone number")
    if not missing:
        return ""
    if len(missing) == 2 and missing == ["name", "email or phone number"]:
        fields = "your name, email, and phone number"
    elif len(missing) == 2:
        fields = f"your {missing[0]} and {missing[1]}"
    else:
        fields = "your email and phone number (either one is enough)" if missing[0] == "email or phone number" else f"your {missing[0]}"
    return (
        "Thank you for contacting TrailerPlace. Before we get started, could I get "
        f"{fields}? Sharing contact details is optional, and I can still help with your trailer search."
    )


def _initial_contact_followup_text(session: dict[str, Any]) -> str:
    if session.get("customer_full_name"):
        return (
            "Thanks, I've saved your name. Could I also get either your email address or phone number? "
            "Sharing it is optional, and I can continue with your trailer search either way."
        )
    return (
        "Thanks, I've saved that contact method. Could I also get your name? "
        "Sharing it is optional, and I can continue with your trailer search either way."
    )


def _is_contact_refusal(message: str) -> bool:
    text = (message or "").strip().lower()
    return bool(
        re.search(
            r"\b(no|nope|nah|skip|pass|not now|don't want|do not want|rather not|"
            r"won't share|will not share|dont want)\b",
            text,
        )
    )


def _is_contact_explanation_question(message: str) -> bool:
    text = (message or "").strip().lower()
    return bool(
        re.search(
            r"\bwhy\b.*\b(?:need|want|ask|asking|require|get|collect)\b.*\b(?:contact|details?|info|information|email|phone|number|name|them|this)\b"
            r"|\b(?:what|how)\b.*\b(?:use|using)\b.*\b(?:contact|details?|info|information|email|phone|number|name)\b"
            r"|\b(?:why|what for)\b",
            text,
        )
    )


def _contact_explanation_text() -> str:
    return (
        "We ask because it helps our team contact you later if the need arises, "
        "for example if there is a good trailer match or a follow-up detail to confirm. "
        "It is optional, and I can keep helping here."
    )


def _classify_contact_prompt_reply(session: dict[str, Any], latest_message: str) -> ContactPromptReplyDecision:
    pending = str(session.get("pending_initial_user_message") or "").strip()
    try:
        result = _contact_prompt_reply_llm().invoke(
            [
                SystemMessage(
                    content=(
                            "The customer just replied to an optional contact request. Decide the next action.\n\n"

                            "## CONTACT VALUE RECOGNITION\n"
                            "Treat standalone names (e.g. 'ibrahim', 'john'), phone numbers (e.g. '03304388550', '033-438-8550'), "
                            "or emails (e.g. 'ibrahim@example.com') as partial or complete contact replies — not new trailer requests. "
                            "This applies whether the assistant asked for all fields or one specific missing field.\n\n"

                            "## ACTION RULES (evaluate in order)\n"
                            "1. `answer_contact_question` — reply asks why contact is needed, how it will be used, or if it's required.\n"
                            "2. `acknowledge_contact_details` — reply supplies name + phone or name + email AND no saved request exists. "
                            "   Use this only when no meaningful non-contact content remains.\n"
                            "3. `acknowledge_and_continue` — reply supplies contact details AND also contains any meaningful "
                            "   non-contact content: an answer, question, request, correction, preference, small talk, or requirement. "
                            "   Set remaining_message to that content with only contact-related text removed. Preserve its meaning, "
                            "   wording, constraints, and multiple requirements; do not classify or answer it.\n"
                            "4. `resume_saved_request` — reply only provides or declines contact details AND a saved original request exists. "
                            "   Store any supplied field, then resume the saved request.\n"
                            "5. `resume_saved_request_with_update` — reply adds or refines the saved "
                            "   request, including 'it should be 20ft', 'make it gooseneck', or 'I will haul cattle'.\n"
                            "   If contact details are also present, put only the non-contact update in remaining_message.\n"
                            "6. `decline_contact_details` — user declines or skips contact and makes no other actionable request.\n"
                            "7. `route_latest_request` — reply clearly replaces the saved request with a different trailer, "
                            "   store, financing, service, parts, "
                            "   trade-in, human-contact, or inventory request that should replace the saved request.\n\n"

                            "Judge intent from context, not fixed wording."
                        )
                ),
                HumanMessage(
                    content=(
                        f"Saved original request: {pending!r}\n"
                        f"Contact now stored: {_has_contact(session)!r}\n"
                        f"Latest user reply after optional contact prompt: {latest_message!r}"
                    )
                ),
            ]
        )
        decision = _model_dump(result)
        action = str(decision.get("action") or "resume_saved_request")
        allowed = {
            "answer_contact_question", "acknowledge_contact_details", "acknowledge_and_continue",
            "decline_contact_details",
            "resume_saved_request", "resume_saved_request_with_update", "route_latest_request",
        }
        if action not in allowed:
            action = "route_latest_request"
        remaining_message = str(decision.get("remaining_message") or "").strip() or None
        if action == "acknowledge_and_continue" and not remaining_message:
            action = "route_latest_request"
        logger.info(
            "contact_prompt_reply_decision | action=%s | reason=%r",
            action,
            decision.get("reason") or "",
        )
        return ContactPromptReplyDecision(
            action=action,
            remaining_message=remaining_message,
            reason=str(decision.get("reason") or ""),
        )
    except Exception:
        logger.exception("Contact prompt reply LLM failed; using deterministic fallback")
        # Deterministic recovery so a transient LLM outage never discards the
        # customer's saved original request. If they refused/skipped contact,
        # resume the saved request; if they clearly stated a new actionable
        # request, route that; otherwise default to resuming (never lose intent).
        if pending and _is_contact_refusal(latest_message):
            return ContactPromptReplyDecision(
                action="resume_saved_request", reason="LLM unavailable; refusal detected, resuming saved request."
            )
        if _has_actionable_intent(latest_message):
            return ContactPromptReplyDecision(
                action="route_latest_request", reason="LLM unavailable; latest message looks actionable."
            )
        if pending:
            return ContactPromptReplyDecision(
                action="resume_saved_request", reason="LLM unavailable; resuming saved request by default."
            )
        return ContactPromptReplyDecision(action="route_latest_request", reason="LLM unavailable.")


def _should_resume_pending_after_contact_ask(session: dict[str, Any], latest_message: str) -> bool:
    return _classify_contact_prompt_reply(session, latest_message).action == "resume_saved_request"


def _contact_prompt_bridge_text(
    *,
    action: str,
    latest_message: str,
    saved_request: str,
) -> str:
    try:
        response = _contact_prompt_bridge_llm().invoke(
            [
                SystemMessage(
                    content=(
                        "Write a short, natural opening for a TrailerPlace sales chat response. "
                        "The user just replied to an optional contact-details request. "
                        "Address only that contact-related reply. "
                        "Write acknowledgment text only. Never include or anticipate the next trailer question, "
                        "category list, recommendation, or search response; the application appends that separately. "
                        "Do not answer the trailer request yourself, do not mention specific inventory, "
                        "do not repeat the full saved request, and do not ask any trailer-search or "
                        "qualification question. "
                        "Never ask what type, kind, size, category, length, width, weight, hitch, color, "
                        "or budget the customer wants; the next response will handle trailer details. "
                        "If action is answer_contact_question, explain that contact details help the team "
                        "reach the customer later if the need arises and that sharing them is optional. "
                        "If action is resume_saved_request and contact details were provided, thank them briefly and naturally. "
                        "If action is decline_contact_details, never thank them for sharing contact details. Warmly respect "
                        "their choice without guilt or pressure, reassure them that it is no problem, and smoothly keep them "
                        "engaged with their trailer request. Use natural sales-friendly wording such as 'No problem at all—I "
                        "understand your preference, and I'll continue helping with your trailer search.' Vary the wording naturally "
                        "rather than copying a fixed template. Apply the same respectful approach when resume_saved_request "
                        "represents a refusal or skip. For resume_saved_request_with_update, briefly acknowledge that you "
                        "will continue with the added requirement; do not repeat or interpret the trailer details yourself. "
                        "Never use stalling or readiness language such as 'whenever you're ready', 'when you're ready', "
                        "or 'I'm here to help whenever'. The trailer response immediately following this opening continues the flow. "
                        "Keep it to 1-2 concise sentences, no markdown list. End without a question mark."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Action: {action}\n"
                        f"User contact reply: {latest_message!r}\n"
                        f"An original trailer request will resume: {bool(saved_request)!r}"
                    )
                ),
            ]
        )
        text = re.sub(r"\s+", " ", str(response.content or "")).strip()
        return text
    except Exception:
        logger.exception("Contact prompt bridge LLM failed; using fallback bridge")
        if action == "answer_contact_question":
            return _contact_explanation_text()
        return "No problem, you do not have to share contact details. I can keep helping here."


def _without_latest_user_message(messages: list[dict[str, Any]], latest_message: str) -> list[dict[str, Any]]:
    if not messages:
        return []
    last = messages[-1]
    if (
        isinstance(last, dict)
        and last.get("role") == "user"
        and str(last.get("content") or "") == str(latest_message or "")
    ):
        return list(messages[:-1])
    return list(messages)

def _replace_latest_user_message(
    messages: list[dict[str, Any]], original: str, replacement: str
) -> list[dict[str, Any]]:
    updated = list(messages)
    if updated and updated[-1].get("role") == "user" and str(updated[-1].get("content") or "") == original:
        updated[-1] = {**updated[-1], "content": replacement}
    return updated


def _active_question_followup(session: dict[str, Any]) -> str:
    awaiting = str(session.get("awaiting_slot") or "").strip()
    pending = session.get("pending_questions") or []
    if awaiting:
        for item in pending:
            if str(item.get("slot") or "") == awaiting:
                return str(item.get("question") or "").strip()
    if pending:
        return str(pending[0].get("question") or "").strip()
    return ""


def _append_active_question_if_present(session: dict[str, Any], text: str) -> str:
    question = _active_question_followup(session)
    base = str(text or "").strip()
    if question and question not in base:
        return f"{base}\n\n{question}" if base else question
    return base


def _send_one_pending_contact_action_if_ready(
    session: dict[str, Any],
) -> tuple[str | None, dict[str, Any] | None]:
    action = session.get("pending_contact_action")
    if not action or not _has_contact(session):
        return None, None
    action_type = action.get("type")
    if action_type == "interest":
        item_name = str(action.get("item_name") or "that trailer").strip()
        _persist_email_transcript_snapshot(session)
        result = send_interested_listing_email(
            session_id=session.get("session_id") or "",
            full_name=session.get("customer_full_name") or "",
            email=session.get("customer_email"),
            phone=session.get("customer_phone") or "",
            item_name=item_name,
        )
        logger.info("deferred_contact_action_sent | type=interest | result=%s", json.dumps(result, default=str))
        session["pending_contact_action"] = None
        fallback = (
            f"Thanks, I saved your contact information and sent your interest in \"{item_name}\" "
            "to our team so they can follow up."
        )
        reply = compose_email_tool_reply(
            email_purpose=f"customer interest in {item_name}",
            latest_message=str((session.get("messages") or [{}])[-1].get("content") or ""),
            conversation_context=_email_context_summary(session),
            fallback=fallback,
        )
        return reply, {"tool": "send_interested_listing_email", "result": result}
    if action_type == "faq":
        category = str(action.get("faq_category") or "contact_human")
        summary = str(action.get("summary") or "")
        _persist_email_transcript_snapshot(session)
        result = send_non_sales_faq_email(
            session_id=session.get("session_id") or "",
            full_name=session.get("customer_full_name") or "",
            email=session.get("customer_email"),
            phone=session.get("customer_phone") or "",
            faq_category=category,
            summary=summary,
            user_message=str(action.get("user_message") or ""),
            context_summary=str(action.get("context_summary") or _email_context_summary(session)),
        )
        logger.info("deferred_contact_action_sent | type=faq | result=%s", json.dumps(result, default=str))
        session["pending_contact_action"] = None
        reply = compose_email_tool_reply(
            email_purpose=summary or category,
            latest_message=str((session.get("messages") or [{}])[-1].get("content") or ""),
            conversation_context=_email_context_summary(session),
            fallback="Thanks, I saved your contact information and sent that request to our team so they can help.",
        )
        return reply, {"tool": "send_non_sales_faq_email", "result": result}
    if action_type == "escalation_alert":
        _persist_email_transcript_snapshot(session)
        result = send_escalation_alert_email(
            session_id=session.get("session_id") or "",
            full_name=session.get("customer_full_name") or "",
            email=session.get("customer_email"),
            phone=session.get("customer_phone") or "",
            summary=str(action.get("summary") or "Customer requested an unsupported business action."),
            user_message=str(action.get("user_message") or ""),
            context_summary=str(action.get("context_summary") or _email_context_summary(session)),
        )
        logger.info("deferred_contact_action_sent | type=escalation_alert | result=%s", json.dumps(result, default=str))
        session["pending_contact_action"] = None
        reply = compose_email_tool_reply(
            email_purpose=str(action.get("summary") or "customer escalation request"),
            latest_message=str((session.get("messages") or [{}])[-1].get("content") or ""),
            conversation_context=_email_context_summary(session),
            fallback=_ESCALATION_SENT_REPLY,
        )
        return reply, {"tool": "send_escalation_alert_email", "result": result}
    session["pending_contact_action"] = None
    return None, None


def _send_pending_contact_action_if_ready(
    session: dict[str, Any],
) -> tuple[str | None, list[dict[str, Any]]]:
    actions = list(session.get("pending_contact_actions") or [])
    legacy = session.get("pending_contact_action")
    if legacy and legacy not in actions:
        actions.append(legacy)
    if not actions or not _has_contact(session):
        return None, []
    session["pending_contact_actions"] = []
    session["pending_contact_action"] = None
    replies: list[str] = []
    tool_events: list[dict[str, Any]] = []
    for action in actions:
        session["pending_contact_action"] = action
        reply, tool_event = _send_one_pending_contact_action_if_ready(session)
        if reply and reply not in replies:
            replies.append(reply)
        if tool_event:
            tool_events.append(tool_event)
    combined = "\n\n".join(replies) if replies else None
    if combined:
        combined = _append_active_question_if_present(session, combined)
    tracker = dict(session.get("active_question_tracker") or {})
    if combined and tracker.get("confusion_sent"):
        combined = _strip_repeated_question(combined, str(tracker.get("question") or ""))
    return combined, tool_events


def _main_smalltalk_response(session: dict[str, Any], user_message: str) -> str:
    try:
        # temperature 0.2: smalltalk allows a little phrasing variety, but the
        # prompt has hard constraints (no greeting, no contact ask). 0.4 was high
        # enough to leak those constraints; 0.2 keeps some warmth while staying
        # on-policy.
        response = make_llm(model=_CHAT_MODEL, temperature=0.2).invoke(
            [
                SystemMessage(
                    content=(
                        f"{TRAILERPLACE_PERSONA_SECTION}\n\n"
                        f"{TRAILERPLACE_KNOWLEDGE_SECTION}\n\n"
                        f"{TRAILERPLACE_ACTION_SECTION}\n\n"
                        "You are the Trailer Place sales chat assistant. Reply naturally to the "
                        "latest message, then invite them to share what trailer or service "
                        "help they need. If they ask what TrailerPlace has, carries, sells, or "
                        "what services are offered, say TrailerPlace carries many trailer types "
                        "including utility, dump, equipment, flatbed, car hauler, livestock, "
                        "enclosed, tilt, roll-off, and gooseneck trailer options with bumper pull "
                        "or gooseneck hitch setups depending on model, and can help with financing, "
                        "trade-ins, delivery, and service or spare parts. Do not mention rentals, "
                        "repairs, or custom modifications unless the user explicitly asks. Do not "
                        "ask for contact details here. Keep it brief."
                        "Your purpose is to inform the user. not greet them. So do not greet the user or ask how they are doing. Just reply to their message. "
                        "Do not claim that an email or escalation was sent from smalltalk; those actions must be routed through graph tools."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Customer name: {session.get('customer_full_name') or ''}\n"
                        f"Recent messages: {(session.get('messages') or [])[-6:]}\n"
                        f"Latest user message: {user_message}"
                    )
                ),
            ]
        )
        text = str(response.content or "").strip()
        return text or "How can I help you today?"
    except Exception:
        logger.exception("Main smalltalk LLM failed; using fallback response")
        return "How can I help you today?"


def _has_actionable_intent(message: str) -> bool:
    text = (message or "").strip()
    if not text or _GREETING_RE.match(text):
        return False
    category = resolve_category_from_text(text)
    if category.category or category.needs_clarification:
        return True
    return bool(_GENERAL_INTENT_RE.search(text))


def _has_trailer_search_context(session: dict[str, Any]) -> bool:
    return bool(
        session.get("trailer_category")
        or session.get("slots_collected")
        or session.get("metadata_filters_collected")
        or session.get("make_category_options")
        or session.get("last_listings")
        or session.get("already_shown_listing_urls")
    )


def _has_metadata_update_intent(message: str) -> bool:
    text = (message or "").strip()
    if not text:
        return False
    if _METADATA_UPDATE_RE.search(text):
        return True
    return bool(resolve_make_from_text(text, use_llm_fallback=False).make)



def _route_decision(session: dict[str, Any], user_message: str, has_prior_listings: bool) -> RoutingDecision:
    """Merged main-phase router (E5): one structured call replacing the
    catalogue-overview, unsupported-business-action, and ROUTE/SMALLTALK gates.
    Called only after the deterministic pre-checks are inconclusive."""
    text = (user_message or "").strip()
    if not text or not os.getenv("OPENAI_API_KEY"):
        return RoutingDecision(route="other", reason="no_text_or_key")
    try:
        decision = _routing_llm().invoke(
            [
                SystemMessage(
                    content=(
                        "Classify the latest customer message into exactly one route for a trailer-sales "
                        "assistant. Return structured fields only. Evaluate in this priority order:\n\n"
                        f"1. route='escalation' — the customer asks TrailerPlace/the team to perform a real-world "
                        f"action or commitment the chatbot cannot complete directly: {escalation_actions()}. "
                        "Action requests only ('reserve this', 'remind me tomorrow', 'send a quote', 'schedule a "
                        "call at 3 PM'). Requests to stop/skip questions or to show results/options/inventory are "
                        "NOT escalation. Information questions ('what does it cost?', 'do you offer financing?', "
                        "'how do reservations work?') are NOT escalation.\n"
                        "2. route='catalogue_overview' — the customer asks broadly what trailer types, options, "
                        "lineup, inventory categories, products, or services TrailerPlace has/carries/sells, without "
                        "enough specific constraints to search ('what trailers do you offer?', 'what are the "
                        "options?', 'what do you guys carry?'). NOT this when they give constraints "
                        "(category/length/make/budget/hitch/features), want recommendations for a use case, ask to "
                        "show/search trailers, answer a qualification question, express purchase interest, ask about "
                        "a specific listing, or ask for more/next options after results.\n"
                        "3. route='listings_followup' — the message could refer to previously shown listings: "
                        "selection, ordinal reference ('the 4th one'), follow-up constraints, asking for more "
                        "options, or purchase intent.\n"
                        "4. route='other' — none of the above (ordinary smalltalk or a question answerable without a tool)."
                    )
                ),
                HumanMessage(
                    content=json.dumps(
                        {
                            "latest_user_message": text,
                            "awaiting_slot": session.get("awaiting_slot"),
                            "pending_questions": session.get("pending_questions") or [],
                            "trailer_category": session.get("trailer_category"),
                            "slots_collected": session.get("slots_collected") or {},
                            "metadata_filters_collected": session.get("metadata_filters_collected") or {},
                            "has_prior_listings": has_prior_listings,
                            "recent_messages": (session.get("messages") or [])[-6:],
                        },
                        default=str,
                    )
                ),
            ]
        )
        logger.info("route_decision | route=%s | reason=%r | latest_message=%r", decision.route, decision.reason, text)
        return decision
    except Exception:
        logger.exception("Merged router failed; keeping existing routing behavior")
        return RoutingDecision(route="other", reason="exception")


def _should_route_to_graph(session: dict[str, Any], user_message: str) -> bool:
    """
    LLM-first routing for the main phase.
    This prevents regex misses (for example "I like the 4th one") from falling into smalltalk.
    """
    if not _has_contact(session) and _is_contact_only_message(user_message):
        return False

    if session.get("awaiting_slot") or session.get("pending_questions"):
        return True

    if session.get("pending_category_suggestion"):
        return True

    if _has_trailer_search_context(session) and _has_metadata_update_intent(user_message):
        return True

    # Keep deterministic fast-path for obvious intent. (Escalation would also
    # route to graph, so evaluating this before the merged router below changes
    # no outcome and saves an LLM call for these messages.)
    if _has_actionable_intent(user_message):
        return True

    # One merged router replaces the escalation / catalogue-overview /
    # ROUTE-SMALLTALK gates. Map its route to the same precedence as before:
    # escalation -> graph; catalogue overview -> not graph; a listings follow-up
    # -> graph only when listings were shown; otherwise smalltalk.
    has_prior_listings = bool(session.get("last_listings") or session.get("already_shown_listing_urls"))
    route = _route_decision(session, user_message, has_prior_listings).route
    if route == "escalation":
        return True
    if route == "catalogue_overview":
        return False
    if not has_prior_listings:
        return False
    return route == "listings_followup"


def _is_contact_only_message(message: str) -> bool:
    if _has_explicit_confusion_language(message):
        return False
    contact = _regex_contact(message or "")
    has_contact_detail = any(contact.get(key) for key in ("full_name", "email", "phone"))
    return has_contact_detail and not _has_actionable_intent(message) and not _has_listing_followup_intent(message)


def _has_listing_followup_intent(message: str) -> bool:
    text = (message or "").strip().lower()
    if not text:
        return False
    return bool(
        re.search(
            r"\b(?:interested|interest|like|want|take|buy|purchase|call|contact|follow\s*up)\b.*"
            r"\b(?:trailer|item|listing|option|one|first|second|third|fourth|fifth|#?\d+)\b"
            r"|\b(?:trailer|item|listing|option|one|first|second|third|fourth|fifth|#?\d+)\b.*"
            r"\b(?:interested|interest|like|want|take|buy|purchase|call|contact|follow\s*up)\b",
            text,
        )
    )


def _has_listing_context(session: dict[str, Any]) -> bool:
    return bool(session.get("last_listings") or session.get("already_shown_listing_urls"))


def _is_result_navigation_request(message: str) -> bool:
    text = (message or "").strip().lower()
    if not text:
        return False
    return bool(
        re.search(
            r"\bshow\s+more(?:\s+(?:results?|options?|trailers?|like\s+this))?\b"
            r"|\bmore(?:\s+(?:results?|options?|trailers?|like\s+this))?\b"
            r"|\bnext(?:\s+(?:ones?|results?|options?|trailers?))?\b"
            r"|\bany\s+others?\b"
            r"|\banother\s+(?:one|option|trailer|result)\b"
            r"|\bshow\s+me\s+another\b",
            text,
        )
    )


def _confusion_repeat_threshold_for_message(session: dict[str, Any], message: str) -> tuple[int, str]:
    if _has_explicit_confusion_language(message):
        return 1, "explicit_confusion"
    if _has_listing_context(session) and _is_result_navigation_request(message):
        return _RESULT_NAV_CONFUSION_REPEAT_THRESHOLD, "result_navigation"
    return _CONFUSION_REPEAT_THRESHOLD, "default"


def _has_store_or_contact_info_question(message: str) -> bool:
    text = (message or "").strip().lower()
    if not text:
        return False
    return bool(
        re.search(
            r"\b(?:how|what|where|when|can|could|do)\b.*\b(?:contact|call|phone|number|reach|location|located|address|hours|open|store|sales)\b"
            r"|\b(?:contact|call|phone|number|reach|location|located|address|hours|open|store|sales)\b.*\b(?:you|guys|trailerplace|team)\b",
            text,
        )
    )


def _has_explicit_confusion_language(message: str) -> bool:
    text = (message or "").strip().lower()
    if not text:
        return False
    return bool(
        re.search(
            r"\b(?:i\s+am|i'm|im|getting|feel(?:ing)?)\s+confused\b"
            r"|\bconfused\b"
            r"|\bi\s+don['’]?t\s+understand\b"
            r"|\bthis\s+is\s+not\s+helping\b"
            r"|\bi\s+don['’]?t\s+know\s+what\s+to\s+choose\b"
            r"|\bi\s+keep\s+asking\b"
            r"|\bsame\s+thing\s+again\b",
            text,
        )
    )


def _normalized_confusion_text(message: str) -> str:
    text = (message or "").lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\b\d{7,}\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _messages_materially_similar(left: str, right: str) -> bool:
    left_norm = _normalized_confusion_text(left)
    right_norm = _normalized_confusion_text(right)
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm:
        return True
    if left_norm in right_norm or right_norm in left_norm:
        return min(len(left_norm), len(right_norm)) >= 12
    return SequenceMatcher(None, left_norm, right_norm).ratio() >= _CONFUSION_SIMILARITY_THRESHOLD


def _has_repeated_confusion_candidate(
    session: dict[str, Any],
    latest_message: str,
    recent_user_turns: list[str],
    repeat_threshold: int,
) -> bool:
    if _has_listing_context(session) and _is_result_navigation_request(latest_message):
        return sum(1 for message in recent_user_turns if _is_result_navigation_request(message)) >= repeat_threshold
    return (
        sum(
            1
            for message in recent_user_turns
            if _messages_materially_similar(latest_message, message)
        )
        >= repeat_threshold
    )


def _is_confusion_eligible_user_message(message: str, session: dict[str, Any] | None = None) -> bool:
    if _is_contact_only_message(message):
        return False
    if _has_store_or_contact_info_question(message):
        return False
    if _has_explicit_confusion_language(message):
        return True
    if session and _has_listing_context(session) and _is_result_navigation_request(message):
        return True
    return _has_actionable_intent(message)


def _recent_confusion_user_messages(session: dict[str, Any], limit: int = _CONFUSION_HISTORY_WINDOW) -> list[str]:
    messages = session.get("messages") or []
    user_turns = [
        str(m.get("content") or "").strip()
        for m in messages
        if (
            isinstance(m, dict)
            and m.get("role") == "user"
            and str(m.get("content") or "").strip()
            and _is_confusion_eligible_user_message(str(m.get("content") or ""), session)
        )
    ]
    return user_turns[-limit:]


def _is_confused_user_turn(session: dict[str, Any], user_message: str) -> tuple[bool, int]:
    if session.get("awaiting_slot") or session.get("pending_questions"):
        return False, 0
    if not _is_confusion_eligible_user_message(user_message, session):
        return False, 0
    recent_user_turns = _recent_confusion_user_messages(session)
    repeat_threshold, threshold_type = _confusion_repeat_threshold_for_message(session, user_message)
    has_explicit_confusion = _has_explicit_confusion_language(user_message)
    if len(recent_user_turns) < repeat_threshold:
        return False, 0
    if not has_explicit_confusion and not _has_repeated_confusion_candidate(
        session,
        user_message,
        recent_user_turns,
        repeat_threshold,
    ):
        return False, 0

    try:
        recent_messages = (session.get("messages") or [])[-8:]
        result = _confusion_llm().invoke(
            [
                SystemMessage(
                    content=(
                        "Detect whether the latest user message shows genuine customer confusion. "
                        "Return structured output with:\n"
                        "- similar_repeat_count: number of recent user messages including the latest that are materially similar to the latest.\n"
                        "- confusion_score: 0-100 confidence that the customer is genuinely confused and needs human follow-up.\n"
                        "- is_answer_to_assistant_question: true when the latest message is a direct answer or clarification to the assistant's last question.\n"
                        "- confused: true only when confusion_score is at least 85.\n"
                        "Do not classify normal answers to assistant questions as confused, even if they reuse words like trailer, livestock,utility, or need. "
                        "Do not classify natural narrowing or clarification as confused, e.g. user asks for a trailer, assistant asks type, user says utility trailer. "
                        "Result-navigation requests after listings, such as show more results, more options, next, or any others, are normal workflow. "
                        "Only consider result-navigation requests confused when they are repeated excessively or include real confusion language; code applies a higher repeat threshold for these. "
                        "Do not classify normal store or contact-info questions as confused, such as asking how to contact TrailerPlace, the phone number, location, or hours. "
                        "A user saying they are interested in a listing and then asking how to contact the store is a normal sales flow, not confusion. "
                        "similar_repeat_count must count only materially duplicate or repeated user requests, not generally related conversation turns. "
                        "Classify as confused only for high-confidence cases such as spamming the same simple request repeatedly, asking the same question again and again without progress, "
                        "or indirectly indicating inability to decide/understand (for example: I am confused, I don't know what to choose, this is not helping, I keep asking the same thing)."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Latest user message: {user_message}\n"
                        f"Recent eligible user messages (oldest to latest): {recent_user_turns}\n"
                        f"Recent full conversation messages (oldest to latest): {recent_messages}"
                    )
                ),
            ]
        )
        decision = _model_dump(result)
        similar_repeat_count = int(decision.get("similar_repeat_count") or 0)
        confusion_score = int(decision.get("confusion_score") or 0)
        confused = bool(decision.get("confused"))
        is_answer = bool(decision.get("is_answer_to_assistant_question"))
        final_confused = (
            confused
            and not is_answer
            and confusion_score >= _CONFUSION_SCORE_THRESHOLD
            and similar_repeat_count >= repeat_threshold
        )
        logger.info(
            "confusion_detection_result | confused=%s | final_confused=%s | confusion_score=%s | threshold=%s | similar_repeat_count=%s | repeat_threshold=%s | threshold_type=%s | is_answer_to_assistant_question=%s",
            confused,
            final_confused,
            confusion_score,
            _CONFUSION_SCORE_THRESHOLD,
            similar_repeat_count,
            repeat_threshold,
            threshold_type,
            is_answer,
        )
        return final_confused, similar_repeat_count
    except Exception:
        logger.exception("Confusion detector failed; skipping escalation check")
        return False, 0


def _handle_confusion_escalation(session: dict[str, Any], request: ChatRequest, similar_repeat_count: int) -> ChatResponse:
    if not _has_contact(session):
        session["pending_contact_action"] = {
            "type": "faq",
            "faq_category": "contact_human",
            "summary": (
                "Customer appears confused due to repeated similar requests; "
                "please reach out from the sales department."
            ),
            "user_message": request.message,
            "context_summary": _email_context_summary(session),
        }
        assistant_text = (
            "I can have our sales team help with this. Could you please share your name and either "
            "your phone number or email address so they can contact you?"
        )
    elif not session.get("confusion_escalated"):
        _persist_email_transcript_snapshot(session)
        context_summary = _email_context_summary(session)
        send_non_sales_faq_email(
            session_id=session.get("session_id") or "",
            full_name=session.get("customer_full_name") or "",
            email=session.get("customer_email"),
            phone=session.get("customer_phone") or "",
            faq_category="contact_human",
            summary=(
                "Customer appears confused due to repeated similar requests; "
                "please reach out from the sales department."
            ),
            user_message=request.message,
            context_summary=context_summary,
        )
        session["confusion_escalated"] = True
        assistant_text = compose_email_tool_reply(
            email_purpose="sales follow-up for repeated customer confusion",
            latest_message=request.message,
            conversation_context=context_summary,
            fallback=_CONFUSION_ESCALATION_REPLY,
        )
    else:
        assistant_text = _CONFUSION_ESCALATION_REPLY
    session["confusion_signal_count"] = int(similar_repeat_count)
    session["messages"].append({"role": "assistant", "content": assistant_text})
    _persist(session)
    _log_chat_turn(request.session_id, request.message, assistant_text)
    return ChatResponse(
        assistant_text=assistant_text,
        sales_phase="main",
        onboarding_api_messages=request.onboarding_api_messages,
        customer_full_name=session.get("customer_full_name"),
        customer_email=session.get("customer_email") or "",
        customer_phone=session.get("customer_phone"),
        contact_status=session.get("contact_status"),
        main_prior_messages=session.get("messages") or [],
        listings=[],
    )


def _update_active_question_tracker(session: dict[str, Any], result: dict[str, Any]) -> None:
    question = str(result.get("active_question_text") or "").strip()
    slot = str(result.get("active_question_slot") or result.get("awaiting_slot") or "").strip()
    if "active_question_attempts" in result:
        attempts = dict(result.get("active_question_attempts") or {})
        count = int(attempts.get(slot) or 0) if slot else 0
        session["active_question_attempts"] = attempts
        session["active_question_unanswered_count"] = count
        session["active_question_tracker"] = (
            {"slot": slot, "question": question, "unanswered_count": count}
            if slot
            else None
        )
        return
    attempts = dict(session.get("active_question_attempts") or {})
    if not slot:
        session["active_question_tracker"] = None
        session["active_question_unanswered_count"] = 0
        return
    previous_count = int(attempts.get(slot) or 0)
    unanswered = bool(result.get("active_question_was_unanswered"))
    count = previous_count + (1 if unanswered else 0)
    if result.get("active_question_was_resolved") or result.get("repeated_unanswered_question_escalation"):
        attempts.pop(slot, None)
        count = 0
    else:
        attempts[slot] = count
    session["active_question_attempts"] = attempts
    session["active_question_tracker"] = {
        "slot": slot,
        "question": question,
        "unanswered_count": count,
    }
    session["active_question_unanswered_count"] = count


def _strip_repeated_question(text: str, question: str) -> str:
    value = str(text or "").strip()
    if not question:
        return value
    return value.replace(f"\n\n{question}", "").replace(question, "").strip()


def _conversation_payload(session: dict[str, Any]) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    messages = session.get("messages") or []
    for idx in range(0, len(messages), 2):
        user_msg = messages[idx] if idx < len(messages) else None
        assistant_msg = messages[idx + 1] if idx + 1 < len(messages) else None
        turn = {
            "user": user_msg.get("content") if user_msg else None,
            "chatbot": assistant_msg.get("content") if assistant_msg else None,
        }
        if assistant_msg:
            for key in ("trailer_category", "metadata_filters_collected"):
                if assistant_msg.get(key) not in (None, "", [], {}):
                    turn[key] = assistant_msg[key]
            feedback = assistant_msg.get("feedback")
            if feedback in (None, ""):
                feedback = assistant_msg.get("user_feedback")
            if feedback not in (None, ""):
                turn["feedback"] = feedback
        turns.append(turn)
    return turns


def _log_chat_turn(session_id: str, user_message: str, assistant_text: str) -> None:
    """Log one exchange for support and debugging."""
    max_len = 8000
    reply = assistant_text if len(assistant_text) <= max_len else assistant_text[:max_len] + "...(truncated)"
    logger.info(
        "chat_turn | session_id=%s | user_question=%r | chatbot_response=%r",
        session_id,
        user_message,
        reply,
    )


def _persist(session: dict[str, Any]) -> None:
    enqueue_upsert_conversation(
        session_id=session["session_id"],
        lead_id=session.get("lead_id"),
        conversation=_conversation_payload(session),
    )


def _resolve_listing_reference(
    session: dict[str, Any],
    user_message: str,
) -> ListingReferenceDecision:
    listings = list(session.get("last_listings") or [])
    if not listings:
        return ListingReferenceDecision(reason="no_current_listing_set")
    context_listings = [
        {
            "index": index,
            "title": item.get("title"),
            "url": item.get("url"),
            "stock_number": item.get("stock_number"),
            "model": item.get("model"),
            "price": item.get("price_display") or item.get("price"),
        }
        for index, item in enumerate(listings, 1)
    ]
    try:
        gate = _listing_reference_intent_llm().invoke(
            [
                SystemMessage(content=(
                    "Decide whether LATEST_MESSAGE explicitly refers to one item in CURRENT_IDENTIFIERS. "
                    "Use only these inputs; do not infer a reference from prior conversation. Approve only: "
                    "(1) an ordinal/number such as first, #2, second, or last; "
                    "(2) a displayed title, model, or stock number; or "
                    "(3) a demonstrative such as that one or the gray one. "
                    "Generic quote, financing, reservation, delivery, category-shopping, and trailer-use "
                    "questions are not listing references, even when they mention a trailer category. "
                    "Set reference_intent=interest only when the latest message expresses interest in a "
                    "specific referenced item; use details for a question about one. Return structured output."
                )),
                HumanMessage(content=json.dumps({
                    "latest_message": user_message,
                    "current_identifiers": context_listings,
                }, default=str)),
            ]
        )
    except Exception:
        logger.exception("listing_reference_intent_gate_failed")
        return ListingReferenceDecision(reason="listing_reference_intent_gate_failed")

    logger.info(
        "listing_reference_intent_gate | explicit=%s | kind=%s | intent=%s | confidence=%s | reason=%r",
        gate.has_explicit_listing_reference,
        gate.reference_kind,
        gate.reference_intent,
        gate.confidence,
        gate.reason,
    )
    if (
        not gate.has_explicit_listing_reference
        or gate.reference_kind == "none"
        or gate.reference_intent == "none"
        or gate.confidence not in {"medium", "high"}
    ):
        return ListingReferenceDecision(
            reason=gate.reason or "no_explicit_current_turn_listing_reference"
        )

    try:
        proposed = _listing_reference_llm().invoke(
            [
                SystemMessage(content=(
                    "Resolve whether the latest message refers to a listing in CURRENT_LISTINGS. "
                    "Return structured output only. Use one-based displayed indexes. "
                    "Set has_explicit_listing_reference=true only when the latest message contains evidence that "
                    "points to a displayed item: an ordinal/number (#2, second, last), a shown title/model/stock "
                    "number, or an unmistakable demonstrative reference such as 'that one' or 'the gray one'. "
                    "General shopping, a new category request, or wording such as 'I am also looking for a car "
                    "hauler' does not refer to a shown item: set has_explicit_listing_reference=false and "
                    "is_listing_selection=false, even though it expresses shopping interest. "
                    "reference_intent=interest only for explicit liking, buying, selecting, or asking the team "
                    "to follow up about a listing. Use details for questions about a listing. "
                    "Copy selected_title and selected_url exactly from CURRENT_LISTINGS; never use older results "
                    "or invent a listing. If a phrase such as 'that one' is ambiguous, mark it as a listing "
                    "selection but leave the selected fields empty with low confidence."
                )),
                HumanMessage(content=json.dumps({
                    "latest_message": user_message,
                    "current_listings": context_listings,
                    "intent_gate": _model_dump(gate),
                }, default=str)),
            ]
        )
    except Exception:
        logger.exception("listing_reference_llm_failed")
        return ListingReferenceDecision(reason="listing_reference_llm_failed")

    if not proposed.has_explicit_listing_reference:
        rejected = proposed.model_copy(update={
            "is_listing_selection": False,
            "reference_intent": "none",
            "selected_index": None,
            "selected_title": None,
            "selected_url": None,
            "confidence": "low",
            "reason": proposed.reason or "no_explicit_current_listing_reference",
        })
        logger.info(
            "listing_reference_resolution | selected=false | explicit_reference=false | reason=%r",
            rejected.reason,
        )
        return rejected

    index = proposed.selected_index
    if (
        not proposed.is_listing_selection
        or proposed.reference_intent == "none"
        or proposed.confidence not in {"medium", "high"}
        or not index
        or index < 1
        or index > len(listings)
    ):
        logger.info(
            "listing_reference_resolution | selected=%s | intent=%s | index=%r | confidence=%s | reason=%r",
            proposed.is_listing_selection,
            proposed.reference_intent,
            index,
            proposed.confidence,
            proposed.reason,
        )
        return proposed

    selected = listings[index - 1]
    title = str(selected.get("title") or "")
    url = str(selected.get("url") or "")
    if str(proposed.selected_title or "") != title or str(proposed.selected_url or "") != url:
        logger.warning(
            "listing_reference_validation_failed | index=%s | proposed_title=%r | proposed_url=%r",
            index,
            proposed.selected_title,
            proposed.selected_url,
        )
        return ListingReferenceDecision(reason="selected_listing_identity_mismatch")
    resolved = proposed.model_copy(update={
        "selected_title": title,
        "selected_url": url,
    })
    logger.info(
        "listing_reference_resolution | selected=true | intent=%s | index=%s | title=%r | url=%r | confidence=%s",
        resolved.reference_intent,
        index,
        title,
        url,
        resolved.confidence,
    )
    return resolved


def _listing_reference_response(
    session: dict[str, Any],
    request: ChatRequest,
    decision: ListingReferenceDecision,
) -> ChatResponse | None:
    if not decision.is_listing_selection:
        return None
    tool_events: list[dict[str, Any]] = []
    confident = (
        decision.confidence in {"medium", "high"}
        and decision.selected_index is not None
        and bool(decision.selected_title)
    )
    if not confident:
        titles = [
            f"{index}. {item.get('title')}"
            for index, item in enumerate(session.get("last_listings") or [], 1)
            if item.get("title")
        ]
        assistant_text = "Which current listing do you mean? Please choose a number."
        if titles:
            assistant_text += "\n\n" + "\n".join(titles)
        logger.info("listing_reference_ambiguous | listing_count=%s", len(titles))
    elif decision.reference_intent == "details":
        return None
    elif decision.reference_intent == "interest":
        title = str(decision.selected_title)
        if not _has_contact(session):
            session["pending_contact_action"] = {
                "type": "interest",
                "item_name": title,
                "selected_listing_url": decision.selected_url,
            }
            assistant_text = (
                f"I can send your interest in **{title}** to our team. "
                "Please share your name and either an email address or phone number."
            )
        else:
            _persist_email_transcript_snapshot(session)
            result = send_interested_listing_email(
                session_id=session.get("session_id") or "",
                full_name=session.get("customer_full_name") or "",
                email=session.get("customer_email"),
                phone=session.get("customer_phone") or "",
                item_name=title,
            )
            tool_events.append({
                "tool": "send_interested_listing_email",
                "result": result,
            })
            logger.info(
                "listing_reference_field_extraction_bypassed | index=%s | title=%r | email_result=%s",
                decision.selected_index,
                title,
                json.dumps(result, default=str),
            )
            assistant_text = (
                f"Your interest in **{title}** has been logged. Our team will reach out soon."
            )
    else:
        return None

    session["messages"].append({"role": "assistant", "content": assistant_text})
    _persist(session)
    _log_chat_turn(request.session_id, request.message, assistant_text)
    return ChatResponse(
        assistant_text=assistant_text,
        sales_phase="main",
        onboarding_api_messages=request.onboarding_api_messages,
        customer_full_name=session.get("customer_full_name"),
        customer_email=session.get("customer_email") or "",
        customer_phone=session.get("customer_phone"),
        contact_status=session.get("contact_status"),
        main_prior_messages=session.get("messages") or [],
        listings=[],
        thinking_context={
            "listing_reference": _model_dump(decision),
            "tool_events": tool_events,
        },
    )


def _inventory_lookup_response(
    session: dict[str, Any],
    request: ChatRequest,
    user_message: str,
) -> ChatResponse | None:
    reference_decision = _resolve_listing_reference(session, user_message)
    reference_response = _listing_reference_response(session, request, reference_decision)
    if reference_response is not None:
        return reference_response
    has_shown_results = bool(session.get("has_shown_search_results"))
    direct_candidate = is_potential_direct_inventory_lookup(user_message)
    validated_extraction = (
        validated_direct_inventory_extraction(user_message)
        if direct_candidate
        else None
    )
    if direct_candidate and validated_extraction is None:
        return None
    if not validated_extraction:
        if not has_shown_results and _has_trailer_search_context(session):
            return None
        if session.get("awaiting_slot") or session.get("pending_questions"):
            return None
        if has_shown_results and _has_trailer_search_context(session) and _has_metadata_update_intent(user_message):
            return None
        if not should_attempt_chat_lookup(
            user_message,
            last_listings=session.get("last_listings") or [],
        ):
            return None
    try:
        result = search_trailers(
            user_message,
            last_listings=session.get("last_listings") or [],
            extraction=validated_extraction,
            for_chat=True,
        )
    except Exception:
        logger.exception("inventory_lookup_failed")
        return None

    assistant_text = str(result.get("reply") or "").strip()
    if not assistant_text:
        return None

    listings = result.get("top_matches") or []
    assistant_msg = {
        "role": "assistant",
        "content": assistant_text,
        "tool_events": [
            {
                "tool": "inventory_matcher",
                "entity_type": result.get("entity_type"),
                "confidence": result.get("confidence"),
                "result_count": len(listings),
            }
        ],
        "listings": listings,
    }
    session["messages"].append(assistant_msg)
    session["last_listings"] = listings
    shown = list(session.get("already_shown_listing_urls") or [])
    shown.extend([str(item.get("url")) for item in listings if item.get("url")])
    session["already_shown_listing_urls"] = sorted(set(shown))
    logger.info(
        "inventory_reference_state_replaced | listing_count=%s | accumulated_shown_url_count=%s",
        len(listings),
        len(session["already_shown_listing_urls"]),
    )
    if listings:
        session["has_shown_search_results"] = True
        _send_results_shown_notification(session)
    session["sales_phase"] = "main"
    _persist(session)
    _log_chat_turn(request.session_id, request.message, assistant_text)
    return ChatResponse(
        assistant_text=assistant_text,
        sales_phase="main",
        onboarding_api_messages=request.onboarding_api_messages,
        customer_full_name=session.get("customer_full_name"),
        customer_email=session.get("customer_email") or "",
        customer_phone=session.get("customer_phone"),
        contact_status=session.get("contact_status"),
        main_prior_messages=session.get("messages") or [],
        listings=[],
        thinking_context={
            "inventory_lookup": {
                "entity_type": result.get("entity_type"),
                "confidence": result.get("confidence"),
                "extraction": result.get("extraction") or {},
            }
        },
    )


def _listing_reference_llm():
    return make_llm(structured_output=ListingReferenceDecision)


def _listing_reference_intent_llm():
    return make_llm(structured_output=ListingReferenceIntentDecision)


def _legacy_messages(conversation: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for turn in conversation or []:
        if turn.get("user") is not None:
            messages.append({"role": "user", "content": turn.get("user")})
        if turn.get("chatbot") is not None:
            messages.append({"role": "assistant", "content": turn.get("chatbot")})
    return messages


def handle_chat(request: ChatRequest) -> ChatResponse:
    """Process and commit one idempotent, fully resumable turn."""
    if not persistence_enabled():
        return _handle_chat_in_memory(request)

    from fastapi import HTTPException

    try:
        with durable_turn(request.session_id, request.turn_id, request.message) as (
            db_session,
            row,
            receipt,
        ):
            if receipt:
                return ChatResponse.model_validate(receipt.response)
            if row and row.closed_at is not None:
                raise HTTPException(status_code=409, detail="Session is closed")

            if row and isinstance(row.state_snapshot, dict):
                loaded = deepcopy(row.state_snapshot)
                loaded["session_id"] = request.session_id
            else:
                loaded = _new_session(request.session_id)
                if row:
                    loaded["messages"] = _legacy_messages(row.conversation)
            with _lock:
                _sessions[request.session_id] = loaded

            with capture_email_events() as email_events:
                response = _handle_chat_in_memory(request)
            snapshot = deepcopy(_sessions[request.session_id])
            json.dumps(snapshot)  # Fail before success if state is not JSON-safe.
            payload = _model_dump(response)

            if row is None:
                lead_id = snapshot.get("lead_id")
                if not lead_id:
                    raise RuntimeError("A durable conversation requires a lead")
                row = _create_or_get_conversation(
                    db_session,
                    session_id=request.session_id,
                    lead_id=lead_id,
                    conversation=_conversation_payload(snapshot),
                )
                if row is None:
                    raise RuntimeError("Conversation could not be created or reloaded")
            row.conversation = _conversation_payload(snapshot)
            row.state_snapshot = snapshot
            row.state_schema_version = 1
            row.state_version = int(row.state_version or 0) + 1
            db_session.add(
                ChatbotTurn(
                    session_id=request.session_id,
                    turn_id=request.turn_id,
                    request_message=request.message,
                    response=payload,
                )
            )
            for index, event in enumerate(email_events):
                db_session.add(
                    ChatbotOutbox(
                        session_id=request.session_id,
                        turn_id=request.turn_id,
                        event_key=f"{event['event_type']}:{index}",
                        event_type=event["event_type"],
                        payload=event["payload"],
                    )
                )
            committed_response = response
        try:
            deliver_pending_outbox()
        except Exception:
            logger.exception("outbox_drain_failed")
        return committed_response
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("durable_chat_turn_failed | session_id=%s", request.session_id)
        raise HTTPException(
            status_code=503,
            detail="Conversation could not be saved; retry this turn",
        ) from exc


def _invoke_graph(
    session: dict[str, Any],
    user_message: str,
    already_shown: list[str],
    *,
    suppress_active_question_progress: bool = False,
) -> dict[str, Any]:
    graph_state = {
        # When True, the active qualification question is frozen for this turn:
        # the user is providing/declining the contact details we need to send a
        # deferred email, so their reply is not a failed answer and must not
        # advance the unanswered counter or trigger the skip-after-two escalation.
        "suppress_active_question_progress": suppress_active_question_progress,
        "session_id": session["session_id"],
        "user_message": user_message,
        "messages": session.get("messages") or [],
        "customer_full_name": session.get("customer_full_name"),
        "customer_email": session.get("customer_email"),
        "customer_phone": session.get("customer_phone"),
        "lead_id": session.get("lead_id"),
        "contact_status": session.get("contact_status"),
        "contact_request_allowed": _contact_request_allowed(session),
        "sales_phase": "main",
        "active_search_request_text": session.get("active_search_request_text") or "",
        "trailer_category": session.get("trailer_category"),
        "pending_category_change": deepcopy(session.get("pending_category_change")),
        "pending_category_suggestion": deepcopy(session.get("pending_category_suggestion")),
        "category_needs_clarification": bool(session.get("category_needs_clarification")),
        "category_clarification_key": session.get("category_clarification_key"),
        "slots_collected": deepcopy(session.get("slots_collected") or {}),
        "slots_skipped": list(session.get("slots_skipped") or []),
        "metadata_filters_collected": deepcopy(session.get("metadata_filters_collected") or {}),
        "defaulted_metadata_filters": list(session.get("defaulted_metadata_filters") or []),
        "requested_non_metadata_features": list(session.get("requested_non_metadata_features") or []),
        "make_category_options": list(session.get("make_category_options") or []),
        "awaiting_slot": session.get("awaiting_slot"),
        "pending_questions": deepcopy(session.get("pending_questions") or []),
        "asked_questions": list(session.get("asked_questions") or []),
        "already_shown_listing_urls": sorted(
            set((session.get("already_shown_listing_urls") or []) + (already_shown or []))
        ),
        "last_listings": deepcopy(session.get("last_listings") or []),
        "tool_events": [],
        "has_shown_search_results": bool(session.get("has_shown_search_results")),
        "active_category_cycle_id": int(session.get("active_category_cycle_id") or 1),
        "initial_contact_request_asked": bool(session.get("initial_contact_request_asked")),
        "pending_contact_action": deepcopy(session.get("pending_contact_action")),
        "pending_contact_actions": deepcopy(session.get("pending_contact_actions") or []),
        "active_question_text": session.get("active_question_text"),
        "active_question_tracker": deepcopy(session.get("active_question_tracker")),
        "active_question_unanswered_count": int(session.get("active_question_unanswered_count") or 0),
        "active_question_attempts": deepcopy(session.get("active_question_attempts") or {}),
        "pending_initial_user_message": session.get("pending_initial_user_message"),
    }
    return build_chatbot_graph().invoke(graph_state)


def _reset_search_state_for_category_switch(session: dict[str, Any], old_category: str, new_category: str) -> None:
    session["trailer_category"] = None
    session["pending_category_change"] = None
    session["category_needs_clarification"] = False
    session["category_clarification_key"] = None
    session["slots_collected"] = {}
    session["slots_skipped"] = []
    session["metadata_filters_collected"] = {}
    session["defaulted_metadata_filters"] = []
    session["requested_non_metadata_features"] = []
    session["active_search_request_text"] = ""
    session["make_category_options"] = []
    session["awaiting_slot"] = None
    session["pending_questions"] = []
    session["asked_questions"] = []
    session["already_shown_listing_urls"] = []
    session["last_listings"] = []
    session["tool_events"] = []
    session["active_question_tracker"] = None
    session["active_question_unanswered_count"] = 0
    session["active_question_attempts"] = {}
    session["has_shown_search_results"] = False
    session["active_category_cycle_id"] = int(session.get("active_category_cycle_id") or 1) + 1
    logger.info(
        "category_switch_reset | session_id=%s | old_category=%r | new_category=%r",
        session.get("session_id"),
        old_category,
        new_category,
    )
    logger.info(
        "category_cycle_reset | session_id=%s | active_category_cycle_id=%s | has_shown_search_results=%s",
        session.get("session_id"),
        session.get("active_category_cycle_id"),
        session.get("has_shown_search_results"),
    )


def _handle_chat_in_memory(request: ChatRequest) -> ChatResponse:
    session = _get_session(request.session_id)
    _ensure_lead(session)
    had_contact_before_turn = _has_contact(session)
    contact_before_turn = {
        "customer_full_name": session.get("customer_full_name"),
        "customer_full_name_confidence": session.get("customer_full_name_confidence"),
        "customer_email": session.get("customer_email"),
        "customer_phone": session.get("customer_phone"),
    }
    session["messages"].append({"role": "user", "content": request.message})
    was_awaiting_initial_contact = bool(session.get("awaiting_initial_contact_reply"))
    had_pending_contact_action = bool(
        session.get("pending_contact_action") or session.get("pending_contact_actions")
    )
    had_expiring_email = any(
        bool(action.get("expires_after_next_turn"))
        for action in (
            list(session.get("pending_contact_actions") or [])
            + ([session.get("pending_contact_action")] if session.get("pending_contact_action") else [])
        )
        if isinstance(action, dict)
    )
    contact_changed_this_turn = _apply_contact_from_request_and_message(session, request)
    contact_became_available = not had_contact_before_turn and _has_contact(session)
    session["sales_phase"] = "main"

    _send_pending_results_notification_if_ready(session)
    deferred_text, deferred_tool_events = _send_pending_contact_action_if_ready(session)
    resume_after_deferred_email = bool(
        deferred_text and (session.get("awaiting_slot") or session.get("pending_questions"))
    )
    deferred_email_prefix = deferred_text if (had_expiring_email or resume_after_deferred_email) else None
    if deferred_text and not had_expiring_email and not resume_after_deferred_email:
        assistant_text = deferred_text
        _update_active_question_tracker(
            session,
            {
                "active_question_text": session.get("active_question_text"),
                "awaiting_slot": session.get("awaiting_slot"),
                "assistant_text": assistant_text,
            },
        )
        session["messages"].append({"role": "assistant", "content": assistant_text})
        _persist(session)
        _log_chat_turn(request.session_id, request.message, assistant_text)
        return ChatResponse(
            assistant_text=assistant_text,
            sales_phase="main",
            onboarding_api_messages=request.onboarding_api_messages,
            customer_full_name=session.get("customer_full_name"),
            customer_email=session.get("customer_email") or "",
            customer_phone=session.get("customer_phone"),
            contact_status=session.get("contact_status"),
            main_prior_messages=session.get("messages") or [],
            listings=[],
            thinking_context={"tool_events": deferred_tool_events},
        )
    if had_expiring_email and not _has_contact(session):
        session["pending_contact_actions"] = [
            action
            for action in (session.get("pending_contact_actions") or [])
            if not action.get("expires_after_next_turn")
        ]
        if (session.get("pending_contact_action") or {}).get("expires_after_next_turn"):
            session["pending_contact_action"] = None

    inventory_response = _inventory_lookup_response(session, request, request.message)
    if inventory_response:
        return inventory_response

    if (
        not session.get("initial_contact_request_asked")
        and not _has_full_initial_details(session)
    ):
        session["initial_contact_request_asked"] = True
        session["awaiting_initial_contact_reply"] = True
        if _should_save_initial_message_for_resume(request.message):
            session["pending_initial_user_message"] = request.message
        assistant_text = _initial_contact_request_text(session)
        session["messages"].append({"role": "assistant", "content": assistant_text})
        _persist(session)
        _log_chat_turn(request.session_id, request.message, assistant_text)
        return ChatResponse(
            assistant_text=assistant_text,
            sales_phase="main",
            onboarding_api_messages=request.onboarding_api_messages,
            customer_full_name=session.get("customer_full_name"),
            customer_email=session.get("customer_email") or "",
            customer_phone=session.get("customer_phone"),
            contact_status=session.get("contact_status"),
            main_prior_messages=session.get("messages") or [],
            listings=[],
        )
    pending_initial = str(session.get("pending_initial_user_message") or "").strip()
    contact_reply_action: str | None = None
    contact_reply_latest_message = ""
    contact_reply_saved_request = ""
    if was_awaiting_initial_contact or contact_became_available:
        session["awaiting_initial_contact_reply"] = False
        contact_reply_decision = _classify_contact_prompt_reply(session, request.message)
        if contact_reply_decision.action == "decline_contact_details":
            session["contact_status"] = "contact_declined"
        if (
            contact_changed_this_turn
            and not _has_contact(session)
            and not session.get("initial_contact_followup_asked")
            and contact_reply_decision.action != "decline_contact_details"
        ):
            session["initial_contact_followup_asked"] = True
            session["awaiting_initial_contact_reply"] = True
            assistant_text = _initial_contact_followup_text(session)
            session["messages"].append({"role": "assistant", "content": assistant_text})
            _persist(session)
            logger.info(
                "partial_contact_followup | session_id=%s | has_name=%s | has_contact_method=%s",
                request.session_id,
                bool(session.get("customer_full_name")),
                bool(session.get("customer_email") or session.get("customer_phone")),
            )
            _log_chat_turn(request.session_id, request.message, assistant_text)
            return ChatResponse(
                assistant_text=assistant_text,
                sales_phase="main",
                onboarding_api_messages=request.onboarding_api_messages,
                customer_full_name=session.get("customer_full_name"),
                customer_email=session.get("customer_email") or "",
                customer_phone=session.get("customer_phone"),
                contact_status=session.get("contact_status"),
                main_prior_messages=session.get("messages") or [],
                listings=[],
            )
        if contact_reply_decision.action in {
            "route_latest_request", "resume_saved_request_with_update"
        } and not _has_contact(session):
            for key, value in contact_before_turn.items():
                session[key] = value
            _sync_contact_status(session)
            contact_changed_this_turn = False
            _persist_contact(session)
        if pending_initial and contact_reply_decision.action in {
            "acknowledge_contact_details", "decline_contact_details"
        }:
            contact_reply_decision.action = "resume_saved_request"
        session["pending_initial_user_message"] = None
        if not pending_initial and contact_reply_decision.action in {
            "acknowledge_contact_details", "decline_contact_details", "answer_contact_question"
        }:
            if contact_reply_decision.action == "acknowledge_contact_details":
                assistant_text = (
                    _CONTACT_ONLY_ACK
                    if _has_contact(session)
                    else _initial_contact_request_text(session)
                )
            elif contact_reply_decision.action == "answer_contact_question":
                assistant_text = _contact_explanation_text()
            else:
                assistant_text = "No problem—you do not have to share contact details. How can I help?"
            session["messages"].append({"role": "assistant", "content": assistant_text})
            _persist(session)
            _log_chat_turn(request.session_id, request.message, assistant_text)
            return ChatResponse(
                assistant_text=assistant_text,
                sales_phase="main",
                onboarding_api_messages=request.onboarding_api_messages,
                customer_full_name=session.get("customer_full_name"),
                customer_email=session.get("customer_email") or "",
                customer_phone=session.get("customer_phone"),
                contact_status=session.get("contact_status"),
                main_prior_messages=session.get("messages") or [],
                listings=[],
            )
        if pending_initial and contact_reply_decision.action in {
            "answer_contact_question", "resume_saved_request", "resume_saved_request_with_update"
        }:
            contact_reply_action = contact_reply_decision.action
            contact_reply_latest_message = request.message
            contact_reply_saved_request = pending_initial
            effective_message = (
                f"{pending_initial} | Additional requirement: {contact_reply_decision.remaining_message or request.message}"
                if contact_reply_decision.action == "resume_saved_request_with_update"
                else pending_initial
            )
        elif contact_reply_decision.action == "acknowledge_and_continue":
            contact_reply_action = contact_reply_decision.action
            contact_reply_latest_message = request.message
            effective_message = contact_reply_decision.remaining_message or request.message
        else:
            effective_message = request.message
    else:
        effective_message = request.message
    context_session = session
    if contact_reply_action:
        context_messages = (
            _replace_latest_user_message(session.get("messages") or [], request.message, effective_message)
            if contact_reply_action == "acknowledge_and_continue"
            else _without_latest_user_message(session.get("messages") or [], request.message)
        )
        context_session = {**session, "messages": context_messages}
    actionable_intent = _has_actionable_intent(effective_message)

    has_active_qualification_question = bool(session.get("awaiting_slot") or session.get("pending_questions"))
    if not has_active_qualification_question:
        confused_turn, similar_repeat_count = _is_confused_user_turn(session, effective_message)
        if confused_turn:
            return _handle_confusion_escalation(session, request, similar_repeat_count)

    should_route_graph = _should_route_to_graph(session, effective_message)
    logger.info(
        "main_phase_route_decision | session_id=%s | should_route_graph=%s | has_last_listings=%s | actionable_intent=%s",
        request.session_id,
        should_route_graph,
        bool(session.get("last_listings") or session.get("already_shown_listing_urls")),
        actionable_intent,
    )

    if not should_route_graph:
        assistant_text = _main_smalltalk_response(context_session, effective_message)
        assistant_text = _enforce_contact_response_policy(session, assistant_text)
        if contact_reply_action:
            bridge = (
                _CONTACT_CONTINUE_ACK
                if contact_reply_action == "acknowledge_and_continue"
                else _contact_prompt_bridge_text(
                    action=contact_reply_action,
                    latest_message=contact_reply_latest_message,
                    saved_request=contact_reply_saved_request,
                )
            )
            if bridge:
                assistant_text = f"{bridge}\n\n{assistant_text}"
        session["messages"].append({"role": "assistant", "content": assistant_text})
        _persist(session)
        _log_chat_turn(request.session_id, request.message, assistant_text)
        return ChatResponse(
            assistant_text=assistant_text,
            sales_phase="main",
            onboarding_api_messages=request.onboarding_api_messages,
            customer_full_name=session.get("customer_full_name"),
            customer_email=session.get("customer_email") or "",
            customer_phone=session.get("customer_phone"),
            contact_status=session.get("contact_status"),
            main_prior_messages=session.get("messages") or [],
            listings=[],
        )

    result = _invoke_graph(
        context_session,
        effective_message,
        request.already_shown_listing_urls,
        # A pending contact action at the start of this turn means we are
        # collecting the missing name/phone/email needed to send a deferred email.
        # Freeze the active qualification question's counter for such turns.
        suppress_active_question_progress=had_pending_contact_action,
    )
    logger.info(
        "graph_result | session_id=%s | action=%r | category=%r | tool_events=%s",
        request.session_id,
        (result.get("mind_decision") or {}).get("action"),
        result.get("trailer_category"),
        json.dumps(result.get("tool_events") or [], default=str),
    )
    for key in (
        "trailer_category",
        "pending_category_change",
        "pending_category_suggestion",
        "category_needs_clarification",
        "category_clarification_key",
        "slots_collected",
        "slots_skipped",
        "metadata_filters_collected",
        "defaulted_metadata_filters",
        "requested_non_metadata_features",
        "active_search_request_text",
        "make_category_options",
        "awaiting_slot",
        "pending_questions",
        "asked_questions",
        "already_shown_listing_urls",
        "last_listings",
        "pending_contact_action",
        "pending_contact_actions",
        "active_question_text",
        "active_question_attempts",
    ):
        if key in result:
            session[key] = result[key]
    pending_action = session.get("pending_contact_action")
    if pending_action:
        queued_actions = list(session.get("pending_contact_actions") or [])
        if pending_action not in queued_actions:
            queued_actions.append(pending_action)
        session["pending_contact_actions"] = queued_actions
    repeated_question_escalation = bool(result.get("repeated_unanswered_question_escalation"))
    _update_active_question_tracker(session, result)
    if repeated_question_escalation:
        session["active_question_tracker"] = None
        session["active_question_unanswered_count"] = 0
    tool_events = [
        *deferred_tool_events,
        *(result.get("tool_events") or []),
    ]
    if any(
        e.get("tool") == "pinecone_search" and int(e.get("result_count") or 0) > 0
        for e in tool_events
        if isinstance(e, dict)
    ) or bool(result.get("last_listings")):
        session["has_shown_search_results"] = True
        logger.info(
            "category_cycle_results_shown | session_id=%s | active_category_cycle_id=%s | has_shown_search_results=%s",
            request.session_id,
            session.get("active_category_cycle_id"),
            session.get("has_shown_search_results"),
        )
    assistant_text = (result.get("assistant_text") or "").strip() or " "
    assistant_text = _enforce_contact_response_policy(
        session, assistant_text, result.get("active_question_text")
    )
    if repeated_question_escalation:
        confusion_action = {
            "type": "faq",
            "faq_category": "contact_human",
            "summary": "Customer did not answer the same qualification question after two attempts; sales follow-up requested.",
            "user_message": request.message,
            "context_summary": _email_context_summary(session),
            "expires_after_next_turn": True,
        }
        if _has_contact(session):
            _persist_email_transcript_snapshot(session)
            send_non_sales_faq_email(
                session_id=session.get("session_id") or "",
                full_name=session.get("customer_full_name") or "",
                email=session.get("customer_email"),
                phone=session.get("customer_phone") or "",
                faq_category="contact_human",
                summary=confusion_action["summary"],
                user_message=request.message,
                context_summary=confusion_action["context_summary"],
            )
            next_question = str(result.get("active_question_text") or "")
            has_search_results = bool(result.get("last_listings"))
            base_reply = "" if has_search_results else _strip_repeated_question(assistant_text, next_question)
            email_reply = compose_email_tool_reply(
                email_purpose=confusion_action["summary"],
                latest_message=request.message,
                conversation_context=confusion_action["context_summary"],
                base_reply=base_reply,
                next_question="" if has_search_results else next_question,
                fallback="I've asked our sales team to follow up.",
            )
            assistant_text = (
                f"{email_reply}\n\n{assistant_text}".strip()
                if has_search_results
                else email_reply
            )
        else:
            queued_actions = list(session.get("pending_contact_actions") or [])
            queued_actions.append(confusion_action)
            session["pending_contact_actions"] = queued_actions
            contact_notice = (
                "If you'd like me to send this to our team, please share your name and either "
                "your phone number or email address."
            )
            assistant_text = f"{contact_notice}\n\n{assistant_text}".strip()
        logger.info(
            "repeated_unanswered_question_escalation | session_id=%s | skipped_slot=%r",
            request.session_id,
            result.get("skipped_unanswered_slot"),
        )
    if deferred_email_prefix:
        assistant_text = f"{deferred_email_prefix}\n\n{assistant_text}".strip()
    if contact_reply_action:
        bridge = (
            _CONTACT_CONTINUE_ACK
            if contact_reply_action == "acknowledge_and_continue"
            else _contact_prompt_bridge_text(
                action=contact_reply_action,
                latest_message=contact_reply_latest_message,
                saved_request=contact_reply_saved_request,
            )
        )
        if bridge:
            assistant_text = f"{bridge}\n\n{assistant_text}"
    assistant_msg = {
        "role": "assistant",
        "content": assistant_text,
        "qualification_question": result.get("active_question_text"),
        "tool_events": tool_events,
        "listings": result.get("last_listings") or [],
        "trailer_category": result.get("trailer_category"),
        "slots_collected": result.get("slots_collected") or {},
        "metadata_filters_collected": result.get("metadata_filters_collected") or {},
    }
    session["messages"].append(assistant_msg)
    if any(
        e.get("tool") == "pinecone_search" and int(e.get("result_count") or 0) > 0
        for e in tool_events
        if isinstance(e, dict)
    ):
        _send_results_shown_notification(session)
    session["sales_phase"] = "main"
    _persist(session)
    _log_chat_turn(request.session_id, request.message, assistant_text)

    return ChatResponse(
        assistant_text=assistant_text,
        sales_phase="main",
        onboarding_api_messages=request.onboarding_api_messages,
        customer_full_name=session.get("customer_full_name"),
        customer_email=session.get("customer_email") or "",
        customer_phone=session.get("customer_phone"),
        contact_status=session.get("contact_status"),
        main_prior_messages=session.get("messages") or [],
        listings=[],
        thinking_context={
            "category": session.get("trailer_category"),
            "slots": session.get("slots_collected") or {},
            "metadata_filters": session.get("metadata_filters_collected") or {},
            "tool_events": tool_events,
        },
    )
