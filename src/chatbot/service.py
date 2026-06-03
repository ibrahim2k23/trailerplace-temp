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
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from src.chatbot.categories import resolve_category_from_text
from src.chatbot.graph import build_chatbot_graph
from src.chatbot.inventory_matcher import search_trailers
from src.chatbot.inventory_matcher import should_attempt_chat_lookup
from src.chatbot.make_resolver import resolve_make_from_text
from src.chatbot.tools.email_tools import (
    send_interested_listing_email,
    send_non_sales_faq_email,
)
from src.conversation_store import (
    create_or_get_soft_lead,
    enqueue_upsert_conversation,
    update_lead_contact,
)
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
    r"gooseneck|bumper\s*[- ]?\s*pull|hitch|payload|capacity|"
    r"financ(?:e|ing)|trade(?:-|\s)?in|service|parts|human|contact|"
    r"store|hours|location|interested"
    r")\b",
    re.I,
)
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
_CONFUSION_REPEAT_THRESHOLD = 2
_RESULT_NAV_CONFUSION_REPEAT_THRESHOLD = 3
_CONFUSION_HISTORY_WINDOW = 6
_CONFUSION_SCORE_THRESHOLD = 85
_CONFUSION_SIMILARITY_THRESHOLD = 0.86
_CONTACT_ONLY_ACK = (
    "Thanks for sharing your contact details. I've saved them. "
    "Let me know if you need help with anything else."
)
_INITIAL_CONTACT_REQUEST = (
    "Thank you for contacting TrailerPlace. Before we get started, could I get your name, "
    "email, and phone number? Sharing contact details is optional, and I can still help with your trailer search."
)


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
        "resume_saved_request",
        "route_latest_request",
    ] = "resume_saved_request"
    reason: str = ""


def _model_dump(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


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
        "trailer_category": None,
        "slots_collected": {},
        "slots_skipped": [],
        "metadata_filters_collected": {},
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
        "pending_contact_action": None,
        "pending_initial_user_message": None,
    }


def _get_session(session_id: str) -> dict[str, Any]:
    with _lock:
        if session_id not in _sessions:
            _sessions[session_id] = _new_session(session_id)
        return _sessions[session_id]


def reset_session(session_id: str) -> None:
    with _lock:
        _sessions.pop(session_id, None)


def _contact_llm():
    return ChatOpenAI(model=_CHAT_MODEL, temperature=0).with_structured_output(
        ContactExtraction,
        method="function_calling",
    )


def _confusion_llm():
    return ChatOpenAI(model=_CHAT_MODEL, temperature=0).with_structured_output(
        ConfusionDetectionDecision,
        method="function_calling",
    )


def _contact_prompt_reply_llm():
    model = (os.getenv("OPENAI_MODEL") or _CHAT_MODEL).strip()
    return ChatOpenAI(model=model, temperature=0).with_structured_output(
        ContactPromptReplyDecision,
        method="function_calling",
    )


def _contact_prompt_bridge_llm():
    model = (os.getenv("OPENAI_MODEL") or _CHAT_MODEL).strip()
    return ChatOpenAI(model=model, temperature=0.3)


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
                    content=(
                        "Extract customer contact details from the message using natural language understanding. "
                        "Return null for missing fields. Do not guess. "
                        "Only extract a name when the user is clearly giving their personal name. "
                        "Set name_confidence to high only for explicit self-identification such as "
                        "'my name is Minahil', 'I am Minahil', 'Minahil here', or 'folks call me Tex'. "
                        "Set name_confidence to none for sentence fragments, objections, questions, "
                        "or trailer requests. "
                        "Do not treat phrases like 'I'm looking for', 'I'm interested in', "
                        "'I need', 'I want', 'I'm not comfortable', or 'why do you need them' as a name."
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
    session["contact_status"] = "contact_available" if _has_contact(session) else "missing_contact"


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
        fields = f"your {missing[0]}"
    return (
        "Thank you for contacting TrailerPlace. Before we get started, could I get "
        f"{fields}? Sharing contact details is optional, and I can still help with your trailer search."
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
    if not pending:
        return ContactPromptReplyDecision(action="route_latest_request", reason="No saved request.")
    try:
        result = _contact_prompt_reply_llm().invoke(
            [
                SystemMessage(
                    content=(
                        "The assistant previously asked for optional contact details after saving "
                        "the customer's original request. Decide the next action.\n"
                        "Return action=answer_contact_question when the latest reply asks why contact "
                        "details are needed, how they will be used, or whether they are required.\n"
                        "Return action=resume_saved_request when the latest reply is only providing "
                        "contact details, declining/skipping contact details, or is unclear/ambiguous "
                        "after the contact prompt.\n"
                        "Return action=route_latest_request only when the latest reply contains a "
                        "clear new actionable trailer, store, financing, service, parts, trade-in, "
                        "human-contact, or inventory request that should replace the saved request.\n"
                        "Do not rely on fixed wording; judge the user's intent in context."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Saved original request: {pending!r}\n"
                        f"Latest user reply after optional contact prompt: {latest_message!r}"
                    )
                ),
            ]
        )
        decision = _model_dump(result)
        action = str(decision.get("action") or "resume_saved_request")
        if action not in {"answer_contact_question", "resume_saved_request", "route_latest_request"}:
            action = "resume_saved_request"
        logger.info(
            "contact_prompt_reply_decision | action=%s | reason=%r",
            action,
            decision.get("reason") or "",
        )
        return ContactPromptReplyDecision(action=action, reason=str(decision.get("reason") or ""))
    except Exception:
        logger.exception("Contact prompt reply LLM failed; using regex fallback")
        if _is_contact_explanation_question(latest_message):
            return ContactPromptReplyDecision(action="answer_contact_question", reason="Regex fallback contact question.")
        if _has_actionable_intent(latest_message) and not _is_contact_refusal(latest_message):
            return ContactPromptReplyDecision(action="route_latest_request", reason="Regex fallback actionable request.")
        return ContactPromptReplyDecision(action="resume_saved_request", reason="Regex fallback resume saved request.")


def _should_resume_pending_after_contact_ask(session: dict[str, Any], latest_message: str) -> bool:
    return _classify_contact_prompt_reply(session, latest_message).action == "resume_saved_request"


def _contact_prompt_bridge_text(
    *,
    action: str,
    latest_message: str,
    saved_request: str,
    graph_response: str,
) -> str:
    try:
        response = _contact_prompt_bridge_llm().invoke(
            [
                SystemMessage(
                    content=(
                        "Write a short, natural opening for a TrailerPlace sales chat response. "
                        "The user just replied to an optional contact-details request. "
                        "Address only that contact-related reply. "
                        "Do not answer the trailer request yourself, do not mention specific inventory, "
                        "do not repeat the full saved request, and do not ask any trailer-search or "
                        "qualification question. "
                        "Never ask what type, kind, size, category, length, width, weight, hitch, color, "
                        "or budget the customer wants; the next response will handle trailer details. "
                        "If action is answer_contact_question, explain that contact details help the team "
                        "reach the customer later if the need arises and that sharing them is optional. "
                        "If action is resume_saved_request, acknowledge their preference or ambiguity without pressure "
                        "and say you can keep helping here. "
                        "Keep it to 1-2 concise sentences, no markdown list. End without a question mark."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Action: {action}\n"
                        f"User contact reply: {latest_message!r}\n"
                        f"Saved trailer request being resumed: {saved_request!r}\n"
                        f"Trailer/search response that will follow this opening: {graph_response!r}"
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


def _message_for_routing(session: dict[str, Any], latest_message: str) -> str:
    pending = str(session.get("pending_initial_user_message") or "").strip()
    if not pending:
        return latest_message
    if _should_resume_pending_after_contact_ask(session, latest_message):
        session["pending_initial_user_message"] = None
        return pending
    session["pending_initial_user_message"] = None
    return latest_message


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


def _send_pending_contact_action_if_ready(session: dict[str, Any]) -> str | None:
    action = session.get("pending_contact_action")
    if not action or not _has_contact(session):
        return None
    action_type = action.get("type")
    if action_type == "interest":
        item_name = str(action.get("item_name") or "that trailer").strip()
        result = send_interested_listing_email(
            session_id=session.get("session_id") or "",
            full_name=session.get("customer_full_name") or "",
            email=session.get("customer_email"),
            phone=session.get("customer_phone") or "",
            item_name=item_name,
        )
        logger.info("deferred_contact_action_sent | type=interest | result=%s", json.dumps(result, default=str))
        session["pending_contact_action"] = None
        return (
            f"Thanks, I saved your contact information and sent your interest in \"{item_name}\" "
            "to our team so they can follow up."
        )
    if action_type == "faq":
        category = str(action.get("faq_category") or "contact_human")
        summary = str(action.get("summary") or "")
        result = send_non_sales_faq_email(
            session_id=session.get("session_id") or "",
            full_name=session.get("customer_full_name") or "",
            email=session.get("customer_email"),
            phone=session.get("customer_phone") or "",
            faq_category=category,
            summary=summary,
            user_message=str(action.get("user_message") or ""),
        )
        logger.info("deferred_contact_action_sent | type=faq | result=%s", json.dumps(result, default=str))
        session["pending_contact_action"] = None
        return "Thanks, I saved your contact information and sent that request to our team so they can help."
    session["pending_contact_action"] = None
    return None


def _main_smalltalk_response(session: dict[str, Any], user_message: str) -> str:
    try:
        response = ChatOpenAI(model=_CHAT_MODEL, temperature=0.4).invoke(
            [
                SystemMessage(
                    content=(
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
                        "Your purpose is to inform the user. not greet them. So do not greet the user or ask how they are doing. Just reply to their message."
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


def _should_route_to_graph(session: dict[str, Any], user_message: str) -> bool:
    """
    LLM-first routing for the main phase.
    This prevents regex misses (for example "I like the 4th one") from falling into smalltalk.
    """
    if _is_contact_only_message(user_message):
        return False

    if session.get("awaiting_slot") or session.get("pending_questions"):
        return True

    # Keep deterministic fast-paths for obvious intent.
    if _has_actionable_intent(user_message):
        return True
    if _has_trailer_search_context(session) and _has_metadata_update_intent(user_message):
        return True

    # If we have shown listings before, let the model decide whether this turn is
    # a listings follow-up (interest, comparison, ordinal reference, show more, etc.).
    if not (session.get("last_listings") or session.get("already_shown_listing_urls")):
        return False

    try:
        response = ChatOpenAI(model=_CHAT_MODEL, temperature=0).invoke(
            [
                SystemMessage(
                    content=(
                        "Classify whether the latest user message should go to the trailer-planner graph. "
                        "Respond with exactly one token: ROUTE or SMALLTALK.\n"
                        "Choose ROUTE for any message that could refer to shown listings, trailer selection, "
                        "ordinal references, follow-up constraints, asking for more options, or purchase intent."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Latest user message: {user_message}\n"
                        f"Current category: {session.get('trailer_category')}\n"
                        f"Has shown listings before: {bool(session.get('last_listings') or session.get('already_shown_listing_urls'))}\n"
                        f"Recent messages: {(session.get('messages') or [])[-6:]}"
                    )
                ),
            ]
        )
        label = str(response.content or "").strip().upper()
        return label.startswith("ROUTE")
    except Exception:
        logger.exception("Route classifier failed; falling back to deterministic intent check")
        return _has_actionable_intent(user_message)


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
        }
        assistant_text = (
            "I can have our sales team help with this. Could you please share your phone number "
            "or email address so they can contact you?"
        )
    elif not session.get("confusion_escalated"):
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
        )
        session["confusion_escalated"] = True
        assistant_text = _CONFUSION_ESCALATION_REPLY
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


def _inventory_lookup_response(
    session: dict[str, Any],
    request: ChatRequest,
    user_message: str,
) -> ChatResponse | None:
    if not should_attempt_chat_lookup(
        user_message,
        last_listings=session.get("last_listings") or [],
    ):
        return None
    try:
        result = search_trailers(
            user_message,
            last_listings=session.get("last_listings") or [],
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
    if listings:
        session["has_shown_search_results"] = True
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


def _invoke_graph(session: dict[str, Any], user_message: str, already_shown: list[str]) -> dict[str, Any]:
    graph_state = {
        "session_id": session["session_id"],
        "user_message": user_message,
        "messages": session.get("messages") or [],
        "customer_full_name": session.get("customer_full_name"),
        "customer_email": session.get("customer_email"),
        "customer_phone": session.get("customer_phone"),
        "lead_id": session.get("lead_id"),
        "contact_status": session.get("contact_status"),
        "sales_phase": "main",
        "trailer_category": session.get("trailer_category"),
        "slots_collected": deepcopy(session.get("slots_collected") or {}),
        "slots_skipped": list(session.get("slots_skipped") or []),
        "metadata_filters_collected": deepcopy(session.get("metadata_filters_collected") or {}),
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
        "pending_initial_user_message": session.get("pending_initial_user_message"),
    }
    return build_chatbot_graph().invoke(graph_state)


def _reset_search_state_for_category_switch(session: dict[str, Any], old_category: str, new_category: str) -> None:
    session["trailer_category"] = None
    session["slots_collected"] = {}
    session["slots_skipped"] = []
    session["metadata_filters_collected"] = {}
    session["make_category_options"] = []
    session["awaiting_slot"] = None
    session["pending_questions"] = []
    session["asked_questions"] = []
    session["already_shown_listing_urls"] = []
    session["last_listings"] = []
    session["tool_events"] = []
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


def handle_chat(request: ChatRequest) -> ChatResponse:
    session = _get_session(request.session_id)
    _ensure_lead(session)
    session["messages"].append({"role": "user", "content": request.message})
    had_pending_contact_action = bool(session.get("pending_contact_action"))
    _apply_contact_from_request_and_message(session, request)
    session["sales_phase"] = "main"

    deferred_text = _send_pending_contact_action_if_ready(session)
    if deferred_text:
        assistant_text = deferred_text
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

    inventory_response = _inventory_lookup_response(session, request, request.message)
    if inventory_response:
        return inventory_response

    if (
        not session.get("initial_contact_request_asked")
        and not _has_full_initial_details(session)
    ):
        session["initial_contact_request_asked"] = True
        if _has_actionable_intent(request.message):
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
    if had_pending_contact_action and not _has_contact(session):
        session["pending_contact_action"] = None

    pending_initial = str(session.get("pending_initial_user_message") or "").strip()
    contact_reply_action: str | None = None
    contact_reply_latest_message = ""
    contact_reply_saved_request = ""
    if pending_initial:
        contact_reply_decision = _classify_contact_prompt_reply(session, request.message)
        session["pending_initial_user_message"] = None
        if contact_reply_decision.action in {"answer_contact_question", "resume_saved_request"}:
            contact_reply_action = contact_reply_decision.action
            contact_reply_latest_message = request.message
            contact_reply_saved_request = pending_initial
            effective_message = pending_initial
        else:
            effective_message = request.message
    else:
        effective_message = request.message
    context_session = session
    if contact_reply_action:
        context_session = {
            **session,
            "messages": _without_latest_user_message(
                session.get("messages") or [],
                request.message,
            ),
        }
    if (
        not contact_reply_action
        and not had_pending_contact_action
        and _has_contact(session)
        and _is_contact_only_message(request.message)
    ):
        assistant_text = _CONTACT_ONLY_ACK
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
        if contact_reply_action:
            bridge = _contact_prompt_bridge_text(
                action=contact_reply_action,
                latest_message=contact_reply_latest_message,
                saved_request=contact_reply_saved_request,
                graph_response=assistant_text,
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

    # Deterministic reset for explicit category switch, then let planner drive action.
    resolution = resolve_category_from_text(effective_message)
    new_category = resolution.category
    current_category = session.get("trailer_category")
    if new_category and current_category and new_category != current_category:
        gate_open = bool(session.get("has_shown_search_results"))
        logger.info(
            "category_switch_candidate | session_id=%s | old_category=%r | new_category=%r | has_shown_search_results=%s | active_category_cycle_id=%s",
            request.session_id,
            current_category,
            new_category,
            gate_open,
            session.get("active_category_cycle_id"),
        )
        if gate_open:
            logger.info(
                "category_switch_applied | session_id=%s | old_category=%r | new_category=%r | gate_before=true | active_category_cycle_id=%s",
                request.session_id,
                current_category,
                new_category,
                session.get("active_category_cycle_id"),
            )
            _reset_search_state_for_category_switch(session, current_category, new_category)
        else:
            logger.info(
                "category_switch_ignored_pre_results | session_id=%s | old_category=%r | new_category=%r | active_category_cycle_id=%s",
                request.session_id,
                current_category,
                new_category,
                session.get("active_category_cycle_id"),
            )

    result = _invoke_graph(
        context_session,
        effective_message,
        request.already_shown_listing_urls,
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
        "slots_collected",
        "slots_skipped",
        "metadata_filters_collected",
        "make_category_options",
        "awaiting_slot",
        "pending_questions",
        "asked_questions",
        "already_shown_listing_urls",
        "last_listings",
        "pending_contact_action",
    ):
        if key in result:
            session[key] = result[key]
    tool_events = result.get("tool_events") or []
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
    if contact_reply_action:
        bridge = _contact_prompt_bridge_text(
            action=contact_reply_action,
            latest_message=contact_reply_latest_message,
            saved_request=contact_reply_saved_request,
            graph_response=assistant_text,
        )
        if bridge:
            assistant_text = f"{bridge}\n\n{assistant_text}"
    assistant_msg = {
        "role": "assistant",
        "content": assistant_text,
        "tool_events": tool_events,
        "listings": result.get("last_listings") or [],
        "trailer_category": result.get("trailer_category"),
        "slots_collected": result.get("slots_collected") or {},
        "metadata_filters_collected": result.get("metadata_filters_collected") or {},
    }
    session["messages"].append(assistant_msg)
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
            "tool_events": result.get("tool_events") or [],
        },
    )
