from __future__ import annotations

import json
import logging
import re
from copy import deepcopy
from threading import RLock
from typing import Any, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from src.chatbot.categories import resolve_category_from_text
from src.chatbot.graph import build_chatbot_graph
from src.chatbot.tools.email_tools import send_non_sales_faq_email
from src.conversation_store import (
    create_or_get_soft_lead,
    enqueue_upsert_conversation,
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
    r"financ(?:e|ing)|trade(?:-|\s)?in|service|parts|human|contact|"
    r"store|hours|location|interested"
    r")\b",
    re.I,
)
_CONFUSION_ESCALATION_REPLY = (
    "I've forwarded your request to our sales department, and they will reach out to you soon."
)
_CONFUSION_REPEAT_THRESHOLD = 2
_RESULT_NAV_CONFUSION_REPEAT_THRESHOLD = 3
_CONFUSION_HISTORY_WINDOW = 6
_CONFUSION_SCORE_THRESHOLD = 85


class ContactExtraction(BaseModel):
    full_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None


class ConfusionDetectionDecision(BaseModel):
    similar_repeat_count: int = 0
    confused: bool = False
    confusion_score: int = 0
    is_answer_to_assistant_question: bool = False


def _model_dump(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _new_session(session_id: str) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "sales_phase": "onboarding",
        "customer_full_name": None,
        "customer_email": None,
        "customer_phone": None,
        "lead_id": None,
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
        name = re.split(r"\s+(?:my\s+)?(?:email|phone|number)\b", name_match.group(1), flags=re.I)[0]
        name = re.sub(r"[,.;]+$", "", name).strip()
    return {
        "full_name": name,
        "email": email_match.group(0).strip() if email_match else None,
        "phone": phone_match.group(0).strip() if phone_match else None,
    }


def _extract_contact(message: str, current: dict[str, Any]) -> dict[str, Optional[str]]:
    try:
        result = _contact_llm().invoke(
            [
                SystemMessage(
                    content=(
                        "Extract customer contact details from the message. "
                        "Return null for missing fields. Do not guess."
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
        data = {}
    fallback = _regex_contact(message)
    return {
        "full_name": data.get("full_name") or fallback.get("full_name"),
        "email": data.get("email") or fallback.get("email"),
        "phone": data.get("phone") or fallback.get("phone"),
    }


def _contact_fallback_response(session: dict[str, Any]) -> str:
    has_name = bool(session.get("customer_full_name"))
    has_phone = bool(session.get("customer_phone"))
    if has_name and has_phone:
        return "Thank you for providing your contact information. How can I help you today?"
    if not has_name and not has_phone:
        return (
            "Hello! Thank you for contacting Trailer Place. Before we continue, "
            "can you please share your name, email, and phone number?"
        )
    if not has_name:
        return (
            "Thanks, I have your phone number. Before we continue, can you please share your name?"
        )
    return "Thanks, I have your name. Before we continue, can you please share your phone number?"


def _violates_contact_gate(text: str, *, contact_complete: bool) -> bool:
    if not contact_complete:
        return False
    low = (text or "").lower()
    email_request_markers = (
        "provide your email",
        "share your email",
        "email address",
        "complete your contact information",
        "complete your contact details",
    )
    return any(marker in low for marker in email_request_markers)


def _invalid_contact_request(text: str, *, has_name: bool, has_phone: bool) -> bool:
    low = (text or "").lower()
    if not has_name and not has_phone:
        return not all(word in low for word in ("name", "email", "phone"))
    if has_name and not has_phone:
        return "phone" not in low and "number" not in low
    if has_phone and not has_name:
        return "name" not in low
    return False


def _onboarding_llm_response(session: dict[str, Any], user_message: str) -> str:
    """Let the model write humane onboarding copy while code enforces the gate."""
    has_name = bool(session.get("customer_full_name"))
    has_email = bool(session.get("customer_email"))
    has_phone = bool(session.get("customer_phone"))
    contact_complete = has_name and has_phone
    recent_messages = (session.get("messages") or [])[-6:]
    try:
        response = ChatOpenAI(model=_CHAT_MODEL, temperature=0.2).invoke(
            [
                SystemMessage(
                    content=(
                        "You are the Trailer Place sales chat assistant. Write one short, natural "
                        "customer-facing reply for the onboarding/contact step.\n"
                        "Rules:\n"
                        "- Be humane and respond to the user's latest message, including questions like 'why?'.\n"
                        "- Before trailer help, collect contact details and keep the conversation realistic.\n"
                        "- On the first contact request, ask for name, email, and phone number together.\n"
                        "- Internally, contact is complete when name and phone are known. Email is not required, but do not say that.\n"
                        "- If name and phone are known, thank the user and ask how you can help today.\n"
                        "- If name and phone are known, it is forbidden to ask for email, email address, or completion of contact information.\n"
                        "- If only one of name or phone is missing, ask only for the missing required field, not email.\n"
                        "- Do not answer trailer inventory questions or choose trailer categories in this onboarding reply.\n"
                        "- Keep it to 1-3 sentences."
                    )
                ),
                HumanMessage(
                    content=(
                        "Known contact state:\n"
                        f"- name_known: {has_name}\n"
                        f"- email_known: {has_email}\n"
                        f"- phone_known: {has_phone}\n"
                        f"- contact_complete_name_and_phone: {contact_complete}\n"
                        f"- known_name: {session.get('customer_full_name') or ''}\n"
                        "Recent messages:\n"
                        f"{recent_messages}\n"
                        f"Latest user message: {user_message}"
                    )
                ),
            ]
        )
        text = str(response.content or "").strip()
        if _violates_contact_gate(text, contact_complete=contact_complete):
            return _contact_fallback_response(session)
        if _invalid_contact_request(text, has_name=has_name, has_phone=has_phone):
            return _contact_fallback_response(session)
        return text or _contact_fallback_response(session)
    except Exception:
        logger.exception("Onboarding response LLM failed; using fallback response")
        return _contact_fallback_response(session)


def _main_smalltalk_response(session: dict[str, Any], user_message: str) -> str:
    try:
        response = ChatOpenAI(model=_CHAT_MODEL, temperature=0.4).invoke(
            [
                SystemMessage(
                    content=(
                        "You are the Trailer Place sales chat assistant. The customer's "
                        "contact information is already captured. Reply naturally to the "
                        "latest message, then invite them to share what trailer or service "
                        "help they need. Keep it brief."
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


def _should_route_to_graph(session: dict[str, Any], user_message: str) -> bool:
    """
    LLM-first routing for the main phase.
    This prevents regex misses (for example "I like the 4th one") from falling into smalltalk.
    """
    if session.get("awaiting_slot") or session.get("pending_questions"):
        return True

    # Keep deterministic fast-paths for obvious intent.
    if _has_actionable_intent(user_message):
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
    contact = _regex_contact(message or "")
    has_contact_detail = any(contact.get(key) for key in ("full_name", "email", "phone"))
    return has_contact_detail and not _has_actionable_intent(message)


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
    if _has_listing_context(session) and _is_result_navigation_request(message):
        return _RESULT_NAV_CONFUSION_REPEAT_THRESHOLD, "result_navigation"
    return _CONFUSION_REPEAT_THRESHOLD, "default"


def _is_confusion_eligible_user_message(message: str, session: dict[str, Any] | None = None) -> bool:
    if _is_contact_only_message(message):
        return False
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
    if not _is_confusion_eligible_user_message(user_message, session):
        return False, 0
    recent_user_turns = _recent_confusion_user_messages(session)
    repeat_threshold, threshold_type = _confusion_repeat_threshold_for_message(session, user_message)
    if len(recent_user_turns) < _CONFUSION_REPEAT_THRESHOLD:
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
    if not session.get("confusion_escalated"):
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
    session["confusion_signal_count"] = int(similar_repeat_count)
    assistant_text = _CONFUSION_ESCALATION_REPLY
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


def _invoke_graph(session: dict[str, Any], user_message: str, already_shown: list[str]) -> dict[str, Any]:
    graph_state = {
        "session_id": session["session_id"],
        "user_message": user_message,
        "messages": session.get("messages") or [],
        "customer_full_name": session.get("customer_full_name"),
        "customer_email": session.get("customer_email"),
        "customer_phone": session.get("customer_phone"),
        "lead_id": session.get("lead_id"),
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
    session["messages"].append({"role": "user", "content": request.message})
    contact_was_complete = bool(session.get("customer_full_name") and session.get("customer_phone"))
    actionable_intent = _has_actionable_intent(request.message)
    extracted_any_contact = False

    if request.customer_full_name:
        session["customer_full_name"] = request.customer_full_name
    if request.customer_email is not None:
        session["customer_email"] = request.customer_email or None
    if request.customer_phone:
        session["customer_phone"] = request.customer_phone

    if not (session.get("customer_full_name") and session.get("customer_phone")):
        contact = _extract_contact(request.message, session)
        for source, target in (
            ("full_name", "customer_full_name"),
            ("email", "customer_email"),
            ("phone", "customer_phone"),
        ):
            if contact.get(source):
                session[target] = contact[source]
                extracted_any_contact = True

    if session.get("customer_full_name") and session.get("customer_phone") and not session.get("lead_id"):
        try:
            session["lead_id"] = create_or_get_soft_lead(
                session_id=request.session_id,
                full_name=session["customer_full_name"],
                email=session.get("customer_email"),
                phone=session["customer_phone"],
                item_of_interest="Trailer inquiry",
            )
        except Exception:
            logger.exception("Soft lead creation failed")
        session["sales_phase"] = "main"

    if not (session.get("customer_full_name") and session.get("customer_phone")):
        assistant_text = _onboarding_llm_response(session, request.message)
        session["messages"].append({"role": "assistant", "content": assistant_text})
        _persist(session)
        _log_chat_turn(request.session_id, request.message, assistant_text)
        return ChatResponse(
            assistant_text=assistant_text,
            sales_phase="onboarding",
            onboarding_api_messages=request.onboarding_api_messages
            + [{"role": "user", "content": request.message}, {"role": "assistant", "content": assistant_text}],
            customer_full_name=session.get("customer_full_name"),
            customer_email=session.get("customer_email"),
            customer_phone=session.get("customer_phone"),
            main_prior_messages=[],
            listings=[],
        )

    if not contact_was_complete and not actionable_intent:
        assistant_text = _onboarding_llm_response(session, request.message)
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
            main_prior_messages=session.get("messages") or [],
            listings=[],
        )

    has_active_qualification_question = bool(session.get("awaiting_slot") or session.get("pending_questions"))
    if not has_active_qualification_question:
        confused_turn, similar_repeat_count = _is_confused_user_turn(session, request.message)
        if confused_turn:
            return _handle_confusion_escalation(session, request, similar_repeat_count)

    should_route_graph = _should_route_to_graph(session, request.message)
    logger.info(
        "main_phase_route_decision | session_id=%s | should_route_graph=%s | has_last_listings=%s | actionable_intent=%s",
        request.session_id,
        should_route_graph,
        bool(session.get("last_listings") or session.get("already_shown_listing_urls")),
        actionable_intent,
    )

    if contact_was_complete and not should_route_graph:
        assistant_text = _main_smalltalk_response(session, request.message)
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
            main_prior_messages=session.get("messages") or [],
            listings=[],
        )

    # Deterministic reset for explicit category switch, then let planner drive action.
    resolution = resolve_category_from_text(request.message)
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
        session,
        request.message,
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
        main_prior_messages=session.get("messages") or [],
        listings=[],
        thinking_context={
            "category": session.get("trailer_category"),
            "slots": session.get("slots_collected") or {},
            "metadata_filters": session.get("metadata_filters_collected") or {},
            "tool_events": result.get("tool_events") or [],
        },
    )
