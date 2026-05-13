from __future__ import annotations

import json
import logging
import os
import re
from functools import lru_cache
from typing import Any, Literal, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from trailer_fields import get_trailer_fields_as_dict
from src.chatbot.categories import resolve_category_from_text
from src.chatbot.formatting import format_listing_results
from src.chatbot.mini_llm_classifier import (
    HaulClassificationDecision,
    classify_haul_requirements,
)
from src.chatbot.prompts import MIND_SYSTEM_PROMPT
from src.chatbot.state import ChatbotState, QuestionItem
from src.models import TrailerListing
from src.normalizer import normalize_hitch, normalize_subcategory
from src.chatbot.tools.email_tools import (
    FAQ_CATEGORY_LABELS,
    send_interested_listing_email,
    send_non_sales_faq_email,
)
from src.chatbot.tools.pinecone_search import search_pinecone_listings

load_dotenv()

logger = logging.getLogger(__name__)
_SITE_URL = (os.getenv("TRAILERPLACE_SITE_URL") or "https://trailerplace.com").strip()
_FAQ_REPLY_FALLBACKS: dict[str, str] = {
    "contact_human": (
        "You can reach our team at 979-532-1486. Happy to keep helping with your trailer search too!"
    ),
    "financing": (
        "We offer financing. Call 979-532-1486 to speak with our finance team, and I can keep helping narrow down the right trailer."
    ),
    "trade_in": "Our sales team handles trade-in appraisals. Call 979-532-1486.",
    "service_parts": "Our service and parts team can help. Reach them at 979-532-1486.",
    "store_info": (
        f"We're located in Wharton, TX. Call 979-532-1486 or visit {_SITE_URL}. "
        "We also offer financing and delivery."
    ),
}
_FAQ_GENERIC_FALLBACK = "Thanks, I sent that request to the team so they can help you with it."
_INTEREST_GENERIC_FALLBACK = (
    'Your interest in "{item_name}" has been logged. Our team will reach out to you soon. '
    f"In the meantime, feel free to visit {_SITE_URL} or call us at 979-532-1486."
)
_INTEREST_GENERIC_FALLBACK_NO_ITEM = (
    f"Your interest has been logged. Our team will reach out to you soon. In the meantime, feel free to visit {_SITE_URL} "
    "or call us at 979-532-1486."
)
_INTEREST_SAFE_FALLBACK = (
    "Great, I sent your interest in that trailer to the team. They can follow up with you shortly."
)


class MindDecision(BaseModel):
    action: Literal[
        "ask_next_question",
        "pinecone_search",
        "send_interested_listing_email",
        "send_non_sales_faq_email",
        "respond",
    ] = "respond"
    assistant_text: str = ""
    trailer_category: Optional[str] = None
    slots_collected_update: dict[str, Any] = Field(default_factory=dict)
    metadata_filters_update: dict[str, Any] = Field(default_factory=dict)
    optional_question_slots_to_queue: list[str] = Field(default_factory=list)
    selected_listing_title: Optional[str] = None
    selected_listing_url: Optional[str] = None
    faq_category: Optional[str] = None
    faq_summary: Optional[str] = None


class FilterExtractionDecision(BaseModel):
    length_ft: Optional[str] = None
    width_ft: Optional[str] = None
    payload_lbs: Optional[str] = None
    max_price: Optional[str] = None
    hitch_type: Optional[str] = None
    subcategory: Optional[str] = None
    color: Optional[str] = None
    slot_updates: dict[str, Any] = Field(default_factory=dict)


@lru_cache(maxsize=1)
def _mind_llm():
    model = (os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip()
    return ChatOpenAI(model=model, temperature=0).with_structured_output(
        MindDecision,
        method="function_calling",
    )


@lru_cache(maxsize=1)
def _filter_extractor_llm():
    model = (
        os.getenv("FILTER_EXTRACTOR_MODEL")
        or os.getenv("OPENAI_MODEL")
        or "gpt-4o-mini"
    ).strip()
    return ChatOpenAI(model=model, temperature=0).with_structured_output(
        FilterExtractionDecision,
        method="function_calling",
    )


def _model_dump(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _safe_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=True, default=str)


_METADATA_FILTER_KEYS = {
    "length_ft",
    "width_ft",
    "payload_lbs",
    "max_price",
    "hitch_type",
    "subcategory",
    "color",
}
_ALLOWED_HITCH_TYPES = {"Gooseneck", "Bumper Pull"}
_CONFIDENT_CLASSIFICATIONS = {"medium", "high"}
_DYNAMIC_WIDTH_SLOT = "item_or_trailer_width_ft"
_DYNAMIC_WIDTH_QUESTION = "About how wide is the item, or what trailer width do you need?"


def _category_slots(category: str | None) -> set[str]:
    if not category:
        return set()
    spec = get_trailer_fields_as_dict(category)
    return set(spec.get("required_slots") or []) | set(spec.get("optional_slots") or [])


def _first_match(patterns: tuple[str, ...], text: str) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            return " ".join(g for g in match.groups() if g) or match.group(0)
    return None


def _normalize_allowed_hitch(value: Any) -> str | None:
    hitch = normalize_hitch(str(value or "").strip())
    return hitch if hitch in _ALLOWED_HITCH_TYPES else None


def _explicit_subcategory_requested(message: str) -> bool:
    text = str(message or "").lower()
    return bool(
        re.search(
            r"\b(?:sub\s*category|subcategory)\s+(?:should\s+be|is|to|as|=)\b"
            r"|\b(?:filter|search)\s+(?:by|for|with)\s+(?:sub\s*category|subcategory)\b"
            r"|\bmake\s+the\s+(?:sub\s*category|subcategory)\b",
            text,
        )
    )


def _sanitize_metadata_filter_update(key: str, value: Any, latest_message: str) -> tuple[str, Any] | None:
    if value in (None, "") or key not in _METADATA_FILTER_KEYS:
        return None
    if key == "hitch_type":
        hitch = _normalize_allowed_hitch(value)
        if not hitch:
            logger.info("metadata_filter_rejected | key=hitch_type | value=%r", value)
            return None
        return key, hitch
    if key == "subcategory":
        if not _explicit_subcategory_requested(latest_message):
            logger.info("metadata_filter_rejected | key=subcategory | value=%r | reason=not_explicit", value)
            return None
        subcategory = normalize_subcategory(str(value))
        if not subcategory:
            logger.info("metadata_filter_rejected | key=subcategory | value=%r | reason=invalid", value)
            return None
        return key, subcategory
    return key, value


def _fallback_filter_extraction(state: ChatbotState) -> FilterExtractionDecision:
    latest = state.get("user_message") or ""
    if not latest.strip():
        return FilterExtractionDecision()

    text = latest.strip()
    updates: dict[str, Any] = {}
    slot_updates: dict[str, Any] = {}
    shorthand_dimension = re.search(
        r"\b(\d+(?:\.\d+)?)\s*(?:ft|feet|foot|')?\s*[xX]\s*"
        r"(\d+(?:\.\d+)?)\s*(?:ft|feet|foot|')?"
        r"(?:\s*[xX]\s*(\d+(?:\.\d+)?)\s*(?:ft|feet|foot|')?)?\b",
        text,
    )
    if shorthand_dimension:
        # Trailer shorthand is width x length x optional height.
        updates["width_ft"] = shorthand_dimension.group(1)
        updates["length_ft"] = shorthand_dimension.group(2)

    width = _first_match(
        (
            r"\b(\d+(?:\.\d+)?)\s*(ft|feet|foot|')\s*(?:wide|width)\b",
            r"\b(?:width|wide)\D{0,40}?(?:at\s+least|atleast|min(?:imum)?|should\s+be|is|of)?\D{0,20}?(\d+(?:\.\d+)?)\s*(ft|feet|foot|')\b",
        ),
        text,
    )
    width_scrubbed = re.sub(
        r"\b\d+(?:\.\d+)?\s*(?:ft|feet|foot|')\s*(?:wide|width)\b",
        " ",
        text,
        flags=re.I,
    )
    width_scrubbed = re.sub(
        r"\b(?:width|wide)\D{0,60}?\d+(?:\.\d+)?\s*(?:ft|feet|foot|')\b",
        " ",
        width_scrubbed,
        flags=re.I,
    )
    length = _first_match(
        (
            r"\b(\d+(?:\.\d+)?)\s*(ft|feet|foot|')\s*(?:long|length|trailer)\b",
            r"\b(?:length|long|deck\s+length|trailer\s+length|size)\D{0,30}(\d+(?:\.\d+)?)\s*(ft|feet|foot|')\b",
            r"\b(?:make\s+it|change\s+it\s+to|instead)\D{0,20}(\d+(?:\.\d+)?)\s*(ft|feet|foot|')\b",
            r"\b(\d+(?:\.\d+)?)\s*(ft|feet|foot|')\b",
        ),
        width_scrubbed,
    )
    payload = _first_match(
        (
            r"\b(\d+(?:\.\d+)?)\s*(k|m)?\s*(lbs?|pounds?|#)\b",
            r"\b(?:payload|load|haul|weight)\D{0,30}(\d+(?:\.\d+)?)\s*(k|m)?\b",
        ),
        text,
    )
    price = _first_match(
        (
            r"\b(?:under|below|max|budget)\D{0,20}\$?\s*(\d[\d,]*(?:\.\d+)?)\s*(k|m)?\b",
            r"\$\s*(\d[\d,]*(?:\.\d+)?)\s*(k|m)?\b",
        ),
        text,
    )

    if length and "length_ft" not in updates:
        updates["length_ft"] = length
    if width and "width_ft" not in updates:
        updates["width_ft"] = width
    if payload:
        updates["payload_lbs"] = payload
    if price:
        updates["max_price"] = price

    lower = text.lower()
    if "gooseneck" in lower:
        updates["hitch_type"] = "gooseneck"
    elif "bumper pull" in lower or "bumper-pull" in lower:
        updates["hitch_type"] = "bumper pull"

    color_match = re.search(
        r"\b(black|white|gray|grey|silver|red|blue|green|yellow|orange|tan)\b",
        lower,
    )
    if color_match:
        updates["color"] = color_match.group(1)

    return FilterExtractionDecision(**updates, slot_updates=slot_updates)


def _extract_filter_decision(state: ChatbotState, category: str | None) -> FilterExtractionDecision:
    latest = state.get("user_message") or ""
    recent = [
        str(m.get("content") or "")
        for m in reversed((state.get("messages") or [])[-8:])
        if isinstance(m, dict) and m.get("role") == "user"
    ]
    context = {
        "listing_model_fields": sorted(TrailerListing.model_fields.keys()),
        "latest_user_message": latest,
        "recent_user_messages": recent,
        "current_category": category,
        "allowed_category_slots": sorted(_category_slots(category)),
        "slots_collected": state.get("slots_collected") or {},
        "metadata_filters_collected": state.get("metadata_filters_collected") or {},
    }
    try:
        return _filter_extractor_llm().invoke(
            [
                SystemMessage(
                    content=(
                        "Extract only explicit trailer search filters and category slot values from the latest user message. "
                        "Use recent messages only as context, not as new updates.\n"
                        "Rules:\n"
                        "- Return null for fields not explicitly mentioned in the latest user message.\n"
                        "- Width phrases such as 'width should be at least 6 ft' or '6 ft wide' map only to width_ft, never length_ft.\n"
                        "- Length phrases must mention length, long, deck length, trailer length, size, trailer size, or an ambiguous 'make it 14 ft' update.\n"
                        "- Trailer shorthand like '6x12' means width_ft=6 and length_ft=12; '6x12x5' means width_ft=6, length_ft=12, height=5. Do not store height unless there is an allowed slot/filter for it.\n"
                        "- Payload/load/haul weight maps to payload_lbs, not GVWR.\n"
                        "- hitch_type can only be gooseneck or bumper pull; return null for any other value.\n"
                        "- subcategory must only be returned when the latest message explicitly asks for a subcategory filter.\n"
                        "- Do not infer subcategory from category words like Tilt, Utility, Dump, Aluminum, or Enclosed.\n"
                        "- Preserve existing values by returning null unless the latest message updates that exact field.\n"
                        "- slot_updates may include only allowed category slots and only when the latest message provides the value."
                    )
                ),
                HumanMessage(
                    content=(
                        "Return structured extraction for this context:\n"
                        f"{_safe_json(context)}"
                    )
                ),
            ]
        )
    except Exception:
        logger.exception("Filter extractor LLM failed; using regex fallback")
        return _fallback_filter_extraction(state)


def _metadata_filters_from_extraction(
    extraction: FilterExtractionDecision,
    latest_message: str = "",
) -> dict[str, Any]:
    data = _model_dump(extraction)
    updates: dict[str, Any] = {}
    for key in _METADATA_FILTER_KEYS:
        sanitized = _sanitize_metadata_filter_update(key, data.get(key), latest_message)
        if sanitized:
            clean_key, clean_value = sanitized
            updates[clean_key] = clean_value
    return updates


def _metadata_filters_from_decision(
    decision: dict[str, Any],
    latest_message: str = "",
) -> dict[str, Any]:
    raw = decision.get("metadata_filters_update") or {}
    updates: dict[str, Any] = {}
    for key, value in raw.items():
        sanitized = _sanitize_metadata_filter_update(str(key), value, latest_message)
        if sanitized:
            clean_key, clean_value = sanitized
            updates[clean_key] = clean_value
    return updates


def _slot_updates_from_extraction(
    extraction: FilterExtractionDecision,
    allowed_category_slots: set[str],
) -> dict[str, Any]:
    raw = dict(extraction.slot_updates or {})
    return {
        k: v
        for k, v in raw.items()
        if k in allowed_category_slots and v not in (None, "")
    }


def _slot_updates_from_metadata(category: str | None, metadata_filters: dict[str, Any]) -> dict[str, Any]:
    allowed = _category_slots(category)
    if not allowed:
        return {}
    updates: dict[str, Any] = {}

    def pick(candidates: tuple[str, ...]) -> str | None:
        for slot in candidates:
            if slot in allowed:
                return slot
        return None

    length_slot = pick(("trailer_length_ft", "haul_length_ft", "vehicle_length_ft", "trailer_size", "cargo_size"))
    width_slot = pick(("item_or_trailer_width_ft", "trailer_width_ft", "width_ft", "trailer_size", "cargo_size"))
    payload_slot = pick(("haul_weight_lbs", "payload_need", "total_weight"))

    if length_slot and metadata_filters.get("length_ft"):
        updates[length_slot] = metadata_filters["length_ft"]
    if width_slot and width_slot != length_slot and metadata_filters.get("width_ft"):
        updates[width_slot] = metadata_filters["width_ft"]
    if payload_slot and metadata_filters.get("payload_lbs"):
        updates[payload_slot] = metadata_filters["payload_lbs"]
    for key in ("hitch_type", "color", "max_price"):
        if key in allowed and metadata_filters.get(key):
            updates[key] = metadata_filters[key]
    return updates


def _apply_haul_classification_effects(
    *,
    category: str | None,
    slots: dict[str, Any],
    metadata_filters: dict[str, Any],
    classification: HaulClassificationDecision,
) -> tuple[list[str], dict[str, str]]:
    if not category:
        return [], {}

    spec = get_trailer_fields_as_dict(category)
    required_slots = list(spec.get("required_slots") or [])
    dynamic_questions: dict[str, str] = {}
    confident = classification.confidence in _CONFIDENT_CLASSIFICATIONS
    cat = category.strip().lower()

    if confident and classification.matched_item and not slots.get("haul_item"):
        if "haul_item" in _category_slots(category):
            slots["haul_item"] = classification.matched_item
            logger.info(
                "classification_haul_item_filled | category=%r | matched_item=%r | reason=%r",
                category,
                classification.matched_item,
                classification.reason,
            )

    if cat == "utility" and confident and classification.is_lightweight_utility_load:
        required_slots = [slot for slot in required_slots if slot != "haul_weight_lbs"]
        if not metadata_filters.get("payload_lbs") and not slots.get("haul_weight_lbs"):
            metadata_filters["payload_lbs"] = "1500 lbs"
            logger.info(
                "utility_lightweight_payload_default_applied | matched_item=%r | reason=%r",
                classification.matched_item,
                classification.reason,
            )

    if (
        cat not in {"utility", "enclosed"}
        and confident
        and classification.needs_width_question
    ):
        dynamic_questions[_DYNAMIC_WIDTH_SLOT] = _DYNAMIC_WIDTH_QUESTION
        if metadata_filters.get("width_ft"):
            slots[_DYNAMIC_WIDTH_SLOT] = metadata_filters["width_ft"]
        elif _DYNAMIC_WIDTH_SLOT not in required_slots:
            required_slots.append(_DYNAMIC_WIDTH_SLOT)
            logger.info(
                "dynamic_width_question_added | category=%r | matched_item=%r | reason=%r",
                category,
                classification.matched_item,
                classification.reason,
            )

    return required_slots, dynamic_questions


def _filter_pending_for_effective_requirements(
    pending: list[QuestionItem],
    *,
    category: str | None,
    required_slots: list[str],
    dynamic_questions: dict[str, str],
    classification: HaulClassificationDecision,
) -> list[QuestionItem]:
    if not category:
        return pending

    cat = category.strip().lower()
    confident = classification.confidence in _CONFIDENT_CLASSIFICATIONS
    allowed = _category_slots(category)
    effective = set(required_slots) | set(dynamic_questions) | allowed
    filtered: list[QuestionItem] = []

    for question in pending:
        slot = str(question.get("slot") or "")
        if not slot:
            filtered.append(question)
            continue
        if (
            cat == "utility"
            and confident
            and classification.is_lightweight_utility_load
            and slot == "haul_weight_lbs"
        ):
            logger.info(
                "utility_lightweight_weight_question_removed | matched_item=%r",
                classification.matched_item,
            )
            continue
        if slot == _DYNAMIC_WIDTH_SLOT and slot not in effective:
            logger.info("dynamic_width_question_removed | category=%r", category)
            continue
        filtered.append(question)

    return filtered


def _last_listing(state: ChatbotState) -> dict[str, Any] | None:
    listings = state.get("last_listings") or []
    return listings[-1] if listings else None


def _fallback_decision(state: ChatbotState) -> MindDecision:
    text = (state.get("user_message") or "").lower()
    last = _last_listing(state)
    if any(word in text for word in ("interested", "i want", "call me", "buy this", "that one")) and last:
        return MindDecision(
            action="send_interested_listing_email",
            selected_listing_title=last.get("title"),
            selected_listing_url=last.get("url"),
        )
    faq_terms = {
        "financing": "financing",
        "finance": "financing",
        "trade": "trade_in",
        "service": "service_parts",
        "parts": "service_parts",
        "human": "contact_human",
        "contact": "contact_human",
        "store": "store_info",
        "hours": "store_info",
        "location": "store_info",
    }
    for term, category in faq_terms.items():
        if term in text:
            return MindDecision(
                action="send_non_sales_faq_email",
                faq_category=category,
                faq_summary=FAQ_CATEGORY_LABELS.get(category),
            )
    if state.get("trailer_category"):
        return MindDecision(action="pinecone_search")
    return MindDecision(
        action="respond",
        assistant_text="What kind of trailer are you looking for?",
    )


def _queue_questions(
    *,
    category: str,
    slots: dict[str, Any],
    pending: list[QuestionItem],
    optional_slots: list[str],
    required_slots: list[str] | None = None,
    questions_override: dict[str, str] | None = None,
) -> list[QuestionItem]:
    spec = get_trailer_fields_as_dict(category)
    questions = spec.get("questions") or {}
    questions = {**questions, **(questions_override or {})}
    queued_slots = {q.get("slot") for q in pending}
    next_queue = list(pending)

    def add_slot(slot: str, required: bool) -> None:
        if not slot or slot in slots or slot in queued_slots:
            return
        question = questions.get(slot)
        if not question:
            return
        next_queue.append({"slot": slot, "question": question, "required": required})
        queued_slots.add(slot)

    for slot in (required_slots if required_slots is not None else (spec.get("required_slots") or [])):
        add_slot(slot, True)
    allowed_optional = set(spec.get("optional_slots") or [])
    for slot in optional_slots or []:
        if slot in allowed_optional:
            add_slot(slot, False)
    return next_queue


def _is_numeric_slot(slot: str) -> bool:
    slot_l = (slot or "").lower()
    numeric_tokens = (
        "length_ft",
        "width_ft",
        "weight_lbs",
        "gvwr",
        "payload",
        "capacity",
        "size",
    )
    return any(tok in slot_l for tok in numeric_tokens)


def _validate_slot_value(slot: str, value: Any) -> tuple[bool, str]:
    text = str(value or "").strip()
    if not text:
        return False, "empty_value"
    if not _is_numeric_slot(slot):
        return True, ""
    lower = text.lower()
    has_number = bool(re.search(r"\d+(?:\.\d+)?", lower))
    if not has_number:
        return False, "missing_number"

    slot_l = slot.lower()
    is_length_slot = "length_ft" in slot_l or "width_ft" in slot_l or slot_l.endswith("_size")
    is_weight_slot = "weight_lbs" in slot_l or "gvwr" in slot_l or "payload" in slot_l or "capacity" in slot_l

    if is_weight_slot and re.search(r"\b(ft|feet|foot|in|inch|inches)\b|['\"]", lower):
        return False, "length_unit_in_weight_slot"
    if is_length_slot and re.search(r"\b(lb|lbs|pound|pounds|kg|kilogram|ton|tons)\b", lower):
        return False, "weight_unit_in_length_slot"
    return True, ""


def _required_slot_state(
    category: str,
    slots: dict[str, Any],
    *,
    required_slots: list[str] | None = None,
    questions_override: dict[str, str] | None = None,
) -> tuple[list[str], list[str], dict[str, str]]:
    spec = get_trailer_fields_as_dict(category)
    required_slots = list(required_slots if required_slots is not None else (spec.get("required_slots") or []))
    questions = dict(spec.get("questions") or {})
    questions.update(questions_override or {})
    missing: list[str] = []
    invalid: list[str] = []
    for slot in required_slots:
        if slot not in slots:
            missing.append(slot)
            continue
        ok, _reason = _validate_slot_value(slot, slots.get(slot))
        if not ok:
            invalid.append(slot)
    return missing, invalid, questions


def _mind_node(state: ChatbotState) -> ChatbotState:
    resolution = resolve_category_from_text(state.get("user_message") or "")
    if resolution.needs_clarification and not state.get("trailer_category"):
        clarification = resolution.clarification_question or ""
        return {
            **state,
            "category_needs_clarification": True,
            "assistant_text": clarification,
            "mind_decision": {"action": "respond", "assistant_text": clarification},
        }

    category_locked_pre_results = bool(state.get("trailer_category")) and not bool(
        state.get("has_shown_search_results")
    )
    deterministic_hint = None if category_locked_pre_results else resolution.category
    if category_locked_pre_results:
        logger.info(
            "category_lock_active | phase=mind_node | category=%r | has_shown_search_results=%s | proposed_category=%r",
            state.get("trailer_category"),
            state.get("has_shown_search_results"),
            resolution.category,
        )

    context = {
        "user_message": state.get("user_message"),
        "customer": {
            "name": state.get("customer_full_name"),
            "email": state.get("customer_email"),
            "phone": state.get("customer_phone"),
        },
        "current_category": state.get("trailer_category"),
        "deterministic_category_hint": deterministic_hint,
        "listing_model_fields": sorted(TrailerListing.model_fields.keys()),
        "slots_collected": state.get("slots_collected") or {},
        "metadata_filters_collected": state.get("metadata_filters_collected") or {},
        "awaiting_slot": state.get("awaiting_slot"),
        "pending_questions": state.get("pending_questions") or [],
        "asked_questions": state.get("asked_questions") or [],
        "last_listings": state.get("last_listings") or [],
        "already_shown_listing_urls": state.get("already_shown_listing_urls") or [],
        "recent_messages": (state.get("messages") or [])[-8:],
    }
    try:
        decision = _mind_llm().invoke(
            [
                SystemMessage(content=MIND_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        "Return the next action as structured output for this state:\n"
                        f"{_safe_json(context)}"
                    )
                ),
            ]
        )
        if deterministic_hint and not decision.trailer_category:
            decision.trailer_category = deterministic_hint
    except Exception:
        logger.exception("Mind LLM failed; using fallback decision")
        decision = _fallback_decision(state)
        if deterministic_hint and not decision.trailer_category:
            decision.trailer_category = deterministic_hint

    return {**state, "mind_decision": _model_dump(decision)}


def _apply_mind_node(state: ChatbotState) -> ChatbotState:
    decision = dict(state.get("mind_decision") or {})
    slots_before = dict(state.get("slots_collected") or {})
    slots = dict(slots_before)
    metadata_filters_before = dict(state.get("metadata_filters_collected") or {})
    metadata_filters = dict(metadata_filters_before)
    invalid_required_slot: str | None = None
    awaiting_slot = state.get("awaiting_slot")
    if awaiting_slot and awaiting_slot not in slots and (state.get("user_message") or "").strip():
        candidate_value = (state.get("user_message") or "").strip()
        is_valid, reason = _validate_slot_value(awaiting_slot, candidate_value)
        if is_valid:
            slots[awaiting_slot] = candidate_value
            awaiting_slot = None
        else:
            invalid_required_slot = awaiting_slot
            logger.info(
                "slot_validation_failed | slot=%s | value=%r | reason=%s",
                awaiting_slot,
                candidate_value,
                reason,
            )
    category_before = state.get("trailer_category")
    category_proposed = decision.get("trailer_category") or category_before
    category_locked_pre_results = bool(category_before) and not bool(
        state.get("has_shown_search_results")
    )
    if category_locked_pre_results and category_proposed and category_proposed != category_before:
        logger.info(
            "category_change_blocked_pre_results | current_category=%r | proposed_category=%r | has_shown_search_results=%s",
            category_before,
            category_proposed,
            state.get("has_shown_search_results"),
        )
        category = category_before
    else:
        category = category_proposed

    category_changed = bool(category_before and category and category != category_before)
    reset_result_state = False
    if category_changed:
        logger.info(
            "category_change_clears_filters | old_category=%r | new_category=%r",
            category_before,
            category,
        )
        slots = {}
        metadata_filters = {}
        awaiting_slot = None
        reset_result_state = True

    allowed_category_slots = _category_slots(category)
    for key, value in (decision.get("slots_collected_update") or {}).items():
        if value in (None, ""):
            continue
        if category and key not in allowed_category_slots:
            continue
        is_valid, reason = _validate_slot_value(key, value)
        if not is_valid:
            logger.info(
                "slot_validation_failed | slot=%s | value=%r | reason=%s",
                key,
                value,
                reason,
            )
            if not invalid_required_slot:
                invalid_required_slot = key
            continue
        slots[key] = value

    latest_message = state.get("user_message") or ""
    for key, value in _metadata_filters_from_decision(decision, latest_message).items():
        metadata_filters[key] = value
    extraction = _extract_filter_decision(
        {
            **state,
            "trailer_category": category,
            "slots_collected": slots,
            "metadata_filters_collected": metadata_filters,
        },
        category,
    )
    extracted_metadata = _metadata_filters_from_extraction(extraction, latest_message)
    extracted_slots = _slot_updates_from_extraction(extraction, allowed_category_slots)
    logger.info(
        "filter_extraction_applied | category=%r | extracted_metadata=%s | extracted_slots=%s",
        category,
        json.dumps(extracted_metadata, default=str),
        json.dumps(extracted_slots, default=str),
    )
    for key, value in extracted_metadata.items():
        if key in _METADATA_FILTER_KEYS and value not in (None, ""):
            metadata_filters[key] = value
    for key, value in extracted_slots.items():
        is_valid, reason = _validate_slot_value(key, value)
        if is_valid:
            slots[key] = value
        else:
            logger.info(
                "slot_validation_failed | slot=%s | value=%r | reason=%s",
                key,
                value,
                reason,
            )

    for key, value in _slot_updates_from_metadata(category, metadata_filters).items():
        if key in slots or value in (None, ""):
            continue
        is_valid, reason = _validate_slot_value(key, value)
        if is_valid:
            slots[key] = value
        else:
            logger.info(
                "slot_validation_failed | slot=%s | value=%r | reason=%s",
                key,
                value,
                reason,
            )
            if not invalid_required_slot:
                invalid_required_slot = key
    if awaiting_slot and awaiting_slot in slots:
        awaiting_slot = None

    haul_classification = classify_haul_requirements(
        category=category,
        user_message=state.get("user_message") or "",
        recent_messages=state.get("messages") or [],
        slots_collected=slots,
        metadata_filters_collected=metadata_filters,
    )
    required_slots_override, dynamic_questions = _apply_haul_classification_effects(
        category=category,
        slots=slots,
        metadata_filters=metadata_filters,
        classification=haul_classification,
    )
    if awaiting_slot and awaiting_slot in slots:
        awaiting_slot = None
    logger.info(
        "haul_classification_applied | category=%r | classification=%s | required_slots=%s | dynamic_questions=%s",
        category,
        json.dumps(_model_dump(haul_classification), default=str),
        json.dumps(required_slots_override, default=str),
        json.dumps(dynamic_questions, default=str),
    )

    missing_required: list[str] = []
    invalid_required: list[str] = []
    questions_by_slot: dict[str, str] = {}
    if category:
        missing_required, invalid_required, questions_by_slot = _required_slot_state(
            category,
            slots,
            required_slots=required_slots_override,
            questions_override=dynamic_questions,
        )
        for slot in invalid_required:
            if slot in slots:
                del slots[slot]
        if invalid_required:
            missing_required = list(dict.fromkeys(missing_required + invalid_required))
    logger.info(
        "required_completion_state | category=%r | missing=%s | invalid=%s",
        category,
        json.dumps(missing_required),
        json.dumps(invalid_required),
    )

    pending_source = [] if reset_result_state else (state.get("pending_questions") or [])
    pending = [
        q for q in pending_source
        if q.get("slot") not in slots
    ]
    pending = _filter_pending_for_effective_requirements(
        pending,
        category=category,
        required_slots=required_slots_override,
        dynamic_questions=dynamic_questions,
        classification=haul_classification,
    )
    if category:
        pending = _queue_questions(
            category=category,
            slots=slots,
            pending=pending,
            optional_slots=decision.get("optional_question_slots_to_queue") or [],
            required_slots=required_slots_override,
            questions_override=dynamic_questions,
        )

    valid_actions = {
        "ask_next_question",
        "pinecone_search",
        "send_interested_listing_email",
        "send_non_sales_faq_email",
        "respond",
    }
    action = decision.get("action") or "respond"
    if action not in valid_actions:
        action = "respond"
    required_complete = bool(category) and not missing_required and not invalid_required
    if required_complete and action not in {"send_interested_listing_email", "send_non_sales_faq_email"}:
        if action != "pinecone_search":
            logger.info(
                "action_corrected_to_search | previous_action=%s | category=%r",
                action,
                category,
            )
        action = "pinecone_search"
    should_ask = action == "ask_next_question" or (
        action in {"pinecone_search", "respond"} and bool(pending)
    )

    asked = [] if reset_result_state else list(state.get("asked_questions") or [])
    assistant_text = decision.get("assistant_text") or ""
    if should_ask and pending:
        next_question = pending.pop(0)
        assistant_text = next_question.get("question") or "Could you share a little more detail?"
        if next_question.get("slot"):
            awaiting_slot = str(next_question["slot"])
            asked.append(awaiting_slot)
        action = "respond"
    elif action == "ask_next_question" and not pending:
        logger.info(
            "ask_next_empty_queue_corrected | category=%r | required_complete=%s",
            category,
            required_complete,
        )
        if required_complete and category:
            action = "pinecone_search"
        else:
            action = "respond"
            unresolved_slot = invalid_required_slot or (missing_required[0] if missing_required else None)
            if unresolved_slot:
                assistant_text = questions_by_slot.get(unresolved_slot) or "Could you share that detail?"
                awaiting_slot = unresolved_slot
            elif not assistant_text.strip():
                assistant_text = "Could you share a little more detail so I can narrow this down?"
    elif action == "pinecone_search" and not category:
        assistant_text = "What kind of trailer are you looking for?"
        action = "respond"
    elif invalid_required_slot and action != "pinecone_search":
        action = "respond"
        slot_prompt = questions_by_slot.get(invalid_required_slot)
        if slot_prompt:
            assistant_text = slot_prompt
            awaiting_slot = invalid_required_slot
        elif not assistant_text.strip():
            assistant_text = "Could you please confirm that requirement?"
    elif action == "respond" and not assistant_text:
        assistant_text = "How can I help with your trailer search?"

    slots_after = dict(slots)
    metadata_filters_after = dict(metadata_filters)
    slot_changes = {
        k: {"before": slots_before.get(k), "after": slots_after.get(k)}
        for k in set(slots_before.keys()) | set(slots_after.keys())
        if slots_before.get(k) != slots_after.get(k)
    }
    metadata_filter_changes = {
        k: {"before": metadata_filters_before.get(k), "after": metadata_filters_after.get(k)}
        for k in set(metadata_filters_before.keys()) | set(metadata_filters_after.keys())
        if metadata_filters_before.get(k) != metadata_filters_after.get(k)
    }
    logger.info(
        "mind_decision_applied | action=%s | category_before=%r | category_after=%r | slot_changes=%s | metadata_filter_changes=%s | pending_count=%s",
        action,
        category_before,
        category,
        json.dumps(slot_changes, default=str),
        json.dumps(metadata_filter_changes, default=str),
        len(pending),
    )

    decision["action"] = action
    return {
        **state,
        "trailer_category": category,
        "slots_collected": slots,
        "metadata_filters_collected": metadata_filters,
        "awaiting_slot": awaiting_slot,
        "pending_questions": pending,
        "asked_questions": asked,
        "already_shown_listing_urls": [] if reset_result_state else state.get("already_shown_listing_urls"),
        "last_listings": [] if reset_result_state else state.get("last_listings"),
        "assistant_text": assistant_text,
        "selected_listing_title": decision.get("selected_listing_title"),
        "selected_listing_url": decision.get("selected_listing_url"),
        "mind_decision": decision,
    }


def _route_after_mind(state: ChatbotState) -> str:
    action = (state.get("mind_decision") or {}).get("action")
    if action == "pinecone_search":
        return "pinecone_search"
    if action == "send_interested_listing_email":
        return "send_interested_listing_email"
    if action == "send_non_sales_faq_email":
        return "send_non_sales_faq_email"
    return END


def _pinecone_search_node(state: ChatbotState) -> ChatbotState:
    logger.info(
        "pinecone_search_execute | category=%r | slots=%s | metadata_filters=%s | shown_count=%s",
        state.get("trailer_category"),
        json.dumps(state.get("slots_collected") or {}, default=str),
        json.dumps(state.get("metadata_filters_collected") or {}, default=str),
        len(state.get("already_shown_listing_urls") or []),
    )
    listings = search_pinecone_listings(
        category=state.get("trailer_category"),
        slots=state.get("slots_collected") or {},
        metadata_filters=state.get("metadata_filters_collected") or {},
        user_message=state.get("user_message") or "",
        already_shown_urls=state.get("already_shown_listing_urls") or [],
    )
    assistant_text = format_listing_results(
        listings,
        category=state.get("trailer_category"),
        slots=state.get("slots_collected") or {},
        user_message=state.get("user_message") or "",
    )
    shown = list(state.get("already_shown_listing_urls") or [])
    shown.extend([str(x.get("url")) for x in listings if x.get("url")])
    events = list(state.get("tool_events") or [])
    events.append({"tool": "pinecone_search", "result_count": len(listings)})
    return {
        **state,
        "assistant_text": assistant_text,
        "last_listings": listings,
        "already_shown_listing_urls": sorted(set(shown)),
        "tool_events": events,
    }


def _interest_email_node(state: ChatbotState) -> ChatbotState:
    decision = state.get("mind_decision") or {}
    title = decision.get("selected_listing_title") or state.get("selected_listing_title")
    if not title and _last_listing(state):
        title = _last_listing(state).get("title")
    if not title:
        return {
            **state,
            "assistant_text": "Which listing are you interested in?",
        }
    result = send_interested_listing_email(
        session_id=state.get("session_id") or "",
        full_name=state.get("customer_full_name") or "",
        email=state.get("customer_email"),
        phone=state.get("customer_phone") or "",
        item_name=str(title),
    )
    planner_reply = str(decision.get("assistant_text") or state.get("assistant_text") or "").strip()
    if planner_reply:
        assistant_text = planner_reply
        reply_source = "planner_assistant_text"
    elif title:
        assistant_text = _INTEREST_GENERIC_FALLBACK.format(item_name=str(title).strip())
        reply_source = "interest_fallback"
    else:
        assistant_text = _INTEREST_GENERIC_FALLBACK_NO_ITEM
        reply_source = "generic_fallback"
    if not assistant_text.strip():
        assistant_text = _INTEREST_SAFE_FALLBACK
        reply_source = "safe_fallback"
    logger.debug(
        "interest_reply_source=%s title_present=%s",
        reply_source,
        bool(str(title or "").strip()),
    )
    events = list(state.get("tool_events") or [])
    events.append({"tool": "send_interested_listing_email", "result": result})
    return {
        **state,
        "assistant_text": assistant_text,
        "tool_events": events,
    }


def _faq_email_node(state: ChatbotState) -> ChatbotState:
    decision = state.get("mind_decision") or {}
    category = (decision.get("faq_category") or "contact_human").strip().lower()
    summary = decision.get("faq_summary") or FAQ_CATEGORY_LABELS.get(category)
    result = send_non_sales_faq_email(
        session_id=state.get("session_id") or "",
        full_name=state.get("customer_full_name") or "",
        email=state.get("customer_email"),
        phone=state.get("customer_phone") or "",
        faq_category=category,
        summary=summary,
    )
    planner_reply = str(decision.get("assistant_text") or state.get("assistant_text") or "").strip()
    category_reply = _FAQ_REPLY_FALLBACKS.get(category, "").strip()
    if planner_reply:
        assistant_text = planner_reply
        reply_source = "planner_assistant_text"
    elif category_reply:
        assistant_text = category_reply
        reply_source = "category_fallback"
    else:
        assistant_text = _FAQ_GENERIC_FALLBACK
        reply_source = "generic_fallback"
    logger.debug(
        "faq_reply_source=%s category=%s summary=%r",
        reply_source,
        category,
        summary,
    )
    events = list(state.get("tool_events") or [])
    events.append({"tool": "send_non_sales_faq_email", "result": result})
    return {
        **state,
        "assistant_text": assistant_text,
        "tool_events": events,
    }


@lru_cache(maxsize=1)
def build_chatbot_graph():
    graph = StateGraph(ChatbotState)
    graph.add_node("mind", _mind_node)
    graph.add_node("apply_mind", _apply_mind_node)
    graph.add_node("pinecone_search", _pinecone_search_node)
    graph.add_node("send_interested_listing_email", _interest_email_node)
    graph.add_node("send_non_sales_faq_email", _faq_email_node)
    graph.set_entry_point("mind")
    graph.add_edge("mind", "apply_mind")
    graph.add_conditional_edges("apply_mind", _route_after_mind)
    graph.add_edge("pinecone_search", END)
    graph.add_edge("send_interested_listing_email", END)
    graph.add_edge("send_non_sales_faq_email", END)
    return graph.compile()
