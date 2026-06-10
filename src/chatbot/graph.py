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

from trailer_fields import get_trailer_fields_as_dict, list_all_categories
from src.chatbot.categories import resolve_category_from_text
from src.chatbot.formatting import format_listing_results
from src.chatbot.mini_llm_classifier import (
    HaulClassificationDecision,
    classify_haul_requirements,
)
from src.chatbot.mini_preference_classifier import (
    PreferenceNullDecision,
    classify_no_preference,
)
from src.chatbot.make_inventory import categories_for_make
from src.chatbot.make_resolver import resolve_make_from_text
from src.chatbot.prompts import MIND_SYSTEM_PROMPT
from src.chatbot.state import ChatbotState, QuestionItem
from src.models import TrailerListing
from src.normalizer import normalize_category, normalize_hitch, normalize_subcategory
from src.chatbot.tools.email_tools import (
    FAQ_CATEGORY_LABELS,
    send_interested_listing_email,
    send_non_sales_faq_email,
)
from src.chatbot.tools.pinecone_search import (
    PineconeListingSearchResult,
    search_pinecone_listing_result,
)

load_dotenv()

logger = logging.getLogger(__name__)
_SITE_URL = (os.getenv("TRAILERPLACE_SITE_URL") or "https://trailerplace.com").strip()
_CATALOGUE_REDIRECT_REPLY = (
    f"To see the full lineup of available trailers, the best place to browse is [TrailerPlace]({_SITE_URL}). "
    "You can compare inventory at your pace, and I can still help narrow things down whenever you have a "
    "trailer type, size, or use case in mind."
)
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
    make: Optional[str] = None
    color: Optional[str] = None
    slot_updates: dict[str, Any] = Field(default_factory=dict)


class RequestedFeatureExtractionDecision(BaseModel):
    requested_non_metadata_features: list[str] = Field(default_factory=list)
    reason: str = ""


class PineconeListingMatchDecision(BaseModel):
    position: int = 0
    match_level: Literal["full", "partial", "alternative", "unknown"] = "unknown"
    confirmed_requirements: list[str] = Field(default_factory=list)
    missing_or_unconfirmed_requirements: list[str] = Field(default_factory=list)
    customer_label: str = ""
    short_reason: str = ""
    sales_blurb: str = ""


class PineconeMatchFramingDecision(BaseModel):
    intro_text: str = ""
    overall_match_level: Literal["full", "mixed", "partial_only", "no_exact", "unknown"] = "unknown"
    full_match_count: int = 0
    partial_match_count: int = 0
    alternative_count: int = 0
    requested_non_metadata_features: list[str] = Field(default_factory=list)
    per_listing_match: list[PineconeListingMatchDecision] = Field(default_factory=list)
    reason: str = ""


class PineconeSalesIntroDecision(BaseModel):
    intro_text: str = ""
    reason: str = ""


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


@lru_cache(maxsize=1)
def _requested_feature_extractor_llm():
    model = (
        os.getenv("REQUESTED_FEATURE_EXTRACTOR_MODEL")
        or os.getenv("OPENAI_MODEL")
        or "gpt-4o-mini"
    ).strip()
    return ChatOpenAI(model=model, temperature=0).with_structured_output(
        RequestedFeatureExtractionDecision,
        method="function_calling",
    )


def _result_interest_followup_llm_enabled() -> bool:
    return (os.getenv("RESULT_INTEREST_FOLLOWUP_LLM_ENABLED") or "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


@lru_cache(maxsize=1)
def _result_interest_followup_llm():
    model = (
        os.getenv("RESULT_INTEREST_FOLLOWUP_MODEL")
        or os.getenv("OPENAI_MODEL")
        or "gpt-4o-mini"
    ).strip()
    return ChatOpenAI(model=model, temperature=0.3)


def _pinecone_match_framing_llm_enabled() -> bool:
    return (os.getenv("PINECONE_MATCH_FRAMING_LLM_ENABLED") or "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


@lru_cache(maxsize=1)
def _pinecone_match_audit_llm():
    model = (
        os.getenv("PINECONE_MATCH_AUDIT_MODEL")
        or os.getenv("PINECONE_MATCH_FRAMING_MODEL")
        or os.getenv("OPENAI_MODEL")
        or "gpt-4o-mini"
    ).strip()
    return ChatOpenAI(model=model, temperature=0).with_structured_output(
        PineconeMatchFramingDecision,
        method="function_calling",
    )


@lru_cache(maxsize=1)
def _pinecone_sales_intro_llm():
    model = (
        os.getenv("PINECONE_SALES_INTRO_MODEL")
        or os.getenv("PINECONE_MATCH_FRAMING_MODEL")
        or os.getenv("OPENAI_MODEL")
        or "gpt-4o-mini"
    ).strip()
    return ChatOpenAI(model=model, temperature=0.25).with_structured_output(
        PineconeSalesIntroDecision,
        method="function_calling",
    )


def _model_dump(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _safe_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=True, default=str)


_PINECONE_INTERNAL_LISTING_KEYS = {"match_evidence_text"}
_PINECONE_MATCH_FRAMING_FALLBACK = (
    "Here are the strongest available options I found based on your search."
)
_PINECONE_MATCH_LEVEL_ORDER = {"full": 0, "partial": 1, "alternative": 2, "unknown": 3}


def _public_listing(listing: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in dict(listing or {}).items()
        if key not in _PINECONE_INTERNAL_LISTING_KEYS
    }


def _safe_match_level(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in _PINECONE_MATCH_LEVEL_ORDER:
        return normalized
    if normalized == "no_exact":
        return "alternative"
    return "unknown"


def _normalize_per_listing_matches(data: dict[str, Any], listing_count: int) -> list[dict[str, Any]]:
    raw_matches = data.get("per_listing_match") or []
    normalized: list[dict[str, Any]] = []
    seen_positions: set[int] = set()
    if isinstance(raw_matches, list):
        for raw in raw_matches:
            item = raw if isinstance(raw, dict) else _model_dump(raw) if isinstance(raw, BaseModel) else {}
            try:
                position = int(item.get("position") or 0)
            except Exception:
                position = 0
            if position < 1 or position > listing_count or position in seen_positions:
                continue
            level = _safe_match_level(item.get("match_level"))
            seen_positions.add(position)
            normalized.append(
                {
                    "position": position,
                    "match_level": level,
                    "confirmed_requirements": [
                        str(x).strip()
                        for x in (item.get("confirmed_requirements") or [])
                        if str(x).strip()
                    ],
                    "missing_or_unconfirmed_requirements": [
                        str(x).strip()
                        for x in (item.get("missing_or_unconfirmed_requirements") or [])
                        if str(x).strip()
                    ],
                    "customer_label": str(item.get("customer_label") or "").strip(),
                    "short_reason": str(item.get("short_reason") or "").strip(),
                    "sales_blurb": str(item.get("sales_blurb") or "").strip(),
                }
            )
    return normalized


def _normalized_requirement_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _requirement_confirms_feature(confirmed_requirement: Any, requested_feature: Any) -> bool:
    confirmed = _normalized_requirement_text(confirmed_requirement)
    requested = _normalized_requirement_text(requested_feature)
    if not confirmed or not requested:
        return False
    return requested in confirmed or confirmed in requested


def _enforce_requested_feature_consistency(
    per_listing_match: list[dict[str, Any]],
    requested_features: list[Any],
) -> list[dict[str, Any]]:
    clean_requested = [
        str(feature).strip()
        for feature in requested_features
        if str(feature or "").strip()
    ]
    if not clean_requested:
        return per_listing_match

    enforced: list[dict[str, Any]] = []
    for item in per_listing_match:
        updated = dict(item)
        if _safe_match_level(updated.get("match_level")) != "full":
            enforced.append(updated)
            continue

        confirmed = updated.get("confirmed_requirements") or []
        missing_features = [
            feature
            for feature in clean_requested
            if not any(_requirement_confirms_feature(req, feature) for req in confirmed)
        ]
        if missing_features:
            existing_missing = [
                str(x).strip()
                for x in (updated.get("missing_or_unconfirmed_requirements") or [])
                if str(x).strip()
            ]
            for feature in missing_features:
                if not any(_requirement_confirms_feature(existing, feature) for existing in existing_missing):
                    existing_missing.append(feature)
            updated["missing_or_unconfirmed_requirements"] = existing_missing
            updated["match_level"] = "partial" if confirmed else "alternative"
            if not updated.get("customer_label"):
                updated["customer_label"] = "Strong available option"
        enforced.append(updated)
    return enforced


def _normalize_requested_feature_list(features: list[Any]) -> list[str]:
    clean: list[str] = []
    seen: set[str] = set()
    for feature in features or []:
        value = re.sub(r"\s+", " ", str(feature or "").strip())
        key = _normalized_requirement_text(value)
        if not key or key in seen:
            continue
        seen.add(key)
        clean.append(value)
    return clean


def _structured_listing_context_matches(
    *,
    fact: dict[str, Any],
    category: str | None,
    metadata_filters: dict[str, Any],
) -> bool:
    fact_category = normalize_category(fact.get("category"))
    requested_category = normalize_category(category)
    if requested_category and fact_category != requested_category:
        return False

    requested_make = _normalized_requirement_text(metadata_filters.get("make"))
    if requested_make and requested_make != _normalized_requirement_text(fact.get("make")):
        return False

    requested_hitch = normalize_hitch(metadata_filters.get("hitch_type"))
    fact_hitch = normalize_hitch(fact.get("hitch_type"))
    if requested_hitch and requested_hitch != fact_hitch:
        return False

    requested_color = _normalized_requirement_text(metadata_filters.get("color"))
    if requested_color and requested_color != _normalized_requirement_text(fact.get("color")):
        return False

    requested_subcategory = normalize_subcategory(metadata_filters.get("subcategory"))
    fact_subcategory = normalize_subcategory(fact.get("subcategory"))
    if requested_subcategory and requested_subcategory != fact_subcategory:
        return False

    return True


def _sanitize_requested_feature_analysis(
    *,
    data: dict[str, Any],
    facts: list[dict[str, Any]],
    requested_features: list[Any],
    category: str | None,
    metadata_filters: dict[str, Any],
) -> dict[str, Any]:
    clean_requested = _normalize_requested_feature_list(requested_features)
    allowed_keys = {_normalized_requirement_text(feature) for feature in clean_requested}
    by_position = {
        int(fact.get("position") or 0): fact
        for fact in facts
        if str(fact.get("position") or "").isdigit()
    }
    sanitized: list[dict[str, Any]] = []

    for item in _normalize_per_listing_matches(data, len(facts)):
        updated = dict(item)
        original_missing = [
            str(value).strip()
            for value in (updated.get("missing_or_unconfirmed_requirements") or [])
            if str(value).strip()
        ]
        filtered_missing = [
            value
            for value in original_missing
            if _normalized_requirement_text(value) in allowed_keys
        ]
        updated["missing_or_unconfirmed_requirements"] = filtered_missing
        if (
            filtered_missing != original_missing
            and not filtered_missing
            and _structured_listing_context_matches(
                fact=by_position.get(int(updated.get("position") or 0), {}),
                category=category,
                metadata_filters=metadata_filters,
            )
        ):
            updated["match_level"] = "full"
        sanitized.append(updated)

    sanitized = _enforce_requested_feature_consistency(sanitized, clean_requested)
    full_count, partial_count, alternative_count = _match_count_summary(sanitized)
    overall_match_level = "unknown"
    if full_count > 0 and full_count == len(facts):
        overall_match_level = "full"
    elif full_count > 0:
        overall_match_level = "mixed"
    elif partial_count > 0:
        overall_match_level = "partial_only"
    elif alternative_count > 0:
        overall_match_level = "no_exact"

    return {
        **data,
        "intro_text": "",
        "requested_non_metadata_features": clean_requested,
        "per_listing_match": sanitized,
        "full_match_count": full_count,
        "partial_match_count": partial_count,
        "alternative_count": alternative_count,
        "validated_listing_count": len(facts),
        "overall_match_level": overall_match_level,
    }


def _fallback_match_analysis(*, source: str, listing_count: int = 0) -> dict[str, Any]:
    return {
        "overall_match_level": "unknown",
        "full_match_count": 0,
        "partial_match_count": 0,
        "alternative_count": 0,
        "requested_non_metadata_features": [],
        "per_listing_match": [],
        "source": source,
        "validated_listing_count": listing_count,
    }


def _log_pinecone_match_audit(
    *,
    match_analysis: dict[str, Any],
    facts: list[dict[str, Any]],
    query_text: str,
) -> None:
    logger.info(
        "pinecone_match_audit_summary | source=%r | overall_match_level=%r | full_match_count=%s | "
        "partial_match_count=%s | alternative_count=%s | requested_non_metadata_features=%s | query_text=%r",
        match_analysis.get("source") or "audit",
        match_analysis.get("overall_match_level"),
        match_analysis.get("full_match_count"),
        match_analysis.get("partial_match_count"),
        match_analysis.get("alternative_count"),
        _safe_json(match_analysis.get("requested_non_metadata_features") or []),
        query_text,
    )
    by_position = {
        int(item.get("position") or 0): item
        for item in (match_analysis.get("per_listing_match") or [])
        if str(item.get("position") or "").isdigit()
    }
    for fact in facts:
        position = int(fact.get("position") or 0)
        item = by_position.get(position) or {}
        logger.info(
            "pinecone_match_audit_listing | position=%s | match_level=%r | title=%r | url=%r | "
            "confirmed_requirements=%s | missing_or_unconfirmed_requirements=%s | customer_label=%r | short_reason=%r",
            position,
            item.get("match_level") or "unknown",
            fact.get("title"),
            fact.get("url"),
            _safe_json(item.get("confirmed_requirements") or []),
            _safe_json(item.get("missing_or_unconfirmed_requirements") or []),
            item.get("customer_label") or "",
            item.get("short_reason") or "",
        )


def _log_pinecone_intro_fallback(
    *,
    source: str,
    reason: str,
    intro_candidate: str = "",
    match_analysis: dict[str, Any] | None = None,
) -> None:
    data = match_analysis or {}
    logger.info(
        "pinecone_match_intro_fallback | source=%r | reason=%r | overall_match_level=%r | "
        "full_match_count=%s | partial_match_count=%s | alternative_count=%s | intro_candidate=%r",
        source,
        reason,
        data.get("overall_match_level"),
        data.get("full_match_count"),
        data.get("partial_match_count"),
        data.get("alternative_count"),
        intro_candidate,
    )


def _match_count_summary(per_listing_match: list[dict[str, Any]]) -> tuple[int, int, int]:
    full = sum(1 for item in per_listing_match if item.get("match_level") == "full")
    partial = sum(1 for item in per_listing_match if item.get("match_level") == "partial")
    alternative = sum(1 for item in per_listing_match if item.get("match_level") == "alternative")
    return full, partial, alternative


def _invalid_pinecone_intro(
    *,
    text: str,
    data: dict[str, Any],
    listing_count: int,
) -> bool:
    lower = text.lower()
    full_count = int(data.get("full_match_count") or 0)
    per_listing_match = data.get("per_listing_match") or []
    all_full = listing_count > 0 and full_count >= listing_count
    if full_count <= 0 and re.search(r"\b(?:fully|exact(?:ly)?|perfect(?:ly)?)\s+match", lower):
        return True
    if full_count <= 0 and re.search(
        r"\b(?:strong|clear|close|closest|best|top|solid)\s+match(?:es)?\b"
        r"|\bclosely\s+match(?:es)?\b"
        r"|\bbest-fitting\b",
        lower,
    ):
        return True
    if not all_full and re.search(r"\b(?:all|every|each)\b.{0,40}\b(?:fully|exact(?:ly)?)\s+match", lower):
        return True
    if not all_full and re.search(r"\bmeet(?:s)?\s+(?:your\s+)?(?:specifications|requirements|needs)\b", lower):
        return True
    if re.search(
        r"\b(?:exceeds?|does not meet|doesn't meet|not meet|mismatch|too wide|too tall|too short|"
        r"too long|shorter than|longer than|requirement(?:s)? of|required value)\b",
        lower,
    ):
        return True
    requested_features = [
        str(x).strip().lower()
        for x in (data.get("requested_non_metadata_features") or [])
        if str(x).strip()
    ]
    if requested_features:
        confirmed_features = {
            str(req).strip().lower()
            for item in per_listing_match
            if item.get("match_level") == "full"
            for req in (item.get("confirmed_requirements") or [])
        }
        for feature in requested_features:
            feature_claim = re.search(
                rf"\b(?:features?|includes?|has|have|with)\b.{{0,50}}{re.escape(feature)}"
                rf"|{re.escape(feature)}.{{0,35}}\b(?:option|match|available|included)\b",
                lower,
                re.I,
            )
            if feature_claim and not any(feature in confirmed or confirmed in feature for confirmed in confirmed_features):
                return True
    return False


def _apply_pinecone_match_validation(
    listings: list[dict[str, Any]],
    match_analysis: dict[str, Any],
) -> list[dict[str, Any]]:
    if not listings or not match_analysis:
        return listings
    by_position = {
        int(item.get("position")): item
        for item in (match_analysis.get("per_listing_match") or [])
        if str(item.get("position") or "").isdigit()
    }
    enriched: list[tuple[int, int, dict[str, Any]]] = []
    for original_idx, listing in enumerate(listings, 1):
        next_listing = dict(listing)
        validation = by_position.get(original_idx)
        if validation:
            validation = {**validation, "requested_non_metadata_features": match_analysis.get("requested_non_metadata_features") or []}
            next_listing["match_validation"] = validation
            order = _PINECONE_MATCH_LEVEL_ORDER.get(_safe_match_level(validation.get("match_level")), 3)
        else:
            order = 3
        enriched.append((order, original_idx, next_listing))
    enriched.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in enriched]


_ACTIVE_REQUEST_SUFFIX_MARKER = " | Current requirements: "


def _active_request_base(text: str) -> str:
    base = str(text or "").strip()
    if _ACTIVE_REQUEST_SUFFIX_MARKER in base:
        base = base.split(_ACTIVE_REQUEST_SUFFIX_MARKER, 1)[0].strip()
    return re.sub(r"\s+", " ", base)


def _active_requirement_parts(
    *,
    slots: dict[str, Any],
    metadata_filters: dict[str, Any],
) -> list[str]:
    parts: list[str] = []
    seen: set[str] = set()

    def add(label: str, value: Any) -> None:
        clean = re.sub(r"\s+", " ", str(value or "").strip())
        if not clean:
            return
        key = label.lower()
        if key in seen:
            return
        seen.add(key)
        parts.append(f"{label} {clean}")

    add("make", metadata_filters.get("make"))
    add("subcategory", metadata_filters.get("subcategory"))
    add("length", metadata_filters.get("length_ft") or slots.get("trailer_length_ft"))
    add("width", metadata_filters.get("width_ft") or slots.get("trailer_width_ft"))
    add("height", metadata_filters.get("height_ft"))
    add("payload", metadata_filters.get("payload_lbs"))
    add("hitch", metadata_filters.get("hitch_type"))
    add("color", metadata_filters.get("color"))
    add("max price", metadata_filters.get("max_price"))
    return parts


_ACTIVE_REQUEST_FILLER_WORDS = {
    "a",
    "about",
    "an",
    "and",
    "am",
    "be",
    "for",
    "i",
    "it",
    "looking",
    "me",
    "need",
    "please",
    "should",
    "that",
    "the",
    "trailer",
    "trailers",
    "want",
    "with",
}


def _looks_like_contact_or_admin_text(text: str) -> bool:
    clean = str(text or "").strip()
    if not clean:
        return True
    if re.search(r"[\w.+-]+@[\w.-]+\.[a-z]{2,}", clean, re.I):
        return True
    if re.search(r"\d{7,}", clean) and not re.search(
        r"\b(?:trailer|trailers|ft|feet|foot|wide|width|length|hitch|gate|gates|wheels|tires|ramp|door|doors)\b",
        clean,
        re.I,
    ):
        return True
    if re.search(r"\b(?:name|phone|email|number|contact)\b", clean, re.I) and re.search(r"\d{7,}", clean):
        return True
    if re.fullmatch(r"[\d\s()+.-]{7,}", clean):
        return True
    if re.fullmatch(r"\b(?:yes|no|nah|nope|ok|okay|thanks|thank you)\b", clean, re.I):
        return True
    return False


def _freeform_request_addition_from_text(message: str) -> str:
    text = re.sub(r"\s+", " ", str(message or "").strip())
    if not text or not re.search(r"[a-z]", text, re.I):
        return ""
    if _looks_like_contact_or_admin_text(text):
        return ""
    if re.search(r"\b(?:instead|actually|make\s+it)\b", text, re.I):
        return ""

    cleaned = text.lower()
    cleaned = re.sub(r"\b\d+(?:\.\d+)?\s*(?:ft|feet|foot|in|inch|inches|lbs?|pounds?|k)\b", " ", cleaned)
    cleaned = re.sub(r"\b\d+(?:\.\d+)?\s*[xX]\s*\d+(?:\.\d+)?\b", " ", cleaned)
    cleaned = re.sub(r"\$\s*\d[\d,]*(?:\.\d+)?", " ", cleaned)
    cleaned = re.sub(r"\b(?:length|wide|width|tall|height|payload|capacity|gvwr|max|maximum|price|budget)\b", " ", cleaned)
    cleaned = re.sub(r"\b(?:gooseneck|bumper\s+pull|black|white|gray|grey|silver|red|blue|green|yellow|orange|tan)\b", " ", cleaned)
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned).strip()
    tokens = [
        token
        for token in cleaned.split()
        if token not in _ACTIVE_REQUEST_FILLER_WORDS and len(token) > 1
    ]
    if not tokens:
        return ""
    return " ".join(tokens)


def _freeform_request_additions_from_state(
    *,
    state: ChatbotState,
    latest_message: str,
    base: str,
) -> list[str]:
    seen: set[str] = {_normalized_requirement_text(base)}
    additions: list[str] = []
    candidates = [
        str(item.get("content") or "")
        for item in (state.get("messages") or [])
        if str(item.get("role") or "").lower() == "user"
    ]
    if latest_message:
        candidates.append(str(latest_message))

    for message in candidates:
        addition = _freeform_request_addition_from_text(message)
        key = _normalized_requirement_text(addition)
        if not key:
            continue
        if key in seen or key in _normalized_requirement_text(base):
            continue
        if any(key in _normalized_requirement_text(existing) or _normalized_requirement_text(existing) in key for existing in additions):
            continue
        seen.add(key)
        additions.append(addition)
    return additions


def _updated_active_search_request_text(
    *,
    state: ChatbotState,
    latest_message: str,
    slots: dict[str, Any],
    metadata_filters: dict[str, Any],
    reset_active_request: bool = False,
) -> str:
    previous = _active_request_base(state.get("active_search_request_text") or "")
    latest = _active_request_base(latest_message)
    base = latest if reset_active_request or not previous else previous
    if not base:
        return ""
    if previous and not reset_active_request:
        additions = _freeform_request_additions_from_state(
            state=state,
            latest_message=latest_message,
            base=base,
        )
        for additional in additions:
            base = f"{base}; {additional}"
    parts = _active_requirement_parts(slots=slots, metadata_filters=metadata_filters)
    if parts:
        return f"{base}{_ACTIVE_REQUEST_SUFFIX_MARKER}{'; '.join(parts)}"
    return base


def _pinecone_listing_facts(listings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "position": idx,
            "title": str(item.get("title") or "").strip(),
            "url": item.get("url"),
            "category": item.get("category"),
            "subcategory": item.get("subcategory"),
            "make": item.get("make"),
            "model": item.get("model"),
            "price": item.get("price_display") or item.get("price"),
            "length": item.get("length"),
            "width": item.get("width"),
            "hitch_type": item.get("hitch_type"),
            "color": item.get("color"),
            "gvwr": item.get("gvwr"),
            "payload_capacity": item.get("payload_capacity"),
            "relevance_score": item.get("relevance_score"),
            "match_evidence_text": str(item.get("match_evidence_text") or "")[:1800],
        }
        for idx, item in enumerate(listings[:6], 1)
    ]


def _extract_requested_non_metadata_features(
    *,
    user_message: str,
    latest_user_message: str | None,
    category: str | None,
    slots: dict[str, Any],
    metadata_filters: dict[str, Any],
) -> list[str]:
    if not os.getenv("OPENAI_API_KEY"):
        return []
    try:
        decision = _requested_feature_extractor_llm().invoke(
            [
                SystemMessage(
                    content="""You identify only user-requested trailer features that are not already represented by structured search filters.

Return structured data only.

Use only the request context provided. Do not infer requested features from inventory listings, titles, specs, examples, or search results.

Requested non-metadata features are user-asked equipment, configuration, or build details that are not captured by normal structured filters. Examples of structured filters to exclude are trailer type/category, make, subcategory, hitch type, color, trailer length, trailer width, height, payload, GVWR, and budget/price.

Do not rewrite structured constraints as non-metadata features. Do not add likely, implied, or common-for-category features. If the user did not explicitly ask for a non-metadata feature, return an empty list.
"""
                ),
                HumanMessage(
                    content=_safe_json(
                        {
                            "active_search_request_text": user_message,
                            "latest_user_message": latest_user_message or user_message,
                            "category": category,
                            "slots": slots,
                            "metadata_filters": metadata_filters,
                        }
                    )
                ),
            ]
        )
    except Exception:
        logger.exception("requested_feature_extractor_llm_failed")
        return []

    data = _model_dump(decision)
    clean = _normalize_requested_feature_list(data.get("requested_non_metadata_features") or [])
    logger.info(
        "requested_feature_extraction | category=%r | features=%s | latest_user_message=%r",
        category,
        _safe_json(clean),
        latest_user_message or user_message,
    )
    return clean


def _pinecone_match_audit(
    *,
    user_message: str,
    latest_user_message: str | None,
    category: str | None,
    slots: dict[str, Any],
    metadata_filters: dict[str, Any],
    search_result: PineconeListingSearchResult,
    facts: list[dict[str, Any]],
    requested_non_metadata_features: list[str],
) -> dict[str, Any]:
    listings = search_result.listings or []
    decision = _pinecone_match_audit_llm().invoke(
        [
            SystemMessage(
                content="""You are an internal trailer match auditor. Return structured audit data only; do not write customer-facing intro copy.

Classify each supplied Pinecone result as full, partial, or alternative against the user's complete active request.

Core rule:
For overall fit, focus on category/use case, make/model/hitch/color when requested, and the supplied requested non-metadata features.

Length, width, payload capacity, and GVWR alone do not make a full match. Do not use length, width, payload capacity, or GVWR to make a listing full, partial, or alternative. Those fields are handled elsewhere by deterministic metadata filters and reranking. You may include them in confirmed_requirements only as factual context, but they must not rescue a missing feature fit.

Strict feature validation:
Match requested non-metadata features by exact feature concept, not by broad category words.

Use language understanding for natural wording variants, but be strict about feature concepts. Slide gate, sliding gate, slider gate, and swing-slide gate may be the same feature if evidence confirms that concept. Butterfly gate is not the same requested feature as sliding gate. Ramp gate, rear gate, side gate, escape door, divider gate, and butterfly gate do not satisfy sliding gate unless evidence explicitly says it is also sliding/slide/swing-slide. Offroad wheels, off-road tires, and all-terrain tires may satisfy an offroad wheel request only when evidence supports that concept.

Evidence rules:
Use only supplied listing facts and match_evidence_text. Do not invent features, prices, specs, availability, or reasons. Missing, unclear, implied, common-for-category, or merely related evidence is unconfirmed.

Important:
Treat the supplied requested_non_metadata_features as authoritative. Do not add, infer, broaden, or substitute any new requested features from listing text or general trailer knowledge. If the supplied list is empty, return an empty requested_non_metadata_features list.

Return:
* requested_non_metadata_features: user-requested features not represented by normal structured filters.
* per_listing_match for every supplied listing position.
* match_level: full only if all requested non-metadata features are explicitly confirmed and the listing fits the main trailer type/use case. Use partial when some important requested features are confirmed but others are missing/unconfirmed. Use alternative when the listing is relevant but the requested feature combination is not confirmed.
* confirmed_requirements: only confirmed facts/features.
* missing_or_unconfirmed_requirements: requested features absent, ambiguous, merely similar, unsupported, or replaced by a different feature type. missing_or_unconfirmed_requirements must include requested features that are not confirmed.
* sales_blurb: one positive customer-facing sentence, max 28 words, highlighting confirmed strengths only. Do not say partial match, alternative, mismatch, missing, requirement, exceeds, does not meet, or fully/exactly/perfectly matches unless match_level is full.

Set intro_text empty. Do not write the intro.
"""
            ),
            HumanMessage(
                content=_safe_json(
                    {
                        "user_message": user_message,
                        "active_search_request_text": user_message,
                        "latest_user_message": latest_user_message or user_message,
                        "category": category,
                        "slots": slots,
                        "metadata_filters": metadata_filters,
                        "requested_non_metadata_features": requested_non_metadata_features,
                        "pinecone_embedding_query_text": search_result.query_text,
                        "pinecone_metadata_filter": search_result.metadata_filter,
                        "rerank_debug": search_result.rerank_debug,
                        "make_debug": search_result.make_debug,
                        "listings": facts,
                    }
                )
            ),
        ]
    )
    data = _model_dump(decision)
    data = _sanitize_requested_feature_analysis(
        data=data,
        facts=facts,
        requested_features=requested_non_metadata_features,
        category=category,
        metadata_filters=metadata_filters,
    )
    return data


def _pinecone_sales_intro_text(
    *,
    user_message: str,
    latest_user_message: str | None,
    category: str | None,
    slots: dict[str, Any],
    metadata_filters: dict[str, Any],
    search_result: PineconeListingSearchResult,
    facts: list[dict[str, Any]],
    match_analysis: dict[str, Any],
) -> str:
    decision = _pinecone_sales_intro_llm().invoke(
        [
            SystemMessage(
                content="""You write only the short customer-facing intro before trailer recommendation results.

Use the supplied audit counts and per-listing verdicts as truth. Do not re-decide match quality.

Tone:
Professional trailer salesperson: positive, helpful, commercially smart, and honest.

Rules:
Use no markdown, bullets, labels, or quotes. Max 55 words. Do not ask for contact details, phone number, email, callback, or sales follow-up.

If full_match_count is 0, say the exact requested combination is not clearly shown/currently shown, then positively position the displayed trailers as strongest available options, practical choices, or useful comparison options. Do not call them strong matches, close matches, best matches, top matches, closest matches, best-fitting options, or exact matches.

If full_match_count is greater than 0, say confirmed fits are shown first, followed by relevant options worth comparing.

Do not mention length, width, payload capacity, or GVWR mismatches. Do not list specific mismatches. Do not say all results meet the user's needs, request, specifications, or criteria unless every displayed listing is full.
"""
            ),
            HumanMessage(
                content=_safe_json(
                    {
                        "user_message": user_message,
                        "active_search_request_text": user_message,
                        "latest_user_message": latest_user_message or user_message,
                        "category": category,
                        "slots": slots,
                        "metadata_filters": metadata_filters,
                        "pinecone_embedding_query_text": search_result.query_text,
                        "audit": match_analysis,
                        "listings": [
                            {
                                **{k: v for k, v in fact.items() if k != "match_evidence_text"},
                                "match_validation": next(
                                    (
                                        item
                                        for item in match_analysis.get("per_listing_match", [])
                                        if int(item.get("position") or 0) == int(fact.get("position") or 0)
                                    ),
                                    {},
                                ),
                            }
                            for fact in facts
                        ],
                    }
                )
            ),
        ]
    )
    return str((_model_dump(decision) if isinstance(decision, BaseModel) else {}).get("intro_text") or "").strip()


def _pinecone_match_framing_text(
    *,
    user_message: str,
    latest_user_message: str | None = None,
    category: str | None,
    slots: dict[str, Any],
    metadata_filters: dict[str, Any],
    search_result: PineconeListingSearchResult,
) -> tuple[str, dict[str, Any]]:
    listings = search_result.listings or []
    if not listings:
        return "", {}
    if not _pinecone_match_framing_llm_enabled() or not os.getenv("OPENAI_API_KEY"):
        data = _fallback_match_analysis(
            source="neutral_fallback",
            listing_count=len(listings),
        )
        _log_pinecone_intro_fallback(
            source="neutral_fallback",
            reason="match_framing_llm_disabled_or_missing_openai_api_key",
            match_analysis=data,
        )
        return _PINECONE_MATCH_FRAMING_FALLBACK, data

    facts = _pinecone_listing_facts(listings)
    try:
        requested_non_metadata_features = _extract_requested_non_metadata_features(
            user_message=user_message,
            latest_user_message=latest_user_message,
            category=category,
            slots=slots,
            metadata_filters=metadata_filters,
        )
    except Exception:
        logger.exception("requested_feature_extraction_failed")
        requested_non_metadata_features = []
    try:
        data = _pinecone_match_audit(
            user_message=user_message,
            latest_user_message=latest_user_message,
            category=category,
            slots=slots,
            metadata_filters=metadata_filters,
            search_result=search_result,
            facts=facts,
            requested_non_metadata_features=requested_non_metadata_features,
        )
        _log_pinecone_match_audit(
            match_analysis={**data, "source": "audit"},
            facts=facts,
            query_text=search_result.query_text,
        )
    except Exception:
        logger.exception("pinecone_match_audit_llm_failed")
        data = _fallback_match_analysis(
            source="neutral_fallback_audit_exception",
            listing_count=len(listings),
        )
        _log_pinecone_intro_fallback(
            source="neutral_fallback_audit_exception",
            reason="match_audit_llm_exception",
            match_analysis=data,
        )
        return _PINECONE_MATCH_FRAMING_FALLBACK, data

    try:
        text = _pinecone_sales_intro_text(
            user_message=user_message,
            latest_user_message=latest_user_message,
            category=category,
            slots=slots,
            metadata_filters=metadata_filters,
            search_result=search_result,
            facts=facts,
            match_analysis=data,
        )
    except Exception:
        logger.exception("pinecone_sales_intro_llm_failed")
        _log_pinecone_intro_fallback(
            source="neutral_fallback_intro_exception",
            reason="sales_intro_llm_exception",
            match_analysis=data,
        )
        return _PINECONE_MATCH_FRAMING_FALLBACK, {
            **data,
            "intro_text": _PINECONE_MATCH_FRAMING_FALLBACK,
            "source": "neutral_fallback_intro_exception",
        }

    text = re.sub(r"\s+", " ", str(text or "").strip())
    if text.startswith('"') and text.endswith('"') and len(text) > 1:
        text = text[1:-1].strip()
    if not text:
        _log_pinecone_intro_fallback(
            source="neutral_fallback_empty_intro_text",
            reason="sales_intro_empty_text",
            match_analysis=data,
        )
        return _PINECONE_MATCH_FRAMING_FALLBACK, {
            **data,
            "intro_text": _PINECONE_MATCH_FRAMING_FALLBACK,
            "source": "neutral_fallback_empty_intro_text",
        }
    if re.search(
        r"\b(?:phone|email|contact|call|reach|callback|follow\s*up|sales\s+team)\b",
        text,
        re.I,
    ):
        _log_pinecone_intro_fallback(
            source="neutral_fallback_blocked_contact_text",
            reason="sales_intro_contact_or_followup_text_blocked",
            intro_candidate=text,
            match_analysis=data,
        )
        return _PINECONE_MATCH_FRAMING_FALLBACK, {
            **data,
            "intro_text": _PINECONE_MATCH_FRAMING_FALLBACK,
            "source": "neutral_fallback_blocked_contact_text",
        }
    if _invalid_pinecone_intro(text=text, data=data, listing_count=len(listings)):
        _log_pinecone_intro_fallback(
            source="neutral_fallback_invalid_match_claim",
            reason="sales_intro_invalid_or_unsupported_match_claim",
            intro_candidate=text,
            match_analysis=data,
        )
        return _PINECONE_MATCH_FRAMING_FALLBACK, {
            **data,
            "intro_text": _PINECONE_MATCH_FRAMING_FALLBACK,
            "source": "neutral_fallback_invalid_match_claim",
        }
    if len(text) > 420:
        text = text[:420].rsplit(" ", 1)[0].strip()
    if not text.endswith((".", "!", "?")):
        text += "."
    logger.info(
        "pinecone_match_intro_applied | source=%r | overall_match_level=%r | full_match_count=%s | "
        "partial_match_count=%s | alternative_count=%s | intro_text=%r",
        "llm",
        data.get("overall_match_level"),
        data.get("full_match_count"),
        data.get("partial_match_count"),
        data.get("alternative_count"),
        text,
    )
    return text, {**data, "intro_text": text, "source": "llm"}


def _result_interest_followup_text(
    *,
    user_message: str,
    category: str | None,
    slots: dict[str, Any],
    listings: list[dict[str, Any]],
) -> str:
    if not listings or not _result_interest_followup_llm_enabled() or not os.getenv("OPENAI_API_KEY"):
        return ""

    all_full = bool(listings) and all(
        (
            item.get("match_validation")
            if isinstance(item.get("match_validation"), dict)
            else {}
        ).get("match_level")
        == "full"
        for item in listings
    )
    facts = [
        {
            "position": idx,
            "title": str(item.get("title") or "").strip(),
            "category": item.get("category"),
            "length": item.get("length"),
            "width": item.get("width"),
            "price": item.get("price_display") or item.get("price"),
            "match_level": (
                item.get("match_validation")
                if isinstance(item.get("match_validation"), dict)
                else {}
            ).get("match_level"),
        }
        for idx, item in enumerate(listings[:5], 1)
    ]

    try:
        response = _result_interest_followup_llm().invoke(
            [
                SystemMessage(
                    content=(
                        "Write exactly one short customer-facing follow-up sentence after trailer search results. "
                        "Ask in a positive sales-advisor tone whether any shown trailer stands out or should be compared further. "
                        "Do not ask for phone number, email, contact details, store visit, callback, or team follow-up. "
                        "Do not invent listing facts. Use only the supplied context. "
                        "Do not say the options meet the customer's needs/specifications unless all_listings_full_match is true. "
                        "No markdown, no bullets, no quotes, max 22 words."
                    )
                ),
                HumanMessage(
                    content=_safe_json(
                        {
                            "user_message": user_message,
                            "category": category,
                            "slots": slots,
                            "all_listings_full_match": all_full,
                            "listings": facts,
                        }
                    )
                ),
            ]
        )
        text = re.sub(r"\s+", " ", str(response.content or "").strip())
        if not text:
            return ""
        if re.search(
            r"\b(?:phone|email|contact|call|reach|follow\s*up|team|sales|website|visit)\b",
            text,
            re.I,
        ):
            return ""
        if not all_full and re.search(r"\bmeet(?:s)?\s+(?:your\s+)?(?:needs|specifications|requirements)\b", text, re.I):
            return ""
        if text.startswith('"') and text.endswith('"') and len(text) > 1:
            text = text[1:-1].strip()
        if not text.endswith(("?", ".", "!")):
            text += "?"
        return text
    except Exception:
        logger.exception("result_interest_followup_llm_failed")
        return ""


def _has_contact(state: ChatbotState) -> bool:
    return bool(
        state.get("customer_full_name")
        and (state.get("customer_phone") or state.get("customer_email"))
    )


def _optional_contact_request(reason: str) -> str:
    clean_reason = str(reason or "this request").strip()
    return (
        "Could you please share your phone number or email address so our team can "
        f"contact you regarding {clean_reason}?"
    )


_METADATA_FILTER_KEYS = {
    "length_ft",
    "width_ft",
    "payload_lbs",
    "max_price",
    "hitch_type",
    "subcategory",
    "make",
    "color",
}
_ALLOWED_HITCH_TYPES = {"Gooseneck", "Bumper Pull"}
_CONFIDENT_CLASSIFICATIONS = {"medium", "high"}
_CONFIDENT_PREFERENCE_NULL = {"medium", "high"}
_DYNAMIC_WIDTH_SLOT = "item_or_trailer_width_ft"
_DYNAMIC_WIDTH_QUESTION = "About how wide is the load, or what trailer width do you need?"
_DYNAMIC_WIDTH_EXCLUDED_CATEGORIES = {"utility", "enclosed", "livestock", "aluminum", "flatbed"}
_FLATBED_DEFAULT_WIDTH_FT = "8 ft"
_GENERIC_CATEGORY_CHOICE_SLOT = "generic_category_choice"
_GENERIC_CATEGORY_QUESTION = "What type of trailer are you looking for?"
_MAKE_CATEGORY_CHOICE_SLOT = "make_category_choice"
_MAKE_GENERIC_LENGTH_SLOT = "trailer_length_ft"
_MAKE_GENERIC_PAYLOAD_SLOT = "haul_weight_lbs"
_MAKE_GENERIC_QUESTIONS = {
    _MAKE_GENERIC_LENGTH_SLOT: "What trailer length would you prefer?",
    _MAKE_GENERIC_PAYLOAD_SLOT: "What payload or weight capacity do you need?",
}
_SLOT_METADATA_FILTER_MAP = {
    "base_category": ("subcategory",),
    "bin_size": ("length_ft",),
    "cargo_size": ("length_ft", "width_ft"),
    "haul_length_ft": ("length_ft",),
    "haul_weight_lbs": ("payload_lbs",),
    "item_or_trailer_width_ft": ("width_ft",),
    "max_price": ("max_price",),
    "payload_need": ("payload_lbs",),
    "total_weight": ("payload_lbs",),
    "trailer_length_ft": ("length_ft",),
    "trailer_size": ("length_ft", "width_ft"),
    "trailer_width_ft": ("width_ft",),
    "vehicle_length_ft": ("length_ft",),
    "width_ft": ("width_ft",),
}


def _has_width_requirement(slots: dict[str, Any], metadata_filters: dict[str, Any]) -> bool:
    if metadata_filters.get("width_ft"):
        return True
    width_slots = (
        _DYNAMIC_WIDTH_SLOT,
        "trailer_width_ft",
        "width_ft",
        "trailer_size",
        "cargo_size",
    )
    return any(bool(slots.get(slot)) for slot in width_slots)


def _apply_flatbed_default_width(category: str | None, slots: dict[str, Any], metadata_filters: dict[str, Any]) -> None:
    if str(category or "").strip().lower() != "flatbed":
        return
    if _has_width_requirement(slots, metadata_filters):
        return
    metadata_filters["width_ft"] = _FLATBED_DEFAULT_WIDTH_FT
    logger.info("flatbed_default_width_applied | width_ft=%r", _FLATBED_DEFAULT_WIDTH_FT)


def _is_usable_classifier_haul_item(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    if re.search(r"\btrailers?\b", text):
        return False
    category_names = {cat.lower() for cat in list_all_categories()}
    return text not in category_names


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


_KNOWN_COLOR_WORDS = {
    "black",
    "white",
    "gray",
    "grey",
    "silver",
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "tan",
}


def _slot_is_length_like(slot: str | None) -> bool:
    slot_l = str(slot or "").lower()
    return (
        "length" in slot_l
        or "width" in slot_l
        or "size" in slot_l
        or "cargo" in slot_l
        or slot_l == _DYNAMIC_WIDTH_SLOT
    )


def _slot_is_weight_like(slot: str | None) -> bool:
    slot_l = str(slot or "").lower()
    return any(token in slot_l for token in ("weight", "payload", "capacity", "gvwr", "total_weight"))


def _has_dimension_shorthand(message: str) -> bool:
    return bool(
        re.search(
            r"\b\d+(?:\.\d+)?\s*(?:ft|feet|foot|')?\s*[xX]\s*"
            r"\d+(?:\.\d+)?\s*(?:ft|feet|foot|')?"
            r"(?:\s*[xX]\s*\d+(?:\.\d+)?\s*(?:ft|feet|foot|')?)?\b",
            message or "",
        )
    )


def _explicit_length_requested(message: str, awaiting_slot: str | None = None) -> bool:
    text = str(message or "").lower()
    if _has_dimension_shorthand(text):
        return True
    if re.search(r"\b(?:length|long|deck\s+length|trailer\s+length|size|trailer\s+size)\b", text):
        return bool(re.search(r"\d+(?:\.\d+)?", text))
    if re.search(r"\b(?:make\s+it|change\s+it\s+to|instead)\D{0,20}\d+(?:\.\d+)?\s*(?:ft|feet|foot|')\b", text):
        return True
    if re.search(r"\b\d+(?:\.\d+)?\s*(?:ft|feet|foot|')\s*(?:(?:\w+\s+){0,3})?(?:long|length|trailer)\b", text):
        return True
    return bool(_slot_is_length_like(awaiting_slot) and re.search(r"\b\d+(?:\.\d+)?\s*(?:ft|feet|foot|')\b", text))


def _explicit_width_requested(message: str, awaiting_slot: str | None = None) -> bool:
    text = str(message or "").lower()
    if _has_dimension_shorthand(text):
        return True
    if re.search(r"\b(?:width|wide)\b", text):
        return bool(re.search(r"\d+(?:\.\d+)?", text))
    return bool(str(awaiting_slot or "") == _DYNAMIC_WIDTH_SLOT and re.search(r"\b\d+(?:\.\d+)?\s*(?:ft|feet|foot|')\b", text))


def _explicit_payload_requested(message: str, awaiting_slot: str | None = None) -> bool:
    text = str(message or "").lower()
    has_weight_unit = bool(re.search(r"\b\d[\d,]*(?:\.\d+)?\s*(?:k|m)?\s*(?:lbs?|pounds?|#)\b", text))
    if has_weight_unit:
        return True
    if re.search(r"\b(?:payload|load|haul|carry|weight|weighs?)\b", text):
        return bool(re.search(r"\d[\d,]*(?:\.\d+)?", text))
    return bool(_slot_is_weight_like(awaiting_slot) and has_weight_unit)


def _explicit_price_requested(message: str) -> bool:
    text = str(message or "").lower()
    if "$" in text:
        return bool(re.search(r"\d[\d,]*(?:\.\d+)?", text))
    return bool(
        re.search(r"\b(?:price|budget|cost|under|below|max|maximum|afford|dollars?)\b", text)
        and re.search(r"\d[\d,]*(?:\.\d+)?", text)
    )


def _explicit_color_requested(message: str) -> bool:
    text = str(message or "").lower()
    return any(re.search(rf"\b{re.escape(color)}\b", text) for color in _KNOWN_COLOR_WORDS)


def _explicit_hitch_requested(message: str) -> bool:
    text = str(message or "").lower()
    return bool(
        re.search(
            r"\bgoose\s*neck\b"
            r"|\bgooseneck\b"
            r"|\bbumper\s*[- ]?\s*pull\b"
            r"|\bbumperpull\b"
            r"|\bhitch\s+type\s+(?:should\s+be|is|to|as|=)\b"
            r"|\bprefer\s+\w+(?:\s+\w+){0,4}\s+hitch\b"
            r"|\bmake\s+it\s+\w+(?:\s+\w+){0,4}\s+hitch\b",
            text,
        )
    )


def _message_has_filter_evidence(key: str, latest_message: str, awaiting_slot: str | None = None) -> bool:
    key = str(key)
    if key == "length_ft":
        return _explicit_length_requested(latest_message, awaiting_slot)
    if key == "width_ft":
        return _explicit_width_requested(latest_message, awaiting_slot)
    if key == "payload_lbs":
        return _explicit_payload_requested(latest_message, awaiting_slot)
    if key == "max_price":
        return _explicit_price_requested(latest_message)
    if key == "hitch_type":
        return _explicit_hitch_requested(latest_message)
    if key == "subcategory":
        return _explicit_subcategory_requested(latest_message)
    if key == "make":
        return _explicit_make_requested(latest_message)
    if key == "color":
        return _explicit_color_requested(latest_message)
    return False


def _value_text_appears_in_message(value: Any, latest_message: str) -> bool:
    value_text = str(value or "").strip().lower()
    message = str(latest_message or "").lower()
    if not value_text or not message:
        return False
    compact_value = re.sub(r"\s+", " ", value_text)
    return compact_value in re.sub(r"\s+", " ", message)


def _message_has_slot_evidence(slot: str, value: Any, latest_message: str, awaiting_slot: str | None = None) -> bool:
    slot = str(slot)
    if awaiting_slot and slot == awaiting_slot:
        return bool(str(latest_message or "").strip())
    if _slot_is_weight_like(slot):
        return _explicit_payload_requested(latest_message, awaiting_slot)
    if _slot_is_length_like(slot):
        return _explicit_length_requested(latest_message, awaiting_slot) or _explicit_width_requested(latest_message, awaiting_slot)
    if slot == "hitch_type":
        return _explicit_hitch_requested(latest_message)
    if slot == "color":
        return _explicit_color_requested(latest_message)
    if slot == "max_price":
        return _explicit_price_requested(latest_message)
    return _value_text_appears_in_message(value, latest_message)


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


def _explicit_make_requested(message: str) -> bool:
    return bool(resolve_make_from_text(message or "", use_llm_fallback=False).make)


def _should_apply_inferred_make(
    latest_message: str,
    *,
    category: str | None,
    existing_make: str | None,
) -> bool:
    if _message_has_filter_evidence("make", latest_message):
        return True
    if existing_make:
        return False
    return not bool(category)


def _sanitize_metadata_filter_update(
    key: str,
    value: Any,
    latest_message: str,
    awaiting_slot: str | None = None,
) -> tuple[str, Any] | None:
    if value in (None, "") or key not in _METADATA_FILTER_KEYS:
        return None
    if not _message_has_filter_evidence(key, latest_message, awaiting_slot):
        logger.info(
            "metadata_filter_rejected | key=%s | value=%r | reason=not_explicit",
            key,
            value,
        )
        return None
    if key == "hitch_type":
        hitch = _normalize_allowed_hitch(value)
        if not hitch:
            logger.info("metadata_filter_rejected | key=hitch_type | value=%r", value)
            return None
        return key, hitch
    if key == "subcategory":
        subcategory = normalize_subcategory(str(value))
        if not subcategory:
            logger.info("metadata_filter_rejected | key=subcategory | value=%r | reason=invalid", value)
            return None
        return key, subcategory
    if key == "make":
        resolution = resolve_make_from_text(str(value), use_llm_fallback=False)
        if not resolution.make:
            resolution = resolve_make_from_text(latest_message or "", use_llm_fallback=False)
        if not resolution.make:
            logger.info("metadata_filter_rejected | key=make | value=%r | reason=unknown", value)
            return None
        return key, resolution.make
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

    make_resolution = resolve_make_from_text(text, use_llm_fallback=False)
    if make_resolution.make:
        updates["make"] = make_resolution.make

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
                        "- Return null for fields not explicitly mentioned in the latest user message; never infer updates from recent messages.\n"
                        "- Width phrases such as 'width should be at least 6 ft' or '6 ft wide' map only to width_ft, never length_ft.\n"
                        "- Length phrases must mention length, long, deck length, trailer length, size, trailer size, or an ambiguous 'make it 14 ft' update.\n"
                        "- Trailer shorthand like '6x12' means width_ft=6 and length_ft=12; '6x12x5' means width_ft=6, length_ft=12, height=5. Do not store height unless there is an allowed slot/filter for it.\n"
                        "- Payload/load/haul weight maps to payload_lbs, not GVWR.\n"
                        "- hitch_type can only be gooseneck or bumper pull and must be explicitly named in the latest message; return null otherwise.\n"
                        "- make is the trailer manufacturer only; never use Gooseneck as make unless the user explicitly says it is the brand/manufacturer.\n"
                        "- subcategory must only be returned when the latest message explicitly asks for a subcategory filter.\n"
                        "- Do not infer subcategory from category words like Tilt, Utility, Dump, Aluminum, or Enclosed.\n"
                        "- Preserve existing values by returning null unless the latest message updates that exact field.\n"
                        "- slot_updates may include only allowed category slots and only when the latest message provides direct evidence for that slot."
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
    awaiting_slot: str | None = None,
) -> dict[str, Any]:
    data = _model_dump(extraction)
    updates: dict[str, Any] = {}
    for key in _METADATA_FILTER_KEYS:
        sanitized = _sanitize_metadata_filter_update(key, data.get(key), latest_message, awaiting_slot)
        if sanitized:
            clean_key, clean_value = sanitized
            updates[clean_key] = clean_value
    return updates


def _metadata_filters_from_decision(
    decision: dict[str, Any],
    latest_message: str = "",
    awaiting_slot: str | None = None,
) -> dict[str, Any]:
    raw = decision.get("metadata_filters_update") or {}
    updates: dict[str, Any] = {}
    for key, value in raw.items():
        sanitized = _sanitize_metadata_filter_update(str(key), value, latest_message, awaiting_slot)
        if sanitized:
            clean_key, clean_value = sanitized
            updates[clean_key] = clean_value
    return updates


def _slot_updates_from_extraction(
    extraction: FilterExtractionDecision,
    allowed_category_slots: set[str],
    latest_message: str = "",
    awaiting_slot: str | None = None,
) -> dict[str, Any]:
    raw = dict(extraction.slot_updates or {})
    updates: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in allowed_category_slots or value in (None, ""):
            continue
        if not _message_has_slot_evidence(str(key), value, latest_message, awaiting_slot):
            logger.info("slot_update_rejected | slot=%s | value=%r | reason=not_explicit", key, value)
            continue
        updates[key] = value
    return updates


def _slot_updates_from_decision(
    decision: dict[str, Any],
    allowed_category_slots: set[str],
    latest_message: str = "",
    awaiting_slot: str | None = None,
) -> dict[str, Any]:
    raw = dict(decision.get("slots_collected_update") or {})
    updates: dict[str, Any] = {}
    for key, value in raw.items():
        if value in (None, ""):
            continue
        if allowed_category_slots and key not in allowed_category_slots:
            continue
        if not _message_has_slot_evidence(str(key), value, latest_message, awaiting_slot):
            logger.info("slot_update_rejected | slot=%s | value=%r | reason=not_explicit", key, value)
            continue
        updates[key] = value
    return updates


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


def _aluminum_base_category_subcategory(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    lower = text.lower()

    cleaned = re.sub(r"\b(?:aluminum|trailers?)\b", " ", lower, flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return None

    category = normalize_category(cleaned)
    allowed = {cat for cat in list_all_categories() if cat != "Aluminum"}
    if category in allowed:
        return category
    return None


def _apply_aluminum_base_category_filter(
    category: str | None,
    slots: dict[str, Any],
    metadata_filters: dict[str, Any],
) -> None:
    if (category or "").strip().lower() != "aluminum" or "base_category" not in slots:
        return

    subcategory = _aluminum_base_category_subcategory(slots.get("base_category"))
    if subcategory:
        before = metadata_filters.get("subcategory")
        metadata_filters["subcategory"] = subcategory
        if before != subcategory:
            logger.info(
                "aluminum_base_category_subcategory_applied | base_category=%r | subcategory=%r",
                slots.get("base_category"),
                subcategory,
            )
    elif metadata_filters.pop("subcategory", None) is not None:
        logger.info(
            "aluminum_base_category_subcategory_cleared | base_category=%r",
            slots.get("base_category"),
        )


def _metadata_filters_for_skipped_slot(slot: str) -> tuple[str, ...]:
    if slot in _SLOT_METADATA_FILTER_MAP:
        return _SLOT_METADATA_FILTER_MAP[slot]
    slot_l = slot.lower()
    if "width" in slot_l:
        return ("width_ft",)
    if "length" in slot_l or slot_l.endswith("_size"):
        return ("length_ft",)
    if "weight" in slot_l or "payload" in slot_l or "capacity" in slot_l:
        return ("payload_lbs",)
    if slot_l in _METADATA_FILTER_KEYS:
        return (slot_l,)
    return ()


def _valid_preference_targets(
    *,
    category: str | None,
    awaiting_slot: str | None,
    pending: list[QuestionItem],
) -> set[str]:
    del category
    targets: set[str] = set()
    if awaiting_slot:
        targets.add(str(awaiting_slot))
    for question in pending or []:
        slot = str(question.get("slot") or "")
        if slot:
            targets.add(slot)
    return targets


def _apply_preference_null_decision(
    *,
    category: str | None,
    awaiting_slot: str | None,
    pending: list[QuestionItem],
    metadata_filters: dict[str, Any],
    slots_skipped: set[str],
    decision: PreferenceNullDecision,
) -> tuple[str | None, set[str], dict[str, Any]]:
    data = _model_dump(decision)
    if not data.get("has_no_preference") or data.get("confidence") not in _CONFIDENT_PREFERENCE_NULL:
        return awaiting_slot, slots_skipped, {}

    allowed_slots = _valid_preference_targets(
        category=category,
        awaiting_slot=awaiting_slot,
        pending=pending,
    )
    target_slots = {
        str(slot)
        for slot in data.get("target_slots") or []
        if str(slot) in allowed_slots
    }
    if not target_slots and awaiting_slot:
        target_slots.add(str(awaiting_slot))

    target_filters = {
        str(key)
        for key in data.get("target_metadata_filters") or []
        if str(key) in _METADATA_FILTER_KEYS
    }
    for slot in target_slots:
        target_filters.update(_metadata_filters_for_skipped_slot(slot))

    removed_filters: dict[str, Any] = {}
    for key in target_filters:
        if key in metadata_filters:
            removed_filters[key] = metadata_filters.pop(key)

    slots_skipped.update(target_slots)
    if awaiting_slot in slots_skipped:
        awaiting_slot = None

    logger.info(
        "preference_null_applied | category=%r | slots=%s | filters=%s | removed_filters=%s | reason=%r",
        category,
        json.dumps(sorted(target_slots)),
        json.dumps(sorted(target_filters)),
        json.dumps(removed_filters, default=str),
        data.get("reason"),
    )
    return awaiting_slot, slots_skipped, removed_filters


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

    matched_item_is_usable = _is_usable_classifier_haul_item(classification.matched_item)
    if confident and matched_item_is_usable and classification.matched_item and not slots.get("haul_item"):
        if "haul_item" in _category_slots(category):
            slots["haul_item"] = classification.matched_item
            logger.info(
                "classification_haul_item_filled | category=%r | matched_item=%r | reason=%r",
                category,
                classification.matched_item,
                classification.reason,
            )

    known_haul_item = bool(slots.get("haul_item") or matched_item_is_usable)
    if cat == "utility" and confident and classification.is_lightweight_utility_load and known_haul_item:
        required_slots = [slot for slot in required_slots if slot != "haul_weight_lbs"]
        if not metadata_filters.get("payload_lbs") and not slots.get("haul_weight_lbs"):
            metadata_filters["payload_lbs"] = "1500 lbs"
            logger.info(
                "utility_lightweight_payload_default_applied | matched_item=%r | reason=%r",
                classification.matched_item,
                classification.reason,
            )

    if (
        cat not in _DYNAMIC_WIDTH_EXCLUDED_CATEGORIES
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
    slots_skipped: set[str] | None = None,
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
    skipped = set(slots_skipped or set())

    def add_slot(slot: str, required: bool) -> None:
        if not slot or slot in slots or slot in skipped or slot in queued_slots:
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
    slots_skipped: set[str] | None = None,
    required_slots: list[str] | None = None,
    questions_override: dict[str, str] | None = None,
) -> tuple[list[str], list[str], dict[str, str]]:
    spec = get_trailer_fields_as_dict(category)
    required_slots = list(required_slots if required_slots is not None else (spec.get("required_slots") or []))
    questions = dict(spec.get("questions") or {})
    questions.update(questions_override or {})
    missing: list[str] = []
    invalid: list[str] = []
    skipped = set(slots_skipped or set())
    for slot in required_slots:
        if slot in skipped:
            continue
        if slot not in slots:
            missing.append(slot)
            continue
        ok, _reason = _validate_slot_value(slot, slots.get(slot))
        if not ok:
            invalid.append(slot)
    return missing, invalid, questions


def _last_assistant_text(messages: list[dict[str, Any]] | None) -> str:
    for message in reversed(messages or []):
        if isinstance(message, dict) and message.get("role") == "assistant":
            return str(message.get("content") or "").strip()
    return ""


def _is_generic_trailer_type_question(text: str) -> bool:
    return bool(
        re.search(
            r"\bwhat\s+(?:kind|type)\s+of\s+trailer\b"
            r"|\bwhat\s+kind\s+of\s+trailer\s+are\s+you\s+looking\s+for\b"
            r"|\bwhich\s+(?:kind|type|category)\s+(?:of\s+trailer\s+)?(?:should|do|would|are)\b",
            text or "",
            re.I,
        )
    )


def _has_obvious_catalogue_intent(text: str) -> bool:
    normalized = _normalize_choice_text(text)
    return bool(
        re.search(
            r"\b(?:show|list|see|view|browse|send|give)\s+(?:me\s+)?(?:all|everything|the\s+whole|the\s+full)\s+"
            r"(?:of\s+)?(?:your\s+|the\s+)?(?:trailers?|products?|inventory|stock)\b"
            r"|\b(?:browse|see|view)\s+(?:your\s+|the\s+)?(?:inventory|stock|catalog(?:ue)?)\b"
            r"|\b(?:full|complete|whole)\s+(?:trailer\s+)?(?:inventory|catalog(?:ue)?|stock|lineup)\b"
            r"|\b(?:catalog|catalogue)\b"
            r"|\b(?:everything|all)\s+(?:you\s+have|you\s+carry|in\s+stock)\b",
            normalized,
            re.I,
        )
    )


def _latest_message_has_inventory_constraint(text: str) -> bool:
    return bool(
        resolve_category_from_text(text).category
        or resolve_make_from_text(text, use_llm_fallback=False).make
        or _explicit_length_requested(text)
        or _explicit_width_requested(text)
        or _explicit_payload_requested(text)
        or _explicit_price_requested(text)
        or _explicit_hitch_requested(text)
        or _explicit_subcategory_requested(text)
        or _explicit_color_requested(text)
    )


def _has_inventory_constraints(
    state: ChatbotState,
    *,
    category: str | None = None,
    latest_message: str = "",
) -> bool:
    return bool(
        category
        or state.get("trailer_category")
        or (state.get("slots_collected") or {})
        or (state.get("metadata_filters_collected") or {})
        or _latest_message_has_inventory_constraint(latest_message)
    )


def _catalogue_redirect_allowed(
    state: ChatbotState,
    *,
    category: str | None,
    latest_message: str,
    no_preference_decision: PreferenceNullDecision | None = None,
) -> bool:
    if _has_inventory_constraints(state, category=category, latest_message=latest_message):
        return False
    if _has_obvious_catalogue_intent(latest_message):
        return True
    if state.get("awaiting_slot") == _GENERIC_CATEGORY_CHOICE_SLOT:
        return False

    previous_assistant = _last_assistant_text(state.get("messages") or [])
    if not _is_generic_trailer_type_question(previous_assistant):
        return False

    if no_preference_decision is not None:
        data = _model_dump(no_preference_decision)
        if data.get("has_no_preference") and data.get("confidence") in _CONFIDENT_PREFERENCE_NULL:
            return True

    return _is_no_category_preference(latest_message)


def _catalogue_redirect_state(
    state: ChatbotState,
    *,
    decision: dict[str, Any],
    category: str | None,
    slots: dict[str, Any],
    slots_skipped: set[str],
    metadata_filters: dict[str, Any],
) -> ChatbotState:
    decision["action"] = "respond"
    logger.info("catalogue_redirect_applied | user_message=%r", state.get("user_message"))
    return {
        **state,
        "trailer_category": category,
        "slots_collected": slots,
        "slots_skipped": sorted(slots_skipped),
        "metadata_filters_collected": metadata_filters,
        "make_category_options": [],
        "awaiting_slot": None,
        "pending_questions": [],
        "assistant_text": _CATALOGUE_REDIRECT_REPLY,
        "mind_decision": decision,
    }


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
            "contact_status": state.get("contact_status"),
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

    if (
        not deterministic_hint
        and not state.get("trailer_category")
        and decision.trailer_category
    ):
        logger.info(
            "llm_category_ignored_without_deterministic_match | proposed_category=%r | user_message=%r",
            decision.trailer_category,
            state.get("user_message"),
        )
        decision.trailer_category = None

    return {**state, "mind_decision": _model_dump(decision)}


def _normalize_choice_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _category_from_choice(text: str, options: list[str]) -> str | None:
    resolution = resolve_category_from_text(text)
    if resolution.category and resolution.category in options:
        return resolution.category

    normalized = _normalize_choice_text(text)
    for option in options:
        option_l = _normalize_choice_text(option)
        if normalized == option_l or option_l in normalized:
            return option
    return None


def _is_no_category_preference(text: str) -> bool:
    return bool(
        re.search(
            r"\b("
            r"any|either|whatever|no\s+preference|no\s+idea|not\s+sure|don't\s+know|dont\s+know|"
            r"do\s+not\s+know|can't\s+choose|cant\s+choose|you\s+pick|recommend"
            r")\b",
            text or "",
            re.I,
        )
        or bool(
            re.search(
                r"\b(?:no|not|none)\s+(?:specific\s+)?(?:type|category|kind)\b"
                r"|\b(?:no|not|none)\s+(?:specific\s+)?(?:type|category|kind)\s+in\s+mind\b"
                r"|\b(?:i\s+have|have)\s+no\s+(?:type|category|kind)\s+in\s+mind\b",
                text or "",
                re.I,
            )
        )
        or bool(
            re.search(
                r"\b(?:type|category|kind)\s+(?:doesn['\u2019]?t|does\s+not)\s+matter\b"
                r"|\b(?:type|category|kind)\s+(?:is\s+)?(?:irrelevant|unimportant)\b"
                r"|\bi\s+don['\u2019]?t\s+care\s+about\s+(?:the\s+)?(?:type|category|kind)\b",
                text or "",
                re.I,
            )
        )
    )


def _classify_make_category_no_preference(
    *,
    user_message: str,
    make_category_options: list[str],
    metadata_filters: dict[str, Any],
    slots: dict[str, Any],
) -> bool:
    category_text = ", ".join(make_category_options)
    active_question = (
        f"Which category should I use: {category_text}?"
        if category_text
        else "Which category should I use?"
    )
    decision = classify_no_preference(
        category=None,
        user_message=user_message,
        awaiting_slot=_MAKE_CATEGORY_CHOICE_SLOT,
        pending_questions=[],
        slots_collected=slots,
        metadata_filters_collected=metadata_filters,
        allowed_category_slots=[],
        active_question=active_question,
    )
    data = _model_dump(decision)
    if data.get("has_no_preference") and data.get("confidence") in _CONFIDENT_PREFERENCE_NULL:
        return True
    return _is_no_category_preference(user_message)


def _classify_generic_category_no_preference(
    *,
    user_message: str,
    metadata_filters: dict[str, Any],
    slots: dict[str, Any],
) -> bool:
    decision = classify_no_preference(
        category=None,
        user_message=user_message,
        awaiting_slot=_GENERIC_CATEGORY_CHOICE_SLOT,
        pending_questions=[],
        slots_collected=slots,
        metadata_filters_collected=metadata_filters,
        allowed_category_slots=[],
        active_question=_GENERIC_CATEGORY_QUESTION,
    )
    data = _model_dump(decision)
    if data.get("has_no_preference") and data.get("confidence") in _CONFIDENT_PREFERENCE_NULL:
        return True
    return _is_no_category_preference(user_message)


def _make_category_question(make: str, categories: tuple[str, ...]) -> str:
    category_text = ", ".join(categories)
    return (
        f"We have {make} trailers in these categories: {category_text}. "
        "Which category should I look at?"
    )


def _make_only_missing_slots(
    *,
    category: str | None,
    metadata_filters: dict[str, Any],
    slots: dict[str, Any],
    slots_skipped: set[str],
) -> list[str]:
    if category or not metadata_filters.get("make"):
        return []

    missing: list[str] = []
    if (
        not metadata_filters.get("length_ft")
        and not slots.get(_MAKE_GENERIC_LENGTH_SLOT)
        and _MAKE_GENERIC_LENGTH_SLOT not in slots_skipped
    ):
        missing.append(_MAKE_GENERIC_LENGTH_SLOT)
    if (
        not metadata_filters.get("payload_lbs")
        and not slots.get(_MAKE_GENERIC_PAYLOAD_SLOT)
        and _MAKE_GENERIC_PAYLOAD_SLOT not in slots_skipped
    ):
        missing.append(_MAKE_GENERIC_PAYLOAD_SLOT)
    return missing


def _generic_category_no_preference_active(slots_skipped: set[str]) -> bool:
    return _GENERIC_CATEGORY_CHOICE_SLOT in slots_skipped


def _generic_no_category_missing_slots(
    *,
    category: str | None,
    metadata_filters: dict[str, Any],
    slots: dict[str, Any],
    slots_skipped: set[str],
) -> list[str]:
    if category or not _generic_category_no_preference_active(slots_skipped):
        return []

    missing: list[str] = []
    if (
        not metadata_filters.get("length_ft")
        and not slots.get(_MAKE_GENERIC_LENGTH_SLOT)
        and _MAKE_GENERIC_LENGTH_SLOT not in slots_skipped
    ):
        missing.append(_MAKE_GENERIC_LENGTH_SLOT)
    if (
        not metadata_filters.get("payload_lbs")
        and not slots.get(_MAKE_GENERIC_PAYLOAD_SLOT)
        and _MAKE_GENERIC_PAYLOAD_SLOT not in slots_skipped
    ):
        missing.append(_MAKE_GENERIC_PAYLOAD_SLOT)
    return missing


def _queue_make_only_questions(
    pending: list[QuestionItem],
    missing_slots: list[str],
) -> list[QuestionItem]:
    existing = {str(q.get("slot") or "") for q in pending}
    for slot in missing_slots:
        if slot not in existing:
            pending.append(
                {
                    "slot": slot,
                    "question": _MAKE_GENERIC_QUESTIONS[slot],
                    "required": True,
                }
            )
    return pending


def _queue_generic_no_category_questions(
    pending: list[QuestionItem],
    missing_slots: list[str],
) -> list[QuestionItem]:
    return _queue_make_only_questions(pending, missing_slots)


def _has_generic_trailer_request(text: str) -> bool:
    normalized = _normalize_choice_text(text)
    if not normalized:
        return False
    if _has_obvious_catalogue_intent(normalized):
        return False
    if resolve_category_from_text(normalized).category:
        return False
    return bool(
        re.search(r"\btrailers?\b", normalized)
        or _explicit_length_requested(normalized)
        or _explicit_width_requested(normalized)
        or _explicit_payload_requested(normalized)
        or _explicit_price_requested(normalized)
        or _explicit_hitch_requested(normalized)
        or _explicit_color_requested(normalized)
    )


def _apply_make_resolution(
    *,
    latest_message: str,
    category: str | None,
    metadata_filters: dict[str, Any],
) -> tuple[str | None, list[str], str | None]:
    explicit_category = resolve_category_from_text(latest_message).category
    if explicit_category and not category:
        category = explicit_category

    resolution = resolve_make_from_text(
        latest_message,
        use_llm_fallback=True,
    )
    if not resolution.make:
        return category, [], None

    existing_make = str(metadata_filters.get("make") or "").strip() or None
    if not _should_apply_inferred_make(
        latest_message,
        category=category,
        existing_make=existing_make,
    ):
        logger.info(
            "make_resolution_suppressed | resolved_make=%r | existing_make=%r | category=%r | reason=%s",
            resolution.make,
            existing_make,
            category,
            "existing_make_retained" if existing_make else "not_explicit_with_category_context",
        )
        return category, [], None

    metadata_filters["make"] = resolution.make
    logger.info(
        "make_resolution_applied | make=%r | category=%r | explicit_make=%s | match_type=%r",
        resolution.make,
        category,
        _message_has_filter_evidence("make", latest_message),
        resolution.match_type,
    )
    available_categories = list(categories_for_make(resolution.make))
    if category:
        return category, [], None
    if len(available_categories) == 1:
        return available_categories[0], [], None
    if len(available_categories) > 1:
        return category, available_categories, _make_category_question(
            resolution.make,
            tuple(available_categories),
        )
    return category, [], None


def _apply_explicit_filter_extraction(
    *,
    state: ChatbotState,
    category: str | None,
    slots: dict[str, Any],
    metadata_filters: dict[str, Any],
    latest_message: str,
    awaiting_slot: str | None,
    apply_slot_updates: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    allowed_category_slots = _category_slots(category)
    extraction = _extract_filter_decision(
        {
            **state,
            "trailer_category": category,
            "slots_collected": slots,
            "metadata_filters_collected": metadata_filters,
        },
        category,
    )
    extracted_metadata = _metadata_filters_from_extraction(extraction, latest_message, awaiting_slot)
    extracted_slots = (
        _slot_updates_from_extraction(
            extraction,
            allowed_category_slots,
            latest_message,
            awaiting_slot,
        )
        if apply_slot_updates and category
        else {}
    )
    logger.info(
        "filter_extraction_applied | category=%r | extracted_metadata=%s | extracted_slots=%s",
        category,
        json.dumps(extracted_metadata, default=str),
        json.dumps(extracted_slots, default=str),
    )
    for key, value in extracted_metadata.items():
        if key in _METADATA_FILTER_KEYS and value not in (None, ""):
            metadata_filters[key] = value
    return extracted_slots, extracted_metadata


def _apply_mind_node(state: ChatbotState) -> ChatbotState:
    decision = dict(state.get("mind_decision") or {})
    slots_before = dict(state.get("slots_collected") or {})
    slots = dict(slots_before)
    slots_skipped_before = set(state.get("slots_skipped") or [])
    slots_skipped = set(slots_skipped_before)
    metadata_filters_before = dict(state.get("metadata_filters_collected") or {})
    metadata_filters = dict(metadata_filters_before)
    invalid_required_slot: str | None = None
    awaiting_slot = state.get("awaiting_slot")
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
    make_changed = False
    reset_result_state = False
    if category_changed:
        logger.info(
            "category_change_clears_filters | old_category=%r | new_category=%r",
            category_before,
            category,
        )
        slots = {}
        slots_skipped = set()
        metadata_filters = {}
        awaiting_slot = None
        reset_result_state = True

    latest_message = state.get("user_message") or ""
    make_category_options = list(state.get("make_category_options") or [])
    if _catalogue_redirect_allowed(
        state,
        category=category,
        latest_message=latest_message,
    ):
        return _catalogue_redirect_state(
            state,
            decision=decision,
            category=category,
            slots=slots,
            slots_skipped=slots_skipped,
            metadata_filters=metadata_filters,
        )

    if awaiting_slot == _MAKE_CATEGORY_CHOICE_SLOT:
        chosen_category = _category_from_choice(latest_message, make_category_options)
        if chosen_category:
            category = chosen_category
            awaiting_slot = None
            make_category_options = []
            reset_result_state = True
        elif _classify_make_category_no_preference(
            user_message=latest_message,
            make_category_options=make_category_options,
            metadata_filters=metadata_filters,
            slots=slots,
        ):
            awaiting_slot = None
            make_category_options = []
            slots_skipped.add(_MAKE_CATEGORY_CHOICE_SLOT)
        else:
            assistant_text = (
                "Which category should I use: "
                f"{', '.join(make_category_options)}?"
                if make_category_options
                else "Which category should I use?"
            )
            decision["action"] = "respond"
            return {
                **state,
                "trailer_category": category,
                "slots_collected": slots,
                "slots_skipped": sorted(slots_skipped),
                "metadata_filters_collected": metadata_filters,
                "active_search_request_text": _updated_active_search_request_text(
                    state=state,
                    latest_message=latest_message,
                    slots=slots,
                    metadata_filters=metadata_filters,
                    reset_active_request=category_changed or make_changed,
                ),
                "make_category_options": make_category_options,
                "awaiting_slot": _MAKE_CATEGORY_CHOICE_SLOT,
                "pending_questions": [],
                "assistant_text": assistant_text,
                "mind_decision": decision,
            }

    if awaiting_slot == _GENERIC_CATEGORY_CHOICE_SLOT:
        chosen_category = resolve_category_from_text(latest_message).category
        if chosen_category:
            category = chosen_category
            awaiting_slot = None
            reset_result_state = True
        elif _classify_generic_category_no_preference(
            user_message=latest_message,
            metadata_filters=metadata_filters,
            slots=slots,
        ):
            awaiting_slot = None
            slots_skipped.add(_GENERIC_CATEGORY_CHOICE_SLOT)

    category, category_options, make_question = _apply_make_resolution(
        latest_message=latest_message,
        category=category,
        metadata_filters=metadata_filters,
    )
    make_before = str(metadata_filters_before.get("make") or "").strip()
    make_after = str(metadata_filters.get("make") or "").strip()
    make_changed = bool(make_before and make_after and make_before.lower() != make_after.lower())
    if make_changed:
        logger.info(
            "make_change_clears_filters | old_make=%r | new_make=%r",
            make_before,
            make_after,
        )
        slots = {}
        slots_skipped = set()
        metadata_filters = {"make": make_after}
        awaiting_slot = None
        reset_result_state = True
    if _MAKE_CATEGORY_CHOICE_SLOT in slots_skipped:
        category_options = []
        make_question = None
    if category_options and make_question:
        _apply_explicit_filter_extraction(
            state=state,
            category=category,
            slots=slots,
            metadata_filters=metadata_filters,
            latest_message=latest_message,
            awaiting_slot=awaiting_slot,
            apply_slot_updates=False,
        )
        decision["action"] = "respond"
        return {
            **state,
            "trailer_category": category,
            "slots_collected": slots,
            "slots_skipped": sorted(slots_skipped),
            "metadata_filters_collected": metadata_filters,
            "active_search_request_text": _updated_active_search_request_text(
                state=state,
                latest_message=latest_message,
                slots=slots,
                metadata_filters=metadata_filters,
                reset_active_request=category_changed or make_changed,
            ),
            "make_category_options": category_options,
            "awaiting_slot": _MAKE_CATEGORY_CHOICE_SLOT,
            "pending_questions": [],
            "assistant_text": make_question,
            "mind_decision": decision,
        }

    allowed_category_slots = _category_slots(category)
    preference_decision = classify_no_preference(
        category=category,
        user_message=latest_message,
        awaiting_slot=awaiting_slot,
        pending_questions=[] if reset_result_state else (state.get("pending_questions") or []),
        slots_collected=slots,
        metadata_filters_collected=metadata_filters,
        allowed_category_slots=sorted(allowed_category_slots),
        active_question=_last_assistant_text(state.get("messages") or []),
    )
    if _catalogue_redirect_allowed(
        state,
        category=category,
        latest_message=latest_message,
        no_preference_decision=preference_decision,
    ):
        return _catalogue_redirect_state(
            state,
            decision=decision,
            category=category,
            slots=slots,
            slots_skipped=slots_skipped,
            metadata_filters=metadata_filters,
        )
    awaiting_slot, slots_skipped, _removed_filters = _apply_preference_null_decision(
        category=category,
        awaiting_slot=awaiting_slot,
        pending=[] if reset_result_state else (state.get("pending_questions") or []),
        metadata_filters=metadata_filters,
        slots_skipped=slots_skipped,
        decision=preference_decision,
    )
    evidence_awaiting_slot = awaiting_slot

    if awaiting_slot and awaiting_slot not in slots and awaiting_slot not in slots_skipped and latest_message.strip():
        candidate_value = latest_message.strip()
        is_valid, reason = _validate_slot_value(awaiting_slot, candidate_value)
        if is_valid:
            slots[awaiting_slot] = candidate_value
            slots_skipped.discard(str(awaiting_slot))
            awaiting_slot = None
        else:
            invalid_required_slot = awaiting_slot
            logger.info(
                "slot_validation_failed | slot=%s | value=%r | reason=%s",
                awaiting_slot,
                candidate_value,
                reason,
            )

    for key, value in _slot_updates_from_decision(
        decision,
        allowed_category_slots,
        latest_message,
        evidence_awaiting_slot,
    ).items():
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
        slots_skipped.discard(str(key))

    for key, value in _metadata_filters_from_decision(decision, latest_message, evidence_awaiting_slot).items():
        metadata_filters[key] = value
    extracted_slots, _extracted_metadata = _apply_explicit_filter_extraction(
        state=state,
        category=category,
        slots=slots,
        metadata_filters=metadata_filters,
        latest_message=latest_message,
        awaiting_slot=evidence_awaiting_slot,
        apply_slot_updates=True,
    )
    for key, value in extracted_slots.items():
        is_valid, reason = _validate_slot_value(key, value)
        if is_valid:
            slots[key] = value
            slots_skipped.discard(str(key))
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
            slots_skipped.discard(str(key))
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

    if awaiting_slot == _GENERIC_CATEGORY_CHOICE_SLOT and not category:
        decision["action"] = "respond"
        return {
            **state,
            "trailer_category": category,
            "slots_collected": slots,
            "slots_skipped": sorted(slots_skipped),
            "metadata_filters_collected": metadata_filters,
            "active_search_request_text": _updated_active_search_request_text(
                state=state,
                latest_message=latest_message,
                slots=slots,
                metadata_filters=metadata_filters,
                reset_active_request=category_changed or make_changed,
            ),
            "make_category_options": make_category_options,
            "awaiting_slot": _GENERIC_CATEGORY_CHOICE_SLOT,
            "pending_questions": [],
            "assistant_text": _GENERIC_CATEGORY_QUESTION,
            "mind_decision": decision,
        }

    if (
        not category
        and awaiting_slot != _GENERIC_CATEGORY_CHOICE_SLOT
        and not _generic_category_no_preference_active(slots_skipped)
        and not (metadata_filters.get("make") and _MAKE_CATEGORY_CHOICE_SLOT in slots_skipped)
        and _has_generic_trailer_request(latest_message)
    ):
        decision["action"] = "respond"
        return {
            **state,
            "trailer_category": category,
            "slots_collected": slots,
            "slots_skipped": sorted(slots_skipped),
            "metadata_filters_collected": metadata_filters,
            "active_search_request_text": _updated_active_search_request_text(
                state=state,
                latest_message=latest_message,
                slots=slots,
                metadata_filters=metadata_filters,
                reset_active_request=category_changed or make_changed,
            ),
            "make_category_options": make_category_options,
            "awaiting_slot": _GENERIC_CATEGORY_CHOICE_SLOT,
            "pending_questions": [],
            "assistant_text": _GENERIC_CATEGORY_QUESTION,
            "mind_decision": decision,
        }

    _apply_aluminum_base_category_filter(category, slots, metadata_filters)
    _apply_flatbed_default_width(category, slots, metadata_filters)

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
            slots_skipped=slots_skipped,
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
        if q.get("slot") not in slots and q.get("slot") not in slots_skipped
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
            slots_skipped=slots_skipped,
            pending=pending,
            optional_slots=decision.get("optional_question_slots_to_queue") or [],
            required_slots=required_slots_override,
            questions_override=dynamic_questions,
        )
    make_only_missing = _make_only_missing_slots(
        category=category,
        metadata_filters=metadata_filters,
        slots=slots,
        slots_skipped=slots_skipped,
    )
    if make_only_missing:
        pending = _queue_make_only_questions(pending, make_only_missing)
    generic_no_category_missing = _generic_no_category_missing_slots(
        category=category,
        metadata_filters=metadata_filters,
        slots=slots,
        slots_skipped=slots_skipped,
    )
    if generic_no_category_missing:
        pending = _queue_generic_no_category_questions(pending, generic_no_category_missing)

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
    make_only_complete = bool(metadata_filters.get("make")) and not category and not make_only_missing
    generic_no_category_complete = (
        _generic_category_no_preference_active(slots_skipped)
        and not category
        and not generic_no_category_missing
    )
    required_complete = (
        (bool(category) and not missing_required and not invalid_required)
        or make_only_complete
        or generic_no_category_complete
    )
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
    elif action == "pinecone_search" and not category and not make_only_complete and not generic_no_category_complete:
        assistant_text = _GENERIC_CATEGORY_QUESTION
        awaiting_slot = _GENERIC_CATEGORY_CHOICE_SLOT
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
    active_search_request_text = _updated_active_search_request_text(
        state=state,
        latest_message=latest_message,
        slots=slots,
        metadata_filters=metadata_filters,
        reset_active_request=category_changed or make_changed,
    )
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
        "slots_skipped": sorted(slots_skipped),
        "metadata_filters_collected": metadata_filters,
        "active_search_request_text": active_search_request_text,
        "make_category_options": make_category_options,
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
    latest_user_message = state.get("user_message") or ""
    active_search_request_text = (
        str(state.get("active_search_request_text") or "").strip()
        or latest_user_message
    )
    logger.info(
        "pinecone_search_execute | category=%r | slots=%s | metadata_filters=%s | shown_count=%s",
        state.get("trailer_category"),
        json.dumps(state.get("slots_collected") or {}, default=str),
        json.dumps(state.get("metadata_filters_collected") or {}, default=str),
        len(state.get("already_shown_listing_urls") or []),
    )
    search_result = search_pinecone_listing_result(
        category=state.get("trailer_category"),
        slots=state.get("slots_collected") or {},
        metadata_filters=state.get("metadata_filters_collected") or {},
        user_message=active_search_request_text,
        already_shown_urls=state.get("already_shown_listing_urls") or [],
    )
    if isinstance(search_result, list):
        search_result = PineconeListingSearchResult(
            listings=search_result,
            query_text=active_search_request_text,
            metadata_filter=None,
        )
    listings_with_evidence = search_result.listings
    intro_text, match_analysis = _pinecone_match_framing_text(
        user_message=active_search_request_text,
        latest_user_message=latest_user_message,
        category=state.get("trailer_category"),
        slots=state.get("slots_collected") or {},
        metadata_filters=state.get("metadata_filters_collected") or {},
        search_result=search_result,
    )
    search_result.match_analysis = match_analysis
    listings_with_evidence = _apply_pinecone_match_validation(listings_with_evidence, match_analysis)
    listings = [_public_listing(item) for item in listings_with_evidence]
    assistant_text = format_listing_results(
        listings,
        category=state.get("trailer_category"),
        slots=state.get("slots_collected") or {},
        user_message=active_search_request_text,
    )
    if intro_text:
        assistant_text = f"{intro_text}\n\n{assistant_text}"
    followup = _result_interest_followup_text(
        user_message=active_search_request_text,
        category=state.get("trailer_category"),
        slots=state.get("slots_collected") or {},
        listings=listings,
    )
    if followup:
        assistant_text = f"{assistant_text}\n\n{followup}"
    shown = list(state.get("already_shown_listing_urls") or [])
    shown.extend([str(x.get("url")) for x in listings if x.get("url")])
    events = list(state.get("tool_events") or [])
    event = {"tool": "pinecone_search", "result_count": len(listings)}
    if match_analysis:
        event["match_analysis"] = match_analysis
    events.append(event)
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
    if not _has_contact(state):
        events = list(state.get("tool_events") or [])
        events.append({"tool": "send_interested_listing_email", "result": {"status": "deferred_missing_contact"}})
        return {
            **state,
            "assistant_text": (
                "Would you like to share your phone number or email address so our team can "
                f"follow up with you about your interest in {title}?"
            ),
            "pending_contact_action": {
                "type": "interest",
                "item_name": str(title),
                "selected_listing_url": decision.get("selected_listing_url") or state.get("selected_listing_url"),
            },
            "tool_events": events,
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
    if category not in FAQ_CATEGORY_LABELS:
        category = "contact_human"
    summary = FAQ_CATEGORY_LABELS[category]
    if not _has_contact(state):
        events = list(state.get("tool_events") or [])
        events.append({"tool": "send_non_sales_faq_email", "result": {"status": "deferred_missing_contact"}})
        return {
            **state,
            "assistant_text": _optional_contact_request(summary.lower()),
            "pending_contact_action": {
                "type": "faq",
                "faq_category": category,
                "summary": summary,
                "user_message": state.get("user_message") or "",
            },
            "tool_events": events,
        }
    result = send_non_sales_faq_email(
        session_id=state.get("session_id") or "",
        full_name=state.get("customer_full_name") or "",
        email=state.get("customer_email"),
        phone=state.get("customer_phone") or "",
        faq_category=category,
        summary=summary,
        user_message=state.get("user_message") or "",
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
