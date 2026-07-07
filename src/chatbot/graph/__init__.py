from __future__ import annotations

import json
import logging
import os
import re
from functools import lru_cache
from typing import Any, Literal, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field, field_validator, model_validator

from trailer_fields import get_trailer_fields_as_dict, list_all_categories
from src.chatbot.categories import (
    CANONICAL_CATEGORIES,
    category_prompt_block,
    category_clarification_question,
    resolve_categories_from_text,
    resolve_category_clarification_answer,
    resolve_category_from_text,
)
from src.chatbot.formatting import format_listing_results
from src.chatbot.email_reply import compose_email_tool_reply
from src.chatbot.mini_llm_classifier import (
    HaulClassificationDecision,
    classify_haul_requirements,
)
from src.chatbot.mini_preference_classifier import (
    PreferenceNullDecision,
    classify_no_preference,
)
from src.chatbot.make_inventory import categories_for_make, make_prompt_block
from src.chatbot.make_resolver import resolve_make_from_text
from src.chatbot.constants import (
    DYNAMIC_WIDTH_EXCLUDED_CATEGORIES,
    compact_listings,
    compact_recent_messages,
)
from src.chatbot.graph.apply_mind import build_state_return
from src.chatbot.llm import make_llm, safe_invoke
from src.chatbot.prompts import MIND_SYSTEM_PROMPT, TRAILERPLACE_KNOWLEDGE_SECTION
from src.chatbot.state import ChatbotState, QuestionItem
from src.models import TrailerListing
from src.normalizer import normalize_category, normalize_hitch, normalize_subcategory
from src.chatbot.tools.email_tools import (
    FAQ_CATEGORY_LABELS,
    send_escalation_alert_email,
    send_interested_listing_email,
    send_non_sales_faq_email,
)
from src.chatbot.tools.pinecone_search import (
    PineconeListingSearchResult,
    search_pinecone_listing_result,
)
from src.conversation_store import persist_messages_snapshot

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
_ESCALATION_SENT_REPLY = (
    "I've sent your query to our team, and they'll reach out to you soon. "
    "In the meantime, I can keep helping you narrow down the right trailer."
)
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


def _compact_recent_context(state: ChatbotState) -> str:
    parts: list[str] = []
    category = str(state.get("trailer_category") or "").strip()
    if category:
        parts.append(f"Category: {category}")
    slots = state.get("slots_collected") or {}
    if slots:
        parts.append(f"Slots: {_safe_json(slots)}")
    metadata = state.get("metadata_filters_collected") or {}
    if metadata:
        parts.append(f"Metadata filters: {_safe_json(metadata)}")
    listings = state.get("last_listings") or []
    if listings:
        titles = [
            str(item.get("title") or "").strip()
            for item in listings[:3]
            if str(item.get("title") or "").strip()
        ]
        if titles:
            parts.append(f"Recent listings shown: {'; '.join(titles)}")
    recent = state.get("messages") or []
    if recent:
        transcript = _format_recent_message_transcript(recent, limit=4)
        if transcript:
            parts.append(f"Recent conversation:\n{transcript}")
    return " | ".join(parts)[:1800]


def _active_question_followup(state: ChatbotState) -> str:
    awaiting = str(state.get("awaiting_slot") or "").strip()
    if awaiting:
        queued = _queued_question_for_slot(state.get("pending_questions") or [], awaiting)
        if queued:
            return queued
    pending_change = state.get("pending_category_change") or {}
    if awaiting == "category_filter_confirmation" and pending_change:
        return str(pending_change.get("question") or "").strip()
    pending_suggestion = state.get("pending_category_suggestion") or {}
    if awaiting == "category_suggestion_confirmation" and pending_suggestion:
        return _category_suggestion_prompt(pending_suggestion)
    if awaiting:
        try:
            definition = _slot_definition(awaiting, category=state.get("trailer_category"))
            return str(definition.get("question") or "").strip()
        except Exception:
            logger.debug("active_question_definition_unavailable | slot=%r", awaiting)
    pending = state.get("pending_questions") or []
    if pending:
        return str(pending[0].get("question") or "").strip()
    return ""


def _persist_email_transcript_snapshot(state: ChatbotState) -> None:
    persist_messages_snapshot(
        session_id=str(state.get("session_id") or ""),
        lead_id=state.get("lead_id"),
        messages=state.get("messages") or [],
    )


def _append_active_question_if_present(state: ChatbotState, text: str) -> str:
    question = _active_question_followup(state)
    base = str(text or "").strip()
    if question and question not in base:
        return f"{base}\n\n{question}" if base else question
    return base


def _category_suggestion_options(payload: dict[str, Any]) -> list[str]:
    raw = payload.get("categories") or []
    options: list[str] = []
    for item in raw:
        category = str((item or {}).get("category") if isinstance(item, dict) else item).strip()
        if category in CANONICAL_CATEGORIES and category not in options:
            options.append(category)
    for key in ("category", "recommended_category"):
        category = str(payload.get(key) or "").strip()
        if category in CANONICAL_CATEGORIES and category not in options:
            options.append(category)
    return options


def _category_suggestion_prompt(payload: dict[str, Any], *, confirm_recommended: bool = False) -> str:
    recommended = str(payload.get("recommended_category") or payload.get("category") or "").strip()
    if (confirm_recommended or payload.get("status") == "awaiting_recommended_confirmation") and recommended:
        return f"I recommend {recommended}. Would you like to continue with {recommended} trailers?"
    options = _category_suggestion_options(payload)
    if options:
        return (
            f"A few trailer types could fit: {', '.join(options)}. "
            'You can choose one, or say "recommend one" and I will pick the best fit.'
        )
    return "I can recommend a trailer type for that use case. Would you like me to recommend one?"


_SEARCH_PROMISE_RE = re.compile(
    r"\b(?:let me|i(?:'ll| will| can))\s+"
    r"(?:find|search|look|pull up|look up|show|recommend)\b[^.?!]*(?:trailers?|options?|listings?)?[^.?!]*(?:[.?!]|$)"
    r"|\bi\s+can\s+help\s+you\s+(?:find|search|look|pull up|look up|show|recommend)\b[^.?!]*(?:trailers?|options?|listings?)?[^.?!]*(?:[.?!]|$)"
    r"|\bplease hold on\b[^.?!]*(?:[.?!]|$)",
    re.I,
)


def _strip_search_promises(text: Any) -> str:
    clean = _SEARCH_PROMISE_RE.sub("", str(text or "")).strip()
    clean = re.sub(r"[ \t]{2,}", " ", clean)
    return re.sub(r"\n{3,}", "\n\n", clean).strip()


def _has_categoryless_spec_update(message: str) -> bool:
    return bool(
        _explicit_length_requested(message)
        or _explicit_width_requested(message)
        or _explicit_payload_requested(message)
        or _explicit_price_requested(message)
        or _explicit_color_requested(message)
        or _explicit_hitch_requested(message)
    )


def _has_categoryless_trailer_features(
    *,
    metadata_filters: dict[str, Any],
    slots: dict[str, Any],
    requested_features: list[str],
) -> bool:
    return bool(
        metadata_filters
        or requested_features
        or any(
            key not in {_GENERIC_CATEGORY_CHOICE_SLOT, _GENERIC_HAUL_USE_SLOT}
            for key in slots
        )
    )


class MindDecision(BaseModel):
    action: Literal[
        "ask_trailer_category",
        "ask_next_question",
        "pinecone_search",
        "send_interested_listing_email",
        "send_non_sales_faq_email",
        "send_escalation_alert_email",
        "respond",
    ] = "respond"
    assistant_text: str = ""
    trailer_category: Optional[str] = None
    category_resolution_kind: Literal["explicit", "recommendation", "none"] = "none"
    category_confidence: Literal["low", "medium", "high"] = "low"
    category_reasoning: str = ""
    category_suggestion_response: Literal["accept", "reject", "recommend_one", "none"] = "none"
    category_recommendations: list[dict[str, Any]] = Field(default_factory=list)
    recommended_category: Optional[str] = None
    slots_collected_update: dict[str, Any] = Field(default_factory=dict)
    metadata_filters_update: dict[str, Any] = Field(default_factory=dict)
    optional_question_slots_to_queue: list[str] = Field(default_factory=list)
    selected_listing_title: Optional[str] = None
    selected_listing_url: Optional[str] = None
    faq_category: Optional[str] = None
    faq_summary: Optional[str] = None
    escalation_summary: Optional[str] = None
    unsupported_request: Optional[str] = None

    @field_validator("category_recommendations", mode="before")
    @classmethod
    def normalize_category_recommendations(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return value
        return [
            {
                "category": item,
                "confidence": "medium",
                "reasoning": "",
            }
            if isinstance(item, str)
            else item
            for item in value
        ]


class FilterExtractionDecision(BaseModel):
    length_ft: Optional[str] = None
    width_ft: Optional[str] = None
    height_ft: Optional[str] = None
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


class FieldExtractionAdjudicationDecision(BaseModel):
    metadata_filters_update: dict[str, Any] = Field(default_factory=dict)
    slots_collected_update: dict[str, Any] = Field(default_factory=dict)
    requested_non_metadata_features: list[str] = Field(default_factory=list)
    rejected_candidates: list[dict[str, Any]] = Field(default_factory=list)
    clarification_needed: Optional[str] = None
    confidence: Literal["low", "medium", "high"] = "low"
    reason: str = ""

    @field_validator("rejected_candidates", mode="before")
    @classmethod
    def normalize_empty_rejected_candidates(cls, value: Any) -> Any:
        # Function-calling models occasionally encode an empty JSON array as
        # an empty object. They are equivalent for this optional diagnostic
        # field, so normalize that harmless shape mismatch before validation.
        if value == {}:
            return []
        return value


class NonRecommendationTurnDecision(BaseModel):
    turn_type: Literal[
        "contact_or_store_info",
        "supported_faq",
        "unsupported_business_action",
        "catalogue_request",
        "trailer_shopping_missing_category",
        "trailer_shopping_with_category_or_filters",
        "answer_to_active_question",
        "smalltalk_or_other",
    ] = "smalltalk_or_other"
    action: Literal[
        "respond",
        "ask_trailer_category",
        "send_non_sales_faq_email",
        "send_escalation_alert_email",
        "continue_recommendation_flow",
    ] = "continue_recommendation_flow"
    faq_category: Optional[str] = None
    faq_summary: Optional[str] = None
    escalation_summary: Optional[str] = None
    assistant_text: str = ""
    should_store_freeform_fields: bool = True
    reason: str = ""
    confidence: Literal["low", "medium", "high"] = "low"


class QuestionTurnDecision(BaseModel):
    counter_question_topic: Literal[
        "trailer_categories", "hitch_types", "makes", "dimensions",
        "payload", "faq", "other", "none",
    ] = "none"
    answered_active_question: bool = False
    no_preference_for_active_question: bool = False
    search_now_requested: bool = False
    skip_remaining_questions: bool = False
    active_slot_value: Optional[str] = None
    metadata_filters_update: dict[str, Any] = Field(default_factory=dict)
    slots_collected_update: dict[str, Any] = Field(default_factory=dict)
    requested_non_metadata_features: list[str] = Field(default_factory=list)
    email_action: Literal[
        "none",
        "send_non_sales_faq_email",
        "send_escalation_alert_email",
    ] = "none"
    faq_category: Optional[str] = None
    faq_summary: Optional[str] = None
    escalation_summary: Optional[str] = None
    unsupported_request: Optional[str] = None
    reply_to_user: str = ""
    rephrased_question: str = ""
    retry_slot: Optional[str] = None
    confidence: Literal["low", "medium", "high"] = "low"
    reason: str = ""

    @model_validator(mode="before")
    @classmethod
    def normalize_structured_active_slot_value(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        value = normalized.get("active_slot_value")
        if isinstance(value, dict):
            metadata = dict(normalized.get("metadata_filters_update") or {})
            dimension_parts: list[str] = []
            for key in ("length_ft", "width_ft", "height_ft"):
                raw = value.get(key)
                if raw in (None, ""):
                    continue
                text = format(raw, "g") if isinstance(raw, float) else str(raw).strip()
                if isinstance(raw, (int, float)) or re.fullmatch(r"\d+(?:\.\d+)?", text):
                    text = f"{text} ft"
                metadata.setdefault(key, text)
                dimension_parts.append(text)
            if dimension_parts:
                normalized["active_slot_value"] = " × ".join(dimension_parts)
                normalized["metadata_filters_update"] = metadata
            else:
                normalized["active_slot_value"] = ", ".join(
                    f"{str(key).replace('_', ' ')}: {value}"
                    for key, value in value.items()
                    if value not in (None, "")
                )
        elif isinstance(value, (list, tuple, set)):
            normalized["active_slot_value"] = ", ".join(
                str(item).strip() for item in value if str(item).strip()
            )
        elif isinstance(value, bool):
            normalized["active_slot_value"] = "yes" if value else "no"
        return normalized

    @field_validator("active_slot_value", mode="before")
    @classmethod
    def coerce_numeric_active_slot_value(cls, value: Any) -> Any:
        """Accept JSON numeric answers without invalidating the full LLM response."""
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            return format(value, "g")
        return value


class CategoryTransitionDecision(BaseModel):
    final_category: Optional[str] = None
    approve_category_change: bool = False
    explicit_category_switch: bool = False
    confidence: Literal["low", "medium", "high"] = "low"
    reason: str = ""


class MakeVerificationDecision(BaseModel):
    approve_make: bool = False
    verified_make: Optional[str] = None
    explicit_evidence: str = ""
    confidence: Literal["low", "medium", "high"] = "low"
    reason: str = ""


class CategoryFilterConfirmationDecision(BaseModel):
    keep_fields: list[str] = Field(default_factory=list)
    discard_fields: list[str] = Field(default_factory=list)
    updates: dict[str, Any] = Field(default_factory=dict)
    resolved: bool = False
    reasoning: str = ""
    confidence: Literal["low", "medium", "high"] = "low"


class OfficeTrailerClarificationDecision(BaseModel):
    answered_clarification: bool = False
    resolved_category: Optional[Literal["Fiber", "Enclosed"]] = None
    email_action: Literal[
        "none",
        "send_non_sales_faq_email",
        "send_escalation_alert_email",
    ] = "none"
    faq_category: Optional[str] = None
    faq_summary: Optional[str] = None
    escalation_summary: Optional[str] = None
    unsupported_request: Optional[str] = None
    reply_to_user: str = ""
    confidence: Literal["low", "medium", "high"] = "low"
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


@lru_cache(maxsize=1)
def _mind_llm():
    return make_llm(structured_output=MindDecision)


@lru_cache(maxsize=1)
def _category_filter_confirmation_llm():
    return make_llm(
        model_env="CATEGORY_FILTER_CONFIRMATION_MODEL",
        structured_output=CategoryFilterConfirmationDecision,
    )


@lru_cache(maxsize=1)
def _filter_extractor_llm():
    return make_llm(
        model_env="FILTER_EXTRACTOR_MODEL",
        structured_output=FilterExtractionDecision,
    )


@lru_cache(maxsize=1)
def _requested_feature_extractor_llm():
    return make_llm(
        model_env="REQUESTED_FEATURE_EXTRACTOR_MODEL",
        structured_output=RequestedFeatureExtractionDecision,
    )


@lru_cache(maxsize=1)
def _field_extraction_adjudicator_llm():
    return make_llm(
        model_env="FIELD_EXTRACTION_ADJUDICATOR_MODEL",
        structured_output=FieldExtractionAdjudicationDecision,
    )


@lru_cache(maxsize=1)
def _non_recommendation_turn_llm():
    return make_llm(
        model_env="NON_RECOMMENDATION_TURN_MODEL",
        structured_output=NonRecommendationTurnDecision,
    )


@lru_cache(maxsize=1)
def _question_turn_adjudicator_llm():
    return make_llm(
        model_env="QUESTION_TURN_ADJUDICATOR_MODEL",
        structured_output=QuestionTurnDecision,
    )


@lru_cache(maxsize=1)
def _active_turn_reconciler_llm():
    return make_llm(
        model_env="ACTIVE_TURN_RECONCILER_MODEL",
        structured_output=QuestionTurnDecision,
    )


@lru_cache(maxsize=1)
def _category_transition_llm():
    return make_llm(
        model_env="CATEGORY_TRANSITION_MODEL",
        structured_output=CategoryTransitionDecision,
    )


@lru_cache(maxsize=1)
def _make_verification_llm():
    return make_llm(
        model_env="MAKE_VERIFICATION_MODEL",
        structured_output=MakeVerificationDecision,
    )


@lru_cache(maxsize=1)
def _office_trailer_clarification_llm():
    return make_llm(
        model_env="OFFICE_TRAILER_CLARIFICATION_MODEL",
        structured_output=OfficeTrailerClarificationDecision,
    )


def _pinecone_match_framing_llm_enabled() -> bool:
    return (os.getenv("PINECONE_MATCH_FRAMING_LLM_ENABLED") or "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _pinecone_audit_model_name() -> str:
    # Single source of truth for the auditor model name so the diagnostic log
    # (below) can never drift from the model actually invoked.
    return (
        os.getenv("PINECONE_MATCH_AUDIT_MODEL")
        or os.getenv("PINECONE_MATCH_FRAMING_MODEL")
        or "gpt-5-mini"
    ).strip()


@lru_cache(maxsize=1)
def _pinecone_match_audit_llm():
    # Reasoning model with strict JSON schema; its model resolution deliberately
    # does not fall back to OPENAI_MODEL, so pass the resolved name explicitly.
    return make_llm(
        model=_pinecone_audit_model_name(),
        temperature=None,
        use_responses_api=True,
        reasoning={"effort": "minimal"},
        structured_output=PineconeMatchFramingDecision,
        method="json_schema",
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


# Leading negation on the extracted feature itself ("no tarp", "without ramps").
_FEATURE_NEGATION_PREFIX_RE = re.compile(r"^(?:no|not|without|never|skip|avoid)\b", re.I)
# A negation clause immediately before the feature mention in the source text.
_FEATURE_NEGATION_CONTEXT_RE = re.compile(
    r"\b(?:no|not|don'?t|do not|does not|doesn'?t|without|never|skip|avoid|no need for)\b[^.?!;]*$",
    re.I,
)
# Bare informational question with no request verb — the extractor sometimes
# invents abstract "features" from these ("cargo types", "load capacity").
_INFORMATIONAL_QUESTION_RE = re.compile(r"^\s*(?:what|which|how|can|could|does|do|are|is)\b", re.I)
_REQUEST_VERB_RE = re.compile(
    r"\b(?:need|want|require|looking for|must have|should have|include|includes|"
    r"with a|with an|prefer|add|equipped with|comes with)\b",
    re.I,
)


def _filter_negated_or_ungrounded_features(features: list[str], message: str) -> list[str]:
    """Drop features that the customer negated or that were invented from an
    informational question.

    The LLM feature extractor is strong on positive requests but occasionally
    turns a negation ("I don't need a tarp") into a requested feature ("no
    tarp"/"tarp") or fabricates abstract features from a bare informational
    question. Those wrongly become Pinecone search requirements, so we strip them
    deterministically after extraction.
    """
    if not features:
        return features
    low_message = (message or "").lower()
    stripped = low_message.strip()
    is_bare_info_question = bool(
        "?" in low_message
        and _INFORMATIONAL_QUESTION_RE.match(stripped)
        and not _REQUEST_VERB_RE.search(low_message)
    )

    kept: list[str] = []
    for feature in features:
        feature_low = feature.lower().strip()
        # 1) The feature phrase itself is a negation.
        if _FEATURE_NEGATION_PREFIX_RE.match(feature_low):
            continue
        # 2) The feature is mentioned in the message inside a negation clause.
        idx = low_message.find(feature_low)
        if idx != -1 and _FEATURE_NEGATION_CONTEXT_RE.search(low_message[:idx]):
            continue
        # 3) Informational question with no request verb, and this feature is not
        #    literally grounded in the customer's words → it was invented.
        if is_bare_info_question and feature_low not in low_message:
            continue
        kept.append(feature)
    return kept


def _build_match_short_reason(
    *,
    item: dict[str, Any],
    fact: dict[str, Any],
    category: str | None,
    metadata_filters: dict[str, Any],
    requested_features: list[str],
) -> str:
    level = _safe_match_level(item.get("match_level"))
    confirmed = [
        str(value).strip()
        for value in (item.get("confirmed_requirements") or [])
        if str(value).strip()
    ]
    missing = [
        str(value).strip()
        for value in (item.get("missing_or_unconfirmed_requirements") or [])
        if str(value).strip()
    ]
    context_match = _structured_listing_context_matches(
        fact=fact,
        category=category,
        metadata_filters=metadata_filters,
    )

    if level == "full":
        if requested_features:
            return "full: category/structured filters align and all requested features are confirmed"
        return "full: category/structured filters align and no requested feature is left unconfirmed"
    if level == "partial":
        partial_reasons: list[str] = []
        if missing:
            partial_reasons.append(f"requested features still unconfirmed: {', '.join(missing)}")
        if confirmed:
            partial_reasons.append("relevant fit signals are confirmed")
        if not partial_reasons:
            partial_reasons.append("relevant listing, but exact requested combination is not fully confirmed")
        return f"partial: {'; '.join(partial_reasons)}"
    if level == "alternative":
        alternative_reasons: list[str] = []
        if not context_match:
            alternative_reasons.append("structured category/filter mismatch")
        if missing:
            alternative_reasons.append(f"requested features not confirmed: {', '.join(missing)}")
        if not alternative_reasons:
            alternative_reasons.append("relevant inventory option, but requested combination is not confirmed")
        return f"alternative: {'; '.join(alternative_reasons)}"
    return "unknown: match auditor did not return a recognized match level"


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
        fact = by_position.get(int(updated.get("position") or 0), {})
        context_match = _structured_listing_context_matches(
            fact=fact,
            category=category,
            metadata_filters=metadata_filters,
        )
        if not clean_requested and context_match:
            updated["match_level"] = "full"
        elif (
            filtered_missing != original_missing
            and not filtered_missing
            and context_match
        ):
            updated["match_level"] = "full"
        updated["short_reason"] = _build_match_short_reason(
            item=updated,
            fact=fact,
            category=category,
            metadata_filters=metadata_filters,
            requested_features=clean_requested,
        )
        sanitized.append(updated)

    sanitized = _enforce_requested_feature_consistency(sanitized, clean_requested)
    for item in sanitized:
        item["short_reason"] = _build_match_short_reason(
            item=item,
            fact=by_position.get(int(item.get("position") or 0), {}),
            category=category,
            metadata_filters=metadata_filters,
            requested_features=clean_requested,
        )
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
        "intro_text": re.sub(r"\s+", " ", str(data.get("intro_text") or "").strip()),
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
            "confirmed_requirements=%s | missing_or_unconfirmed_requirements=%s | customer_label=%r | short_reason=%r | sales_blurb=%r",
            position,
            item.get("match_level") or "unknown",
            fact.get("title"),
            fact.get("url"),
            _safe_json(item.get("confirmed_requirements") or []),
            _safe_json(item.get("missing_or_unconfirmed_requirements") or []),
            item.get("customer_label") or "",
            item.get("short_reason") or "",
            item.get("sales_blurb") or "",
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
    if full_count > 0 and re.search(
        r"\b(?:not\s+currently\s+shown|not\s+clearly\s+shown|do\s+not\s+currently\s+show|"
        r"don't\s+currently\s+show|exact\s+requested\s+combination\s+is\s+not)\b",
        lower,
    ):
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


_AUDIT_IGNORED_KEY_PARTS = (
    "length",
    "width",
    "height",
    "size",
    "weight",
    "payload",
    "gvwr",
    "axle_capacity",
)


def _audit_sanitize_mapping(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _audit_sanitize_mapping(item)
            for key, item in value.items()
            if not any(part in str(key).lower() for part in _AUDIT_IGNORED_KEY_PARTS)
        }
    if isinstance(value, list):
        return [_audit_sanitize_mapping(item) for item in value]
    return value


def _audit_sanitize_text(value: Any) -> str:
    text = str(value or "")
    ignored_label = re.compile(
        r"^\s*(?:current\s+requirements\s*:\s*)?(?:filter_)?(?:trailer_)?"
        r"(?:length(?:_ft)?|width(?:_ft)?|height(?:_ft)?|size|payload(?:_lbs|\s+capacity)?|gvwr|"
        r"dry\s+weight|axle\s+capacity|weight(?:\s+capacity)?)\s*(?:[:;,]|\s+\d)",
        re.I,
    )
    pieces = re.split(r"([|\n;])", text)
    text = "".join(
        "" if ignored_label.search(piece) else piece
        for piece in pieces
    )
    text = re.sub(r"\b\d+(?:\.\d+)?\s*[xX]\s*\d+(?:\.\d+)?(?:\s*[xX]\s*\d+(?:\.\d+)?)?\b", " ", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\s*(?:ft|feet|foot|inches?|inch)\b", " ", text, flags=re.I)
    text = re.sub(r"\b\d[\d,]*(?:\.\d+)?\s*(?:lbs?|pounds?|tons?|#)(?=\s|$|[|,;])", " ", text, flags=re.I)
    text = re.sub(r"\b\d+(?:\.\d+)?\s*['\"]", " ", text)
    text = re.sub(r"\s*\|\s*(?=\||$)", " ", text)
    return re.sub(r"\s+", " ", text).strip(" |;,.-")


def _pinecone_listing_facts(listings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "position": idx,
            "title": _audit_sanitize_text(item.get("title")),
            "category": item.get("category"),
            "subcategory": item.get("subcategory"),
            "make": item.get("make"),
            "model": _audit_sanitize_text(item.get("model")),
            "price": item.get("price_display") or item.get("price"),
            "hitch_type": item.get("hitch_type"),
            "color": item.get("color"),
            "relevance_score": item.get("relevance_score"),
            "match_evidence_text": _audit_sanitize_text(item.get("match_evidence_text"))[:1800],
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
    clean = _filter_negated_or_ungrounded_features(clean, latest_user_message or user_message)
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
    # Scenario boundaries below are relative to the actual number of listings
    # shown, not a hardcoded 6. The previous magic constant was tied to
    # SEARCH_MAX_RECOMMENDATIONS (default 5), so Scenario C either never fired or
    # misfired whenever that knob changed.
    listing_count = len(listings)
    max_partial_count = max(listing_count - 1, 1)
    audit_user_message = _audit_sanitize_text(user_message)
    audit_latest_message = _audit_sanitize_text(latest_user_message or user_message)
    audit_slots = _audit_sanitize_mapping(slots)
    audit_metadata_filters = _audit_sanitize_mapping(metadata_filters)
    audit_messages = [
        SystemMessage(
                content=f"""You are an internal trailer match auditor. Return structured audit data only.

Classify each supplied Pinecone result as full, partial, or alternative against the user's complete active request.

Core rule:
For overall fit, focus on category/use case, make/model/hitch/color when requested, and the supplied requested non-metadata features.

Length, width, height, payload capacity, GVWR, and other weights are handled outside this audit and removed from your context. Never infer, evaluate, mention, or return them.

Strict feature validation:
Match requested non-metadata features by exact feature concept, not by broad category words.

Use language understanding for natural wording variants, but be strict about feature concepts. Slide gate, sliding gate, slider gate, and swing-slide gate may be the same feature if evidence confirms that concept. Butterfly gate is not the same requested feature as sliding gate. Ramp gate, rear gate, side gate, escape door, divider gate, and butterfly gate do not satisfy sliding gate unless evidence explicitly says it is also sliding/slide/swing-slide. Offroad wheels, off-road tires, and all-terrain tires may satisfy an offroad wheel request only when evidence supports that concept.

Evidence rules:
Use only supplied listing facts and match_evidence_text. Do not invent features, prices, specs, availability, or reasons. Missing, unclear, implied, common-for-category, or merely related evidence is unconfirmed.

Important:
Treat the supplied requested_non_metadata_features as authoritative. Do not add, infer, broaden, or substitute any new requested features from listing text or general trailer knowledge. If the supplied list is empty, return an empty requested_non_metadata_features list.

Critical defaulting rule:
If requested_non_metadata_features is empty, and the listing matches the requested category/use case plus any explicitly requested make, hitch, color, or subcategory filters, classify it as full. Do not use partial or alternative in that case unless there is a mismatch on one of those requested category/structured filters.

Return:

* intro_text: Write 1–2 short sentences for the customer, shown before the listings. Maximum 55 words total. Plain text only — no markdown, no bullets, no labels, no quotation marks. Never ask for contact details, phone number, email, callback, or any sales follow-up.

Pick exactly ONE scenario below based on full_match_count and follow it exactly. Do not mix wording from other scenarios.

SCENARIO A — full_match_count = 0:
Say that the exact requested combination is not currently shown. Then describe the listed trailers positively using phrases like "strongest available options," "practical choices," or "useful options to compare."
Never use these words/phrases here: strong match, close match, best match, top match, closest match, best-fitting, exact match.
Example: "We don't have that exact combination right now, but here are some practical options worth comparing."

SCENARIO B — full_match_count is 1 to {max_partial_count}:
Say that confirmed fits are shown first, followed by other relevant options worth comparing.
Example: "Your confirmed fits are listed first, followed by other relevant trailers worth comparing."

SCENARIO C — full_match_count = {listing_count} (every shown listing is a full match):
Confidently state that every trailer shown matches the trailer type and features the customer asked for. End with a short, confident line that this is a ready-to-compare lineup built for their request. Write it like an upbeat, confident marketer — not a flat confirmation. Do NOT claim it matches "exactly" or that it meets their exact dimensions or specifications, because dimensions are audited separately and are not part of this check.
Example: "Every trailer below matches the trailer type and features you asked for — a ready-to-compare lineup built around your request."

Rules for ALL scenarios:(extremely important — follow these carefully)
- Never mention dimensions, payload capacity, GVWR, or other weights.
- Never list specific mismatches.
- Never say "all results meet your needs/request/specifications/criteria" unless full_match_count = {listing_count}.

* requested_non_metadata_features: user-requested features not represented by normal structured filters.
* per_listing_match for every supplied listing position.
* match_level: full if the listing fits the main trailer type/use case and all explicitly requested non-metadata features are confirmed. If requested_non_metadata_features is empty, use full whenever the listing matches the requested category/use case plus any explicitly requested make, hitch, color, or subcategory filters. Use partial when some explicitly requested non-metadata features are confirmed but others are missing/unconfirmed. Use alternative when the listing is relevant but the explicitly requested feature combination is not confirmed.
* confirmed_requirements: only confirmed facts/features.
* missing_or_unconfirmed_requirements: requested features absent, ambiguous, merely similar, unsupported, or replaced by a different feature type. missing_or_unconfirmed_requirements must include requested features that are not confirmed.
* sales_blurb: Talk like an experienced sales representative who wants to make a trailer sale. Make one positive customer-facing sentence, max 28 words, highlighting confirmed strengths only. It should be engaging for the customer. Do not say partial match, alternative, mismatch, missing, requirement, exceeds, does not meet, or fully/exactly/perfectly matches unless match_level is full."""
        ),
        HumanMessage(
                content=_safe_json(
                    {
                        "user_message": audit_user_message,
                        "active_search_request_text": audit_user_message,
                        "latest_user_message": audit_latest_message,
                        "category": category,
                        "slots": audit_slots,
                        "metadata_filters": audit_metadata_filters,
                        "requested_non_metadata_features": requested_non_metadata_features,
                        "pinecone_embedding_query_text": _audit_sanitize_text(search_result.query_text),
                        "pinecone_metadata_filter": _audit_sanitize_mapping(search_result.metadata_filter),
                        "make_debug": search_result.make_debug,
                        "listings": facts,
                    }
                )
        ),
    ]
    output_schema = (
        PineconeMatchFramingDecision.model_json_schema()
        if hasattr(PineconeMatchFramingDecision, "model_json_schema")
        else PineconeMatchFramingDecision.schema()
    )
    logger.info(
        "pinecone_match_audit_input_json | %s",
        _safe_json(
            {
                "model": _pinecone_audit_model_name(),
                "temperature": None,
                "use_responses_api": True,
                "reasoning": {"effort": "minimal"},
                "structured_output_method": "json_schema",
                "structured_output_schema": output_schema,
                "messages": [
                    {
                        "role": "system" if isinstance(message, SystemMessage) else "user",
                        "content": message.content,
                    }
                    for message in audit_messages
                ],
            }
        ),
    )

    decision = _pinecone_match_audit_llm().invoke(audit_messages)
    raw_data = _model_dump(decision)
    logger.info(
        "pinecone_match_audit_output_json | %s",
        _safe_json(raw_data),
    )

    raw_counts = (
        raw_data.get("overall_match_level"),
        raw_data.get("full_match_count"),
        raw_data.get("partial_match_count"),
        raw_data.get("alternative_count"),
    )
    data = raw_data
    data = _sanitize_requested_feature_analysis(
        data=data,
        facts=facts,
        requested_features=requested_non_metadata_features,
        category=category,
        metadata_filters=metadata_filters,
    )
    validated_counts = (
        data.get("overall_match_level"),
        data.get("full_match_count"),
        data.get("partial_match_count"),
        data.get("alternative_count"),
    )
    if validated_counts != raw_counts:
        full_count = int(data.get("full_match_count") or 0)
        listing_count = int(data.get("validated_listing_count") or len(facts))
        if full_count == listing_count and listing_count:
            data["intro_text"] = (
                "Every trailer below is a confirmed match for the audited features—a ready-to-compare lineup built around your request."
            )
        elif full_count:
            data["intro_text"] = (
                "Your confirmed fits are listed first, followed by other relevant trailers worth comparing."
            )
        else:
            data["intro_text"] = (
                "We don't have that exact feature combination right now, but here are some practical options worth comparing."
            )
        data["reason"] = "Final match summary regenerated after deterministic validation."
    logger.info(
        "pinecone_match_audit_validated_output_json | %s",
        _safe_json(data),
    )
    return data


def _pinecone_match_framing_text(
    *,
    user_message: str,
    latest_user_message: str | None = None,
    category: str | None,
    slots: dict[str, Any],
    metadata_filters: dict[str, Any],
    search_result: PineconeListingSearchResult,
    requested_non_metadata_features: list[str] | None = None,
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
    if requested_non_metadata_features is None:
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

    text = re.sub(r"\s+", " ", str(data.get("intro_text") or "").strip())
    if text.startswith('"') and text.endswith('"') and len(text) > 1:
        text = text[1:-1].strip()
    if not text:
        _log_pinecone_intro_fallback(
            source="neutral_fallback_empty_intro_text",
            reason="match_audit_empty_intro_text",
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
            reason="match_audit_contact_or_followup_text_blocked",
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
            reason="match_audit_invalid_or_unsupported_match_claim",
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
    del user_message, category, slots
    if not listings:
        return ""
    return "Do any of these trailers interest you?"


def _has_contact(state: ChatbotState) -> bool:
    return bool(
        state.get("customer_full_name")
        and (state.get("customer_phone") or state.get("customer_email"))
    )


def _missing_contact_request(state: ChatbotState, reason: str) -> str:
    # Never interpolate the internal email_purpose (reason) into customer text —
    # it leaked strings like "customer asked to contact a person." Use a generic
    # phrase instead.
    del reason
    clean_reason = "your request"
    if state.get("customer_full_name"):
        return (
            f"To send {clean_reason} to our team, could you share either an email address "
            "or phone number? Sharing it is optional, and we can keep working on your trailer search."
        )
    if state.get("customer_phone") or state.get("customer_email"):
        return (
            f"To send {clean_reason} to our team, could you share your name? "
            "Sharing it is optional, and we can keep working on your trailer search."
        )
    return (
        f"To send {clean_reason} to our team, could you share your name and either an email "
        "address or phone number? Sharing them is optional, and we can keep working on your trailer search."
    )


_METADATA_FILTER_KEYS = {
    "length_ft",
    "width_ft",
    "height_ft",
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
_DYNAMIC_WIDTH_EXCLUDED_CATEGORIES = DYNAMIC_WIDTH_EXCLUDED_CATEGORIES
_FLATBED_DEFAULT_WIDTH_FT = "8 ft"
_CATEGORY_CLARIFICATION_SLOT = "category_clarification"
_GENERIC_CATEGORY_CHOICE_SLOT = "generic_category_choice"
_CATEGORY_FILTER_CONFIRMATION_SLOT = "category_filter_confirmation"
_CATEGORY_SUGGESTION_SLOT = "category_suggestion_confirmation"
_COMMON_CATEGORY_FILTERS = ("length_ft", "width_ft", "height_ft", "payload_lbs", "hitch_type")
_GENERIC_CATEGORY_QUESTION = "What type of trailer are you looking for?"
_GENERIC_HAUL_USE_SLOT = "generic_haul_use"
_MAKE_CATEGORY_CHOICE_SLOT = "make_category_choice"
_MAKE_GENERIC_LENGTH_SLOT = "trailer_length_ft"
_MAKE_GENERIC_PAYLOAD_SLOT = "haul_weight_lbs"
_MAKE_GENERIC_QUESTIONS = {
    _MAKE_GENERIC_LENGTH_SLOT: "What trailer length would you prefer?",
    _MAKE_GENERIC_PAYLOAD_SLOT: "What payload or weight capacity do you need?",
}
_PSEUDO_SLOT_DEFINITIONS: dict[str, dict[str, Any]] = {
    _CATEGORY_CLARIFICATION_SLOT: {
        "question": "",
        "answer_guidance": (
            "Store the user's clarification when a broad trailer phrase could map to more than one canonical "
            "category. Resolve the category from the clarification instead of asking the generic category question."
        ),
        "mapped_metadata_fields": [],
    },
    _GENERIC_CATEGORY_CHOICE_SLOT: {
        "question": _GENERIC_CATEGORY_QUESTION,
        "answer_guidance": "Store the trailer category the customer chooses. Accept supported trailer-type names only.",
        "mapped_metadata_fields": [],
    },
    _GENERIC_HAUL_USE_SLOT: {
        "question": "What will you be hauling or using the trailer for?",
        "answer_guidance": (
            "When current_category is unknown, store the free-form cargo, material, equipment, "
            "or use case the customer gives, including broad phrases after haul/carry/move such as "
            "'some heavy items', 'a car', 'equipment', or 'tools', as well as debris, a mower, "
            "a skid steer, hay, or furniture. Store any substantive direct haul/use answer unless "
            "the user refuses/skips, asks a counter-question, or answers another field. Do not store trailer category names here."
        ),
        "mapped_metadata_fields": [],
    },
    _MAKE_CATEGORY_CHOICE_SLOT: {
        "question": "",
        "answer_guidance": "Store the category the customer chooses from the presented make-specific category options.",
        "mapped_metadata_fields": [],
    },
    _DYNAMIC_WIDTH_SLOT: {
        "question": _DYNAMIC_WIDTH_QUESTION,
        "answer_guidance": (
            "Store the required trailer or cargo width when the answer includes a digit, number word, range, or approximation. "
            "If a cooperative answer has no usable number and is not a counter-question or another-field answer, skip width as no preference."
        ),
        "mapped_metadata_fields": ["width_ft"],
    },
}
_SLOT_METADATA_FILTER_MAP = {
    "base_category": ("subcategory",),
    "bin_size": ("length_ft",),
    "cargo_size": ("length_ft", "width_ft"),
    "haul_length_ft": ("length_ft",),
    "haul_weight_lbs": ("payload_lbs",),
    "item_or_trailer_width_ft": ("width_ft",),
    "max_price": ("max_price",),
    "payload_capacity": ("payload_lbs",),
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


def _apply_flatbed_default_width(
    category: str | None,
    slots: dict[str, Any],
    metadata_filters: dict[str, Any],
    defaulted_fields: set[str] | None = None,
) -> None:
    if str(category or "").strip().lower() != "flatbed":
        return
    if _has_width_requirement(slots, metadata_filters):
        return
    metadata_filters["width_ft"] = _FLATBED_DEFAULT_WIDTH_FT
    if defaulted_fields is not None:
        defaulted_fields.add("width_ft")
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
        return {_GENERIC_HAUL_USE_SLOT}
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


def _explicit_height_requested(message: str, awaiting_slot: str | None = None) -> bool:
    text = str(message or "").lower()
    if _has_dimension_shorthand(text):
        return True
    if re.search(
        r"\b\d+(?:\.\d+)?\s*(?:ft|feet|foot|'|in|inch|inches)\s*(?:(?:high|tall)\b|(?:\w+\s+){0,3}(?:side|sides|wall|walls)\b)",
        text,
    ):
        return True
    if re.search(
        r"\b(?:height|high|tall|side|sides|wall|walls)\b",
        text,
    ) and re.search(r"\d+(?:\.\d+)?", text):
        return True
    return bool(
        _slot_is_length_like(awaiting_slot)
        and re.search(r"\b\d+(?:\.\d+)?\s*(?:ft|feet|foot|'|in|inch|inches)\b", text)
        and re.search(r"\b(?:height|high|tall|side|sides|wall|walls)\b", text)
    )


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
    if key == "height_ft":
        return _explicit_height_requested(latest_message, awaiting_slot)
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


def _feature_has_message_evidence(feature: str, latest_message: str) -> bool:
    def concepts(text: str) -> set[str]:
        words = re.findall(r"[a-z0-9]+", str(text or "").lower())
        aliases = {
            "sliding": "slide",
            "slider": "slide",
            "slides": "slide",
            "gates": "gate",
            "doors": "door",
            "ramps": "ramp",
            "wheels": "wheel",
            "tires": "wheel",
            "tyres": "wheel",
            "allterrain": "offroad",
        }
        normalized: set[str] = set()
        for word in words:
            compact = word.replace("-", "")
            normalized.add(aliases.get(compact, compact))
        compact_text = re.sub(r"[^a-z0-9]+", "", str(text or "").lower())
        if "allterrain" in compact_text or "offroad" in compact_text:
            normalized.add("offroad")
        return normalized

    ignored = {
        "a",
        "an",
        "and",
        "for",
        "of",
        "the",
        "trailer",
        "trailers",
        "with",
    }
    feature_concepts = concepts(feature) - ignored
    message_concepts = concepts(latest_message)
    return bool(feature_concepts) and feature_concepts.issubset(message_concepts)


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


_CONFIDENT_FIELD_EXTRACTION = {"medium", "high"}
_LLM_ADJUDICATED_METADATA_KEYS = _METADATA_FILTER_KEYS - {"make"}


def _metadata_field_definitions() -> dict[str, str]:
    return {
        "length_ft": "Trailer/deck/cargo/bin length in feet. Accept any length phrasing: '12ft', '12 foot', '12 footer', 'twelve feet', etc. Store open-ended answers like 'any length' as 'null'",
        "width_ft": "Trailer/cargo/load width in feet. Never use a width value as length.Store open-ended answers like 'any width' as 'null'",
        "height_ft": "Side-wall or usable cargo height in feet. Triggered by: 'X inch sides', 'X ft walls', 'X ft sides', etc.",
        "payload_lbs": "Haul/carried weight or capacity in lbs. Not GVWR unless the user specifically says GVWR.",
        "max_price": "Maximum price or budget. Only set when the user gives a concrete upper limit.",
        "hitch_type": "Hitch preference: 'gooseneck' or 'bumper pull' only. Set only on explicit user selection.",
        "color": "Requested trailer color.",
        "subcategory": "Aluminum category only: the underlying trailer type (e.g. utility, equipment, enclosed).",
    }


def _slot_field_definitions(category: str | None) -> dict[str, dict[str, Any]]:
    definitions = {
        str(key): dict(value)
        for key, value in _PSEUDO_SLOT_DEFINITIONS.items()
    }
    if not category:
        return definitions
    spec = get_trailer_fields_as_dict(category)
    questions = spec.get("questions") or {}
    answer_guidance = spec.get("answer_guidance") or {}
    slots = (spec.get("required_slots") or []) + (spec.get("optional_slots") or [])
    for slot in slots:
        slot_key = str(slot)
        definitions[slot_key] = {
            "question": questions.get(slot_key) or "",
            "answer_guidance": answer_guidance.get(slot_key) or "",
            "mapped_metadata_fields": list(_SLOT_METADATA_FILTER_MAP.get(slot_key, ())),
        }
    return definitions


def _slot_definition(
    slot: str | None,
    *,
    category: str | None,
    questions_override: dict[str, str] | None = None,
    queued_question: str | None = None,
    make_category_options: list[str] | None = None,
) -> dict[str, Any]:
    slot_key = str(slot or "")
    definition = dict(_slot_field_definitions(category).get(slot_key) or {})
    if queued_question:
        definition["question"] = queued_question
    elif questions_override and slot_key in questions_override:
        definition["question"] = questions_override.get(slot_key) or definition.get("question") or ""
    if slot_key == _MAKE_CATEGORY_CHOICE_SLOT and make_category_options:
        definition["question"] = (
            "Which category should I use: "
            f"{', '.join(make_category_options)}?"
        )
    return {
        "question": str(definition.get("question") or "").strip(),
        "answer_guidance": str(definition.get("answer_guidance") or "").strip(),
        "mapped_metadata_fields": list(definition.get("mapped_metadata_fields") or []),
    }


def _last_assistant_question(messages: list[dict[str, Any]]) -> str:
    for item in reversed(messages or []):
        if str(item.get("role") or "").lower() == "assistant":
            return str(item.get("content") or "")
    return ""


def _queued_question_for_slot(pending_questions: list[QuestionItem], slot: str | None) -> str:
    slot_key = str(slot or "")
    for item in pending_questions or []:
        if str(item.get("slot") or "") == slot_key:
            return str(item.get("question") or "").strip()
    return ""


def _active_question_context(
    *,
    category: str | None,
    awaiting_slot: str | None,
    pending_questions: list[QuestionItem],
    questions_override: dict[str, str] | None = None,
    make_category_options: list[str] | None = None,
    messages: list[dict[str, Any]] | None = None,
) -> tuple[str | None, dict[str, Any]]:
    active_slot = str(awaiting_slot or "").strip()
    if not active_slot:
        last_assistant = _last_assistant_question(messages or [])
        for item in pending_questions or []:
            candidate = str(item.get("slot") or "").strip()
            candidate_question = str(item.get("question") or "").strip()
            if candidate and candidate_question and candidate_question == last_assistant:
                active_slot = candidate
                break
    if not active_slot:
        return None, {}
    queued_question = _queued_question_for_slot(pending_questions, active_slot)
    displayed_question = ""
    for item in reversed(messages or []):
        if str(item.get("role") or "") == "assistant":
            displayed_question = str(item.get("qualification_question") or "").strip()
            if displayed_question:
                break
    definition = _slot_definition(
        active_slot,
        category=category,
        questions_override=questions_override,
        queued_question=displayed_question or queued_question or _last_assistant_question(messages or []),
        make_category_options=make_category_options,
    )
    return active_slot, definition


def _slot_value_to_metadata_updates(slot: str, value: Any, category: str | None) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    for metadata_key in _SLOT_METADATA_FILTER_MAP.get(str(slot), ()):
        sanitized = _canonicalize_adjudicated_metadata(key=str(metadata_key), value=value, category=category)
        if sanitized:
            clean_key, clean_value = sanitized
            updates[clean_key] = clean_value
    return updates


def _question_turn_fallback_reply(question_text: str, latest_message: str) -> str:
    text = str(latest_message or "").strip()
    if not text:
        return ""
    if "?" in text:
        return "I can help with that, and I still need one detail to narrow this down."
    if re.search(r"\b(?:later|not sure|dont know|don't know|idk|maybe|whatever)\b", text, re.I):
        return "No problem."
    return ""


def _trim_fallback_active_slot_value(active_slot: str, value: Any) -> Any:
    text = str(value or "").strip()
    if active_slot != "haul_item" or not text:
        return value
    parts = re.split(r"\s*(?:[.!?]+|\band\b)\s+(?=(?:the\s+)?trailer\b|\bit\b|\bi\s+)", text, maxsplit=1, flags=re.I)
    return parts[0].strip() if parts and parts[0].strip() else value


def _preserve_latest_haul_item_phrase(active_slot: str, value: Any, latest_message: str) -> Any:
    text = str(value or "").strip()
    latest = str(latest_message or "").strip()
    if active_slot != "haul_item" or not text or not latest:
        return value

    first_phrase = _trim_fallback_active_slot_value(active_slot, latest)
    normalized_value = " ".join(text.lower().split())
    normalized_phrase = " ".join(first_phrase.lower().split())
    if normalized_phrase in {normalized_value, f"a {normalized_value}", f"an {normalized_value}", f"the {normalized_value}"}:
        return first_phrase
    return value


def _sanitize_question_turn_decision(data: dict[str, Any]) -> QuestionTurnDecision:
    confidence = str(data.get("confidence") or "low").lower()
    if confidence not in {"medium", "high"}:
        confidence = "low"
    email_action = str(data.get("email_action") or "none").strip()
    if email_action not in {
        "none",
        "send_non_sales_faq_email",
        "send_escalation_alert_email",
    }:
        email_action = "none"
    counter_topic = str(data.get("counter_question_topic") or "none").strip()
    if counter_topic not in {
        "trailer_categories", "hitch_types", "makes", "dimensions",
        "payload", "faq", "other", "none",
    }:
        counter_topic = "none"
    return QuestionTurnDecision(
        counter_question_topic=counter_topic,  # type: ignore[arg-type]
        answered_active_question=bool(data.get("answered_active_question")),
        no_preference_for_active_question=bool(data.get("no_preference_for_active_question")),
        search_now_requested=bool(
            data.get("search_now_requested") or data.get("skip_remaining_questions")
        ),
        skip_remaining_questions=bool(data.get("skip_remaining_questions")),
        active_slot_value=str(data.get("active_slot_value") or "").strip() or None,
        metadata_filters_update=dict(data.get("metadata_filters_update") or {}),
        slots_collected_update=dict(data.get("slots_collected_update") or {}),
        requested_non_metadata_features=[
            str(item).strip()
            for item in (data.get("requested_non_metadata_features") or [])
            if str(item).strip()
        ],
        email_action=email_action,  # type: ignore[arg-type]
        faq_category=str(data.get("faq_category") or "").strip() or None,
        faq_summary=str(data.get("faq_summary") or "").strip() or None,
        escalation_summary=str(data.get("escalation_summary") or "").strip() or None,
        unsupported_request=str(data.get("unsupported_request") or "").strip() or None,
        reply_to_user=str(data.get("reply_to_user") or "").strip(),
        rephrased_question=str(data.get("rephrased_question") or "").strip(),
        retry_slot=str(data.get("retry_slot") or "").strip() or None,
        confidence=confidence,
        reason=str(data.get("reason") or ""),
    )


def _sanitize_office_trailer_clarification_decision(data: dict[str, Any]) -> OfficeTrailerClarificationDecision:
    confidence = str(data.get("confidence") or "low").lower()
    if confidence not in {"medium", "high"}:
        confidence = "low"
    email_action = str(data.get("email_action") or "none").strip()
    if email_action not in {
        "none",
        "send_non_sales_faq_email",
        "send_escalation_alert_email",
    }:
        email_action = "none"
    resolved_category = str(data.get("resolved_category") or "").strip() or None
    if resolved_category not in {"Fiber", "Enclosed"}:
        resolved_category = None
    return OfficeTrailerClarificationDecision(
        answered_clarification=bool(data.get("answered_clarification")),
        resolved_category=resolved_category,  # type: ignore[arg-type]
        email_action=email_action,  # type: ignore[arg-type]
        faq_category=str(data.get("faq_category") or "").strip() or None,
        faq_summary=str(data.get("faq_summary") or "").strip() or None,
        escalation_summary=str(data.get("escalation_summary") or "").strip() or None,
        unsupported_request=str(data.get("unsupported_request") or "").strip() or None,
        reply_to_user=str(data.get("reply_to_user") or "").strip(),
        confidence=confidence,
        reason=str(data.get("reason") or ""),
    )


def _fallback_office_trailer_clarification_decision(
    *,
    latest_message: str,
    clarification_key: str | None,
    active_question: str,
) -> OfficeTrailerClarificationDecision:
    resolution = resolve_category_clarification_answer(latest_message, clarification_key)
    if resolution.category in {"Fiber", "Enclosed"}:
        return OfficeTrailerClarificationDecision(
            answered_clarification=True,
            resolved_category=resolution.category,
            confidence="high",
            reason="deterministic_office_trailer_clarification_answer",
        )
    return OfficeTrailerClarificationDecision(
        reply_to_user=_question_turn_fallback_reply(active_question, latest_message),
        confidence="low",
        reason="deterministic_office_trailer_clarification_unanswered",
    )


def _adjudicate_office_trailer_clarification_turn(
    *,
    state: ChatbotState,
    active_question: str,
    latest_message: str,
) -> QuestionTurnDecision:
    clarification_key = str(state.get("category_clarification_key") or "").strip() or None
    if not os.getenv("OPENAI_API_KEY"):
        fallback = _fallback_office_trailer_clarification_decision(
            latest_message=latest_message,
            clarification_key=clarification_key,
            active_question=active_question,
        )
    else:
        context = {
            "latest_user_message": latest_message,
            "recent_messages": (state.get("messages") or [])[-8:],
            "active_question": active_question,
            "clarification_key": clarification_key,
            "valid_outcomes": {
                "Fiber": "Fiber/telecom work, splicing, telecom setup, fiber optic use, cooldown trailer for fiber crews, or similar.",
                "Enclosed": "General office trailer, office work, office use, office only, jobsite office, site office, mobile office, workspace, admin office, crew office, or similar.",
            },
            "tool_rules": {
                "send_non_sales_faq_email": "Use for contact/human help, financing, trade-in, service/parts, or store/location info.",
                "send_escalation_alert_email": "Use when the customer asks TrailerPlace/the team to perform an unsupported real-world action: hold/reserve; send a reminder/follow-up; create/send a quote, invoice, contract, application, or paperwork; call/text/email them; schedule a call, meeting, appointment, delivery, pickup, service, or installation; or make a future timing commitment. General information questions do not qualify.",
            },
        }
        try:
            decision = _office_trailer_clarification_llm().invoke(
                [
                    SystemMessage(
                        content=(
                            "You evaluate the user's answer to a specific office trailer clarification question. "
                            "Return structured data only.\n"
                            "Decide whether the customer answered whether this should route to Fiber or to Enclosed.\n"
                            "Set answered_clarification=true only when the user clearly answered the clarification.\n"
                            "Use resolved_category=Fiber when they mean fiber/telecom work.\n"
                            "Use resolved_category=Enclosed when they mean a more general office trailer or office work.\n"
                            "If the user asks a counter-question or says something unrelated, do not resolve a category. "
                            "Provide a short reply_to_user instead.\n"
                            "You may set email_action to send_non_sales_faq_email or send_escalation_alert_email using the existing rules.\n"
                            "Do not invent a category if the user did not answer the clarification. "
                            "Use medium or high confidence only when clearly supported."
                        )
                    ),
                    HumanMessage(content=_safe_json(context)),
                ]
            )
            fallback = _sanitize_office_trailer_clarification_decision(
                _model_dump(decision) if isinstance(decision, BaseModel) else {}
            )
        except Exception:
            logger.exception("Office trailer clarification adjudicator failed; using fallback")
            fallback = _fallback_office_trailer_clarification_decision(
                latest_message=latest_message,
                clarification_key=clarification_key,
                active_question=active_question,
            )

    return QuestionTurnDecision(
        answered_active_question=bool(
            fallback.answered_clarification
            and fallback.resolved_category in {"Fiber", "Enclosed"}
            and fallback.confidence in _CONFIDENT_CLASSIFICATIONS
        ),
        active_slot_value=fallback.resolved_category,
        email_action=fallback.email_action,
        faq_category=fallback.faq_category,
        faq_summary=fallback.faq_summary,
        escalation_summary=fallback.escalation_summary,
        unsupported_request=fallback.unsupported_request,
        reply_to_user=fallback.reply_to_user,
        confidence=fallback.confidence,
        reason=fallback.reason or "office_trailer_clarification",
    )


def _skip_remaining_questions_requested(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:don['\u2019]?t|do\s+not|won['\u2019]?t|will\s+not)\s+"
            r"(?:want\s+to\s+)?answer\b[^.?!]*(?:more|any\s+more|additional|remaining)\s+questions?\b"
            r"|\b(?:don['\u2019]?t|do\s+not|stop)\b[^.?!]*\bask(?:ing)?\b[^.?!]*"
            r"(?:more|any\s+more|additional|remaining)\s+questions?\b"
            r"|\b(?:skip|stop)\b[^.?!]*(?:the\s+)?(?:rest|remaining|questions?)\b"
            r"|\b(?:that['\u2019]?s|that\s+is)\s+enough\s+questions?\b",
            text or "",
            re.I,
        )
    )


def _immediate_search_requested(text: str) -> bool:
    return bool(
        _skip_remaining_questions_requested(text)
        or re.search(
            r"\b(?:show|let\s+me\s+see|give|find|search|pull\s+up)\b[^.?!]*"
            r"(?:results?|trailers?|listings?|options?|inventory|what\s+you\s+have)\b"
            r"|\bwhat\b[^.?!]*(?:trailers?|listings?|options?)\b[^.?!]*"
            r"(?:have|available|in\s+stock)\b",
            text or "",
            re.I,
        )
    )


def _fallback_question_turn_decision(
    *,
    state: ChatbotState,
    category: str | None,
    active_slot: str,
    active_question: str,
    active_definition: dict[str, Any],
    latest_message: str,
    pending_questions: list[QuestionItem],
    make_category_options: list[str],
) -> QuestionTurnDecision:
    del active_definition
    text = str(latest_message or "").strip()
    if not text:
        return QuestionTurnDecision(reason="empty_latest_message", confidence="low")

    if category and _immediate_search_requested(text):
        skip_all = _skip_remaining_questions_requested(text)
        return QuestionTurnDecision(
            search_now_requested=True,
            skip_remaining_questions=skip_all,
            confidence="high",
            reason="deterministic_immediate_search_request",
        )

    if active_slot == _MAKE_CATEGORY_CHOICE_SLOT:
        chosen_category = _category_from_choice(text, make_category_options)
        if chosen_category:
            return QuestionTurnDecision(
                answered_active_question=True,
                active_slot_value=chosen_category,
                confidence="high",
                reason="resolved_make_category_choice",
            )
        if _classify_make_category_no_preference(
            user_message=text,
            make_category_options=make_category_options,
            metadata_filters=state.get("metadata_filters_collected") or {},
            slots=state.get("slots_collected") or {},
        ):
            return QuestionTurnDecision(
                no_preference_for_active_question=True,
                confidence="high",
                reason="make_category_choice_no_preference",
            )
        return QuestionTurnDecision(
            reply_to_user=_question_turn_fallback_reply(active_question, text),
            reason="unanswered_make_category_choice",
            confidence="low",
        )

    if active_slot == _GENERIC_CATEGORY_CHOICE_SLOT:
        chosen_category = resolve_category_from_text(text).category
        if chosen_category:
            return QuestionTurnDecision(
                answered_active_question=True,
                active_slot_value=chosen_category,
                confidence="high",
                reason="resolved_generic_category_choice",
            )
        if _classify_generic_category_no_preference(
            user_message=text,
            metadata_filters=state.get("metadata_filters_collected") or {},
            slots=state.get("slots_collected") or {},
        ):
            return QuestionTurnDecision(
                no_preference_for_active_question=True,
                confidence="high",
                reason="generic_category_choice_no_preference",
            )
        return QuestionTurnDecision(
            reply_to_user=_question_turn_fallback_reply(active_question, text),
            reason="unanswered_generic_category_choice",
            confidence="low",
        )

    if active_slot == _CATEGORY_CLARIFICATION_SLOT and str(state.get("category_clarification_key") or "").strip() == "office_trailer_use":
        return _adjudicate_office_trailer_clarification_turn(
            state=state,
            active_question=active_question,
            latest_message=text,
        )

    preference_decision = classify_no_preference(
        category=category,
        user_message=text,
        awaiting_slot=active_slot,
        pending_questions=pending_questions,
        slots_collected=state.get("slots_collected") or {},
        metadata_filters_collected=state.get("metadata_filters_collected") or {},
        allowed_category_slots=sorted(_category_slots(category)),
        active_question=active_question,
    )
    pref_data = _model_dump(preference_decision)
    if pref_data.get("has_no_preference") and pref_data.get("confidence") in _CONFIDENT_PREFERENCE_NULL:
        return QuestionTurnDecision(
            no_preference_for_active_question=True,
            confidence="high",
            reason=str(pref_data.get("reason") or "active_question_no_preference"),
        )

    temp_slots = dict(state.get("slots_collected") or {})
    temp_metadata = dict(state.get("metadata_filters_collected") or {})
    extracted_slots, _extracted_metadata, extracted_features = _apply_explicit_filter_extraction(
        state=state,
        category=category,
        slots=temp_slots,
        metadata_filters=temp_metadata,
        latest_message=text,
        awaiting_slot=active_slot,
        apply_slot_updates=True,
    )
    active_value = extracted_slots.get(active_slot)
    if active_value in (None, ""):
        active_value = _slot_updates_from_metadata(category, temp_metadata).get(active_slot)
    if active_value not in (None, ""):
        active_value = _trim_fallback_active_slot_value(active_slot, active_value)
        active_value = _preserve_latest_haul_item_phrase(active_slot, active_value, text)
        return QuestionTurnDecision(
            answered_active_question=True,
            active_slot_value=str(active_value).strip(),
            metadata_filters_update={
                key: value
                for key, value in temp_metadata.items()
                if key not in (state.get("metadata_filters_collected") or {})
                or (state.get("metadata_filters_collected") or {}).get(key) != value
            },
            slots_collected_update={
                key: value
                for key, value in temp_slots.items()
                if key not in (state.get("slots_collected") or {})
                or (state.get("slots_collected") or {}).get(key) != value
            },
            requested_non_metadata_features=extracted_features,
            confidence="high",
            reason="fallback_explicit_extraction_answered",
        )

    return QuestionTurnDecision(
        reply_to_user=_question_turn_fallback_reply(active_question, text),
        requested_non_metadata_features=extracted_features,
        confidence="low",
        reason="fallback_unanswered_active_question",
    )


def _reconcile_category_transition(
    *,
    state: ChatbotState,
    persisted_category: str,
    proposed_category: str,
    active_slot: str | None,
    active_question: str,
    latest_message: str,
) -> CategoryTransitionDecision:
    if not os.getenv("OPENAI_API_KEY"):
        return CategoryTransitionDecision(
            final_category=persisted_category,
            reason="category_transition_llm_unavailable",
        )
    context = {
        "persisted_category": persisted_category,
        "mind_proposed_category": proposed_category,
        "active_slot": active_slot,
        "active_question": active_question,
        "latest_user_message": latest_message,
        "recent_messages": (state.get("messages") or [])[-8:],
    }
    try:
        return _category_transition_llm().invoke(
            [
                SystemMessage(content=(
    "You verify a proposed trailer-category transition before any category state is changed. "
    "Return structured data only.\n\n"

    "## APPROVE the transition when the latest message:\n"
    "- Explicitly names a different canonical category with shopping intent, even with soft phrasing:\n"
    "  'I am also looking for a dump trailer', 'now I need a dump trailer', 'I want a dump trailer too',\n"
    "  'actually I want Equipment', 'Equipment instead', 'let's do Dump', 'switch to Enclosed'.\n"
    "- Uses additive or sequential language ('also', 'now', 'too', 'as well', 'next') with a new category name.\n"
    "- Clearly abandons the current category: 'forget Livestock, I want Dump', 'never mind, show me flatbeds'.\n\n"

    "## REJECT the transition when the latest message:\n"
    "- Uses a category-like word as cargo/haul context, not as a trailer request:\n"
    "  'haul assorted equipment' (cargo) while on Utility must NOT switch to Equipment.\n"
    "- Mentions a category incidentally: comparisons, recommendations, informational questions.\n"
    "- Is ambiguous with no clear shopping intent for the new category.\n"
    "- On any genuine doubt, reject and return persisted_category.\n\n"

    "## KEY DISTINCTION\n"
    "Shopping intent for a new trailer type = APPROVE.\n"
    "Category word used as cargo description or incidental mention = REJECT.\n"
    "Soft or additive phrasing ('also', 'now', 'too') + canonical category name = APPROVE.\n\n"

    "## TRAILER TYPES & SYNONYM MAPPING\n"
    f"{category_prompt_block()}\n\n"

    "## TRAILER BRANDS / MAKES\n"
    "(Informational only — do NOT infer category from make)\n"
    f"{make_prompt_block()}"
)),
                HumanMessage(content=f"Verify this category transition:\n{_safe_json(context)}"),
            ]
        )
    except Exception:
        logger.exception("category_transition_llm_failed; preserving persisted category")
        return CategoryTransitionDecision(
            final_category=persisted_category,
            reason="category_transition_llm_failed",
        )


def _verify_make_candidate(
    *,
    candidate_make: str,
    candidate_match_type: str,
    latest_message: str,
    category: str | None,
    recent_messages: list[dict[str, Any]],
) -> MakeVerificationDecision:
    if not os.getenv("OPENAI_API_KEY"):
        return MakeVerificationDecision(reason="make_verification_llm_unavailable")
    context = {
        "deterministic_candidate": candidate_make,
        "candidate_match_type": candidate_match_type,
        "latest_user_message": latest_message,
        "current_category": category,
        "recent_messages": recent_messages[-6:],
    }
    # safe_invoke centralizes the fallback: any error (including a pydantic
    # ValidationError from an off-schema enum) rejects the candidate rather than
    # crashing the turn.
    return safe_invoke(
        _make_verification_llm(),
        [
            SystemMessage(content=(
                "Verify whether the user explicitly requested the deterministic manufacturer candidate. "
                "Return structured data only. Approve only when the latest message uses the candidate as a "
                "trailer brand/make. Reject lexical collisions and ordinary descriptions: 'general cargo' is "
                "not Cargo Craft, 'diamond plate' is not Diamond C, and generic iron/aluminum wording is not a make. "
                "verified_make must equal the supplied candidate when approved. On uncertainty, reject.\n\n"
                "## TRAILER TYPES & SYNONYM MAPPING\n"
                "(Map spelling mistakes and synonyms to canonical categories)\n"
                f"{category_prompt_block()}\n\n"
                "## TRAILER BRANDS / MAKES\n"
                "(Use for informational answers only — do NOT infer category from make)\n"
                f"{make_prompt_block()}"
            )),
            HumanMessage(content=f"Verify this make candidate:\n{_safe_json(context)}"),
        ],
        fallback=MakeVerificationDecision(reason="make_verification_llm_failed"),
        role="make_verification",
    )


def _reconcile_active_question_turn(
    *,
    state: ChatbotState,
    category: str | None,
    active_slot: str,
    active_question: str,
    latest_message: str,
    mind_decision: dict[str, Any],
    adjudicator_decision: QuestionTurnDecision,
    extracted_slots: dict[str, Any],
    extracted_metadata: dict[str, Any],
    extracted_features: list[str],
    no_preference_decision: PreferenceNullDecision,
) -> QuestionTurnDecision:
    """Use an LLM as the final semantic authority for an active Q&A turn."""
    if not os.getenv("OPENAI_API_KEY"):
        return adjudicator_decision
    context = {
        "latest_user_message": latest_message,
        "recent_messages": (state.get("messages") or [])[-8:],
        "persisted_category": category,
        "active_slot": active_slot,
        "active_question": active_question,
        "existing_slots": state.get("slots_collected") or {},
        "existing_metadata_filters": state.get("metadata_filters_collected") or {},
        "mind_proposal": mind_decision,
        "question_adjudicator_proposal": _model_dump(adjudicator_decision),
        "field_extractor_proposal": {
            "slots_collected_update": extracted_slots,
            "metadata_filters_update": extracted_metadata,
            "requested_non_metadata_features": extracted_features,
        },
        "no_preference_classifier_proposal": _model_dump(no_preference_decision),
    }
    try:
        return _active_turn_reconciler_llm().invoke(
            [
                SystemMessage(content=(
                    "You are the final semantic state-transition authority for one active trailer Q&A turn. "
                    "Reconcile the mind, question adjudicator, and field extractor. Return QuestionTurnDecision only.\n\n"
                    "1. Preserve the persisted category unless the user explicitly asks to switch trailer type. "
                    "Cargo words such as equipment, cars, livestock, cargo, or debris do not change category.\n"
                    "2. Free-text slots (haul_item, haul_material, vehicle_type, use_case and similar) accept any "
                    "substantive direct answer, however broad or informal. Preserve its meaning. Reject only an "
                    "unrelated counter-question, explicit refusal/skip, or content that answers a different field.\n"
                    "3. Numeric slots require a digit or an unambiguous number written in words. Accept ranges and "
                    "approximations. If there is no usable number, set no_preference_for_active_question=true and "
                    "do not invent or retry a value.\n"
                    "3a. Width specifically requires a numeric measurement. 'Flexible', 'normal', 'standard', "
                    "'whatever fits', and 'no specific measurement' mean no preference: return no active value and "
                    "no width_ft update.\n"
                    "4. For choice/preference slots, 'either', 'anything standard', 'whatever works', and flexible "
                    "wording mean no preference.\n"
                    "4a. For every other constrained field, accept a recognizable field value; otherwise a "
                    "cooperative vague answer means no preference, not rejection and not a retry.\n"
                    "5. Never overwrite an existing slot with an answer to another slot. Merge valid extractor "
                    "updates only when explicitly supported by the latest message.\n"
                    "6. Compound dimensions must be separated in metadata updates: '16 by 7 feet' means "
                    "length_ft='16 ft' and width_ft='7 ft'; never copy the whole phrase into both fields.\n"
                    "6a. For active cargo_size, one usable cargo length fully answers the slot; width and height "
                    "are optional. 'About 18 feet long' means active_slot_value='18 ft', length_ft='18 ft'. "
                    "'18 by 8 feet; height is not important' means active_slot_value='18 ft × 8 ft', "
                    "length_ft='18 ft', width_ft='8 ft'. Never repeat cargo_size for missing width or height.\n"
                    "7. A make requires an explicitly named manufacturer. Generic 'cargo' never means Cargo Craft.\n"
                    "8. search_now_requested and skip_remaining_questions are semantic intent decisions. Set them "
                    "only when the user's message actually expresses those intents.\n"
                    "9. Prefer the best-supported interpretation across the three proposals; confidence reflects "
                    "the evidence. Treat the no-preference classifier as advisory evidence, not an automatic override. "
                    "The result is authoritative.\n"
                    "9a. A counter-question is unresolved, not no preference. When counter_question_topic is not "
                    "'none', set answered_active_question=false and no_preference_for_active_question=false, do not "
                    "supply or update the active slot, answer the counter-question, and re-ask the same active question. "
                    "Counter-questions count toward the existing unanswered-attempt limit.\n"
                    "10. For every numeric range, choose and store only the smallest stated value. Examples: "
                    "'15 to 18 ft' becomes '15 ft'; '5,000-10,000 lbs' becomes '5000 lbs'.\n"
                    "11. 'Either A or B', 'either is fine', and equivalent wording mean no preference for a fixed-choice "
                    "field; store neither option.\n"
                    "12. Roll Off bin_size maps its numeric value directly to trailer length_ft for Pinecone. "
                    "'15 yd' means length_ft='15 ft', never 45 ft; use the smallest number in a range.\n"
                    "13. Explicit requests to stop/skip questions or show results are supported search controls, "
                    "never email actions; set search_now_requested=true and, for stop/skip requests, "
                    "skip_remaining_questions=true. Otherwise use send_escalation_alert_email when the user "
                    "asks the business to perform an unsupported real-world action: for example reserve/holding a trailer or an item, send a reminder or "
                    "future follow-up, create/send a quote/invoice/paperwork, call/text/email them, schedule a call/"
                    "meeting/appointment/delivery/pickup/service, or make a future timing commitment. Such a request "
                    "does not answer or skip the active slot. Do not escalate ordinary informational questions."
                )),
                HumanMessage(content=f"Reconcile this active turn:\n{_safe_json(context)}"),
            ]
        )
    except Exception:
        logger.exception("active_turn_reconciler_llm_failed; using adjudicator proposal")
        return adjudicator_decision


def _adjudicate_active_question_turn(
    *,
    state: ChatbotState,
    category: str | None,
    active_slot: str,
    active_question: str,
    active_definition: dict[str, Any],
    latest_message: str,
    pending_questions: list[QuestionItem],
    make_category_options: list[str],
) -> QuestionTurnDecision:
    if active_slot == _CATEGORY_CLARIFICATION_SLOT and str(state.get("category_clarification_key") or "").strip() == "office_trailer_use":
        return _adjudicate_office_trailer_clarification_turn(
            state=state,
            active_question=active_question,
            latest_message=latest_message,
        )
    if not os.getenv("OPENAI_API_KEY"):
        fallback = _fallback_question_turn_decision(
            state=state,
            category=category,
            active_slot=active_slot,
            active_question=active_question,
            active_definition=active_definition,
            latest_message=latest_message,
            pending_questions=pending_questions,
            make_category_options=make_category_options,
        )
        return fallback
    context = {
        "latest_user_message": latest_message,
        "recent_messages": (state.get("messages") or [])[-8:],
        "current_category": category,
        "active_slot": active_slot,
        "active_question": active_question,
        "active_slot_definition": active_definition,
        "pending_questions": pending_questions,
        "slot_definitions": _slot_field_definitions(category),
        "metadata_field_definitions": _metadata_field_definitions(),
        "existing_slots_collected": state.get("slots_collected") or {},
        "existing_metadata_filters_collected": state.get("metadata_filters_collected") or {},
        "make_category_options": make_category_options,
        "supported_hitch_types": ["Bumper Pull", "Gooseneck"],
        "unanswered_count_before_this_turn": int(
            (state.get("active_question_attempts") or {}).get(active_slot)
            or state.get("active_question_unanswered_count")
            or 0
        ),
    }
    try:
        decision = _question_turn_adjudicator_llm().invoke(
            [
            SystemMessage(
                content=(
                    f"{TRAILERPLACE_KNOWLEDGE_SECTION}\n\n"
                    "You evaluate the latest user turn while a trailer qualification question is active. "
                    "Return structured data only.\n\n"

                    "## HARD RULES (apply before anything else)\n"
                    "1. CRITICAL HAUL-ITEM RULE: CATEGORY ≠ HAUL ITEM. 'I want an equipment trailer' sets category only — not haul_item/generic_haul_use. "
                    "   Accept a category-like term as haul cargo only when explicitly framed as cargo or as a direct answer to the active haul question.\n"
                    "2. HITCH TYPES ONLY IN hitch_type. Gooseneck and Bumper Pull are hitch configurations only. "
                    "   Never use either as active_slot_value for a category/base-category question, make, or any other slot. "
                    "   If stated while a different question is active, store in metadata_filters_update.hitch_type and leave the active question unanswered unless the same message also answers it. "
                    "   Informational hitch questions ('what hitches do you have?', 'which is better?') → no hitch_type update.\n"
                    "3. NO MAKE INFERENCE. Makes are for user education only — never infer category from make.\n\n"

                    "## FOUR MUTUALLY EXCLUSIVE RESPONSE STATES\n"
                    "Every user turn must be classified into exactly one primary state. Do not combine:\n"
                    "| State | Field | When to set |\n"
                    "|---|---|---|\n"
                    "| Answered | answered_active_question=true | Message clearly answers the active slot |\n"
                    "| No preference | no_preference_for_active_question=true | User says no preference for the active slot only |\n"
                    "| Search now | search_now_requested=true | User wants to see inventory/listings immediately |\n"
                    "| Skip all | skip_remaining_questions=true | User refuses further questions (always also sets search_now_requested=true) |\n"
                    "These are mutually exclusive for the primary intent. A message may combine answered + search_now (e.g. 'gravel, but just show me trailers now').\n\n"

                    "## IMMEDIATE SEARCH / STOP QUALIFICATION\n"
                    "Set search_now_requested=true when the user clearly wants to see matching inventory now:\n"
                    "- 'Show me the results / trailers / options'\n"
                    "- 'What options do you have?' / 'What do you have available?'\n"
                    "- 'Let me see what's available' / 'Search with what I already gave you'\n"
                    "- 'What dump trailers do you have?' / 'Show me dump trailers'\n\n"
                    "Also set skip_remaining_questions=true when the user refuses further qualification:\n"
                    "- 'I don't want to answer more questions'\n"
                    "- 'That's enough questions / Skip the rest'\n"
                    "- 'Just show me what you have / Use whatever you already have'\n\n"
                    "Rules:\n"
                    "- search_now_requested=true is only valid when current_category is already resolved.\n"
                    "- Preserve all valid slots and metadata filters already collected.\n"
                    "- Do not fabricate missing slot values.\n"
                    "- Do not set answered_active_question=true unless the message separately answers the active slot.\n"
                    "- Do not place the search request into active_slot_value.\n"
                    "- When search_now_requested=true: leave reply_to_user and rephrased_question empty.\n"
                    "- skip_remaining_questions=true must always also set search_now_requested=true.\n\n"

                    "## SEARCH REQUEST VS INFORMATIONAL QUESTION\n"
                    "| User says | Result |\n"
                    "|---|---|\n"
                    "| 'Show me dump trailers' | search_now_requested=true |\n"
                    "| 'What dump trailers do you have available?' | search_now_requested=true |\n"
                    "| 'What options do you have for dump trailers?' | search_now_requested=true |\n"
                    "| 'What trailer types do you carry?' | counter_question_topic=trailer_categories, search_now_requested=false |\n"
                    "| 'Can you explain the different trailer types?' | counter_question_topic=trailer_categories, search_now_requested=false |\n\n"

                    "## OPEN-ENDED / VAGUE ANSWERS\n"
                    "- Dimensions (length_ft, width_ft, height_ft) and numeric fields (payload_lbs, max_price): "
                    "  vague answers ('any', 'doesn't matter', 'no preference', 'no idea') → null. Store concrete values only.\n"
                    "- hitch_type: store only 'gooseneck' or 'bumper pull'. Any non-specific answer → null.\n"
                    "- Haul/use fields (generic_haul_use, haul_item, haul_material): store whatever the user says, even if broad — "
                    "  'anything', 'all types of material', 'various equipment'. Capture the phrase as-is.\n\n"

                    "## AUTHORITATIVE LOOSE-ANSWER POLICY\n"
                    "- Numeric fields (weight, payload, length, width, height, capacity, crew size) require a digit "
                    "or an unambiguous number written in words. Accept ranges and approximations; for every range "
                    "store only its smallest stated value ('15 to 18 ft' -> '15 ft'). If no usable "
                    "number is present, set no_preference_for_active_question=true; never retry or invent a value.\n"
                    "- For width, 'flexible', 'normal', 'standard', 'whatever fits', and 'no specific measurement' "
                    "mean no preference. Return no active-slot value and no width_ft update.\n"
                    "- Roll Off bin_size is a search proxy for trailer length: copy its chosen numeric value directly "
                    "to length_ft ('15 yd' -> length_ft='15 ft'), without converting yards to feet.\n"
                    "- Free-text haul/use fields accept any substantive direct answer, including 'random things', "
                    "'general cargo', and 'assorted equipment'. Reject only explicit refusal/skip, a counter-question, "
                    "or content answering a different field.\n"
                    "- 'Either', 'whatever works', 'standard', and flexible wording mean no preference for a choice slot.\n"
                    "- Example: 'Either bumper pull or gooseneck is fine' means no preference: store neither hitch.\n"
                    "- For every other constrained field, accept a recognizable field value; otherwise a cooperative "
                    "vague answer means no preference, not rejection and not a retry.\n"
                    "- Incidental cargo words never change an already selected category during active Q&A.\n\n"

                    "## ACTIVE QUESTION EVALUATION\n"
                    "- Treat the active slot definition and question text as authoritative.\n"
                    "- answered_active_question=true + active_slot_value when the question is clearly answered.\n"
                    "- active_slot_value must be a scalar string, never an object or array. "
                    "For composite dimensions use a string such as '20 ft × 8 ft × 7 ft' and also put "
                    "individual dimensions in metadata_filters_update.\n"
                    "- For active cargo_size, one usable length fully answers the question. Width and height are "
                    "optional. '18 by 8 feet; height is not important' returns active_slot_value='18 ft × 8 ft', "
                    "length_ft='18 ft', width_ft='8 ft'; 'about 18 feet long' returns active_slot_value='18 ft'.\n"
                    "- For multiple items or amenities, use one comma-separated string. "
                    "For yes/no fields, use the strings 'yes' or 'no', not JSON booleans.\n"
                    "- no_preference_for_active_question=true when the user says no preference for the active slot only — this does NOT trigger a search.\n"
                    "  Example: 'Any length is fine' → no_preference_for_active_question=true, search_now_requested=false.\n"
                    "- If unanswered: do not fabricate a value. Provide reply_to_user addressing their comment/question.\n"
                    "- Interpret answers semantically: '14 footer', 'twenty-foot', '14' all answer a length question.\n"
                    "- Use history only to resolve an explicit reference — never copy an old value as a new answer.\n\n"

                    "## COMBINED-INTENT EXAMPLES\n"
                    "Active: 'What material will you haul?' | User: 'Gravel, but just show me the trailers now.'\n"
                    "→ answered_active_question=true, active_slot_value='gravel', search_now_requested=true, skip_remaining_questions=false\n\n"
                    "Active: 'What material will you haul?' | User: 'I don't want more questions. Show me what's available.'\n"
                    "→ answered_active_question=false, no_preference_for_active_question=false, search_now_requested=true, skip_remaining_questions=true\n\n"
                    "Active: 'What length do you need?' | User: 'Any length is fine.'\n"
                    "→ no_preference_for_active_question=true, search_now_requested=false\n\n"
                    "Active: 'What length do you need?' | User: 'What kinds of dump trailers are available?'\n"
                    "→ search_now_requested=true, answered_active_question=false\n\n"

                    "## REPHRASED QUESTION (first unanswered reply only)\n"
                    "- Provide rephrased_question: a natural rewording of the same active slot — do not add requirements or change meaning.\n"
                    "- Structure: answer the counter-question as a statement → brief contextual bridge → naturally rephrased active question.\n"
                    "- The rephrased question must appear exactly once, as the final sentence of reply_to_user.\n"
                    "- Never ask a follow-up about the counter-question topic. The only question in reply_to_user is the rephrased active slot question.\n"
                    "- When unanswered_count_before_this_turn >= 1: leave rephrased_question empty (app skips the question).\n"
                    "- When search_now_requested=true: leave rephrased_question and reply_to_user empty.\n\n"

                    "## COUNTER-QUESTION HANDLING\n"
                    "A counter-question never answers or skips the active slot and never means no preference. "
                    "Set answered_active_question=false and no_preference_for_active_question=false, preserve the "
                    "active slot, answer the counter-question, then repeat the active question. It counts as an "
                    "unanswered attempt; after the existing two-attempt limit, answer it and let the app advance.\n"
                    "Classify counter_question_topic by the noun being asked about. 'Type' alone ≠ trailer category.\n"
                    "| User asks | counter_question_topic | reply_to_user content |\n"
                    "|---|---|---|\n"
                    "| 'Which trailer types do you carry?' | trailer_categories | List all canonical trailer categories |\n"
                    "| 'Which hitch types do you carry?' | hitch_types | 'Bumper Pull and Gooseneck' — never list trailer categories |\n"
                    "| 'Which makes do you carry?' | makes | List canonical inventory makes |\n"
                    "If the user asks what types/makes are available, list them directly — do not say 'I can help with that'.\n\n"

                    "## EMAIL ACTIONS (priority over active question)\n"
                    "An email-triggering request interrupts but never answers or skips the active qualification "
                    "question. Set answered_active_question=false and no_preference_for_active_question=false. "
                    "Provide a natural reply and let the app preserve or advance the question flow.\n"
                    "- send_non_sales_faq_email: financing, trade-in, service/parts, store/location, human contact.\n"
                    "- send_escalation_alert_email: requests that the business perform an unsupported real-world action "
                    "— hold/reserve; send a reminder/follow-up; create/send a quote, invoice, contract, application, "
                    "or paperwork; call/text/email the customer; schedule a call, meeting, appointment, delivery, "
                    "pickup, service, or installation; or make a future timing commitment.\n"
                    "- Escalate action requests, not ordinary questions. 'Remind me tomorrow' and 'schedule a call at "
                    "3 PM' escalate; 'what time are you open?' and 'how do reservations work?' do not.\n"
                    "- Never set send_interested_listing_email during active qualification.\n"
                    "- Do not set any email action for broad catalogue browsing.\n"
                    "Examples: 'How do I contact you?' → send_non_sales_faq_email, faq_category=contact_human. "
                    "'Can you call me tomorrow?' → send_escalation_alert_email.\n\n"

                    "## ALUMINUM BASE-CATEGORY RULES\n"
                    "When current_category=Aluminum and active_slot=base_category:\n"
                    "- A plain canonical category reply answers the active question. Set answered_active_question=true, active_slot_value=canonical category.\n"
                    "  Examples: 'utility' → Utility; 'an enclosed one' → Enclosed; 'I need it for equipment' → Equipment.\n"
                    "- Do NOT return Aluminum as active_slot_value. Do NOT put the answer in requested_non_metadata_features.\n"
                    "- Switch away from Aluminum ONLY on explicit rejection: 'not aluminum', 'I don't want aluminum anymore', 'instead of aluminum make it enclosed'.\n"
                    "- Multiple possible base categories with no clear preference → answered_active_question=false, ask user to choose one.\n"
                    "- Unrecognized reply → leave unanswered, let the app clarify.\n\n"

                    "## CATEGORY HANDLING DURING ACTIVE QnA\n"
                    "- User may change category mid-flow. If so, set category_update to the new canonical category and preserve any still-valid field updates.\n"
                    "- When current_category is unknown and the user gives only a use case/haul item: offer 2–3 suitable canonical trailer types each with a one-line description, then ask which they prefer. Do not ask dimensions/features before type is chosen.\n"
                    "- Map spelling mistakes and synonyms to canonical categories.\n\n"

                    "## GENERAL EXTRACTION RULES\n"
                    "- You may also extract metadata_filters_update, slots_collected_update, and requested_non_metadata_features from the same message.\n"
                    "- requested_non_metadata_features: user-requested equipment/config/features not covered by metadata or slots. Never infer from listing text.\n"
                    "- Do not invent updates. Use medium or high confidence only when clearly supported.\n"
                    "- Use recent messages and the previous assistant question only to interpret the latest message — not as new updates.\n\n"

                    "## CUSTOMER-FACING MARKDOWN FORMAT\n"
                    "- Use short paragraphs for ordinary answers.\n"
                    "- For multiple options, put a blank line before the list.\n"
                    "- Format each option as `- **Name** — short practical description`.\n"
                    "- Never place the first bullet inline with introductory prose.\n"
                    "- Put a blank line after the list before a closing question.\n"
                    "- Do not mix inline options, bullets, and numbered lists in one response.\n"
                    "- Keep a single direct answer as prose rather than forcing a list.\n\n"

                    "## TRAILER TYPES AND MAPPING TERMS\n"
                    f"{category_prompt_block()}\n\n"
                    "## TRAILER BRANDS / MAKES (education only — never infer category)\n"
                    f"{make_prompt_block()}"
                )
            ),
                HumanMessage(content=_safe_json(context)),
            ]
        )
        return _sanitize_question_turn_decision(
            _model_dump(decision) if isinstance(decision, BaseModel) else {}
        )
    except Exception:
        logger.exception("Question turn adjudicator failed; using fallback")
        fallback = _fallback_question_turn_decision(
            state=state,
            category=category,
            active_slot=active_slot,
            active_question=active_question,
            active_definition=active_definition,
            latest_message=latest_message,
            pending_questions=pending_questions,
            make_category_options=make_category_options,
        )
        return fallback


def _valid_question_retry(decision: QuestionTurnDecision, active_slot: str) -> bool:
    reply = str(decision.reply_to_user or "").strip()
    questions = re.findall(r"[^?]*\?", reply)
    return bool(
        decision.retry_slot == active_slot
        and len(questions) == 1
        and reply.endswith("?")
        and str(decision.rephrased_question or "").strip().endswith("?")
    )


def _question_retry_validation_errors(
    decision: QuestionTurnDecision,
    active_slot: str,
) -> list[str]:
    question = str(decision.rephrased_question or "").strip()
    reply = str(decision.reply_to_user or "").strip()
    errors: list[str] = []
    if not question:
        errors.append("missing_rephrased_question")
    if decision.retry_slot != active_slot:
        errors.append(f"retry_slot_mismatch:{decision.retry_slot!r}")
    question_count = len(re.findall(r"[^?]*\?", reply))
    if question_count != 1:
        errors.append(f"reply_question_count:{question_count}")
    if reply and not reply.endswith("?"):
        errors.append("reply_does_not_end_with_question")
    if question and not question.endswith("?"):
        errors.append("rephrased_question_not_question")
    return errors


def _repair_question_retry(
    *,
    state: ChatbotState,
    decision: QuestionTurnDecision,
    active_slot: str,
    active_question: str,
    latest_message: str,
    direct_answer_fallback: str = "",
) -> QuestionTurnDecision:
    if _valid_question_retry(decision, active_slot):
        logger.info(
            "question_retry_source | source=initial_adjudicator | slot=%r | output=%s",
            active_slot,
            _safe_json(_model_dump(decision)),
        )
        return decision
    logger.warning(
        "question_retry_validation_failed | source=initial_adjudicator | slot=%r | errors=%s | output=%s",
        active_slot,
        _safe_json(_question_retry_validation_errors(decision, active_slot)),
        _safe_json(_model_dump(decision)),
    )
    try:
        repaired = _question_turn_adjudicator_llm().invoke(
            [
                SystemMessage(
                    content=(
                        "Repair this unanswered qualification response. Answer the customer's counter-question "
                        "with a statement, then naturally rephrase the exact active qualification question. "
                        "Do not ask about the counter-question topic. The only question in reply_to_user must be "
                        "rephrased_question, exactly once at the end. Set retry_slot exactly to active_slot. "
                        "Repair wording fields only. Never change email actions, FAQ/escalation fields, "
                        "slot decisions, extracted values, or any other adjudication semantics."
                    )
                ),
                HumanMessage(
                    content=_safe_json(
                        {
                            "active_slot": active_slot,
                            "exact_active_question": active_question,
                            "latest_user_message": latest_message,
                            "prior_decision": _model_dump(decision),
                            "recent_messages": (state.get("messages") or [])[-6:],
                        }
                    )
                ),
            ]
        )
        candidate = _sanitize_question_turn_decision(
            _model_dump(repaired) if isinstance(repaired, BaseModel) else {}
        )
        logger.info(
            "question_retry_repair_output | slot=%r | output=%s",
            active_slot,
            _safe_json(_model_dump(candidate)),
        )
        if _valid_question_retry(candidate, active_slot):
            repaired_decision = decision.model_copy(
                update={
                    "reply_to_user": candidate.reply_to_user,
                    "rephrased_question": candidate.rephrased_question,
                    "retry_slot": active_slot,
                }
            )
            logger.info(
                "question_retry_source | source=repair_llm | slot=%r | output=%s",
                active_slot,
                _safe_json(_model_dump(repaired_decision)),
            )
            return repaired_decision
        logger.warning(
            "question_retry_validation_failed | source=repair_llm | slot=%r | errors=%s | output=%s",
            active_slot,
            _safe_json(_question_retry_validation_errors(candidate, active_slot)),
            _safe_json(_model_dump(candidate)),
        )
    except Exception:
        logger.exception("Question retry repair failed")
    source = str(decision.reply_to_user or "").strip()
    if not source or source == active_question or not re.search(r"[.!](?:\s|$)", source):
        source = str(direct_answer_fallback or "").strip()
    statements = [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+", source)
        if part.strip() and not part.strip().endswith("?")
    ]
    direct_answer = " ".join(statements).strip()
    reply = f"{direct_answer} {active_question}".strip()
    fallback = decision.model_copy(
        update={
            "reply_to_user": reply,
            "rephrased_question": active_question,
            "retry_slot": active_slot,
        }
    )
    logger.warning(
        "question_retry_source | source=deterministic_fallback | slot=%r | direct_answer_source=%r | output=%s",
        active_slot,
        "adjudicator" if source == str(decision.reply_to_user or "").strip() else "mind",
        _safe_json(_model_dump(fallback)),
    )
    return fallback


def _normalize_length_or_width_value(value: Any) -> Any:
    text = str(value or "").strip()
    if not text:
        return value
    match = re.fullmatch(
        r"(\d+(?:\.\d+)?)\s*(ft|feet|foot|footer|footers|'|in|inch|inches|\"|mm|millimeters?|cm|centimeters?|m|meters?|metres?|yd|yards?)?",
        text,
        re.I,
    )
    if not match:
        return value
    amount = float(match.group(1))
    unit = str(match.group(2) or "ft").lower()
    if unit in {"in", "inch", "inches", '"'}:
        feet = amount / 12
    elif unit in {"mm", "millimeter", "millimeters"}:
        feet = amount / 304.8
    elif unit in {"cm", "centimeter", "centimeters"}:
        feet = amount / 30.48
    elif unit in {"m", "meter", "meters", "metre", "metres"}:
        feet = amount * 3.280839895
    elif unit in {"yd", "yard", "yards"}:
        feet = amount * 3
    else:
        feet = amount
    return f"{feet:g} ft"


def _normalize_roll_off_bin_size_as_length(value: Any) -> Any:
    """Map a Roll Off bin's yard-size number directly to trailer feet."""
    text = str(value or "").strip()
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(?:yd|yds|yard|yards)?", text, re.I)
    if not match:
        return value
    return f"{float(match.group(1)):g} ft"


def _dimension_shorthand_updates(text: str) -> dict[str, Any]:
    match = re.search(
        r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:ft|feet|foot|')?\s*[xX]\s*"
        r"(\d+(?:\.\d+)?)\s*(?:ft|feet|foot|')?"
        r"(?:\s*[xX]\s*(\d+(?:\.\d+)?)\s*(?:ft|feet|foot|')?)?(?!\d)",
        text or "",
        re.I,
    )
    if not match:
        return {}
    updates: dict[str, Any] = {
        "width_ft": match.group(1),
        "length_ft": match.group(2),
    }
    if match.group(3):
        updates["height_ft"] = match.group(3)
    return updates


def _normalize_payload_value(value: Any) -> Any:
    text = str(value or "").strip()
    if not text:
        return value
    match = re.fullmatch(
        r"(\d+(?:\.\d+)?)\s*(k)?\s*(lbs?|pounds?|#|kg|kilograms?|tons?|tonnes?|oz|ounces?)?",
        text,
        re.I,
    )
    if not match:
        return value
    amount = float(match.group(1))
    multiplier = 1000 if match.group(2) else 1
    unit = str(match.group(3) or "lbs").lower()
    if unit in {"kg", "kilogram", "kilograms"}:
        pounds = amount * multiplier * 2.2046226218
    elif unit in {"ton", "tons"}:
        pounds = amount * multiplier * 2000
    elif unit in {"tonne", "tonnes"}:
        pounds = amount * multiplier * 2204.6226218
    elif unit in {"oz", "ounce", "ounces"}:
        pounds = amount * multiplier / 16
    else:
        pounds = amount * multiplier
    return f"{pounds:g} lbs"


def _canonicalize_color(value: Any) -> Any:
    text = str(value or "").strip()
    return text.lower() if text else value


def _canonicalize_adjudicated_metadata(
    *,
    key: str,
    value: Any,
    category: str | None,
) -> tuple[str, Any] | None:
    if value in (None, "") or key not in _METADATA_FILTER_KEYS:
        return None
    if key == "make":
        return None
    if key == "subcategory" and normalize_category(category) != "Aluminum":
        logger.info("metadata_filter_rejected | key=subcategory | value=%r | reason=non_aluminum_category", value)
        return None
    if key == "subcategory":
        subcategory = normalize_subcategory(str(value))
        if not subcategory:
            logger.info("metadata_filter_rejected | key=subcategory | value=%r | reason=invalid", value)
            return None
        return key, subcategory
    if key == "hitch_type":
        hitch = _normalize_allowed_hitch(value)
        if not hitch:
            logger.info("metadata_filter_rejected | key=hitch_type | value=%r | reason=invalid", value)
            return None
        return key, hitch
    if key in {"length_ft", "width_ft", "height_ft"}:
        if key == "length_ft" and normalize_category(category) == "Roll Off":
            return key, _normalize_roll_off_bin_size_as_length(value)
        return key, _normalize_length_or_width_value(value)
    if key == "payload_lbs":
        return key, _normalize_payload_value(value)
    if key == "color":
        return key, _canonicalize_color(value)
    return key, value


def _canonicalize_adjudicated_slots(
    *,
    raw: dict[str, Any],
    allowed_category_slots: set[str],
    metadata_updates: dict[str, Any],
) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    for key, value in raw.items():
        key = str(key)
        if key not in allowed_category_slots or value in (None, ""):
            continue
        mapped_fields = _SLOT_METADATA_FILTER_MAP.get(key, ())
        dimension_values = [
            metadata_updates[field]
            for field in mapped_fields
            if field in {"length_ft", "width_ft", "height_ft"}
            and metadata_updates.get(field) not in (None, "")
        ]
        if len(dimension_values) > 1:
            updates[key] = " × ".join(str(value) for value in dimension_values)
            continue
        mapped_value = next(
            (metadata_updates[field] for field in mapped_fields if metadata_updates.get(field) not in (None, "")),
            None,
        )
        if mapped_value is not None:
            updates[key] = mapped_value
        elif any(field in {"length_ft", "width_ft"} for field in mapped_fields):
            updates[key] = _normalize_length_or_width_value(value)
        elif any(field == "payload_lbs" for field in mapped_fields):
            updates[key] = _normalize_payload_value(value)
        else:
            updates[key] = value
    return updates


def _legacy_field_updates_from_filter_extraction(
    *,
    state: ChatbotState,
    category: str | None,
    awaiting_slot: str | None,
    apply_slot_updates: bool,
) -> FieldExtractionAdjudicationDecision:
    extraction = _extract_filter_decision(state, category)
    metadata = _metadata_filters_from_extraction(
        extraction,
        state.get("user_message") or "",
        awaiting_slot,
    )
    slots = (
        _slot_updates_from_extraction(
            extraction,
            _category_slots(category),
            state.get("user_message") or "",
            awaiting_slot,
        )
        if apply_slot_updates
        else {}
    )
    return FieldExtractionAdjudicationDecision(
        metadata_filters_update=metadata,
        slots_collected_update=slots,
        requested_non_metadata_features=[],
        confidence="medium" if metadata or slots else "low",
        reason="legacy_filter_extraction_fallback",
    )


def _extract_field_updates(
    *,
    state: ChatbotState,
    category: str | None,
    awaiting_slot: str | None,
    apply_slot_updates: bool,
    reset_active_request: bool = False,
) -> FieldExtractionAdjudicationDecision:
    latest = state.get("user_message") or ""
    if not os.getenv("OPENAI_API_KEY"):
        return FieldExtractionAdjudicationDecision(
            confidence="low",
            reason="field_extraction_llm_unavailable",
        )

    allowed_category_slots = _category_slots(category)
    recent = [
        str(m.get("content") or "")
        for m in reversed((state.get("messages") or [])[-8:])
        if isinstance(m, dict) and m.get("role") == "user"
    ]
    context = {
        "latest_user_message": latest,
        "recent_user_messages": recent,
        "current_category": category,
        "category_changed_or_search_reset": reset_active_request,
        "awaiting_slot": awaiting_slot,
        "previous_assistant_question": _last_assistant_question(state.get("messages") or []),
        "metadata_field_definitions": _metadata_field_definitions(),
        "category_slot_definitions": _slot_field_definitions(category),
        "existing_slots_collected": state.get("slots_collected") or {},
        "existing_metadata_filters_collected": state.get("metadata_filters_collected") or {},
        "allowed_metadata_fields": sorted(_LLM_ADJUDICATED_METADATA_KEYS),
        "allowed_category_slots": sorted(allowed_category_slots if apply_slot_updates else set()),
        "generic_haul_use_slot": _GENERIC_HAUL_USE_SLOT,
    }
    try:
        decision = _field_extraction_adjudicator_llm().invoke(
            [
                SystemMessage(
    content=(
        "Extract all explicit trailer search updates from the latest user message in one pass. "
        "Return structured data only. Always return rejected_candidates as a JSON array ([] if none).\n\n"

        "## HARD RULES (apply before extracting anything)\n"
        "1. CRITICAL HAUL-ITEM RULE: CATEGORY ≠ HAUL ITEM. A category names the trailer type, not its cargo.\n"
        "   - 'I want an equipment trailer' → sets category only. haul_item/generic_haul_use = nothing.\n"
        "   - 'I need to haul equipment' → sets generic_haul_use='equipment'.\n"
        "   - A category-like term may answer an active haul-item question from the assistant.\n"
        "   - History alone must never create a haul item.\n"
        "2. HITCH TYPES ONLY GO IN hitch_type. Gooseneck and Bumper Pull are hitch configurations.\n"
        "   Never put either into base_category, generic_haul_use, haul_item, subcategory, make, or features.\n"
        "   Only extract hitch_type when the user explicitly selects or prefers one in the latest message.\n"
        "   Informational questions ('what hitches do you have?', 'which is better?') → no hitch_type update.\n"
        "3. NO MAKE EXTRACTION. Make/manufacturer is handled by a separate resolver.\n"
        "4. NO subcategory UNLESS current_category=Aluminum.\n\n"

        "## FIELD EXTRACTION RULES\n"
        "- Use recent messages and the previous assistant question only to interpret the latest message — not as new updates.\n"
        "- Never return an existing/historical value unless the latest message states, changes, confirms, or clearly refers to it.\n"
        "- LOOSE / OPEN-ENDED ANSWERS: Store vague or open-ended answers as-is in the relevant field.\n"
        "  Examples: 'any length' → length_ft='any'; 'all types of material' → generic_haul_use='all types of material';\n"
        "  'no budget limit' → do not set max_price; 'doesn't matter' for a field → leave that field null.\n"
        "  The goal is to capture what the user said, even if it is non-specific.\n"
        "- UNKNOWN CATEGORY: Extract any stated item, cargo, material, equipment, or use case into generic_haul_use.\n"
        "  Include broad phrases: 'some heavy items', 'construction materials', 'a car and tools'.\n"
        "  Do not require a precise named object. Do not put haul/use phrases into requested_non_metadata_features.\n"
        "- requested_non_metadata_features: only user-requested equipment/config/features not covered by metadata or slots.\n"
        "  Do not infer from inventory/listing text.\n\n"

        "## UNIT NORMALIZATION\n"
        "- Lengths and widths → feet with suffix 'ft'. Convert inches, 'footer', 'foot' etc.\n"
        "  Compact notation AxB = width A x length B; AxBxC = width A x length B x height C.\n"
        "- Weights and payloads → pounds with suffix 'lbs'. Convert tons, kg, etc.\n"
        "- payload_lbs is haul/carried weight — NOT GVWR unless user specifically says GVWR.\n"
        "- Side/wall height: '3 inch sides', '3 ft walls' → height_ft.\n\n"

        "## AUTHORITATIVE ACTIVE-SLOT POLICY\n"
        "- For a free-text active slot, extract any substantive direct answer even when broad or informal. "
        "Do not overwrite it using text that answers a different active slot.\n"
        "- Numeric fields require a digit or an unambiguous number written in words. Accept ranges and "
        "approximations; for every range store only its smallest stated value ('5000-10000 lbs' -> '5000 lbs'). "
        "Return no numeric update when no usable number exists.\n"
        "- Width requires a numeric measurement. Never extract 'flexible', 'normal', 'standard', 'whatever fits', "
        "or 'no specific measurement' as width_ft.\n"
        "- Compound dimensions must be separated: '16 by 7 feet' means length_ft='16 ft' and width_ft='7 ft'. "
        "Never copy the complete compound phrase into both fields.\n"
        "- For active cargo_size, one usable length fully answers the field; width and height are optional. "
        "'18 by 8 feet; height is not important' stores cargo_size='18 ft × 8 ft', length_ft='18 ft', "
        "width_ft='8 ft'. 'About 18 feet long' stores cargo_size='18 ft', length_ft='18 ft'.\n"
        "- For Roll Off bin_size, map the chosen numeric value directly into length_ft for Pinecone: "
        "'15 yd' becomes length_ft='15 ft', not 45 ft. For ranges, use the smallest value.\n"
        "- An already resolved category is stable during active Q&A. Cargo wording is not a category switch.\n"
        "- Extract a make only from an explicitly named manufacturer; generic cargo language is never a make.\n\n"

        "## CONFIDENCE\n"
        "If confidence is low, leave updates empty and optionally set clarification_needed.\n"
        "The field definitions and category slot definitions passed in context are authoritative."
        
        "EXTREMELY IMPORTANT- LOOSE / OPEN-ENDED ANSWERS:\n"
"  • Dimensions (length_ft, width_ft, height_ft): vague answers ('any length', 'doesn't matter', 'no preference') → set to null. Only store a concrete measurement.\n"
"  • Haul/use fields (generic_haul_use, haul_item, haul_material): store whatever the user says, even if broad — 'all types of material', 'anything', 'various equipment','random things',it does not matter what they say. as long it's not a counter question or a value for any other field. Capture the phrase as-is.\n"
"  • hitch_type: only store 'gooseneck' or 'bumper pull'. Any other answer or non-specific reply ('either', 'doesn't matter', 'any') → set to null.\n"
"  • max_price: only set when the user gives a concrete upper limit. Vague answers → null.\n"
"  • payload_lbs: store a concrete weight only. Vague answers ('any weight', 'doesn't matter') → null.\n"
"IF ONE OF THE ANSWERS IF LOOSE/OPEN-ENDED, DO NOT SET OTHER FIELDS UNLESS EXPLICITLY STATED. For example, if the user says 'I want a trailer for gravel, any length is fine', set generic_haul_use='gravel' and length_ft=null. Do not set other fields unless explicitly stated."
    )
),
                HumanMessage(content=f"Return field extraction for this context:\n{_safe_json(context)}"),
            ]
        )
    except Exception:
        logger.exception("field_extraction_adjudicator_llm_failed")
        return FieldExtractionAdjudicationDecision(
            confidence="low",
            reason="field_extraction_adjudicator_exception",
        )

    data = _model_dump(decision)
    # Deterministic A×B orientation override. The documented convention is
    # "AxB = width A x length B; AxBxC adds height C" (stated in the extractor
    # prompt), but the LLM occasionally transposes width and length. When the
    # latest message contains that shorthand, a regex parse is authoritative, so
    # we overwrite the LLM's width/length/height with it. Injecting before
    # normalization lets the corrected values also mirror into the category slots.
    shorthand_dims = _dimension_shorthand_updates(latest)
    if shorthand_dims:
        merged_metadata = dict(data.get("metadata_filters_update") or {})
        merged_metadata.update(shorthand_dims)
        data["metadata_filters_update"] = merged_metadata
    data = _normalize_llm_field_mappings(data, category)
    confidence = str(data.get("confidence") or "low").lower()
    if confidence not in _CONFIDENT_FIELD_EXTRACTION:
        return FieldExtractionAdjudicationDecision(
            requested_non_metadata_features=[],
            rejected_candidates=data.get("rejected_candidates") or [],
            clarification_needed=data.get("clarification_needed"),
            confidence="low",
            reason=data.get("reason") or "low_confidence",
        )

    metadata_updates: dict[str, Any] = {}
    for key, value in (data.get("metadata_filters_update") or {}).items():
        sanitized = _canonicalize_adjudicated_metadata(key=str(key), value=value, category=category)
        if sanitized:
            clean_key, clean_value = sanitized
            metadata_updates[clean_key] = clean_value

    raw_slot_updates = {
        str(key): value
        for key, value in (data.get("slots_collected_update") or {}).items()
        if value not in (None, "")
    }
    slot_updates = (
        _canonicalize_adjudicated_slots(
            raw=raw_slot_updates,
            allowed_category_slots=allowed_category_slots,
            metadata_updates=metadata_updates,
        )
        if apply_slot_updates
        else {}
    )
    features = [
        feature
        for feature in _normalize_requested_feature_list(
            data.get("requested_non_metadata_features") or []
        )
        if _feature_has_message_evidence(feature, latest)
    ]
    for feature in _normalize_requested_feature_list(
        data.get("requested_non_metadata_features") or []
    ):
        if feature not in features:
            logger.info(
                "requested_feature_rejected | feature=%r | reason=not_in_latest_message",
                feature,
            )
    slot_updates, features = _remove_generic_haul_use_feature_duplicates(slot_updates, features)
    return FieldExtractionAdjudicationDecision(
        metadata_filters_update=metadata_updates,
        slots_collected_update=slot_updates,
        requested_non_metadata_features=features,
        rejected_candidates=data.get("rejected_candidates") or [],
        clarification_needed=data.get("clarification_needed"),
        confidence=confidence if confidence in {"medium", "high"} else "low",
        reason=data.get("reason") or "",
    )


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
    return _canonicalize_adjudicated_metadata(key=key, value=value, category=None)


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
        if shorthand_dimension.group(3):
            updates["height_ft"] = shorthand_dimension.group(3)

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
    height = _first_match(
        (
            r"\b(\d+(?:\.\d+)?)\s*(ft|feet|foot|'|in|inch|inches)\s*(?:high|tall)\b",
            r"\b(\d+(?:\.\d+)?)\s*(ft|feet|foot|'|in|inch|inches)\s*(?:(?:\w+\s+){0,3})?(?:side|sides|wall|walls)\b",
            r"\b(?:height|high|tall|side|sides|wall|walls)\D{0,40}?(\d+(?:\.\d+)?)\s*(ft|feet|foot|'|in|inch|inches)\b",
        ),
        text,
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
    if height and "height_ft" not in updates:
        updates["height_ft"] = height
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
                        "- Understand natural length expressions too: '14 footer' means 14 ft and 'twenty-foot trailer' means 20 ft.\n"
                        "- Return all dimensions converted to feet with suffix 'ft' and all weights converted to pounds with suffix 'lbs'.\n"
                        "- Trailer shorthand like '6x12' means width_ft=6 and length_ft=12; '6x12x5' means width_ft=6, length_ft=12, height=5. Do not store height unless there is an allowed slot/filter for it.\n"
                        "- Payload/load/haul weight maps to payload_lbs, not GVWR.\n"
                        "- hitch_type can only be Gooseneck or Bumper Pull and must be explicitly named in the latest message; return null otherwise.\n"
                        "- Gooseneck and Bumper Pull are strictly hitch types. Never return either as make, category, subcategory, base_category, haul item, use case, or another slot value.\n"
                        "- make is the trailer manufacturer only; never use Gooseneck or Bumper Pull as make.\n"
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
        logger.exception("Filter extractor LLM failed; returning no updates")
        return FilterExtractionDecision()


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
    category: str | None = None,
) -> dict[str, Any]:
    raw = decision.get("metadata_filters_update") or {}
    updates: dict[str, Any] = {}
    for key, value in raw.items():
        key = str(key)
        if key == "make":
            resolution = resolve_make_from_text(str(value), use_llm_fallback=False)
            if not resolution.make:
                resolution = resolve_make_from_text(latest_message or "", use_llm_fallback=False)
            if resolution.make:
                updates["make"] = resolution.make
            continue
        sanitized = _canonicalize_adjudicated_metadata(key=key, value=value, category=category)
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
        updates[key] = value
    return updates


_HAUL_ITEM_SLOT_ALIASES = {
    "generic_haul_use",
    "haul_item",
    "haul_material",
    "cargo_type",
    "cargo_item",
    "use_case",
    "vehicle_type",
    "equipment_list",
}
_HAUL_ITEM_SLOT_TARGETS = (
    "haul_material",
    "haul_item",
    "vehicle_type",
    "equipment_list",
    "cargo_type",
    "cargo_item",
    "use_case",
    "fiber_use_case",
)


def _normalize_llm_field_mappings(
    decision: dict[str, Any],
    category: str | None,
) -> dict[str, Any]:
    normalized = dict(decision)
    raw_slots = dict(normalized.get("slots_collected_update") or {})
    raw_metadata = dict(normalized.get("metadata_filters_update") or {})
    allowed = _category_slots(category)

    # Rescue searchable fields and category-slot aliases returned in the wrong slot dictionary.
    for key in tuple(raw_slots):
        metadata_key = key if key in _METADATA_FILTER_KEYS else None
        if metadata_key is None and key not in allowed:
            mapped_fields = _SLOT_METADATA_FILTER_MAP.get(key, ())
            if len(mapped_fields) == 1:
                metadata_key = mapped_fields[0]
        if metadata_key is None:
            continue
        sanitized = _canonicalize_adjudicated_metadata(
            key=metadata_key,
            value=raw_slots.pop(key),
            category=category,
        )
        if sanitized:
            clean_key, clean_value = sanitized
            raw_metadata.setdefault(clean_key, clean_value)

    # Map generic/incorrect haul-field aliases to the selected category's haul field.
    haul_value = next(
        (
            value
            for key, value in raw_slots.items()
            if key in _HAUL_ITEM_SLOT_ALIASES and value not in (None, "")
        ),
        None,
    )
    if haul_value is not None:
        target = next((slot for slot in _HAUL_ITEM_SLOT_TARGETS if slot in allowed), None)
        if target:
            raw_slots.setdefault(target, haul_value)
            for key in tuple(raw_slots):
                if key in _HAUL_ITEM_SLOT_ALIASES and key != target and key not in allowed:
                    raw_slots.pop(key, None)

    # Normalize metadata values, then mirror each concept into the category slot.
    clean_metadata: dict[str, Any] = {}
    for key, value in raw_metadata.items():
        sanitized = _canonicalize_adjudicated_metadata(
            key=str(key),
            value=value,
            category=category,
        )
        if sanitized:
            clean_key, clean_value = sanitized
            clean_metadata[clean_key] = clean_value
    for slot, value in _slot_updates_from_metadata(category, clean_metadata).items():
        raw_slots.setdefault(slot, value)

    normalized["slots_collected_update"] = raw_slots
    normalized["metadata_filters_update"] = clean_metadata
    return normalized


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
    payload_slot = pick(("haul_weight_lbs", "payload_need", "payload_capacity", "total_weight"))

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


def _normalized_generic_haul_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    for char in ",.;:!?()[]{}\"'":
        text = text.replace(char, " ")
    return " ".join(text.split())


def _remove_generic_haul_use_feature_duplicates(
    slot_updates: dict[str, Any],
    requested_features: list[str],
) -> tuple[dict[str, Any], list[str]]:
    generic_value = slot_updates.get(_GENERIC_HAUL_USE_SLOT)
    generic_text = _normalized_generic_haul_text(generic_value)
    if not generic_text:
        return slot_updates, requested_features

    duplicate_forms = {
        generic_text,
        f"haul {generic_text}",
        f"hauling {generic_text}",
        f"carry {generic_text}",
        f"carrying {generic_text}",
        f"move {generic_text}",
        f"moving {generic_text}",
        f"use for {generic_text}",
        f"using for {generic_text}",
    }
    cleaned = [
        feature
        for feature in requested_features
        if _normalized_generic_haul_text(feature) not in duplicate_forms
    ]
    return slot_updates, cleaned


def _apply_generic_haul_use_to_category_slot(category: str | None, slots: dict[str, Any]) -> None:
    generic_value = slots.get(_GENERIC_HAUL_USE_SLOT)
    if not category or generic_value in (None, ""):
        return

    allowed = _category_slots(category)
    target_slot: str | None = None
    for candidate in ("haul_material", "haul_item", "cargo_type", "cargo_item", "use_case"):
        if candidate in allowed:
            target_slot = candidate
            break
    if not target_slot:
        return

    if not slots.get(target_slot):
        slots[target_slot] = generic_value
    slots.pop(_GENERIC_HAUL_USE_SLOT, None)


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


def _explicitly_rejects_aluminum(message: str) -> bool:
    return bool(
        re.search(
            r"\b(?:not|no)\s+aluminum\b"
            r"|\b(?:do\s+not|don't|dont)\s+want\s+(?:an?\s+)?aluminum\b"
            r"|\binstead\s+of\s+(?:an?\s+)?aluminum\b",
            str(message or ""),
            re.I,
        )
    )


def _apply_aluminum_category_guardrail(
    *,
    latest_message: str,
    current_category: str | None,
    awaiting_slot: str | None,
    decision: dict[str, Any],
) -> None:
    mentioned = resolve_categories_from_text(latest_message)
    non_aluminum = [category for category in mentioned if category != "Aluminum"]
    answering_base_category = current_category == "Aluminum" and awaiting_slot == "base_category"
    rejects_aluminum = _explicitly_rejects_aluminum(latest_message)

    if answering_base_category and not rejects_aluminum:
        if decision.get("trailer_category") != "Aluminum":
            logger.info(
                "aluminum_base_category_prevents_category_switch | proposed=%r | message=%r",
                decision.get("trailer_category"),
                latest_message,
            )
        decision["trailer_category"] = "Aluminum"
        return

    if rejects_aluminum:
        return

    if "Aluminum" not in mentioned or not non_aluminum:
        return

    selected = _aluminum_base_category_subcategory(
        (decision.get("slots_collected_update") or {}).get("base_category")
    )
    if selected not in non_aluminum:
        selected = non_aluminum[0] if len(non_aluminum) == 1 else None

    decision["trailer_category"] = "Aluminum"
    decision["category_resolution_kind"] = "explicit"
    if selected:
        updates = dict(decision.get("slots_collected_update") or {})
        updates["base_category"] = selected
        decision["slots_collected_update"] = updates
    logger.info(
        "aluminum_category_guardrail_applied | mentioned=%s | base_category=%r",
        mentioned,
        selected,
    )


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
    defaulted_fields: set[str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    if not category:
        return [], {}

    spec = get_trailer_fields_as_dict(category)
    required_slots = list(spec.get("required_slots") or [])
    dynamic_questions: dict[str, str] = {}
    confident = classification.confidence in _CONFIDENT_CLASSIFICATIONS
    cat = category.strip().lower()

    matched_item_is_usable = _is_usable_classifier_haul_item(classification.matched_item)
    if confident and matched_item_is_usable and classification.matched_item:
        allowed_slots = _category_slots(category)
        target_slot: str | None = None
        for candidate in ("haul_material", "haul_item"):
            if candidate in allowed_slots and not slots.get(candidate):
                target_slot = candidate
                break
        if target_slot:
            slots[target_slot] = classification.matched_item
            logger.info(
                "classification_haul_item_filled | category=%r | target_slot=%r | matched_item=%r | reason=%r",
                category,
                target_slot,
                classification.matched_item,
                classification.reason,
            )

    known_haul_item = bool(slots.get("haul_item") or slots.get("haul_material") or matched_item_is_usable)
    if cat == "utility" and confident and classification.is_lightweight_utility_load and known_haul_item:
        required_slots = [slot for slot in required_slots if slot != "haul_weight_lbs"]
        if not metadata_filters.get("payload_lbs") and not slots.get("haul_weight_lbs"):
            metadata_filters["payload_lbs"] = "1500 lbs"
            if defaulted_fields is not None:
                defaulted_fields.add("payload_lbs")
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


def _apply_cargo_size_extraction_safeguard(
    *,
    decision: QuestionTurnDecision,
    active_slot: str,
    extracted_metadata: dict[str, Any],
    no_preference_decision: PreferenceNullDecision,
) -> QuestionTurnDecision:
    """Accept current-turn cargo length when semantic reconciliation misses it."""
    preference_data = _model_dump(no_preference_decision)
    if (
        active_slot != "cargo_size"
        or decision.counter_question_topic != "none"
        or decision.email_action != "none"
        or decision.skip_remaining_questions
        or (
            preference_data.get("has_no_preference")
            and preference_data.get("confidence") in _CONFIDENT_PREFERENCE_NULL
        )
    ):
        return decision

    length = str(extracted_metadata.get("length_ft") or "").strip()
    if not length or not _validate_slot_value("trailer_length_ft", length)[0]:
        return decision
    width = str(extracted_metadata.get("width_ft") or "").strip()
    if width and not _validate_slot_value("trailer_width_ft", width)[0]:
        width = ""

    value = f"{length} × {width}" if width else length
    metadata_updates = dict(decision.metadata_filters_update or {})
    metadata_updates["length_ft"] = length
    if width:
        metadata_updates["width_ft"] = width
    slot_updates = dict(decision.slots_collected_update or {})
    slot_updates["cargo_size"] = value
    return decision.model_copy(
        update={
            "answered_active_question": True,
            "no_preference_for_active_question": False,
            "active_slot_value": value,
            "metadata_filters_update": metadata_updates,
            "slots_collected_update": slot_updates,
            "reply_to_user": "",
            "rephrased_question": "",
            "retry_slot": None,
            "confidence": "high",
            "reason": "current_turn_cargo_length_extraction_answered",
        }
    )


def _apply_active_slot_extraction_safeguard(
    *,
    decision: QuestionTurnDecision,
    active_slot: str,
    extracted_slots: dict[str, Any],
    extracted_metadata: dict[str, Any],
    no_preference_decision: PreferenceNullDecision,
) -> QuestionTurnDecision:
    """Resolve an active slot from valid current-turn extraction evidence."""
    preference_data = _model_dump(no_preference_decision)
    if (
        decision.answered_active_question
        or decision.no_preference_for_active_question
        or decision.counter_question_topic != "none"
        or decision.email_action != "none"
        or decision.search_now_requested
        or decision.skip_remaining_questions
        or (
            preference_data.get("has_no_preference")
            and preference_data.get("confidence") in _CONFIDENT_PREFERENCE_NULL
        )
    ):
        return decision

    value = extracted_slots.get(active_slot)
    if value in (None, "") or not _validate_slot_value(active_slot, value)[0]:
        return decision

    slot_updates = dict(decision.slots_collected_update or {})
    slot_updates[active_slot] = value
    metadata_updates = dict(decision.metadata_filters_update or {})
    for key in _SLOT_METADATA_FILTER_MAP.get(active_slot, ()):
        extracted_value = extracted_metadata.get(key)
        if extracted_value not in (None, ""):
            metadata_updates[key] = extracted_value
    logger.info(
        "active_slot_extraction_recovered | slot=%r | value=%r",
        active_slot,
        value,
    )
    return decision.model_copy(
        update={
            "answered_active_question": True,
            "no_preference_for_active_question": False,
            "active_slot_value": value,
            "slots_collected_update": slot_updates,
            "metadata_filters_update": metadata_updates,
            "reply_to_user": "",
            "rephrased_question": "",
            "retry_slot": None,
            "confidence": "high",
            "reason": "current_turn_active_slot_extraction_answered",
        }
    )


def _enforce_skipped_slot_invariants(
    *,
    slots: dict[str, Any],
    metadata_filters: dict[str, Any],
    slots_skipped: set[str],
) -> None:
    """No-preference/skip state has final priority over every update source."""
    for slot in slots_skipped:
        removed_slot = slots.pop(slot, None)
        removed_filters = {
            key: metadata_filters.pop(key)
            for key in _SLOT_METADATA_FILTER_MAP.get(slot, ())
            if key in metadata_filters
        }
        if removed_slot is not None or removed_filters:
            logger.info(
                "skipped_slot_cleanup_applied | slot=%r | removed_slot=%r | removed_filters=%s",
                slot,
                removed_slot,
                json.dumps(removed_filters, default=str),
            )


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
    if state.get("awaiting_slot") or state.get("pending_questions"):
        return False
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
    category_needs_clarification: bool = False,
    category_clarification_key: str | None = None,
) -> ChatbotState:
    decision["action"] = "respond"
    logger.info("catalogue_redirect_applied | user_message=%r", state.get("user_message"))
    return {
        **state,
        "trailer_category": category,
        "category_needs_clarification": category_needs_clarification,
        "category_clarification_key": category_clarification_key,
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
            "category_clarification_key": resolution.clarification_key,
            "assistant_text": clarification,
            "mind_decision": {
                "action": "respond",
                "assistant_text": clarification,
                "category_clarification_key": resolution.clarification_key,
            },
        }

    deterministic_hint = resolution.category
    deterministic_hint_tier = resolution.match_tier

    context = {
        "user_message": state.get("user_message"),
        "customer": {
            "name": state.get("customer_full_name"),
            "email": state.get("customer_email"),
            "phone": state.get("customer_phone"),
            "contact_status": state.get("contact_status"),
            "contact_request_allowed": bool(state.get("contact_request_allowed")),
        },
        "current_category": state.get("trailer_category"),
        "deterministic_category_hint": deterministic_hint,
        # D4: send a compact listing projection (index/title/url/price) instead of
        # full listing objects, and drop the full TrailerListing field list — both
        # were pure bloat/distraction on the planner's hot path (audit issue #5).
        "slots_collected": state.get("slots_collected") or {},
        "metadata_filters_collected": state.get("metadata_filters_collected") or {},
        "awaiting_slot": state.get("awaiting_slot"),
        "pending_questions": state.get("pending_questions") or [],
        "pending_category_change": state.get("pending_category_change"),
        "pending_category_suggestion": state.get("pending_category_suggestion"),
        "asked_questions": state.get("asked_questions") or [],
        "last_listings": compact_listings(state.get("last_listings")),
        "already_shown_listing_urls": state.get("already_shown_listing_urls") or [],
        "recent_messages": compact_recent_messages(state.get("messages")),
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
    except Exception:
        logger.exception("Mind LLM failed; using fallback decision")
        decision = _fallback_decision(state)
        if deterministic_hint and not decision.trailer_category:
            decision.trailer_category = deterministic_hint
    logger.info("mind_structured_output | %s", _safe_json(_model_dump(decision)))
    decision.assistant_text = _strip_search_promises(decision.assistant_text)

    if decision.trailer_category and decision.trailer_category not in CANONICAL_CATEGORIES:
        logger.info(
            "llm_category_ignored_noncanonical | proposed_category=%r | user_message=%r",
            decision.trailer_category,
            state.get("user_message"),
        )
        decision.trailer_category = None
    # The deterministic hint is advisory, not an unconditional override. The old
    # code stamped the hint as explicit/high even when it came from a cargo word
    # that beat the customer's explicitly named type, defeating a correct LLM
    # proposal. Now: trust the hint when the LLM is silent or agrees, or when the
    # hint is a naming-tier (explicitly named type) match; otherwise keep the
    # LLM's own canonical category but lower confidence so the downstream
    # category-transition reconciler still guards the change.
    if deterministic_hint:
        llm_category = decision.trailer_category
        if not llm_category or llm_category == deterministic_hint:
            decision.trailer_category = deterministic_hint
            decision.category_resolution_kind = "explicit"
            decision.category_confidence = "high"
        elif deterministic_hint_tier == "naming":
            logger.info(
                "category_hint_disagreement | hint=%r tier=%r llm=%r | preferring naming-tier hint",
                deterministic_hint,
                deterministic_hint_tier,
                llm_category,
            )
            decision.trailer_category = deterministic_hint
            decision.category_resolution_kind = "explicit"
            decision.category_confidence = "high"
        else:
            logger.info(
                "category_hint_disagreement | hint=%r tier=%r llm=%r | keeping LLM category at medium confidence",
                deterministic_hint,
                deterministic_hint_tier,
                llm_category,
            )
            decision.category_confidence = "medium"
    pending_suggestion = dict(state.get("pending_category_suggestion") or {})
    if pending_suggestion:
        latest_choice_text = _normalize_choice_text(state.get("user_message") or "")
        explicit_recommend_request = bool(
            re.search(
                r"\b(?:recommend|suggest|pick|choose)\s+(?:me\s+)?(?:one|the\s+best|best)\b"
                r"|\bwhich\s+(?:one|option)\s+(?:is|would\s+be)\s+best\b",
                latest_choice_text,
            )
        )
        if explicit_recommend_request:
            decision.category_suggestion_response = "recommend_one"
        suggested = str(
            decision.trailer_category
            or decision.recommended_category
            or pending_suggestion.get("recommended_category")
            or pending_suggestion.get("category")
            or ""
        ).strip()
        options = _category_suggestion_options(pending_suggestion)
        if (
            decision.trailer_category in CANONICAL_CATEGORIES
            and decision.category_resolution_kind == "explicit"
            and decision.action == "pinecone_search"
        ):
            logger.info("category_suggestion_decision | %s", _safe_json(_model_dump(decision)))
            return {
                **state,
                "pending_category_suggestion": None,
                "awaiting_slot": None,
                "mind_decision": _model_dump(decision),
            }
        if decision.category_suggestion_response == "recommend_one":
            recommended = str(
                decision.recommended_category
                or pending_suggestion.get("recommended_category")
                or suggested
                or ""
            ).strip()
            if recommended in CANONICAL_CATEGORIES:
                pending_suggestion["recommended_category"] = recommended
                pending_suggestion["status"] = "awaiting_recommended_confirmation"
                decision.trailer_category = None
                decision.action = "respond"
                decision.assistant_text = _category_suggestion_prompt(
                    pending_suggestion,
                    confirm_recommended=True,
                )
                logger.info("category_suggestion_decision | %s", _safe_json(_model_dump(decision)))
                return {
                    **state,
                    "pending_category_suggestion": pending_suggestion,
                    "awaiting_slot": _CATEGORY_SUGGESTION_SLOT,
                    "mind_decision": _model_dump(decision),
                }
        if decision.category_suggestion_response == "accept" and suggested in CANONICAL_CATEGORIES:
            decision.trailer_category = suggested
            decision.category_resolution_kind = "explicit"
            decision.action = "pinecone_search"
            logger.info("category_suggestion_decision | %s", _safe_json(_model_dump(decision)))
            return {**state, "pending_category_suggestion": None, "awaiting_slot": None, "mind_decision": _model_dump(decision)}
        if decision.category_suggestion_response == "reject":
            decision.trailer_category = None
            decision.action = "respond"
            decision.assistant_text = "No problem. Which trailer type would you like to explore instead?"
            logger.info("category_suggestion_decision | %s", _safe_json(_model_dump(decision)))
            return {**state, "pending_category_suggestion": None, "awaiting_slot": None, "mind_decision": _model_dump(decision)}
        if resolution.category and (decision.trailer_category == resolution.category or resolution.category in options):
            decision.trailer_category = resolution.category
            decision.action = "pinecone_search"
            logger.info("category_suggestion_decision | %s", _safe_json(_model_dump(decision)))
            return {**state, "pending_category_suggestion": None, "awaiting_slot": None, "mind_decision": _model_dump(decision)}
        if _has_categoryless_spec_update(state.get("user_message") or ""):
            decision.trailer_category = None
            decision.action = "respond"
            logger.info("category_suggestion_specs_deferred | %s", _safe_json(_model_dump(decision)))
            return {**state, "mind_decision": _model_dump(decision)}
        decision.trailer_category = None
        decision.action = "respond"
        decision.assistant_text = decision.assistant_text or _category_suggestion_prompt(pending_suggestion)
        logger.info("category_suggestion_decision | %s", _safe_json(_model_dump(decision)))
        return {**state, "mind_decision": _model_dump(decision)}

    recommended_candidate = decision.trailer_category or decision.recommended_category
    if not state.get("trailer_category") and not recommended_candidate and decision.category_recommendations:
        recommendations = [
            item for item in (decision.category_recommendations or [])
            if isinstance(item, dict) and str(item.get("category") or "").strip() in CANONICAL_CATEGORIES
        ]
        if recommendations:
            pending_payload = {
                "categories": recommendations,
                "reasoning": decision.category_reasoning,
                "confidence": decision.category_confidence,
                "status": "awaiting_choice_or_recommendation",
            }
            decision.action = "respond"
            decision.assistant_text = decision.assistant_text or _category_suggestion_prompt(pending_payload)
            logger.info("category_recommendation_decision | %s", _safe_json(_model_dump(decision)))
            return {
                **state,
                "pending_category_suggestion": pending_payload,
                "awaiting_slot": _CATEGORY_SUGGESTION_SLOT,
                "mind_decision": _model_dump(decision),
            }
    if (
        recommended_candidate
        and not state.get("trailer_category")
        and resolution.category != recommended_candidate
        and decision.category_resolution_kind != "explicit"
    ):
        recommended = recommended_candidate
        if recommended in CANONICAL_CATEGORIES and (
            decision.category_resolution_kind == "recommendation"
            or decision.category_resolution_kind == "none"
        ):
            confidence = (
                decision.category_confidence
                if decision.category_confidence in {"medium", "high"}
                else "medium"
            )
            recommendations = [
                item for item in (decision.category_recommendations or [])
                if isinstance(item, dict) and str(item.get("category") or "").strip() in CANONICAL_CATEGORIES
            ]
            if not recommendations:
                recommendations = [{
                    "category": recommended,
                    "confidence": confidence,
                    "reasoning": decision.category_reasoning,
                }]
            pending_payload = {
                "categories": recommendations,
                "recommended_category": (
                    decision.recommended_category
                    if decision.recommended_category in CANONICAL_CATEGORIES
                    else recommended
                ),
                "category": recommended,
                "reasoning": decision.category_reasoning,
                "confidence": confidence,
                "status": "awaiting_recommended_confirmation",
            }
            decision.trailer_category = None
            decision.category_resolution_kind = "recommendation"
            decision.category_confidence = confidence
            decision.action = "respond"
            confirm_text = _category_suggestion_prompt(pending_payload, confirm_recommended=True)
            base_text = _strip_search_promises(decision.assistant_text)
            decision.assistant_text = (
                base_text
                if "?" in base_text and recommended.lower() in base_text.lower()
                else f"{base_text} {confirm_text}".strip()
            )
            logger.info("category_recommendation_decision | %s", _safe_json(_model_dump(decision)))
            return {
                **state,
                "pending_category_suggestion": pending_payload,
                "awaiting_slot": _CATEGORY_SUGGESTION_SLOT,
                "mind_decision": _model_dump(decision),
            }
        logger.info("inferred_category_not_applied | proposed=%r | kind=%r | confidence=%r", recommended, decision.category_resolution_kind, decision.category_confidence)
        decision.trailer_category = None
        decision.action = "respond"
        if not decision.assistant_text:
            decision.assistant_text = "Could you share a little more detail so I can narrow this down?"
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


def _classify_non_recommendation_turn(
    *,
    state: ChatbotState,
    category: str | None,
    slots: dict[str, Any],
    metadata_filters: dict[str, Any],
    latest_message: str,
    awaiting_slot: str | None,
    pending_questions: list[QuestionItem],
    active_qna_slot: str | None = None,
    active_qna_question: str = "",
) -> NonRecommendationTurnDecision:
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not str(latest_message or "").strip() or not api_key or api_key.startswith("test"):
        return NonRecommendationTurnDecision(reason="classifier_unavailable", confidence="low")
    context = {
        "latest_user_message": latest_message,
        "current_category": category,
        "slots_collected": slots,
        "metadata_filters_collected": metadata_filters,
        "awaiting_slot": awaiting_slot,
        "pending_questions": pending_questions,
        "active_qna_slot": active_qna_slot,
        "active_qna_question": active_qna_question,
        "has_shown_search_results": bool(state.get("has_shown_search_results")),
        "last_listings": state.get("last_listings") or [],
        "recent_messages": (state.get("messages") or [])[-8:],
        "contact_status": state.get("contact_status"),
        "customer_has_contact": bool(state.get("customer_email") or state.get("customer_phone")),
    }
    try:
        decision = _non_recommendation_turn_llm().invoke(
            [
                SystemMessage(
                    content=(
                        "Act only as a safety classifier for FAQ/contact and unsupported business-action tools. "
                        "Return structured data only. For catalogue questions, category questions, recommendations, "
                        "active QnA answers, and ordinary shopping, return continue_recommendation_flow without replacement text.\n\n"
                        "Choose send_non_sales_faq_email when the customer asks how to contact TrailerPlace, asks for the phone number, "
                        "location, store info, sales contact, financing, trade-in, service, or parts. Use faq_category contact_human for "
                        "general contact/sales-contact questions and store_info for location/store visit questions.\n\n"
                        "Examples: 'How can I contact you guys?' means send_non_sales_faq_email/contact_human; "
                        "'Where are you located?' means send_non_sales_faq_email/store_info; "
                        "'Can you call me tomorrow?' means send_escalation_alert_email, not contact_human.\n\n"
                        "Choose send_escalation_alert_email when the customer asks TrailerPlace/the team to perform an unsupported "
                        "real-world action or commitment: for example hold/reserving a trailer or an item; send a reminder or future follow-up; create/send "
                        "a quote, invoice, contract, application, or paperwork; call/text/email them; schedule a call, "
                        "meeting, appointment, delivery, pickup, inspection, service, or installation; promise future "
                        "timing; buy a trailer from the customer; or make a custom arrangement. The user must request "
                        "an action. Ordinary questions about price, hours, financing, or reservation policy do not qualify.\n\n"
                        "Choose continue_recommendation_flow when normal trailer QnA/search should continue and existing field extraction "
                        "should handle freeform details. Set should_store_freeform_fields=true only for clear trailer-shopping constraints "
                        "or active question answers. Do not infer inventory details from listings.\n\n"
                        "Gooseneck and Bumper Pull are strictly hitch types, never trailer categories, makes, haul items, "
                        "or use cases. A message containing only one of these still contains trailer-shopping filter data; "
                        "continue the recommendation flow so field extraction can store it as hitch_type. Do not treat "
                        "'gooseneck trailer' or 'bumper pull trailer' as a resolved trailer category."
                    )
                ),
                HumanMessage(content=_safe_json(context)),
            ]
        )
        result = decision if isinstance(decision, NonRecommendationTurnDecision) else NonRecommendationTurnDecision()
        logger.info("pre_generic_structured_output | %s", _safe_json(_model_dump(result)))
        return result
    except Exception:
        logger.exception("Non-recommendation turn classifier failed; continuing existing flow")
        return NonRecommendationTurnDecision(reason="classifier_failed", confidence="low")


def _non_recommendation_tool_state(
    *,
    state: ChatbotState,
    turn_decision: NonRecommendationTurnDecision,
    mind_decision: dict[str, Any],
    category: str | None,
    slots: dict[str, Any],
    slots_skipped: set[str],
    metadata_filters: dict[str, Any],
    requested_non_metadata_features: list[str],
    latest_message: str,
    category_changed: bool,
    make_changed: bool,
    make_category_options: list[str],
    awaiting_slot: str | None,
    pending_questions: list[QuestionItem],
) -> ChatbotState | None:
    if turn_decision.confidence not in {"medium", "high"}:
        return None
    if turn_decision.action not in {"send_non_sales_faq_email", "send_escalation_alert_email"}:
        return None

    decision = dict(mind_decision)
    decision["action"] = turn_decision.action
    assistant_text = str(turn_decision.assistant_text or decision.get("assistant_text") or "").strip()
    if turn_decision.action == "send_non_sales_faq_email":
        faq_category = (turn_decision.faq_category or "contact_human").strip().lower()
        if faq_category not in FAQ_CATEGORY_LABELS:
            faq_category = "contact_human"
        decision["faq_category"] = faq_category
        decision["faq_summary"] = turn_decision.faq_summary or FAQ_CATEGORY_LABELS.get(faq_category)
    else:
        decision["escalation_summary"] = (
            turn_decision.escalation_summary
            or f"Customer requested an unsupported business action: {latest_message[:240]}"
        )
        decision["unsupported_request"] = latest_message
    if assistant_text:
        decision["assistant_text"] = assistant_text
    output_slots = slots if turn_decision.should_store_freeform_fields else dict(state.get("slots_collected") or {})
    output_metadata = (
        metadata_filters
        if turn_decision.should_store_freeform_fields
        else dict(state.get("metadata_filters_collected") or {})
    )
    output_features = (
        requested_non_metadata_features
        if turn_decision.should_store_freeform_fields
        else list(state.get("requested_non_metadata_features") or [])
    )
    output_category = category if turn_decision.should_store_freeform_fields else state.get("trailer_category")

    return {
        **state,
        "trailer_category": output_category,
        "slots_collected": output_slots,
        "slots_skipped": sorted(slots_skipped),
        "metadata_filters_collected": output_metadata,
        "requested_non_metadata_features": output_features,
        "active_search_request_text": _updated_active_search_request_text(
            state=state,
            latest_message=latest_message if turn_decision.should_store_freeform_fields else "",
            slots=output_slots,
            metadata_filters=output_metadata,
            reset_active_request=category_changed or make_changed,
        ),
        "make_category_options": make_category_options,
        "awaiting_slot": awaiting_slot,
        "pending_questions": pending_questions,
        "assistant_text": assistant_text,
        "mind_decision": decision,
    }


def _apply_make_resolution(
    *,
    latest_message: str,
    category: str | None,
    metadata_filters: dict[str, Any],
    recent_messages: list[dict[str, Any]] | None = None,
) -> tuple[str | None, list[str], str | None]:
    resolution = resolve_make_from_text(
        latest_message,
        use_llm_fallback=False,
    )
    if not resolution.make:
        return category, [], None
    verification = _verify_make_candidate(
        candidate_make=resolution.make,
        candidate_match_type=resolution.match_type,
        latest_message=latest_message,
        category=category,
        recent_messages=recent_messages or [],
    )
    if not (
        verification.approve_make
        and verification.confidence in _CONFIDENT_CLASSIFICATIONS
        and verification.verified_make == resolution.make
    ):
        logger.info(
            "make_candidate_rejected | candidate=%r | match_type=%r | confidence=%r | evidence=%r | reason=%r",
            resolution.make,
            resolution.match_type,
            verification.confidence,
            verification.explicit_evidence,
            verification.reason,
        )
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
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    extraction = _extract_field_updates(
        state={
            **state,
            "trailer_category": category,
            "slots_collected": slots,
            "metadata_filters_collected": metadata_filters,
        },
        category=category,
        awaiting_slot=awaiting_slot,
        apply_slot_updates=apply_slot_updates,
        reset_active_request=bool(state.get("trailer_category") and category and state.get("trailer_category") != category),
    )
    extracted_metadata = dict(extraction.metadata_filters_update or {})
    extracted_slots = dict(extraction.slots_collected_update or {})
    if category:
        extracted_slots.pop(_GENERIC_HAUL_USE_SLOT, None)
    bare_trailer_inches = re.search(
        r"\b(\d+(?:\.\d+)?)\s*(?:\"|in|inch|inches)\s+(?:(?:\w+)\s+){0,3}trailer\b",
        latest_message or "",
        re.I,
    )
    if bare_trailer_inches and not re.search(
        r"\b(?:wide|width|high|height|tall)\b", latest_message or "", re.I
    ):
        length_value = _normalize_length_or_width_value(
            f"{bare_trailer_inches.group(1)} inches"
        )
        if extracted_metadata.get("width_ft") == length_value:
            extracted_metadata.pop("width_ft", None)
            metadata_filters.pop("width_ft", None)
        extracted_metadata["length_ft"] = length_value
        logger.info("bare_trailer_dimension_forced_to_length | value=%r", length_value)
    logger.info(
        "field_extraction_applied | category=%r | confidence=%r | extracted_metadata=%s | extracted_slots=%s | requested_non_metadata_features=%s",
        category,
        extraction.confidence,
        json.dumps(extracted_metadata, default=str),
        json.dumps(extracted_slots, default=str),
        json.dumps(extraction.requested_non_metadata_features or [], default=str),
    )
    for key, value in extracted_metadata.items():
        if key in _METADATA_FILTER_KEYS and value not in (None, ""):
            metadata_filters[key] = value
    return extracted_slots, extracted_metadata, list(extraction.requested_non_metadata_features or [])


def _legacy_apply_explicit_filter_extraction(
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


def _category_filter_confirmation_text(values: dict[str, Any], category: str) -> str:
    labels = {
        "length_ft": "length",
        "width_ft": "width",
        "height_ft": "height",
        "payload_lbs": "payload capacity",
        "hitch_type": "hitch type",
    }
    def display_value(key: str, value: Any) -> str:
        text = str(value or "").strip()
        number = re.search(r"\d+(?:\.\d+)?", text)
        if key in {"length_ft", "width_ft", "height_ft"}:
            normalized = str(_normalize_length_or_width_value(text)).strip()
            normalized_number = re.search(r"\d+(?:\.\d+)?", normalized)
            return f"{normalized_number.group(0)} ft" if normalized_number else normalized
        if key == "payload_lbs":
            normalized = str(_normalize_payload_value(text)).strip()
            if re.search(r"\b(?:lb|lbs|pound|pounds)\b", normalized, re.I):
                return normalized
            return f"{number.group(0)} lbs" if number else normalized
        return text.replace("_", " ") if key == "hitch_type" else text

    details = ", ".join(
        f"{labels[key]} {display_value(key, value)}" for key, value in values.items()
    )
    return f"Should I keep your previous {details} for the {category} trailer?"


def _yes_no_answer(text: str) -> bool | None:
    normalized = re.sub(r"[^a-z ]", " ", str(text or "").lower()).strip()
    positive = bool(re.search(r"\b(?:yes|yeah|yep|sure|keep|same|carry|retain|continue)\b", normalized))
    negative = bool(re.search(r"\b(?:no|nope|reset|clear|remove|discard|different|start over)\b", normalized))
    selective = bool(re.search(r"\b(?:length|width|height|payload|capacity|weight|hitch)\b", normalized))
    if positive and not negative and not selective:
        return True
    if negative and not positive and not selective:
        return False
    return None


def _adjudicate_category_filter_confirmation(
    message: str, carry_filters: dict[str, Any]
) -> CategoryFilterConfirmationDecision:
    try:
        return _category_filter_confirmation_llm().invoke(
            [
                SystemMessage(content=(
                    "Interpret a customer's reply about carrying old trailer filters into a new category. "
                    "Allowed fields are length_ft, width_ft, height_ft, payload_lbs, hitch_type. "
                    "Gooseneck and Bumper Pull are strictly hitch_type values and must never be interpreted as "
                    "categories, makes, subcategories, or other fields. "
                    "Put explicitly retained fields in keep_fields, explicitly rejected fields in discard_fields, "
                    "and explicitly changed values in updates. A changed field is also resolved. "
                    "Resolve pronouns from the supplied pending fields: when only length_ft is pending, "
                    "'change it to 20 ft' updates length_ft. Do not decide unspecified fields. "
                    "Set resolved=true when at least one field is addressed. Explain the interpretation in reasoning "
                    "and set confidence to low, medium, or high.\n"
                    "Example 1: pending={length_ft: 15 ft}, reply='change it to 20ft' -> "
                    "updates={length_ft: 20 ft}, resolved=true, confidence=high.\n"
                    "Example 2: pending={length_ft: 15 ft, width_ft: 7 ft, hitch_type: bumper pull}, "
                    "reply='keep the width, drop the length, and make it gooseneck' -> "
                    "keep_fields=[width_ft], discard_fields=[length_ft], updates={hitch_type: gooseneck}, "
                    "resolved=true, confidence=high."
                )),
                HumanMessage(content=_safe_json({
                    "reply": message,
                    "filters_awaiting_confirmation": carry_filters,
                })),
            ]
        )
    except Exception:
        logger.exception("category_filter_confirmation_llm_failed")
        return CategoryFilterConfirmationDecision()


def _apply_mind_node(state: ChatbotState) -> ChatbotState:
    decision = dict(state.get("mind_decision") or {})
    original_mind_action = str(decision.get("action") or "respond")
    original_mind_text = str(decision.get("assistant_text") or "").strip()
    slots_before = dict(state.get("slots_collected") or {})
    slots = dict(slots_before)
    slots_skipped_before = set(state.get("slots_skipped") or [])
    slots_skipped = set(slots_skipped_before)
    metadata_filters_before = dict(state.get("metadata_filters_collected") or {})
    metadata_filters = dict(metadata_filters_before)
    defaulted_metadata_filters = set(state.get("defaulted_metadata_filters") or [])
    requested_non_metadata_features = list(state.get("requested_non_metadata_features") or [])
    invalid_required_slot: str | None = None
    awaiting_slot = state.get("awaiting_slot")
    category_before = state.get("trailer_category")
    latest_message = state.get("user_message") or ""
    _apply_aluminum_category_guardrail(
        latest_message=latest_message,
        current_category=category_before,
        awaiting_slot=awaiting_slot,
        decision=decision,
    )
    for field in tuple(defaulted_metadata_filters):
        if _message_has_filter_evidence(field, latest_message, awaiting_slot):
            defaulted_metadata_filters.discard(field)
    pending_category_change = dict(state.get("pending_category_change") or {})
    pending_category_suggestion = dict(state.get("pending_category_suggestion") or {})
    if pending_category_suggestion and not decision.get("trailer_category"):
        extracted_slots, _extracted_metadata, extracted_features = _apply_explicit_filter_extraction(
            state=state,
            category=None,
            slots=slots,
            metadata_filters=metadata_filters,
            latest_message=latest_message,
            awaiting_slot=None,
            apply_slot_updates=True,
        )
        slots.update(extracted_slots)
        requested_non_metadata_features = extracted_features
        state = {
            **state,
            "slots_collected": slots,
            "metadata_filters_collected": metadata_filters,
            "requested_non_metadata_features": requested_non_metadata_features,
        }
    if (
        pending_category_suggestion
        and not decision.get("trailer_category")
        and not _has_categoryless_spec_update(latest_message)
    ):
        suggestion_text = str(decision.get("assistant_text") or "").strip()
        if not suggestion_text:
            suggestion_text = _category_suggestion_prompt(pending_category_suggestion)
        decision["action"] = "respond"
        return build_state_return(
            state,
            assistant_text=suggestion_text,
            slots_collected=slots,
            metadata_filters_collected=metadata_filters,
            requested_non_metadata_features=requested_non_metadata_features,
            awaiting_slot=_CATEGORY_SUGGESTION_SLOT,
            pending_questions=[],
            mind_decision=decision,
        )
    if pending_category_change:
        carry_filters = dict(pending_category_change.get("carry_filters") or {})
        confirmation = _yes_no_answer(latest_message)
        question = str(pending_category_change.get("question") or "Should I keep the previous filters?")
        if confirmation is None:
            field_decision = _adjudicate_category_filter_confirmation(
                latest_message, carry_filters
            )
            logger.info(
                "category_filter_confirmation_adjudicated | resolved=%s | confidence=%r | keep=%s | discard=%s | updates=%s | reasoning=%r",
                field_decision.resolved,
                field_decision.confidence,
                field_decision.keep_fields,
                field_decision.discard_fields,
                field_decision.updates,
                field_decision.reasoning,
            )
            if field_decision.resolved and field_decision.confidence in {"medium", "high"}:
                keep = set(field_decision.keep_fields) & set(carry_filters)
                discard = set(field_decision.discard_fields) & set(carry_filters)
                for key in keep:
                    metadata_filters[key] = carry_filters[key]
                for key, value in field_decision.updates.items():
                    if key not in _COMMON_CATEGORY_FILTERS:
                        continue
                    sanitized = _canonicalize_adjudicated_metadata(
                        key=key, value=value, category=None
                    )
                    if sanitized:
                        clean_key, clean_value = sanitized
                        metadata_filters[clean_key] = clean_value
                addressed = keep | discard | (set(field_decision.updates) & set(carry_filters))
                remaining = {
                    key: value for key, value in carry_filters.items() if key not in addressed
                }
                if remaining:
                    question = _category_filter_confirmation_text(
                        remaining, str(pending_category_change.get("new_category") or category_before)
                    )
                    pending_category_change["carry_filters"] = remaining
                    pending_category_change["question"] = question
                    return build_state_return(
                        state,
                        metadata_filters_collected=metadata_filters,
                        pending_category_change=pending_category_change,
                        assistant_text=question,
                        awaiting_slot=_CATEGORY_FILTER_CONFIRMATION_SLOT,
                        pending_questions=[],
                        mind_decision={**decision, "action": "respond"},
                    )
                confirmation = False
            else:
                reply = _question_turn_fallback_reply(question, latest_message)
                return build_state_return(
                    state,
                    assistant_text=f"{reply}\n\n{question}" if reply else question,
                    awaiting_slot=_CATEGORY_FILTER_CONFIRMATION_SLOT,
                    pending_questions=[],
                    mind_decision={**decision, "action": "respond"},
                )
        if confirmation:
            for key, value in carry_filters.items():
                if key not in metadata_filters:
                    metadata_filters[key] = value
        else:
            for key in carry_filters:
                metadata_filters.pop(key, None)
        state = {**state, "pending_category_change": None}
        awaiting_slot = None
        decision["action"] = "ask_next_question"
    latest_message_resolution = resolve_category_from_text(latest_message)
    category_needs_clarification = bool(state.get("category_needs_clarification"))
    category_clarification_key = str(state.get("category_clarification_key") or "").strip() or None
    if (
        latest_message_resolution.needs_clarification
        and not category_before
        and awaiting_slot != _CATEGORY_CLARIFICATION_SLOT
    ):
        proposed_category = decision.get("trailer_category")
        if proposed_category:
            logger.info(
                "ambiguous_category_blocks_direct_assignment | proposed_category=%r | clarification_key=%r | user_message=%r",
                proposed_category,
                latest_message_resolution.clarification_key,
                latest_message,
            )
            decision.pop("trailer_category", None)
        category_needs_clarification = True
        category_clarification_key = (
            latest_message_resolution.clarification_key
            or category_clarification_key
        )
    category_proposed = decision.get("trailer_category") or category_before
    if (
        category_before
        and category_proposed
        and category_proposed != category_before
    ):
        transition_slot = str(awaiting_slot or "").strip() or None
        transition_question = (
            str(state.get("active_question_text") or "").strip()
            or _last_assistant_question(state.get("messages") or [])
        )
        transition = _reconcile_category_transition(
            state=state,
            persisted_category=str(category_before),
            proposed_category=str(category_proposed),
            active_slot=transition_slot,
            active_question=transition_question,
            latest_message=latest_message,
        )
        transition_approved = bool(
            transition.approve_category_change
            and transition.explicit_category_switch
            and transition.confidence in _CONFIDENT_CLASSIFICATIONS
            and transition.final_category == category_proposed
        )
        if not transition_approved:
            logger.info(
                "category_transition_rejected | persisted=%r | proposed=%r | confidence=%r | reason=%r",
                category_before,
                category_proposed,
                transition.confidence,
                transition.reason,
            )
            category_proposed = category_before
            decision["trailer_category"] = category_before
        else:
            logger.info(
                "category_transition_approved | persisted=%r | proposed=%r | confidence=%r | reason=%r",
                category_before,
                category_proposed,
                transition.confidence,
                transition.reason,
            )
    category = category_proposed
    decision = _normalize_llm_field_mappings(decision, category)

    category_changed = bool(category_before and category and category != category_before)
    make_changed = False
    reset_result_state = False
    if category_changed:
        logger.info(
            "category_change_clears_filters | old_category=%r | new_category=%r",
            category_before,
            category,
        )
        old_common_filters = {
            key: metadata_filters_before[key]
            for key in _COMMON_CATEGORY_FILTERS
            if metadata_filters_before.get(key) not in (None, "")
            and key not in defaulted_metadata_filters
        }
        slots = {}
        slots_skipped = set()
        metadata_filters = {}
        defaulted_metadata_filters = set()
        requested_non_metadata_features = []
        awaiting_slot = None
        reset_result_state = True
        explicit_slots, explicit_metadata, _explicit_features = _apply_explicit_filter_extraction(
            state=state,
            category=category,
            slots=slots,
            metadata_filters=metadata_filters,
            latest_message=latest_message,
            awaiting_slot=None,
            apply_slot_updates=True,
        )
        slots.update(explicit_slots)
        carry_filters = {
            key: value for key, value in old_common_filters.items() if key not in explicit_metadata
        }
        if carry_filters:
            question = _category_filter_confirmation_text(carry_filters, category)
            decision["action"] = "respond"
            return build_state_return(
                state,
                trailer_category=category,
                pending_category_change={
                    "old_category": category_before,
                    "new_category": category,
                    "carry_filters": carry_filters,
                    "question": question,
                },
                slots_collected=slots,
                slots_skipped=[],
                metadata_filters_collected=metadata_filters,
                defaulted_metadata_filters=[],
                requested_non_metadata_features=[],
                active_search_request_text=latest_message,
                make_category_options=[],
                awaiting_slot=_CATEGORY_FILTER_CONFIRMATION_SLOT,
                pending_questions=[],
                asked_questions=[],
                already_shown_listing_urls=[],
                last_listings=[],
                active_question_attempts={},
                active_question_unanswered_count=0,
                active_question_tracker=None,
                assistant_text=question,
                mind_decision=decision,
            )

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
            category_needs_clarification=category_needs_clarification,
            category_clarification_key=category_clarification_key,
        )

    if category_needs_clarification and not category and awaiting_slot != _CATEGORY_CLARIFICATION_SLOT:
        clarification_question = (
            category_clarification_question(category_clarification_key)
            or str(state.get("assistant_text") or decision.get("assistant_text") or "").strip()
            or "Could you clarify which type you mean?"
        )
        decision["action"] = "respond"
        return build_state_return(
            state,
            trailer_category=None,
            category_needs_clarification=True,
            category_clarification_key=category_clarification_key,
            slots_collected=slots,
            slots_skipped=sorted(slots_skipped),
            metadata_filters_collected=metadata_filters,
            requested_non_metadata_features=requested_non_metadata_features,
            active_search_request_text=_updated_active_search_request_text(
                state=state,
                latest_message="",
                slots=slots,
                metadata_filters=metadata_filters,
                reset_active_request=category_changed or make_changed,
            ),
            make_category_options=make_category_options,
            awaiting_slot=_CATEGORY_CLARIFICATION_SLOT,
            pending_questions=[],
            assistant_text=clarification_question,
            mind_decision=decision,
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
            _apply_explicit_filter_extraction(
                state=state,
                category=category,
                slots=slots,
                metadata_filters=metadata_filters,
                latest_message=latest_message,
                awaiting_slot=None,
                apply_slot_updates=False,
            )
            assistant_text = (
                "Which category should I use: "
                f"{', '.join(make_category_options)}?"
                if make_category_options
                else "Which category should I use?"
            )
            reply_prefix = _question_turn_fallback_reply(assistant_text, latest_message)
            if reply_prefix:
                assistant_text = f"{reply_prefix}\n\n{assistant_text}"
            decision["action"] = "respond"
            return build_state_return(
                state,
                trailer_category=category,
                category_needs_clarification=category_needs_clarification,
                category_clarification_key=category_clarification_key,
                slots_collected=slots,
                slots_skipped=sorted(slots_skipped),
                metadata_filters_collected=metadata_filters,
                requested_non_metadata_features=requested_non_metadata_features,
                active_search_request_text=_updated_active_search_request_text(
                    state=state,
                    latest_message=latest_message,
                    slots=slots,
                    metadata_filters=metadata_filters,
                    reset_active_request=category_changed or make_changed,
                ),
                make_category_options=make_category_options,
                awaiting_slot=_MAKE_CATEGORY_CHOICE_SLOT,
                pending_questions=[],
                assistant_text=assistant_text,
                mind_decision=decision,
            )

    if awaiting_slot == _GENERIC_CATEGORY_CHOICE_SLOT:
        chosen_category = resolve_category_from_text(latest_message).category
        if chosen_category:
            category = chosen_category
            awaiting_slot = None
            reset_result_state = True
        elif latest_message.strip():
            question_turn = _adjudicate_active_question_turn(
                state={
                    **state,
                    "trailer_category": category,
                    "slots_collected": slots,
                    "metadata_filters_collected": metadata_filters,
                },
                category=category,
                active_slot=_GENERIC_CATEGORY_CHOICE_SLOT,
                active_question=_GENERIC_CATEGORY_QUESTION,
                active_definition=_slot_definition(
                    _GENERIC_CATEGORY_CHOICE_SLOT,
                    category=category,
                    queued_question=_GENERIC_CATEGORY_QUESTION,
                ),
                latest_message=latest_message,
                pending_questions=[],
                make_category_options=make_category_options,
            )
            if question_turn.no_preference_for_active_question:
                awaiting_slot = None
                slots_skipped.add(_GENERIC_CATEGORY_CHOICE_SLOT)
                continue_generic_category_flow = True
            else:
                continue_generic_category_flow = False

            if continue_generic_category_flow:
                pass
            else:
                _apply_explicit_filter_extraction(
                    state=state,
                    category=category,
                    slots=slots,
                    metadata_filters=metadata_filters,
                    latest_message=latest_message,
                    awaiting_slot=None,
                    apply_slot_updates=False,
                )
                make_resolution = resolve_make_from_text(latest_message, use_llm_fallback=False)
                if make_resolution.make:
                    metadata_filters["make"] = make_resolution.make
                assistant_text = _GENERIC_CATEGORY_QUESTION
                if question_turn.reply_to_user:
                    assistant_text = f"{question_turn.reply_to_user}\n\n{assistant_text}"
                decision["action"] = "respond"
                return build_state_return(
                    state,
                    trailer_category=category,
                    category_needs_clarification=category_needs_clarification,
                    category_clarification_key=category_clarification_key,
                    slots_collected=slots,
                    slots_skipped=sorted(slots_skipped),
                    metadata_filters_collected=metadata_filters,
                    requested_non_metadata_features=requested_non_metadata_features,
                    active_search_request_text=_updated_active_search_request_text(
                        state=state,
                        latest_message="",
                        slots=slots,
                        metadata_filters=metadata_filters,
                        reset_active_request=category_changed or make_changed,
                    ),
                    make_category_options=make_category_options,
                    awaiting_slot=_GENERIC_CATEGORY_CHOICE_SLOT,
                    pending_questions=[],
                    assistant_text=assistant_text,
                    mind_decision=decision,
                )

    category, category_options, make_question = _apply_make_resolution(
        latest_message=latest_message,
        category=category,
        metadata_filters=metadata_filters,
        recent_messages=state.get("messages") or [],
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
        requested_non_metadata_features = []
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
        return build_state_return(
            state,
            trailer_category=category,
            category_needs_clarification=category_needs_clarification,
            category_clarification_key=category_clarification_key,
            slots_collected=slots,
            slots_skipped=sorted(slots_skipped),
            metadata_filters_collected=metadata_filters,
            requested_non_metadata_features=requested_non_metadata_features,
            active_search_request_text=_updated_active_search_request_text(
                state=state,
                latest_message=latest_message,
                slots=slots,
                metadata_filters=metadata_filters,
                reset_active_request=category_changed or make_changed,
            ),
            make_category_options=category_options,
            awaiting_slot=_MAKE_CATEGORY_CHOICE_SLOT,
            pending_questions=[],
            assistant_text=make_question,
            mind_decision=decision,
        )

    _apply_generic_haul_use_to_category_slot(category, slots)

    allowed_category_slots = _category_slots(category)
    pending_source_for_turn = [] if reset_result_state else (state.get("pending_questions") or [])
    active_qna_slot, active_qna_definition = _active_question_context(
        category=category,
        awaiting_slot=awaiting_slot,
        pending_questions=pending_source_for_turn,
        make_category_options=make_category_options,
        messages=state.get("messages") or [],
    )
    active_qna_question = str(active_qna_definition.get("question") or "").strip()
    active_question_attempts = (
        {}
        if category_changed
        else dict(state.get("active_question_attempts") or {})
    )
    if category_changed:
        logger.info(
            "category_change_resets_question_attempts | old_category=%r | new_category=%r",
            category_before,
            category,
        )
    active_qna_unanswered = False
    active_qna_reply = ""
    active_qna_retry_question = ""
    active_question_was_unanswered = False
    active_question_was_resolved = False
    repeated_unanswered_escalation = False
    skipped_unanswered_slot: str | None = None
    active_qna_counter_topic = "none"
    active_qna_email_action = "none"
    active_qna_faq_category: str | None = None
    active_qna_faq_summary: str | None = None
    active_qna_escalation_summary: str | None = None
    active_qna_unsupported_request: str | None = None
    active_qna_search_now = False
    active_qna_skip_remaining = False
    if active_qna_slot and latest_message.strip():
        question_turn = _adjudicate_active_question_turn(
            state={
                **state,
                "trailer_category": category,
                "slots_collected": slots,
                "metadata_filters_collected": metadata_filters,
            },
            category=category,
            active_slot=active_qna_slot,
            active_question=active_qna_question,
            active_definition=active_qna_definition,
            latest_message=latest_message,
            pending_questions=pending_source_for_turn,
            make_category_options=make_category_options,
        )
        supplemental_slots, _supplemental_metadata, supplemental_features = _apply_explicit_filter_extraction(
            state=state,
            category=category,
            slots=slots,
            metadata_filters=metadata_filters,
            latest_message=latest_message,
            awaiting_slot=active_qna_slot,
            apply_slot_updates=True,
        )
        active_no_preference = classify_no_preference(
            category=category,
            user_message=latest_message,
            awaiting_slot=active_qna_slot,
            pending_questions=pending_source_for_turn,
            slots_collected=slots,
            metadata_filters_collected=metadata_filters,
            allowed_category_slots=sorted(allowed_category_slots),
            active_question=active_qna_question,
        )
        question_turn = _reconcile_active_question_turn(
            state=state,
            category=category,
            active_slot=str(active_qna_slot),
            active_question=active_qna_question,
            latest_message=latest_message,
            mind_decision=decision,
            adjudicator_decision=question_turn,
            extracted_slots=supplemental_slots,
            extracted_metadata=_supplemental_metadata,
            extracted_features=supplemental_features,
            no_preference_decision=active_no_preference,
        )
        normalized_question_turn = _normalize_llm_field_mappings(
            _model_dump(question_turn),
            category,
        )
        question_turn.slots_collected_update = dict(
            normalized_question_turn.get("slots_collected_update") or {}
        )
        question_turn.metadata_filters_update = dict(
            normalized_question_turn.get("metadata_filters_update") or {}
        )
        question_turn = _apply_cargo_size_extraction_safeguard(
            decision=question_turn,
            active_slot=str(active_qna_slot),
            extracted_metadata=_supplemental_metadata,
            no_preference_decision=active_no_preference,
        )
        question_turn = _apply_active_slot_extraction_safeguard(
            decision=question_turn,
            active_slot=str(active_qna_slot),
            extracted_slots=supplemental_slots,
            extracted_metadata=_supplemental_metadata,
            no_preference_decision=active_no_preference,
        )
        if question_turn.counter_question_topic != "none":
            # A counter-question cannot answer or skip the active qualification slot.
            # Preserve explicit updates for unrelated fields.
            question_turn.answered_active_question = False
            question_turn.no_preference_for_active_question = False
            question_turn.active_slot_value = None
            question_turn.slots_collected_update.pop(str(active_qna_slot), None)
            for key in _SLOT_METADATA_FILTER_MAP.get(str(active_qna_slot), ()):
                question_turn.metadata_filters_update.pop(str(key), None)
        if question_turn.email_action != "none":
            # Email requests interrupt qualification; they do not resolve the active slot.
            question_turn.answered_active_question = False
            question_turn.no_preference_for_active_question = False
            question_turn.active_slot_value = None
            question_turn.slots_collected_update.pop(str(active_qna_slot), None)
            for key in _SLOT_METADATA_FILTER_MAP.get(str(active_qna_slot), ()):
                question_turn.metadata_filters_update.pop(str(key), None)
        if (
            not question_turn.answered_active_question
            and not question_turn.no_preference_for_active_question
            and not question_turn.search_now_requested
            and int(
                (state.get("active_question_attempts") or {}).get(active_qna_slot)
                or state.get("active_question_unanswered_count")
                or 0
            ) == 0
        ):
            question_turn = _repair_question_retry(
                state=state,
                decision=question_turn,
                active_slot=active_qna_slot,
                active_question=active_qna_question,
                latest_message=latest_message,
                direct_answer_fallback=str(decision.get("assistant_text") or ""),
            )
        logger.info(
            "question_turn_adjudicated | slot=%r | answered=%s | no_preference=%s | search_now=%s | skip_remaining=%s | counter_topic=%r | reply=%r | email_action=%r | confidence=%r | reason=%r",
            active_qna_slot,
            question_turn.answered_active_question,
            question_turn.no_preference_for_active_question,
            question_turn.search_now_requested,
            question_turn.skip_remaining_questions,
            question_turn.counter_question_topic,
            question_turn.reply_to_user,
            question_turn.email_action,
            question_turn.confidence,
            question_turn.reason,
        )
        active_qna_counter_topic = question_turn.counter_question_topic
        active_qna_email_action = question_turn.email_action
        active_qna_faq_category = question_turn.faq_category
        active_qna_faq_summary = question_turn.faq_summary
        active_qna_escalation_summary = question_turn.escalation_summary
        active_qna_unsupported_request = question_turn.unsupported_request
        active_qna_retry_question = question_turn.rephrased_question
        active_qna_search_now = bool(question_turn.search_now_requested and category)
        active_qna_skip_remaining = bool(
            question_turn.skip_remaining_questions and active_qna_search_now
        )
        if question_turn.no_preference_for_active_question:
            active_question_was_resolved = True
            slots_skipped.add(str(active_qna_slot))
            awaiting_slot = None if awaiting_slot == active_qna_slot else awaiting_slot
            for key in _SLOT_METADATA_FILTER_MAP.get(str(active_qna_slot), ()):
                metadata_filters.pop(str(key), None)
        elif (
            active_qna_slot == _CATEGORY_CLARIFICATION_SLOT
            and question_turn.answered_active_question
            and question_turn.active_slot_value in {"Fiber", "Enclosed"}
        ):
            active_question_was_resolved = True
            category = str(question_turn.active_slot_value)
            category_needs_clarification = False
            category_clarification_key = None
            awaiting_slot = None if awaiting_slot == active_qna_slot else awaiting_slot
            reset_result_state = True
        elif question_turn.answered_active_question and question_turn.active_slot_value not in (None, ""):
            active_slot_value = _trim_fallback_active_slot_value(active_qna_slot, question_turn.active_slot_value)
            active_slot_value = _preserve_latest_haul_item_phrase(active_qna_slot, active_slot_value, latest_message)
            is_valid, reason = _validate_slot_value(active_qna_slot, active_slot_value)
            if is_valid:
                active_question_was_resolved = True
                slots[active_qna_slot] = active_slot_value
                slots_skipped.discard(str(active_qna_slot))
                awaiting_slot = None if awaiting_slot == active_qna_slot else awaiting_slot
                for key, value in _slot_value_to_metadata_updates(
                    active_qna_slot,
                    active_slot_value,
                    category,
                ).items():
                    metadata_filters[key] = value
            else:
                invalid_required_slot = active_qna_slot
                active_qna_unanswered = True
                active_qna_reply = question_turn.reply_to_user
                logger.info(
                    "slot_validation_failed | slot=%s | value=%r | reason=%s",
                    active_qna_slot,
                    active_slot_value,
                    reason,
                )
        elif active_qna_search_now:
            active_question_was_resolved = True
            awaiting_slot = None
            active_qna_reply = ""
            active_qna_retry_question = ""
        else:
            active_qna_unanswered = True
            active_qna_reply = question_turn.reply_to_user
        active_question_was_unanswered = active_qna_unanswered
        # While the service is collecting the missing contact details required to
        # send a deferred email, the active qualification question is frozen: the
        # user's contact reply is not a failed answer, so it does not increment the
        # unanswered counter or trigger the skip-after-two escalation. The hold
        # releases once contact is provided (email sent) or declined, after which
        # the next question's counter advances normally.
        suppress_active_question_progress = bool(
            state.get("suppress_active_question_progress")
        )
        previous_unanswered_count = int(active_question_attempts.get(active_qna_slot) or 0)
        if active_question_was_resolved:
            active_question_attempts.pop(active_qna_slot, None)
        elif active_qna_unanswered and not suppress_active_question_progress:
            active_question_attempts[active_qna_slot] = previous_unanswered_count + 1

        if (
            active_qna_unanswered
            and not suppress_active_question_progress
            and active_question_attempts.get(active_qna_slot, 0) >= 2
        ):
            repeated_unanswered_escalation = True
            skipped_unanswered_slot = str(active_qna_slot)
            slots_skipped.add(skipped_unanswered_slot)
            active_question_attempts.pop(active_qna_slot, None)
            awaiting_slot = None
            active_qna_unanswered = False

        for key, value in _slot_updates_from_decision(
            {"slots_collected_update": question_turn.slots_collected_update},
            allowed_category_slots,
            latest_message,
            None,
        ).items():
            if str(key) == str(active_qna_slot):
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
            slots_skipped.discard(str(key))

        for key, value in _metadata_filters_from_decision(
            {"metadata_filters_update": question_turn.metadata_filters_update},
            latest_message,
            None,
            category,
        ).items():
            if str(key) in _SLOT_METADATA_FILTER_MAP.get(str(active_qna_slot), ()) and active_qna_unanswered:
                continue
            metadata_filters[key] = value
        if active_qna_skip_remaining:
            for slot in allowed_category_slots:
                if slot not in slots:
                    slots_skipped.add(str(slot))
            awaiting_slot = None
            pending_source_for_turn = []
            active_question_attempts.clear()
        for key, value in supplemental_slots.items():
            if str(key) == str(active_qna_slot):
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
            slots_skipped.discard(str(key))
        if question_turn.requested_non_metadata_features:
            requested_non_metadata_features = list(question_turn.requested_non_metadata_features or [])
        else:
            requested_non_metadata_features = list(supplemental_features or [])
        preference_decision = PreferenceNullDecision(
            has_no_preference=question_turn.no_preference_for_active_question,
            target_slots=[active_qna_slot] if question_turn.no_preference_for_active_question else [],
            confidence=question_turn.confidence,
            reason=question_turn.reason,
        )
    else:
        preference_decision = classify_no_preference(
            category=category,
            user_message=latest_message,
            awaiting_slot=awaiting_slot,
            pending_questions=pending_source_for_turn,
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
                category_needs_clarification=category_needs_clarification,
                category_clarification_key=category_clarification_key,
            )
        awaiting_slot, slots_skipped, _removed_filters = _apply_preference_null_decision(
            category=category,
            awaiting_slot=awaiting_slot,
            pending=pending_source_for_turn,
            metadata_filters=metadata_filters,
            slots_skipped=slots_skipped,
            decision=preference_decision,
        )
        evidence_awaiting_slot = awaiting_slot

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

        for key, value in _metadata_filters_from_decision(
            decision,
            latest_message,
            evidence_awaiting_slot,
            category,
        ).items():
            metadata_filters[key] = value
        extracted_slots, _extracted_metadata, extracted_features = _apply_explicit_filter_extraction(
            state=state,
            category=category,
            slots=slots,
            metadata_filters=metadata_filters,
            latest_message=latest_message,
            awaiting_slot=evidence_awaiting_slot,
            apply_slot_updates=True,
        )
        requested_non_metadata_features = extracted_features
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

    should_check_pre_generic_turn = (
        (decision.get("action") or "respond") in {"respond", "ask_next_question", "pinecone_search"}
        and not bool(state.get("has_shown_search_results"))
        and bool(str(latest_message or "").strip())
    )
    pre_generic_turn_decision: NonRecommendationTurnDecision | None = None
    pre_generic_tool_state: ChatbotState | None = None
    if should_check_pre_generic_turn:
        pre_generic_turn_decision = _classify_non_recommendation_turn(
            state=state,
            category=category,
            slots=slots,
            metadata_filters=metadata_filters,
            latest_message=latest_message,
            awaiting_slot=awaiting_slot,
            pending_questions=list(state.get("pending_questions") or []),
        )
        pre_generic_tool_state = _non_recommendation_tool_state(
            state=state,
            turn_decision=pre_generic_turn_decision,
            mind_decision=decision,
            category=category,
            slots=slots,
            slots_skipped=slots_skipped,
            metadata_filters=metadata_filters,
            requested_non_metadata_features=requested_non_metadata_features,
            latest_message=latest_message,
            category_changed=category_changed,
            make_changed=make_changed,
            make_category_options=make_category_options,
            awaiting_slot=awaiting_slot,
            pending_questions=list(state.get("pending_questions") or []),
        )
    if pre_generic_tool_state is not None:
        return pre_generic_tool_state

    if decision.get("action") == "ask_trailer_category":
        if category:
            # A resolved category enters its normal required-slot qualification flow.
            decision["action"] = "ask_next_question"
        else:
            assistant_text = str(decision.get("assistant_text") or "").strip()
            if not assistant_text:
                assistant_text = _GENERIC_CATEGORY_QUESTION
            return build_state_return(
                state,
                trailer_category=None,
                category_needs_clarification=category_needs_clarification,
                category_clarification_key=category_clarification_key,
                slots_collected=slots,
                slots_skipped=sorted(slots_skipped),
                metadata_filters_collected=metadata_filters,
                requested_non_metadata_features=requested_non_metadata_features,
                active_search_request_text=_updated_active_search_request_text(
                    state=state,
                    latest_message=latest_message,
                    slots=slots,
                    metadata_filters=metadata_filters,
                    reset_active_request=category_changed or make_changed,
                ),
                make_category_options=make_category_options,
                awaiting_slot=_GENERIC_CATEGORY_CHOICE_SLOT,
                pending_questions=[],
                assistant_text=assistant_text,
                mind_decision={**decision, "assistant_text": assistant_text},
            )

    product_counter_topics = {"trailer_categories", "hitch_types", "makes", "dimensions", "payload"}
    if (
        decision.get("action") == "send_non_sales_faq_email"
        and active_qna_counter_topic in product_counter_topics
    ):
        logger.info(
            "product_counter_question_overrides_faq | topic=%r | faq_category=%r",
            active_qna_counter_topic,
            decision.get("faq_category"),
        )
        decision["action"] = "respond"
        if active_qna_reply:
            decision["assistant_text"] = active_qna_reply
    if decision.get("action") == "send_non_sales_faq_email" and decision.get("faq_category") not in FAQ_CATEGORY_LABELS:
        logger.info(
            "faq_action_rejected_missing_category | faq_category=%r | user_message=%r",
            decision.get("faq_category"),
            latest_message,
        )
        decision["action"] = "respond"
    _apply_aluminum_base_category_filter(category, slots, metadata_filters)
    _apply_flatbed_default_width(category, slots, metadata_filters, defaulted_metadata_filters)

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
        defaulted_fields=defaulted_metadata_filters,
    )
    _enforce_skipped_slot_invariants(
        slots=slots,
        metadata_filters=metadata_filters,
        slots_skipped=slots_skipped,
    )
    if invalid_required_slot in slots_skipped:
        invalid_required_slot = None
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
        "send_escalation_alert_email",
        "respond",
    }
    action = decision.get("action") or "respond"
    if action not in valid_actions:
        action = "respond"
    if category_changed and action == "respond" and original_mind_text:
        logger.info(
            "stale_mind_response_discarded_after_category_change | old_category=%r | new_category=%r",
            category_before,
            category,
        )
        original_mind_text = ""
        decision["assistant_text"] = ""
        action = "ask_next_question"
        decision["action"] = action
    explicit_immediate_search = bool(
        category and _immediate_search_requested(str(state.get("user_message") or ""))
    )
    if (active_qna_search_now or explicit_immediate_search) and category:
        if active_qna_email_action:
            logger.info(
                "search_override_escalation | email_action=%r | category=%r | skip_remaining=%s",
                active_qna_email_action,
                category,
                active_qna_skip_remaining,
            )
        action = "pinecone_search"
        decision["action"] = action
        awaiting_slot = None
        active_qna_unanswered = False
        active_question_was_unanswered = False
        if active_qna_skip_remaining or _skip_remaining_questions_requested(
            str(state.get("user_message") or "")
        ):
            active_qna_skip_remaining = True
            pending = []
    elif active_qna_email_action in {"send_non_sales_faq_email", "send_escalation_alert_email"}:
        action = active_qna_email_action
        decision["action"] = action
        if active_qna_faq_category:
            decision["faq_category"] = active_qna_faq_category
        if active_qna_faq_summary:
            decision["faq_summary"] = active_qna_faq_summary
        if active_qna_escalation_summary:
            decision["escalation_summary"] = active_qna_escalation_summary
        if active_qna_unsupported_request:
            decision["unsupported_request"] = active_qna_unsupported_request
    elif active_qna_slot and action == "send_interested_listing_email":
        action = "respond"
        decision["action"] = action
    if action == "send_non_sales_faq_email" and decision.get("faq_category") not in FAQ_CATEGORY_LABELS:
        logger.info(
            "faq_action_rejected_after_qna | faq_category=%r | counter_topic=%r",
            decision.get("faq_category"),
            active_qna_counter_topic,
        )
        action = "respond"
        decision["action"] = action
        if active_qna_reply:
            decision["assistant_text"] = active_qna_reply
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
    qualification_completed_this_turn = False
    if category and required_complete:
        if category_before != category:
            qualification_completed_this_turn = True
        else:
            prior_missing, prior_invalid, _ = _required_slot_state(
                category,
                slots_before,
                slots_skipped=slots_skipped_before,
                required_slots=required_slots_override,
                questions_override=dynamic_questions,
            )
            qualification_completed_this_turn = bool(prior_missing or prior_invalid)
    continue_search_after_email = bool(
        active_qna_email_action in {"send_non_sales_faq_email", "send_escalation_alert_email"}
        and repeated_unanswered_escalation
        and required_complete
        and category
        and not bool(state.get("has_shown_search_results"))
    )
    has_shown_results = bool(state.get("has_shown_search_results"))
    explicit_search = bool(
        action == "pinecone_search"
        or active_qna_search_now
    )
    auto_search_first_results = bool(
        required_complete
        and not has_shown_results
        and (
            active_question_was_resolved
            or qualification_completed_this_turn
            or not original_mind_text
        )
    )
    preserve_mind_response = bool(
        has_shown_results
        and original_mind_text
        and original_mind_action != "pinecone_search"
        and action not in {
            "send_interested_listing_email",
            "send_non_sales_faq_email",
            "send_escalation_alert_email",
        }
        and not active_qna_search_now
    )
    if preserve_mind_response:
        action = "respond"
        decision["action"] = action
        decision["assistant_text"] = original_mind_text
        logger.info(
            "mind_response_preserved | action=%r | category=%r | required_complete=%s | active_question_resolved=%s",
            original_mind_action,
            category,
            required_complete,
            active_question_was_resolved,
        )
    if auto_search_first_results and not preserve_mind_response and action not in {
        "send_interested_listing_email",
        "send_non_sales_faq_email",
        "send_escalation_alert_email",
    }:
        logger.info(
            "auto_search_first_results | category=%r | active_question_resolved=%s | qualification_completed_this_turn=%s",
            category,
            active_question_was_resolved,
            qualification_completed_this_turn,
        )
        if action != "pinecone_search":
            decision["action"] = "pinecone_search"
        action = "pinecone_search"
    elif has_shown_results and explicit_search:
        logger.info("explicit_post_results_search | category=%r", category)
    elif has_shown_results and required_complete and action not in {
        "pinecone_search",
        "send_interested_listing_email",
        "send_non_sales_faq_email",
        "send_escalation_alert_email",
    }:
        logger.info(
            "post_results_auto_search_suppressed | action=%r | category=%r",
            action,
            category,
        )
    should_ask = (
        not preserve_mind_response
        and not active_qna_search_now
        and not active_qna_unanswered
        and (
        action == "ask_next_question" or (
        action in {"pinecone_search", "respond"} and bool(pending)
        )
        )
    )

    asked = [] if reset_result_state else list(state.get("asked_questions") or [])
    assistant_text = decision.get("assistant_text") or ""
    displayed_active_question = ""
    if (
        active_qna_unanswered
        and not active_qna_search_now
        and active_qna_slot
        and action not in {"send_non_sales_faq_email", "send_escalation_alert_email"}
    ):
        action = "respond"
        awaiting_slot = active_qna_slot
        assistant_text = active_qna_reply or active_qna_retry_question
        displayed_active_question = active_qna_retry_question or active_qna_question
        if not assistant_text.strip():
            assistant_text = active_qna_question or "Could you please confirm that requirement?"
    elif should_ask and pending:
        next_question = pending.pop(0)
        assistant_text = next_question.get("question") or "Could you share a little more detail?"
        displayed_active_question = str(next_question.get("question") or "").strip()
        if repeated_unanswered_escalation and active_qna_reply:
            bridge = active_qna_reply.replace(active_qna_retry_question, "").strip()
            if bridge:
                assistant_text = f"{bridge}\n\n{assistant_text}"
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
        if auto_search_first_results and category:
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
        if not assistant_text.strip():
            assistant_text = "Could you share a little more detail so I can narrow this down?"
            awaiting_slot = None
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

    if action in {"send_non_sales_faq_email", "send_escalation_alert_email"} and not awaiting_slot and pending:
        next_slot = str(pending[0].get("slot") or "").strip()
        if next_slot:
            awaiting_slot = next_slot

    slots_after = dict(slots)
    metadata_filters_after = dict(metadata_filters)
    active_search_request_text = _updated_active_search_request_text(
        state=state,
        latest_message="" if active_qna_unanswered else latest_message,
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
    return build_state_return(
        state,
        trailer_category=category,
        category_needs_clarification=category_needs_clarification,
        category_clarification_key=category_clarification_key,
        slots_collected=slots,
        slots_skipped=sorted(slots_skipped),
        metadata_filters_collected=metadata_filters,
        defaulted_metadata_filters=sorted(defaulted_metadata_filters),
        requested_non_metadata_features=requested_non_metadata_features,
        active_search_request_text=active_search_request_text,
        make_category_options=make_category_options,
        awaiting_slot=awaiting_slot,
        pending_questions=pending,
        asked_questions=asked,
        already_shown_listing_urls=[] if reset_result_state else state.get("already_shown_listing_urls"),
        last_listings=[] if reset_result_state else state.get("last_listings"),
        assistant_text=assistant_text,
        selected_listing_title=decision.get("selected_listing_title"),
        selected_listing_url=decision.get("selected_listing_url"),
        mind_decision=decision,
        repeated_unanswered_question_escalation=repeated_unanswered_escalation,
        skipped_unanswered_slot=skipped_unanswered_slot,
        continue_search_after_email=continue_search_after_email,
        active_question_was_unanswered=active_question_was_unanswered,
        active_question_was_resolved=active_question_was_resolved,
        active_question_slot=active_qna_slot,
        active_question_attempts=active_question_attempts,
        active_question_text=(
            displayed_active_question or None
        ),
    )


def _route_after_mind(state: ChatbotState) -> str:
    action = (state.get("mind_decision") or {}).get("action")
    if action == "pinecone_search":
        return "pinecone_search"
    if action == "send_interested_listing_email":
        return "send_interested_listing_email"
    if action == "send_non_sales_faq_email":
        return "send_non_sales_faq_email"
    if action == "send_escalation_alert_email":
        return "send_escalation_alert_email"
    return "finalize_active_question"


def _finalize_active_question_node(state: ChatbotState) -> ChatbotState:
    return {
        **state,
        "active_question_text": (
            state.get("active_question_text")
            or _active_question_followup(state)
            or None
        ),
    }


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
        requested_non_metadata_features=state.get("requested_non_metadata_features"),
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
        event.update(
            {
                "overall_match_level": match_analysis.get("overall_match_level"),
                "full_match_count": match_analysis.get("full_match_count"),
                "partial_match_count": match_analysis.get("partial_match_count"),
                "alternative_count": match_analysis.get("alternative_count"),
                "source": match_analysis.get("source"),
            }
        )
    events.append(event)
    return {
        **state,
        "assistant_text": assistant_text,
        "active_question_text": None,
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
                _missing_contact_request(state, f"your interest in {title}")
            ),
            "pending_contact_action": {
                "type": "interest",
                "item_name": str(title),
                "selected_listing_url": decision.get("selected_listing_url") or state.get("selected_listing_url"),
            },
            "tool_events": events,
        }
    _persist_email_transcript_snapshot(state)
    result = send_interested_listing_email(
        session_id=state.get("session_id") or "",
        full_name=state.get("customer_full_name") or "",
        email=state.get("customer_email"),
        phone=state.get("customer_phone") or "",
        item_name=str(title),
    )
    planner_reply = str(decision.get("assistant_text") or state.get("assistant_text") or "").strip()
    fallback = (
        _INTEREST_GENERIC_FALLBACK.format(item_name=str(title).strip())
        if title
        else _INTEREST_GENERIC_FALLBACK_NO_ITEM
    )
    assistant_text = compose_email_tool_reply(
        email_purpose=f"customer interest in {title or 'a trailer'}",
        latest_message=state.get("user_message") or "",
        conversation_context=_compact_recent_context(state),
        base_reply=planner_reply,
        next_question=str(state.get("active_question_text") or _active_question_followup(state)),
        fallback=fallback or _INTEREST_SAFE_FALLBACK,
    )
    reply_source = "email_reply_llm"
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
    category = str(decision.get("faq_category") or "").strip().lower()
    if category not in FAQ_CATEGORY_LABELS:
        logger.warning("faq_email_blocked_invalid_category | category=%r", category)
        return {
            **state,
            "assistant_text": str(decision.get("assistant_text") or "").strip()
            or "I can help with that here. Could you clarify what information you need?",
        }
    summary = FAQ_CATEGORY_LABELS[category]
    if not _has_contact(state):
        # Informational FAQs are self-service: the direct fallback reply already
        # carries the phone/site, so answer it immediately instead of deferring
        # an internal team-email that just loops asking for the customer's
        # contact (the bug the live harness caught for "how do I contact you").
        events = list(state.get("tool_events") or [])
        events.append({"tool": "send_non_sales_faq_email", "result": {"status": "answered_without_contact"}})
        reply = _FAQ_REPLY_FALLBACKS.get(category, "").strip() or _FAQ_GENERIC_FALLBACK
        return build_state_return(state, assistant_text=reply, tool_events=events)
    _persist_email_transcript_snapshot(state)
    result = send_non_sales_faq_email(
        session_id=state.get("session_id") or "",
        full_name=state.get("customer_full_name") or "",
        email=state.get("customer_email"),
        phone=state.get("customer_phone") or "",
        faq_category=category,
        summary=summary,
        user_message=state.get("user_message") or "",
        context_summary=_compact_recent_context(state),
    )
    planner_reply = str(decision.get("assistant_text") or state.get("assistant_text") or "").strip()
    category_reply = _FAQ_REPLY_FALLBACKS.get(category, "").strip()
    fallback = category_reply or _FAQ_GENERIC_FALLBACK
    assistant_text = compose_email_tool_reply(
        email_purpose=f"{category} customer request",
        latest_message=state.get("user_message") or "",
        conversation_context=_compact_recent_context(state),
        base_reply=planner_reply,
        next_question=str(state.get("active_question_text") or _active_question_followup(state)),
        fallback=_append_active_question_if_present(state, fallback),
    )
    reply_source = "email_reply_llm"
    logger.debug(
        "faq_reply_source=%s category=%s summary=%r",
        reply_source,
        category,
        summary,
    )
    events = list(state.get("tool_events") or [])
    events.append({"tool": "send_non_sales_faq_email", "result": result})
    email_state = {
        **state,
        "assistant_text": assistant_text,
        "tool_events": events,
    }
    if state.get("continue_search_after_email"):
        search_state = _pinecone_search_node(email_state)
        search_state["assistant_text"] = (
            f"{assistant_text}\n\n{search_state.get('assistant_text') or ''}".strip()
        )
        return search_state
    return email_state


def _escalation_email_node(state: ChatbotState) -> ChatbotState:
    decision = state.get("mind_decision") or {}
    summary = str(
        decision.get("escalation_summary")
        or decision.get("unsupported_request")
        or state.get("user_message")
        or "Customer requested an unsupported business action."
    ).strip()
    user_message = str(decision.get("unsupported_request") or state.get("user_message") or "").strip()
    if not _has_contact(state):
        events = list(state.get("tool_events") or [])
        events.append({"tool": "send_escalation_alert_email", "result": {"status": "deferred_missing_contact"}})
        return {
            **state,
            "assistant_text": _missing_contact_request(state, "this request"),
            "pending_contact_action": {
                "type": "escalation_alert",
                "summary": summary,
                "user_message": user_message,
                "context_summary": _compact_recent_context(state),
            },
            "tool_events": events,
        }
    _persist_email_transcript_snapshot(state)
    result = send_escalation_alert_email(
        session_id=state.get("session_id") or "",
        full_name=state.get("customer_full_name") or "",
        email=state.get("customer_email"),
        phone=state.get("customer_phone") or "",
        summary=summary,
        user_message=user_message,
        context_summary=_compact_recent_context(state),
    )
    events = list(state.get("tool_events") or [])
    events.append({"tool": "send_escalation_alert_email", "result": result})
    assistant_text = compose_email_tool_reply(
        email_purpose=summary,
        latest_message=state.get("user_message") or "",
        conversation_context=_compact_recent_context(state),
        base_reply=str(decision.get("assistant_text") or ""),
        next_question=str(state.get("active_question_text") or _active_question_followup(state)),
        fallback=_append_active_question_if_present(state, _ESCALATION_SENT_REPLY),
    )
    email_state = {
        **state,
        "assistant_text": assistant_text,
        "tool_events": events,
    }
    if state.get("continue_search_after_email"):
        search_state = _pinecone_search_node(email_state)
        search_state["assistant_text"] = (
            f"{assistant_text}\n\n{search_state.get('assistant_text') or ''}".strip()
        )
        return search_state
    return email_state


@lru_cache(maxsize=1)
def build_chatbot_graph():
    graph = StateGraph(ChatbotState)
    graph.add_node("mind", _mind_node)
    graph.add_node("apply_mind", _apply_mind_node)
    graph.add_node("pinecone_search", _pinecone_search_node)
    graph.add_node("send_interested_listing_email", _interest_email_node)
    graph.add_node("send_non_sales_faq_email", _faq_email_node)
    graph.add_node("send_escalation_alert_email", _escalation_email_node)
    graph.add_node("finalize_active_question", _finalize_active_question_node)
    graph.set_entry_point("mind")
    graph.add_edge("mind", "apply_mind")
    graph.add_conditional_edges("apply_mind", _route_after_mind)
    graph.add_edge("pinecone_search", "finalize_active_question")
    graph.add_edge("send_interested_listing_email", "finalize_active_question")
    graph.add_edge("send_non_sales_faq_email", "finalize_active_question")
    graph.add_edge("send_escalation_alert_email", "finalize_active_question")
    graph.add_edge("finalize_active_question", END)
    return graph.compile()
