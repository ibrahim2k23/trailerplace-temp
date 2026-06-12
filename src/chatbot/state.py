from __future__ import annotations

from typing import Any, Optional, TypedDict


class QuestionItem(TypedDict, total=False):
    slot: str
    question: str
    required: bool


class ChatbotState(TypedDict, total=False):
    session_id: str
    user_message: str
    active_search_request_text: str
    messages: list[dict[str, Any]]
    assistant_text: str
    customer_full_name: Optional[str]
    customer_email: Optional[str]
    customer_phone: Optional[str]
    lead_id: Optional[str]
    contact_status: str
    sales_phase: str
    trailer_category: Optional[str]
    category_needs_clarification: bool
    category_clarification_key: Optional[str]
    slots_collected: dict[str, Any]
    slots_skipped: list[str]
    metadata_filters_collected: dict[str, Any]
    requested_non_metadata_features: list[str]
    make_category_options: list[str]
    awaiting_slot: Optional[str]
    pending_questions: list[QuestionItem]
    asked_questions: list[str]
    already_shown_listing_urls: list[str]
    last_listings: list[dict[str, Any]]
    selected_listing_title: Optional[str]
    selected_listing_url: Optional[str]
    tool_events: list[dict[str, Any]]
    mind_decision: dict[str, Any]
    has_shown_search_results: bool
    active_category_cycle_id: int
    initial_contact_request_asked: bool
    pending_contact_action: Optional[dict[str, Any]]
    pending_initial_user_message: Optional[str]
