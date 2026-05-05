"""
LangGraph session state for the TrailerPlace conversational agent.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Optional, Sequence, TypedDict

from langchain_core.messages import BaseMessage


class SessionState(TypedDict):
    # ── Conversation history ─────────────────────────────────────────────────
    messages: Annotated[Sequence[BaseMessage], operator.add]

    # ── Routing ──────────────────────────────────────────────────────────────
    # High-level intent: 'sales' | 'finance' | 'service' | 'trade_in' | None
    intent: Optional[str]

    # ── Trailer type (persisted once identified) ─────────────────────────────
    # Canonical category from normalizer.CATEGORY_MAP, e.g. 'Dump', 'Equipment'
    trailer_type: Optional[str]

    # ── Slot collection ───────────────────────────────────────────────────────
    # All qualification answers keyed by slot name, e.g. {"haul_weight_lbs": 9500}
    slots_collected: dict[str, Any]

    # Required slots for the current trailer_type (populated by trailer_fields tool)
    required_slots: list[str]

    # Optional slots for the current trailer_type (populated by trailer_fields tool)
    optional_slots: list[str]

    # ── Search & recommendations ──────────────────────────────────────────────
    search_results: list[dict[str, Any]]
    # Category that produced the current non-empty search_results (for routing).
    # Must match trailer_type to enter recommendation_node; cleared on category pivot.
    search_results_for_category: Optional[str]
    # Listings to return to the HTTP API / Streamlit for this user turn only (cleared each chat()).
    # Distinct from search_results, which stays in state for the recommendation node's prompt.
    api_listings_this_turn: list[dict[str, Any]]
    # True only when specialist just ran a successful search_trailers this graph invoke; gates
    # recommendation_node so FAQ turns are not overridden by the recommendation prompt.
    recommendation_entry_due: bool

    # ── Interest logging ──────────────────────────────────────────────────────
    is_interested: bool
    interested_item: Optional[str]

    # ── Customer info (optional; may be filled from onboarding or API) ───────
    customer_full_name: Optional[str]
    customer_email: Optional[str]
    customer_phone: Optional[str]

    # ── Session tracking ──────────────────────────────────────────────────────
    session_id: Optional[str]
    # URLs supplied by the HTTP client each turn (split UI/API); merged with disk store for show-more.
    client_shown_urls: list[str]

    # ── Node routing helpers ──────────────────────────────────────────────────
    # Tracks which node should handle the next turn
    next_node: Optional[str]
