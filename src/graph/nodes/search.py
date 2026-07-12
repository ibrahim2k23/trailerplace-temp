from __future__ import annotations

import logging
from typing import Any

from src.config import settings
from src.domain.slot_map import normalize_slot_targets
from src.graph.apply_analysis import _current_user_text
from src.search.pinecone_search import search_pinecone_listings

logger = logging.getLogger(__name__)


def _build_metadata_filters(state: dict) -> dict[str, Any]:
    """Translate answered slots into Pinecone metadata filter targets.

    Only category/make/hitch_type/subcategory/min-length become hard filters
    here; width/payload/height stay in ``slots`` for pinecone_search's fit
    rerank to weigh but are never passed as Pinecone `$eq`/`$gte` filters
    themselves (milestone.md M6 step 2).
    """
    category = state.get("category") or ""
    slots = state.get("slots", {}) or {}
    filters: dict[str, Any] = {}
    for slot_name, value in slots.items():
        if value is None:
            continue
        filters.update(normalize_slot_targets(category, slot_name, value))

    hitch_value = slots.get("hitch_type")
    if isinstance(hitch_value, list) and len(hitch_value) == 1:
        filters["hitch_type"] = hitch_value[0]

    if state.get("brand_preference"):
        filters["make"] = state["brand_preference"]

    return filters


def search_node(state: dict) -> dict:
    assert state.get("qualification_complete"), "search_node requires qualification_complete"
    outcome = state.setdefault("turn_outcome", {})
    category = state.get("category")
    slots = state.get("slots", {}) or {}
    metadata_filters = _build_metadata_filters(state)
    user_message = _current_user_text(state)
    shown_urls = state.get("shown_urls", []) or []

    logger.info(
        "TOOL search: session=%s category=%s filters=%s already_shown=%d",
        state.get("session_id"), category, metadata_filters, len(shown_urls),
    )

    results = search_pinecone_listings(
        category=category,
        slots=slots,
        metadata_filters=metadata_filters,
        user_message=user_message,
        already_shown_urls=shown_urls,
        top_k=settings.search_top_k,
        max_recommendations=settings.search_max_recommendations,
    )

    brand_relaxed = False
    if not results and metadata_filters.get("make"):
        relaxed_filters = {key: value for key, value in metadata_filters.items() if key != "make"}
        logger.info(
            "TOOL search: session=%s zero results with make filter, relaxing and retrying filters=%s",
            state.get("session_id"), relaxed_filters,
        )
        results = search_pinecone_listings(
            category=category,
            slots=slots,
            metadata_filters=relaxed_filters,
            user_message=user_message,
            already_shown_urls=shown_urls,
            top_k=settings.search_top_k,
            max_recommendations=settings.search_max_recommendations,
        )
        brand_relaxed = True

    logger.info(
        "TOOL search: session=%s results=%d brand_relaxed=%s urls=%s",
        state.get("session_id"), len(results), brand_relaxed, [r.get("url") for r in results],
    )

    # NOT recorded as shown here: respond decides what actually reaches the customer, and it
    # is respond that records it. Marking them shown from this side told the team we had
    # presented six trailers the customer never saw, and locked those six out of every later
    # "show me more".
    state["last_search_filters"] = metadata_filters
    state["search_pending"] = False

    outcome["listings"] = results
    outcome["search_ran"] = True
    outcome["result_count"] = len(results)
    if brand_relaxed:
        outcome["brand_relaxed"] = True

    if results:
        filter_desc = ", ".join(f"{key}={value}" for key, value in metadata_filters.items())
        description = f"Pinecone search — {len(results)} results — {category or 'Unknown'}"
        if filter_desc:
            description += f" ({filter_desc})"
        outcome.setdefault("system_email_triggers", []).append(
            {"kind": "results_shown", "description": description}
        )
    return state
