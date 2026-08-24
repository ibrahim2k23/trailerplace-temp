"""A volunteered axle rating must still reach the reranker, in every category.

No category ASKS for axle capacity any more - it was Utility's third qualification question
and was removed. That leaves this path with no spec declaring the slot, so nothing in the
category specs would fail if it quietly stopped being stored. These tests are what stands in
for that: "7,000 lb axles" said out loud, in any category, still has to end up ordering the
results. See _ALWAYS_VALID_NUMERIC_SLOTS in src/graph/apply_analysis.py.
"""
from __future__ import annotations

import pytest

from src.graph.apply_analysis import apply_analysis_to_state
from src.graph.nodes.search import _build_metadata_filters
from src.graph.state import new_session_state
from src.search.listing_search import _listing_filters, _required_axle_capacity_lbs_from_filters
from tests.unit.llm_helpers import sample_analysis

CATEGORIES = ["Utility", "Equipment", "Dump", "Livestock", "Enclosed", "Car Hauler", "Flatbed"]


def _state_after_axle_answer(category: str, *, axle_lbs: float = 7000.0) -> dict:
    state = new_session_state("s1")
    state["category"] = category
    state["turn"] = sample_analysis(
        category_mentioned=category,
        intent="qualification_answer",
        extracted={
            "trailer_length_ft": None,
            "trailer_width_ft": None,
            "trailer_height_ft": None,
            "payload_lbs": None,
            "axle_capacity_lbs": axle_lbs,
            "hitch_type": None,
            "haul_item": None,
            "brand_preference": None,
            "non_metadata_features": ["7000 lb axles"],
            "numeric_no_preference": [],
        },
        slot_answers=[{"slot_name": "axle_capacity_lbs", "raw_answer": "7,000 lb axles"}],
    )
    apply_analysis_to_state(state)
    return state


@pytest.mark.parametrize("category", CATEGORIES)
def test_volunteered_axle_capacity_reaches_the_rerank_in_every_category(category):
    state = _state_after_axle_answer(category)
    slots = state.get("slots", {})
    filters = _build_metadata_filters(state)

    assert slots.get("axle_capacity_lbs") == 7000.0, "the slot must be stored off-spec"
    assert filters.get("axle_capacity_lbs") == 7000.0, "and become a filter target"
    assert _required_axle_capacity_lbs_from_filters(slots, filters) == 7000.0


@pytest.mark.parametrize("category", CATEGORIES)
def test_axle_capacity_is_never_a_hard_sql_gate(category):
    """It has to ORDER results, not empty the screen: only ~70% of the catalogue carries a
    rating, so gating the query on it would hide the other 30% outright."""
    state = _state_after_axle_answer(category)
    hard_filters = _listing_filters(category, state.get("slots", {}), _build_metadata_filters(state))
    assert "axle" not in str(hard_filters).lower()


@pytest.mark.parametrize("category", CATEGORIES)
def test_an_axle_phrase_is_not_also_kept_as_a_search_feature(category):
    """"7000 lb axles" is the axle slot and nothing else - as a feature it matches nearly
    every tandem trailer and can only ever score 0 in the feature rerank."""
    state = _state_after_axle_answer(category)
    features = state.get("requested_features") or []
    assert not any("axle" in str(feature).lower() for feature in features)
