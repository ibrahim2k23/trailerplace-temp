from __future__ import annotations

from src.graph.apply_analysis import lookup_requested as _lookup_requested
from src.graph.build import _route
from src.graph.state import new_session_state
from tests.unit.llm_helpers import sample_analysis


def _lookup(**overrides) -> dict:
    inventory = {
        "is_lookup": True,
        "year": None,
        "make": "Diamond C",
        "model_text": "Lpx",
        "stock_number": None,
        "wants": "general",
        "confidence": "high",
    }
    inventory.update(overrides)
    return inventory


def _state_with(analysis) -> dict:
    state = new_session_state("s1")
    state["turn"] = analysis
    return state


def test_lookup_routes_on_its_own_intent():
    state = _state_with(sample_analysis(intent="inventory_lookup", inventory_lookup=_lookup()))
    assert _route(state) == "inventory_lookup"


def test_lookup_rides_along_with_contact_info_provided():
    # "My name is Ibrahim ... I am looking for Diamond C Lpx" — the analyze prompt keeps the
    # dominant intent on the contact handover, but the lookup must still fire.
    state = _state_with(sample_analysis(intent="contact_info_provided", inventory_lookup=_lookup()))
    assert _route(state) == "inventory_lookup"


def test_lookup_rides_along_with_general_question():
    # "What about Diamond C Lpx?"
    state = _state_with(sample_analysis(intent="general_question", inventory_lookup=_lookup()))
    assert _route(state) == "inventory_lookup"


def test_lookup_rides_along_mid_qualification():
    state = _state_with(sample_analysis(intent="qualification_answer", inventory_lookup=_lookup()))
    state["category"] = "Dump"
    state["pending_question_slot"] = "haul_material"
    assert _route(state) == "inventory_lookup"


def test_year_make_lookup_routes():
    state = _state_with(
        sample_analysis(intent="inventory_lookup", inventory_lookup=_lookup(year=2025, model_text=None))
    )
    assert _route(state) == "inventory_lookup"


def test_stock_only_lookup_routes():
    state = _state_with(
        sample_analysis(intent="inventory_lookup", inventory_lookup=_lookup(make=None, model_text=None, stock_number="02570"))
    )
    assert _route(state) == "inventory_lookup"


def test_make_alone_is_not_a_lookup():
    # A bare brand is a brand preference, never an identifier lookup — even if the
    # extractor wrongly flags is_lookup=true.
    analysis = sample_analysis(intent="general_question", inventory_lookup=_lookup(model_text=None))
    assert _lookup_requested(analysis) is False
    assert _route(_state_with(analysis)) != "inventory_lookup"


def test_low_confidence_never_routes_to_lookup():
    analysis = sample_analysis(intent="inventory_lookup", inventory_lookup=_lookup(confidence="low"))
    assert _lookup_requested(analysis) is False


def test_no_lookup_block_routes_normally():
    state = _state_with(sample_analysis(intent="general_question"))
    assert _route(state) != "inventory_lookup"


def _empty_extracted() -> dict:
    return {
        "trailer_length_ft": None, "trailer_width_ft": None, "trailer_height_ft": None,
        "payload_lbs": None, "axle_capacity_lbs": None, "hitch_type": None, "haul_item": None,
        "brand_preference": None, "non_metadata_features": [], "numeric_no_preference": [],
    }


def _lookup_turn_state(analysis) -> dict:
    from src.graph.apply_analysis import apply_analysis_to_state

    state = new_session_state("s1")
    state["messages"] = [{"role": "user", "content": "my name is Ibrahim, I am looking for Iron Bull Dtb"}]
    state["turn"] = analysis
    apply_analysis_to_state(state)
    return state


def test_lookup_turn_does_not_record_brand_or_ask_brand_categories():
    # Seen live: "I am looking for Iron Bull Dtb" recorded brand_preference="Iron Bull" and fired
    # the which-category-for-that-brand question, which then talked over the lookup's results.
    analysis = sample_analysis(
        intent="inventory_lookup",
        category_mentioned=None,
        slot_answers=[],
        inventory_lookup=_lookup(make="Iron Bull", model_text="Dtb"),
        extracted={**_empty_extracted(), "brand_preference": "Iron Bull"},
    )
    state = _lookup_turn_state(analysis)
    assert state.get("brand_preference") is None
    assert state.get("pending_brand_categories") is None


def test_lookup_turn_clears_stale_brand_category_question():
    analysis = sample_analysis(
        intent="contact_info_provided",
        category_mentioned=None,
        slot_answers=[],
        inventory_lookup=_lookup(make="Iron Bull", model_text="Dtb"),
        extracted=_empty_extracted(),
    )
    from src.graph.apply_analysis import apply_analysis_to_state

    state = new_session_state("s1")
    state["messages"] = [{"role": "user", "content": "I am looking for Iron Bull Dtb"}]
    state["pending_brand_categories"] = {"brand": "Iron Bull", "categories": ["Dump", "Tilt"]}
    state["turn"] = analysis
    apply_analysis_to_state(state)
    assert state.get("pending_brand_categories") is None


def test_non_lookup_brand_mention_still_records_preference():
    analysis = sample_analysis(
        intent="general_question",
        category_mentioned=None,
        slot_answers=[],
        extracted={**_empty_extracted(), "brand_preference": "Iron Bull"},
    )
    from src.graph.apply_analysis import apply_analysis_to_state

    state = new_session_state("s1")
    state["messages"] = [{"role": "user", "content": "do you carry Iron Bull trailers?"}]
    state["turn"] = analysis
    apply_analysis_to_state(state)
    assert state.get("brand_preference") == "Iron Bull"
