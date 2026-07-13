from __future__ import annotations

import pytest

from src.domain.slot_map import brand_is_actually_a_hitch, sanitize_non_metadata_features
from src.graph.apply_analysis import apply_analysis_to_state
from src.graph.nodes.search import _build_metadata_filters
from src.graph.state import new_session_state
from tests.unit.llm_helpers import sample_analysis
from tests.unit.test_apply_analysis import _empty_extracted, say


def apply_with(state, analysis):
    state["turn"] = analysis
    return apply_analysis_to_state(state)


def test_hitch_stated_as_a_feature_becomes_a_search_filter():
    # Seen live: "it should be goose neck only as well" came back as a non_metadata_feature,
    # so no hitch filter was built and bumper-pull trailers stayed in the running.
    state = new_session_state("s1")
    state["category"] = "Livestock"
    say(state, "18ft would be nice. it should be goose neck only as well")
    apply_with(
        state,
        sample_analysis(
            intent="requirement_change",
            category_mentioned="Livestock",
            extracted={**_empty_extracted(), "trailer_length_ft": 18.0, "non_metadata_features": ["gooseneck hitch only"]},
            slot_answers=[{"slot_name": "trailer_length_ft", "raw_answer": "18ft"}],
        ),
    )
    assert state["slots"]["hitch_type"] == ["Gooseneck"]
    assert state["non_metadata_features"] == []
    assert _build_metadata_filters(state) == {"length_ft": 18.0, "hitch_type": "Gooseneck"}


def test_gooseneck_is_a_hitch_unless_they_call_it_a_brand():
    state = new_session_state("s1")
    state["category"] = "Livestock"
    say(state, "I want a gooseneck")
    apply_with(state, sample_analysis(extracted={**_empty_extracted(), "brand_preference": "Gooseneck"}, slot_answers=[]))
    assert state["brand_preference"] is None
    assert state["slots"]["hitch_type"] == ["Gooseneck"]

    other = new_session_state("s2")
    other["category"] = "Livestock"
    say(other, "I want a trailer made by Gooseneck")
    apply_with(other, sample_analysis(extracted={**_empty_extracted(), "brand_preference": "Gooseneck"}, slot_answers=[]))
    assert other["brand_preference"] == "Gooseneck"
    assert "hitch_type" not in other["slots"]


def test_a_real_make_is_still_a_make():
    state = new_session_state("s1")
    state["category"] = "Livestock"
    say(state, "I like Galyean")
    apply_with(state, sample_analysis(extracted={**_empty_extracted(), "brand_preference": "Galyean"}, slot_answers=[]))
    assert state["brand_preference"] == "Galyean"


@pytest.mark.parametrize(
    "feature, verdict",
    [
        ("gooseneck hitch only", "hitch"),
        ("bumper pull preferred", "hitch"),
        ("18 ft long", "drop"),        # a measurement belongs in its slot, not the feature list
        ("7000 lbs payload", "drop"),
        ("$20,000 budget", "drop"),
        ("under 25k", "drop"),
        ("16 ft ramps", "keep"),       # a real feature that merely mentions a size
        ("side rails", "keep"),
        ("winch", "keep"),
        ("butterfly gates", "keep"),
    ],
)
def test_only_unsearchable_preferences_stay_in_the_feature_list(feature, verdict):
    kept, hitch = sanitize_non_metadata_features([feature])
    actual = "hitch" if hitch else ("keep" if kept else "drop")
    assert actual == verdict


def test_brand_is_actually_a_hitch_reads_the_framing():
    assert brand_is_actually_a_hitch("Gooseneck", "it should be goose neck only")
    assert not brand_is_actually_a_hitch("Gooseneck", "I want the Gooseneck brand")
    assert not brand_is_actually_a_hitch("Galyean", "I like Galyean")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("insulated enclosed trailer", "insulated"),
        ("black 16 ft Cargo Craft enclosed trailer with a winch", "winch"),
        ("Diamond C equipment trailer with a rear ramp door", "rear ramp door"),
        ("gooseneck livestock trailer with butterfly gates", "butterfly gates"),
        ("white utility trailer", None),
    ],
)
def test_feature_sanitizer_removes_metadata_identity_words(raw, expected):
    kept, hitch = sanitize_non_metadata_features([raw])
    assert kept == ([expected] if expected else [])
    if raw.startswith("gooseneck"):
        assert hitch == ["Gooseneck"]


def test_combined_size_slots_store_feet_not_a_sentence():
    # trailer_size / cargo_size had no value kind, so they kept the raw sentence — which then
    # travelled verbatim into the Pinecone query text.
    state = new_session_state("s1")
    state["category"] = "Utility"
    say(state, "I'd rather keep it 18ft")
    apply_with(
        state,
        sample_analysis(
            intent="qualification_answer",
            category_mentioned="Utility",
            extracted={**_empty_extracted(), "trailer_length_ft": 18.0},
            slot_answers=[{"slot_name": "trailer_size", "raw_answer": "I'd rather keep it 18ft"}],
        ),
    )
    assert state["slots"]["trailer_size"] == 18.0
    assert state["slots"]["length_ft"] == 18.0
    assert "width_ft" not in state["slots"]


def test_weight_slots_store_pounds_not_a_sentence():
    state = new_session_state("s1")
    state["category"] = "Utility"
    say(state, "probably about 2 tons all in")
    apply_with(
        state,
        sample_analysis(
            intent="qualification_answer",
            category_mentioned="Utility",
            extracted=_empty_extracted(),
            slot_answers=[{"slot_name": "haul_weight_lbs", "raw_answer": "about 2 tons"}],
        ),
    )
    assert state["slots"]["haul_weight_lbs"] == 4000.0
    assert state["slots"]["payload_lbs"] == 4000.0
