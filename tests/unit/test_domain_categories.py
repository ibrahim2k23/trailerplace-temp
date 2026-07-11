from src.domain.categories import (
    CANONICAL_CATEGORIES,
    category_clarification_question,
    resolve_category_from_text,
)
from src.domain.trailer_fields import get_trailer_fields, list_all_categories


def test_category_resolution_salience_and_guards():
    assert resolve_category_from_text("I need a dump trailer").category == "Dump"

    tilt = resolve_category_from_text("tilt trailer to haul a tractor")
    assert tilt.category == "Tilt"
    assert tilt.match_tier == "naming"

    assert resolve_category_from_text("gooseneck").category is None


def test_office_clarification_and_field_coverage():
    resolution = resolve_category_from_text("I need an office trailer")
    assert resolution.needs_clarification is True
    assert category_clarification_question(resolution.clarification_key)

    for category in CANONICAL_CATEGORIES:
        assert get_trailer_fields(category).category == category

    assert set(list_all_categories()) - set(CANONICAL_CATEGORIES) == {"Welding"}
