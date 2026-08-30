"""A category we hold no stock in must not be qualified for.

The respond prompt already forbade it. The model ignored that: a concrete "ask them this"
in the decision lines beat a general rule further up the prompt, so a Diesel Tank request
was asked its fuel type and then its tank capacity before reaching a search that could only
ever return nothing. Withholding the question is what makes the rule stick.
"""
from __future__ import annotations

import pytest

from src.domain.categories import CANONICAL_CATEGORIES, unstocked_categories
from src.graph.nodes.qualification import qualification_node
from src.graph.state import new_session_state

@pytest.fixture(autouse=True)
def _fresh_catalogue():
    """Read the catalogue afresh, before and after.

    load_make_inventory is lru_cached and other suites patch it. Deciding which categories
    are stocked at COLLECTION time and asserting at RUN time compared two different answers,
    which is a flaw in the test rather than in the guard.
    """
    from src.domain import brands

    brands.load_make_inventory.cache_clear()
    yield
    brands.load_make_inventory.cache_clear()


def _state(category: str) -> dict:
    state = new_session_state("s1")
    state["category"] = category
    return state


def test_an_unstocked_category_is_asked_nothing():
    for category in unstocked_categories():
        state = qualification_node(_state(category))

        assert state["turn_outcome"].get("next_question") is None, category
        assert state["turn_outcome"]["unstocked_category"] == category


def test_an_unstocked_category_never_reaches_a_search():
    """qualification_complete is what gates the search node; an empty search helps nobody."""
    for category in unstocked_categories():
        state = qualification_node(_state(category))

        assert state["qualification_complete"] is False, category


def test_a_stocked_category_is_still_qualified_normally():
    """The guard must be narrow: everything we do carry keeps its questions."""
    stocked = [c for c in CANONICAL_CATEGORIES if c not in unstocked_categories()]
    assert stocked, "the catalogue must be readable for this to test anything"
    for category in stocked:
        state = qualification_node(_state(category))

        assert state["turn_outcome"].get("unstocked_category") is None, category
        assert state["turn_outcome"].get("next_question"), category


def test_concession_asks_for_a_length():
    """The question that distinguishes an 8.5x14 from an 8.5x22 - the same width, a
    different business. Whether Concession is STOCKED depends on live data, so that is
    covered by test_a_category_with_listings_is_never_reported_unstocked instead; unit
    tests run with the database disabled and fall back to the workbook.
    """
    from src.domain.trailer_fields import get_trailer_fields

    spec = get_trailer_fields("Concession")
    assert "trailer_length_ft" in spec.required
    assert "length" in spec.questions["trailer_length_ft"].lower()


def test_the_unstocked_list_comes_from_the_live_catalogue():
    """Not a hardcoded list: a category sells out, and the answer changes with it."""
    from src.domain.brands import stocked_categories

    stocked = set(stocked_categories())
    assert stocked, "the catalogue must be readable for this guard to mean anything"
    for category in unstocked_categories():
        assert category not in stocked


# --- the alias trap ------------------------------------------------------------------------


def test_no_alias_folds_one_canonical_category_into_another():
    """The bug this guards: _CATEGORY_ALIASES mapped Concession -> Enclosed.

    The aliases exist to fold NON-canonical catalogue labels ("Landscape", "Tank") onto the
    canonical category they mean. Concession was in there from when those units were filed
    under Enclosed. Once the workbook re-tagged them, that entry relabelled every concession
    trailer before the stock count ran - so the bot reported none while holding two, and
    would have sent a paying customer elsewhere.
    """
    from src.domain.brands import _CATEGORY_ALIASES

    for label, target in _CATEGORY_ALIASES.items():
        assert label not in CANONICAL_CATEGORIES, (
            f"{label!r} is a canonical category being folded into {target!r}; "
            "that hides real stock from stocked_categories()"
        )
        assert target in CANONICAL_CATEGORIES, f"{label!r} maps to non-canonical {target!r}"


def test_a_category_with_listings_is_never_reported_unstocked():
    """Ties the claim to the catalogue rather than to any hand-maintained list."""
    from src.domain.brands import _make_category_pairs_from_db, _display_category

    pairs = _make_category_pairs_from_db()
    if not pairs:
        pytest.skip("database not reachable")
    with_listings = {
        _display_category(category) for _make, category in pairs
    } & set(CANONICAL_CATEGORIES)
    overlap = with_listings & set(unstocked_categories())
    assert not overlap, f"{overlap} have listings but are advertised as out of stock"
