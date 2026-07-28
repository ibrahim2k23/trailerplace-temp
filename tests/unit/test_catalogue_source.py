"""The catalogue the prompts advertise must be the catalogue search can return.

Brands, stocked categories and direct lookups all read trailer_listings, so an
advertised brand is by construction a brand search can find. These tests pin
that wiring, plus the fallbacks that keep the bot answering when the database is
unavailable — without a fallback, an unreachable database would have the bot
claim we stock nothing at all.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.domain import brands
from src.domain.categories import CANONICAL_CATEGORIES, advertised_categories_line


@pytest.fixture(autouse=True)
def _clear_catalogue_caches():
    """load_make_inventory is lru_cached for the process; isolate each test."""
    brands.load_make_inventory.cache_clear()
    yield
    brands.load_make_inventory.cache_clear()


def _rows(*pairs: tuple[str, str]):
    return list(pairs)


def test_makes_come_from_the_listings_table_when_available():
    with patch.object(brands, "_make_category_pairs_from_db", return_value=_rows(
        ("Diamond C", "Dump"), ("Diamond C", "Equipment"), ("Galyean", "Livestock"),
    )), patch.object(brands, "_make_category_pairs_from_workbook") as workbook:
        inventory = brands.load_make_inventory()

    assert inventory.canonical_makes == ("Diamond C", "Galyean")
    assert inventory.categories_by_make["Diamond C"] == ("Dump", "Equipment")
    # The workbook must not even be opened when the table answered.
    workbook.assert_not_called()


def test_workbook_is_the_fallback_when_the_table_is_empty():
    with patch.object(brands, "_make_category_pairs_from_db", return_value=[]), \
         patch.object(brands, "_make_category_pairs_from_workbook",
                      return_value=_rows(("Haulmark", "Enclosed"))):
        inventory = brands.load_make_inventory()

    # An empty table means "not ingested yet" far more often than "sold out",
    # so falling back beats advertising no brands at all.
    assert inventory.canonical_makes == ("Haulmark",)


def test_a_database_error_falls_back_instead_of_raising():
    """Brand advertising must never be able to take the chatbot down."""
    with patch.object(brands, "_make_category_pairs_from_workbook",
                      return_value=_rows(("Haulmark", "Enclosed"))), \
         patch("src.db.database_enabled", return_value=True), \
         patch("src.db.get_session_factory", side_effect=RuntimeError("connection refused")):
        inventory = brands.load_make_inventory()

    assert inventory.canonical_makes == ("Haulmark",)


def test_only_categories_we_stock_are_advertised():
    with patch.object(brands, "_make_category_pairs_from_db", return_value=_rows(
        ("Diamond C", "Dump"), ("Haulmark", "Enclosed"),
    )):
        assert brands.stocked_categories() == ("Enclosed", "Dump")
        # Canonical order is preserved, not database or alphabetical order.
        assert advertised_categories_line() == "Enclosed, Dump"


def test_a_category_with_no_stock_is_not_advertised():
    with patch.object(brands, "_make_category_pairs_from_db", return_value=_rows(
        ("Diamond C", "Dump"),
    )):
        assert "Livestock" not in brands.stocked_categories()


def test_non_canonical_categories_never_reach_the_prompt():
    """The catalogue carries rows like "Welding" that nothing downstream handles."""
    with patch.object(brands, "_make_category_pairs_from_db", return_value=_rows(
        ("Diamond C", "Dump"), ("Weld Co", "Welding"),
    )):
        assert brands.stocked_categories() == ("Dump",)
        assert "Weld Co" not in brands.known_makes()


def test_empty_catalogue_still_advertises_every_canonical_category():
    """An unreachable catalogue must not make the bot claim we sell nothing."""
    with patch.object(brands, "_make_category_pairs_from_db", return_value=[]), \
         patch.object(brands, "_make_category_pairs_from_workbook", return_value=[]):
        assert brands.stocked_categories() == tuple(CANONICAL_CATEGORIES)


def test_inventory_matcher_prefers_the_table_over_the_workbook():
    from src.search import inventory_matcher as matcher

    row = SimpleNamespace(
        __table__=SimpleNamespace(columns=[SimpleNamespace(name=name) for name in (
            "title", "url", "year", "make", "model", "trim", "category", "subcategory",
            "stock_number", "price_display", "condition", "length", "width", "gvwr",
            "payload_capacity", "hitch_type", "color", "axles", "trailer_material",
            "floor", "match_evidence_text", "info_json_source",
        )]),
        title="2026 Diamond C Dump", url="https://x.test/1", year="2026", make="Diamond C",
        model="LPD", trim="", category="Dump", subcategory="", stock_number="12345",
        price_display="$18,500", condition="New", length="20 ft", width="8 ft",
        gvwr="14000 lbs", payload_capacity="9000 lbs", hitch_type="Bumper Pull",
        color="Black", axles="2", trailer_material="Steel", floor="Steel",
        match_evidence_text="Diamond C | Dump | hydraulic hoist", info_json_source="x",
    )

    matcher.prepared_inventory.cache_clear()
    try:
        with patch("src.db.database_enabled", return_value=True), \
             patch("src.search.listing_search.fetch_listings", return_value=[row]), \
             patch.object(matcher, "load_inventory") as workbook_loader:
            frame = matcher.prepared_inventory()

        assert len(frame) == 1
        # price_display becomes the card's price, matching what search returns.
        assert frame.iloc[0]["price"] == "$18,500"
        workbook_loader.assert_not_called()

        result = matcher.lookup_inventory(
            year=None, make=None, model_text=None, stock_number="12345", limit=5
        )
        assert result["match_status"] == "exact"
        match = result["matches"][0]
        assert match["stock_number"] == "12345"
        # Evidence is ingest's flattened text, so lookups and the feature
        # reranker quote identical evidence for the same trailer.
        assert match["match_evidence_text"] == "Diamond C | Dump | hydraulic hoist"
    finally:
        matcher.prepared_inventory.cache_clear()
