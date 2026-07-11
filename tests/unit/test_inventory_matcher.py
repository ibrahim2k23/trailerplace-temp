from __future__ import annotations

import pandas as pd
import pytest

from src.search import inventory_matcher as matcher


def _fixture_df() -> pd.DataFrame:
    rows = [
        {
            "title": "2026 Iron Bull FHG24K Dump Trailer",
            "url": "https://example.com/1",
            "year": 2026,
            "make": "Iron Bull Trailers",
            "model": "FHG24K",
            "trim": "",
            "category": "Dump",
            "subcategory": "",
            "stock_number": "12914",
            "price": "$18,500",
            "condition": "New",
            "length": "14",
            "width": "83 inches",
            "gvwr": "24000",
            "payload_capacity": "18000",
            "hitch_type": "Bumper Pull",
            "color": "Black",
            "axles": "2",
            "trailer_material": "Steel",
            "floor": "Steel",
            "dealer_notes": "Call 979-532-1486 for details.",
            "info_specs_json": "",
        },
        {
            "title": "2025 Iron Bull FHG20K Dump Trailer",
            "url": "https://example.com/2",
            "year": 2025,
            "make": "Iron Bull Trailers",
            "model": "FHG20K",
            "trim": "",
            "category": "Dump",
            "subcategory": "",
            "stock_number": "12915",
            "price": "$16,900",
            "condition": "New",
            "length": "14",
            "width": "83 inches",
            "gvwr": "20000",
            "payload_capacity": "15000",
            "hitch_type": "Bumper Pull",
            "color": "Black",
            "axles": "2",
            "trailer_material": "Steel",
            "floor": "Steel",
            "dealer_notes": "",
            "info_specs_json": "",
        },
        {
            "title": "2026 Diamond C 7210S-BT Dump Trailer 8.5'X18'",
            "url": "https://example.com/3",
            "year": 2026,
            "make": "Diamond C",
            "model": "7210S-BT",
            "trim": "",
            "category": "Dump",
            "subcategory": "",
            "stock_number": "30001",
            "price": "$21,000",
            "condition": "New",
            "length": "18",
            "width": "102 inches",
            "gvwr": "21000",
            "payload_capacity": "16000",
            "hitch_type": "Bumper Pull",
            "color": "Grey",
            "axles": "2",
            "trailer_material": "Steel",
            "floor": "Steel",
            "dealer_notes": "",
            "info_specs_json": "",
        },
        {
            "title": "2026 Diamond C 7210-BT Dump Trailer",
            "url": "https://example.com/4",
            "year": 2026,
            "make": "Diamond C",
            "model": "7210-BT",
            "trim": "",
            "category": "Dump",
            "subcategory": "",
            "stock_number": "30002",
            "price": "$20,500",
            "condition": "New",
            "length": "18",
            "width": "102 inches",
            "gvwr": "21000",
            "payload_capacity": "16000",
            "hitch_type": "Bumper Pull",
            "color": "Grey",
            "axles": "2",
            "trailer_material": "Steel",
            "floor": "Steel",
            "dealer_notes": "",
            "info_specs_json": "",
        },
        {
            "title": "2023 Diamond C TSB 7K Utility Trailer",
            "url": "https://example.com/5",
            "year": 2023,
            "make": "Diamond C",
            "model": "TSB7K",
            "trim": "",
            "category": "Utility",
            "subcategory": "",
            "stock_number": "40010",
            "price": "$9,800",
            "condition": "New",
            "length": "16",
            "width": "83 inches",
            "gvwr": "7000",
            "payload_capacity": "5000",
            "hitch_type": "Bumper Pull",
            "color": "Black",
            "axles": "2",
            "trailer_material": "Aluminum",
            "floor": "Aluminum",
            "dealer_notes": "",
            "info_specs_json": "",
        },
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def prepared_df():
    return matcher.prepare_inventory(_fixture_df())


@pytest.fixture(autouse=True)
def _patch_prepared_inventory(monkeypatch, prepared_df):
    monkeypatch.setattr(matcher, "prepared_inventory", lambda: prepared_df)


def test_stock_number_exact_match():
    result = matcher.lookup_inventory(year=None, make=None, model_text=None, stock_number="12914")
    assert result["match_status"] == "exact"
    assert result["matches"][0]["stock_number"] == "12914"


def test_year_make_returns_all_matching_rows():
    result = matcher.lookup_inventory(year=2026, make="Diamond C", model_text=None, stock_number=None)
    assert result["match_status"] == "exact"
    urls = {m["url"] for m in result["matches"]}
    assert "https://example.com/3" in urls
    assert "https://example.com/4" in urls


def test_make_model_partial_iron_bull_fhg():
    result = matcher.lookup_inventory(year=2026, make="Iron Bull Trailers", model_text="FHG 24K", stock_number=None)
    assert result["matches"]
    assert result["matches"][0]["stock_number"] == "12914"


def test_model_typo_7210_bt_matches_diamond_c_row(prepared_df):
    # Narrow the fixture to a single "7210" row so the typo match is deterministic
    # (a separate test covers the multi-candidate ambiguous case).
    single = prepared_df[prepared_df["stock_number"] == "30001"]
    result = matcher.match_inventory(
        matcher._Identifiers(possible_make="Diamond C", possible_model_text="7210 bt", possible_model_code="bt"),
        df=single,
    )
    assert result["top_matches"]
    assert result["top_matches"][0]["stock_number"] == "30001"


def test_no_exact_year_falls_back_to_same_make_other_year():
    result = matcher.lookup_inventory(year=2030, make="Iron Bull Trailers", model_text="FHG24K", stock_number=None)
    assert result["match_status"] == "no_exact"
    assert result["matches"]
    assert result["matches"][0]["make"] == "Iron Bull Trailers"


def test_two_close_models_are_ambiguous():
    result = matcher.lookup_inventory(year=2026, make="Diamond C", model_text="7210 BT", stock_number=None)
    assert result["match_status"] == "ambiguous"
    assert len(result["matches"]) > 1


def test_card_dict_shape_matches_app_parser():
    from src.models import TrailerListing

    result = matcher.lookup_inventory(year=None, make=None, model_text=None, stock_number="12914")
    d = result["matches"][0]
    listing = TrailerListing(
        listing_id=str(d.get("url") or d.get("title") or ""),
        title=str(d.get("title") or ""),
        condition=str(d.get("condition") or "New"),
        price=d.get("price"),
        price_display=str(d.get("price") or "") or None,
        payments_from=None,
        category_subcategory=str(d.get("category") or ""),
        make=str(d.get("make") or ""),
        color=str(d.get("color") or ""),
        hitch_type=d.get("hitch_type"),
        year=d.get("year"),
        length=d.get("length"),
        width=d.get("width"),
        axles=d.get("axles"),
        gvwr=d.get("gvwr"),
        payload_capacity=d.get("payload_capacity"),
        trailer_material=d.get("material"),
        floor=d.get("floor"),
        url=str(d.get("url") or ""),
        score=d.get("relevance_score"),
    )
    assert listing.color == "Black"
    assert listing.trailer_material == "Steel"
    assert listing.axles == "2"


def test_normalize_text_cleans_mojibake_and_dimension_forms():
    assert matcher.normalize_text("8.5'X18'") == "8 5 x18"
    assert matcher.normalize_text("Trailers & Co�") == "trailer and co"


def test_requested_label_uses_year_make_model():
    result = matcher.lookup_inventory(year=2099, make="Nonexistent", model_text="ZZZ", stock_number=None)
    assert result["requested_label"] == "2099 Nonexistent ZZZ"
