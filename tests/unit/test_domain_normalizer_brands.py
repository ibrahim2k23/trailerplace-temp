from pathlib import Path

import pandas as pd

from src.domain import brands
from src.domain.normalizer import (
    clean_dealer_notes,
    normalize_category,
    normalize_hitch,
    normalize_make,
)


def test_normalizers():
    assert normalize_hitch("goose neck") == "Gooseneck"
    assert normalize_category("enclose") == "Enclosed"
    assert normalize_make("dimond c") == "Diamond C"

    cleaned = clean_dealer_notes("Call 979-532-1486 at trailerplace.com. Financing and delivery available.")
    assert "979-532-1486" not in cleaned
    assert "trailerplace.com" not in cleaned
    assert "Financing and delivery available" not in cleaned


def test_brands_from_fixture_excel(tmp_path, monkeypatch):
    fixture = Path(tmp_path) / "listings.xlsx"
    pd.DataFrame(
        [
            {"make": "Diamond C Trailers", "category": "Utility"},
            {"make": "Dimond C", "category": "Dump"},
            {"make": "Gooseneck", "category": "Equipment"},
        ]
    ).to_excel(fixture, index=False)

    monkeypatch.setattr(brands, "_LISTINGS_FILE", fixture)
    brands.load_make_inventory.cache_clear()
    try:
        assert brands.categories_for_make("Diamond C") == ("Dump", "Utility")
        assert set(brands.make_filter_values("Diamond C")) >= {"Diamond C Trailers", "Diamond C"}

        block = brands.make_prompt_block()
        assert "- Diamond C: Dump, Utility" in block
        assert "- Gooseneck:" not in block
        assert "Gooseneck and Bumper Pull are strictly hitch types" in block

        from src.domain.categories import make_prompt_block

        assert make_prompt_block() == block
    finally:
        brands.load_make_inventory.cache_clear()
