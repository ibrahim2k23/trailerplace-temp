from __future__ import annotations

import pandas as pd

from src.search.ingest import build_flattened_evidence_text, build_record


def test_all_flattened_info_and_specification_fields_enter_evidence_text():
    text = build_flattened_evidence_text(
        {
            "make": "Cargo Craft",
            "length": "16 ft",
            "interior_wall_thickness": "1 inch",
            "rear_door_opening": "76 inches",
            "electrical_options": ["30 amp service", "LED outlets"],
        },
        "2026 Cargo Craft Enclosed",
        ["insulated walls"],
    )

    assert "Make: Cargo Craft" in text
    assert "Length: 16 ft" in text
    assert "Interior Wall Thickness: 1 inch" in text
    assert "Rear Door Opening: 76 inches" in text
    assert "Electrical Options: 30 amp service; LED outlets" in text
    assert "Features: insulated walls" in text


def test_msrp_remains_excluded_from_evidence_text():
    text = build_flattened_evidence_text(
        {"price": "$10,000", "msrp": "$15,000"}, "Trailer", []
    )
    assert "Price: $10,000" in text
    assert "15,000" not in text


def _row(**fields) -> pd.Series:
    base = {
        "stock_number": "12345", "title": "2026 Iron Bull Utility", "url": "https://x/1",
        "category": "Utility", "make": "Iron Bull", "info_specs_json": "",
    }
    return pd.Series({**base, **fields})


def test_axle_capacity_is_ingested_from_the_workbook_column():
    """The column has always been in the workbook; ingest simply never read it."""
    record = build_record(_row(axle_capacity="3500 lbs", gvwr="7000 lbs"), 0)
    assert record["axle_capacity"] == "3500 lbs"
    assert record["axle_capacity_lbs_num"] == 3500.0


def test_axle_count_in_the_capacity_column_is_dropped():
    """Four catalogue rows carry the axle COUNT ("2 lbs") against a 14,000 lb GVWR.

    Parsed as a capacity it would rank the trailer as absurdly weak and print
    "Axle capacity: 2 lbs" on the customer's card, so both forms are dropped.
    """
    record = build_record(_row(axle_capacity="2 lbs", gvwr="14000 lbs"), 0)
    assert record["axle_capacity"] is None
    assert record["axle_capacity_lbs_num"] is None
    # The rest of the row is untouched.
    assert record["gvwr_lbs_num"] == 14000.0


def test_axle_capacity_participates_in_the_content_hash():
    """Without this, adding the field leaves every existing hash unchanged and an
    incremental re-ingest silently skips every row it should have rewritten."""
    with_axle = build_record(_row(axle_capacity="3500 lbs"), 0)
    without = build_record(_row(), 0)
    assert with_axle["content_hash"] != without["content_hash"]
    assert "Axle Capacity: 3500 lbs" in with_axle["match_evidence_text"]
