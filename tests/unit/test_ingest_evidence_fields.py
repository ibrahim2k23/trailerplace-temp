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


def test_axle_count_is_ingested_from_the_workbook_column():
    """New column: the scraper extracts it, recovering it from the title, model
    and feature lines on the 94 of 262 listings that carry no axle fields."""
    record = build_record(_row(axle_count=2, axles="2"), 0)
    assert record["axle_count"] == 2


def test_axle_count_is_recovered_where_the_listing_had_no_axles_field():
    record = build_record(_row(axle_count=3), 0)
    assert record["axle_count"] == 3, "this is the whole point of extracting it"


def test_a_count_written_back_as_a_float_still_reads_as_a_whole_number():
    """Excel round-trips a column with any gap in it as float, so "2.0" arrives
    as often as "2"."""
    assert build_record(_row(axle_count=2.0), 0)["axle_count"] == 2
    assert build_record(_row(axle_count="2.0"), 0)["axle_count"] == 2


def test_a_capacity_in_the_count_column_is_dropped():
    """The mirror of the "2 lbs" corruption: "axles": "8000" is a rating, not a
    count, and nothing on the lot has 8000 axles."""
    record = build_record(_row(axles="8000"), 0)
    assert record["axle_count"] is None
    assert record["axles"] == "8000", "the raw text is still published as-is"


def test_the_axles_text_is_the_fallback_for_workbooks_predating_the_column():
    record = build_record(_row(axles="2"), 0)
    assert record["axle_count"] == 2


def test_axle_count_participates_in_the_content_hash():
    """Without this, an incremental re-ingest silently skips every row whose only
    change is the newly extracted count."""
    with_count = build_record(_row(axle_count=2), 0)
    without = build_record(_row(), 0)
    assert with_count["content_hash"] != without["content_hash"]


def test_total_axle_capacity_is_the_count_times_the_per_axle_rating():
    """Two 7,500 lb axles carry 15,000 lb between them."""
    record = build_record(_row(axle_count=2, axle_capacity="7500 lbs"), 0)
    assert record["total_axle_capacity_lbs_num"] == 15000.0
    assert record["axle_capacity_lbs_num"] == 7500.0, "the per-axle rating is unchanged"


def test_total_axle_capacity_needs_both_parts():
    """A capacity with no count must not silently assume two axles."""
    capacity_only = build_record(_row(axle_capacity="5200 lbs"), 0)
    assert capacity_only["axle_capacity_lbs_num"] == 5200.0
    assert capacity_only["total_axle_capacity_lbs_num"] is None

    count_only = build_record(_row(axle_count=2), 0)
    assert count_only["axle_count"] == 2
    assert count_only["total_axle_capacity_lbs_num"] is None


def test_a_dropped_corrupt_capacity_leaves_no_total():
    """The "2 lbs" rows: the capacity is refused, so the total cannot be built."""
    record = build_record(_row(axle_count=2, axle_capacity="2 lbs", gvwr="14000 lbs"), 0)
    assert record["axle_capacity_lbs_num"] is None
    assert record["total_axle_capacity_lbs_num"] is None


def test_a_single_axle_total_equals_its_per_axle_rating():
    record = build_record(_row(axle_count=1, axle_capacity="3500 lbs"), 0)
    assert record["total_axle_capacity_lbs_num"] == 3500.0


def test_a_tri_axle_total_multiplies_by_three():
    record = build_record(_row(axle_count=3, axle_capacity="7000 lbs"), 0)
    assert record["total_axle_capacity_lbs_num"] == 21000.0


# ----- the axles field is not always a bare number -------------------------
#
# Every string below is a real value from the catalogue. Reading the first
# number in them takes 10,400 as a count on the second one; the range guard then
# discards it, so nothing wrong was stored, but three listings silently lost a
# count they had plainly stated.

import pytest  # noqa: E402

from src.search.ingest import parse_axle_count  # noqa: E402


@pytest.mark.parametrize("value,expected", [
    ("2 x 7,000 lb", 2),
    ("Tandem | 10,400 lbs", 2),                    # 10,400 is the total, not a count
    ("Single | Total: 3,500 lbs", 1),
    ("Tandem | Total: 7,000 lbs", 2),
    ("Type: Spring | Count: 2 | Rating: 7,000# (Dexter)", 2),
])
def test_a_described_axle_field_still_yields_its_count(value, expected):
    assert parse_axle_count(value) == expected


@pytest.mark.parametrize("value", [
    "2000# Rubber torsion axle - No brakes - Easy lube hubs",
    "3500# Rubber torsion axle (Rated at 2990#) - No brakes - Easy lube hub",
])
def test_a_rating_with_no_stated_count_yields_nothing(value):
    """One fact about the axles, and it is not how many there are."""
    assert parse_axle_count(value) is None


@pytest.mark.parametrize("value,expected", [
    (2, 2), (2.0, 2), ("2", 2), ("2.0", 2), ("  3 ", 3),
    ("2 Axles", 2), ("Tri-Axle", 3), ("2-7,000# Straight Axles", 2),
    ("SA", 1), ("TA", 2),
])
def test_the_ordinary_forms_all_read_as_numbers(value, expected):
    assert parse_axle_count(value) == expected


@pytest.mark.parametrize("value", ["8000", 11, 0, "", None, "two", "no axle data"])
def test_anything_that_is_not_a_count_yields_nothing(value):
    assert parse_axle_count(value) is None


def test_a_described_axle_field_reaches_the_stored_row_as_a_number():
    """End to end: the column is an integer, so text could never land in it -
    the risk was a real count being dropped, not text being stored."""
    record = build_record(_row(axles="Tandem | 10,400 lbs", axle_capacity="5200 lbs"), 0)
    assert record["axle_count"] == 2
    assert record["total_axle_capacity_lbs_num"] == 10400.0, (
        "and the total now agrees with the 10,400 the field stated all along"
    )


def test_the_scrapers_total_is_used_when_the_workbook_carries_one():
    record = build_record(_row(axle_count=2, axle_capacity="7500 lbs",
                               total_axle_capacity=15000), 0)
    assert record["total_axle_capacity_lbs_num"] == 15000.0


def test_an_older_workbook_without_the_column_still_gets_a_total():
    record = build_record(_row(axle_count=2, axle_capacity="7500 lbs"), 0)
    assert record["total_axle_capacity_lbs_num"] == 15000.0
