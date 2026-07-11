import pytest

from src.domain.slot_map import (
    _SLOT_METADATA_FILTER_MAP,
    normalize_answer_for_slot,
    normalize_hitch_answer,
    normalize_slot_value,
    normalize_subcategory_answer,
)
from src.domain.trailer_fields import list_all_categories, get_trailer_fields
from src.domain.units import parse_dimensions, parse_length_ft, parse_length_ft_loose, parse_weight_lbs, parse_weight_lbs_loose


@pytest.mark.parametrize(
    "text, expected",
    [
        ("8x25", (8.0, 25.0, None)),
        ("25x8", (8.0, 25.0, None)),           # either order: width is the smaller number
        ("8' x 25'", (8.0, 25.0, None)),
        ("8.5X20 ft", (8.5, 20.0, None)),
        ("7 by 16", (7.0, 16.0, None)),
        ("8x25x6.5", (8.0, 25.0, 6.5)),         # a=width, b=length, c=height - positional, not sorted
        ("20x8x7 ft", (20.0, 8.0, 7.0)),
        ("25 ft", None),
        ("as long as possible", None),
        ("20 yard bin", None),
    ],
)
def test_parse_dimensions(text, expected):
    assert parse_dimensions(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("15-18 ft", 15.0),
        ("15 to 18 ft", 15.0),
        ("15 through 18 feet", 15.0),
        ("25-20 ft", 20.0),          # reversed order still yields the smaller value
        ("20-25", 20.0),             # bare numeric range, no unit
        ("around 20 something", 20.0),   # loose last-resort: any embedded number
        ("maybe 20?", 20.0),
        ("as long as you can get", None),
        ("20 ft", 20.0),
    ],
)
def test_parse_length_ft_loose(text, expected):
    assert parse_length_ft_loose(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("5,000-10,000 lbs", 5000.0),
        ("5k-10k", 5000.0),
        ("10k-5k", 5000.0),        # reversed order still yields the smaller value
        ("5000-10000", 5000.0),
        ("around 900 lbs", 900.0),
        ("as heavy as possible", None),
    ],
)
def test_parse_weight_lbs_loose(text, expected):
    assert parse_weight_lbs_loose(text) == expected


def test_size_pair_normalizes_per_target_and_per_slot():
    # "8x25" carries both dimensions; each target/slot pulls out the one it asks for.
    assert normalize_slot_value("Livestock", "length_ft", "8x25") == 25.0
    assert normalize_slot_value("Livestock", "width_ft", "8x25") == 8.0
    assert normalize_answer_for_slot("Livestock", "trailer_length_ft", "8x25") == 25.0
    assert normalize_answer_for_slot("Livestock", "trailer_width_ft", "8x25") == 8.0
    # A triple answer also yields height.
    assert normalize_slot_value("Enclosed", "height_ft", "8x20x7") == 7.0
    # Roll Off bin sizes are yardage, not a WxL pair.
    assert normalize_slot_value("Roll Off", "length_ft", "20 yard") == 20.0
    # A numeric slot with an unparseable answer is null (no preference), never raw text.
    assert normalize_answer_for_slot("Livestock", "trailer_length_ft", "as big as you have") is None
    # A range resolves to its smallest side even through the slot-answer path.
    assert normalize_answer_for_slot("Livestock", "trailer_length_ft", "15-18 ft") == 15.0
    # A free-text slot is never coerced.
    assert normalize_answer_for_slot("Livestock", "haul_item", "cattle") == "cattle"


def test_hitch_answer_normalization():
    assert normalize_hitch_answer("Gooseneck") == ["Gooseneck"]
    assert normalize_hitch_answer("goose neck please") == ["Gooseneck"]
    assert normalize_hitch_answer("bumper pull") == ["Bumper Pull"]
    # Either/any/vague/unrecognized-typo all collapse to null - no preference, not a
    # 2-item "both" list.
    assert normalize_hitch_answer("either is fine") is None
    assert normalize_hitch_answer("no preference") is None
    assert normalize_hitch_answer("goosenek") is None
    assert normalize_hitch_answer("") is None


def test_subcategory_answer_normalization():
    assert normalize_subcategory_answer("utility") == "Utility"
    assert normalize_subcategory_answer("an equipment trailer please") == "Equipment"
    # Vague replies to Aluminum's open-ended base_category question are null, not junk text.
    assert normalize_subcategory_answer("not sure") is None
    assert normalize_subcategory_answer("whatever's in stock") is None
    assert normalize_subcategory_answer("") is None


def test_slot_map_keys_exist_or_are_runtime_injected():
    slots = set()
    for category in list_all_categories():
        spec = get_trailer_fields(category)
        slots.update(spec.required)
        slots.update(spec.optional)

    missing = set(_SLOT_METADATA_FILTER_MAP) - slots
    # item_or_trailer_width_ft: runtime-injected, never a spec-declared slot.
    # trailer_height_ft: never itself an asked slot name - populated by direct LLM
    # extraction or a WxLxH answer - but still needs a metadata-filter target so height
    # reaches the fit rerank.
    assert missing == {"item_or_trailer_width_ft", "trailer_height_ft"}


def test_slot_normalization():
    assert normalize_slot_value("Roll Off", "length_ft", "15 yd") == 15.0
    assert normalize_slot_value("Utility", "length_ft", 14.0) == 14.0
    assert normalize_slot_value("Utility", "length_ft", "83 inches") == pytest.approx(6.92, rel=0.01)
    assert normalize_slot_value("Equipment", "payload_lbs", "2 tons") == 4000


def test_units_listing_string_parsers():
    assert parse_length_ft("83 inches") == pytest.approx(6.92, rel=0.01)
    assert parse_length_ft("7'6\"") == 7.5
    assert parse_length_ft("20 ft") == 20
    assert parse_weight_lbs("2 tons") == 4000
    assert parse_weight_lbs("5k") == 5000
    assert parse_weight_lbs("7,000 lbs") == 7000
