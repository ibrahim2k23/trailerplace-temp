from __future__ import annotations

import re
from typing import Any

from src.domain.categories import resolve_categories_from_text
from src.domain.normalizer import normalize_category
from src.domain.units import (
    _range_candidates,
    parse_dimensions,
    parse_length_ft_loose,
    parse_weight_lbs_loose,
)

_ALLOWED_HITCH_TYPES = {"Gooseneck", "Bumper Pull"}

# What KIND of value each slot itself holds. `_SLOT_METADATA_FILTER_MAP` says where a
# slot's answer gets copied for filtering; this says how the slot's OWN value should be
# parsed before storing, so a numeric slot never ends up holding raw text like "8x25".
_SLOT_VALUE_KIND = {
    "haul_length_ft": "length_ft",
    "trailer_length_ft": "length_ft",
    "vehicle_length_ft": "length_ft",
    "item_or_trailer_width_ft": "width_ft",
    "trailer_width_ft": "width_ft",
    "trailer_height_ft": "height_ft",
    "haul_weight_lbs": "payload_lbs",
    "payload_lbs": "payload_lbs",
    "payload_need": "payload_lbs",
    "total_weight": "payload_lbs",
    "bin_size": "length_ft",
    "hitch_type": "hitch_type",
    "base_category": "subcategory",
}

# Real answers are phrases ("goose neck please"), not the exact HITCH_MAP keys - match the
# known terms as a word-bounded substring, longest first so "bumper pull" outranks a
# coincidental shorter match.
_HITCH_TERM_ORDER = sorted(
    {
        "bumper pull": "Bumper Pull", "bumperpull": "Bumper Pull", "bumper-pull": "Bumper Pull",
        "tag along": "Bumper Pull", "tag-along": "Bumper Pull",
        "gooseneck": "Gooseneck", "goose neck": "Gooseneck", "goose-neck": "Gooseneck",
    }.items(),
    key=lambda pair: -len(pair[0]),
)


def normalize_hitch_answer(raw_answer: Any) -> Any:
    """Canonicalize a free-text hitch answer to the same ``["Gooseneck"]``-style
    single-item list the LLM's constrained extraction produces for a clear preference -
    so a raw slot_answers entry can never downgrade that clean value to unnormalized text.

    "either"/"any"/"no preference", a genuine typo the term list doesn't catch, or any
    other unrecognized text all mean the same thing here: no specific hitch preference
    was given, so the slot is null - never invented, never a middle-ground list of both.
    """
    text = str(raw_answer or "").strip().lower()
    if not text:
        return None
    for term, canonical in _HITCH_TERM_ORDER:
        if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text):
            return [canonical]
    return None


def normalize_subcategory_answer(raw_answer: Any) -> Any:
    """Only a recognizable trailer-type name counts as an Aluminum ``base_category``
    answer. The question is open-ended ("utility, equipment, enclosed, or something
    else?"), but a vague reply to that ("not sure", "whatever's in stock") is not a real
    preference, so it resolves to null rather than being stored as junk text. Reuses
    categories.py's own term list so this can never recognize a name the category
    resolver itself wouldn't.
    """
    text = str(raw_answer or "").strip()
    if not text:
        return None
    matches = resolve_categories_from_text(text)
    return matches[0] if matches else None


_SLOT_METADATA_FILTER_MAP = {
    "base_category": ("subcategory",),
    "bin_size": ("length_ft",),
    "cargo_size": ("length_ft", "width_ft", "height_ft"),
    "haul_length_ft": ("length_ft",),
    "haul_weight_lbs": ("payload_lbs",),
    "item_or_trailer_width_ft": ("width_ft",),
    "payload_need": ("payload_lbs",),
    "trailer_length_ft": ("length_ft",),
    "trailer_height_ft": ("height_ft",),
    "trailer_size": ("length_ft", "width_ft"),
    "vehicle_length_ft": ("length_ft",),
}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _normalize_roll_off_bin_size_as_length(value: Any) -> float | None:
    """Bin size is yardage, mapped directly to a length_ft NUMBER (15 yd -> 15, not a unit
    conversion) - so it follows the same range/vague rules as any other measurement: a
    range ("15-20 yd") takes the smallest side, and a vague answer with no number at all
    ("as big as you have") is null.
    """
    if _is_number(value):
        return float(value)
    text = str(value or "").strip().lower().replace(",", "")
    if not text:
        return None
    bounds = _range_candidates(text)
    if bounds:
        nums = [float(m.group(0)) for c in bounds if (m := re.search(r"\d+(?:\.\d+)?", c))]
        if nums:
            return min(nums)
    matches = re.findall(r"\d+(?:\.\d+)?", text)
    return float(matches[0]) if matches else None


def normalize_slot_value(category: str, key: str, value: Any) -> Any:
    if _is_number(value):
        return float(value)

    if key in {"length_ft", "width_ft", "height_ft"}:
        if key == "length_ft" and normalize_category(category) == "Roll Off":
            return _normalize_roll_off_bin_size_as_length(value)
        # "8x25"/"8x25x6.5" carries multiple dimensions: pull out the one this key asks for.
        dims = parse_dimensions(value)
        if dims:
            width_ft, length_ft, height_ft = dims
            return {"width_ft": width_ft, "length_ft": length_ft, "height_ft": height_ft}[key]
        return parse_length_ft_loose(value)

    if key == "payload_lbs":
        return parse_weight_lbs_loose(value)

    return value


def slot_value_kind(slot_name: str) -> str | None:
    """The parse kind for a slot's own value ("length_ft"/"width_ft"/...), or None if free text."""
    return _SLOT_VALUE_KIND.get(slot_name)


def is_recognized_slot_value(slot_name: str, value: Any) -> bool:
    """True when ``value`` is already in the clean, parsed form for this slot's kind
    (a number for a dimension/payload slot, a single-item canonical list for hitch_type,
    a resolved category name for subcategory) rather than raw, unnormalized text. Used to
    stop a vague answer in the SAME slot from clobbering a good value another part of this
    same turn already produced."""
    kind = _SLOT_VALUE_KIND.get(slot_name)
    if kind is None:
        return False
    if kind == "hitch_type":
        return isinstance(value, list) and len(value) == 1 and value[0] in _ALLOWED_HITCH_TYPES
    if kind == "subcategory":
        return isinstance(value, str) and bool(value)
    return _is_number(value)


def normalize_answer_for_slot(category: str, slot_name: str, value: Any) -> Any:
    """The value to STORE under ``slot_name`` itself.

    Numeric slots (length/width/height/payload) keep numbers, resolving a range to its
    smallest side and falling back to any plain number in the text as a last resort; a
    truly vague answer with no extractable number is null (no preference), never raw
    text. hitch_type and subcategory (Aluminum's base_category) work the same way: a
    recognized answer is canonicalized, anything else - including "either"/"any" for
    hitch_type, or a vague reply for subcategory - is null. Free-text slots (haul_item and
    similar) have no kind here and are stored completely as-is, however vague.
    """
    kind = _SLOT_VALUE_KIND.get(slot_name)
    if kind is None:
        return value
    if kind == "hitch_type":
        return normalize_hitch_answer(value)
    if kind == "subcategory":
        return normalize_subcategory_answer(value)
    normalized = normalize_slot_value(category, kind, value)
    return normalized if _is_number(normalized) else None
