from __future__ import annotations

import re
from typing import Any

from src.domain.categories import resolve_categories_from_text
from src.domain.normalizer import normalize_category
from src.domain.units import (
    _range_candidates,
    parse_dimensions,
    parse_length_ft,
    parse_length_ft_loose,
    parse_weight_lbs,
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
    # Combined size questions ("size preference (length / width)?"). Their own value is the
    # LENGTH in feet — the width/height they also carry are stored under their own slots.
    # Without a kind here they kept the raw sentence ("I'd rather keep it 18ft"), which then
    # travelled verbatim into the search query text.
    "cargo_size": "length_ft",
    "trailer_size": "length_ft",
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


# "I want it in Gooseneck" is a hitch. "I want a Gooseneck" is... also usually a hitch —
# even though Gooseneck is a real make we carry. We only read it as the make when the
# customer frames it as one.
_EXPLICIT_BRAND_RE = re.compile(r"\b(brand|make|manufacturer|manufactured|made\s+by)\b", re.IGNORECASE)


def brand_is_actually_a_hitch(brand: Any, user_text: str) -> bool:
    """True when a 'brand' the extractor reported is really a hitch preference.

    Gooseneck is both a hitch type and a make in our catalogue. Read as a make it silently
    narrows the search to one manufacturer; read as a hitch it filters on the thing the
    customer actually cares about. So it is only a make when they say so.
    """
    return normalize_hitch_answer(brand) is not None and not _EXPLICIT_BRAND_RE.search(user_text or "")


# A "feature" is something we CANNOT filter on. Sizes, weights, hitch type, the Aluminum
# sub-category and price all have homes of their own; parked in non_metadata_features they
# are silently dropped from the search instead of filtering it.
_PRICE_RE = re.compile(
    r"\$|\b(price[ds]?|pricing|budget|cost|costs|afford\w*)\b|\b(?:under|below|over|up\s+to)\s*\d",
    re.IGNORECASE,
)
_MEASUREMENT_WORDS = re.compile(
    r"\b(ft|foot|feet|in|inch|inches|lb|lbs|pound|pounds|ton|tons|kg|"
    r"long|length|wide|width|tall|height|high|weigh\w*|weight|payload|capacity|gvwr|"
    r"trailer|trailers|about|around|roughly|approx\w*|max|maximum|min|minimum|at|least|"
    r"under|below|over|up|to|less|more|than|"
    r"a|an|the|of|is|be|should|only|please|prefer\w*|need|want|k)\b",
    re.IGNORECASE,
)


def _is_only_a_measurement_or_price(feature: str) -> bool:
    """True when the phrase says nothing beyond a size, a weight or a price.

    "18 ft long" and "$20,000 budget" are requirements, not features. "16 ft ramps" IS a
    feature — after the number and the filler words, a real noun ("ramps") survives.
    """
    text = str(feature or "").strip().lower()
    if not text:
        return True
    is_measurement = parse_length_ft(text) is not None or parse_weight_lbs(text) is not None
    is_price = bool(_PRICE_RE.search(text))
    if not (is_measurement or is_price):
        return False
    remainder = _PRICE_RE.sub(" ", text)
    remainder = re.sub(r"[\d.,'\"×x-]+", " ", remainder)
    remainder = _MEASUREMENT_WORDS.sub(" ", remainder)
    return not re.search(r"[a-z]", remainder)


def sanitize_non_metadata_features(features: Any) -> tuple[list[str], Any]:
    """Split the extractor's feature list into real features and a hitch preference.

    Returns ``(features_to_keep, hitch_value_or_None)``. A hitch stated as a feature
    ("gooseneck hitch only") is a filter we own, not a nice-to-have, so it is lifted out;
    a bare size/weight/price is dropped (its value is already in its own slot).
    """
    kept: list[str] = []
    hitch: Any = None
    for feature in features or []:
        found = normalize_hitch_answer(feature)
        if found:
            hitch = hitch or found
            continue
        if _is_only_a_measurement_or_price(feature):
            continue
        kept.append(feature)
    return kept, hitch


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


_DIMENSION_TARGETS = {"length_ft", "width_ft", "height_ft"}


def normalize_slot_targets(category: str, slot_name: str, value: Any) -> dict[str, Any]:
    """Every Pinecone metadata target this one answer fills, with its parsed number.

    A combined size question ("Do you have a size preference (length / width)?") maps to
    several dimension targets, but a lone number answering it states ONE of them - and it
    is the length. Copying that number into width_ft (and height_ft) too would invent a
    requirement the customer never gave and wreck the fit rerank, so a single number only
    spreads across the targets when the answer really does carry several dimensions
    ("8x25", "8x20x7").
    """
    targets = _SLOT_METADATA_FILTER_MAP.get(slot_name, ())
    dimension_targets = set(targets) & _DIMENSION_TARGETS
    one_number_only = len(dimension_targets) > 1 and parse_dimensions(value) is None
    filled: dict[str, Any] = {}
    for target in targets:
        if one_number_only and target in {"width_ft", "height_ft"}:
            continue
        parsed = normalize_slot_value(category, target, value)
        if parsed is not None:
            filled[target] = float(parsed) if _is_number(parsed) else parsed
    return filled


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
