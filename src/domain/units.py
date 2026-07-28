"""Single source of truth for parsing weight/length text into numbers.

Previously two near-duplicate parsers existed: ``ingest.parse_lbs`` /
``parse_length_ft`` (used to write the numeric columns of trailer_listings) and
``listing_search._parse_number`` / ``_parse_length_ft`` (used at query time and,
crucially, to re-parse the *same* raw catalog strings during reranking). When the
two diverge, a filter built at query time can disagree with the rerank's view of
the same listing.

These functions are intentionally kept **behaviour-equivalent to the ingest
parsers** so re-running ingest produces the exact same index values. The ingest
length parser was already the broader of the two (it handles yards/metres/cm/mm),
so the query side simply gains that coverage. The only additions over the ingest
weight parser are a few typo-normalisations (``lbd``/``lbss``/``punds``) carried
over from the query-side parser; these rewrite nothing in clean catalog cells, so
they do not change indexed values.
"""

from __future__ import annotations

import math
import re
from typing import Any, Optional


def _is_missing(value: Any) -> bool:
    """True for None or a float NaN (pandas cells arrive as float('nan'))."""
    return value is None or (isinstance(value, float) and math.isnan(value))


def parse_weight_lbs(value: Any) -> Optional[float]:
    """Parse weight-like text and normalize to pounds. Behaviour-equivalent to
    the original ``ingest.parse_lbs`` (plus harmless typo normalization)."""
    if _is_missing(value):
        return None
    if isinstance(value, (int, float)):
        try:
            out = float(value)
            return out if out > 0 else None
        except (TypeError, ValueError):
            return None
    s = str(value).strip().lower()
    if not s:
        return None
    s = s.replace(",", "")
    # Typo normalization carried from the query-side parser. These only match
    # malformed unit tokens, so clean catalog cells are untouched.
    s = re.sub(r"\blbd\b", "lbs", s)
    s = re.sub(r"\blbss\b", "lbs", s)
    s = re.sub(r"\bpunds\b", "pounds", s)

    m = re.search(r"\b(\d+(?:\.\d+)?)\s*(k|m)\b", s)
    if m:
        qty = float(m.group(1))
        mult = 1000.0 if m.group(2) == "k" else 1_000_000.0
        out = qty * mult
        return out if out > 0 else None

    m = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:lb|lbs|pound|pounds|#)\b", s)
    if m:
        out = float(m.group(1))
        return out if out > 0 else None
    m = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:kg|kgs|kilogram|kilograms)\b", s)
    if m:
        out = float(m.group(1)) * 2.2046226218
        return out if out > 0 else None
    m = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:ton|tons|tonne|tonnes)\b", s)
    if m:
        out = float(m.group(1)) * 2000.0
        return out if out > 0 else None

    m = re.search(r"\b(\d+(?:\.\d+)?)\b", s)
    if not m:
        return None
    out = float(m.group(1))
    return out if out > 0 else None


# "8x25", "8x25x6.5", "8' x 25'", "8.5 X 20 ft", "7 by 16" — how customers actually give a
# trailer size, optionally with a third (height) number. Neither length parser handles
# these (no unit marker, not a bare number), so an answer like "8x25" used to parse to
# None and the length was lost from both the metadata filter and the fit rerank.
# Deliberately NOT folded into parse_length_ft: that one must stay behaviour-equivalent to
# the ingest parser that wrote the index.
_DIMENSION_UNIT = r"(?:'|′|ft\.?|feet|foot)?"
_DIMENSION_TRIPLE_RE = re.compile(
    rf"(?<![\d.])(\d+(?:\.\d+)?)\s*{_DIMENSION_UNIT}\s*(?:x|×|by)\s*(\d+(?:\.\d+)?)\s*{_DIMENSION_UNIT}"
    rf"(?:\s*(?:x|×|by)\s*(\d+(?:\.\d+)?)\s*{_DIMENSION_UNIT})?",
    re.IGNORECASE,
)


def parse_dimensions(value: Any) -> Optional[tuple[float, float, Optional[float]]]:
    """Parse a size phrase into ``(width_ft, length_ft, height_ft)``; None when it isn't one.

    Two numbers ("8x25", "25x8"): order is ambiguous in how customers write it, so the
    smaller is taken as width — trailers top out around 8.5 ft wide, while length runs far
    longer.

    Three numbers ("8x25x6.5"): trailer spec sheets and the customer both use a fixed
    W x L x H convention, so these are read positionally — first is width, second is
    length, third is height — rather than sorted by size.
    """
    if _is_missing(value) or isinstance(value, (int, float)):
        return None
    match = _DIMENSION_TRIPLE_RE.search(str(value).strip().lower())
    if not match:
        return None
    first, second = float(match.group(1)), float(match.group(2))
    if first <= 0 or second <= 0:
        return None
    third_raw = match.group(3)
    if third_raw is not None:
        third = float(third_raw)
        return first, second, (third if third > 0 else None)
    return min(first, second), max(first, second), None


_RANGE_SEP_RE = r"(?:-|–|—|\bto\b|\bthrough\b)"
_RANGE_RE = re.compile(
    rf"(\d[\d.]*)\s*((?:(?!to\b|through\b)[a-z]+)*)\s*{_RANGE_SEP_RE}\s*(\d[\d.]*)\s*([a-z]+)?",
    re.IGNORECASE,
)


def _range_candidates(text: str) -> Optional[tuple[str, str]]:
    """Split "15-18 ft" / "15 to 18 feet" / "5k-10k" into two independently-parseable
    number+unit strings, e.g. ("15 ft", "18 ft"). When only one side states a unit, the
    other side borrows it (customers rarely repeat the unit on both numbers)."""
    match = _RANGE_RE.search(text)
    if not match:
        return None
    num1, unit1, num2, unit2 = match.group(1), match.group(2) or "", match.group(3), match.group(4) or ""
    if unit2 and not unit1:
        unit1 = unit2
    elif unit1 and not unit2:
        unit2 = unit1
    return f"{num1}{unit1}", f"{num2}{unit2}"


def parse_length_ft_loose(value: Any) -> Optional[float]:
    """Like ``parse_length_ft``, but for a live qualification answer rather than a catalog
    cell: a range ("15-18 ft") resolves to the SMALLEST side, and a number with no
    recognizable unit at all ("about 20 something") is still read as a last resort. Kept
    separate from ``parse_length_ft`` so ingest's stricter, unit-required parsing of
    catalog text is untouched."""
    if _is_missing(value):
        return None
    if isinstance(value, (int, float)):
        return parse_length_ft(value)
    text = str(value).strip().lower().replace(",", "")
    if not text:
        return None
    candidates = _range_candidates(text)
    if candidates:
        parsed = [p for p in (parse_length_ft(c) for c in candidates) if p is not None]
        if parsed:
            return min(parsed)
    direct = parse_length_ft(text)
    if direct is not None:
        return direct
    loose = re.search(r"\d+(?:\.\d+)?", text)
    return float(loose.group(0)) if loose else None


def parse_weight_lbs_loose(value: Any) -> Optional[float]:
    """Like ``parse_weight_lbs``, but for a live qualification answer: a range
    ("5k-10k", "5,000-10,000 lbs") resolves to the SMALLEST side. Kept separate from
    ``parse_weight_lbs`` so ingest's parsing of catalog text is untouched."""
    if _is_missing(value):
        return None
    if isinstance(value, (int, float)):
        return parse_weight_lbs(value)
    text = str(value).strip().lower().replace(",", "")
    if not text:
        return None
    candidates = _range_candidates(text)
    if candidates:
        parsed = [p for p in (parse_weight_lbs(c) for c in candidates) if p is not None]
        if parsed:
            return min(parsed)
    return parse_weight_lbs(text)


def parse_length_ft(value: Any) -> Optional[float]:
    """Parse length-like text and normalize to feet. Behaviour-equivalent to the
    original ``ingest.parse_length_ft`` (the superset covering ft/in/yd/m/cm/mm)."""
    if _is_missing(value):
        return None
    if isinstance(value, (int, float)):
        try:
            out = float(value)
            return out if out > 0 else None
        except (TypeError, ValueError):
            return None
    s = str(value).strip().lower()
    if not s:
        return None

    ft_m = re.search(r"(\d+(?:\.\d+)?)\s*(?:ft|feet|['′]|`(?!`))", s)
    in_m = re.search(r"(\d+(?:\.\d+)?)\s*(?:in|inch|inches|\"|``|″)", s)
    if ft_m:
        ft = float(ft_m.group(1))
        inches = float(in_m.group(1)) if in_m else 0.0
        out = ft + (inches / 12.0)
        return out if out > 0 else None
    if in_m:
        out = float(in_m.group(1)) / 12.0
        return out if out > 0 else None
    yd_m = re.search(r"(\d+(?:\.\d+)?)\s*(?:yd|yds|yard|yards)\b", s)
    if yd_m:
        out = float(yd_m.group(1)) * 3.0
        return out if out > 0 else None
    m_m = re.search(r"(\d+(?:\.\d+)?)\s*(?:m|meter|meters|metre|metres)\b", s)
    if m_m:
        out = float(m_m.group(1)) * 3.280839895
        return out if out > 0 else None
    cm_m = re.search(r"(\d+(?:\.\d+)?)\s*(?:cm|centimeter|centimeters|centimetre|centimetres)\b", s)
    if cm_m:
        out = float(cm_m.group(1)) / 30.48
        return out if out > 0 else None
    mm_m = re.search(r"(\d+(?:\.\d+)?)\s*(?:mm|millimeter|millimeters|millimetre|millimetres)\b", s)
    if mm_m:
        out = float(mm_m.group(1)) / 304.8
        return out if out > 0 else None
    if re.fullmatch(r"\d+(?:\.\d+)?", s):
        out = float(s)
        return out if out > 0 else None
    return None
