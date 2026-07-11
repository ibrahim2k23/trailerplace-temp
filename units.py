"""Single source of truth for parsing weight/length text into numbers.

Previously two near-duplicate parsers existed: ``ingest.parse_lbs`` /
``parse_length_ft`` (used to write numeric metadata into the Pinecone index) and
``pinecone_search._parse_number`` / ``_parse_length_ft`` (used at query time and,
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
