from __future__ import annotations

import logging
import re
from typing import Any

# (display label, listing keys in preference order)
_BULLET_FIELDS: list[tuple[str, tuple[str, ...]]] = [
    ("Category", ("category",)),
    ("Price", ("price_display", "price")),
    ("Length", ("length",)),
    ("Width", ("width",)),
    ("GVWR", ("gvwr",)),
    ("Payload Capacity", ("payload_capacity",)),
    ("Hitch Type", ("hitch_type",)),
    ("Color", ("color",)),
]

logger = logging.getLogger(__name__)

_SALES_SAFE_FALLBACK = (
    "This is one of the stronger available options to compare, with the confirmed specs shown above."
)
_NEGATIVE_MATCH_LANGUAGE_RE = re.compile(
    r"\b(?:partial match|close alternative based on|exceeds?|does not meet|doesn't meet|"
    r"not meet|mismatch|too wide|too tall|too short|too long|shorter than|longer than|"
    r"requirement(?:s)? of|required value)\b",
    re.I,
)
_FULL_MATCH_LANGUAGE_RE = re.compile(
    r"\b(?:fully|exact(?:ly)?|perfect(?:ly)?)\s+match(?:es|ed)?\b"
    r"|\bmeet(?:s)?\s+(?:your\s+)?(?:specifications|requirements|needs)\b"
    r"|\bperfect\s+for\s+(?:your\s+)?(?:specifications|requirements|needs|livestock needs|hauling needs)\b",
    re.I,
)
# Softer superlative match-claims ("strong/best/closest match"). These overclaim
# fit on a listing that is not a confirmed full match, so we hold the per-listing
# sales blurb to the same standard the intro validator (_invalid_pinecone_intro)
# applies to the batch intro text.
_STRONG_MATCH_LANGUAGE_RE = re.compile(
    r"\b(?:strong|clear|close|closest|best|top|solid)\s+match(?:es)?\b"
    r"|\bclosely\s+match(?:es)?\b"
    r"|\bbest-fitting\b",
    re.I,
)
def _first_value(listing: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = listing.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def _bullet_relevance_hint(text: str) -> dict[str, int]:
    """Light keyword hints so we order bullets toward stated customer needs."""
    low = (text or "").lower()
    scores: dict[str, int] = {label: 0 for label, _ in _BULLET_FIELDS}
    if re.search(r"\b(price|budget|\$|dollar|cost|afford)\b", low):
        scores["Price"] += 3
    if re.search(r"\b(length|ft|feet|deck|long|short)\b", low):
        scores["Length"] += 3
    if re.search(r"\b(gvwr|weight|lbs|ton|heavy|capacity)\b", low):
        scores["GVWR"] += 2
        scores["Payload Capacity"] += 2
    if re.search(r"\b(payload|carry|haul weight)\b", low):
        scores["Payload Capacity"] += 3
    if re.search(r"\b(hitch|bumper|gooseneck|pull)\b", low):
        scores["Hitch Type"] += 3
    if re.search(r"\b(color|colour|black|white|red|blue)\b", low):
        scores["Color"] += 3
    return scores


def _slots_hint_text(slots: dict[str, Any]) -> str:
    parts: list[str] = []
    for k, v in sorted((slots or {}).items()):
        if v not in (None, "", [], {}):
            parts.append(f"{k}={v}")
    return " ".join(parts).lower()


def _ordered_bullets(
    listing: dict[str, Any],
    *,
    user_message: str,
    slots: dict[str, Any],
) -> list[str]:
    hint = _bullet_relevance_hint(user_message + " " + _slots_hint_text(slots))
    hitch_slot = str(slots.get("hitch_type") or "").lower()
    if hitch_slot:
        hint["Hitch Type"] += 2

    mandatory_labels = {"Length", "Width"}
    mandatory_lines: list[str] = []
    candidates: list[tuple[int, str, str]] = []
    for label, keys in _BULLET_FIELDS:
        value = _first_value(listing, *keys)
        if value in (None, ""):
            continue
        line = f"{label}: {value}"
        if label in mandatory_labels:
            mandatory_lines.append(line)
            continue
        score = hint.get(label, 0)
        if label == "Hitch Type" and hitch_slot and str(value).lower():
            if hitch_slot in str(value).lower() or str(value).lower() in hitch_slot:
                score += 2
        candidates.append((score, label, line))

    candidates.sort(key=lambda x: (-x[0], x[1]))
    remaining_capacity = max(0, 6 - len(mandatory_lines))
    return mandatory_lines + [c[2] for c in candidates[:remaining_capacity]]


def _safe_sales_blurb(
    line: str,
    *,
    listing: dict[str, Any],
    fallback_line: str = _SALES_SAFE_FALLBACK,
) -> str:
    clean = re.sub(r"\s+", " ", str(line or "").strip())
    if clean.startswith('"') and clean.endswith('"') and len(clean) > 1:
        clean = clean[1:-1].strip()
    if not clean:
        return fallback_line
    if _NEGATIVE_MATCH_LANGUAGE_RE.search(clean):
        return fallback_line

    validation = listing.get("match_validation") if isinstance(listing.get("match_validation"), dict) else {}
    match_level = str((validation or {}).get("match_level") or "").strip().lower()
    requested = [
        str(x).strip().lower()
        for x in ((validation or {}).get("requested_non_metadata_features") or [])
        if str(x).strip()
    ]
    confirmed = [
        str(x).strip().lower()
        for x in ((validation or {}).get("confirmed_requirements") or [])
        if str(x).strip()
    ]
    low_clean = clean.lower()
    if match_level != "full" and (
        _FULL_MATCH_LANGUAGE_RE.search(clean) or _STRONG_MATCH_LANGUAGE_RE.search(clean)
    ):
        return fallback_line
    for feature in requested:
        if feature and feature in low_clean and not any(feature in c or c in feature for c in confirmed):
            return fallback_line
    if not clean.endswith((".", "!", "?")):
        clean += "."
    return clean


def _why_it_fits_body(
    listing: dict[str, Any],
    *,
    category: str | None,
    slots: dict[str, Any],
    user_message: str,
) -> str:
    """Return one natural, customer-facing fit note."""
    validation = listing.get("match_validation") if isinstance(listing.get("match_validation"), dict) else {}
    sales_blurb = _safe_sales_blurb(
        str((validation or {}).get("sales_blurb") or ""),
        listing=listing,
        fallback_line="",
    )
    if sales_blurb:
        return sales_blurb
    match_level = str((validation or {}).get("match_level") or "").strip().lower()
    missing = [
        str(x).strip()
        for x in ((validation or {}).get("missing_or_unconfirmed_requirements") or [])
        if str(x).strip()
    ]
    focus = "your needs"
    for key in (
        "haul_item",
        "vehicle_type",
        "haul_material",
        "use_case",
        "fiber_use_case",
        "base_category",
    ):
        if slots.get(key):
            focus = str(slots[key]).strip()
            break

    feature_bits: list[str] = []
    if listing.get("length"):
        feature_bits.append(f"{listing.get('length')} length")
    if listing.get("hitch_type"):
        feature_bits.append(f"{listing.get('hitch_type')} hitch")
    if listing.get("payload_capacity"):
        feature_bits.append(f"{listing.get('payload_capacity')} payload capacity")
    elif listing.get("gvwr"):
        feature_bits.append(f"{listing.get('gvwr')} GVWR")
    if listing.get("color"):
        feature_bits.append(f"{listing.get('color')} finish")

    top_features = ", ".join(feature_bits[:3])

    if match_level in {"alternative", "unknown"} and missing:
        if top_features:
            return f"This option is worth comparing for its confirmed strengths, including {top_features}."
        return _SALES_SAFE_FALLBACK
    if match_level == "partial":
        if top_features:
            return f"This option brings useful strengths to the table, including {top_features}."
        return _SALES_SAFE_FALLBACK
    if category and top_features:
        return f"A strong {category.lower()} match for {focus}, with {top_features}."
    if category:
        return f"A strong {category.lower()} match for {focus}, and well-aligned with what you asked for."
    if top_features:
        return f"A strong match for {focus}, with {top_features}."
    if (user_message or "").strip():
        return "A solid option based on your request and the specs available on this listing."
    return "A strong option based on your current search filters."


_NO_SPECS_BULLET = "- *(No spec fields on this listing.)*"
LISTING_CARD_SEPARATOR = "\n\n---\n\n"


def render_listing_card(
    index: int,
    *,
    title: str,
    url: str = "",
    bullet_lines: list[str],
    why_line: str | None = None,
) -> str:
    """Single source of truth for one trailer card's layout.

    Both the Pinecone search path (``format_listing_results``) and the direct
    inventory-lookup path (``inventory_matcher._format_listing_block``) render
    cards through this so the header/bullet/separator structure can't drift.
    ``bullet_lines`` are already ``- ``-prefixed; ``why_line`` is the optional
    match-fit note that only the search path supplies.
    """
    title_txt = str(title or "").strip() or "Trailer listing"
    url_txt = str(url or "").strip()
    header = (
        f"Trailer #{index}: [{title_txt}]({url_txt})"
        if url_txt
        else f"Trailer #{index}: {title_txt}"
    )
    body = "\n".join(bullet_lines) if bullet_lines else _NO_SPECS_BULLET
    parts = [header, "", body]
    if why_line:
        parts.extend(["", why_line])
    return "\n".join(parts)


def format_listing_results(
    listings: list[dict[str, Any]],
    *,
    category: str | None,
    slots: dict[str, Any],
    user_message: str,
) -> str:
    if not listings:
        return (
            "I could not find a strong inventory match with those details yet. "
            "If you can loosen one requirement, I can search again."
        )

    sections: list[str] = []
    for i, listing in enumerate(listings, 1):
        raw_bullets = _ordered_bullets(listing, user_message=user_message, slots=slots or {})
        bullet_lines = [f"- {text}" for text in raw_bullets]

        why = _why_it_fits_body(
            listing,
            category=category,
            slots=slots or {},
            user_message=user_message,
        )

        sections.append(render_listing_card(
            i,
            title=str(listing.get("title") or "Trailer listing").strip(),
            url=str(listing.get("url") or "").strip(),
            bullet_lines=bullet_lines,
            why_line=why,
        ))

    return LISTING_CARD_SEPARATOR.join(sections)
