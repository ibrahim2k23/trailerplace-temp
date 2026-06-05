from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

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


def _why_it_fits_llm_enabled() -> bool:
    return (os.getenv("WHY_IT_FITS_LLM_ENABLED") or "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


@lru_cache(maxsize=1)
def _why_it_fits_llm() -> ChatOpenAI:
    model = (os.getenv("WHY_IT_FITS_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip()
    return ChatOpenAI(model=model, temperature=0.4)


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
    if match_level != "full" and _FULL_MATCH_LANGUAGE_RE.search(clean):
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


def _why_it_fits_llm_line(
    listing: dict[str, Any],
    *,
    category: str | None,
    slots: dict[str, Any],
    user_message: str,
    fallback_line: str,
) -> str:
    if not _why_it_fits_llm_enabled():
        return fallback_line

    focus = ""
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

    context = {
        "title": listing.get("title"),
        "category": category,
        "focus": focus,
        "user_message": user_message,
        "price": listing.get("price_display") or listing.get("price"),
        "length": listing.get("length"),
        "hitch_type": listing.get("hitch_type"),
        "gvwr": listing.get("gvwr"),
        "payload_capacity": listing.get("payload_capacity"),
        "color": listing.get("color"),
        "match_validation": listing.get("match_validation") if isinstance(listing.get("match_validation"), dict) else {},
    }

    try:
        response = _why_it_fits_llm().invoke(
            [
                SystemMessage(
                    content=(
                        """Write exactly one short customer-facing sentence for Why-it-fits.
Use only supplied listing context, confirmed_requirements, and match_validation. Do not invent specs, feature matches, prices, availability, discounts, condition, stock status, links, or reasons.
Tone: positive trailer sales advisor, natural, commercially smart, and not robotic. Sell the trailer honestly by highlighting confirmed strengths only.
If match_validation.match_level is full, you may confidently position the trailer as a strong fit using confirmed specs and features.
If match_validation.match_level is partial or alternative, do not claim or imply that the trailer fully matches the user's request. Instead, highlight confirmed strengths, practical value, trailer type, build quality, brand, condition, use case, or confirmed features that make it worth comparing.
Do not mention requested non-metadata features unless they appear in confirmed_requirements.
Do not mention length, width, payload capacity, or GVWR unless that specific value is explicitly confirmed as matching the user's requested value or requested range.
If length, width, payload capacity, or GVWR is different from the user's request, outside the requested range, missing, unclear, approximate, or not explicitly confirmed as matching, do not mention that field at all.
Never use negative or mismatch language such as "partial match", "close alternative", "based on", "exceeds", "does not meet", "mismatch", "missing", "unclear", "too wide", "too tall", "shorter than", "longer than", "outside", "different from", or "requirement".
Do not say it meets the user's needs, request, specifications, or criteria unless match_validation.match_level is full.
Avoid repeating "this is in the X category" or restating generic category labels unless it adds meaningful customer value.
No markdown, no bullets, no quotes, no emojis, max 26 words.
"""
                    )
                ),
                HumanMessage(content=str(context)),
            ]
        )
        line = str(response.content or "").strip()
        line = re.sub(r"\s+", " ", line)
        if not line:
            return fallback_line
        if line.startswith('"') and line.endswith('"') and len(line) > 1:
            line = line[1:-1].strip()
        return _safe_sales_blurb(line, listing=listing, fallback_line=fallback_line)
    except Exception:
        logger.exception("Why-it-fits LLM generation failed; falling back to deterministic line")
        return fallback_line


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
        title = str(listing.get("title") or "Trailer listing").strip()
        url = str(listing.get("url") or "").strip()

        if url:
            line1 = f"Trailer #{i}: [{title}]({url})"
        else:
            line1 = f"Trailer #{i}: {title}"

        raw_bullets = _ordered_bullets(listing, user_message=user_message, slots=slots or {})
        bullet_lines = [f"- {text}" for text in raw_bullets]

        why_fallback = _why_it_fits_body(
            listing,
            category=category,
            slots=slots or {},
            user_message=user_message,
        )
        why = _why_it_fits_llm_line(
            listing,
            category=category,
            slots=slots or {},
            user_message=user_message,
            fallback_line=why_fallback,
        )

        block_parts = [line1]
        if bullet_lines:
            block_parts.extend(["", "\n".join(bullet_lines)])
        else:
            block_parts.extend(["", "- *(No spec fields on this listing.)*"])
        block_parts.extend(["", why])

        sections.append("\n".join(block_parts))

    return "\n\n---\n\n".join(sections)
