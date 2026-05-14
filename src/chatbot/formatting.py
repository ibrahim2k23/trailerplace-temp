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
    ("Price", ("price_display", "price")),
    ("Length", ("length",)),
    ("Width", ("width",)),
    ("GVWR", ("gvwr",)),
    ("Payload Capacity", ("payload_capacity",)),
    ("Hitch Type", ("hitch_type",)),
    ("Color", ("color",)),
]

logger = logging.getLogger(__name__)


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


def _why_it_fits_body(
    listing: dict[str, Any],
    *,
    category: str | None,
    slots: dict[str, Any],
    user_message: str,
) -> str:
    """Return one natural, customer-facing fit note."""
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

    if category and top_features:
        return f"A strong {category.lower()} match for {focus}, with {top_features}."
    if category:
        return f"A strong {category.lower()} match for {focus}, and well-aligned with what you asked for."
    if top_features:
        return f"A strong match for {focus}, with {top_features}."
    if (user_message or "").strip():
        return "A solid fit based on your request and the specs available on this listing."
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
    }

    try:
        response = _why_it_fits_llm().invoke(
            [
                SystemMessage(
                    content=(
                        "Write exactly one short customer-facing sentence for Why-it-fits. "
                        "Use specific specs from context when available. "
                        "Tone: helpful sales advisor, natural, not robotic. "
                        "Avoid repeating this is in the X category. "
                        "No markdown, no quotes, no emojis, max 26 words."
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
        if not line.endswith((".", "!", "?")):
            line += "."
        return line
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
            line1 = f"{i}. [{title}]({url})"
        else:
            line1 = f"{i}. {title}"

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
