"""Pinecone semantic search for trailer listings (ported from the reference ``pinecone_search.py``).

Logic is unchanged from the reference file; only import paths were updated to
the new ``src.domain.*`` layout (M1 ported ``normalizer.py``/``units.py``/
``make_aliases.py``/``make_inventory.py`` there).
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Optional

from openai import OpenAI
from pinecone import Pinecone
from rapidfuzz import fuzz

from src import config
from src.domain.brands import make_filter_values
from src.domain.make_aliases import MAKE_ALIASES as MAKE_ALIAS_MAP
from src.llm import usage
from src.domain.normalizer import normalize_category, normalize_hitch, normalize_subcategory
# Single-sourced parsers (units.py) so query/rerank parse raw catalog strings the
# same way ingest did when it wrote the index. _parse_number/_parse_length_ft are
# kept as local aliases to avoid churning the many call sites.
from src.domain.units import (
    parse_length_ft as _parse_length_ft,
    parse_weight_lbs as _parse_number,
)
from src.search.feature_ranker import ValidatedFeatureRerank, rank_non_metadata_features

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
RERANK_ENABLED = (os.getenv("RERANK_ENABLED") or "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
RERANK_WARN_RATIO = float((os.getenv("RERANK_WARN_RATIO") or "1.35").strip())
RERANK_EXTREME_RATIO = float((os.getenv("RERANK_EXTREME_RATIO") or "1.9").strip())
RERANK_LENGTH_WEIGHT = float((os.getenv("RERANK_LENGTH_WEIGHT") or "8.0").strip())
RERANK_MISSING_DIM_PENALTY = float((os.getenv("RERANK_MISSING_DIM_PENALTY") or "0.35").strip())
RERANK_VERBOSE_LOGS = (os.getenv("RERANK_VERBOSE_LOGS") or "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
MAKE_RERANK_VERBOSE_LOGS = (os.getenv("MAKE_RERANK_VERBOSE_LOGS") or "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
FEATURE_RERANK_WEIGHT = config.settings.feature_rerank_weight
FIT_RERANK_WEIGHT = config.settings.feature_fit_weight
FEATURE_MATCH_THRESHOLD = 0.70
FEATURE_RERANK_VERBOSE_LOGS = (
    os.getenv("FEATURE_RERANK_VERBOSE_LOGS") or "1"
).strip().lower() in {"1", "true", "yes", "on"}

# Stage precedence:
# 1) fit rerank (length-first, strict no under-length)
# 2) category make-priority reorder
# Strict under-length behavior remains unchanged by make-priority.
CATEGORY_MAKE_PREFERENCES: dict[str, list[str]] = {
    "Aluminum": ["Aluma"],
    "Car Hauler": ["Diamond C", "Iron Bull Trailers", "P&C"],
    "Diesel Tank": ["East Texas Trailers"],
    "Dump": ["Iron Bull Trailers", "Diamond C", "Texas Pride"],
    "Enclosed": ["Cargo Craft", "Haulmark", "Stallion"],
    "Equipment": ["Diamond C", "Iron Bull Trailers", "P&C"],
    "Fiber": ["Cargo Craft", "Haulmark", "Stallion"],
    "Flatbed": ["Diamond C", "Iron Bull Trailers", "P&C"],
    "Livestock": ["Galyean", "Gooseneck", "Calico Trailers"],
    "Race Trailer": ["Haulmark", "Cargo Craft"],
    "Roll Off": ["Iron Bull Trailers", "East Texas Trailers"],
    "Tilt": ["Diamond C", "Iron Bull Trailers", "Aluma"],
    "Utility": ["Diamond C", "Iron Bull Trailers", "East Texas Trailers", "P&C"],
}


@dataclass
class PineconeListingSearchResult:
    listings: list[dict[str, Any]]
    query_text: str
    metadata_filter: dict[str, Any] | None
    rerank_debug: dict[str, Any] = field(default_factory=dict)
    make_debug: dict[str, Any] = field(default_factory=dict)
    match_analysis: dict[str, Any] = field(default_factory=dict)


_ALLOWED_HITCH_TYPES = {"Gooseneck", "Bumper Pull"}


@lru_cache(maxsize=1)
def _openai_client() -> OpenAI:
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


@lru_cache(maxsize=1)
def _pinecone_index():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name = os.getenv("PINECONE_INDEX_NAME", "trailerplace-listings").strip()
    return pc.Index(index_name)


def _embed(text: str) -> list[float]:
    response = _openai_client().embeddings.create(model=EMBEDDING_MODEL, input=text)
    # The one embedding a search turn is allowed to make (M9 cost audit).
    tokens = getattr(response, "usage", None)
    usage.record_embedding(EMBEDDING_MODEL, prompt_tokens=getattr(tokens, "prompt_tokens", 0) or 0)
    return response.data[0].embedding


def _metadata_filter(
    category: str | None,
    slots: dict[str, Any],
    metadata_filters: dict[str, Any] | None = None,
    category_only: bool = False,
) -> dict[str, Any]:
    metadata_filters = metadata_filters or {}
    filters: list[dict[str, Any]] = []
    normalized_category = normalize_category(category) if category else None
    if category:
        filters.append({"category": {"$eq": normalized_category}})

    if category_only:
        # The last-resort pass: every hard filter but the category is dropped so we can show the
        # customer the closest alternatives instead of an empty screen. Their requirements are not
        # thrown away — they still shape the embedding query and the fit rerank, so what comes back
        # is ordered by how near it gets. Only the all-or-nothing $eq/$gte gates are gone.
        return filters[0] if filters else {}

    make_value = metadata_filters.get("make")
    if make_value:
        values = [value for value in make_filter_values(str(make_value)) if value]
        if len(values) == 1:
            filters.append({"make": {"$eq": values[0]}})
        elif values:
            filters.append({"make": {"$in": values}})

    hitch_value = metadata_filters.get("hitch_type") or slots.get("hitch_type")
    if hitch_value:
        hitch = normalize_hitch(str(hitch_value))
        if hitch in _ALLOWED_HITCH_TYPES:
            filters.append({"hitch_type": {"$eq": hitch}})
        else:
            logger.info("pinecone_hitch_filter_rejected | value=%r | normalized=%r", hitch_value, hitch)

    subcategory_value = metadata_filters.get("subcategory")
    if normalized_category == "Aluminum" and subcategory_value:
        subcategory = normalize_subcategory(str(subcategory_value))
        if subcategory:
            filters.append({"subcategory": {"$eq": subcategory}})

    min_length = (
        _parse_length_ft(metadata_filters.get("length_ft"))
        or _parse_length_ft(slots.get("haul_length_ft"))
        or _parse_length_ft(slots.get("vehicle_length_ft"))
        or _parse_length_ft(slots.get("trailer_length_ft"))
        or _parse_length_ft(slots.get("trailer_size"))
    )
    if min_length:
        filters.append({"length_ft_num": {"$gte": min_length}})

    if not filters:
        return {}
    if len(filters) == 1:
        return filters[0]
    return {"$and": filters}


def narrowing_filters_present(
    category: str | None,
    slots: dict[str, Any],
    metadata_filters: dict[str, Any] | None = None,
) -> bool:
    """Does the hard filter constrain anything BEYOND the category?

    If it does not, a category-only retry would run the identical query for a second time and come
    back just as empty — the category really has nothing left to show.
    """
    full = _metadata_filter(category, slots, metadata_filters)
    return full != _metadata_filter(category, slots, metadata_filters, category_only=True)


# The query vector is compared against listing vectors built by normalizer.build_embedding_text
# ("Title | Make: X | Category: Livestock | Length: 24 ft 0 in | ... | Details: ..."), so the query
# mirrors that shape: the customer's requirements as a spec sheet, not as chat.
_SLOT_QUERY_LABELS: dict[str, str] = {
    "make": "Make",
    "brand": "Make",
    "subcategory": "Subcategory",
    "hitch_type": "Hitch Type",
    "color": "Color",
    "axles": "Axles",
    "trailer_material": "Material",
    "floor": "Floor",
    "length_ft": "Length",
    "trailer_length_ft": "Length",
    "haul_length_ft": "Length",
    "vehicle_length_ft": "Length",
    "trailer_size": "Length",
    "width_ft": "Width",
    "trailer_width_ft": "Width",
    "item_or_trailer_width_ft": "Width",
    "height_ft": "Height",
    "payload_lbs": "Payload Capacity",
    "payload_need": "Payload Capacity",
    "haul_weight_lbs": "Payload Capacity",
    "total_weight": "Payload Capacity",
}

_QUERY_LABEL_ORDER = [
    "Make",
    "Category",
    "Subcategory",
    "Hitch Type",
    "Length",
    "Width",
    "Height",
    "Payload Capacity",
    "Color",
    "Axles",
    "Material",
    "Floor",
]

_QUERY_LABEL_UNITS = {"Length": "ft", "Width": "ft", "Height": "ft", "Payload Capacity": "lbs"}


def _join_values(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item).strip() for item in value if str(item or "").strip())
    return str(value).strip()


def _format_field_value(label: str, value: Any) -> str:
    text = _join_values(value)
    unit = _QUERY_LABEL_UNITS.get(label)
    if not unit or not text:
        return text
    if isinstance(value, bool):
        return text
    if isinstance(value, (int, float)):
        return f"{value:g} {unit}"
    return f"{text} {unit}" if re.fullmatch(r"\d+(\.\d+)?", text) else text


def _add_detail(details: list[str], text: str) -> None:
    """Add a free-text preference, keeping the most specific phrasing of it.

    Slot and feature lists overlap ("gate preferences: sliding gates" from a slot, "sliding
    gates" from the feature list); repeating the same words dilutes the embedding.
    """
    text = text.strip(" ;,")
    if not text:
        return
    lowered = text.lower()
    if any(lowered in existing.lower() for existing in details):
        return
    details[:] = [existing for existing in details if existing.lower() not in lowered]
    details.append(text)


def _detail_phrase(key: str, value: Any) -> str:
    text = _join_values(value)
    if not text:
        return ""
    label = key.replace("_", " ").strip()
    if not label or label.lower() in text.lower() or text.lower() in label.lower():
        return text
    return f"{label}: {text}"


def _query_text(
    category: str | None,
    slots: dict[str, Any],
    metadata_filters: dict[str, Any],
    requested_features: list[str] | None = None,
) -> str:
    """Embed everything the customer asked for - and nothing else.

    The raw user message is deliberately NOT prepended: it is one turn of chat ("20ft sounds
    good to me"), it drowns the accumulated requirements in conversational filler, and every
    requirement it does carry is already in slots/filters by the time we search.
    """
    fields: dict[str, str] = {}
    details: list[str] = []

    if category:
        fields["Category"] = str(category).strip()

    # Filters first, then slots: a filter is the resolved value, a slot may be the raw phrasing.
    for source in (metadata_filters or {}, slots or {}):
        for key, value in source.items():
            if value in (None, "", [], {}):
                continue
            label = _SLOT_QUERY_LABELS.get(key)
            if label:
                text = _format_field_value(label, value)
                if text:
                    fields.setdefault(label, text)
                continue
            _add_detail(details, _detail_phrase(key, value))

    for feature in requested_features or []:
        _add_detail(details, str(feature or ""))

    ordered = [label for label in _QUERY_LABEL_ORDER if fields.get(label)]
    ordered += [label for label in fields if label not in _QUERY_LABEL_ORDER]
    parts = [f"{label}: {fields[label]}" for label in ordered]
    if details:
        parts.append(f"Details: {'; '.join(details)}")
    return " | ".join(parts)


def _clean_match(match: Any) -> dict[str, Any]:
    metadata = dict(getattr(match, "metadata", None) or match.get("metadata", {}) or {})
    score = getattr(match, "score", None)
    if score is None and isinstance(match, dict):
        score = match.get("score")
    price = metadata.get("price_display") or metadata.get("price")
    raw_features = metadata.get("features") or []
    if isinstance(raw_features, str):
        raw_features = [raw_features]
    return {
        "title": metadata.get("title") or "",
        "condition": metadata.get("condition") or "New",
        "price": price,
        "price_display": metadata.get("price_display") or (f"${metadata['price']:,.0f}" if metadata.get("price") else None),
        "category": metadata.get("category") or "",
        "subcategory": metadata.get("subcategory") or "",
        "make": metadata.get("make") or "",
        "model": metadata.get("model") or "",
        "trim": metadata.get("trim") or "",
        "stock_number": metadata.get("stock_number") or metadata.get("stock") or "",
        "color": metadata.get("color") or "",
        "hitch_type": metadata.get("hitch_type"),
        "year": metadata.get("year"),
        "length": metadata.get("length"),
        "width": metadata.get("width"),
        "height": metadata.get("height"),
        "axles": metadata.get("axles"),
        "gvwr": metadata.get("gvwr"),
        "payload_capacity": metadata.get("payload_capacity"),
        "material": metadata.get("trailer_material"),
        "floor": metadata.get("floor"),
        "url": metadata.get("url") or "",
        "relevance_score": score,
        # Kept only while ranking. search_pinecone_listings strips this internal
        # evidence before results enter conversation state or the API response.
        "features": [
            str(value).strip()
            for value in raw_features
            if str(value or "").strip()
        ],
        "match_evidence_text": metadata.get("match_evidence_text") or "",
    }


def _required_length_ft_from_filters(slots: dict[str, Any], metadata_filters: dict[str, Any]) -> Optional[float]:
    return (
        _parse_length_ft(metadata_filters.get("length_ft"))
        or _parse_length_ft(slots.get("haul_length_ft"))
        or _parse_length_ft(slots.get("vehicle_length_ft"))
        or _parse_length_ft(slots.get("trailer_length_ft"))
        or _parse_length_ft(slots.get("trailer_size"))
    )


def _required_payload_lbs_from_filters(slots: dict[str, Any], metadata_filters: dict[str, Any]) -> Optional[float]:
    return _parse_number(
        metadata_filters.get("payload_lbs")
        or slots.get("haul_weight_lbs")
        or slots.get("payload_need")
        or slots.get("total_weight")
    )


def _required_width_ft_from_filters(slots: dict[str, Any], metadata_filters: dict[str, Any]) -> Optional[float]:
    return _parse_length_ft(
        metadata_filters.get("width_ft")
        or slots.get("item_or_trailer_width_ft")
        or slots.get("trailer_width_ft")
        or slots.get("width_ft")
    )


def _required_height_ft_from_filters(slots: dict[str, Any], metadata_filters: dict[str, Any]) -> Optional[float]:
    return _parse_length_ft(
        metadata_filters.get("height_ft")
        or slots.get("height_ft")
    )


def _norm_key(value: Any) -> str:
    s = str(value or "").strip().lower()
    s = re.sub(r"[^\w&]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _canonical_make(value: Any) -> str:
    key = _norm_key(value)
    if not key:
        return ""
    return MAKE_ALIAS_MAP.get(key, str(value or "").strip())


def _brand_order_for_category(
    listings: list[dict[str, Any]],
    *,
    category: str | None,
) -> tuple[list[str], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    cat = normalize_category(category) if category else None
    preferred = CATEGORY_MAKE_PREFERENCES.get(cat) or []
    preferred_norm: list[str] = []
    seen_preferred: set[str] = set()
    for item in preferred:
        canonical = _canonical_make(item)
        if canonical and canonical not in seen_preferred:
            preferred_norm.append(canonical)
            seen_preferred.add(canonical)

    buckets: dict[str, list[dict[str, Any]]] = {}
    brand_order: list[str] = []
    remainder: list[dict[str, Any]] = []

    for item in listings:
        brand = _canonical_make(item.get("make"))
        if not brand:
            remainder.append(item)
            continue
        if brand not in buckets:
            buckets[brand] = []
            brand_order.append(brand)
        buckets[brand].append(item)

    ordered_brands: list[str] = []
    seen_brands: set[str] = set()

    for brand in preferred_norm:
        if brand in buckets and brand not in seen_brands:
            ordered_brands.append(brand)
            seen_brands.add(brand)

    for brand in brand_order:
        if brand not in seen_brands:
            ordered_brands.append(brand)
            seen_brands.add(brand)

    return ordered_brands, buckets, remainder


def _brand_quota_template(brand_count: int, max_recommendations: int) -> list[int]:
    if brand_count <= 1:
        return [max_recommendations]
    if brand_count == 2:
        return [3, 3]
    return [3, 2, 1]


def _apply_category_make_preference(
    listings: list[dict[str, Any]],
    *,
    category: str | None,
    max_recommendations: int = 6,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not listings or not category:
        return listings, {"applied": False, "reason": "missing_listings_or_category"}

    cat = normalize_category(category)
    preferred = CATEGORY_MAKE_PREFERENCES.get(cat)
    if not preferred:
        return listings, {"applied": False, "reason": "no_category_preference", "category": cat}

    ordered_brands, buckets, remainder = _brand_order_for_category(listings, category=category)
    brand_count = len(ordered_brands)
    quota_template = _brand_quota_template(brand_count, max_recommendations)
    quota_brand_count = 1 if brand_count <= 1 else 2 if brand_count == 2 else 3
    quota_brands = ordered_brands[:quota_brand_count]

    ranked: list[dict[str, Any]] = []
    selected_counts: dict[str, int] = {}
    selected_urls: set[str] = set()

    def _take_from_brand(brand: str, limit: int) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        taken: list[dict[str, Any]] = []
        for item in buckets.get(brand, []):
            if len(taken) >= limit:
                break
            url = str(item.get("url") or "").strip()
            if url and url in selected_urls:
                continue
            taken.append(item)
            if url:
                selected_urls.add(url)
        return taken

    # First pass: enforce the 3/2/1 or 3/3 style quota for the highest-priority brands.
    for idx, brand in enumerate(quota_brands):
        taken = _take_from_brand(brand, quota_template[idx])
        ranked.extend(taken)
        selected_counts[brand] = len(taken)

    def _backfill_from_brand(brand: str) -> None:
        remaining = []
        already_selected = selected_counts.get(brand, 0)
        for item in buckets.get(brand, [])[already_selected:]:
            url = str(item.get("url") or "").strip()
            if url and url in selected_urls:
                continue
            remaining.append(item)
        if not remaining:
            return
        ranked.extend(remaining)
        selected_counts[brand] = selected_counts.get(brand, 0) + len(remaining)
        for item in remaining:
            url = str(item.get("url") or "").strip()
            if url:
                selected_urls.add(url)

    # Backfill from lower-priority brands first, then from any leftovers of the quota brands.
    for brand in ordered_brands[quota_brand_count:]:
        _backfill_from_brand(brand)

    if len(ranked) < max_recommendations:
        for brand in quota_brands:
            if len(ranked) >= max_recommendations:
                break
            already_selected = selected_counts.get(brand, 0)
            extras = []
            for item in buckets.get(brand, [])[already_selected:]:
                url = str(item.get("url") or "").strip()
                if url and url in selected_urls:
                    continue
                extras.append(item)
                if len(ranked) + len(extras) >= max_recommendations:
                    break
            if not extras:
                continue
            ranked.extend(extras)
            selected_counts[brand] = selected_counts.get(brand, 0) + len(extras)
            for item in extras:
                url = str(item.get("url") or "").strip()
                if url:
                    selected_urls.add(url)

    if len(ranked) < max_recommendations and remainder:
        for item in remainder:
            if len(ranked) >= max_recommendations:
                break
            url = str(item.get("url") or "").strip()
            if url and url in selected_urls:
                continue
            ranked.append(item)
            if url:
                selected_urls.add(url)

    ranked = ranked[:max_recommendations]
    selected_by_brand = {
        brand: count
        for brand, count in selected_counts.items()
        if count > 0
    }
    before_top = [str(x.get("make") or "") for x in listings[:8]]
    after_top = [str(x.get("make") or "") for x in ranked[:8]]
    logger.info(
        "make_rerank_summary | category=%r | brand_order=%s | quota_template=%s | selected_by_brand=%s | total=%s | before_top=%s | after_top=%s",
        cat,
        ordered_brands,
        quota_template,
        selected_by_brand,
        len(listings),
        before_top,
        after_top,
    )
    if MAKE_RERANK_VERBOSE_LOGS:
        for i, item in enumerate(ranked, 1):
            mk = _canonical_make(item.get("make"))
            logger.info(
                "make_rerank_item | rank=%s | title=%r | make=%r | canonical_make=%r | brand_rank=%s",
                i,
                item.get("title"),
                item.get("make"),
                mk,
                ordered_brands.index(mk) + 1 if mk in ordered_brands else None,
            )

    return ranked, {
        "applied": True,
        "category": cat,
        "brand_order": ordered_brands,
        "quota_template": quota_template,
        "selected_by_brand": selected_by_brand,
        "brand_count": brand_count,
    }


def _rerank_listings_by_fit(
    listings: list[dict[str, Any]],
    *,
    required_length_ft: Optional[float],
    required_payload_lbs: Optional[float],
    required_width_ft: Optional[float],
    required_height_ft: Optional[float],
    warn_ratio: float,
    extreme_ratio: float,
    length_weight: float,
    missing_dim_penalty: float,
    retain_all: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    needs_present = any(x is not None for x in (required_length_ft, required_payload_lbs, required_width_ft, required_height_ft))
    if not listings or not needs_present:
        return listings, {"applied": False, "reason": "missing_clear_requirements_or_no_listings"}

    entries: list[dict[str, Any]] = []
    for idx, listing in enumerate(listings, 1):
        length_ft = _parse_length_ft(listing.get("length"))
        gvwr_lbs = _parse_number(listing.get("gvwr"))
        payload_lbs = _parse_number(listing.get("payload_capacity"))
        width_ft = _parse_length_ft(listing.get("width"))
        height_ft = _parse_length_ft(listing.get("height"))

        # Weight requirement can be satisfied by payload or gvwr if payload missing.
        weight_from = "payload_capacity"
        weight_cap = payload_lbs
        if weight_cap is None:
            weight_cap = gvwr_lbs
            if weight_cap is not None:
                weight_from = "gvwr"
            else:
                weight_from = "unknown"

        length_ratio = (
            (length_ft / required_length_ft)
            if required_length_ft is not None and length_ft is not None and required_length_ft > 0
            else None
        )
        weight_ratio = (
            (weight_cap / required_payload_lbs)
            if required_payload_lbs is not None and weight_cap is not None and required_payload_lbs > 0
            else None
        )
        width_ratio = (
            (width_ft / required_width_ft)
            if required_width_ft is not None and width_ft is not None and required_width_ft > 0
            else None
        )
        height_ratio = (
            (height_ft / required_height_ft)
            if required_height_ft is not None and height_ft is not None and required_height_ft > 0
            else None
        )

        fail_count = 0
        missing_count = 0
        penalty = 0.0

        if required_length_ft is not None:
            if length_ratio is None:
                missing_count += 1
                penalty += missing_dim_penalty
            elif length_ratio < 1.0:
                # Strict under-length disallow.
                fail_count += 1
                penalty += (1.0 - length_ratio) * (length_weight * 2.0)
            else:
                over = length_ratio - 1.0
                penalty += over * length_weight
                if length_ratio > warn_ratio:
                    penalty += (length_ratio - warn_ratio) * (length_weight * 1.8)
                if length_ratio > extreme_ratio:
                    penalty += (length_ratio - extreme_ratio) * (length_weight * 3.2)

        if required_payload_lbs is not None:
            if weight_ratio is None:
                missing_count += 1
                penalty += missing_dim_penalty
            elif weight_ratio < 1.0:
                fail_count += 1
                penalty += (1.0 - weight_ratio) * 6.0
            else:
                over = weight_ratio - 1.0
                penalty += over * 1.4
                if weight_ratio > warn_ratio:
                    penalty += (weight_ratio - warn_ratio) * 2.0
                if weight_ratio > extreme_ratio:
                    penalty += (weight_ratio - extreme_ratio) * 4.0
                if weight_from == "gvwr":
                    penalty += 0.2
                    logger.info(
                        "payload_rerank_gvwr_fallback | title=%r | gvwr=%r | payload_capacity=%r",
                        listing.get("title"),
                        listing.get("gvwr"),
                        listing.get("payload_capacity"),
                    )

        if required_width_ft is not None:
            if width_ratio is None:
                missing_count += 1
                penalty += missing_dim_penalty
            elif width_ratio < 1.0:
                fail_count += 1
                penalty += (1.0 - width_ratio) * 4.0
            else:
                over = width_ratio - 1.0
                penalty += over * 1.2
                if width_ratio > warn_ratio:
                    penalty += (width_ratio - warn_ratio) * 1.8
                if width_ratio > extreme_ratio:
                    penalty += (width_ratio - extreme_ratio) * 3.2

        if required_height_ft is not None:
            if height_ratio is None:
                missing_count += 1
                penalty += missing_dim_penalty
            elif height_ratio < 1.0:
                fail_count += 1
                penalty += (1.0 - height_ratio) * 4.0
            else:
                over = height_ratio - 1.0
                penalty += over * 1.0
                if height_ratio > warn_ratio:
                    penalty += (height_ratio - warn_ratio) * 1.5
                if height_ratio > extreme_ratio:
                    penalty += (height_ratio - extreme_ratio) * 2.6

        length_overage = (
            max(0.0, float(length_ratio) - 1.0)
            if required_length_ft is not None and length_ratio is not None
            else 999.0
        )
        base_score = float(listing.get("relevance_score") or 0.0)
        entries.append(
            {
                "listing": listing,
                "fetch_pos": idx,
                "base_score": base_score,
                "penalty": round(penalty, 6),
                "fail_count": fail_count,
                "missing_count": missing_count,
                "length_ratio": None if length_ratio is None else round(length_ratio, 6),
                "weight_ratio": None if weight_ratio is None else round(weight_ratio, 6),
                "width_ratio": None if width_ratio is None else round(width_ratio, 6),
                "height_ratio": None if height_ratio is None else round(height_ratio, 6),
                "length_overage": round(length_overage, 6),
                "fit_score": round(base_score - penalty, 6),
            }
        )

    # The legacy/no-feature path drops failing entries whenever a non-failing
    # option exists. Feature-aware search sets retain_all=True so every hard-
    # filtered Pinecone candidate reaches the combined feature/fit scorer.
    nonfailing = [e for e in entries if e["fail_count"] == 0]
    fallback_pool = entries if retain_all or not nonfailing else nonfailing
    ranked_entries = sorted(
        fallback_pool,
        key=lambda e: (
            e["fail_count"] if retain_all else 0,
            e["missing_count"],
            e["length_overage"],
            e["penalty"],
            -e["base_score"],
        ),
    )
    fit_rank_by_fetch_pos = {
        entry["fetch_pos"]: rank for rank, entry in enumerate(ranked_entries, 1)
    }
    for entry in entries:
        entry["fit_rank"] = fit_rank_by_fetch_pos.get(entry["fetch_pos"])

    if RERANK_VERBOSE_LOGS:
        decision_rank = {id(e): i for i, e in enumerate(ranked_entries, 1)}
        for e in entries:
            logger.info(
                "rerank_score | fetched_pos=%s | decision_rank=%s | title=%r | base_score=%.6f | penalty=%.6f | fit_score=%.6f | length_ratio=%s | weight_ratio=%s | width_ratio=%s | height_ratio=%s | fail_count=%s | missing_count=%s",
                e["fetch_pos"],
                decision_rank.get(id(e)),
                e["listing"].get("title"),
                e["base_score"],
                e["penalty"],
                e["fit_score"],
                e["length_ratio"],
                e["weight_ratio"],
                e["width_ratio"],
                e["height_ratio"],
                e["fail_count"],
                e["missing_count"],
            )

    logger.info(
        "rerank_summary | applied=true | required_length_ft=%s | required_payload_lbs=%s | required_width_ft=%s | required_height_ft=%s | candidates=%s | kept_pool=%s",
        required_length_ft,
        required_payload_lbs,
        required_width_ft,
        required_height_ft,
        len(entries),
        len(fallback_pool),
    )

    return [e["listing"] for e in ranked_entries], {
        "applied": True,
        "required_length_ft": required_length_ft,
        "required_payload_lbs": required_payload_lbs,
        "required_width_ft": required_width_ft,
        "required_height_ft": required_height_ft,
        "candidate_count": len(entries),
        "retained_candidate_count": len(ranked_entries),
        "fit_entries": entries if retain_all else [],
    }


def _normalize_feature_phrase(value: Any) -> str:
    text = str(value or "").casefold()
    # Treat common feature abbreviations such as A/C as one token ("ac").
    text = re.sub(r"(?<=\w)/(?=\w)", "", text)
    text = re.sub(r"[^\w]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _feature_token_similarity(requested_token: str, candidate_token: str) -> float:
    """Compare word forms without a hand-maintained feature vocabulary.

    Edit similarity naturally connects productive forms such as insulation/insulated,
    electrical/electric, rails/rail, and lights/light. Very short tokens (AC, LED, TV)
    must match exactly because fuzzy matching them creates too many coincidences.
    """
    if min(len(requested_token), len(candidate_token)) <= 3:
        return 1.0 if requested_token == candidate_token else 0.0
    return fuzz.ratio(requested_token, candidate_token) / 100.0


def _feature_match_details(
    requested_features: list[str], match_sources: list[str]
) -> tuple[float, list[dict[str, Any]]]:
    stored = []
    all_candidate_tokens: list[tuple[str, str]] = []
    for source in match_sources:
        normalized = _normalize_feature_phrase(source)
        if not normalized:
            continue
        tokens = normalized.split()
        stored.append((source, tokens))
        all_candidate_tokens.extend((source, token) for token in tokens)
    details: list[dict[str, Any]] = []
    scores: list[float] = []

    for requested in requested_features:
        requested_norm = _normalize_feature_phrase(requested)
        requested_tokens = requested_norm.split()
        token_matches: list[dict[str, Any]] = []
        for requested_token in requested_tokens:
            best_source = ""
            best_candidate_token = ""
            best_similarity = 0.0
            for source, candidate_token in all_candidate_tokens:
                similarity = _feature_token_similarity(requested_token, candidate_token)
                if similarity > best_similarity:
                    best_source = source
                    best_candidate_token = candidate_token
                    best_similarity = similarity
            token_matches.append(
                {
                    "requested_token": requested_token,
                    "matched_token": best_candidate_token or None,
                    "source": best_source or None,
                    "similarity": round(best_similarity, 6),
                    "accepted": best_similarity >= FEATURE_MATCH_THRESHOLD,
                }
            )

        best_raw = (
            sum(match["similarity"] for match in token_matches) / len(token_matches)
            if token_matches
            else 0.0
        )
        # Every word in a clean feature phrase matters. This prevents "enclosed" alone
        # from satisfying "insulated enclosed" while retaining inflectional matches.
        accepted = bool(token_matches) and all(match["accepted"] for match in token_matches)
        score = best_raw if accepted else 0.0
        scores.append(score)
        best_feature = max(
            stored,
            key=lambda item: (
                sum(
                    max(
                        (_feature_token_similarity(token, candidate) for candidate in item[1]),
                        default=0.0,
                    )
                    for token in requested_tokens
                )
                / len(requested_tokens)
                if requested_tokens
                else 0.0
            ),
            default=(None, []),
        )[0]
        details.append(
            {
                "requested": requested,
                "best_stored_feature": best_feature,
                "raw_similarity": round(best_raw, 6),
                "accepted": accepted,
                "score": round(score, 6),
                "token_matches": token_matches,
            }
        )

    coverage = sum(scores) / len(scores) if scores else 0.0
    return round(coverage, 6), details


def _candidate_feature_match_sources(listing: dict[str, Any]) -> list[str]:
    """Return deduplicated searchable phrases from all candidate evidence.

    match_evidence_text is the exact readable text embedded at ingest time, so
    splitting it into fields lets feature reranking inspect core fields and
    arbitrary info/specification values in addition to the explicit features.
    """
    candidates: list[str] = []
    candidates.extend(str(value) for value in (listing.get("features") or []))
    candidates.extend(
        str(listing.get(key) or "") for key in ("title", "model", "trim")
    )
    evidence = str(listing.get("match_evidence_text") or "")
    candidates.extend(re.split(r"[|\n;]+", evidence))

    unique: dict[str, str] = {}
    for candidate in candidates:
        text = re.sub(r"\s+", " ", str(candidate or "")).strip(" ,")
        normalized = _normalize_feature_phrase(text)
        if normalized:
            unique.setdefault(normalized, text)
    return list(unique.values())


def _combined_feature_fit_rerank(
    listings: list[dict[str, Any]],
    *,
    requested_features: list[str],
    required_length_ft: Optional[float],
    required_payload_lbs: Optional[float],
    required_width_ft: Optional[float],
    required_height_ft: Optional[float],
    metadata_filter: dict[str, Any] | None,
    query_text: str,
    semantic_rerank: ValidatedFeatureRerank | None = None,
    semantic_fallback_reason: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rank every hard-filtered Pinecone candidate with an 85/15 blend.

    Unlike the legacy fit reranker, this never removes dimensionally failing
    candidates. Fit only supplies the secondary 15% ordering signal. When a
    validated GPT assessment is supplied, Python derives feature coverage from
    its evidence-grounded booleans; otherwise the deterministic matcher is the
    availability fallback.
    """
    if not listings:
        return listings, {
            "applied": False,
            "reason": "no_listings",
            "requested_features": requested_features,
        }

    _, fit_debug = _rerank_listings_by_fit(
        listings,
        required_length_ft=required_length_ft,
        required_payload_lbs=required_payload_lbs,
        required_width_ft=required_width_ft,
        required_height_ft=required_height_ft,
        warn_ratio=RERANK_WARN_RATIO,
        extreme_ratio=RERANK_EXTREME_RATIO,
        length_weight=RERANK_LENGTH_WEIGHT,
        missing_dim_penalty=RERANK_MISSING_DIM_PENALTY,
        retain_all=True,
    )
    fit_entries = fit_debug.get("fit_entries") or []
    if not fit_entries:
        # With no dimension/weight requirement, the existing order is the
        # Pinecone cosine order, so it becomes the secondary fit-order signal.
        fit_entries = [
            {
                "listing": listing,
                "fetch_pos": pos,
                "fit_rank": pos,
                "base_score": float(listing.get("relevance_score") or 0.0),
                "penalty": 0.0,
                "fail_count": 0,
                "missing_count": 0,
                "length_ratio": None,
                "weight_ratio": None,
                "width_ratio": None,
                "height_ratio": None,
                "length_overage": 999.0,
                "fit_score": float(listing.get("relevance_score") or 0.0),
            }
            for pos, listing in enumerate(listings, 1)
        ]

    candidate_count = len(fit_entries)
    scored: list[dict[str, Any]] = []
    for entry in fit_entries:
        listing = entry["listing"]
        fit_rank = int(entry.get("fit_rank") or entry["fetch_pos"])
        fit_order_score = (
            1.0
            if candidate_count == 1
            else 1.0 - ((fit_rank - 1) / (candidate_count - 1))
        )
        stored_features = list(listing.get("features") or [])
        match_sources = _candidate_feature_match_sources(listing)
        candidate_id = f"C{int(entry['fetch_pos']):03d}"
        semantic_source = "deterministic_fallback"
        if semantic_rerank is not None and candidate_id in semantic_rerank.assessments_by_id:
            assessment = semantic_rerank.assessments_by_id[candidate_id]
            matched_count = sum(1 for match in assessment.feature_matches if match.matched)
            feature_coverage = round(matched_count / len(requested_features), 6)
            feature_matches = [match.model_dump() for match in assessment.feature_matches]
            semantic_source = "gpt_semantic"
        else:
            feature_coverage, feature_matches = _feature_match_details(
                requested_features, match_sources
            )
        final_score = (
            FEATURE_RERANK_WEIGHT * feature_coverage
            + FIT_RERANK_WEIGHT * fit_order_score
        )
        scored.append(
            {
                "listing": listing,
                "candidate_id": candidate_id,
                "fetch_pos": entry["fetch_pos"],
                "fit_rank": fit_rank,
                "pinecone_score": float(listing.get("relevance_score") or 0.0),
                "feature_coverage": round(feature_coverage, 6),
                "fit_order_score": round(fit_order_score, 6),
                "final_score": round(final_score, 6),
                "feature_matches": feature_matches,
                "semantic_source": semantic_source,
                "stored_features": stored_features,
                "match_source_count": len(match_sources),
                "penalty": entry.get("penalty", 0.0),
                "fail_count": entry.get("fail_count", 0),
                "missing_count": entry.get("missing_count", 0),
                "length_ratio": entry.get("length_ratio"),
                "weight_ratio": entry.get("weight_ratio"),
                "width_ratio": entry.get("width_ratio"),
                "height_ratio": entry.get("height_ratio"),
            }
        )

    scored.sort(
        key=lambda item: (
            -item["final_score"],
            -item["feature_coverage"],
            item["fit_rank"],
            -item["pinecone_score"],
            item["fetch_pos"],
        )
    )

    analysis_candidates: list[dict[str, Any]] = []
    for final_rank, item in enumerate(scored, 1):
        listing = item["listing"]
        diagnostic = {
            "fetch_position": item["fetch_pos"],
            "candidate_id": item["candidate_id"],
            "fit_position": item["fit_rank"],
            "final_position": final_rank,
            "title": listing.get("title"),
            "url": listing.get("url"),
            "make": listing.get("make"),
            "pinecone_cosine_similarity": item["pinecone_score"],
            "requested_features": requested_features,
            "stored_features": item["stored_features"],
            "match_source_count": item["match_source_count"],
            "feature_matches": item["feature_matches"],
            "semantic_source": item["semantic_source"],
            "feature_coverage": item["feature_coverage"],
            "fit_order_score": item["fit_order_score"],
            "feature_weight": FEATURE_RERANK_WEIGHT,
            "fit_weight": FIT_RERANK_WEIGHT,
            "final_score": item["final_score"],
            "length": listing.get("length"),
            "width": listing.get("width"),
            "height": listing.get("height"),
            "payload_capacity": listing.get("payload_capacity"),
            "gvwr": listing.get("gvwr"),
            "length_ratio": item["length_ratio"],
            "weight_ratio": item["weight_ratio"],
            "width_ratio": item["width_ratio"],
            "height_ratio": item["height_ratio"],
            "fit_penalty": item["penalty"],
            "fit_fail_count": item["fail_count"],
            "fit_missing_count": item["missing_count"],
            "rank_movement": {
                "pinecone_to_fit": item["fetch_pos"] - item["fit_rank"],
                "fit_to_final": item["fit_rank"] - final_rank,
            },
        }
        analysis_candidates.append(diagnostic)
        if FEATURE_RERANK_VERBOSE_LOGS:
            logger.info(
                "feature_fit_candidate | %s",
                json.dumps(diagnostic, ensure_ascii=False, default=str),
            )

    matched_candidates = sum(
        1 for item in scored if item["feature_coverage"] > 0.0
    )
    logger.info(
        "feature_fit_summary | candidates=%s matched_candidates=%s feature_weight=%.2f fit_weight=%.2f threshold=%.2f metadata_filter=%s query_text=%r final_urls=%s",
        candidate_count,
        matched_candidates,
        FEATURE_RERANK_WEIGHT,
        FIT_RERANK_WEIGHT,
        FEATURE_MATCH_THRESHOLD,
        json.dumps(metadata_filter or {}, default=str),
        query_text,
        [item["listing"].get("url") for item in scored],
    )
    return [item["listing"] for item in scored], {
        "applied": True,
        "feature_match_mode": (
            "gpt_semantic"
            if semantic_rerank is not None and not semantic_rerank.fallback_candidate_ids
            else "hybrid_semantic_deterministic"
            if semantic_rerank is not None
            else "deterministic_fallback"
        ),
        "semantic_fallback_reason": semantic_fallback_reason,
        "semantic_model": semantic_rerank.model if semantic_rerank is not None else None,
        "semantic_reasoning_effort": semantic_rerank.reasoning_effort if semantic_rerank is not None else None,
        "semantic_latency_ms": semantic_rerank.latency_ms if semantic_rerank is not None else None,
        "semantic_request_id": semantic_rerank.request_id if semantic_rerank is not None else None,
        "semantic_request_ids": list(semantic_rerank.request_ids) if semantic_rerank is not None else [],
        "semantic_batch_count": semantic_rerank.batch_count if semantic_rerank is not None else 0,
        "semantic_fallback_candidate_ids": sorted(semantic_rerank.fallback_candidate_ids) if semantic_rerank is not None else [],
        "semantic_validation_errors": semantic_rerank.validation_errors_by_id if semantic_rerank is not None else {},
        "candidate_count": candidate_count,
        "matched_candidate_count": matched_candidates,
        "requested_features": requested_features,
        "feature_weight": FEATURE_RERANK_WEIGHT,
        "fit_weight": FIT_RERANK_WEIGHT,
        "feature_match_threshold": FEATURE_MATCH_THRESHOLD,
        "hard_metadata_filter": metadata_filter or {},
        "embedding_query_text": query_text,
        "candidates": analysis_candidates,
    }


def search_pinecone_listing_result(
    *,
    category: str | None,
    slots: dict[str, Any],
    metadata_filters: dict[str, Any] | None = None,
    requested_features: list[str] | None = None,
    already_shown_urls: list[str] | None = None,
    top_k: int | None = None,
    max_recommendations: int | None = None,
    category_only_filters: bool = False,
) -> PineconeListingSearchResult:
    metadata_filters = metadata_filters or {}
    requested_features = [
        str(feature).strip()
        for feature in (requested_features or [])
        if str(feature or "").strip()
    ]
    # "trailer" only when we know literally nothing — the embeddings API rejects an empty input.
    query = _query_text(category, slots, metadata_filters, requested_features) or "trailer"
    top_k = top_k or int(os.getenv("SEARCH_TOP_K", "50"))
    max_recommendations = max_recommendations or int(os.getenv("SEARCH_MAX_RECOMMENDATIONS", "5"))
    metadata_filter = _metadata_filter(category, slots, metadata_filters, category_only_filters) or None
    query_preview = query[:2000] + ("...(truncated)" if len(query) > 2000 else "")
    shown_urls = {str(u).strip() for u in (already_shown_urls or []) if str(u or "").strip()}

    logger.info(
        "pinecone_embedding_query | category=%r | metadata_filter=%s | query_text=%r",
        category,
        json.dumps(metadata_filter, default=str) if metadata_filter else "{}",
        query,
    )
    vector = _embed(query)

    logger.info(
        "pinecone_search | category=%r | top_k=%s | max_recommendations=%s | "
        "already_shown_url_count=%s | metadata_filter=%s | slots=%s | metadata_filters_collected=%s | query_text=%r",
        category,
        top_k,
        max_recommendations,
        len(shown_urls),
        json.dumps(metadata_filter, default=str) if metadata_filter else "{}",
        json.dumps(slots or {}, default=str),
        json.dumps(metadata_filters or {}, default=str),
        query_preview,
    )

    response = _pinecone_index().query(
        vector=vector,
        top_k=top_k,
        include_metadata=True,
        filter=metadata_filter,
    )
    shown = shown_urls
    listings: list[dict[str, Any]] = []
    for match in getattr(response, "matches", None) or response.get("matches", []):
        item = _clean_match(match)
        if item["url"] and item["url"] in shown:
            continue
        listings.append(item)

    rerank_debug: dict[str, Any] = {"applied": False, "reason": "disabled"}
    make_debug: dict[str, Any]
    match_analysis: dict[str, Any] = {}
    if requested_features:
        required_length_ft = _required_length_ft_from_filters(slots, metadata_filters)
        required_payload_lbs = _required_payload_lbs_from_filters(slots, metadata_filters)
        required_width_ft = _required_width_ft_from_filters(slots, metadata_filters)
        required_height_ft = _required_height_ft_from_filters(slots, metadata_filters)
        semantic_rerank: ValidatedFeatureRerank | None = None
        semantic_fallback_reason: str | None = None
        if config.settings.feature_llm_rerank_enabled and listings:
            try:
                semantic_rerank = rank_non_metadata_features(listings, requested_features)
            except Exception as exc:  # noqa: BLE001 - deterministic availability fallback
                semantic_fallback_reason = f"{type(exc).__name__}: {exc}"
                logger.exception(
                    "feature_rank_fallback | reason=%r candidates=%s requested_features=%s",
                    semantic_fallback_reason,
                    len(listings),
                    json.dumps(requested_features, ensure_ascii=False),
                )
        else:
            semantic_fallback_reason = (
                "no_candidates" if not listings else "feature_llm_rerank_disabled"
            )
            logger.info(
                "feature_rank_not_invoked | reason=%s candidates=%s requested_features=%s",
                semantic_fallback_reason,
                len(listings),
                json.dumps(requested_features, ensure_ascii=False),
            )

        listings, match_analysis = _combined_feature_fit_rerank(
            listings,
            requested_features=requested_features,
            required_length_ft=required_length_ft,
            required_payload_lbs=required_payload_lbs,
            required_width_ft=required_width_ft,
            required_height_ft=required_height_ft,
            metadata_filter=metadata_filter,
            query_text=query,
            semantic_rerank=semantic_rerank,
            semantic_fallback_reason=semantic_fallback_reason,
        )
        rerank_debug = {
            "applied": True,
            "mode": "combined_feature_fit",
            "candidate_count": len(listings),
        }
        make_debug = {
            "applied": False,
            "reason": "feature_order_preserved",
            "category": normalize_category(category) if category else None,
        }
    elif RERANK_ENABLED:
        required_length_ft = _required_length_ft_from_filters(slots, metadata_filters)
        required_payload_lbs = _required_payload_lbs_from_filters(slots, metadata_filters)
        required_width_ft = _required_width_ft_from_filters(slots, metadata_filters)
        required_height_ft = _required_height_ft_from_filters(slots, metadata_filters)
        listings, rerank_debug = _rerank_listings_by_fit(
            listings,
            required_length_ft=required_length_ft,
            required_payload_lbs=required_payload_lbs,
            required_width_ft=required_width_ft,
            required_height_ft=required_height_ft,
            warn_ratio=RERANK_WARN_RATIO,
            extreme_ratio=RERANK_EXTREME_RATIO,
            length_weight=RERANK_LENGTH_WEIGHT,
            missing_dim_penalty=RERANK_MISSING_DIM_PENALTY,
            # This is the category-only relaxed retry: the hard metadata gates were dropped at
            # the Pinecone level precisely because nothing met them. If the fit rerank then
            # hard-culls the same under-size candidates (retain_all=False drops fail_count>0
            # whenever any non-failing row exists), the relaxation is undone one stage later and
            # the screen collapses to the lone row that happened to meet the gate — seen live as
            # a 50 ft livestock search returning 1 of 17 candidates. Keep every candidate and let
            # the fit score ORDER them closest-first, which is what "closest alternatives" means.
            retain_all=category_only_filters,
        )
        logger.info("rerank_debug=%s", json.dumps(rerank_debug, default=str))
        listings, make_debug = _apply_category_make_preference(
            listings,
            category=category,
            max_recommendations=max_recommendations,
        )
    else:
        listings, make_debug = _apply_category_make_preference(
            listings,
            category=category,
            max_recommendations=max_recommendations,
        )
    logger.info("make_rerank_debug=%s", json.dumps(make_debug, default=str))

    return PineconeListingSearchResult(
        listings=listings[:max_recommendations],
        query_text=query,
        metadata_filter=metadata_filter,
        rerank_debug=rerank_debug,
        make_debug=make_debug,
        match_analysis=match_analysis,
    )


def search_pinecone_listings(
    *,
    category: str | None,
    slots: dict[str, Any],
    metadata_filters: dict[str, Any] | None = None,
    requested_features: list[str] | None = None,
    already_shown_urls: list[str] | None = None,
    top_k: int | None = None,
    max_recommendations: int | None = None,
    category_only_filters: bool = False,
) -> list[dict[str, Any]]:
    result = search_pinecone_listing_result(
        category=category,
        slots=slots,
        metadata_filters=metadata_filters,
        requested_features=requested_features,
        already_shown_urls=already_shown_urls,
        top_k=top_k,
        max_recommendations=max_recommendations,
        category_only_filters=category_only_filters,
    )
    return [
        {
            key: value
            for key, value in item.items()
            if key not in {"match_evidence_text", "features"}
        }
        for item in result.listings
    ]
