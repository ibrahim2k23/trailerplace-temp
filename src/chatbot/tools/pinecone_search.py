from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

from src.normalizer import normalize_category, normalize_hitch, normalize_subcategory
from src.chatbot.make_inventory import make_filter_values

load_dotenv()

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

MAKE_ALIAS_MAP: dict[str, str] = {
    "cargo craft": "Cargo Craft",
    "cargo craft trailers": "Cargo Craft",
    "iron bull": "Iron Bull Trailers",
    "iron bull trailers": "Iron Bull Trailers",
    "diamond c": "Diamond C",
    "diamond c trailers": "Diamond C",
    "east texas": "East Texas Trailers",
    "east texas trailers": "East Texas Trailers",
    "calico": "Calico Trailers",
    "calico trailers": "Calico Trailers",
    "haulmark": "Haulmark",
    "stallion": "Stallion",
    "p&c": "P&C",
    "p and c": "P&C",
    "p n c": "P&C",
    "p c": "P&C",
    "aluma": "Aluma",
    "galyean": "Galyean",
    "gooseneck": "Gooseneck",
    "texas pride": "Texas Pride",
}

_ALLOWED_HITCH_TYPES = {"Gooseneck", "Bumper Pull"}


def _parse_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if value > 0 else None
    text = str(value).lower().replace(",", "").strip()
    if not text:
        return None
    # Common small typos and variants for unit words.
    text = re.sub(r"\blb\b", "lbs", text)
    text = re.sub(r"\blbd\b", "lbs", text)
    text = re.sub(r"\blbss\b", "lbs", text)
    text = re.sub(r"\bpunds\b", "pounds", text)
    text = re.sub(r"\bkgs\b", "kg", text)
    text = re.sub(r"\bkilograms?\b", "kg", text)
    text = re.sub(r"\btonnes?\b", "ton", text)

    # Normalize unit-bearing values to pounds.
    m = re.search(r"(\d+(?:\.\d+)?)\s*(lbs?|pounds?|#)\b", text)
    if m:
        number = float(m.group(1))
        return number if number > 0 else None
    m = re.search(r"(\d+(?:\.\d+)?)\s*(kg)\b", text)
    if m:
        number = float(m.group(1)) * 2.2046226218
        return number if number > 0 else None
    m = re.search(r"(\d+(?:\.\d+)?)\s*(ton)\b", text)
    if m:
        number = float(m.group(1)) * 2000.0
        return number if number > 0 else None

    match = re.search(r"(\d+(?:\.\d+)?)\s*(k|m)?", text)
    if not match:
        return None
    number = float(match.group(1))
    suffix = match.group(2)
    if suffix == "k":
        number *= 1000
    elif suffix == "m":
        number *= 1_000_000
    return number if number > 0 else None


def _parse_length_ft(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if value > 0 else None
    text = str(value).lower().strip()
    if not text:
        return None
    ft = re.search(r"(\d+(?:\.\d+)?)\s*(?:ft|feet|foot|')", text)
    inches = re.search(r"(\d+(?:\.\d+)?)\s*(?:in|inch|inches|\")", text)
    if ft:
        return float(ft.group(1)) + (float(inches.group(1)) / 12 if inches else 0)
    if inches:
        return float(inches.group(1)) / 12
    return _parse_number(text)


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
    return response.data[0].embedding


def _metadata_filter(
    category: str | None,
    slots: dict[str, Any],
    metadata_filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata_filters = metadata_filters or {}
    filters: list[dict[str, Any]] = []
    normalized_category = normalize_category(category) if category else None
    if category:
        filters.append({"category": {"$eq": normalized_category}})

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


def _query_text(
    category: str | None,
    slots: dict[str, Any],
    metadata_filters: dict[str, Any],
    user_message: str,
) -> str:
    parts = [user_message.strip()]
    if category:
        parts.append(f"Category: {category}")
    for key, value in sorted((slots or {}).items()):
        if value not in (None, "", [], {}):
            parts.append(f"{key}: {value}")
    for key, value in sorted((metadata_filters or {}).items()):
        if value not in (None, "", [], {}):
            parts.append(f"filter_{key}: {value}")
    return " | ".join(p for p in parts if p)


def _clean_match(match: Any) -> dict[str, Any]:
    metadata = dict(getattr(match, "metadata", None) or match.get("metadata", {}) or {})
    score = getattr(match, "score", None)
    if score is None and isinstance(match, dict):
        score = match.get("score")
    price = metadata.get("price_display") or metadata.get("price")
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
        "axles": metadata.get("axles"),
        "gvwr": metadata.get("gvwr"),
        "payload_capacity": metadata.get("payload_capacity"),
        "material": metadata.get("trailer_material"),
        "floor": metadata.get("floor"),
        "url": metadata.get("url") or "",
        "relevance_score": score,
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
    warn_ratio: float,
    extreme_ratio: float,
    length_weight: float,
    missing_dim_penalty: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    needs_present = any(x is not None for x in (required_length_ft, required_payload_lbs, required_width_ft))
    if not listings or not needs_present:
        return listings, {"applied": False, "reason": "missing_clear_requirements_or_no_listings"}

    entries: list[dict[str, Any]] = []
    for idx, listing in enumerate(listings, 1):
        length_ft = _parse_length_ft(listing.get("length"))
        gvwr_lbs = _parse_number(listing.get("gvwr"))
        payload_lbs = _parse_number(listing.get("payload_capacity"))
        width_ft = _parse_length_ft(listing.get("width"))

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
                "length_overage": round(length_overage, 6),
                "fit_score": round(base_score - penalty, 6),
            }
        )

    # Strict policy: prefer non-failing entries first.
    nonfailing = [e for e in entries if e["fail_count"] == 0]
    fallback_pool = entries if not nonfailing else nonfailing
    ranked_entries = sorted(
        fallback_pool,
        key=lambda e: (
            e["missing_count"],
            e["length_overage"],
            e["penalty"],
            -e["base_score"],
        ),
    )

    if RERANK_VERBOSE_LOGS:
        decision_rank = {id(e): i for i, e in enumerate(ranked_entries, 1)}
        for e in entries:
            logger.info(
                "rerank_score | fetched_pos=%s | decision_rank=%s | title=%r | base_score=%.6f | penalty=%.6f | fit_score=%.6f | length_ratio=%s | weight_ratio=%s | width_ratio=%s | fail_count=%s | missing_count=%s",
                e["fetch_pos"],
                decision_rank.get(id(e)),
                e["listing"].get("title"),
                e["base_score"],
                e["penalty"],
                e["fit_score"],
                e["length_ratio"],
                e["weight_ratio"],
                e["width_ratio"],
                e["fail_count"],
                e["missing_count"],
            )

    logger.info(
        "rerank_summary | applied=true | required_length_ft=%s | required_payload_lbs=%s | required_width_ft=%s | candidates=%s | kept_pool=%s",
        required_length_ft,
        required_payload_lbs,
        required_width_ft,
        len(entries),
        len(fallback_pool),
    )

    return [e["listing"] for e in ranked_entries], {
        "applied": True,
        "required_length_ft": required_length_ft,
        "required_payload_lbs": required_payload_lbs,
        "required_width_ft": required_width_ft,
        "candidate_count": len(entries),
    }


def search_pinecone_listing_result(
    *,
    category: str | None,
    slots: dict[str, Any],
    metadata_filters: dict[str, Any] | None = None,
    user_message: str,
    already_shown_urls: list[str] | None = None,
    top_k: int | None = None,
    max_recommendations: int | None = None,
) -> PineconeListingSearchResult:
    metadata_filters = metadata_filters or {}
    query = _query_text(category, slots, metadata_filters, user_message)
    top_k = top_k or int(os.getenv("SEARCH_TOP_K", "50"))
    max_recommendations = max_recommendations or int(os.getenv("SEARCH_MAX_RECOMMENDATIONS", "5"))
    metadata_filter = _metadata_filter(category, slots, metadata_filters) or None
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
    if RERANK_ENABLED:
        required_length_ft = _required_length_ft_from_filters(slots, metadata_filters)
        required_payload_lbs = _required_payload_lbs_from_filters(slots, metadata_filters)
        required_width_ft = _required_width_ft_from_filters(slots, metadata_filters)
        listings, rerank_debug = _rerank_listings_by_fit(
            listings,
            required_length_ft=required_length_ft,
            required_payload_lbs=required_payload_lbs,
            required_width_ft=required_width_ft,
            warn_ratio=RERANK_WARN_RATIO,
            extreme_ratio=RERANK_EXTREME_RATIO,
            length_weight=RERANK_LENGTH_WEIGHT,
            missing_dim_penalty=RERANK_MISSING_DIM_PENALTY,
        )
        logger.info("rerank_debug=%s", json.dumps(rerank_debug, default=str))
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
        match_analysis={},
    )


def search_pinecone_listings(
    *,
    category: str | None,
    slots: dict[str, Any],
    metadata_filters: dict[str, Any] | None = None,
    user_message: str,
    already_shown_urls: list[str] | None = None,
    top_k: int | None = None,
    max_recommendations: int | None = None,
) -> list[dict[str, Any]]:
    result = search_pinecone_listing_result(
        category=category,
        slots=slots,
        metadata_filters=metadata_filters,
        user_message=user_message,
        already_shown_urls=already_shown_urls,
        top_k=top_k,
        max_recommendations=max_recommendations,
    )
    return [
        {key: value for key, value in item.items() if key != "match_evidence_text"}
        for item in result.listings
    ]
