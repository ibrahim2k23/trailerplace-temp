from __future__ import annotations

import json
import logging
import os
import re
from functools import lru_cache
from typing import Any, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

from src.normalizer import normalize_category, normalize_color, normalize_hitch

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


def _metadata_filter(category: str | None, slots: dict[str, Any]) -> dict[str, Any]:
    filters: list[dict[str, Any]] = []
    if category:
        filters.append({"category": {"$eq": normalize_category(category)}})

    if slots.get("hitch_type"):
        hitch = normalize_hitch(str(slots["hitch_type"]))
        if hitch:
            filters.append({"hitch_type": {"$eq": hitch}})
    if slots.get("color"):
        filters.append({"color": {"$eq": normalize_color(str(slots["color"]))}})

    max_price = _parse_number(slots.get("max_price") or slots.get("budget"))
    if max_price:
        filters.append({"price": {"$lte": max_price}})

    min_length = (
        _parse_length_ft(slots.get("haul_length_ft"))
        or _parse_length_ft(slots.get("vehicle_length_ft"))
        or _parse_length_ft(slots.get("trailer_length_ft"))
        or _parse_length_ft(slots.get("trailer_size"))
    )
    if min_length:
        filters.append({"length_ft_num": {"$gte": min_length}})

    min_gvwr = _parse_number(
        slots.get("haul_weight_lbs")
        or slots.get("payload_need")
        or slots.get("total_weight")
    )
    if min_gvwr:
        filters.append({"gvwr_lbs_num": {"$gte": min_gvwr}})

    if not filters:
        return {}
    if len(filters) == 1:
        return filters[0]
    return {"$and": filters}


def _query_text(category: str | None, slots: dict[str, Any], user_message: str) -> str:
    parts = [user_message.strip()]
    if category:
        parts.append(f"Category: {category}")
    for key, value in sorted((slots or {}).items()):
        if value not in (None, "", [], {}):
            parts.append(f"{key}: {value}")
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
    }


def _required_length_ft_from_slots(slots: dict[str, Any]) -> Optional[float]:
    return (
        _parse_length_ft(slots.get("haul_length_ft"))
        or _parse_length_ft(slots.get("vehicle_length_ft"))
        or _parse_length_ft(slots.get("trailer_length_ft"))
        or _parse_length_ft(slots.get("trailer_size"))
    )


def _required_weight_lbs_from_slots(slots: dict[str, Any]) -> Optional[float]:
    return _parse_number(
        slots.get("haul_weight_lbs")
        or slots.get("payload_need")
        or slots.get("total_weight")
    )


def _required_width_ft_from_slots(slots: dict[str, Any]) -> Optional[float]:
    return _parse_length_ft(
        slots.get("item_or_trailer_width_ft")
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


def _apply_category_make_preference(
    listings: list[dict[str, Any]],
    *,
    category: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not listings or not category:
        return listings, {"applied": False, "reason": "missing_listings_or_category"}

    cat = normalize_category(category)
    preferred = CATEGORY_MAKE_PREFERENCES.get(cat)
    if not preferred:
        return listings, {"applied": False, "reason": "no_category_preference", "category": cat}

    preferred_norm = [_canonical_make(x) for x in preferred]
    pref_index = {mk: i for i, mk in enumerate(preferred_norm)}
    buckets: list[list[dict[str, Any]]] = [[] for _ in preferred_norm]
    remainder: list[dict[str, Any]] = []

    for item in listings:
        item_make = _canonical_make(item.get("make"))
        idx = pref_index.get(item_make)
        if idx is None:
            remainder.append(item)
        else:
            buckets[idx].append(item)

    ranked = [it for bucket in buckets for it in bucket] + remainder

    preferred_count = sum(len(bucket) for bucket in buckets)
    before_top = [str(x.get("make") or "") for x in listings[:8]]
    after_top = [str(x.get("make") or "") for x in ranked[:8]]
    logger.info(
        "make_rerank_summary | category=%r | preferred=%s | preferred_found=%s | total=%s | before_top=%s | after_top=%s",
        cat,
        preferred_norm,
        preferred_count,
        len(listings),
        before_top,
        after_top,
    )
    if MAKE_RERANK_VERBOSE_LOGS:
        for i, item in enumerate(ranked, 1):
            mk = _canonical_make(item.get("make"))
            logger.info(
                "make_rerank_item | rank=%s | title=%r | make=%r | canonical_make=%r | preferred_index=%s",
                i,
                item.get("title"),
                item.get("make"),
                mk,
                pref_index.get(mk),
            )

    return ranked, {
        "applied": True,
        "category": cat,
        "preferred_order": preferred_norm,
        "preferred_found": preferred_count,
    }


def _rerank_listings_by_fit(
    listings: list[dict[str, Any]],
    *,
    required_length_ft: Optional[float],
    required_gvwr_lbs: Optional[float],
    required_width_ft: Optional[float],
    warn_ratio: float,
    extreme_ratio: float,
    length_weight: float,
    missing_dim_penalty: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    needs_present = any(x is not None for x in (required_length_ft, required_gvwr_lbs, required_width_ft))
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
            (weight_cap / required_gvwr_lbs)
            if required_gvwr_lbs is not None and weight_cap is not None and required_gvwr_lbs > 0
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

        if required_gvwr_lbs is not None:
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
        "rerank_summary | applied=true | required_length_ft=%s | required_gvwr_lbs=%s | required_width_ft=%s | candidates=%s | kept_pool=%s",
        required_length_ft,
        required_gvwr_lbs,
        required_width_ft,
        len(entries),
        len(fallback_pool),
    )

    return [e["listing"] for e in ranked_entries], {
        "applied": True,
        "required_length_ft": required_length_ft,
        "required_gvwr_lbs": required_gvwr_lbs,
        "required_width_ft": required_width_ft,
        "candidate_count": len(entries),
    }


def search_pinecone_listings(
    *,
    category: str | None,
    slots: dict[str, Any],
    user_message: str,
    already_shown_urls: list[str] | None = None,
    top_k: int | None = None,
    max_recommendations: int | None = None,
) -> list[dict[str, Any]]:
    query = _query_text(category, slots, user_message)
    vector = _embed(query)
    top_k = top_k or int(os.getenv("SEARCH_TOP_K", "50"))
    max_recommendations = max_recommendations or int(os.getenv("SEARCH_MAX_RECOMMENDATIONS", "5"))
    metadata_filter = _metadata_filter(category, slots) or None
    query_preview = query[:2000] + ("...(truncated)" if len(query) > 2000 else "")
    shown_urls = {str(u).strip() for u in (already_shown_urls or []) if str(u or "").strip()}

    logger.info(
        "pinecone_search | category=%r | top_k=%s | max_recommendations=%s | "
        "already_shown_url_count=%s | metadata_filter=%s | slots=%s | query_text=%r",
        category,
        top_k,
        max_recommendations,
        len(shown_urls),
        json.dumps(metadata_filter, default=str) if metadata_filter else "{}",
        json.dumps(slots or {}, default=str),
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

    if RERANK_ENABLED:
        required_length_ft = _required_length_ft_from_slots(slots)
        required_gvwr_lbs = _required_weight_lbs_from_slots(slots)
        required_width_ft = _required_width_ft_from_slots(slots)
        listings, rerank_debug = _rerank_listings_by_fit(
            listings,
            required_length_ft=required_length_ft,
            required_gvwr_lbs=required_gvwr_lbs,
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
    )
    logger.info("make_rerank_debug=%s", json.dumps(make_debug, default=str))

    return listings[:max_recommendations]
