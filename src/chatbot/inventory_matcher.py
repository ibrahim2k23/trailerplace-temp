from __future__ import annotations

import json
import logging
import math
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Optional

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from src.chatbot.make_resolver import resolve_make_from_text
from src.normalizer import clean_dealer_notes

load_dotenv()

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
_LISTINGS_FILE = _ROOT / "listings_final_v5.xlsx"
_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

_MAKE_ONLY = "MAKE_SEARCH"
_YEAR_MAKE = "YEAR_MAKE_SEARCH"
_MODEL_SEARCH = "MODEL_SEARCH"
_YEAR_MAKE_MODEL = "YEAR_MAKE_MODEL_SEARCH"
_STOCK = "STOCK_SEARCH"
_POSSIBLE_MODEL = "POSSIBLE_MODEL_SEARCH"
_UNKNOWN = "UNKNOWN_SEARCH"

_COMMON_MODEL_WORDS = {
    "a",
    "an",
    "and",
    "available",
    "cost",
    "do",
    "does",
    "for",
    "have",
    "how",
    "is",
    "looking",
    "much",
    "price",
    "that",
    "the",
    "this",
    "trailer",
    "trailers",
    "want",
    "what",
    "aluminum",
    "car",
    "cargo",
    "dump",
    "enclosed",
    "equipment",
    "fiber",
    "flatbed",
    "hauler",
    "livestock",
    "race",
    "roll",
    "tilt",
    "utility",
}

_ORDINAL_WORDS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
}
_INVENTORY_EVIDENCE_MAX_CHARS = 3000
_INTERNAL_LISTING_KEYS = {"match_evidence_text"}


def _strip_internal_listing_fields(listing: dict[str, Any] | None) -> dict[str, Any] | None:
    if not listing:
        return listing
    return {
        key: value
        for key, value in dict(listing).items()
        if key not in _INTERNAL_LISTING_KEYS
    }


def _public_listings(listings: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return [
        public
        for item in (listings or [])
        if (public := _strip_internal_listing_fields(item)) is not None
    ]


def _public_match_result(match_result: dict[str, Any]) -> dict[str, Any]:
    public = dict(match_result)
    public["best_match"] = _strip_internal_listing_fields(public.get("best_match"))
    public["top_matches"] = _public_listings(public.get("top_matches") or [])
    return public


class TrailerQueryExtraction(BaseModel):
    year: Optional[int] = None
    possible_make: Optional[str] = None
    possible_model_code: Optional[str] = None
    possible_model_text: Optional[str] = None
    stock_number: Optional[str] = None
    user_wants_price: bool = False
    user_wants_availability: bool = False
    user_wants_details: bool = False
    requested_features: list[str] = Field(default_factory=list)
    should_lookup: bool = False
    lookup_confidence: Literal["low", "medium", "high"] = "low"
    lookup_reason: str = ""
    search_intent: Literal[
        "make",
        "year_make",
        "model",
        "year_make_model",
        "stock",
        "unknown",
    ] = "unknown"


class InventoryFeatureFramingDecision(BaseModel):
    intro_text: str = ""
    configuration_match_level: Literal["full", "partial", "no_exact", "unknown"] = "unknown"
    missing_or_unconfirmed_features: list[str] = Field(default_factory=list)
    full_match_count: int = 0
    listing_match_labels: list[str] = Field(default_factory=list)
    reason: str = ""


class TrailerSearchRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class TrailerSearchResponse(BaseModel):
    reply: str
    entity_type: str
    confidence: float
    best_match: Optional[dict[str, Any]] = None
    top_matches: list[dict[str, Any]] = Field(default_factory=list)
    extraction: dict[str, Any] = Field(default_factory=dict)


def _model_dump(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def normalize_text(text: Any) -> str:
    value = "" if text is None else str(text)
    if value.lower() == "nan":
        return ""
    value = value.lower()
    value = (
        value.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\ufffd", " ")
    )
    value = value.replace("&", " and ")
    value = re.sub(r"\btrailers\b", "trailer", value)
    value = re.sub(r"(?<=\d)\s*[xX]\s*(?=\d)", " x ", value)
    value = re.sub(r"[^a-z0-9#]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _clean_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _stock_text(value: Any) -> str:
    text = _clean_scalar(value)
    if not text:
        return ""
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    return re.sub(r"\D+", "", text)


def extract_model_code(model: Any) -> str:
    text = normalize_text(model)
    for token in text.split():
        if token in _COMMON_MODEL_WORDS or token in {"base", "w", "with"}:
            continue
        if re.fullmatch(r"\d+", token):
            continue
        return token
    return ""


def load_inventory(excel_path: str | Path = _LISTINGS_FILE) -> pd.DataFrame:
    return pd.read_excel(excel_path)


def prepare_inventory(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    for column in (
        "title",
        "url",
        "year",
        "make",
        "model",
        "trim",
        "category",
        "subcategory",
        "stock_number",
        "price",
        "condition",
        "length",
        "width",
        "gvwr",
        "payload_capacity",
        "hitch_type",
        "dealer_notes",
        "info_specs_json",
        "info_spec_json",
    ):
        if column not in prepared.columns:
            prepared[column] = ""

    prepared["make_norm"] = prepared["make"].map(normalize_text)
    prepared["model_norm"] = prepared["model"].map(normalize_text)
    prepared["title_norm"] = prepared["title"].map(normalize_text)
    prepared["category_norm"] = prepared["category"].map(normalize_text)
    prepared["subcategory_norm"] = prepared["subcategory"].map(normalize_text)
    prepared["stock_number_norm"] = prepared["stock_number"].map(_stock_text)
    prepared["year_norm"] = prepared["year"].map(_stock_text)
    prepared["model_code"] = prepared["model"].map(extract_model_code)
    prepared["search_text"] = (
        prepared["year_norm"].astype(str)
        + " "
        + prepared["make_norm"].astype(str)
        + " "
        + prepared["model_norm"].astype(str)
        + " "
        + prepared["model_code"].astype(str)
        + " "
        + prepared["title_norm"].astype(str)
        + " "
        + prepared["stock_number_norm"].astype(str)
        + " "
        + prepared["category_norm"].astype(str)
        + " "
        + prepared["subcategory_norm"].astype(str)
    ).map(lambda value: re.sub(r"\s+", " ", value).strip())
    return prepared


@lru_cache(maxsize=1)
def prepared_inventory() -> pd.DataFrame:
    if not _LISTINGS_FILE.exists():
        logger.warning("inventory_matcher_file_missing | path=%s", _LISTINGS_FILE)
        return prepare_inventory(pd.DataFrame())
    return prepare_inventory(load_inventory(_LISTINGS_FILE))


@lru_cache(maxsize=1)
def _extractor_llm():
    return ChatOpenAI(model=_MODEL, temperature=0).with_structured_output(
        TrailerQueryExtraction,
        method="function_calling",
    )


def _inventory_reply_llm_enabled() -> bool:
    return (os.getenv("INVENTORY_REPLY_LLM_ENABLED") or "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


@lru_cache(maxsize=1)
def _inventory_reply_llm():
    return ChatOpenAI(model=_MODEL, temperature=0.3)


def _inventory_feature_framing_llm_enabled() -> bool:
    return (os.getenv("INVENTORY_FEATURE_FRAMING_LLM_ENABLED") or "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


@lru_cache(maxsize=1)
def _inventory_feature_framing_llm():
    model = (
        os.getenv("INVENTORY_FEATURE_FRAMING_MODEL")
        or os.getenv("OPENAI_MODEL")
        or _MODEL
    ).strip()
    return ChatOpenAI(model=model, temperature=0.25).with_structured_output(
        InventoryFeatureFramingDecision,
        method="function_calling",
    )


def _wants_price(text: str) -> bool:
    return bool(re.search(r"\b(price|cost|how much|payment|payments)\b|\$", text or "", re.I))


def _wants_availability(text: str) -> bool:
    return bool(re.search(r"\b(available|availability|in stock|do you have|have this|have that)\b", text or "", re.I))


def _wants_details(text: str) -> bool:
    return bool(re.search(r"\b(details?|specs?|stock(?: number)?|link|url|show me|tell me)\b", text or "", re.I))


def _fallback_requested_features(text: str) -> list[str]:
    source = text or ""
    low = source.lower()
    features: list[str] = []
    known_patterns = [
        (r"\bslid(?:e|ing)\s+gates?\b", "sliding gates"),
        (r"\bside\s+ramps?\b", "side ramps"),
        (r"\brear\s+doors?\b", "rear doors"),
        (r"\bmesh\s+sides?\b", "mesh sides"),
        (r"\btack\s+rooms?\b", "tack room"),
        (r"\bramp\s+gates?\b", "ramp gate"),
        (r"\bescape\s+doors?\b", "escape door"),
        (r"\bremovable\s+fenders?\b", "removable fenders"),
        (r"\bdovetails?\b", "dovetail"),
    ]
    for pattern, label in known_patterns:
        if re.search(pattern, low):
            features.append(label)

    with_match = re.search(
        r"\bwith\s+(.+?)(?:\b(?:price|cost|available|availability|in\s+stock|do\s+you\s+have|how\s+much)\b|[?.!,]|$)",
        low,
    )
    if with_match:
        phrase = re.sub(r"\b(?:a|an|the|and|or|please|trailer|trailers|model)\b", " ", with_match.group(1))
        phrase = re.sub(r"[^a-z0-9 -]+", " ", phrase)
        phrase = re.sub(r"\s+", " ", phrase).strip(" -")
        if (
            phrase
            and 1 <= len(phrase.split()) <= 6
            and re.search(r"\b(?:gate|gates|ramp|ramps|door|doors|mesh|tack|fender|fenders|dovetail)\b", phrase)
        ):
            features.append(phrase)

    deduped: list[str] = []
    for feature in features:
        feature = re.sub(r"\s+", " ", feature.strip().lower())
        if feature and feature not in deduped:
            deduped.append(feature)
    return deduped


def _extract_year(text: str) -> int | None:
    match = re.search(r"\b(19\d{2}|20\d{2})\b", text or "")
    if not match:
        return None
    return int(match.group(1))


def _extract_stock(text: str, df: pd.DataFrame | None = None) -> str | None:
    source = text or ""
    explicit = re.search(
        r"\b(?:stock|stk|id|item|unit|listing)\s*(?:number|num|no|#)?\s*#?\s*(\d{4,6})\b|#\s*(\d{4,6})\b",
        source,
        re.I,
    )
    if explicit:
        return explicit.group(1) or explicit.group(2)
    numbers = re.findall(r"\b\d{4,6}\b", source)
    if not numbers:
        return None
    years = {str(y) for y in re.findall(r"\b(?:19|20)\d{2}\b", source)}
    stocks = set((df if df is not None else prepared_inventory()).get("stock_number_norm", []))
    for number in numbers:
        if number not in years and number in stocks:
            return number
    return None


def _remove_phrase(text: str, phrase: str | None) -> str:
    if not phrase:
        return text
    out = text
    for token in normalize_text(phrase).split():
        out = re.sub(rf"(?<!\w){re.escape(token)}(?!\w)", " ", out, flags=re.I)
    return re.sub(r"\s+", " ", out).strip()


def _model_candidate_from_text(text: str, make: str | None, year: int | None, stock: str | None) -> str | None:
    normalized = normalize_text(text)
    if year:
        normalized = normalized.replace(str(year), " ")
    if stock:
        normalized = normalized.replace(str(stock), " ")
    normalized = _remove_phrase(normalized, make)
    make_tokens = normalize_text(make).split() if make else []
    tokens = []
    for token in normalized.split():
        if token in _COMMON_MODEL_WORDS or re.fullmatch(r"\d+", token) or token in {"stock", "stk"}:
            continue
        if any(fuzz.ratio(token, make_token) >= 78 for make_token in make_tokens):
            continue
        tokens.append(token)
    candidates = [
        token
        for token in tokens
        if re.search(r"\d", token) or 2 <= len(token) <= 8
    ]
    return candidates[0] if candidates else None


def _fallback_extraction(user_query: str, df: pd.DataFrame | None = None) -> TrailerQueryExtraction:
    df = df if df is not None else prepared_inventory()
    year = _extract_year(user_query)
    stock = _extract_stock(user_query, df)
    make_resolution = resolve_make_from_text(user_query, use_llm_fallback=False)
    make = make_resolution.make
    model_code = _model_candidate_from_text(user_query, make, year, stock)
    model_text = model_code

    if stock:
        intent = "stock"
    elif year and make and model_code:
        intent = "year_make_model"
    elif year and make:
        intent = "year_make"
    elif model_code:
        intent = "model"
    elif make:
        intent = "make"
    else:
        intent = "unknown"
    return TrailerQueryExtraction(
        year=year,
        possible_make=make,
        possible_model_code=model_code,
        possible_model_text=model_text,
        stock_number=stock,
        user_wants_price=_wants_price(user_query),
        user_wants_availability=_wants_availability(user_query),
        user_wants_details=_wants_details(user_query),
        requested_features=_fallback_requested_features(user_query),
        search_intent=intent,  # type: ignore[arg-type]
    )


def extract_trailer_query(user_query: str) -> TrailerQueryExtraction:
    if not os.getenv("OPENAI_API_KEY"):
        return _fallback_extraction(user_query)
    try:
        result = _extractor_llm().invoke(
            [
                SystemMessage(
                    content=(
                        "Extract trailer search entities from user messages. "
                        "Return structured fields only. Do not invent makes, models, stock numbers, or prices. "
                        "A make is a brand/manufacturer. A model code is often short or alphanumeric, "
                        "such as DTB, DMG, FMAX210, HXD207, DEC208. Users may put the model before the make. "
                        "If unsure, put the phrase in possible_model_text and leave exact fields null. "
                        "Put non-structured requested equipment features in requested_features, such as sliding gates, "
                        "side ramps, rear doors, mesh sides, tack rooms, ramp gates, escape doors, or removable fenders. "
                        "Only include requested_features explicitly stated by the user."
                        "\n\nAlso decide whether this message is a direct inventory lookup. Set should_lookup=true "
                        "only when the user explicitly wants availability, price, or details and supplies either "
                        "make + model or year + make. Make-only, model-only, stock-only, vague shopping, and ordinary "
                        "filter/category updates are not direct lookups. Use medium/high confidence only when both "
                        "the lookup intent and required identifiers are clear."
                    )
                ),
                HumanMessage(content=user_query),
            ]
        )
        data = _model_dump(result)
        fallback = _fallback_extraction(user_query)
        if not data.get("possible_make") and fallback.possible_make:
            data["possible_make"] = fallback.possible_make
        if not data.get("year") and fallback.year:
            data["year"] = fallback.year
        if not data.get("possible_model_code") and fallback.possible_model_code:
            data["possible_model_code"] = fallback.possible_model_code
        if not data.get("possible_model_text") and fallback.possible_model_text:
            data["possible_model_text"] = fallback.possible_model_text
        if not data.get("stock_number") and fallback.stock_number:
            data["stock_number"] = fallback.stock_number
        extracted_features = data.get("requested_features") or []
        data["requested_features"] = list(dict.fromkeys([*extracted_features, *fallback.requested_features]))
        for key in ("user_wants_price", "user_wants_availability", "user_wants_details"):
            data[key] = bool(data.get(key)) or bool(getattr(fallback, key))
        return TrailerQueryExtraction(**data)
    except Exception:
        logger.exception("inventory_query_extraction_failed")
        return _fallback_extraction(user_query)


def _spec_blob_text(value: Any) -> str:
    raw = _clean_scalar(value)
    if not raw:
        return ""
    try:
        parsed = json.loads(raw)
    except Exception:
        return raw[:1200]

    values: list[str] = []

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, val in obj.items():
                if val in (None, "", [], {}):
                    continue
                if isinstance(val, (dict, list)):
                    walk(val)
                else:
                    values.append(f"{key}: {val}")
        elif isinstance(obj, list):
            for item in obj:
                walk(item)
        elif obj not in (None, ""):
            values.append(str(obj))

    walk(parsed)
    return " | ".join(values)[:1200]


def _inventory_match_evidence_text(row: pd.Series | dict[str, Any]) -> str:
    get = row.get if isinstance(row, dict) else row.get
    parts = [
        _clean_scalar(get("title")),
        " ".join(
            part
            for part in (
                _clean_scalar(get("year")),
                _clean_scalar(get("make")),
                _clean_scalar(get("model")),
                _clean_scalar(get("trim")),
                _clean_scalar(get("category")),
                _clean_scalar(get("subcategory")),
            )
            if part
        ),
        _clean_scalar(get("search_text")),
        clean_dealer_notes(_clean_scalar(get("dealer_notes"))),
        _spec_blob_text(get("info_specs_json") or get("info_spec_json")),
    ]
    evidence = " | ".join(part for part in parts if part)
    return evidence[:_INVENTORY_EVIDENCE_MAX_CHARS]


def _row_to_listing(row: pd.Series | dict[str, Any], score: float | None = None) -> dict[str, Any]:
    get = row.get if isinstance(row, dict) else row.get
    price = _clean_scalar(get("price"))
    listing = {
        "title": _clean_scalar(get("title")),
        "url": _clean_scalar(get("url")),
        "year": _clean_scalar(get("year")),
        "make": _clean_scalar(get("make")),
        "model": _clean_scalar(get("model")),
        "trim": _clean_scalar(get("trim")),
        "category": _clean_scalar(get("category")),
        "subcategory": _clean_scalar(get("subcategory")),
        "stock_number": _stock_text(get("stock_number")),
        "condition": _clean_scalar(get("condition")) or "New",
        "price": price,
        "price_display": price,
        "length": _clean_scalar(get("length")),
        "width": _clean_scalar(get("width")),
        "gvwr": _clean_scalar(get("gvwr")),
        "payload_capacity": _clean_scalar(get("payload_capacity")),
        "hitch_type": _clean_scalar(get("hitch_type")),
        "match_evidence_text": _inventory_match_evidence_text(row),
    }
    if score is not None:
        listing["match_score"] = round(float(score), 4)
    return listing


def _score_candidate(
    row: pd.Series,
    *,
    query_norm: str,
    make_norm: str,
    model_code_norm: str,
    model_text_norm: str,
) -> dict[str, float]:
    make_score = fuzz.partial_ratio(make_norm, row["make_norm"]) if make_norm else 0.0
    model_code_score = (
        max(
            fuzz.ratio(model_code_norm, row["model_code"]),
            fuzz.partial_ratio(model_code_norm, row["model_norm"]),
            fuzz.partial_ratio(model_code_norm, row["title_norm"]),
        )
        if model_code_norm
        else 0.0
    )
    model_text_score = (
        fuzz.token_set_ratio(model_text_norm, row["model_norm"])
        if model_text_norm
        else 0.0
    )
    title_score = fuzz.token_set_ratio(query_norm, row["title_norm"]) if query_norm else 0.0
    search_score = fuzz.token_set_ratio(query_norm, row["search_text"]) if query_norm else 0.0
    overall = (
        0.30 * make_score
        + 0.35 * model_code_score
        + 0.15 * model_text_score
        + 0.10 * title_score
        + 0.10 * search_score
    )
    return {
        "make_score": float(make_score),
        "model_code_score": float(model_code_score),
        "model_text_score": float(model_text_score),
        "title_score": float(title_score),
        "search_score": float(search_score),
        "overall": float(overall),
    }


def _uses_year_metadata_filter(extraction: TrailerQueryExtraction) -> bool:
    return bool(extraction.year and (extraction.user_wants_price or extraction.user_wants_availability))


def _inventory_metadata_filter(extraction: TrailerQueryExtraction) -> dict[str, Any]:
    if _uses_year_metadata_filter(extraction):
        return {"year": str(extraction.year)}
    return {}


def _candidate_frame(df: pd.DataFrame, extraction: TrailerQueryExtraction) -> pd.DataFrame:
    candidates = df
    if _uses_year_metadata_filter(extraction):
        candidates = candidates[candidates["year_norm"] == str(extraction.year)]
    return candidates


def _best_make_from_query(user_query: str, extraction: TrailerQueryExtraction) -> str:
    return extraction.possible_make or resolve_make_from_text(
        user_query,
        use_llm_fallback=False,
    ).make or ""


def _lookup_intent(extraction: TrailerQueryExtraction) -> str:
    if extraction.user_wants_price:
        return "price"
    if extraction.user_wants_availability:
        return "availability"
    if extraction.user_wants_details:
        return "details"
    return "none"


def _requested_identifiers(extraction: TrailerQueryExtraction) -> dict[str, Any]:
    return {
        "stock_number": _stock_text(extraction.stock_number) or None,
        "year": extraction.year,
        "make": extraction.possible_make or None,
        "model_code": extraction.possible_model_code or None,
        "model_text": extraction.possible_model_text or None,
    }


def _has_specific_lookup_identifiers(extraction: TrailerQueryExtraction) -> bool:
    has_make = bool(extraction.possible_make)
    has_model = bool(extraction.possible_model_code or extraction.possible_model_text)
    return bool(
        (has_make and extraction.year)
        or (has_make and has_model)
    )


def _is_direct_inventory_lookup(extraction: TrailerQueryExtraction) -> bool:
    return _lookup_intent(extraction) != "none" and _has_specific_lookup_identifiers(extraction)


def _filter_same_make_rows(df: pd.DataFrame, make_norm: str) -> pd.DataFrame:
    if not make_norm or df.empty:
        return df.iloc[0:0]
    return df[df["make_norm"].map(lambda value: fuzz.ratio(make_norm, str(value or "")) >= 85)]


def _no_exact_alternative_rows(
    *,
    df: pd.DataFrame,
    extraction: TrailerQueryExtraction,
    query_norm: str,
    make_norm: str,
    top_scored: list[tuple[float, pd.Series, dict[str, float]]],
    limit: int,
) -> list[tuple[float, pd.Series, dict[str, float]]]:
    selected: list[tuple[float, pd.Series, dict[str, float]]] = []
    seen: set[str] = set()

    def add_rows(rows: list[tuple[float, pd.Series, dict[str, float]]]) -> None:
        for item in rows:
            row = item[1]
            key = _stock_text(row.get("stock_number")) or _clean_scalar(row.get("url")) or _clean_scalar(row.get("title"))
            if key in seen:
                continue
            seen.add(key)
            selected.append(item)
            if len(selected) >= limit:
                return

    same_make_df = _filter_same_make_rows(df, make_norm)
    if extraction.year and not same_make_df.empty:
        same_make_other_years = same_make_df[same_make_df["year_norm"] != str(extraction.year)]
        if not same_make_other_years.empty:
            rows: list[tuple[float, pd.Series, dict[str, float]]] = []
            for _, row in same_make_other_years.iterrows():
                scores = _score_candidate(
                    row,
                    query_norm=query_norm,
                    make_norm=make_norm,
                    model_code_norm=normalize_text(extraction.possible_model_code),
                    model_text_norm=normalize_text(extraction.possible_model_text or extraction.possible_model_code),
                )
                year_distance = abs(int(row.get("year_norm") or 0) - extraction.year) if str(row.get("year_norm") or "").isdigit() else 99
                rows.append((max(0.0, 100.0 - year_distance), row, scores))
            rows.sort(key=lambda item: item[0], reverse=True)
            add_rows(rows)

    if len(selected) < limit:
        safe_same_year = [
            item
            for item in top_scored
            if extraction.year
            and str(item[1].get("year_norm") or "") == str(extraction.year)
            and item[2]["overall"] >= 25
        ]
        add_rows(safe_same_year)

    return selected


def match_inventory(
    user_query: str,
    extraction: TrailerQueryExtraction,
    df: pd.DataFrame | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    df = df if df is not None else prepared_inventory()
    query_norm = normalize_text(user_query)
    stock = _stock_text(extraction.stock_number)
    lookup_intent = _lookup_intent(extraction)
    requested_identifiers = _requested_identifiers(extraction)
    direct_lookup = _is_direct_inventory_lookup(extraction)
    metadata_filter = _inventory_metadata_filter(extraction)
    logger.info(
        "inventory_direct_lookup_route | intent=%s | identifiers=%s | metadata_filter=%s | direct=%s | reason=%s",
        lookup_intent,
        requested_identifiers,
        metadata_filter,
        direct_lookup,
        "specific_price_availability_or_details_lookup"
        if direct_lookup
        else "missing_explicit_lookup_intent_or_specific_identifier",
    )
    if stock:
        stock_matches = df[df["stock_number_norm"] == stock]
        if not stock_matches.empty:
            best = _row_to_listing(stock_matches.iloc[0], 100.0)
            logger.info(
                "inventory_direct_lookup_result | exact_match_count=%s | alternative_count=%s | entity_type=%s | no_exact_reason=%s",
                1,
                0,
                _STOCK,
                "",
            )
            return {
                "entity_type": _STOCK,
                "confidence": 1.0,
                "best_match": best,
                "top_matches": [best],
                "should_handle_in_chat": True,
                "debug_scores": [],
                "exact_match_count": 1,
                "alternative_count": 0,
                "requested_identifiers": requested_identifiers,
                "metadata_filter": metadata_filter,
                "no_exact_reason": "",
            }

    possible_make = _best_make_from_query(user_query, extraction)
    make_norm = normalize_text(possible_make)
    model_code_norm = normalize_text(extraction.possible_model_code)
    model_text_norm = normalize_text(extraction.possible_model_text or extraction.possible_model_code)

    candidates = _candidate_frame(df, extraction)
    scored: list[tuple[float, pd.Series, dict[str, float]]] = []
    for _, row in candidates.iterrows():
        scores = _score_candidate(
            row,
            query_norm=query_norm,
            make_norm=make_norm,
            model_code_norm=model_code_norm,
            model_text_norm=model_text_norm,
        )
        scored.append((scores["overall"], row, scores))
    scored.sort(key=lambda item: item[0], reverse=True)
    top_scored = scored[: max(limit, 8)]

    strong_make_rows = [
        item
        for item in top_scored
        if make_norm and item[2]["make_score"] >= 85
    ]
    strong_model_rows = [
        item
        for item in top_scored
        if item[2]["model_code_score"] >= 85 or item[2]["model_text_score"] >= 88
    ]

    entity_type = _UNKNOWN
    should_handle = False
    selected = top_scored
    confidence = 0.0
    exact_match_count = 0
    no_exact_reason = ""

    if extraction.year and make_norm and strong_make_rows and not (model_code_norm or model_text_norm):
        entity_type = _YEAR_MAKE
        selected = strong_make_rows
        confidence = max(item[2]["make_score"] for item in strong_make_rows) / 100
        should_handle = True
        exact_match_count = len(strong_make_rows)
    elif model_code_norm or model_text_norm:
        model_make_rows = [
            item
            for item in top_scored
            if (not make_norm or item[2]["make_score"] >= 75)
            and (item[2]["model_code_score"] >= 75 or item[2]["model_text_score"] >= 78)
            and item[2]["overall"] >= 58
        ]
        if model_make_rows:
            selected = model_make_rows
            confidence = min(0.99, max(item[2]["overall"] for item in model_make_rows) / 100)
            if extraction.year and make_norm:
                entity_type = _YEAR_MAKE_MODEL
            elif model_make_rows[0][2]["model_code_score"] >= 85 or model_make_rows[0][2]["model_text_score"] >= 88:
                entity_type = _MODEL_SEARCH
            else:
                entity_type = _POSSIBLE_MODEL
            should_handle = True
            exact_match_count = len(model_make_rows)
    elif make_norm and strong_make_rows:
        entity_type = _MAKE_ONLY
        selected = strong_make_rows
        confidence = max(item[2]["make_score"] for item in strong_make_rows) / 100
        should_handle = False
    elif top_scored and top_scored[0][2]["overall"] >= 70:
        entity_type = _POSSIBLE_MODEL
        selected = top_scored
        confidence = top_scored[0][2]["overall"] / 100
        should_handle = True

    if direct_lookup and not exact_match_count:
        no_exact_reason = (
            "no_exact_year_filtered_match_fallback_discarded_year"
            if metadata_filter
            else "no_exact_inventory_match_for_requested_identifiers"
        )
        alternative_rows = _no_exact_alternative_rows(
            df=df,
            extraction=extraction,
            query_norm=query_norm,
            make_norm=make_norm,
            top_scored=top_scored,
            limit=limit,
        )
        selected = alternative_rows
        should_handle = True
        confidence = 0.0
        if entity_type == _MAKE_ONLY:
            entity_type = _YEAR_MAKE if extraction.year and make_norm else _POSSIBLE_MODEL

    top_matches = [
        _row_to_listing(row, score)
        for score, row, _scores in selected[:limit]
    ]
    unique_exact = len(top_matches) == 1 and entity_type in {_MODEL_SEARCH, _YEAR_MAKE_MODEL}
    best_match = top_matches[0] if top_matches and (entity_type == _STOCK or unique_exact) else None
    if top_matches and entity_type == _POSSIBLE_MODEL and confidence >= 0.9:
        best_match = top_matches[0]

    debug_scores = [
        {
            "title": _clean_scalar(row.get("title")),
            "stock_number": _stock_text(row.get("stock_number")),
            **scores,
        }
        for _score, row, scores in top_scored[:limit]
    ]
    logger.info(
        "inventory_match_result | entity_type=%s | confidence=%.3f | extracted=%s | top=%s",
        entity_type,
        confidence,
        _model_dump(extraction),
        debug_scores[:3],
    )
    logger.info(
        "inventory_direct_lookup_result | exact_match_count=%s | alternative_count=%s | entity_type=%s | no_exact_reason=%s",
        exact_match_count,
        len(top_matches) if no_exact_reason else max(0, len(top_matches) - exact_match_count),
        entity_type,
        no_exact_reason,
    )
    return {
        "entity_type": entity_type,
        "confidence": round(confidence, 4),
        "best_match": best_match,
        "top_matches": top_matches,
        "should_handle_in_chat": should_handle,
        "debug_scores": debug_scores,
        "exact_match_count": exact_match_count,
        "alternative_count": len(top_matches) if no_exact_reason else max(0, len(top_matches) - exact_match_count),
        "requested_identifiers": requested_identifiers,
        "metadata_filter": metadata_filter,
        "no_exact_reason": no_exact_reason,
    }


def _price_text(listing: dict[str, Any]) -> str:
    return _clean_scalar(listing.get("price_display") or listing.get("price"))


def _item_label(listing: dict[str, Any]) -> str:
    title = _clean_scalar(listing.get("title"))
    if title:
        return title
    parts = [
        _clean_scalar(listing.get("year")),
        _clean_scalar(listing.get("make")),
        _clean_scalar(listing.get("model")),
    ]
    return " ".join(part for part in parts if part).strip() or "that trailer"


def _format_match_line(index: int, listing: dict[str, Any]) -> str:
    bits = [f"{index}. {_item_label(listing)}"]
    stock = _clean_scalar(listing.get("stock_number"))
    if stock:
        bits.append(f"Stock number: {stock}")
    price = _price_text(listing)
    if price:
        bits.append(f"Price: {price}")
    if listing.get("length"):
        bits.append(f"Length: {listing.get('length')}")
    if listing.get("url"):
        bits.append(f"[View listing]({listing.get('url')})")
    return " - ".join(bits)


def _listing_bullets(listing: dict[str, Any]) -> list[str]:
    fields = [
        ("Stock Number", "stock_number"),
        ("Price", "price_display"),
        ("Category", "category"),
        ("Model", "model"),
        ("Length", "length"),
        ("Width", "width"),
        ("GVWR", "gvwr"),
        ("Payload Capacity", "payload_capacity"),
        ("Hitch Type", "hitch_type"),
        ("Condition", "condition"),
    ]
    lines: list[str] = []
    for label, key in fields:
        value = _clean_scalar(listing.get(key))
        if not value and key == "price_display":
            value = _price_text(listing)
        if value:
            lines.append(f"- {label}: {value}")
    return lines


def _format_listing_block(index: int, listing: dict[str, Any]) -> str:
    title = _item_label(listing)
    url = _clean_scalar(listing.get("url"))
    line1 = f"Trailer #{index}: [{title}]({url})" if url else f"Trailer #{index}: {title}"
    bullets = _listing_bullets(listing)
    if not bullets:
        bullets = ["- *(No spec fields on this listing.)*"]
    return "\n\n".join([line1, "\n".join(bullets)])


def _format_listing_blocks(listings: list[dict[str, Any]]) -> str:
    return "\n\n---\n\n".join(
        _format_listing_block(i, listing)
        for i, listing in enumerate(listings[:5], 1)
    )


def _fallback_inventory_intro(
    *,
    user_query: str,
    extraction: TrailerQueryExtraction,
    entity_type: str,
    listings: list[dict[str, Any]],
    exact: bool,
) -> str:
    if not listings:
        return "I could not confidently match that to a trailer in the inventory data."
    count = len(listings)
    first = listings[0]
    price = _price_text(first)
    if count == 1 and extraction.user_wants_price:
        if price:
            return f"Yes, we have that trailer in inventory, and the listed price is {price}."
        return "Yes, we have that trailer in inventory, if you want to know the price, contact our sales team at 979-532-1486."
    if count == 1 and extraction.user_wants_availability:
        return "Yes, we do have that trailer available in inventory."
    if count > 1 and extraction.user_wants_availability:
        return f"Yes, we have {count} matching trailers available in inventory."
    if entity_type == _YEAR_MAKE:
        if count == 1:
            return "We have one matching trailer in inventory for that year and make."
        return f"We have {count} matching trailers in inventory for that year and make."
    if count == 1:
        return "We have one matching trailer in inventory."
    return "We have multiple matching trailers in inventory, so here are the closest options."


def _requested_inventory_label(extraction: TrailerQueryExtraction) -> str:
    parts = [
        str(extraction.year) if extraction.year else "",
        extraction.possible_make or "",
        extraction.possible_model_text or extraction.possible_model_code or "",
    ]
    label = " ".join(part for part in parts if part).strip()
    return label or "that exact trailer"


def _no_exact_inventory_fallback(
    *,
    extraction: TrailerQueryExtraction,
    listings: list[dict[str, Any]],
) -> str:
    requested = _requested_inventory_label(extraction)
    if not listings:
        return f"We do not currently show {requested} in the inventory data."
    if extraction.user_wants_price:
        return (
            f"We do not currently show {requested} in the inventory data, "
            "so any listed price below applies only to the closest available alternative shown."
        )
    if extraction.user_wants_availability:
        return (
            f"We do not currently show {requested} in the inventory data, "
            "but these are the closest available alternatives I found."
        )
    return (
        f"We do not currently show {requested} in the inventory data, "
        "but these are the closest alternatives I found."
    )


def _feature_neutral_fallback(
    *,
    extraction: TrailerQueryExtraction,
    listings: list[dict[str, Any]],
) -> str:
    if not listings:
        return "I could not confidently match that exact configuration to a trailer in the inventory data."
    if extraction.user_wants_price:
        return (
            "I found the closest inventory match based on the confirmed year, make, or model details. "
            "Any listed price below applies to the shown inventory option."
        )
    if extraction.user_wants_availability:
        return (
            "I found the closest inventory match based on the confirmed year, make, or model details. "
            "The requested extra feature is not confirmed in the listing data shown."
        )
    return "I found the closest inventory options based on the confirmed listing details."


def _inventory_feature_framing_llm_intro(
    *,
    user_query: str,
    extraction: TrailerQueryExtraction,
    entity_type: str,
    confidence: float,
    listings: list[dict[str, Any]],
    exact: bool,
    fallback: str,
) -> tuple[str, dict[str, Any]]:
    requested_features = [str(x).strip() for x in (extraction.requested_features or []) if str(x).strip()]
    if not requested_features:
        return fallback, {}
    if not listings:
        return fallback, {}
    if (
        not _inventory_feature_framing_llm_enabled()
        or not _inventory_reply_llm_enabled()
        or not os.getenv("OPENAI_API_KEY")
    ):
        return fallback, {
            "configuration_match_level": "unknown",
            "full_match_count": 0,
            "source": "neutral_fallback",
        }

    facts = [
        {
            "position": idx,
            "title": _item_label(item),
            "stock_number": item.get("stock_number"),
            "price": _price_text(item),
            "year": item.get("year"),
            "make": item.get("make"),
            "model": item.get("model"),
            "trim": item.get("trim"),
            "category": item.get("category"),
            "subcategory": item.get("subcategory"),
            "length": item.get("length"),
            "width": item.get("width"),
            "hitch_type": item.get("hitch_type"),
            "payload_capacity": item.get("payload_capacity"),
            "url_present": bool(item.get("url")),
            "match_score": item.get("match_score"),
            "match_evidence_text": str(item.get("match_evidence_text") or "")[:1800],
        }
        for idx, item in enumerate(listings[:5], 1)
    ]
    try:
        decision = _inventory_feature_framing_llm().invoke(
            [
                SystemMessage(
                    content=(
                        "Write the short intro before inventory lookup results for trailer price or availability questions. "
                        "The user may request an exact year/make/model/stock plus extra non-structured features. "
                        "Use only supplied listing facts and hidden match evidence. Do not invent prices, availability, specs, "
                        "stock numbers, links, discounts, or feature matches. "
                        "If the year/make/model exists but requested features are missing or unconfirmed, explain professionally "
                        "that we do not currently show that exact configuration and that the shown listing(s) are closest available options. "
                        "If a listed price is supplied for an alternative, make clear it applies to the shown inventory option, not an unavailable exact configuration. "
                        "If hidden evidence confirms every requested feature for a listing, answer normally using supplied facts. "
                        "Use warm, polished sales-advisor language. No markdown, no bullets, no quotes. "
                        "intro_text must be one or two customer-facing sentences, max 55 words. "
                        "Do not ask for phone number, email, contact details, callback, or sales follow-up."
                    )
                ),
                HumanMessage(
                    content=(
                        f"User question: {user_query}\n"
                        f"Extraction: {_model_dump(extraction)}\n"
                        f"Requested features: {requested_features}\n"
                        f"Entity type: {entity_type}\n"
                        f"Confidence: {confidence}\n"
                        f"Exact single item before feature check: {exact}\n"
                        f"Inventory facts: {facts}"
                    )
                ),
            ]
        )
        data = _model_dump(decision)
        text = re.sub(r"\s+", " ", str(data.get("intro_text") or "").strip())
        if text.startswith('"') and text.endswith('"') and len(text) > 1:
            text = text[1:-1].strip()
        if not text:
            return fallback, {**data, "source": "neutral_fallback_empty_llm_text"}
        if re.search(
            r"\b(?:phone|email|contact|call|reach|callback|follow\s*up|sales\s+team)\b",
            text,
            re.I,
        ):
            return fallback, {**data, "source": "neutral_fallback_blocked_contact_text"}
        if len(text) > 420:
            text = text[:420].rsplit(" ", 1)[0].strip()
        if not text.endswith((".", "!", "?")):
            text += "."
        return text, {**data, "intro_text": text, "source": "llm"}
    except Exception:
        logger.exception("inventory_feature_framing_llm_failed")
        return fallback, {
            "configuration_match_level": "unknown",
            "full_match_count": 0,
            "source": "neutral_fallback_exception",
        }


def _inventory_intro_llm(
    *,
    user_query: str,
    extraction: TrailerQueryExtraction,
    entity_type: str,
    listings: list[dict[str, Any]],
    exact: bool,
    fallback: str,
    no_exact_reason: str = "",
    exact_match_count: int = 0,
    alternative_count: int = 0,
) -> str:
    if not _inventory_reply_llm_enabled() or not os.getenv("OPENAI_API_KEY"):
        return fallback
    match_status = "no_exact" if no_exact_reason else "exact" if exact_match_count or listings else "unknown"
    facts = [
        {
            "title": _item_label(item),
            "stock_number": item.get("stock_number"),
            "price": _price_text(item),
            "year": item.get("year"),
            "make": item.get("make"),
            "model": item.get("model"),
            "length": item.get("length"),
            "url_present": bool(item.get("url")),
        }
        for item in listings[:5]
    ]
    try:
        response = _inventory_reply_llm().invoke(
            [
                SystemMessage(
                    content=(
                        """Write one short, friendly, professional sentence for a trailer inventory lookup.

                        Use “we” language, such as “we have,” “we found,” etc.

                        Use only the supplied inventory facts. Do not make up price, availability, specs, stock numbers, links, discounts, or details.

                        Match status, exact_match_count, and no_exact_reason are authoritative. Do not reinterpret them.

                        If match_status is exact, never use no-match wording such as “we do not currently show,” “could not find,” or “not available.” Frame the supplied listings as matching inventory.
                        If match_status is no_exact, do not say the requested trailer is available or matching. Say we do not currently show that exact requested trailer/configuration, then frame supplied listings as closest available alternatives.

                        If the customer asks whether a trailer is available and match_status is exact, start with a clear friendly affirmation such as “Yes, we do have...” or “Yes, we currently have...”.

                        If one exact matching trailer is supplied, answer directly using the available facts.
                        If multiple exact matching trailers are supplied, say we have multiple matching options and briefly mention only the supplied differences.
                        If the customer asks about price or availability, answer only from the supplied facts.
                        If price or availability is missing, politely say it is not listed.
                        If match_status is unknown and no listings are supplied, say we do not currently show a matching trailer in the inventory.

                        Sound warm, helpful, polished, and sales-friendly, but stay strictly factual.

                        No markdown, no bullets, no quotes. Return only one customer-facing sentence."""
                    )
                ),
                HumanMessage(
                    content=(
                        f"User question: {user_query}\n"
                        f"Entity type: {entity_type}\n"
                        f"Match status: {match_status}\n"
                        f"Exact match count: {exact_match_count}\n"
                        f"Alternative count: {alternative_count}\n"
                        f"Exact single item: {exact}\n"
                        f"No exact reason: {no_exact_reason}\n"
                        f"Requested identifiers: {_requested_identifiers(extraction)}\n"
                        f"User wants price: {extraction.user_wants_price}\n"
                        f"User wants availability: {extraction.user_wants_availability}\n"
                        f"Inventory facts: {facts}"
                    )
                ),
            ]
        )
        text = re.sub(r"\s+", " ", str(response.content or "").strip())
        if not text:
            return fallback
        if not no_exact_reason and listings and re.search(
            r"\b(?:do\s+not|don't|cannot|can't|could\s+not|couldn't|not\s+currently)\b"
            r".{0,80}\b(?:show|have|find|match|matching|available)\b",
            text,
            re.I,
        ):
            return fallback
        if len(listings) == 1 and re.search(r"\bmultiple\b|\bseveral\b|\boptions\b", text, re.I):
            return fallback
        if len(listings) > 1 and re.search(r"\bonly\s+one\b|\bone\s+matching\b", text, re.I):
            return fallback
        if not text.endswith((".", "!", "?")):
            text += "."
        return text
    except Exception:
        logger.exception("inventory_reply_llm_failed")
        return fallback


def generate_inventory_response(
    user_query: str,
    extraction: TrailerQueryExtraction,
    match_result: dict[str, Any],
) -> str:
    entity_type = match_result.get("entity_type") or _UNKNOWN
    best = match_result.get("best_match")
    top = match_result.get("top_matches") or []
    wants_price = extraction.user_wants_price
    wants_availability = extraction.user_wants_availability
    has_requested_features = bool(extraction.requested_features)
    no_exact_reason = str(match_result.get("no_exact_reason") or "")

    if best:
        listings = [best]
        fallback = (
            _feature_neutral_fallback(extraction=extraction, listings=listings)
            if has_requested_features
            else _fallback_inventory_intro(
                user_query=user_query,
                extraction=extraction,
                entity_type=entity_type,
                listings=listings,
                exact=True,
            )
        )
        if has_requested_features:
            intro, feature_analysis = _inventory_feature_framing_llm_intro(
                user_query=user_query,
                extraction=extraction,
                entity_type=entity_type,
                confidence=float(match_result.get("confidence") or 0.0),
                listings=listings,
                exact=True,
                fallback=fallback,
            )
            match_result["feature_match_analysis"] = feature_analysis
        else:
            intro = _inventory_intro_llm(
                user_query=user_query,
                extraction=extraction,
                entity_type=entity_type,
                listings=listings,
                exact=True,
                fallback=fallback,
                exact_match_count=int(match_result.get("exact_match_count") or len(listings)),
                alternative_count=int(match_result.get("alternative_count") or 0),
            )
        return f"{intro}\n\n{_format_listing_blocks([best])}"

    if top:
        exact_single = len(top) == 1 and not no_exact_reason
        fallback = (
            _feature_neutral_fallback(extraction=extraction, listings=top)
            if has_requested_features
            else _no_exact_inventory_fallback(extraction=extraction, listings=top)
            if no_exact_reason
            else _fallback_inventory_intro(
                user_query=user_query,
                extraction=extraction,
                entity_type=entity_type,
                listings=top,
                exact=exact_single,
            )
        )
        if has_requested_features:
            intro, feature_analysis = _inventory_feature_framing_llm_intro(
                user_query=user_query,
                extraction=extraction,
                entity_type=entity_type,
                confidence=float(match_result.get("confidence") or 0.0),
                listings=top,
                exact=exact_single and not no_exact_reason,
                fallback=fallback,
            )
            match_result["feature_match_analysis"] = feature_analysis
        else:
            intro = _inventory_intro_llm(
                user_query=user_query,
                extraction=extraction,
                entity_type=entity_type,
                listings=top,
                exact=exact_single,
                fallback=fallback,
                no_exact_reason=no_exact_reason,
                exact_match_count=int(match_result.get("exact_match_count") or 0),
                alternative_count=int(match_result.get("alternative_count") or 0),
            )
        return f"{intro}\n\n{_format_listing_blocks(top)}"

    if no_exact_reason:
        return _no_exact_inventory_fallback(extraction=extraction, listings=[])

    return "I could not confidently match that to a trailer in the inventory data."


def _ordinal_reference(message: str) -> int | None:
    text = (message or "").lower()
    numeric = re.search(r"(?:#|\b)(\d+)(?:st|nd|rd|th)?\s+(?:one|trailer|listing|option)\b", text)
    if numeric:
        return int(numeric.group(1))
    for word, idx in _ORDINAL_WORDS.items():
        if re.search(rf"\b{word}\s+(?:one|trailer|listing|option)\b", text):
            return idx
    return None


def _has_context_reference(message: str) -> bool:
    return bool(
        _ordinal_reference(message)
        or re.search(r"\b(this|that|it|that one|this one)\b", message or "", re.I)
    )


def answer_from_last_listings(
    user_query: str,
    last_listings: list[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    listings = last_listings or []
    if not listings or not _has_context_reference(user_query):
        return None
    if not (_wants_price(user_query) or _wants_availability(user_query) or _wants_details(user_query)):
        return None

    ordinal = _ordinal_reference(user_query)
    if ordinal is None:
        if len(listings) != 1:
            extraction = TrailerQueryExtraction(
                user_wants_price=_wants_price(user_query),
                user_wants_availability=_wants_availability(user_query),
                user_wants_details=_wants_details(user_query),
                search_intent="unknown",
            )
            top = [dict(item) for item in listings[:5]]
            public_top = _public_listings(top)
            return {
                "reply": (
                    "Which trailer do you mean? You can say the number, like the first one or #2.\n"
                    + "\n".join(_format_match_line(i, item) for i, item in enumerate(public_top, 1))
                ),
                "entity_type": _POSSIBLE_MODEL,
                "confidence": 0.0,
                "best_match": None,
                "top_matches": public_top,
                "should_handle_in_chat": True,
                "extraction": _model_dump(extraction),
            }
        ordinal = 1
    if ordinal < 1 or ordinal > len(listings):
        return None

    listing = dict(listings[ordinal - 1])
    extraction = TrailerQueryExtraction(
        user_wants_price=_wants_price(user_query),
        user_wants_availability=_wants_availability(user_query),
        user_wants_details=_wants_details(user_query),
        search_intent="stock" if listing.get("stock_number") else "unknown",
    )
    match_result = {
        "entity_type": _STOCK if listing.get("stock_number") else _POSSIBLE_MODEL,
        "confidence": 1.0,
        "best_match": listing,
        "top_matches": [listing],
        "should_handle_in_chat": True,
    }
    return {
        "reply": generate_inventory_response(user_query, extraction, match_result),
        **_public_match_result(match_result),
        "extraction": _model_dump(extraction),
    }


def should_attempt_chat_lookup(
    user_query: str,
    *,
    last_listings: list[dict[str, Any]] | None = None,
) -> bool:
    if answer_from_last_listings(user_query, last_listings):
        return True
    extraction = _fallback_extraction(user_query)
    if _is_direct_inventory_lookup(extraction):
        return True
    return False


def is_potential_direct_inventory_lookup(user_query: str) -> bool:
    """Cheap prefilter used before invoking the inventory validation LLM."""
    return _is_direct_inventory_lookup(_fallback_extraction(user_query))


def validated_direct_inventory_extraction(
    user_query: str,
) -> TrailerQueryExtraction | None:
    """Return one LLM-validated direct lookup extraction, failing closed."""
    if not is_potential_direct_inventory_lookup(user_query):
        return None
    extraction = extract_trailer_query(user_query)
    approved = bool(
        extraction.should_lookup
        and extraction.lookup_confidence in {"medium", "high"}
        and _is_direct_inventory_lookup(extraction)
    )
    logger.info(
        "inventory_lookup_validation | approved=%s | confidence=%s | reason=%r | identifiers=%s",
        approved,
        extraction.lookup_confidence,
        extraction.lookup_reason,
        _requested_identifiers(extraction),
    )
    return extraction if approved else None


def search_trailers(
    user_query: str,
    *,
    last_listings: list[dict[str, Any]] | None = None,
    extraction: TrailerQueryExtraction | None = None,
    for_chat: bool = False,
    limit: int = 5,
) -> dict[str, Any]:
    context_answer = answer_from_last_listings(user_query, last_listings)
    if context_answer:
        return context_answer
    extraction = extraction or extract_trailer_query(user_query)
    match_result = match_inventory(user_query, extraction, limit=limit)
    reply = generate_inventory_response(user_query, extraction, match_result)
    public_match = _public_match_result(match_result)
    result = {
        "reply": reply,
        **public_match,
        "extraction": _model_dump(extraction),
    }
    if for_chat and not result.get("should_handle_in_chat"):
        return {**result, "reply": ""}
    return result
