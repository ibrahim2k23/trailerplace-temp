"""Run once to load all listings from the workbook into the ``trailer_listings`` table.

Usage:
    python -m src.search.ingest
    python -m src.search.ingest --force   # rewrite every row, ignoring content hashes
    python -m src.search.ingest --force --no-wipe  # force rewrite without deleting first

Replaces the former Pinecone ingest: rows are upserted to Postgres and there
are no embeddings. The flattening, feature extraction and change-detection
logic is unchanged — only the destination is different.
"""
import argparse
import hashlib
import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src import db
from src.db_models import TrailerListingRow
from src.domain.normalizer import (
    build_embedding_text,
    normalize_category,
    normalize_color,
    normalize_condition,
    normalize_hitch,
    normalize_make,
    normalize_subcategory,
)
# Weight/length parsing is single-sourced in units.py so the numbers written to
# the table here match how the query/rerank path re-parses the same raw strings.
from src.domain.units import parse_weight_lbs as parse_lbs, parse_length_ft

# src/search/ingest.py -> src/search -> src -> repository root.
# The workbook and .env live at the repository root, not beside this module.
_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(_ROOT / ".env")
load_dotenv()

UPSERT_BATCH_SIZE = 100
MATCH_EVIDENCE_TEXT_MAX_CHARS = 12000
# axle_capacity is a PER-AXLE rating (GVWR is ~2x it on the two-axle rows that dominate the
# catalogue). A handful of rows carry the axle COUNT in that column instead ("2 lbs" against a
# 14,000 lb GVWR); parsed as a capacity it would rank as an absurdly weak trailer and print
# "Axle capacity: 2 lbs" on the card. Real ratings start at 2000, so anything under this floor
# is the count, not a capacity, and is dropped.
MIN_PLAUSIBLE_AXLE_CAPACITY_LBS = 1000
# The mirror of the floor above: a handful of rows carry the axle CAPACITY in the
# count column ("axles": "8000"). Nothing on the lot has more than a handful of
# axles, so anything past this is the capacity, not a count, and is dropped.
MAX_PLAUSIBLE_AXLE_COUNT = 10
MAX_RETRIES = int(os.getenv("INGEST_MAX_RETRIES", "4"))
FEATURE_EXTRACTION_VERSION = os.getenv(
    "FEATURE_EXTRACTION_VERSION", "trailer-features-v2"
)

DATA_FILE = Path(
    os.getenv("LISTINGS_DATA_FILE", str(_ROOT / "listings.xlsx"))
).expanduser()
INGEST_REPORT_FILE = _ROOT / "ingest_report.json"
QUARANTINE_FILE = _ROOT / "ingest_quarantine.json"

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
LOGGER = logging.getLogger("inventory_ingest")
T = TypeVar("T")


def is_blank(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip() == ""


def normalize_json_key(value: Any) -> str:
    """Convert readable JSON labels to stable scalar/metadata keys."""
    key = str(value).strip().lower()
    key = re.sub(r"[\s\-]+", "_", key)
    return re.sub(r"_+", "_", key).strip("_")


def normalize_features(value: Any) -> list[str]:
    """Clean, sort, and case-insensitively deduplicate feature strings."""
    if is_blank(value):
        return []
    candidates = value if isinstance(value, list) else [value]
    unique: dict[str, str] = {}
    for candidate in candidates:
        if is_blank(candidate):
            continue
        feature = re.sub(r"\s+", " ", str(candidate)).strip(" .;,:\t\r\n")
        if feature:
            unique.setdefault(feature.casefold(), feature)
    # Deterministic ordering ensures reordered/case-only duplicates hash alike.
    return [unique[key] for key in sorted(unique)]


def _flatten_mapping(value: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested mappings to underscore-delimited scalar keys."""
    flattened: dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        key = normalize_json_key(raw_key)
        path = f"{prefix}_{key}" if prefix and key else (key or prefix)
        if not path:
            continue
        if isinstance(raw_value, dict):
            flattened.update(_flatten_mapping(raw_value, path))
        elif isinstance(raw_value, list):
            scalar_values = [
                item
                for item in raw_value
                if isinstance(item, (str, int, float, bool)) and not is_blank(item)
            ]
            if scalar_values:
                flattened[path] = scalar_values
        elif not is_blank(raw_value):
            flattened[path] = raw_value
    return flattened


def parse_and_flatten_info_specs(
    row: pd.Series,
) -> tuple[dict[str, Any], list[str], str]:
    """Parse nested info/spec JSON and return merged scalars plus features.

    ``info_specs_json`` is preferred. The older ``info_spec_json`` spelling and
    its legacy flat-object shape remain supported. Normalized specification
    values overwrite normalized info values on collisions.
    """
    plural = row.get("info_specs_json", "")
    singular = row.get("info_spec_json", "")
    if not is_blank(plural):
        raw_value, source = plural, "info_specs_json"
    elif not is_blank(singular):
        raw_value, source = singular, "info_spec_json"
    else:
        return {}, [], "none"

    if isinstance(raw_value, dict):
        blob = raw_value
    else:
        try:
            blob = json.loads(str(raw_value))
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"malformed {source}: {exc}") from exc
    if not isinstance(blob, dict):
        raise ValueError(f"{source} must contain a JSON object")

    if "info" in blob or "specifications" in blob:
        info = blob.get("info", {})
        specifications = blob.get("specifications", {})
        if not isinstance(info, dict):
            raise ValueError(f"{source}.info must be an object")
        if not isinstance(specifications, dict):
            raise ValueError(f"{source}.specifications must be an object")
    else:
        # Compatibility with the original, flat singular JSON column.
        info, specifications = blob, {}

    features = normalize_features(info.get("features"))
    normalized_info = _flatten_mapping(
        {
            key: value
            for key, value in info.items()
            if normalize_json_key(key) != "features"
        }
    )
    normalized_specifications = _flatten_mapping(specifications)
    return {**normalized_info, **normalized_specifications}, features, source


def build_flattened_evidence_text(
    flattened: dict[str, Any], title: str, features: list[str]
) -> str:
    """Build the match-evidence text from every useful flattened JSON value.

    Core trailer fields retain the stable labels produced by
    build_embedding_text. Any remaining info/specification keys are appended
    with human-readable labels so new scraper fields become visible to the
    feature reranker without requiring another code change.

    This is no longer an embedding input — it is the evidence the feature
    reranker (gpt-5-nano) and its deterministic fallback quote from.
    """
    base = build_embedding_text(flattened, title, "").strip()
    parts = [base] if base else []

    core_keys = {
        "title",
        "make",
        "year",
        "model",
        "trim",
        "condition",
        "category",
        "subcategory",
        "hitch_type",
        "color",
        "length",
        "width",
        "gvwr",
        "axles",
        "axle_capacity",
        "axle_count",
        "payload_capacity",
        "dry_weight",
        "trailer_material",
        "floor",
        "price",
        # Locked business rule: MSRP never influences retrieval.
        "msrp",
    }
    additional: list[str] = []
    for key in sorted(flattened):
        if key in core_keys:
            continue
        value = flattened[key]
        if is_blank(value):
            continue
        if isinstance(value, list):
            rendered = "; ".join(
                str(item).strip() for item in value if not is_blank(item)
            )
        else:
            rendered = str(value).strip()
        if not rendered:
            continue
        label = re.sub(r"_+", " ", key).strip().title()
        additional.append(f"{label}: {rendered}")

    if additional:
        parts.append("Additional specifications: " + " | ".join(additional))
    if features:
        parts.append("Features: " + "; ".join(features))
    return "\n".join(parts)


def canonical_content_hash(
    flattened: dict[str, Any], title: str, features: list[str]
) -> str:
    """Stable fingerprint of a row's meaningful content, for skip-unchanged."""
    canonical_scalars: dict[str, Any] = {}
    for key in sorted(flattened):
        value = flattened[key]
        if isinstance(value, list):
            canonical_scalars[key] = sorted(
                {str(item).strip().casefold() for item in value if str(item).strip()}
            )
        elif not is_blank(value):
            canonical_scalars[key] = str(value).strip()
    payload = {
        "title": title.strip(),
        "scalars": canonical_scalars,
        "features": sorted({feature.casefold() for feature in features}),
        "feature_extraction_version": FEATURE_EXTRACTION_VERSION,
    }
    serialized = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def retry_call(operation: Callable[[], T], operation_name: str) -> T:
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return operation()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == MAX_RETRIES:
                break
            delay = 2 ** (attempt - 1) + random.random()
            LOGGER.warning(
                "%s failed; retrying | attempt=%s/%s delay_seconds=%.2f error=%s",
                operation_name,
                attempt,
                MAX_RETRIES,
                delay,
                exc,
            )
            time.sleep(delay)
    assert last_error is not None
    raise last_error


def parse_money(val) -> Optional[float]:
    """Parse currency from Excel/JSON: numbers, '$8,400', '8400', NaN-safe."""
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    if isinstance(val, (int, float)):
        try:
            v = float(val)
            return v if v > 0 else None
        except (TypeError, ValueError):
            return None
    s = str(val).strip()
    if not s:
        return None
    s = re.sub(r"[$,\s]", "", s)
    try:
        v = float(s)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


# The `axles` field is not always a bare number. Real values from the catalogue:
#   "2 x 7,000 lb"                                  -> 2 axles
#   "Tandem | 10,400 lbs"                           -> 2 axles (10,400 is the TOTAL)
#   "Single | Total: 3,500 lbs"                     -> 1 axle
#   "Type: Spring | Count: 2 | Rating: 7,000#"      -> 2 axles
#   "2000# Rubber torsion axle - No brakes"         -> a RATING, no count stated
# Taking the first number in the string reads 10,400 as a count on the second of
# those. The out-of-range guard then discards it, so nothing wrong was ever
# stored - but three listings silently lost a count they had plainly stated.
_AXLE_WORDS = {
    "single": 1, "tandem": 2, "tri": 3, "triple": 3, "quad": 4, "quadruple": 4,
}
# The short forms the spec lists, matched only as whole tokens.
_AXLE_CODES = {"SA": 1, "TA": 2, "TRI": 3, "3A": 3, "QA": 4, "4A": 4, "SPA": 2}
_BARE_NUMBER_RE = re.compile(r"^\d+(?:\.0+)?$")
_COUNT_LABEL_RE = re.compile(r"\b(?:count|qty|quantity|number)\s*[:=]?\s*(\d+)", re.I)
_N_AXLES_RE = re.compile(r"\b(\d+)\s*axles?\b", re.I)
_LEADING_N_RE = re.compile(r"^\s*(\d+)\s*[x\-]\s*[\d$]", re.I)
_WORD_RE = re.compile(
    r"(?<![A-Za-z])(" + "|".join(_AXLE_WORDS) + r")(?![A-Za-z])", re.I
)
_CODE_RE = re.compile(
    r"(?<![A-Za-z0-9])(" + "|".join(_AXLE_CODES) + r")(?![A-Za-z0-9])"
)


def parse_axle_count(val) -> Optional[int]:
    """How many axles, or None when the field does not actually say.

    Reads the count rather than the first digits it can find, because the field
    carries ratings and totals too and those are far larger numbers. A value that
    states only a rating yields None: "2000# Rubber torsion axle" is one fact
    about the axles and it is not how many there are.
    """
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return _in_range(int(val)) if float(val).is_integer() else None

    text = str(val).strip()
    if not text:
        return None
    # Excel round-trips a column with any gap in it as float, so "2.0" arrives as
    # often as "2"; a bare number is the count and nothing else.
    if _BARE_NUMBER_RE.match(text.replace(",", "")):
        return _in_range(int(float(text.replace(",", ""))))

    for pattern in (_COUNT_LABEL_RE, _N_AXLES_RE, _LEADING_N_RE):
        match = pattern.search(text)
        if match:
            return _in_range(int(match.group(1)))

    match = _WORD_RE.search(text)
    if match:
        return _AXLE_WORDS[match.group(1).casefold()]
    match = _CODE_RE.search(text)
    if match:
        return _AXLE_CODES[match.group(1).upper()]
    return None


def _in_range(count: int) -> Optional[int]:
    """A number outside this range is a rating that reached the count field."""
    return count if 1 <= count <= MAX_PLAUSIBLE_AXLE_COUNT else None


def _money_from_info(info: dict, *keys: str) -> Optional[float]:
    for k in keys:
        p = parse_money(info.get(normalize_json_key(k)))
        if p is not None:
            return p
    return None


def build_listing_id(row: pd.Series, row_idx: int) -> str:
    """Stable primary key for a workbook row.

    Unchanged from the Pinecone vector id so re-ingest keeps updating the same
    rows rather than duplicating them.
    """
    stock_number = str(row.get("stock_number", "")).strip()
    url = str(row.get("url", "")).strip()
    if stock_number:
        if url:
            short = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
            return f"stock_{stock_number}_{short}"
        return f"stock_{stock_number}"
    hin = str(row.get("hin", "")).strip()
    if hin:
        return f"hin_{hin}"
    if url:
        short = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
        return f"url_{short}"
    return f"row_{row_idx}"


def build_record(row: pd.Series, row_idx: int) -> dict:
    info, features, json_source = parse_and_flatten_info_specs(row)

    def col(name: str, info_key: Optional[str] = None) -> str:
        val = row.get(name, "")
        if is_blank(val):
            val = info.get(normalize_json_key(info_key or name), "")
        return "" if is_blank(val) else str(val).strip()

    raw_category = col("category")
    raw_subcategory = col("subcategory")
    raw_make = col("make")
    raw_color = col("color")
    raw_hitch = col("hitch_type")
    raw_condition = col("condition")
    title = col("title")
    url = col("url")

    condition = normalize_condition(raw_condition)
    category = normalize_category(raw_category)
    make = normalize_make(raw_make)
    color = normalize_color(raw_color)
    hitch = normalize_hitch(raw_hitch) if raw_hitch else None

    price = parse_money(row.get("price")) or _money_from_info(
        info, "price", "Price", "our_price", "Our Price"
    )

    price_display = f"${price:,.0f}" if price is not None else "Call for price"

    year = col("year") or None
    model = col("model") or None
    trim = col("trim") or None
    length = col("length") or None
    width = col("width") or None
    # Height was read by the search path but never written by the old ingest,
    # so every height requirement silently scored as a missing dimension.
    height = col("height") or None
    axles = col("axles") or None
    gvwr = col("gvwr") or None
    gvwr_lbs_num = parse_lbs(gvwr)
    payload = col("payload_capacity", "payload capacity") or None
    payload_lbs_num = parse_lbs(payload)
    # The scraper extracts this; older workbooks predate the column and fall back
    # to the raw `axles` text, which is a bare number on the rows that have it.
    axle_count = parse_axle_count(col("axle_count", "axle count") or axles)
    axle_capacity = col("axle_capacity", "axle capacity") or None
    axle_capacity_lbs_num = parse_lbs(axle_capacity)
    if axle_capacity_lbs_num is not None and axle_capacity_lbs_num < MIN_PLAUSIBLE_AXLE_CAPACITY_LBS:
        # The axle COUNT landed in the capacity column. Drop the display string too, so the
        # number never reaches the fit rerank and "2 lbs" never reaches a listing card.
        axle_capacity = None
        axle_capacity_lbs_num = None
    # What the axles carry between them. Arithmetic on two known facts, not an
    # inference: it is computed only when both parts are present, so a listing
    # that states a capacity but no count yields nothing here rather than a total
    # that quietly assumes two axles.
    # The scraper works this out and writes it; recomputing here would mean
    # teaching the same arithmetic twice, so its value wins when the workbook
    # carries one. Older workbooks predate the column and fall back to the
    # multiplication.
    total_axle_capacity_lbs_num = parse_lbs(col("total_axle_capacity", "total axle capacity")) or (
        float(axle_count) * axle_capacity_lbs_num
        if axle_count is not None and axle_capacity_lbs_num is not None
        else None
    )
    material = col("trailer_material", "trailer material") or None
    floor = col("floor") or None
    length_ft_num = parse_length_ft(length)
    width_ft_num = parse_length_ft(width)
    height_ft_num = parse_length_ft(height)

    # Direct workbook columns are authoritative. Merge them into the same
    # flattened map used for evidence text and canonical change detection.
    evidence_info = dict(info)
    for key, val in {
        "year": year,
        "make": raw_make,
        "model": model,
        "trim": trim,
        "category": raw_category,
        "subcategory": raw_subcategory,
        "condition": raw_condition,
        "color": raw_color,
        "hitch_type": raw_hitch,
        "length": length,
        "width": width,
        "height": height,
        "axles": axles,
        "gvwr": gvwr,
        # Also feeds canonical_content_hash: without it, adding axle capacity to an existing
        # catalogue leaves every hash unchanged and an incremental re-ingest skips every row.
        "axle_capacity": axle_capacity,
        # Same again for the count, and it bites harder: the capacity only changes
        # on the handful of corrupt rows, so without the count in the hash the
        # other 240 listings would keep their old hash, be skipped as unchanged,
        # and never receive the count at all.
        "axle_count": axle_count,
        # And the total, for the same reason once more. It is usually derived from the two
        # above and so rides along on their changes - but the workbook's own value wins when
        # it carries one, and a row where ONLY that changed would otherwise keep its old
        # hash and be skipped as unchanged.
        "total_axle_capacity": total_axle_capacity_lbs_num,
        "payload_capacity": payload,
        "trailer_material": material,
        "floor": floor,
    }.items():
        if not is_blank(val):
            evidence_info[key] = val

    evidence_text = build_flattened_evidence_text(
        evidence_info, title, features
    )
    content_hash = canonical_content_hash(
        evidence_info, title, features
    )

    sub_norm = normalize_subcategory(raw_subcategory)
    # A subcategory that merely repeats the category carries no information and
    # would make the Aluminum subcategory gate match on noise.
    subcategory = sub_norm if sub_norm and sub_norm.lower() != category.lower() else None

    # One dict per table column. Unlike Pinecone metadata, NULLs are welcome
    # here, so blanks stay blank instead of being dropped from the payload.
    return {
        "listing_id": build_listing_id(row, row_idx),
        "stock_number": col("stock_number") or None,
        "title": title or None,
        "url": url or None,
        "condition": condition,
        "category": category,
        "subcategory": subcategory,
        "make": make,
        "color": color,
        "hitch_type": hitch,
        "price": price,
        "price_display": price_display,
        "year": year,
        "model": model,
        "trim": trim,
        "length": length,
        "width": width,
        "height": height,
        "axles": axles,
        "gvwr": gvwr,
        "axle_capacity": axle_capacity,
        "axle_count": axle_count,
        "payload_capacity": payload,
        "length_ft_num": length_ft_num,
        "width_ft_num": width_ft_num,
        "height_ft_num": height_ft_num,
        "gvwr_lbs_num": gvwr_lbs_num,
        "payload_lbs_num": payload_lbs_num,
        "axle_capacity_lbs_num": axle_capacity_lbs_num,
        "total_axle_capacity_lbs_num": total_axle_capacity_lbs_num,
        "trailer_material": material,
        "floor": floor,
        "features": features,
        "match_evidence_text": evidence_text[:MATCH_EVIDENCE_TEXT_MAX_CHARS],
        "content_hash": content_hash,
        "info_json_source": json_source,
    }


def fetch_existing_hashes() -> dict[str, str]:
    """Current content hash per listing id. One query — the table is small."""
    with db.get_session_factory()() as session:
        rows = session.execute(
            select(TrailerListingRow.listing_id, TrailerListingRow.content_hash)
        ).all()
    return {row[0]: row[1] for row in rows if row[1]}


# Every column the upsert overwrites on conflict. listing_id is the conflict
# target and created_at must survive, so both are excluded.
_UPSERT_COLUMNS = tuple(
    column.name
    for column in TrailerListingRow.__table__.columns
    if column.name not in {"listing_id", "created_at"}
)


def upsert_batch(rows: list[dict[str, Any]]) -> None:
    """Insert or update one batch of listings by primary key."""
    statement = pg_insert(TrailerListingRow).values(rows)
    statement = statement.on_conflict_do_update(
        index_elements=[TrailerListingRow.listing_id],
        set_={
            name: getattr(statement.excluded, name)
            for name in _UPSERT_COLUMNS
            if name != "updated_at"
        }
        # onupdate= only fires for ORM-level updates, not this INSERT ... ON
        # CONFLICT, so the timestamp is set explicitly.
        | {"updated_at": func.now()},
    )
    with db.get_session_factory()() as session:
        session.execute(statement)
        session.commit()


def prune_missing(known_ids: set[str]) -> int:
    """Delete rows whose listing is no longer in the workbook.

    Pinecone never pruned, so retired stock lingered in the index until a
    --force wipe. A DELETE is cheap here, so retired listings go on every run.
    """
    if not known_ids:
        return 0
    with db.get_session_factory()() as session:
        result = session.execute(
            delete(TrailerListingRow).where(TrailerListingRow.listing_id.notin_(known_ids))
        )
        session.commit()
        return int(result.rowcount or 0)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def main(force: bool = False, wipe_on_force: bool = True):
    started = time.perf_counter()
    summary = {
        "input_rows": 0,
        "valid_json_rows": 0,
        "invalid_json_rows": 0,
        "changed_content_hashes": 0,
        "skipped_unchanged_rows": 0,
        "row_upsert_failures": 0,
        "upserted_rows": 0,
        "pruned_rows": 0,
    }
    quarantine: list[dict[str, Any]] = []

    if not db.database_enabled():
        raise RuntimeError(
            "Database settings are incomplete; set HOST/PORT/DATABASE/PGUSER/PASSWORD "
            "before running ingest."
        )

    LOGGER.info("Loading data | path=%s", DATA_FILE)
    df = pd.read_excel(DATA_FILE)
    msrp_cols = [c for c in df.columns if str(c).lower() == "msrp"]
    if msrp_cols:
        df = df.drop(columns=msrp_cols)
    summary["input_rows"] = len(df)
    LOGGER.info("Workbook loaded | rows=%s msrp_excluded=%s", len(df), bool(msrp_cols))

    LOGGER.info("Building flattened records")
    records: list[dict[str, Any]] = []
    for row_idx, (_, row) in enumerate(df.iterrows()):
        stock_number = (
            ""
            if is_blank(row.get("stock_number"))
            else str(row.get("stock_number")).strip()
        )
        try:
            records.append(build_record(row, row_idx))
            summary["valid_json_rows"] += 1
        except ValueError as exc:
            summary["invalid_json_rows"] += 1
            LOGGER.error(
                "Invalid info/spec JSON; row quarantined | stock_number=%s error=%s",
                stock_number or "<missing>",
                exc,
            )
            quarantine.append(
                {
                    "stage": "json_parse",
                    "row_index": row_idx,
                    "stock_number": stock_number,
                    "error": str(exc),
                }
            )

    if force and wipe_on_force:
        LOGGER.warning("Force mode: deleting all existing listings")
        with db.get_session_factory()() as session:
            session.execute(delete(TrailerListingRow))
            session.commit()

    if force:
        changed_records = records
    else:
        existing_hashes = fetch_existing_hashes()
        changed_records = [
            record
            for record in records
            if existing_hashes.get(record["listing_id"]) != record["content_hash"]
        ]
        summary["skipped_unchanged_rows"] = len(records) - len(changed_records)
    summary["changed_content_hashes"] = len(changed_records)
    LOGGER.info(
        "Incremental comparison | valid=%s changed=%s unchanged=%s invalid=%s",
        len(records),
        len(changed_records),
        summary["skipped_unchanged_rows"],
        summary["invalid_json_rows"],
    )

    for i in range(0, len(changed_records), UPSERT_BATCH_SIZE):
        batch = changed_records[i : i + UPSERT_BATCH_SIZE]
        try:
            retry_call(lambda batch=batch: upsert_batch(batch), "Listing upsert batch")
            summary["upserted_rows"] += len(batch)
        except Exception as exc:  # noqa: BLE001
            summary["row_upsert_failures"] += len(batch)
            LOGGER.error(
                "Upsert batch permanently failed | rows=%s error=%s", len(batch), exc
            )
            quarantine.extend(
                {
                    "stage": "upsert",
                    "listing_id": record["listing_id"],
                    "stock_number": record["stock_number"],
                    "error": str(exc),
                }
                for record in batch
            )
        LOGGER.info(
            "Upsert progress | attempted=%s/%s successful=%s",
            min(i + len(batch), len(changed_records)),
            len(changed_records),
            summary["upserted_rows"],
        )

    # Only prune when every row parsed. A workbook that failed halfway would
    # otherwise look like a catalogue that had suddenly shrunk, and we would
    # delete live listings on the strength of a parse error.
    if not summary["invalid_json_rows"] and not summary["row_upsert_failures"]:
        summary["pruned_rows"] = prune_missing(
            {record["listing_id"] for record in records}
        )
    else:
        LOGGER.warning(
            "Skipping prune | invalid_json_rows=%s row_upsert_failures=%s",
            summary["invalid_json_rows"],
            summary["row_upsert_failures"],
        )

    with db.get_session_factory()() as session:
        summary["table_rows"] = int(
            session.execute(select(func.count()).select_from(TrailerListingRow)).scalar() or 0
        )

    summary["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    summary["quarantined_rows"] = len(quarantine)
    write_json(QUARANTINE_FILE, quarantine)
    write_json(INGEST_REPORT_FILE, summary)
    LOGGER.info("Ingestion complete | summary=%s", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Rewrite every row, ignoring hashes")
    parser.add_argument(
        "--no-wipe",
        action="store_true",
        help="With --force, do not delete existing rows before upsert",
    )
    args = parser.parse_args()
    main(force=args.force, wipe_on_force=not args.no_wipe)
