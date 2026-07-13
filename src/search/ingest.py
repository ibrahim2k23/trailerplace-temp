"""Run once to embed all listings and upsert them to Pinecone (ported from the reference ``ingest.py``).

Usage:
    python -m src.search.ingest
    python -m src.search.ingest --force   # re-index even if vectors exist
    python -m src.search.ingest --force --no-wipe  # force re-upsert without clearing index
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
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

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
# the index here match how the query/rerank path re-parses the same raw strings.
from src.domain.units import parse_weight_lbs as parse_lbs, parse_length_ft

# src/search/ingest.py -> src/search -> src -> repository root.
# The workbook and .env live at the repository root, not beside this module.
_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(_ROOT / ".env")
load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = 1536
BATCH_SIZE = 50
MATCH_EVIDENCE_TEXT_MAX_CHARS = 12000
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
    """Flatten nested mappings to underscore-delimited Pinecone-safe keys."""
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


def build_flattened_embedding_text(
    flattened: dict[str, Any], title: str, features: list[str]
) -> str:
    """Build semantic text from every useful flattened JSON value.

    Core trailer fields retain the stable, query-aligned labels produced by
    build_embedding_text. Any remaining info/specification keys are appended
    with human-readable labels so new scraper fields affect similarity without
    requiring another code change.
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


def canonical_embedding_hash(
    flattened: dict[str, Any], title: str, features: list[str]
) -> str:
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
        "embedding_model": EMBEDDING_MODEL,
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


def get_or_create_index(pc: Pinecone):
    existing = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in existing:
        LOGGER.info("Creating Pinecone index | name=%s", INDEX_NAME)
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while True:
            status = pc.describe_index(INDEX_NAME).status
            if status.get("ready", False):
                break
            LOGGER.info("Waiting for Pinecone index readiness")
            time.sleep(2)
        LOGGER.info("Pinecone index ready")
    else:
        LOGGER.info("Using existing Pinecone index | name=%s", INDEX_NAME)
    return pc.Index(INDEX_NAME)


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


def _money_from_info(info: dict, *keys: str) -> Optional[float]:
    for k in keys:
        p = parse_money(info.get(normalize_json_key(k)))
        if p is not None:
            return p
    return None


def build_vector_id(row: pd.Series, row_idx: int) -> str:
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
    axles = col("axles") or None
    gvwr = col("gvwr") or None
    gvwr_lbs_num = parse_lbs(gvwr)
    payload = col("payload_capacity", "payload capacity") or None
    payload_lbs_num = parse_lbs(payload)
    material = col("trailer_material", "trailer material") or None
    floor = col("floor") or None
    length_ft_num = parse_length_ft(length)
    width_ft_num = parse_length_ft(width)

    # Direct workbook columns are authoritative. Merge them into the same
    # flattened map used for embedding and canonical change detection.
    embedding_info = dict(info)
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
        "axles": axles,
        "gvwr": gvwr,
        "payload_capacity": payload,
        "trailer_material": material,
        "floor": floor,
    }.items():
        if not is_blank(val):
            embedding_info[key] = val

    embedding_text = build_flattened_embedding_text(
        embedding_info, title, features
    )
    embedding_hash = canonical_embedding_hash(
        embedding_info, title, features
    )

    metadata: dict = {
        "title": title,
        "condition": condition,
        "category": category,
        "make": make,
        "color": color,
        "url": url,
        "price_display": price_display,
        "match_evidence_text": embedding_text[:MATCH_EVIDENCE_TEXT_MAX_CHARS],
        "embedding_hash": embedding_hash,
        "embedding_model": EMBEDDING_MODEL,
        "feature_extraction_version": FEATURE_EXTRACTION_VERSION,
        "info_json_source": json_source,
    }
    if features:
        metadata["features"] = features
    sub_norm = normalize_subcategory(raw_subcategory)
    if sub_norm and sub_norm.lower() != category.lower():
        metadata["subcategory"] = sub_norm
    for key, val in [
        ("price", price),
        ("hitch_type", hitch),
        ("year", year),
        ("model", model),
        ("trim", trim),
        ("length", length),
        ("width", width),
        ("axles", axles),
        ("gvwr", gvwr),
        ("gvwr_lbs_num", gvwr_lbs_num),
        ("length_ft_num", length_ft_num),
        ("width_ft_num", width_ft_num),
        ("payload_capacity", payload),
        ("payload_lbs_num", payload_lbs_num),
        ("trailer_material", material),
        ("floor", floor),
    ]:
        if val is not None:
            metadata[key] = val

    # Pinecone metadata must remain flat and cannot contain null values.
    metadata = {
        key: value
        for key, value in metadata.items()
        if value is not None and not (isinstance(value, str) and not value.strip())
    }

    vector_id = build_vector_id(row, row_idx)

    return {
        "id": vector_id,
        "stock_number": col("stock_number"),
        "embedding_text": embedding_text,
        "embedding_hash": embedding_hash,
        "metadata": metadata,
    }


def embed_batch(client: OpenAI, texts: list) -> list:
    response = retry_call(
        lambda: client.embeddings.create(model=EMBEDDING_MODEL, input=texts),
        "OpenAI embedding batch",
    )
    embeddings = [item.embedding for item in response.data]
    if len(embeddings) != len(texts):
        raise RuntimeError(
            f"Embedding count mismatch: expected {len(texts)}, got {len(embeddings)}"
        )
    return embeddings


def _response_vectors(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response.get("vectors", {}) or {}
    return getattr(response, "vectors", {}) or {}


def _vector_metadata(vector: Any) -> dict[str, Any]:
    if isinstance(vector, dict):
        return vector.get("metadata", {}) or {}
    return getattr(vector, "metadata", {}) or {}


def fetch_existing_hashes(index: Any, vector_ids: list[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for start in range(0, len(vector_ids), 100):
        ids = vector_ids[start : start + 100]
        response = retry_call(
            lambda ids=ids: index.fetch(ids=ids),
            "Pinecone fetch existing hashes",
        )
        for vector_id, vector in _response_vectors(response).items():
            value = _vector_metadata(vector).get("embedding_hash")
            if value:
                hashes[str(vector_id)] = str(value)
    return hashes


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
        "changed_embedding_hashes": 0,
        "skipped_unchanged_vectors": 0,
        "embedding_failures": 0,
        "vector_upsert_failures": 0,
        "embedded_vectors": 0,
        "upserted_vectors": 0,
    }
    quarantine: list[dict[str, Any]] = []

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

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = get_or_create_index(pc)

    if force and wipe_on_force:
        LOGGER.warning("Force mode: deleting all existing vectors")
        retry_call(lambda: index.delete(delete_all=True), "Pinecone delete all")
        time.sleep(2)

    if force:
        changed_records = records
    else:
        existing_hashes = fetch_existing_hashes(
            index, [record["id"] for record in records]
        )
        changed_records = [
            record
            for record in records
            if existing_hashes.get(record["id"]) != record["embedding_hash"]
        ]
        summary["skipped_unchanged_vectors"] = len(records) - len(changed_records)
    summary["changed_embedding_hashes"] = len(changed_records)
    LOGGER.info(
        "Incremental comparison | valid=%s changed=%s unchanged=%s invalid=%s",
        len(records),
        len(changed_records),
        summary["skipped_unchanged_vectors"],
        summary["invalid_json_rows"],
    )

    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    vectors = []
    for i in range(0, len(changed_records), BATCH_SIZE):
        batch = changed_records[i : i + BATCH_SIZE]
        try:
            embeddings = embed_batch(
                openai_client, [record["embedding_text"] for record in batch]
            )
            for record, embedding in zip(batch, embeddings):
                vectors.append(
                    {
                        "id": record["id"],
                        "values": embedding,
                        "metadata": record["metadata"],
                    }
                )
            summary["embedded_vectors"] += len(batch)
        except Exception as exc:  # noqa: BLE001
            summary["embedding_failures"] += len(batch)
            LOGGER.error(
                "Embedding batch permanently failed | rows=%s error=%s",
                len(batch),
                exc,
            )
            quarantine.extend(
                {
                    "stage": "embedding",
                    "id": record["id"],
                    "stock_number": record["stock_number"],
                    "error": str(exc),
                }
                for record in batch
            )
        LOGGER.info(
            "Embedding progress | processed=%s/%s successful=%s",
            min(i + len(batch), len(changed_records)),
            len(changed_records),
            len(vectors),
        )

    for i in range(0, len(vectors), 100):
        batch = vectors[i : i + 100]
        try:
            retry_call(
                lambda batch=batch: index.upsert(vectors=batch),
                "Pinecone upsert batch",
            )
            summary["upserted_vectors"] += len(batch)
        except Exception as exc:  # noqa: BLE001
            summary["vector_upsert_failures"] += len(batch)
            LOGGER.error(
                "Upsert batch permanently failed | rows=%s error=%s", len(batch), exc
            )
            quarantine.extend(
                {"stage": "upsert", "id": vector["id"], "error": str(exc)}
                for vector in batch
            )
        LOGGER.info(
            "Upsert progress | attempted=%s/%s successful=%s",
            min(i + len(batch), len(vectors)),
            len(vectors),
            summary["upserted_vectors"],
        )

    summary["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    summary["quarantined_rows"] = len(quarantine)
    write_json(QUARANTINE_FILE, quarantine)
    write_json(INGEST_REPORT_FILE, summary)
    LOGGER.info("Ingestion complete | summary=%s", summary)
    LOGGER.info("Pinecone stats | stats=%s", index.describe_index_stats())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-index even if vectors exist")
    parser.add_argument(
        "--no-wipe",
        action="store_true",
        help="With --force, do not clear existing vectors before upsert",
    )
    args = parser.parse_args()
    main(force=args.force, wipe_on_force=not args.no_wipe)
