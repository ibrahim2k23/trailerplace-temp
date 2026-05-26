from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from pinecone import Pinecone


load_dotenv()

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "trailerplace-listings").strip()
OUTPUT_FILE = Path("temp.csv")
FETCH_BATCH_SIZE = 100


def _iter_ids(index) -> Iterable[str]:
    try:
        for page in index.list(namespace=""):
            for vector_id in page:
                yield str(vector_id)
        return
    except Exception:
        pass

    pagination_token = None
    while True:
        kwargs: dict[str, Any] = {"namespace": ""}
        if pagination_token:
            kwargs["pagination_token"] = pagination_token
        page = index.list_paginated(**kwargs)
        for item in page.get("vectors", []):
            vector_id = item.get("id")
            if vector_id:
                yield str(vector_id)
        pagination_token = (page.get("pagination") or {}).get("next")
        if not pagination_token:
            break


def _chunks(items: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True, default=str)
    return value


def main() -> None:
    if not os.getenv("PINECONE_API_KEY"):
        raise RuntimeError("PINECONE_API_KEY is missing")

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(INDEX_NAME)

    ids = list(_iter_ids(index))
    rows: list[dict[str, Any]] = []
    fieldnames = {"id"}

    for batch in _chunks(ids, FETCH_BATCH_SIZE):
        fetched = index.fetch(ids=batch, namespace="")
        vectors = fetched.get("vectors", {}) if isinstance(fetched, dict) else fetched.vectors
        for vector_id, vector in vectors.items():
            metadata = vector.get("metadata", {}) if isinstance(vector, dict) else (vector.metadata or {})
            category = str(metadata.get("category") or "").strip().lower()
            if category != "livestock":
                continue
            row = {"id": vector_id, **{key: _csv_value(value) for key, value in metadata.items()}}
            rows.append(row)
            fieldnames.update(row.keys())

    ordered_fieldnames = ["id"] + sorted(name for name in fieldnames if name != "id")
    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} Livestock rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
