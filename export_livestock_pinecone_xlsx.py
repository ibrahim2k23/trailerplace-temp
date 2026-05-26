from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone


load_dotenv()

DEFAULT_INDEX_NAME = "trailer-recommendations"
DEFAULT_OUTPUT_FILE = Path("temp.xlsx")
DEFAULT_NAMESPACE = "__default__"
FETCH_BATCH_SIZE = 100


def _vector_id(item: Any) -> str | None:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        value = item.get("id")
        return str(value) if value else None
    value = getattr(item, "id", None)
    return str(value) if value else None


def _iter_ids(index, *, namespace: str) -> Iterable[str]:
    try:
        for page in index.list(namespace=namespace):
            for item in page:
                vector_id = _vector_id(item)
                if vector_id:
                    yield vector_id
        return
    except Exception:
        pass

    pagination_token = None
    while True:
        kwargs: dict[str, Any] = {"namespace": namespace}
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


def _cell_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return value


def _vector_metadata(vector: Any) -> dict[str, Any]:
    if isinstance(vector, dict):
        return vector.get("metadata", {}) or {}
    return vector.metadata or {}


def fetch_livestock_rows(index_name: str, *, namespace: str = "") -> list[dict[str, Any]]:
    if not os.getenv("PINECONE_API_KEY"):
        raise RuntimeError("PINECONE_API_KEY is missing")

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(index_name)

    ids = list(_iter_ids(index, namespace=namespace))
    rows: list[dict[str, Any]] = []

    for batch in _chunks(ids, FETCH_BATCH_SIZE):
        fetched = index.fetch(ids=batch, namespace=namespace)
        vectors = fetched.get("vectors", {}) if isinstance(fetched, dict) else fetched.vectors

        for vector_id, vector in vectors.items():
            metadata = _vector_metadata(vector)
            category = str(metadata.get("category") or "").strip().lower()
            if category != "livestock":
                continue

            rows.append(
                {
                    "id": vector_id,
                    **{key: _cell_value(value) for key, value in metadata.items()},
                }
            )

    return rows


def write_xlsx(rows: list[dict[str, Any]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    dataframe = pd.DataFrame(rows)
    if not dataframe.empty and "id" in dataframe.columns:
        ordered_columns = ["id"] + sorted(col for col in dataframe.columns if col != "id")
        dataframe = dataframe[ordered_columns]

    dataframe.to_excel(output_file, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export all Livestock trailers from Pinecone to an XLSX file."
    )
    parser.add_argument(
        "--index",
        default=os.getenv("PINECONE_INDEX_NAME") or DEFAULT_INDEX_NAME,
        help=f"Pinecone index name. Defaults to PINECONE_INDEX_NAME or {DEFAULT_INDEX_NAME}.",
    )
    parser.add_argument(
        "--namespace",
        default=DEFAULT_NAMESPACE,
        help=f"Pinecone namespace. Defaults to {DEFAULT_NAMESPACE}.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_FILE),
        help=f"Output XLSX path. Defaults to {DEFAULT_OUTPUT_FILE}.",
    )
    args = parser.parse_args()

    rows = fetch_livestock_rows(args.index.strip(), namespace=args.namespace)
    output_file = Path(args.output)
    write_xlsx(rows, output_file)

    print(f"Wrote {len(rows)} Livestock rows from '{args.index}' to {output_file}")


if __name__ == "__main__":
    main()
