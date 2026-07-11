from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from src.normalizer import normalize_category, normalize_make


_ROOT = Path(__file__).resolve().parents[2]
_LISTINGS_FILE = _ROOT / "listings_final_v5.xlsx"

_CATEGORY_ALIASES = {
    "Atv Trailer": "Utility",
    "Concession": "Enclosed",
    "Landscape": "Utility",
    "Tank": "Diesel Tank",
}


@dataclass(frozen=True)
class MakeInventory:
    canonical_makes: tuple[str, ...]
    categories_by_make: dict[str, tuple[str, ...]]
    filter_values_by_make: dict[str, tuple[str, ...]]


def _display_make(value: Any) -> str:
    # normalize_make now yields the canonical short display form directly
    # (make_aliases.py), so no post-hoc display override is needed.
    return normalize_make(str(value or "").strip())


def _display_category(value: Any) -> str:
    normalized = normalize_category(str(value or "").strip())
    return _CATEGORY_ALIASES.get(normalized, normalized)


@lru_cache(maxsize=1)
def load_make_inventory() -> MakeInventory:
    if not _LISTINGS_FILE.exists():
        return MakeInventory((), {}, {})

    df = pd.read_excel(_LISTINGS_FILE)
    if "make" not in df.columns or "category" not in df.columns:
        return MakeInventory((), {}, {})

    categories: dict[str, set[str]] = {}
    filter_values: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        raw_make = str(row.get("make") or "").strip()
        raw_category = str(row.get("category") or "").strip()
        if not raw_make or not raw_category:
            continue

        make = _display_make(raw_make)
        category = _display_category(raw_category)
        if category == "Unknown":
            continue

        categories.setdefault(make, set()).add(category)
        variants = filter_values.setdefault(make, set())
        variants.add(raw_make)
        variants.add(normalize_make(raw_make))

    canonical_makes = tuple(sorted(categories))
    return MakeInventory(
        canonical_makes=canonical_makes,
        categories_by_make={make: tuple(sorted(values)) for make, values in categories.items()},
        filter_values_by_make={
            make: tuple(sorted(v for v in values if v and v != "Unknown"))
            for make, values in filter_values.items()
        },
    )


def known_makes() -> tuple[str, ...]:
    return load_make_inventory().canonical_makes


def make_prompt_block() -> str:
    inventory = load_make_inventory()
    prompt_makes = tuple(
        make for make in inventory.canonical_makes
        if make not in {"Gooseneck", "Bumper Pull"}
    )
    lines = [
        "Canonical trailer makes/brands available in the current inventory:",
        "Gooseneck and Bumper Pull are strictly hitch types, never trailer makes/brands. "
        "Do not infer, recommend, or return either one as a make.",
    ]
    if not prompt_makes:
        lines.append("- No makes are currently available from the inventory workbook.")
        return "\n".join(lines)

    for make in prompt_makes:
        categories = tuple(
            category
            for category in inventory.categories_by_make.get(make, ())
            if category not in {"Gooseneck", "Bumper Pull"}
        )
        category_text = ", ".join(categories) if categories else "category unavailable"
        lines.append(f"- {make}: {category_text}")
    return "\n".join(lines)


def categories_for_make(make: str) -> tuple[str, ...]:
    return load_make_inventory().categories_by_make.get(make, ())


def make_filter_values(make: str) -> tuple[str, ...]:
    inventory = load_make_inventory()
    return inventory.filter_values_by_make.get(make, (make,))
