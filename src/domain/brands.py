from __future__ import annotations

import difflib
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.domain.categories import CANONICAL_CATEGORIES
from src.domain.normalizer import normalize_category, normalize_make


_ROOT = Path(__file__).resolve().parents[2]
# The SAME workbook (and env override) the Pinecone ingest reads (src/search/ingest.py):
# the brands/categories the prompts advertise must be the inventory the search can actually
# return. This used to read listings_final_v5.xlsx while ingest read listings.xlsx, so the
# prompts promised makes the index had never seen.
_LISTINGS_FILE = Path(os.getenv("LISTINGS_DATA_FILE", str(_ROOT / "listings.xlsx")))

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

    try:
        import pandas as pd

        df = pd.read_excel(_LISTINGS_FILE)
    except ImportError:
        return MakeInventory((), {}, {})
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
        if category == "Unknown" or category not in CANONICAL_CATEGORIES:
            # Only categories the rest of the system can actually qualify and search. The
            # workbook carries a handful of rows outside the canonical 13 (e.g. "Welding")
            # — advertised in the makes block, they contradicted the same prompt's "we
            # carry exactly 13 categories" line and could be offered in the brand-category
            # question as a choice nothing downstream could handle.
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
        "Canonical trailer makes/brands available in the current inventory, with their available trailer categories:",
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


# Words that appear in a make's name but carry no identifying weight — "trailer" would
# otherwise match the word "trailer" in almost any message and hand us a brand filter the
# customer never asked for.
_GENERIC_MAKE_WORDS = frozenset({"trailer", "trailers", "inc", "llc", "co", "company", "industries", "mfg"})


def brand_mentioned_in_text(brand: str, text: str) -> bool:
    """Did the customer actually NAME this brand in this message?

    The extractor will happily report a make it read off a listing already on screen — so a
    turn that only hands over an email address comes back with brand_preference="Iron Bull
    Trailers", which then silently narrows every later search to one manufacturer. A brand is
    a brand only when they said it, typos included ("dimond c" -> Diamond C).
    """
    words = re.findall(r"[a-z0-9&]+", (text or "").lower())
    brand_words = [
        word
        for word in re.findall(r"[a-z0-9&]+", (brand or "").lower())
        if len(word) >= 3 and word not in _GENERIC_MAKE_WORDS
    ]
    if not words or not brand_words:
        return False
    return any(
        word == brand_word or difflib.SequenceMatcher(None, word, brand_word).ratio() >= 0.8
        for brand_word in brand_words
        for word in words
    )


def categories_for_make(make: str) -> tuple[str, ...]:
    return load_make_inventory().categories_by_make.get(make, ())


def make_filter_values(make: str) -> tuple[str, ...]:
    inventory = load_make_inventory()
    return inventory.filter_values_by_make.get(make, (make,))
