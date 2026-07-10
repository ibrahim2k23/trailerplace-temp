from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Literal

from src.chatbot.make_aliases import MAKE_ALIASES as _ALIASES
from src.chatbot.make_inventory import known_makes


MatchType = Literal["exact", "alias", "partial", "fuzzy", "llm"]


@dataclass(frozen=True)
class MakeResolution:
    make: str | None = None
    confidence: str = "none"
    match_type: MatchType | None = None
    reason: str = ""


_GENERIC_TOKENS = {
    "a",
    "an",
    "and",
    "any",
    "available",
    "brand",
    "by",
    # "cargo" is an everyday word here ("what cargo can it carry?") that also happens to
    # be a token of the makes Cargo Craft and Continental Cargo. Treat it as generic so
    # it can never seed a partial/fuzzy brand match; the full phrase still resolves via
    # the exact/alias tier, which reads the raw text rather than these tokens.
    "cargo",
    "do",
    "for",
    "have",
    "i",
    "looking",
    "made",
    "make",
    "need",
    "price",
    "show",
    "the",
    "trailer",
    "trailers",
    "want",
    "with",
}

_GOOSENECK_BRAND_RE = re.compile(
    r"\b("
    r"gooseneck\s+(?:brand|make|manufacturer)"
    r"|(?:brand|make|manufacturer)\s+(?:called\s+|named\s+)?gooseneck"
    r"|made\s+by\s+gooseneck"
    r")\b",
    re.I,
)


def _norm(value: str) -> str:
    text = str(value or "").lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> list[str]:
    return [token for token in _norm(text).split() if token and token not in _GENERIC_TOKENS]


@lru_cache(maxsize=1)
def _valid_make_map() -> dict[str, str]:
    # Inventory makes are static at runtime; rebuilding this map on every make
    # resolution (a per-query hot path) is wasted work, so cache it. Callers
    # read it without mutating.
    valid = set(known_makes())
    out: dict[str, str] = {}
    for make in valid:
        out[_norm(make)] = make
    for alias, make in _ALIASES.items():
        if make in valid:
            out[_norm(alias)] = make
    return out


def _gooseneck_brand_is_explicit(text: str) -> bool:
    return bool(_GOOSENECK_BRAND_RE.search(text or ""))


def _candidate_allowed(make: str, text: str) -> bool:
    if make != "Gooseneck":
        return True
    return _gooseneck_brand_is_explicit(text)


def _resolve_deterministic(text: str) -> MakeResolution:
    normalized_message = f" {_norm(text)} "
    make_map = _valid_make_map()

    for phrase, make in sorted(make_map.items(), key=lambda item: len(item[0]), reverse=True):
        if not phrase:
            continue
        if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", normalized_message):
            if not _candidate_allowed(make, text):
                continue
            return MakeResolution(make, "high", "exact" if phrase == _norm(make) else "alias")

    message_tokens = _tokens(text)
    for token in message_tokens:
        if len(token) < 4 or token == "aluminum":
            continue
        for phrase, make in sorted(make_map.items()):
            # Only single-word makes may be reached by a prefix match. Allowing a bare
            # word to stand in for one token of a multi-word make ("cargo" -> Cargo
            # Craft, "bull" -> Iron Bull Trailers) matched ordinary sentences; the
            # short forms customers actually type are already in the alias map, which
            # the exact/alias tier above resolves at high confidence.
            if " " in phrase:
                continue
            if phrase.startswith(token):
                if not _candidate_allowed(make, text):
                    continue
                return MakeResolution(make, "medium", "partial")

    best_make: str | None = None
    best_score = 0.0
    ngrams: list[str] = []
    for size in (1, 2, 3):
        for i in range(0, max(0, len(message_tokens) - size + 1)):
            ngrams.append(" ".join(message_tokens[i : i + size]))
    for ngram in ngrams:
        if len(ngram) < 4:
            continue
        for phrase, make in make_map.items():
            if not _candidate_allowed(make, text):
                continue
            score = SequenceMatcher(None, ngram, phrase).ratio()
            if score > best_score:
                best_score = score
                best_make = make
    if best_make and best_score >= 0.86 and _candidate_allowed(best_make, text):
        return MakeResolution(best_make, "medium", "fuzzy", f"score={best_score:.2f}")

    return MakeResolution()


def resolve_make_from_text(text: str, *, use_llm_fallback: bool = True) -> MakeResolution:
    # Makes must always be grounded in the user's text. Keep the argument for
    # caller compatibility, but never allow the LLM to choose a make.
    del use_llm_fallback
    return _resolve_deterministic(text)
