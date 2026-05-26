from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from src.chatbot.make_inventory import known_makes


MatchType = Literal["exact", "alias", "partial", "fuzzy", "llm"]


@dataclass(frozen=True)
class MakeResolution:
    make: str | None = None
    confidence: str = "none"
    match_type: MatchType | None = None
    reason: str = ""


class _LLMMakeResolution(BaseModel):
    make: str | None = None
    confidence: Literal["none", "low", "medium", "high"] = "none"
    reason: str = ""


_ALIASES = {
    "alcom": "Alcom",
    "aluma": "Aluma",
    "alum": "Aluma",
    "ameritrail": "AmeriTrail",
    "ameri trail": "AmeriTrail",
    "ameraitrail": "AmeriTrail",
    "baseline": "Baseline",
    "cm trailers": "CM TRAILERS",
    "cm trailer": "CM TRAILERS",
    "calico trailers": "Calico Trailers",
    "calico": "Calico Trailers",
    "cargo craft": "Cargo Craft",
    "cargo craft trailers": "Cargo Craft",
    "continental cargo": "Continental Cargo",
    "diamond c": "Diamond C",
    "diamond c trailers": "Diamond C",
    "dimond c": "Diamond C",
    "east texas": "East Texas Trailers",
    "east texas trailers": "East Texas Trailers",
    "fairwest": "FAIRWEST",
    "galyean": "Galyean",
    "iron bull": "Iron Bull Trailers",
    "iron bull trailers": "Iron Bull Trailers",
    "j&j": "J&J Trailer",
    "j and j": "J&J Trailer",
    "j j": "J&J Trailer",
    "kaufman": "Kaufman Trailers",
    "kaufman trailers": "Kaufman Trailers",
    "lark united manufacturing": "LARK UNITED MANUFACTURING",
    "liberty": "Liberty",
    "norstar": "Norstar",
    "p&c": "P&C",
    "p and c": "P&C",
    "p n c": "P&C",
    "pace": "Pace",
    "pace american": "Pace American",
    "ranch king": "RANCH KING",
    "rd trailers": "RD TRAILERS",
    "rd trailer": "RD TRAILERS",
    "stallion": "Stallion",
    "star": "Star",
    "texas pride": "Texas Pride",
    "w w": "W-W",
    "w-w": "W-W",
}

_GENERIC_TOKENS = {
    "a",
    "an",
    "and",
    "any",
    "available",
    "brand",
    "by",
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


def _valid_make_map() -> dict[str, str]:
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
        for phrase, make in make_map.items():
            phrase_tokens = phrase.split()
            if any(part.startswith(token) for part in phrase_tokens):
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


@lru_cache(maxsize=1)
def _llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(
        _LLMMakeResolution,
        method="function_calling",
    )


def _resolve_with_llm(text: str) -> MakeResolution:
    valid = known_makes()
    if not valid or not str(text or "").strip():
        return MakeResolution()
    try:
        result = _llm().invoke(
            [
                SystemMessage(
                    content=(
                        "Resolve the trailer make mentioned by the user. "
                        "Choose only from the provided make list. Return null if unclear. "
                        "Never choose Gooseneck unless the user clearly means the brand or manufacturer, "
                        "because gooseneck usually means hitch type."
                    )
                ),
                HumanMessage(
                    content=f"Known makes: {', '.join(valid)}\nUser message: {text}"
                ),
            ]
        )
    except Exception:
        return MakeResolution()

    make = result.make if result.make in valid else None
    if not make or result.confidence not in {"medium", "high"}:
        return MakeResolution()
    if not _candidate_allowed(make, text):
        return MakeResolution(None, "none", None, "gooseneck_without_brand_context")
    return MakeResolution(make, result.confidence, "llm", result.reason)


def resolve_make_from_text(text: str, *, use_llm_fallback: bool = True) -> MakeResolution:
    if use_llm_fallback:
        llm_resolution = _resolve_with_llm(text)
        if llm_resolution.make or llm_resolution.reason == "gooseneck_without_brand_context":
            return llm_resolution

    return _resolve_deterministic(text)
