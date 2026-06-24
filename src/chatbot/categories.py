from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

CANONICAL_CATEGORIES = [
    "Aluminum",
    "Car Hauler",
    "Equipment",
    "Enclosed",
    "Utility",
    "Fiber",
    "Race Trailer",
    "Roll Off",
    "Diesel Tank",
    "Flatbed",
    "Dump",
    "Tilt",
    "Livestock",
]

_SYNONYMS: dict[str, list[str]] = {
    "Aluminum": ["aluminum", "lightweight", "won't rust", "wont rust", "will not rust"],
    "Car Hauler": ["car hauler", "toy hauler", "trailer without sides"],
    "Equipment": [
        "equipment",
        "lowboy",
        "low profile",
        "skid steer",
        "mini ex",
        "mini excavator",
        "mini excuvator",
        "tractor",
        "deckover",
        "deck over",
    ],
    "Enclosed": ["enclosed", "box trailer", "cargo", "v-nose", "v nose","command trailer"],
    "Utility": ["utility", "landscape", "lawnmower", "atv", "bike", "motorcycle", "landscaping", "lawn mower","land scaping"],
    "Fiber": ["fiber", "splicing trailer", "fiber optic "],
    "Race Trailer": ["race trailer", "enclosed car hauler"],
    "Roll Off": ["roll off", "roll-off", "dumpster"],
    "Diesel Tank": ["diesel tank", "fuel tank", "tank trailer"],
    "Flatbed": ["flatbed", "hotshot", "step deck", "dovetail", "platform", "flat bed", "dove tail"],
    "Dump": [
        "dump",
        "scissor lift",
        "hoist",
        "dump trailer", 
        "telescopic",
        "front lift",
    ],
    "Tilt": [
        "tilt",
        "full tilt",
        "gravity dampened tilt",
        "hydraulic dampened tilt",
    ],
    "Livestock": [
        "livestock",
        "live stock",
        "galyean",
        "star trailer",
        "calico trailer",
        "goats",
        "hogs",
        "cattle",
    ],
}

_OFFICE_TERMS = ("office trailer", "cooldown trailer", "cool down trailer")
_FIBER_CONTEXT_TERMS = ("fiber", "telecom", "splicing", "fiber optic")
_GENERAL_OFFICE_TERMS = ("general office", "office only", "jobsite office", "site office")


@dataclass(frozen=True)
class CategoryResolution:
    category: Optional[str]
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    clarification_key: Optional[str] = None


_CATEGORY_CLARIFICATION_RULES: dict[str, dict[str, object]] = {
    "office_trailer_use": {
        "trigger_terms": _OFFICE_TERMS,
        "question": "Will this be for fiber/telecom work specifically, or a more general office trailer?",
        "options": {
            "Fiber": _FIBER_CONTEXT_TERMS,
            "Enclosed": _GENERAL_OFFICE_TERMS,
        },
    },
}


def _matches_any_term(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _direct_category_from_text(text: str) -> Optional[str]:
    for category, terms in _SYNONYMS.items():
        for term in terms:
            if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text):
                return category
    return None


def resolve_categories_from_text(text: str) -> list[str]:
    low = (text or "").lower()
    return [
        category
        for category, terms in _SYNONYMS.items()
        if any(re.search(rf"(?<!\w){re.escape(term)}(?!\w)", low) for term in terms)
    ]


def _clarification_rule(key: str) -> dict[str, object] | None:
    return _CATEGORY_CLARIFICATION_RULES.get(str(key or "").strip())


def category_clarification_question(key: str | None) -> Optional[str]:
    rule = _clarification_rule(str(key or "").strip())
    if not rule:
        return None
    return str(rule.get("question") or "").strip() or None


def resolve_category_clarification_answer(text: str, clarification_key: str | None) -> CategoryResolution:
    low = (text or "").lower()
    rule = _clarification_rule(str(clarification_key or "").strip())
    if not rule:
        return CategoryResolution(_direct_category_from_text(low))

    options = dict(rule.get("options") or {})
    for category, terms in options.items():
        if _matches_any_term(low, tuple(terms)):
            return CategoryResolution(str(category), clarification_key=str(clarification_key or "").strip())

    direct_category = _direct_category_from_text(low)
    if direct_category:
        return CategoryResolution(direct_category, clarification_key=str(clarification_key or "").strip())

    return CategoryResolution(
        None,
        True,
        str(rule.get("question") or "").strip() or None,
        clarification_key=str(clarification_key or "").strip() or None,
    )


def category_prompt_block() -> str:
    lines = [
        "The following are trailer categories and their mapping terms.",
        "Gooseneck and Bumper Pull are strictly hitch types, never trailer categories. "
        "Do not infer, recommend, or return either one as a category.",
    ]
    for category in CANONICAL_CATEGORIES:
        if category in {"Gooseneck", "Bumper Pull"}:
            continue
        terms = ", ".join(_SYNONYMS.get(category, []))
        lines.append(f"- {category}: {terms}")
    for rule in _CATEGORY_CLARIFICATION_RULES.values():
        trigger_terms = ", ".join(rule.get("trigger_terms") or [])
        options = []
        for category, terms in dict(rule.get("options") or {}).items():
            options.append(f"{category} ({', '.join(terms)})")
        lines.append(
            f"- {trigger_terms}: ask a clarification question first. Resolve using: {', '.join(options)}."
        )
    return "\n".join(lines)


def resolve_category_from_text(text: str) -> CategoryResolution:
    low = (text or "").lower()
    for key, rule in _CATEGORY_CLARIFICATION_RULES.items():
        trigger_terms = tuple(rule.get("trigger_terms") or ())
        if _matches_any_term(low, trigger_terms):
            return resolve_category_clarification_answer(low, key)

    return CategoryResolution(_direct_category_from_text(low))
