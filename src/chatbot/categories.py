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
    "Enclosed": ["enclosed", "box trailer", "cargo", "v-nose", "v nose"],
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


def category_prompt_block() -> str:
    lines = ["Canonical trailer categories and disambiguation terms:"]
    for category in CANONICAL_CATEGORIES:
        terms = ", ".join(_SYNONYMS.get(category, []))
        lines.append(f"- {category}: {terms}")
    lines.append(
        "- office trailer / cooldown trailer: ask whether this is for fiber/telecom work. "
        "If yes choose Fiber; otherwise choose Enclosed."
    )
    return "\n".join(lines)


def resolve_category_from_text(text: str) -> CategoryResolution:
    low = (text or "").lower()
    if any(term in low for term in _OFFICE_TERMS):
        if any(term in low for term in _FIBER_CONTEXT_TERMS):
            return CategoryResolution("Fiber")
        if any(term in low for term in _GENERAL_OFFICE_TERMS):
            return CategoryResolution("Enclosed")
        return CategoryResolution(
            None,
            True,
            "Will this be for fiber/telecom work specifically, or a more general office trailer?",
        )

    for category, terms in _SYNONYMS.items():
        for term in terms:
            if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", low):
                return CategoryResolution(category)
    return CategoryResolution(None)
