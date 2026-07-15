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

# Synonym terms are split into two salience tiers so that an explicitly *named*
# trailer type always outranks a mere *cargo/brand* word that only implies a type.
# Without this split the resolver returned the first dict-order match, so a cargo
# word ("tractor") could beat a named type ("tilt") in "tilt trailer to haul a
# tractor" and put the customer on the wrong qualification track.
#
# _NAMING_TERMS: the category word itself or an unambiguous style name for it.
# _CARGO_TERMS: cargo items, brands, or attributes that only *suggest* the category.
_NAMING_TERMS: dict[str, list[str]] = {
    "Aluminum": ["aluminum"],
    "Car Hauler": ["car hauler", "toy hauler"],
    "Equipment": ["equipment", "lowboy", "low profile", "deckover", "deck over"],
    "Enclosed": ["enclosed", "box trailer", "v-nose", "v nose", "command trailer"],
    "Utility": ["utility"],
    "Fiber": ["fiber", "splicing trailer", "fiber optic "],
    "Race Trailer": ["race trailer", "enclosed car hauler"],
    "Roll Off": ["roll off", "roll-off", "dumpster","trailer with bins","bin trailer", "3 bins"],
    "Diesel Tank": ["diesel tank", "fuel tank", "tank trailer"],
    "Flatbed": ["flatbed", "hotshot", "step deck", "dovetail", "platform", "flat bed", "dove tail"],
    "Dump": ["dump", "dump trailer"],
    "Tilt": ["tilt", "full tilt", "gravity dampened tilt", "hydraulic dampened tilt"],
    "Livestock": ["livestock", "live stock"],
}

_CARGO_TERMS: dict[str, list[str]] = {
    "Aluminum": ["lightweight", "won't rust", "wont rust", "will not rust"],
    "Car Hauler": ["trailer without sides","car"],
    "Equipment": ["skid steer", "mini ex", "mini excavator", "mini excuvator", "tractor"],
    "Enclosed": ["cargo"],
    "Utility": ["landscape", "lawnmower", "atv", "bike", "motorcycle", "landscaping", "lawn mower", "land scaping"],
    "Dump": ["scissor lift", "hoist", "telescopic", "front lift"],
    "Livestock": ["galyean", "star trailer", "calico trailer", "goats", "hogs", "cattle"],
    "Roll Off": ["roll off", "roll-off", "dumpster","trailer with bins","bin trailer","3 bins"],
}

# Backward-compatible merged view (naming terms first) for callers that only need
# "every term for this category" — e.g. category_prompt_block().
_SYNONYMS: dict[str, list[str]] = {
    category: _NAMING_TERMS.get(category, []) + _CARGO_TERMS.get(category, [])
    for category in CANONICAL_CATEGORIES
    if _NAMING_TERMS.get(category) or _CARGO_TERMS.get(category)
}

# Categories that NEVER get the dynamically-injected width question, however big the cargo.
# Either the trailer's width is not a real choice (a livestock or dump body comes as it comes,
# and asking a customer how wide their cattle are is nonsense), or the category already asks for
# size its own way. What is left — Car Hauler, Equipment, Tilt — is where a wide load genuinely
# decides whether the trailer works.
#
# Single source of truth: the injection gate in apply_analysis and the needs_width_question rule
# in the Analyze prompt both read this, so the model is never asked to judge width for a category
# the code would refuse to ask about anyway.
WIDTH_EXCLUDED_CATEGORIES = frozenset({
    "Aluminum",
    "Diesel Tank",
    "Dump",
    "Enclosed",
    "Fiber",
    "Flatbed",
    "Livestock",
    "Race Trailer",
    "Roll Off",
    "Utility",
})


def width_excluded_categories_line() -> str:
    """Comma-joined width-exempt categories, for the Analyze prompt."""
    return ", ".join(sorted(WIDTH_EXCLUDED_CATEGORIES))


def width_eligible_categories_line() -> str:
    """Comma-joined categories that CAN take the injected width question."""
    return ", ".join(
        category for category in CANONICAL_CATEGORIES if category not in WIDTH_EXCLUDED_CATEGORIES
    )


_OFFICE_TERMS = ("office trailer", "cooldown trailer", "cool down trailer")
_FIBER_CONTEXT_TERMS = ("fiber", "telecom", "splicing", "fiber optic")
_GENERAL_OFFICE_TERMS = ("general office", "office only", "jobsite office", "site office")


@dataclass(frozen=True)
class CategoryResolution:
    category: Optional[str]
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    clarification_key: Optional[str] = None
    # "naming" when the resolved category came from an explicit type word,
    # "cargo" when it came only from a cargo/brand implication, None when
    # unresolved. Consumers (the mind node) use this to decide how much to
    # trust the deterministic hint versus the LLM's own category proposal.
    match_tier: Optional[str] = None


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


def _ranked_category_matches(text: str) -> list[tuple[str, str, int, int]]:
    """Return category matches ranked by salience.

    Each entry is (category, tier, sort_key_position, sort_key_neg_len). Results
    are ordered so the best match is first, using: naming terms before cargo
    terms; within a tier, the earliest position in the text; then the longest
    (most specific) matching term. Only the strongest match per category is kept.
    """
    low = text or ""
    best_per_category: dict[str, tuple[int, int, int]] = {}
    for tier_rank, term_map in ((0, _NAMING_TERMS), (1, _CARGO_TERMS)):
        for category, terms in term_map.items():
            for term in terms:
                match = re.search(rf"(?<!\w){re.escape(term.strip())}(?!\w)", low)
                if not match:
                    continue
                candidate = (tier_rank, match.start(), -len(term.strip()))
                existing = best_per_category.get(category)
                if existing is None or candidate < existing:
                    best_per_category[category] = candidate

    ranked = sorted(best_per_category.items(), key=lambda item: item[1])
    return [
        (category, "naming" if key[0] == 0 else "cargo", key[1], key[2])
        for category, key in ranked
    ]


def _direct_category_from_text(text: str) -> Optional[str]:
    matches = _ranked_category_matches((text or "").lower())
    return matches[0][0] if matches else None


def _direct_category_tier_from_text(text: str) -> Optional[str]:
    matches = _ranked_category_matches((text or "").lower())
    return matches[0][1] if matches else None


def resolve_categories_from_text(text: str) -> list[str]:
    return [category for category, _tier, _pos, _len in _ranked_category_matches((text or "").lower())]


def resolve_category_matches(text: str) -> list[tuple[str, str]]:
    """Ranked (category, tier) pairs for a message; tier is "naming" or "cargo".

    "naming" means the user said the trailer type itself ("tilt trailer"); "cargo"
    means they only named a load/job that implies the type ("haul a tractor").
    Callers need the tier to tell an explicit choice apart from an implied one.
    """
    return [(category, tier) for category, tier, _pos, _len in _ranked_category_matches((text or "").lower())]


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


def advertised_categories_line() -> str:
    """Comma-joined canonical categories for the 'what we carry' catalogue line.

    Single source of truth so the customer-facing catalogue can never drift from
    the categories the system actually supports (previously the KNOWLEDGE block
    advertised 10 types while 13 were qualifiable).
    """
    return ", ".join(CANONICAL_CATEGORIES)


def category_prompt_block() -> str:
    """Full category catalogue for the Analyze prompt, with the two term tiers split apart.

    TYPE TERMS name the trailer itself ("tilt trailer"); CARGO/TASK TERMS only *imply*
    the category ("haul a tractor" -> Equipment). Analyze must treat the two very
    differently, so they are labelled separately rather than merged into one list.
    """
    lines = [
        "These are the ONLY trailer categories we carry. There are exactly "
        f"{len(CANONICAL_CATEGORIES)}: {', '.join(CANONICAL_CATEGORIES)}.",
        "",
        "For each category below:",
        '  TYPE TERMS  = the user NAMED this trailer type ("I want a tilt trailer") -> an EXPLICIT choice.',
        '  CARGO TERMS = the user named a load or job this category is best suited for ("haul a tractor")',
        "                -> an IMPLIED choice only. It suggests the category; it does not name it.",
        "",
        "Gooseneck and Bumper Pull are strictly hitch types, never trailer categories. "
        "Do not infer, recommend, or return either one as a category.",
        "",
    ]
    for category in CANONICAL_CATEGORIES:
        naming = ", ".join(_NAMING_TERMS.get(category, [])) or "(none)"
        cargo = ", ".join(_CARGO_TERMS.get(category, [])) or "(none)"
        lines.append(f"- {category}")
        lines.append(f"    TYPE TERMS: {naming}")
        lines.append(f"    CARGO TERMS (this category is best suited to haul these): {cargo}")
    for rule in _CATEGORY_CLARIFICATION_RULES.values():
        trigger_terms = ", ".join(rule.get("trigger_terms") or [])
        options = []
        for category, terms in dict(rule.get("options") or {}).items():
            options.append(f"{category} ({', '.join(terms)})")
        lines.append(
            f"- AMBIGUOUS: {trigger_terms} -> ask a clarification question first. Resolve using: {', '.join(options)}."
        )
    return "\n".join(lines)


def category_reference_block() -> str:
    """Customer-facing catalogue for the Respond prompt: what each category is good for.

    Respond needs this to answer "which trailer is best for X?" accurately and to
    explain WHY a suggested switch makes sense - without seeing Analyze's tier logic.
    """
    lines = []
    for category in CANONICAL_CATEGORIES:
        cargo = ", ".join(_CARGO_TERMS.get(category, []))
        aliases = ", ".join(_NAMING_TERMS.get(category, []))
        detail = f" (also called: {aliases})" if aliases else ""
        suited = f" - best suited for: {cargo}" if cargo else ""
        lines.append(f"- {category}{detail}{suited}")
    return "\n".join(lines)


def make_prompt_block() -> str:
    from src.domain.brands import make_prompt_block as _make_prompt_block

    return _make_prompt_block()


def resolve_category_from_text(text: str) -> CategoryResolution:
    low = (text or "").lower()
    for key, rule in _CATEGORY_CLARIFICATION_RULES.items():
        trigger_terms = tuple(rule.get("trigger_terms") or ())
        if _matches_any_term(low, trigger_terms):
            return resolve_category_clarification_answer(low, key)

    return CategoryResolution(
        _direct_category_from_text(low),
        match_tier=_direct_category_tier_from_text(low),
    )
