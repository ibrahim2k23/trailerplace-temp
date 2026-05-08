"""
trailer_fields.py – Tool that returns the required and optional qualification
slots for a given trailer type.  Called by the specialist node once
`trailer_type` is set in the session state.

Slot names map 1-to-1 to the keys used in SessionState["slots_collected"].
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Slot schema
# ---------------------------------------------------------------------------

@dataclass
class TrailerFieldSpec:
    """Required and optional slots for one trailer category."""
    category: str
    required: list[str]
    optional: list[str]
    # Human-readable question prompts keyed by slot name (used in dynamic prompt)
    questions: dict[str, str] = field(default_factory=dict)
    # Brief notes injected into the specialist prompt
    notes: str = ""


# ---------------------------------------------------------------------------
# Per-category field definitions
# ---------------------------------------------------------------------------

_SPECS: dict[str, TrailerFieldSpec] = {

    "Equipment": TrailerFieldSpec(
        category="Equipment",
        required=["haul_item", "haul_weight_lbs", "haul_length_ft"],
        optional=["hitch_type", "loading_style"],
        questions={
            "haul_item":       "What equipment will you be hauling (e.g. skid steer, mini excavator, tractor)?",
            "haul_weight_lbs": "What's the rough total weight of the equipment?",
            "haul_length_ft":  "About how long is the equipment (or what deck length do you need)?",
            "hitch_type":      "Do you prefer a bumper pull or gooseneck hitch?",
            "loading_style":   "How will you load it — ramps, deckover, or drive-over fenders?",
        },
        notes="",
    ),

    "Car Hauler": TrailerFieldSpec(
        category="Car Hauler",
        required=["vehicle_type", "haul_weight_lbs", "vehicle_length_ft"],
        optional=["open_vs_covered"],
        questions={
            "vehicle_type":      "What type of vehicle will you be hauling (make/model or class)?",
            "haul_weight_lbs":   "What's the approximate weight of the vehicle?",
            "vehicle_length_ft": "About how long is the vehicle?",
            "open_vs_covered":   "Are you looking for an open car hauler or a covered/enclosed option?",
        },
    ),

    "Utility": TrailerFieldSpec(
        category="Utility",
        required=["haul_item", "haul_weight_lbs"],
        optional=["trailer_size", "sides_gate_storage"],
        questions={
            "haul_item":           "What will you be hauling on the utility trailer?",
            "haul_weight_lbs":     "What's the rough total weight of your load?",
            "trailer_size":        "Do you have a size preference (length / width)?",
            #"sides_gate_storage":  "Will you need side rails, a rear gate, or tool storage?",
        },
        notes="Utility-only: lightweight haul handling is decided by the assistant agent (not used for other categories).",
    ),

    "Dump": TrailerFieldSpec(
        category="Dump",
        required=["haul_material", "haul_weight_lbs"],
        optional=["dump_mechanism"],
        questions={
            "haul_material":   "What material will you be hauling (dirt, gravel, debris, etc.)?",
            "haul_weight_lbs": "What's the rough haul weight per load?",
            "dump_mechanism":  "Do you have a preference for the dump mechanism — scissor lift, telescopic, or standard?",
        },
    ),

    "Tilt": TrailerFieldSpec(
        category="Tilt",
        required=["haul_item", "haul_weight_lbs"],
        optional=["tilt_style"],
        questions={
            "haul_item":     "What will you be hauling on the tilt trailer?",
            "haul_weight_lbs": "What's the approximate weight of the load?",
            "tilt_style":    "Would you prefer a full-tilt deck or one with a stationary front section?",
        },
    ),

    "Enclosed": TrailerFieldSpec(
        category="Enclosed",
        required=["use_case", "cargo_size"],
        optional=["ac_windows_cabinets", "finished_interior"],
        questions={
            "use_case":              "What will you be using the enclosed trailer for (cargo hauling, mobile workshop, etc.)?",
            "cargo_size":            "What's the rough size of the cargo you need to fit (length × width × height)?",
            "ac_windows_cabinets":   "Will you need AC, windows, or cabinets inside?",
            "finished_interior":     "Does the interior need to be finished (e.g. lined walls, flooring)?",
        },
    ),

    "Livestock": TrailerFieldSpec(
        category="Livestock",
        required=["trailer_length_ft"],
        optional=["gate_preferences"],
        questions={
            "trailer_length_ft": "What length trailer are you looking for?",
            "gate_preferences":  "Any preference on gate style — butterfly, swing, or slant load?",
        },
        notes="Do NOT ask about animal type or count. Only ask about trailer length.",
    ),

    "Roll Off": TrailerFieldSpec(
        category="Roll Off",
        required=["package_scope", "bin_size"],
        optional=["deck_style", "cdl_concern"],
        questions={
            "package_scope": "Are you looking for just the trailer, just bins, or the trailer-and-bins package?",
            "bin_size":      "What size bins are you needing (e.g. 10 yd, 20 yd)?",
            "deck_style":    "Do you prefer a step-deck or standard deck?",
            "cdl_concern":   "Is staying under CDL weight thresholds a concern for you?",
        },
    ),

    "Diesel Tank": TrailerFieldSpec(
        category="Diesel Tank",
        required=["fuel_type", "tank_capacity"],
        optional=["deck_style", "cdl_concern"],
        questions={
            "fuel_type":     "Will you be transporting diesel, gasoline, or another fuel?",
            "tank_capacity": "What tank capacity are you looking for (in gallons)?",
            "deck_style":    "Do you prefer a step-deck or standard deck?",
            "cdl_concern":   "Is staying under CDL weight thresholds a concern for you?",
        },
    ),

    "Flatbed": TrailerFieldSpec(
        category="Flatbed",
        required=["haul_item", "haul_weight_lbs"],
        optional=["deck_style", "cdl_concern"],
        questions={
            "haul_item":     "What will you be hauling on the flatbed?",
            "haul_weight_lbs": "What's the approximate load weight?",
            "deck_style":    "Do you prefer a step deck or standard deck?",
            "cdl_concern":   "Is staying under CDL weight thresholds a concern for you?",
        },
    ),

    "Fiber": TrailerFieldSpec(
        category="Fiber",
        required=["fiber_use_case"],
        optional=["crew_size", "fiber_amenities"],
        questions={
            "fiber_use_case":  "Will this be used for splicing, as an office trailer, or as a cooldown trailer?",
            "crew_size":       "How many crew members need to use it at once?",
            "fiber_amenities": "What amenities do you need — AC, workbench, generator hookup?",
        },
    ),

    "Race Trailer": TrailerFieldSpec(
        category="Race Trailer",
        required=["vehicle_type", "trailer_length_ft"],
        optional=["race_amenities"],
        questions={
            "vehicle_type":      "What type of race vehicle will you be hauling?",
            "trailer_length_ft": "What length trailer are you looking for?",
            "race_amenities":    "Will you need cabinets, a workspace, or living quarters?",
        },
    ),

    "Welding": TrailerFieldSpec(
        category="Welding",
        required=["equipment_list"],
        optional=["total_weight"],
        questions={
            "equipment_list": "What welding equipment will you be mounting or carrying (welder, generator, gas bottles, etc.)?",
            "total_weight":   "Do you have a rough estimate of the total equipment weight?",
        },
    ),

    "Aluminum": TrailerFieldSpec(
        category="Aluminum",
        required=["base_category", "payload_need"],
        optional=["sleeping_need"],
        questions={
            "base_category": (
                "What type of trailer are you looking for in aluminum — utility, equipment, enclosed, "
                "or something else?"
            ),
            "payload_need": "What's the rough total weight of the load?",
            "sleeping_need": "Will you need sleeping accommodations in the trailer?",
        },
        notes="Aluminum is a modifier, not a standalone category. Resolve base_category first.",
    ),
}

# Fallback for unknown categories
_DEFAULT_SPEC = TrailerFieldSpec(
    category="Unknown",
    required=["haul_item"],
    optional=["haul_weight_lbs", "hitch_type"],
    questions={
        "haul_item":       "What will you be hauling or using this trailer for?",
        "haul_weight_lbs": "What's the rough weight of the load?",
        "hitch_type":      "Do you prefer a bumper pull or gooseneck hitch?",
    },
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_trailer_fields(trailer_type: str) -> TrailerFieldSpec:
    """
    Return the TrailerFieldSpec for the given trailer_type.
    Falls back to a generic spec if the type is not recognised.

    Usage:
        spec = get_trailer_fields("Dump")
        print(spec.required)   # ['haul_material', 'haul_weight_lbs']
        print(spec.optional)   # ['dump_mechanism']
    """
    return _SPECS.get(trailer_type, _DEFAULT_SPEC)


def get_trailer_fields_as_dict(trailer_type: str) -> dict:
    """
    Return a JSON-serialisable dict representation of the field spec.
    Suitable for injecting into LLM prompts or tool results.
    """
    spec = get_trailer_fields(trailer_type)
    return {
        "category": spec.category,
        "required_slots": spec.required,
        "optional_slots": spec.optional,
        "questions": spec.questions,
        "notes": spec.notes,
    }


def list_all_categories() -> list[str]:
    """Return all supported trailer categories."""
    return sorted(_SPECS.keys())
