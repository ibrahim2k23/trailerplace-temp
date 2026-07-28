"""Generates the Phase 1-4 name/email category-pipeline scenario YAMLs.

One-off authoring tool, not part of the runtime. Re-run it after editing
CATEGORY_DATA below (e.g. because src/domain/trailer_fields.py's slots or
question order changed) to regenerate scripts/scenarios/category_pipeline/*.yaml.

Phase 1 (per category): contact -> category select -> answer every required/
    optional slot one at a time -> results.
Phase 2 (per category): contact -> one message with the category and every
    answer given up front -> results (no slot re-asked).
Phase 3 (per category): contact -> category select -> a mix of real answers,
    explicit "no preference" (stores null, not re-asked) and explicit declines
    ("skip that one" -> lands in skipped_slots) -> results.
Phase 4 (single conversation): switches category mid-Q&A more than once,
    mixes declines/no-preference/real answers, and checks results after each
    category before switching again.

Usage: python scripts/generate_category_pipeline_scenarios.py
"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "scripts" / "scenarios" / "category_pipeline"

CUSTOMER_NAME = "Ibrahim"
CUSTOMER_EMAIL = "ibrahim@esided.ai"
CONTACT_MESSAGE = f"Hi, I'm {CUSTOMER_NAME}, {CUSTOMER_EMAIL}."

NO_PREF_TEXT = "No preference there -- whatever you'd recommend is fine."
DECLINE_TEXT = "I'd rather skip that one, let's move on."

# Each category: trigger message that names the category, the ordered list of
# (slot, answer_text) covering required then optional slots (matching
# trailer_fields.py's per-category spec), and a single sentence that answers
# everything up front for Phase 2.
CATEGORY_DATA: dict[str, dict] = {
    "Equipment": {
        "trigger": "I'm looking for an equipment trailer.",
        "fields": [
            ("haul_item", "A skid steer loader."),
            ("haul_weight_lbs", "About 8,000 lbs."),
            ("haul_length_ft", "It's around 16 feet long."),
            ("hitch_type", "Gooseneck, please."),
            ("loading_style", "I'll load it with ramps."),
        ],
        "phase2": (
            "I need an equipment trailer to haul a skid steer loader, about 8,000 lbs, "
            "roughly 16 feet long, gooseneck hitch, and I'll load it with ramps."
        ),
    },
    "Car Hauler": {
        "trigger": "I need a car hauler trailer.",
        "fields": [
            ("vehicle_type", "A Ford Mustang GT."),
            ("haul_weight_lbs", "Roughly 3,800 lbs."),
            ("vehicle_length_ft", "About 16 feet."),
            ("open_vs_covered", "I'd like it enclosed and covered."),
        ],
        "phase2": (
            "I'm looking for a car hauler for a Ford Mustang GT, about 3,800 lbs, "
            "around 16 feet long, and I'd like it enclosed and covered."
        ),
    },
    "Utility": {
        "trigger": "I want a utility trailer.",
        "fields": [
            ("haul_item", "Lawn equipment and a riding mower."),
            ("haul_weight_lbs", "Around 1,200 lbs total."),
            ("trailer_size", "Something like 6x12."),
        ],
        "phase2": (
            "I need a utility trailer for lawn equipment and a riding mower, "
            "around 1,200 lbs total, and I'd like something like a 6x12."
        ),
    },
    "Dump": {
        "trigger": "Looking for a dump trailer.",
        "fields": [
            ("haul_material", "Mostly gravel and dirt."),
            ("haul_weight_lbs", "About 6,000 lbs per load."),
            ("dump_mechanism", "A scissor lift mechanism if possible."),
        ],
        "phase2": (
            "I need a dump trailer for hauling gravel and dirt, about 6,000 lbs per load, "
            "with a scissor lift mechanism if possible."
        ),
    },
    "Tilt": {
        "trigger": "I'm interested in a tilt trailer.",
        "fields": [
            ("haul_item", "A farm tractor."),
            ("haul_weight_lbs", "Around 9,000 lbs."),
            ("tilt_style", "A full-tilt deck would be great."),
        ],
        "phase2": (
            "I'm looking for a tilt trailer to haul a farm tractor, around 9,000 lbs, "
            "with a full-tilt deck."
        ),
    },
    "Enclosed": {
        "trigger": "I need an enclosed trailer.",
        "fields": [
            ("use_case", "It'll be a mobile workshop."),
            ("cargo_size", "I need to fit about 8 by 16 by 7 feet of gear."),
            ("ac_windows_cabinets", "Yes, I'd like AC and some cabinets."),
            ("finished_interior", "Lined walls would be nice, and finished flooring too."),
        ],
        "phase2": (
            "I need an enclosed trailer for a mobile workshop, need to fit about 8 by 16 by 7 "
            "feet of gear, with AC, cabinets, and a finished interior with lined walls."
        ),
    },
    "Livestock": {
        "trigger": "I'm looking for a livestock trailer.",
        "fields": [
            ("trailer_length_ft", "About 20 feet."),
            ("gate_preferences", "Butterfly gates if you have them."),
        ],
        "phase2": "I'm looking for a livestock trailer, about 20 feet long, with butterfly gates.",
    },
    "Roll Off": {
        "trigger": "I need a roll off trailer.",
        "fields": [
            ("package_scope", "The full trailer-and-bins package."),
            ("bin_size", "20 yard bins."),
            ("deck_style", "A step deck, please."),
            ("cdl_concern", "Yes, staying under CDL weight matters to me."),
        ],
        "phase2": (
            "I need a roll off trailer, the full trailer-and-bins package, with 20 yard bins, "
            "a step deck, and staying under CDL weight matters to me."
        ),
    },
    "Diesel Tank": {
        "trigger": "I'm shopping for a diesel tank trailer.",
        "fields": [
            ("fuel_type", "Diesel fuel."),
            ("tank_capacity", "About 1,000 gallons."),
            ("deck_style", "Standard deck is fine."),
            ("cdl_concern", "Not really a concern for me."),
        ],
        "phase2": (
            "I'm shopping for a diesel tank trailer, for diesel fuel, about 1,000 gallons, "
            "standard deck, and CDL isn't really a concern for me."
        ),
    },
    "Flatbed": {
        "trigger": "Looking for a flatbed trailer.",
        "fields": [
            ("haul_item", "Steel beams and construction materials."),
            ("haul_weight_lbs", "About 12,000 lbs."),
            ("deck_style", "A step deck would help."),
            ("cdl_concern", "Yes, I want to stay under CDL limits."),
        ],
        "phase2": (
            "Looking for a flatbed trailer to haul steel beams and construction materials, "
            "about 12,000 lbs, step deck, and I want to stay under CDL limits."
        ),
    },
    "Fiber": {
        "trigger": "I need a fiber trailer.",
        "fields": [
            ("fiber_use_case", "It'll be used for splicing work."),
            ("crew_size", "About 4 crew members at once."),
            ("fiber_amenities", "AC and a workbench would be great."),
        ],
        "phase2": (
            "I need a fiber trailer for splicing work, about 4 crew members at once, "
            "with AC and a workbench."
        ),
    },
    "Race Trailer": {
        "trigger": "I want a race trailer.",
        "fields": [
            ("vehicle_type", "A dirt late model race car."),
            ("trailer_length_ft", "24 feet."),
            ("race_amenities", "Cabinets and a small workspace."),
        ],
        "phase2": (
            "I want a race trailer for a dirt late model race car, about 24 feet long, "
            "with cabinets and a small workspace."
        ),
    },
    "Aluminum": {
        "trigger": "I'm interested in an aluminum trailer.",
        "fields": [
            ("base_category", "A utility-style trailer."),
            ("payload_need", "About 2,000 lbs."),
            ("sleeping_need", "No, I won't need sleeping space."),
        ],
        "phase2": (
            "I'm interested in an aluminum trailer, utility-style, about 2,000 lbs payload, "
            "and I won't need sleeping space."
        ),
    },
}


def _slug(category: str) -> str:
    return category.lower().replace(" ", "-")


def _contact_turn(phase: int, category: str) -> dict:
    return {
        "user": CONTACT_MESSAGE,
        "label": f"Phase {phase} | {category} | contact",
        "expect_state": {"customer_name": CUSTOMER_NAME, "customer_email": CUSTOMER_EMAIL},
    }


def build_phase1(category: str, data: dict) -> dict:
    fields = data["fields"]
    turns = [_contact_turn(1, category)]
    turns.append(
        {
            "user": data["trigger"],
            "label": f"Phase 1 | {category} | category-select",
            "expect_state": {"category": category},
        }
    )
    total = len(fields)
    for idx, (slot, answer) in enumerate(fields, start=1):
        turn = {
            "user": answer,
            "label": f"Phase 1 | {category} | Q{idx}/{total} {slot}",
        }
        if idx == total:
            turn["expect_state"] = {"category": category, "qualification_complete": True}
            turn["expect_listings"] = True
        turns.append(turn)
    return {
        "name": f"{_slug(category)}-phase1-full-qna",
        "tags": ["category_pipeline"],
        "phase": 1,
        "category": category,
        "turns": turns,
    }


def build_phase2(category: str, data: dict) -> dict:
    turns = [
        _contact_turn(2, category),
        {
            "user": data["phase2"],
            "label": f"Phase 2 | {category} | all-answers-upfront",
            "expect_state": {"category": category, "qualification_complete": True},
            "expect_listings": True,
        },
    ]
    return {
        "name": f"{_slug(category)}-phase2-prefilled",
        "tags": ["category_pipeline"],
        "phase": 2,
        "category": category,
        "turns": turns,
    }


def build_phase3(category: str, data: dict) -> dict:
    fields = data["fields"]
    turns = [_contact_turn(3, category)]
    turns.append(
        {
            "user": data["trigger"],
            "label": f"Phase 3 | {category} | category-select",
            "expect_state": {"category": category},
        }
    )
    total = len(fields)
    modes = ["answer", "no_preference", "decline"]
    for idx, (slot, answer) in enumerate(fields):
        mode = modes[idx % 3]
        if mode == "answer":
            user_text = answer
        elif mode == "no_preference":
            user_text = NO_PREF_TEXT
        else:
            user_text = DECLINE_TEXT
        turn = {
            "user": user_text,
            "label": f"Phase 3 | {category} | Q{idx + 1}/{total} {slot} ({mode})",
        }
        if mode == "no_preference":
            turn["expect_state"] = {f"slot:{slot}": None}
        elif mode == "decline":
            turn["expect_state"] = {"skipped_slots_contains": slot}
        if idx == total - 1:
            turn.setdefault("expect_state", {})
            turn["expect_state"]["qualification_complete"] = True
            turn["expect_listings"] = True
        turns.append(turn)
    return {
        "name": f"{_slug(category)}-phase3-declines-and-no-preference",
        "tags": ["category_pipeline"],
        "phase": 3,
        "category": category,
        "turns": turns,
    }


def build_phase4() -> dict:
    turns = [
        {
            "user": CONTACT_MESSAGE,
            "label": "Phase 4 | multi-category | contact",
            "expect_state": {"customer_name": CUSTOMER_NAME, "customer_email": CUSTOMER_EMAIL},
        },
        {
            "user": "I need a dump trailer for hauling gravel, about 3 tons.",
            "label": "Phase 4 | Dump | category-select-and-answers",
            "expect_state": {"category": "Dump"},
        },
        {
            "user": NO_PREF_TEXT,
            "label": "Phase 4 | Dump | Q3/3 dump_mechanism (no_preference) -> results",
            "expect_state": {"category": "Dump", "qualification_complete": True},
            "expect_listings": True,
        },
        {
            "user": "Actually, can you switch me to an equipment trailer instead?",
            "label": "Phase 4 | Dump->Equipment | category-switch-offer",
            "expect_state": {"category": "Equipment"},
            "expect_reply_contains_any": ["keep", "carry over", "still apply", "start", "maintain", "adjust"],
        },
        {
            "user": "Just start fresh, don't keep anything.",
            "label": "Phase 4 | Equipment | category-switch-resolved",
            "expect_state": {"category": "Equipment", "pending_category_change": None},
        },
        {
            "user": "I'll be hauling a mini excavator, around 7,500 lbs.",
            "label": "Phase 4 | Equipment | Q1-2/5 haul_item+haul_weight_lbs (answer)",
            "expect_state": {"category": "Equipment"},
        },
        {
            "user": DECLINE_TEXT,
            "label": "Phase 4 | Equipment | Q3/5 haul_length_ft (decline)",
            "expect_state": {"skipped_slots_contains": "haul_length_ft"},
        },
        {
            "user": "Gooseneck hitch, and no preference on loading style.",
            "label": "Phase 4 | Equipment | Q4-5/5 hitch_type+loading_style -> results",
            "expect_state": {"category": "Equipment", "qualification_complete": True},
            "expect_listings": True,
        },
        {
            "user": "Now show me utility trailers instead.",
            "label": "Phase 4 | Equipment->Utility | category-switch-offer",
            "expect_state": {"category": "Utility"},
            "expect_reply_contains_any": ["keep", "carry over", "still apply", "start", "maintain", "adjust"],
        },
        {
            "user": "None of them, fresh start.",
            "label": "Phase 4 | Utility | category-switch-resolved",
            "expect_state": {"category": "Utility", "pending_category_change": None},
        },
        {
            "user": (
                "I need it for hauling landscaping tools, about 900 lbs, and no preference "
                "on trailer size."
            ),
            "label": "Phase 4 | Utility | Q1-2/2 haul_item+haul_weight_lbs, trailer_size (no_preference) -> results",
            "expect_state": {"category": "Utility", "qualification_complete": True},
            "expect_listings": True,
        },
    ]
    return {
        "name": "phase4-multi-category-switch-mixed",
        "tags": ["category_pipeline"],
        "phase": 4,
        "category": "Dump+Equipment+Utility",
        "turns": turns,
    }


def _write(scenario: dict) -> None:
    path = OUT_DIR / f"{scenario['name']}.yaml"
    header = (
        f"# Auto-generated by scripts/generate_category_pipeline_scenarios.py -- "
        f"do not hand-edit, regenerate instead.\n"
    )
    body = yaml.safe_dump(scenario, sort_keys=False, allow_unicode=True, width=100)
    path.write_text(header + body, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for category, data in CATEGORY_DATA.items():
        _write(build_phase1(category, data))
        _write(build_phase2(category, data))
        _write(build_phase3(category, data))
    _write(build_phase4())
    total = len(CATEGORY_DATA) * 3 + 1
    print(f"Wrote {total} scenario files to {OUT_DIR}")


if __name__ == "__main__":
    main()
