from __future__ import annotations

from src.llm.schemas import ReplyOutput, TurnAnalysis


def sample_analysis(**updates) -> TurnAnalysis:
    data = {
        "turn_summary": "The customer wants a dump trailer, about 7x14, to haul dirt.",
        "intent": "category_selection",
        "email_triggers": [],
        "haul_classification": {"is_lightweight_utility_load": False, "needs_width_question": False, "haul_item_matched": None},
        "inventory_lookup": {"is_lookup": False, "year": None, "make": None, "model_text": None, "stock_number": None, "wants": None, "confidence": "low"},
        "category_mentioned": "Dump",
        "is_category_info_only": False,
        "extracted": {
            "trailer_length_ft": 14.0,
            "trailer_width_ft": 7.0,
            "trailer_height_ft": None,
            "payload_lbs": None,
            "axle_capacity_lbs": None,
            "total_axle_capacity_lbs": None,
            "axle_count": None,
            "axle_capacity_basis": None,
            "hitch_type": None,
            "haul_item": "dirt",
            "brand_preference": None,
            "non_metadata_features": [],
            "numeric_no_preference": [],
        },
        "slot_answers": [{"slot_name": "trailer_size", "raw_answer": "7x14"}],
        "contact": {"name": "John", "email": None, "phone": "555-1234"},
        "listing_reference": None,
        "dropped_fields": [],
        "keep_fields_answer": None,
        "kept_fields": [],
        "category_confirm_answer": None,
        "answered_current_question": True,
        "user_question_to_answer": None,
    }
    # `extracted` MERGES rather than replaces. The schema is strict - every field is
    # required, since structured outputs must emit them all - so a test that overrides one
    # extracted field would otherwise have to restate the other dozen, and every new field
    # would break every such test rather than just the ones that care about it.
    extracted_updates = updates.pop("extracted", None)
    data.update(updates)
    if extracted_updates is not None:
        data["extracted"] = {**data["extracted"], **extracted_updates}
    return TurnAnalysis.model_validate(data)


def sample_reply(text: str = "Sure, I can help.", urls: list[str] | None = None) -> ReplyOutput:
    return ReplyOutput(assistant_text=text, cited_listing_urls=urls or [])
