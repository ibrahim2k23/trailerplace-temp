from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.llm.schemas import (
    ContactInfo,
    ExtractedFields,
    HaulClassification,
    InventoryLookup,
    ReplyOutput,
    SlotAnswer,
    TurnAnalysis,
)
from tests.unit.llm_helpers import sample_analysis


def test_turn_analysis_round_trips_json():
    analysis = sample_analysis()
    restored = TurnAnalysis.model_validate_json(analysis.model_dump_json())
    assert restored == analysis
    assert isinstance(restored.slot_answers[0], SlotAnswer)
    assert isinstance(restored.haul_classification, HaulClassification)
    assert isinstance(restored.inventory_lookup, InventoryLookup)


def test_invalid_intent_rejected():
    with pytest.raises(ValidationError):
        sample_analysis(intent="not_real")


def test_nested_models_required():
    with pytest.raises(ValidationError):
        TurnAnalysis.model_validate({"intent": "category_selection"})
    ContactInfo(name=None, email=None, phone=None)
    ExtractedFields(
        trailer_length_ft=None,
        trailer_width_ft=None,
        trailer_height_ft=None,
        payload_lbs=None,
        hitch_type=None,
        haul_item=None,
        brand_preference=None,
        non_metadata_features=[],
        numeric_no_preference=[],
    )


def _assert_strict_objects(schema: dict) -> None:
    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False
            assert "additionalProperties" not in node or node.get("additionalProperties") is False
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(schema)


def test_openai_strict_schema_has_no_open_dicts():
    _assert_strict_objects(TurnAnalysis.model_json_schema())
    _assert_strict_objects(ReplyOutput.model_json_schema())
