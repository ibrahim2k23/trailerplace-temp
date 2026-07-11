from __future__ import annotations

from src.graph.apply_analysis import enforce_haul_classification_invariant
from src.llm.schemas import HaulClassification


def test_haul_invariant_keeps_valid_flags():
    light = HaulClassification(is_lightweight_utility_load=True, needs_width_question=False, haul_item_matched="golf cart")
    width = HaulClassification(is_lightweight_utility_load=False, needs_width_question=True, haul_item_matched="tractor")
    assert enforce_haul_classification_invariant(light).is_lightweight_utility_load is True
    assert enforce_haul_classification_invariant(width).needs_width_question is True


def test_haul_invariant_clears_flags_without_matched_item():
    light = HaulClassification(is_lightweight_utility_load=True, needs_width_question=False, haul_item_matched=None)
    width = HaulClassification(is_lightweight_utility_load=False, needs_width_question=True, haul_item_matched=None)
    assert enforce_haul_classification_invariant(light).is_lightweight_utility_load is False
    assert enforce_haul_classification_invariant(width).needs_width_question is False
