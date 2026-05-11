"""Tests for heavy-duty width qualification and history-based requirement inference."""
from __future__ import annotations

import json
import os
import unittest

os.environ.setdefault("OPENAI_API_KEY", "sk-test-placeholder")
os.environ.setdefault("PINECONE_API_KEY", "pc-test-placeholder")

from langchain_core.messages import HumanMessage

from src.agent_lg import (
    WIDTH_REQUIREMENT_SLOT,
    TrailerAgentLG,
    _extract_width_ft_from_text,
    _enforce_single_required_question_message,
    _infer_dimension_requirements_from_history,
    _inject_width_requirement_slots,
)
from src.agent import TrailerAgent
from src.models import TrailerListing


class TestWidthExtraction(unittest.TestCase):
    def test_extract_width_ft_from_text(self) -> None:
        self.assertAlmostEqual(_extract_width_ft_from_text("I need 8.5 ft wide"), 8.5)
        self.assertAlmostEqual(_extract_width_ft_from_text("width is 102 inches"), 8.5)
        self.assertIsNone(_extract_width_ft_from_text("no width provided here"))


class TestHistoryInference(unittest.TestCase):
    def test_infer_requirements_from_history(self) -> None:
        state = {
            "messages": [
                HumanMessage(content="I need to haul about 7000 lbs"),
                HumanMessage(content="Length should be 20 ft and width 102 inches"),
            ]
        }
        payload, length, width = _infer_dimension_requirements_from_history(state)  # type: ignore[arg-type]
        self.assertEqual(payload, 7000.0)
        self.assertEqual(length, 20.0)
        self.assertAlmostEqual(width or 0.0, 8.5)


class TestWidthSlotInjection(unittest.TestCase):
    def test_inject_width_slot_when_missing(self) -> None:
        req, opt, injected = _inject_width_requirement_slots(
            ["haul_item", "haul_weight_lbs"],
            ["deck_style"],
            slots_collected={},
        )
        self.assertTrue(injected)
        self.assertIn(WIDTH_REQUIREMENT_SLOT, req)
        self.assertEqual(req, ["haul_item", WIDTH_REQUIREMENT_SLOT, "haul_weight_lbs"])
        self.assertNotIn(WIDTH_REQUIREMENT_SLOT, opt)

    def test_inject_width_slot_at_front_when_no_haul_item_slot(self) -> None:
        req, _opt, injected = _inject_width_requirement_slots(
            ["haul_weight_lbs", "haul_length_ft"],
            [],
            slots_collected={},
        )
        self.assertTrue(injected)
        self.assertEqual(req[0], WIDTH_REQUIREMENT_SLOT)

    def test_no_inject_when_already_collected(self) -> None:
        req, _opt, injected = _inject_width_requirement_slots(
            ["haul_item"],
            [],
            slots_collected={WIDTH_REQUIREMENT_SLOT: 8.5},
        )
        self.assertFalse(injected)
        self.assertNotIn(WIDTH_REQUIREMENT_SLOT, req)


class TestFetchTrailerFieldsWidthInjection(unittest.TestCase):
    def test_fetch_fields_injects_width_when_active(self) -> None:
        agent = TrailerAgentLG(customer=None)
        state = {
            "trailer_type": "Dump",
            "utility_lightweight_decided": None,
            "width_requirement_active": True,
            "slots_collected": {},
        }
        body, extra = agent._execute_specialist_tool(
            "fetch_trailer_fields",
            {"trailer_type": "Dump"},
            state,  # type: ignore[arg-type]
        )
        spec = json.loads(body)
        self.assertIn(WIDTH_REQUIREMENT_SLOT, spec["required_slots"])
        self.assertIn(WIDTH_REQUIREMENT_SLOT, spec["questions"])
        self.assertIn(WIDTH_REQUIREMENT_SLOT, extra["required_slots"])

    def test_fetch_fields_skips_width_for_enclosed(self) -> None:
        agent = TrailerAgentLG(customer=None)
        state = {
            "trailer_type": "Enclosed",
            "utility_lightweight_decided": None,
            "width_requirement_active": True,
            "slots_collected": {},
        }
        body, _ = agent._execute_specialist_tool(
            "fetch_trailer_fields",
            {"trailer_type": "Enclosed"},
            state,  # type: ignore[arg-type]
        )
        spec = json.loads(body)
        self.assertNotIn(WIDTH_REQUIREMENT_SLOT, spec["required_slots"])


class TestRequiredQuestionEnforcement(unittest.TestCase):
    def test_enforces_single_question_when_required_missing(self) -> None:
        from langchain_core.messages import AIMessage, HumanMessage

        msgs = [
            HumanMessage(content="a tractor"),
            AIMessage(content="Summary text here. Do you prefer bumper pull?"),
        ]
        changed = _enforce_single_required_question_message(
            msgs,
            trailer_type="Equipment",
            required_slots=["haul_item", "haul_weight_lbs", WIDTH_REQUIREMENT_SLOT],
            slots_collected={"haul_item": "tractor", "haul_weight_lbs": 3000},
        )
        self.assertTrue(changed)
        self.assertEqual(
            msgs[-1].content,
            "What is the approximate width of what you'll haul (or trailer width you need)?",
        )


class TestOptionalBlockedUntilRequired(unittest.TestCase):
    def test_optional_slot_rejected_when_required_missing(self) -> None:
        agent = TrailerAgentLG(customer=None)
        state = {
            "required_slots": ["haul_item", "haul_weight_lbs", WIDTH_REQUIREMENT_SLOT],
            "optional_slots": ["hitch_type", "loading_style"],
            "slots_collected": {"haul_item": "tractor", "haul_weight_lbs": 3000},
        }
        body, _ = agent._execute_specialist_tool(
            "record_slot_answer",
            {"slot": "hitch_type", "value": "Bumper Pull"},
            state,  # type: ignore[arg-type]
        )
        data = json.loads(body)
        self.assertFalse(data.get("ok"))
        self.assertEqual(data.get("error"), "optional_before_required")


class TestWidthRerank(unittest.TestCase):
    @staticmethod
    def _listing(listing_id: str, *, width: str, score: float) -> TrailerListing:
        return TrailerListing(
            listing_id=listing_id,
            title=f"Unit {listing_id}",
            condition="New",
            price=10000.0,
            price_display="$10,000",
            payments_from=None,
            category_subcategory="Equipment",
            make="Test",
            color="Black",
            hitch_type="Bumper Pull",
            year="2026",
            length="20 ft",
            width=width,
            axles="2",
            gvwr="14000 lbs",
            payload_capacity="9000 lbs",
            trailer_material="Steel",
            floor="Wood",
            url=f"https://example.com/{listing_id}",
            score=score,
        )

    def test_width_requirement_changes_rank_order(self) -> None:
        listings = [
            self._listing("wide", width="10 ft", score=0.95),
            self._listing("exact", width="8.5 ft", score=0.90),
        ]
        ranked, debug = TrailerAgent._rerank_by_fit(
            listings,
            required_payload_lbs=None,
            required_length_ft=None,
            required_gvwr_lbs=None,
            required_width_ft=8.5,
            desired_count=3,
        )
        self.assertTrue(debug.get("applied"))
        self.assertEqual(ranked[0].listing_id, "exact")


if __name__ == "__main__":
    unittest.main()

