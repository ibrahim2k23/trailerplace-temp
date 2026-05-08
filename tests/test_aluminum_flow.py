"""Aluminum category: base_category → subcategory, pivot guard, Pinecone filter."""
from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "sk-test-placeholder")
os.environ.setdefault("PINECONE_API_KEY", "pc-test-placeholder")

from src.agent import _build_pinecone_filter
from src.agent_lg import TrailerAgentLG
from src.models import TrailerFilter


class TestAluminumPivotGuard(unittest.TestCase):
    def test_blocks_utility_while_base_category_missing(self) -> None:
        agent = TrailerAgentLG(customer=None)
        state = {
            "trailer_type": "Aluminum",
            "slots_collected": {},
        }
        msg, extra = agent._run_set_trailer_type_tool("Utility", state)  # type: ignore[arg-type]
        self.assertEqual(extra, {})
        data = json.loads(msg)
        self.assertFalse(data.get("ok"))
        self.assertEqual(data.get("error"), "aluminum_base_category_slot_required")

    def test_allows_utility_after_base_category_recorded(self) -> None:
        agent = TrailerAgentLG(customer=None)
        state = {
            "trailer_type": "Aluminum",
            "slots_collected": {"base_category": "utility"},
        }
        msg, extra = agent._run_set_trailer_type_tool("Utility", state)  # type: ignore[arg-type]
        data = json.loads(msg)
        self.assertTrue(data.get("ok"))
        self.assertEqual(extra.get("trailer_type"), "Utility")

    def test_allows_dump_pivot_without_base_category(self) -> None:
        agent = TrailerAgentLG(customer=None)
        state = {
            "trailer_type": "Aluminum",
            "slots_collected": {},
        }
        msg, extra = agent._run_set_trailer_type_tool("Dump", state)  # type: ignore[arg-type]
        data = json.loads(msg)
        self.assertTrue(data.get("ok"))
        self.assertEqual(extra.get("trailer_type"), "Dump")


class TestAluminumSearchInjection(unittest.TestCase):
    @patch("src.agent_lg._run_search", return_value=[])
    def test_injects_subcategory_and_category(self, _mock_run) -> None:
        agent = TrailerAgentLG(customer=None)
        state = {
            "trailer_type": "Aluminum",
            "required_slots": ["base_category", "payload_need"],
            "optional_slots": [],
            "slots_collected": {"base_category": "utility", "payload_need": "8000 lbs"},
            "utility_lightweight_decided": None,
            "session_id": None,
            "client_shown_urls": [],
        }
        extra: dict = {}
        out, extra_out = agent._execute_search_tool(
            {"query": "aluminum utility 8000 lbs"},
            state,  # type: ignore[arg-type]
            extra,
        )
        self.assertIsInstance(out, str)
        la = extra_out.get("last_search_args") or {}
        self.assertEqual(la.get("category_subcategory"), "Aluminum")
        self.assertEqual(la.get("subcategory"), "Utility")
        self.assertEqual(la.get("required_payload_lbs"), 8000.0)
        self.assertTrue(_mock_run.called)
        args, kwargs = _mock_run.call_args
        self.assertEqual(args[0], "aluminum utility 8000 lbs")
        tf: TrailerFilter = args[1]
        self.assertEqual(tf.category_subcategory, "Aluminum")
        self.assertEqual(tf.subcategory, "Utility")


class TestBuildPineconeFilterSubcategory(unittest.TestCase):
    def test_aluminum_and_subcategory_normalized(self) -> None:
        pf = _build_pinecone_filter(
            TrailerFilter(category_subcategory="Aluminum", subcategory="utility")
        )
        self.assertIsNotNone(pf)
        assert pf is not None
        self.assertEqual(pf.get("category"), {"$eq": "Aluminum"})
        self.assertEqual(pf.get("subcategory"), {"$eq": "Utility"})


if __name__ == "__main__":
    unittest.main()
