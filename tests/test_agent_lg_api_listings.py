"""Tests for per-turn API listings vs persisted LangGraph search_results."""
from __future__ import annotations

import os
import unittest
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

# Required before importing modules that read them at import/init time.
os.environ.setdefault("OPENAI_API_KEY", "sk-test-placeholder")
os.environ.setdefault("PINECONE_API_KEY", "pc-test-placeholder")

from langgraph.graph import END

from src.agent_lg import TrailerAgentLG


class TestAgentLgApiListings(unittest.TestCase):
    def test_chat_resets_api_listings_before_invoke(self) -> None:
        captured: dict = {}

        def fake_invoke(state: dict):
            captured["api_at_invoke"] = list(state.get("api_listings_this_turn") or [])
            captured["rec_due_at_invoke"] = bool(state.get("recommendation_entry_due"))
            return {
                **state,
                "messages": list(state.get("messages", [])) + [AIMessage(content="ok")],
            }

        agent = TrailerAgentLG(customer=None)
        agent._state["api_listings_this_turn"] = [{"title": "stale"}]
        agent._state["recommendation_entry_due"] = True
        agent._graph = SimpleNamespace(invoke=fake_invoke)

        agent.chat("hello")

        self.assertEqual(captured["api_at_invoke"], [])
        self.assertFalse(captured["rec_due_at_invoke"])

    def test_chat_returns_api_listings_not_search_results(self) -> None:
        stale = [{"title": "Stale", "url": "https://example.com/a"}]

        def fake_invoke(state: dict):
            return {
                **state,
                "messages": list(state.get("messages", [])) + [AIMessage(content="reply")],
                "search_results": stale,
                "api_listings_this_turn": [],
            }

        agent = TrailerAgentLG(customer=None)
        agent._graph = SimpleNamespace(invoke=fake_invoke)
        reply, listings = agent.chat("thanks")

        self.assertEqual(reply, "reply")
        self.assertEqual(listings, [])
        self.assertEqual(agent._state.get("search_results"), stale)

    def test_chat_returns_non_empty_api_when_search_ran(self) -> None:
        row = {"title": "Dump 1", "url": "https://example.com/t", "rank": 1}

        def fake_invoke(state: dict):
            return {
                **state,
                "messages": list(state.get("messages", [])) + [AIMessage(content="here")],
                "search_results": [row],
                "api_listings_this_turn": [row],
            }

        agent = TrailerAgentLG(customer=None)
        agent._graph = SimpleNamespace(invoke=fake_invoke)
        _reply, listings = agent.chat("find a dump trailer")

        self.assertEqual(listings, [row])

    def test_last_human_message_appended_before_graph(self) -> None:
        seen: list[HumanMessage] = []

        def fake_invoke(state: dict):
            msgs = list(state.get("messages", []))
            self.assertTrue(msgs)
            self.assertIsInstance(msgs[-1], HumanMessage)
            seen.append(msgs[-1])
            return {**state, "messages": msgs + [AIMessage(content="x")]}

        agent = TrailerAgentLG(customer=None)
        agent._graph = SimpleNamespace(invoke=fake_invoke)
        agent.chat("user text")

        self.assertEqual(seen[0].content, "user text")

    def test_route_after_specialist_enters_recommendation_with_results_without_flag(
        self,
    ) -> None:
        """Follow-up interest turns: specialist may not re-search; still route to recommendation."""
        base = {
            "trailer_type": "Utility",
            "search_results": [{"title": "Old", "url": "https://example.com/a"}],
            "search_results_for_category": "Utility",
            "recommendation_entry_due": False,
        }
        self.assertEqual(TrailerAgentLG._route_after_specialist(base), "recommendation_node")

    def test_route_after_specialist_enters_recommendation_when_flag_true(self) -> None:
        base = {
            "trailer_type": "Utility",
            "search_results": [{"title": "New", "url": "https://example.com/b"}],
            "search_results_for_category": "Utility",
            "recommendation_entry_due": True,
        }
        self.assertEqual(TrailerAgentLG._route_after_specialist(base), "recommendation_node")

    def test_route_after_specialist_skips_when_category_mismatch(self) -> None:
        base = {
            "trailer_type": "Utility",
            "search_results": [{"title": "Dump", "url": "https://example.com/c"}],
            "search_results_for_category": "Dump",
            "recommendation_entry_due": True,
        }
        self.assertEqual(TrailerAgentLG._route_after_specialist(base), END)

    def test_route_after_specialist_skips_when_no_results(self) -> None:
        base = {
            "trailer_type": "Utility",
            "search_results": [],
            "search_results_for_category": "Utility",
            "recommendation_entry_due": True,
        }
        self.assertEqual(TrailerAgentLG._route_after_specialist(base), END)

    def test_route_after_specialist_no_trailer_type_to_master(self) -> None:
        base = {
            "trailer_type": None,
            "search_results": [{"title": "x"}],
            "search_results_for_category": None,
            "recommendation_entry_due": True,
        }
        self.assertEqual(TrailerAgentLG._route_after_specialist(base), "master_router_node")


if __name__ == "__main__":
    unittest.main()
