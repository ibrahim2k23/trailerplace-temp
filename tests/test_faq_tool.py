"""Tests for faq_tool (master + recommendation nodes)."""
from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "sk-test-placeholder")
os.environ.setdefault("PINECONE_API_KEY", "pc-test-placeholder")

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src import agent_lg
from src.agent_lg import (
    FAQ_CANONICAL_REPLY_TEXT,
    TrailerAgentLG,
    _ensure_faq_assistant_reply,
    _patch_final_assistant_with_faq_reply_text,
)
from src.models import CustomerContact


class TestExecuteFaqTool(unittest.TestCase):
    def setUp(self) -> None:
        self.cust = CustomerContact(
            full_name="Test User",
            email="t@example.com",
            phone="9795551212",
        )

    def _good_args(self) -> dict:
        return {
            "faq_type": "contact_or_human",
            "title": "Contact inquiry",
            "user_summary": "Asked how to reach a human.",
            "reply_text": FAQ_CANONICAL_REPLY_TEXT["contact_or_human"],
        }

    def test_rejects_invalid_faq_type(self) -> None:
        agent = TrailerAgentLG(customer=self.cust)
        args = dict(self._good_args())
        args["faq_type"] = "not_a_type"
        with patch.object(agent_lg, "enqueue_faq_email_notification") as q:
            body, extra = agent._execute_faq_tool(args, {})
        self.assertEqual(extra, {})
        self.assertFalse(json.loads(body).get("ok"))
        q.assert_not_called()

    def test_rejects_empty_title(self) -> None:
        agent = TrailerAgentLG(customer=self.cust)
        args = dict(self._good_args())
        args["title"] = "   "
        with patch.object(agent_lg, "enqueue_faq_email_notification") as q:
            body, _ = agent._execute_faq_tool(args, {})
        self.assertFalse(json.loads(body).get("ok"))
        q.assert_not_called()

    def test_non_canonical_reply_text_still_enqueues(self) -> None:
        agent = TrailerAgentLG(customer=self.cust)
        args = dict(self._good_args())
        args["reply_text"] = "Call us maybe."
        with patch.object(agent_lg, "enqueue_faq_email_notification") as q:
            body, _ = agent._execute_faq_tool(args, {})
        self.assertTrue(json.loads(body).get("ok"))
        q.assert_called_once()
        data = json.loads(body)
        self.assertEqual(
            data.get("effective_reply_text"),
            FAQ_CANONICAL_REPLY_TEXT["contact_or_human"],
        )

    def test_success_enqueues_email(self) -> None:
        agent = TrailerAgentLG(customer=self.cust)
        state = {"customer_full_name": "X", "customer_email": None, "customer_phone": "1"}
        with patch.object(agent_lg, "enqueue_faq_email_notification") as q:
            body, extra = agent._execute_faq_tool(self._good_args(), state)
        self.assertEqual(extra, {})
        self.assertTrue(json.loads(body).get("ok"))
        q.assert_called_once()
        kwargs = q.call_args.kwargs
        self.assertIn("contact", kwargs["subject"].lower())
        self.assertIn("contact_or_human", kwargs["summary_line"])

    def test_summary_line_uses_deterministic_text_not_raw_question(self) -> None:
        agent = TrailerAgentLG(customer=self.cust)
        args = dict(self._good_args())
        args["faq_type"] = "service_parts"
        args["user_summary"] = "Do you guys have spare parts?"
        args["reply_text"] = FAQ_CANONICAL_REPLY_TEXT["service_parts"]
        with patch.object(agent_lg, "enqueue_faq_email_notification") as q:
            body, _ = agent._execute_faq_tool(args, {})
        self.assertTrue(json.loads(body).get("ok"))
        sent = q.call_args.kwargs["summary_line"]
        self.assertIn("[service_parts]", sent)
        self.assertNotIn("Do you guys have spare parts?", sent)


class TestEnsureFaqAssistantReply(unittest.TestCase):
    def test_appends_when_no_plain_assistant_message(self) -> None:
        text = FAQ_CANONICAL_REPLY_TEXT["contact_or_human"]
        msgs: list = [
            HumanMessage(content="How do I reach you?"),
            AIMessage(content="", tool_calls=[{"name": "faq_tool", "id": "1", "args": {}}]),
            ToolMessage(
                content=json.dumps({"ok": True, "effective_reply_text": text}),
                tool_call_id="1",
            ),
        ]
        _ensure_faq_assistant_reply(msgs, text)
        self.assertIsInstance(msgs[-1], AIMessage)
        self.assertEqual(msgs[-1].content, text)
        self.assertFalse(getattr(msgs[-1], "tool_calls", None) or [])

    def test_replaces_final_assistant_text(self) -> None:
        script = FAQ_CANONICAL_REPLY_TEXT["financing"]
        msgs: list = [
            HumanMessage(content="Do you finance?"),
            AIMessage(content="", tool_calls=[{"name": "faq_tool", "id": "1", "args": {}}]),
            ToolMessage(content='{"ok": true}', tool_call_id="1"),
            AIMessage(content="Generic filler."),
        ]
        _patch_final_assistant_with_faq_reply_text(msgs, script)
        self.assertEqual(msgs[-1].content, script)


class TestSpecialistExcludesFaqTool(unittest.TestCase):
    def test_specialist_node_source_has_no_faq_tool_binding(self) -> None:
        root = Path(__file__).resolve().parents[1]
        src = (root / "src" / "agent_lg.py").read_text(encoding="utf-8")
        spec_start = src.find("def _specialist_node")
        rec_start = src.find("def _recommendation_node")
        self.assertGreater(spec_start, 0)
        self.assertGreater(rec_start, spec_start)
        specialist_block = src[spec_start:rec_start]
        self.assertNotIn("FAQ_TOOL", specialist_block)


class TestMasterRouterBindsFaqTool(unittest.TestCase):
    def test_master_has_faq_in_tools_list(self) -> None:
        root = Path(__file__).resolve().parents[1]
        src = (root / "src" / "agent_lg.py").read_text(encoding="utf-8")
        m_start = src.find("def _master_router_node")
        s_start = src.find("def _specialist_node")
        block = src[m_start:s_start]
        self.assertIn("FAQ_TOOL", block)
        self.assertRegex(block, r"tools\s*=\s*\[\s*FAQ_TOOL")


class TestRecommendationBindsFaqTool(unittest.TestCase):
    def test_recommendation_has_faq_in_tools_list(self) -> None:
        root = Path(__file__).resolve().parents[1]
        src = (root / "src" / "agent_lg.py").read_text(encoding="utf-8")
        r_start = src.find("def _recommendation_node")
        ex_start = src.find("def _execute_log_interest")
        self.assertGreater(r_start, 0)
        self.assertGreater(ex_start, r_start)
        block = src[r_start:ex_start]
        self.assertIn("FAQ_TOOL", block)
        self.assertRegex(block, r"tools\s*=\s*\[\s*\n\s*FAQ_TOOL")
        self.assertIn("_ensure_faq_assistant_reply", block)


if __name__ == "__main__":
    unittest.main()
