from __future__ import annotations

from src.chatbot import graph


def _base_state() -> dict:
    return {
        "mind_decision": {},
        "assistant_text": "",
        "customer_full_name": "Test User",
        "customer_email": "test@example.com",
        "customer_phone": "979-555-1111",
        "tool_events": [],
    }


def test_faq_node_prefers_planner_assistant_text(monkeypatch):
    monkeypatch.setattr(
        graph,
        "send_non_sales_faq_email",
        lambda **kwargs: {"status": "sent", "kwargs": kwargs},
    )
    state = _base_state()
    state["mind_decision"] = {
        "faq_category": "financing",
        "faq_summary": "Customer asked about financing.",
        "assistant_text": "We offer financing. Call 979-532-1486 and I can keep helping with options.",
    }

    out = graph._faq_email_node(state)

    assert out["assistant_text"].startswith("We offer financing.")
    assert out["tool_events"][-1]["tool"] == "send_non_sales_faq_email"


def test_faq_node_uses_category_fallback_when_planner_text_missing(monkeypatch):
    monkeypatch.setattr(
        graph,
        "send_non_sales_faq_email",
        lambda **kwargs: {"status": "sent", "kwargs": kwargs},
    )
    state = _base_state()
    state["mind_decision"] = {
        "faq_category": "trade_in",
        "faq_summary": "Customer asked about trade-in.",
        "assistant_text": "",
    }

    out = graph._faq_email_node(state)

    assert out["assistant_text"] == "Our sales team handles trade-in appraisals. Call 979-532-1486."


def test_faq_node_uses_generic_fallback_for_unknown_category(monkeypatch):
    monkeypatch.setattr(
        graph,
        "send_non_sales_faq_email",
        lambda **kwargs: {"status": "sent", "kwargs": kwargs},
    )
    state = _base_state()
    state["mind_decision"] = {
        "faq_category": "unknown_category",
        "faq_summary": "Unknown",
        "assistant_text": "   ",
    }

    out = graph._faq_email_node(state)

    assert out["assistant_text"] == "Thanks, I sent that request to the team so they can help you with it."
