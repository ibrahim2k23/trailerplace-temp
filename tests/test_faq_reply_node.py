from __future__ import annotations

from src.chatbot import graph


def _base_state() -> dict:
    return {
        "mind_decision": {},
        "assistant_text": "",
        "customer_full_name": "Test User",
        "customer_email": "test@example.com",
        "customer_phone": "979-555-1111",
        "session_id": "session-123",
        "user_message": "Do you sell service parts?",
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
    assert out["tool_events"][-1]["result"]["kwargs"]["session_id"] == "session-123"


def test_faq_node_uses_category_label_not_bad_planner_summary(monkeypatch):
    monkeypatch.setattr(
        graph,
        "send_non_sales_faq_email",
        lambda **kwargs: {"status": "sent", "kwargs": kwargs},
    )
    state = _base_state()
    state["mind_decision"] = {
        "faq_category": "service_parts",
        "faq_summary": "Customer asked to contact a human.",
        "assistant_text": "Our team can help with parts.",
    }

    out = graph._faq_email_node(state)

    kwargs = out["tool_events"][-1]["result"]["kwargs"]
    assert kwargs["faq_category"] == "service_parts"
    assert kwargs["summary"] == "Customer asked about service or parts."
    assert kwargs["user_message"] == "Do you sell service parts?"
    assert kwargs["summary"] != state["mind_decision"]["assistant_text"]


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

    assert out["assistant_text"] == "You can reach our team at 979-532-1486. Happy to keep helping with your trailer search too!"


def test_faq_node_missing_contact_uses_neutral_team_notification_copy():
    state = _base_state()
    state.update(
        {
            "customer_full_name": None,
            "customer_email": None,
            "customer_phone": None,
            "user_message": "How do I contact you guys?",
            "mind_decision": {
                "faq_category": "contact_human",
                "assistant_text": "You can reach our team at 979-532-1486.",
            },
        }
    )

    out = graph._faq_email_node(state)

    assert out["assistant_text"].startswith("To notify our team, could you share your name")
    assert "customer asked to contact a person" not in out["assistant_text"].lower()
    assert out["tool_events"][-1]["result"]["status"] == "deferred_missing_contact"


def test_escalation_node_defers_for_contact_and_repeats_active_question():
    state = _base_state()
    state.update(
        {
            "customer_email": None,
            "customer_phone": None,
            "user_message": "Can you email me a quote?",
            "mind_decision": {
                "action": "send_escalation_alert_email",
                "escalation_summary": "Customer asked for a quote.",
                "unsupported_request": "Can you email me a quote?",
            },
            "awaiting_slot": "haul_weight_lbs",
            "pending_questions": [
                {"slot": "haul_weight_lbs", "question": "How much weight will you be hauling?", "required": True}
            ],
        }
    )

    out = graph._escalation_email_node(state)

    assert out["pending_contact_action"]["type"] == "escalation_alert"
    assert out["tool_events"][-1]["tool"] == "send_escalation_alert_email"
    assert "phone number or email address" in out["assistant_text"]
    assert "How much weight will you be hauling?" in out["assistant_text"]


def test_escalation_node_sends_and_repeats_active_question(monkeypatch):
    calls = []
    monkeypatch.setattr(
        graph,
        "send_escalation_alert_email",
        lambda **kwargs: calls.append(kwargs) or {"status": "sent", "kwargs": kwargs},
    )
    state = _base_state()
    state.update(
        {
            "user_message": "Can you email me a quote?",
            "mind_decision": {
                "action": "send_escalation_alert_email",
                "escalation_summary": "Customer asked for a quote.",
                "unsupported_request": "Can you email me a quote?",
            },
            "messages": [
                {"role": "user", "content": "Can you email me a quote?"},
            ],
            "awaiting_slot": "haul_weight_lbs",
            "pending_questions": [
                {"slot": "haul_weight_lbs", "question": "How much weight will you be hauling?", "required": True}
            ],
        }
    )

    out = graph._escalation_email_node(state)

    assert calls[0]["summary"] == "Customer asked for a quote."
    assert "session_id" not in calls[0]
    assert "Recent conversation:\nUser:\nCan you email me a quote?" in calls[0]["context_summary"]
    assert out["tool_events"][-1]["tool"] == "send_escalation_alert_email"
    assert "I've sent your query to our team" in out["assistant_text"]
    assert "How much weight will you be hauling?" in out["assistant_text"]
