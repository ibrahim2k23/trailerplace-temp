from __future__ import annotations

from src.chatbot import service
from src.models import ChatRequest


def _req(session_id: str, message: str) -> ChatRequest:
    return ChatRequest(
        session_id=session_id,
        message=message,
        customer_full_name="Test User",
        customer_email="test@example.com",
        customer_phone="979-555-1111",
    )


def test_confusion_detector_requires_two_similar_repeats(monkeypatch):
    session = service._new_session("00000000-0000-0000-0000-000000000101")
    session["messages"] = [
        {"role": "user", "content": "I need a 7x16 utility trailer"},
        {"role": "assistant", "content": "Got it."},
        {"role": "user", "content": "I need a 7x16 utility trailer"},
    ]

    class _StubLLM:
        def invoke(self, _messages):
            return service.ConfusionDetectionDecision(similar_repeat_count=2, confused=True)

    monkeypatch.setattr(service, "_confusion_llm", lambda: _StubLLM())
    confused, count = service._is_confused_user_turn(
        session, "I need a 7x16 utility trailer"
    )

    assert confused is True
    assert count == 2


def test_confusion_escalation_sends_email_and_fixed_reply_once(monkeypatch):
    session_id = "00000000-0000-0000-0000-000000000102"
    service.reset_session(session_id)
    email_calls = []
    monkeypatch.setattr(
        service,
        "send_non_sales_faq_email",
        lambda **kwargs: email_calls.append(kwargs) or {"status": "sent"},
    )
    monkeypatch.setattr(service, "_is_confused_user_turn", lambda *_: (True, 2))

    first = service.handle_chat(_req(session_id, "show me a 7x16 utility trailer"))
    second = service.handle_chat(_req(session_id, "show me a 7x16 utility trailer"))

    assert first.assistant_text == service._CONFUSION_ESCALATION_REPLY
    assert second.assistant_text == service._CONFUSION_ESCALATION_REPLY
    assert len(email_calls) == 1
    assert email_calls[0]["faq_category"] == "contact_human"
    assert "appears confused" in email_calls[0]["summary"].lower()
    assert "sales department" in email_calls[0]["summary"].lower()


def test_no_false_positive_when_confusion_not_detected(monkeypatch):
    session_id = "00000000-0000-0000-0000-000000000103"
    service.reset_session(session_id)
    session = service._get_session(session_id)
    session["customer_full_name"] = "Test User"
    session["customer_phone"] = "979-555-1111"
    session["customer_email"] = "test@example.com"
    session["sales_phase"] = "main"
    monkeypatch.setattr(service, "_is_confused_user_turn", lambda *_: (False, 0))
    monkeypatch.setattr(service, "_should_route_to_graph", lambda *_: False)
    monkeypatch.setattr(
        service,
        "_main_smalltalk_response",
        lambda *_: "Happy to help. What kind of trailer do you need?",
    )
    email_calls = []
    monkeypatch.setattr(
        service,
        "send_non_sales_faq_email",
        lambda **kwargs: email_calls.append(kwargs) or {"status": "sent"},
    )

    response = service.handle_chat(_req(session_id, "thanks"))

    assert response.assistant_text == "Happy to help. What kind of trailer do you need?"
    assert email_calls == []
