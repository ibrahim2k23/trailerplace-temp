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
            return service.ConfusionDetectionDecision(
                similar_repeat_count=2,
                confusion_score=90,
                confused=True,
            )

    monkeypatch.setattr(service, "_confusion_llm", lambda: _StubLLM())
    confused, count = service._is_confused_user_turn(
        session, "I need a 7x16 utility trailer"
    )

    assert confused is True
    assert count == 2


def test_confusion_detector_ignores_genuine_answer_to_assistant_question(monkeypatch):
    session = service._new_session("00000000-0000-0000-0000-000000000106")
    session["messages"] = [
        {"role": "user", "content": "I need a trailer"},
        {"role": "assistant", "content": "What kind of trailer are you looking for?"},
        {"role": "user", "content": "A utility trailer"},
    ]

    class _StubLLM:
        def invoke(self, _messages):
            return service.ConfusionDetectionDecision(
                similar_repeat_count=2,
                confusion_score=92,
                is_answer_to_assistant_question=True,
                confused=True,
            )

    monkeypatch.setattr(service, "_confusion_llm", lambda: _StubLLM())

    confused, count = service._is_confused_user_turn(session, "A utility trailer")

    assert confused is False
    assert count == 0


def test_confusion_detector_requires_score_at_least_85(monkeypatch):
    session = service._new_session("00000000-0000-0000-0000-000000000107")
    session["messages"] = [
        {"role": "user", "content": "Do you have utility trailers?"},
        {"role": "assistant", "content": "What will you be hauling?"},
        {"role": "user", "content": "Do you have utility trailers?"},
    ]

    class _StubLLM:
        def invoke(self, _messages):
            return service.ConfusionDetectionDecision(
                similar_repeat_count=2,
                confusion_score=84,
                confused=True,
            )

    monkeypatch.setattr(service, "_confusion_llm", lambda: _StubLLM())

    confused, count = service._is_confused_user_turn(session, "Do you have utility trailers?")

    assert confused is False
    assert count == 2


def test_non_duplicate_actionable_message_does_not_call_confusion_llm(monkeypatch):
    session = service._new_session("00000000-0000-0000-0000-000000000111")
    session["messages"] = [
        {"role": "user", "content": "I need a 7x16 utility trailer"},
        {"role": "assistant", "content": "Here are some options."},
        {"role": "user", "content": "what financing options do you have?"},
    ]

    class _BadLLM:
        def invoke(self, _messages):
            raise AssertionError("Confusion LLM should not run without duplicate/confusion evidence")

    monkeypatch.setattr(service, "_confusion_llm", lambda: _BadLLM())

    confused, count = service._is_confused_user_turn(
        session,
        "what financing options do you have?",
    )

    assert confused is False
    assert count == 0


def test_store_contact_questions_are_not_confusion_eligible():
    session = service._new_session("00000000-0000-0000-0000-000000000112")
    session["last_listings"] = [{"title": "Trailer A"}]

    for message in (
        "how can I contact you guys?",
        "what is your phone number?",
        "where are you located?",
        "what are your hours?",
        "how do I reach sales?",
    ):
        assert service._is_confusion_eligible_user_message(message, session) is False


def test_contact_question_after_interest_does_not_escalate(monkeypatch):
    session_id = "00000000-0000-0000-0000-000000000113"
    service.reset_session(session_id)
    session = service._get_session(session_id)
    session["customer_full_name"] = "Ibrahim"
    session["customer_phone"] = "03304388550"
    session["sales_phase"] = "main"
    session["last_listings"] = [{"title": "Trailer A", "url": "https://example.test/a"}]
    session["has_shown_search_results"] = True
    session["messages"] = [
        {"role": "user", "content": "yeah. I'm interested in the 4th trailer"},
        {"role": "assistant", "content": "Your interest has been logged."},
        {"role": "user", "content": "how can I contact you guys?"},
    ]
    email_calls = []

    class _BadLLM:
        def invoke(self, _messages):
            raise AssertionError("Confusion LLM should not run for contact-info questions")

    monkeypatch.setattr(service, "_confusion_llm", lambda: _BadLLM())
    monkeypatch.setattr(
        service,
        "send_non_sales_faq_email",
        lambda **kwargs: email_calls.append(kwargs) or {"status": "sent"},
    )
    monkeypatch.setattr(service, "_should_route_to_graph", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        service,
        "_main_smalltalk_response",
        lambda *_args, **_kwargs: "You can call us at 979-532-1486.",
    )

    response = service.handle_chat(_req(session_id, "how can I contact you guys?"))

    assert response.assistant_text == "You can call us at 979-532-1486."
    assert email_calls == []


def test_explicit_confusion_outside_active_qna_can_escalate(monkeypatch):
    session = service._new_session("00000000-0000-0000-0000-000000000114")
    session["messages"] = [
        {"role": "user", "content": "I need a trailer"},
        {"role": "assistant", "content": "What kind of trailer do you need?"},
        {"role": "user", "content": "I am confused and I don't know what to choose"},
    ]

    class _StubLLM:
        def invoke(self, _messages):
            return service.ConfusionDetectionDecision(
                similar_repeat_count=1,
                confusion_score=90,
                confused=True,
            )

    monkeypatch.setattr(service, "_confusion_llm", lambda: _StubLLM())

    confused, count = service._is_confused_user_turn(
        session,
        "I am confused and I don't know what to choose",
    )

    assert confused is True
    assert count == 1


def test_explicit_confusion_during_active_qna_does_not_run_confusion_llm(monkeypatch):
    session = service._new_session("00000000-0000-0000-0000-000000000115")
    session["awaiting_slot"] = "haul_item"
    session["messages"] = [
        {"role": "user", "content": "I need an equipment trailer"},
        {"role": "assistant", "content": "What equipment will you be hauling?"},
        {"role": "user", "content": "I am confused"},
    ]

    class _BadLLM:
        def invoke(self, _messages):
            raise AssertionError("Confusion LLM should not run during active Q&A")

    monkeypatch.setattr(service, "_confusion_llm", lambda: _BadLLM())

    confused, count = service._is_confused_user_turn(session, "I am confused")

    assert confused is False
    assert count == 0


def test_result_navigation_after_listings_requires_three_repeats(monkeypatch):
    session = service._new_session("00000000-0000-0000-0000-000000000109")
    session["last_listings"] = [{"title": "Trailer A"}]
    session["messages"] = [
        {"role": "user", "content": "show more results"},
        {"role": "assistant", "content": "Here are more options."},
        {"role": "user", "content": "show more results"},
    ]

    class _StubLLM:
        def invoke(self, _messages):
            return service.ConfusionDetectionDecision(
                similar_repeat_count=2,
                confusion_score=95,
                confused=True,
            )

    monkeypatch.setattr(service, "_confusion_llm", lambda: _StubLLM())

    confused, count = service._is_confused_user_turn(session, "show more results")

    assert confused is False
    assert count == 0


def test_result_navigation_after_listings_escalates_at_three_repeats(monkeypatch):
    session = service._new_session("00000000-0000-0000-0000-000000000110")
    session["already_shown_listing_urls"] = ["https://example.test/trailer-a"]
    session["messages"] = [
        {"role": "user", "content": "show more results"},
        {"role": "assistant", "content": "Here are more options."},
        {"role": "user", "content": "more options"},
        {"role": "assistant", "content": "Here are more options."},
        {"role": "user", "content": "next"},
    ]

    class _StubLLM:
        def invoke(self, _messages):
            return service.ConfusionDetectionDecision(
                similar_repeat_count=3,
                confusion_score=95,
                confused=True,
            )

    monkeypatch.setattr(service, "_confusion_llm", lambda: _StubLLM())

    confused, count = service._is_confused_user_turn(session, "next")

    assert confused is True
    assert count == 3


def test_confusion_detector_excludes_contact_only_message(monkeypatch):
    session = service._new_session("00000000-0000-0000-0000-000000000104")
    session["messages"] = [
        {"role": "user", "content": "Ibrahim. 03304388550"},
        {"role": "assistant", "content": "Thank you, Ibrahim! How can I assist you today?"},
        {"role": "user", "content": "I am looking for a 12 feet utility trailer"},
    ]

    class _BadLLM:
        def invoke(self, _messages):
            raise AssertionError("Confusion LLM should not run on the first eligible main request")

    monkeypatch.setattr(service, "_confusion_llm", lambda: _BadLLM())

    confused, count = service._is_confused_user_turn(
        session,
        "I am looking for a 12 feet utility trailer",
    )

    assert confused is False
    assert count == 0


def test_first_main_request_after_contact_does_not_escalate(monkeypatch):
    session_id = "00000000-0000-0000-0000-000000000105"
    service.reset_session(session_id)
    session = service._get_session(session_id)
    session["customer_full_name"] = "Ibrahim"
    session["customer_phone"] = "03304388550"
    session["sales_phase"] = "main"
    session["messages"] = [
        {"role": "user", "content": "Ibrahim. 03304388550"},
        {"role": "assistant", "content": "Thank you, Ibrahim! How can I assist you today?"},
    ]
    email_calls = []
    monkeypatch.setattr(
        service,
        "send_non_sales_faq_email",
        lambda **kwargs: email_calls.append(kwargs) or {"status": "sent"},
    )
    monkeypatch.setattr(
        service,
        "_invoke_graph",
        lambda *_args, **_kwargs: {
            "assistant_text": "What will you be hauling on the utility trailer?",
            "tool_events": [],
            "last_listings": [],
            "trailer_category": "Utility",
        },
    )

    response = service.handle_chat(_req(session_id, "I am looking for a 12 feet utility trailer"))

    assert response.assistant_text == "What will you be hauling on the utility trailer?"
    assert email_calls == []


def test_active_qualification_answer_bypasses_confusion_detection(monkeypatch):
    session_id = "00000000-0000-0000-0000-000000000108"
    service.reset_session(session_id)
    session = service._get_session(session_id)
    session["customer_full_name"] = "Ibrahim"
    session["customer_phone"] = "03304388550"
    session["sales_phase"] = "main"
    session["trailer_category"] = "Equipment"
    session["awaiting_slot"] = "haul_item"
    session["pending_questions"] = [
        {
            "slot": "haul_weight_lbs",
            "question": "What's the rough total weight of the equipment?",
            "required": True,
        }
    ]
    session["messages"] = [
        {"role": "user", "content": "I am looking for a 6x12 equipment trailer"},
        {
            "role": "assistant",
            "content": "What equipment will you be hauling (e.g. skid steer, mini excavator, tractor)?",
        },
    ]
    email_calls = []
    monkeypatch.setattr(
        service,
        "send_non_sales_faq_email",
        lambda **kwargs: email_calls.append(kwargs) or {"status": "sent"},
    )

    def _bad_confusion(*_args, **_kwargs):
        raise AssertionError("Confusion detection should not run while answering a qualification question")

    monkeypatch.setattr(service, "_is_confused_user_turn", _bad_confusion)
    monkeypatch.setattr(
        service,
        "_invoke_graph",
        lambda *_args, **_kwargs: {
            "assistant_text": "What's the rough total weight of the equipment?",
            "tool_events": [],
            "last_listings": [],
            "trailer_category": "Equipment",
            "awaiting_slot": "haul_weight_lbs",
        },
    )

    response = service.handle_chat(
        _req(session_id, "a car. The trailer should be a bumper pull one as well")
    )

    assert response.assistant_text == "What's the rough total weight of the equipment?"
    assert email_calls == []


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
