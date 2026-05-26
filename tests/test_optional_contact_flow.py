from __future__ import annotations

from src.chatbot import graph, service
from src.models import ChatRequest


def _req(session_id: str, message: str) -> ChatRequest:
    return ChatRequest(session_id=session_id, message=message)


def _decision(action: str) -> service.ContactPromptReplyDecision:
    return service.ContactPromptReplyDecision(action=action)


def _bridge(**kwargs) -> str:
    return f"Bridge[{kwargs['action']}]"


class _ContactLLM:
    def __init__(self, result: service.ContactExtraction | None = None, error: Exception | None = None):
        self.result = result or service.ContactExtraction()
        self.error = error

    def invoke(self, _messages):
        if self.error:
            raise self.error
        return self.result


def test_first_message_without_contact_creates_lead_and_asks_for_details(monkeypatch):
    session_id = "00000000-0000-0000-0000-000000001001"
    service.reset_session(session_id)
    lead_calls = []

    monkeypatch.setattr(
        service,
        "create_or_get_soft_lead",
        lambda **kwargs: lead_calls.append(kwargs) or "00000000-0000-0000-0000-000000009001",
    )
    monkeypatch.setattr(service, "update_lead_contact", lambda **kwargs: "00000000-0000-0000-0000-000000009001")

    response = service.handle_chat(_req(session_id, "I need a utility trailer"))

    assert response.sales_phase == "main"
    assert response.contact_status == "missing_contact"
    assert "Thank you for contacting TrailerPlace" in response.assistant_text
    assert "name, email, and phone number" in response.assistant_text
    assert "optional" in response.assistant_text
    assert service._get_session(session_id)["pending_initial_user_message"] == "I need a utility trailer"
    assert lead_calls[0]["full_name"] is None
    assert lead_calls[0]["email"] is None
    assert lead_calls[0]["phone"] is None


def test_contact_extraction_uses_llm_before_regex_for_flexible_phrasing(monkeypatch):
    monkeypatch.setattr(
        service,
        "_contact_llm",
        lambda: _ContactLLM(
            service.ContactExtraction(
                full_name="Tex Johnson",
                email="tex@example.com",
                phone=None,
                name_confidence="high",
                name_evidence="Folks call me Tex Johnson",
            )
        ),
    )

    contact = service._extract_contact(
        "Folks call me Tex Johnson, best way to reach me is tex@example.com",
        {},
    )

    assert contact["full_name"] == "Tex Johnson"
    assert contact["email"] == "tex@example.com"
    assert contact["phone"] is None
    assert contact["name_confidence"] == "high"


def test_contact_extraction_does_not_treat_im_looking_as_name(monkeypatch):
    monkeypatch.setattr(
        service,
        "_contact_llm",
        lambda: _ContactLLM(service.ContactExtraction()),
    )

    contact = service._extract_contact("Hello! I'm looking for a 6x12 trailer", {})

    assert contact["full_name"] is None
    assert contact["email"] is None
    assert contact["phone"] is None


def test_contact_extraction_rejects_sentence_fragment_name_from_llm(monkeypatch):
    monkeypatch.setattr(
        service,
        "_contact_llm",
        lambda: _ContactLLM(
            service.ContactExtraction(
                full_name="not comfortable sharing such details. could you tell me why",
                phone="03209583349",
                name_confidence="high",
                name_evidence="bad fragment",
            )
        ),
    )

    contact = service._extract_contact(
        "I am not comfortable sharing such details. could you tell me why you need them? My number is 03209583349",
        {},
    )

    assert contact["full_name"] is None
    assert contact["phone"] == "03209583349"


def test_contact_extraction_uses_regex_only_when_llm_fails(monkeypatch):
    monkeypatch.setattr(
        service,
        "_contact_llm",
        lambda: _ContactLLM(error=RuntimeError("model unavailable")),
    )

    contact = service._extract_contact("My name is Alex and my email is alex@example.com", {})

    assert contact["full_name"] == "Alex"
    assert contact["email"] == "alex@example.com"
    assert contact["phone"] is None


def test_refusal_after_initial_contact_request_continues_original_request(monkeypatch):
    session_id = "00000000-0000-0000-0000-000000001004"
    service.reset_session(session_id)
    invoked = []

    monkeypatch.setattr(service, "create_or_get_soft_lead", lambda **kwargs: "00000000-0000-0000-0000-000000009004")
    monkeypatch.setattr(service, "update_lead_contact", lambda **kwargs: "00000000-0000-0000-0000-000000009004")
    monkeypatch.setattr(service, "_classify_contact_prompt_reply", lambda *_args, **_kwargs: _decision("resume_saved_request"))
    monkeypatch.setattr(service, "_contact_prompt_bridge_text", _bridge)
    monkeypatch.setattr(service, "_is_confused_user_turn", lambda *_args, **_kwargs: (False, 0))
    monkeypatch.setattr(service, "_should_route_to_graph", lambda _session, message: invoked.append(message) or True)
    monkeypatch.setattr(
        service,
        "_invoke_graph",
        lambda _session, message, _shown: {
            "assistant_text": f"Continuing with: {message}",
            "tool_events": [],
            "last_listings": [],
        },
    )

    first = service.handle_chat(_req(session_id, "Hello, I am looking for a 6x12 trailer"))
    second = service.handle_chat(_req(session_id, "no I don't want to share my details"))

    assert "Before we get started" in first.assistant_text
    assert second.assistant_text == "Bridge[resume_saved_request]\n\nContinuing with: Hello, I am looking for a 6x12 trailer"
    assert invoked[-1] == "Hello, I am looking for a 6x12 trailer"
    assert service._get_session(session_id)["pending_initial_user_message"] is None


def test_new_actionable_request_after_contact_refusal_replaces_original_request(monkeypatch):
    session_id = "00000000-0000-0000-0000-000000001005"
    service.reset_session(session_id)
    invoked = []

    monkeypatch.setattr(service, "create_or_get_soft_lead", lambda **kwargs: "00000000-0000-0000-0000-000000009005")
    monkeypatch.setattr(service, "update_lead_contact", lambda **kwargs: "00000000-0000-0000-0000-000000009005")
    monkeypatch.setattr(service, "_classify_contact_prompt_reply", lambda *_args, **_kwargs: _decision("route_latest_request"))
    monkeypatch.setattr(service, "_is_confused_user_turn", lambda *_args, **_kwargs: (False, 0))
    monkeypatch.setattr(service, "_should_route_to_graph", lambda _session, message: invoked.append(message) or True)
    monkeypatch.setattr(
        service,
        "_invoke_graph",
        lambda _session, message, _shown: {
            "assistant_text": f"Continuing with: {message}",
            "tool_events": [],
            "last_listings": [],
        },
    )

    service.handle_chat(_req(session_id, "I need a 6x12 utility trailer"))
    response = service.handle_chat(_req(session_id, "No thanks, show me dump trailers instead"))

    assert response.assistant_text == "Continuing with: No thanks, show me dump trailers instead"
    assert invoked[-1] == "No thanks, show me dump trailers instead"
    assert service._get_session(session_id)["pending_initial_user_message"] is None


def test_unclear_reply_after_initial_contact_request_continues_original_request(monkeypatch):
    session_id = "00000000-0000-0000-0000-000000001006"
    service.reset_session(session_id)
    invoked = []

    monkeypatch.setattr(service, "create_or_get_soft_lead", lambda **kwargs: "00000000-0000-0000-0000-000000009006")
    monkeypatch.setattr(service, "update_lead_contact", lambda **kwargs: "00000000-0000-0000-0000-000000009006")
    monkeypatch.setattr(service, "_classify_contact_prompt_reply", lambda *_args, **_kwargs: _decision("resume_saved_request"))
    monkeypatch.setattr(service, "_contact_prompt_bridge_text", _bridge)
    monkeypatch.setattr(service, "_is_confused_user_turn", lambda *_args, **_kwargs: (False, 0))
    monkeypatch.setattr(service, "_should_route_to_graph", lambda _session, message: invoked.append(message) or True)
    monkeypatch.setattr(
        service,
        "_invoke_graph",
        lambda _session, message, _shown: {
            "assistant_text": f"Continuing with: {message}",
            "tool_events": [],
            "last_listings": [],
        },
    )

    service.handle_chat(_req(session_id, "I need a 6x12 utility trailer"))
    response = service.handle_chat(_req(session_id, "maybe later"))

    assert response.assistant_text == "Bridge[resume_saved_request]\n\nContinuing with: I need a 6x12 utility trailer"
    assert invoked[-1] == "I need a 6x12 utility trailer"
    assert service._get_session(session_id)["pending_initial_user_message"] is None


def test_contact_reason_question_is_answered_then_saved_request_continues(monkeypatch):
    session_id = "00000000-0000-0000-0000-000000001008"
    service.reset_session(session_id)

    monkeypatch.setattr(service, "create_or_get_soft_lead", lambda **kwargs: "00000000-0000-0000-0000-000000009008")
    monkeypatch.setattr(service, "update_lead_contact", lambda **kwargs: "00000000-0000-0000-0000-000000009008")
    monkeypatch.setattr(service, "_classify_contact_prompt_reply", lambda *_args, **_kwargs: _decision("answer_contact_question"))
    monkeypatch.setattr(service, "_contact_prompt_bridge_text", _bridge)
    monkeypatch.setattr(service, "_is_confused_user_turn", lambda *_args, **_kwargs: (False, 0))
    monkeypatch.setattr(service, "_should_route_to_graph", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        service,
        "_invoke_graph",
        lambda _session, message, _shown: {
            "assistant_text": f"Continuing with: {message}",
            "tool_events": [],
            "last_listings": [],
        },
    )

    service.handle_chat(_req(session_id, "I need a utility trailer"))
    response = service.handle_chat(_req(session_id, "why do you need them?"))

    assert response.assistant_text == "Bridge[answer_contact_question]\n\nContinuing with: I need a utility trailer"
    assert service._get_session(session_id)["pending_initial_user_message"] is None


def test_resumed_graph_does_not_receive_contact_reply_as_recent_context(monkeypatch):
    session_id = "00000000-0000-0000-0000-000000001014"
    service.reset_session(session_id)
    graph_messages = []

    monkeypatch.setattr(service, "create_or_get_soft_lead", lambda **kwargs: "00000000-0000-0000-0000-000000009014")
    monkeypatch.setattr(service, "update_lead_contact", lambda **kwargs: "00000000-0000-0000-0000-000000009014")
    monkeypatch.setattr(service, "_classify_contact_prompt_reply", lambda *_args, **_kwargs: _decision("resume_saved_request"))
    monkeypatch.setattr(service, "_contact_prompt_bridge_text", _bridge)
    monkeypatch.setattr(service, "_is_confused_user_turn", lambda *_args, **_kwargs: (False, 0))
    monkeypatch.setattr(service, "_should_route_to_graph", lambda *_args, **_kwargs: True)

    def _invoke_graph(session, message, _shown):
        graph_messages.extend(session.get("messages") or [])
        return {
            "assistant_text": f"What type of trailer are you looking for? ({message})",
            "tool_events": [],
            "last_listings": [],
        }

    monkeypatch.setattr(service, "_invoke_graph", _invoke_graph)

    service.handle_chat(_req(session_id, "Hello I'm looking for a trailer"))
    response = service.handle_chat(_req(session_id, "I am not comfortable sharing such details. why do you need them?"))

    assert "Bridge[resume_saved_request]" in response.assistant_text
    assert all("not comfortable" not in str(msg.get("content") or "") for msg in graph_messages)
    assert any("looking for a trailer" in str(msg.get("content") or "") for msg in graph_messages)


def test_contact_prompt_bridge_uses_llm_text(monkeypatch):
    class _BridgeLLM:
        def invoke(self, _messages):
            class _Response:
                content = "Absolutely, you can skip that. I can keep helping here."

            return _Response()

    monkeypatch.setattr(service, "_contact_prompt_bridge_llm", lambda: _BridgeLLM())

    text = service._contact_prompt_bridge_text(
        action="resume_saved_request",
        latest_message="I'm not comfortable sharing that",
        saved_request="I need a 6x12 trailer",
        graph_response="What kind of trailer are you looking for?",
    )

    assert text == "Absolutely, you can skip that. I can keep helping here."


def test_contact_prompt_bridge_prompt_forbids_trailer_questions(monkeypatch):
    captured = {}

    class _BridgeLLM:
        def invoke(self, messages):
            captured["system"] = messages[0].content

            class _Response:
                content = "No worries, I can keep helping here."

            return _Response()

    monkeypatch.setattr(service, "_contact_prompt_bridge_llm", lambda: _BridgeLLM())

    service._contact_prompt_bridge_text(
        action="resume_saved_request",
        latest_message="I'm not comfortable sharing that",
        saved_request="Hello I'm looking for a trailer",
        graph_response="What type of trailer are you looking for?",
    )

    system_prompt = captured["system"]
    assert "do not ask any trailer-search or qualification question" in system_prompt
    assert "Never ask what type" in system_prompt
    assert "End without a question mark" in system_prompt


def test_contact_prompt_reply_classifier_uses_regex_fallback_on_llm_failure(monkeypatch):
    session = service._new_session("00000000-0000-0000-0000-000000001007")
    session["pending_initial_user_message"] = "I need a utility trailer"

    class _FailingLLM:
        def invoke(self, _messages):
            raise RuntimeError("model unavailable")

    monkeypatch.setattr(service, "_contact_prompt_reply_llm", lambda: _FailingLLM())

    assert service._should_resume_pending_after_contact_ask(
        session,
        "no I don't want to share my details",
    ) is True
    assert service._should_resume_pending_after_contact_ask(
        session,
        "show me dump trailers instead",
    ) is False


def test_voluntary_contact_updates_lead_status(monkeypatch):
    session_id = "00000000-0000-0000-0000-000000001002"
    service.reset_session(session_id)
    updates = []

    monkeypatch.setattr(service, "create_or_get_soft_lead", lambda **kwargs: "00000000-0000-0000-0000-000000009002")
    monkeypatch.setattr(
        service,
        "update_lead_contact",
        lambda **kwargs: updates.append(kwargs) or "00000000-0000-0000-0000-000000009002",
    )
    session = service._get_session(session_id)
    session["initial_contact_request_asked"] = True
    monkeypatch.setattr(service, "_should_route_to_graph", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(service, "_main_smalltalk_response", lambda *_args, **_kwargs: "Thanks. How can I help?")

    response = service.handle_chat(_req(session_id, "My name is Alex and my email is alex@example.com"))

    assert response.contact_status == "contact_available"
    assert response.customer_full_name == "Alex"
    assert response.customer_email == "alex@example.com"
    assert response.customer_phone is None
    assert updates[-1]["email"] == "alex@example.com"


def test_explicit_name_statement_corrects_existing_bad_name(monkeypatch):
    session_id = "00000000-0000-0000-0000-000000001015"
    service.reset_session(session_id)

    monkeypatch.setattr(service, "create_or_get_soft_lead", lambda **kwargs: "00000000-0000-0000-0000-000000009015")
    monkeypatch.setattr(service, "update_lead_contact", lambda **kwargs: "00000000-0000-0000-0000-000000009015")
    session = service._get_session(session_id)
    session["initial_contact_request_asked"] = True
    session["customer_full_name"] = "not comfortable sharing such details"
    session["customer_full_name_confidence"] = "low"
    monkeypatch.setattr(
        service,
        "_contact_llm",
        lambda: _ContactLLM(
            service.ContactExtraction(
                full_name="Minahil",
                phone="03209583349",
                name_confidence="high",
                name_evidence="My name is Minahil",
            )
        ),
    )
    monkeypatch.setattr(service, "_should_route_to_graph", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(service, "_main_smalltalk_response", lambda *_args, **_kwargs: "Thanks. How can I help?")

    response = service.handle_chat(_req(session_id, "My name is Minahil and my number is 03209583349"))

    assert response.customer_full_name == "Minahil"
    assert response.customer_phone == "03209583349"
    assert service._get_session(session_id)["customer_full_name_confidence"] == "high"


def test_name_and_phone_are_sufficient_contact(monkeypatch):
    session_id = "00000000-0000-0000-0000-000000001009"
    service.reset_session(session_id)

    monkeypatch.setattr(service, "create_or_get_soft_lead", lambda **kwargs: "00000000-0000-0000-0000-000000009009")
    monkeypatch.setattr(service, "update_lead_contact", lambda **kwargs: "00000000-0000-0000-0000-000000009009")
    session = service._get_session(session_id)
    session["initial_contact_request_asked"] = True
    monkeypatch.setattr(service, "_should_route_to_graph", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(service, "_main_smalltalk_response", lambda *_args, **_kwargs: "Thanks. How can I help?")

    response = service.handle_chat(_req(session_id, "I'm Alex and my phone is 979-555-1212"))

    assert response.contact_status == "contact_available"
    assert response.customer_full_name == "Alex"
    assert response.customer_phone == "979-555-1212"
    assert response.customer_email == ""
    assert "email" not in response.assistant_text.lower()


def test_email_only_contact_asks_for_name(monkeypatch):
    session_id = "00000000-0000-0000-0000-000000001010"
    service.reset_session(session_id)

    monkeypatch.setattr(service, "create_or_get_soft_lead", lambda **kwargs: "00000000-0000-0000-0000-000000009010")
    monkeypatch.setattr(service, "update_lead_contact", lambda **kwargs: "00000000-0000-0000-0000-000000009010")

    response = service.handle_chat(_req(session_id, "alex@example.com"))

    assert response.contact_status == "missing_contact"
    assert response.customer_email == "alex@example.com"
    assert "your name" in response.assistant_text
    assert "phone" not in response.assistant_text.lower()


def test_phone_only_contact_asks_for_name(monkeypatch):
    session_id = "00000000-0000-0000-0000-000000001011"
    service.reset_session(session_id)

    monkeypatch.setattr(service, "create_or_get_soft_lead", lambda **kwargs: "00000000-0000-0000-0000-000000009011")
    monkeypatch.setattr(service, "update_lead_contact", lambda **kwargs: "00000000-0000-0000-0000-000000009011")

    response = service.handle_chat(_req(session_id, "979-555-1212"))

    assert response.contact_status == "missing_contact"
    assert response.customer_phone == "979-555-1212"
    assert "your name" in response.assistant_text
    assert "email" not in response.assistant_text.lower()


def test_opportunistic_contact_capture_fills_missing_fields_during_normal_flow(monkeypatch):
    session_id = "00000000-0000-0000-0000-000000001012"
    service.reset_session(session_id)

    monkeypatch.setattr(service, "create_or_get_soft_lead", lambda **kwargs: "00000000-0000-0000-0000-000000009012")
    monkeypatch.setattr(service, "update_lead_contact", lambda **kwargs: "00000000-0000-0000-0000-000000009012")
    session = service._get_session(session_id)
    session["initial_contact_request_asked"] = True
    monkeypatch.setattr(service, "_is_confused_user_turn", lambda *_args, **_kwargs: (False, 0))
    monkeypatch.setattr(service, "_should_route_to_graph", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        service,
        "_invoke_graph",
        lambda _session, message, _shown: {
            "assistant_text": f"Continuing with: {message}",
            "tool_events": [],
            "last_listings": [],
        },
    )

    response = service.handle_chat(_req(session_id, "I'm Alex, call me at 979-555-1212 and show utility trailers"))

    assert response.contact_status == "contact_available"
    assert response.customer_full_name == "Alex"
    assert response.customer_phone == "979-555-1212"
    assert "Continuing with:" in response.assistant_text


def test_sufficient_contact_is_not_overwritten_or_reclassified(monkeypatch):
    session_id = "00000000-0000-0000-0000-000000001013"
    service.reset_session(session_id)

    monkeypatch.setattr(service, "create_or_get_soft_lead", lambda **kwargs: "00000000-0000-0000-0000-000000009013")
    monkeypatch.setattr(service, "update_lead_contact", lambda **kwargs: "00000000-0000-0000-0000-000000009013")
    session = service._get_session(session_id)
    session["initial_contact_request_asked"] = True
    session["customer_full_name"] = "Original User"
    session["customer_full_name_confidence"] = "high"
    session["customer_email"] = "original@example.com"
    service._sync_contact_status(session)

    def _bad_extract(*_args, **_kwargs):
        raise AssertionError("Contact extraction should not run once sufficient contact exists")

    monkeypatch.setattr(service, "_extract_contact", _bad_extract)
    monkeypatch.setattr(service, "_should_route_to_graph", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(service, "_main_smalltalk_response", lambda *_args, **_kwargs: "How can I help?")

    response = service.handle_chat(_req(session_id, "I'm New User, call me at 979-555-1212"))

    assert response.customer_full_name == "Original User"
    assert response.customer_email == "original@example.com"
    assert response.customer_phone is None


def test_interest_without_contact_is_deferred(monkeypatch):
    sent = []
    monkeypatch.setattr(
        graph,
        "send_interested_listing_email",
        lambda **kwargs: sent.append(kwargs) or {"status": "sent"},
    )
    state = {
        "session_id": "s1",
        "customer_full_name": None,
        "customer_email": None,
        "customer_phone": None,
        "tool_events": [],
        "mind_decision": {"selected_listing_title": "Trailer A"},
    }

    out = graph._interest_email_node(state)

    assert sent == []
    assert out["pending_contact_action"]["type"] == "interest"
    assert out["pending_contact_action"]["item_name"] == "Trailer A"
    assert "phone number or email address" in out["assistant_text"]


def test_pending_interest_sends_once_after_contact(monkeypatch):
    session_id = "00000000-0000-0000-0000-000000001003"
    service.reset_session(session_id)
    session = service._get_session(session_id)
    session["lead_id"] = "00000000-0000-0000-0000-000000009003"
    session["pending_contact_action"] = {"type": "interest", "item_name": "Trailer A"}
    sent = []

    monkeypatch.setattr(service, "create_or_get_soft_lead", lambda **kwargs: session["lead_id"])
    monkeypatch.setattr(service, "update_lead_contact", lambda **kwargs: session["lead_id"])
    monkeypatch.setattr(
        service,
        "send_interested_listing_email",
        lambda **kwargs: sent.append(kwargs) or {"status": "sent"},
    )

    response = service.handle_chat(_req(session_id, "I'm Alex and my email is alex@example.com"))

    assert sent == [
        {
            "session_id": session_id,
            "full_name": "Alex",
            "email": "alex@example.com",
            "phone": "",
            "item_name": "Trailer A",
        }
    ]
    assert response.contact_status == "contact_available"
    assert service._get_session(session_id)["pending_contact_action"] is None


def test_contact_only_after_results_acknowledges_without_graph(monkeypatch):
    session_id = "00000000-0000-0000-0000-000000001016"
    service.reset_session(session_id)
    session = service._get_session(session_id)
    session["initial_contact_request_asked"] = True
    session["last_listings"] = [{"title": "Trailer A", "url": "https://example.test/a"}]
    session["already_shown_listing_urls"] = ["https://example.test/a"]
    session["has_shown_search_results"] = True

    monkeypatch.setattr(service, "create_or_get_soft_lead", lambda **kwargs: "00000000-0000-0000-0000-000000009016")
    monkeypatch.setattr(service, "update_lead_contact", lambda **kwargs: "00000000-0000-0000-0000-000000009016")
    monkeypatch.setattr(
        service,
        "_should_route_to_graph",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("contact-only should not route")),
    )
    monkeypatch.setattr(
        service,
        "_invoke_graph",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("contact-only should not invoke graph")),
    )

    response = service.handle_chat(_req(session_id, "my name is Ibrahim and number is 03304388550"))
    saved = service._get_session(session_id)

    assert response.assistant_text == service._CONTACT_ONLY_ACK
    assert response.contact_status == "contact_available"
    assert response.customer_full_name == "Ibrahim"
    assert response.customer_phone == "03304388550"
    assert saved["last_listings"] == [{"title": "Trailer A", "url": "https://example.test/a"}]


def test_contact_plus_listing_interest_routes_to_graph(monkeypatch):
    session_id = "00000000-0000-0000-0000-000000001017"
    service.reset_session(session_id)
    session = service._get_session(session_id)
    session["initial_contact_request_asked"] = True
    session["last_listings"] = [{"title": "Trailer A", "url": "https://example.test/a"}]
    session["already_shown_listing_urls"] = ["https://example.test/a"]
    session["has_shown_search_results"] = True
    routed = []

    monkeypatch.setattr(service, "create_or_get_soft_lead", lambda **kwargs: "00000000-0000-0000-0000-000000009017")
    monkeypatch.setattr(service, "update_lead_contact", lambda **kwargs: "00000000-0000-0000-0000-000000009017")
    monkeypatch.setattr(service, "_is_confused_user_turn", lambda *_args, **_kwargs: (False, 0))
    monkeypatch.setattr(service, "_should_route_to_graph", lambda _session, message: routed.append(message) or True)
    monkeypatch.setattr(
        service,
        "_invoke_graph",
        lambda _session, message, _shown: {
            "assistant_text": f"Graph handled: {message}",
            "tool_events": [{"tool": "send_interested_listing_email", "status": "sent"}],
            "last_listings": session["last_listings"],
        },
    )

    response = service.handle_chat(
        _req(session_id, "my name is Ibrahim and number is 03304388550 and I am interested in trailer #1")
    )

    assert "Graph handled:" in response.assistant_text
    assert routed[-1].endswith("I am interested in trailer #1")
    assert response.customer_full_name == "Ibrahim"
    assert response.customer_phone == "03304388550"


def test_contact_plus_new_trailer_request_routes_to_graph(monkeypatch):
    session_id = "00000000-0000-0000-0000-000000001018"
    service.reset_session(session_id)
    session = service._get_session(session_id)
    session["initial_contact_request_asked"] = True
    session["last_listings"] = [{"title": "Trailer A", "url": "https://example.test/a"}]
    session["already_shown_listing_urls"] = ["https://example.test/a"]
    session["has_shown_search_results"] = True
    routed = []

    monkeypatch.setattr(service, "create_or_get_soft_lead", lambda **kwargs: "00000000-0000-0000-0000-000000009018")
    monkeypatch.setattr(service, "update_lead_contact", lambda **kwargs: "00000000-0000-0000-0000-000000009018")
    monkeypatch.setattr(service, "_is_confused_user_turn", lambda *_args, **_kwargs: (False, 0))
    monkeypatch.setattr(service, "_should_route_to_graph", lambda _session, message: routed.append(message) or True)
    monkeypatch.setattr(
        service,
        "_invoke_graph",
        lambda _session, message, _shown: {
            "assistant_text": f"Graph handled: {message}",
            "tool_events": [],
            "last_listings": [],
        },
    )

    response = service.handle_chat(
        _req(session_id, "my name is Ibrahim and number is 03304388550 and show me dump trailers")
    )

    assert "Graph handled:" in response.assistant_text
    assert routed[-1].endswith("show me dump trailers")
    assert response.customer_phone == "03304388550"


def test_recommendation_contact_ask_happens_once_without_contact(monkeypatch):
    state = {
        "trailer_category": "Utility",
        "slots_collected": {},
        "metadata_filters_collected": {},
        "user_message": "show utility trailers",
        "already_shown_listing_urls": [],
        "tool_events": [],
        "customer_email": None,
        "customer_phone": None,
        "contact_request_asked_after_recommendation": False,
    }

    def _search(**_kwargs):
        return [{"title": "Trailer A", "url": "https://example.test/a", "length": "12 ft"}]

    monkeypatch.setattr(graph, "search_pinecone_listings", _search)
    monkeypatch.setenv("WHY_IT_FITS_LLM_ENABLED", "0")
    first = graph._pinecone_search_node(state)
    second = graph._pinecone_search_node(first)

    assert "phone number or email address" in first["assistant_text"]
    assert "phone number or email address" not in second["assistant_text"]


def test_metadata_only_followups_route_to_graph_with_search_context():
    base_session = service._new_session("metadata-route")
    base_session["metadata_filters_collected"] = {"length_ft": "12"}

    for message in (
        "bumper pull",
        "under 10000",
        "make it 14 ft",
        "6 feet wide",
        "3000 pounds",
        "black",
        "Diamond C",
    ):
        assert service._should_route_to_graph(base_session, message) is True


def test_business_overview_questions_stay_in_smalltalk_path():
    session = service._new_session("overview-route")

    for message in (
        "what services do you guys offer?",
        "what do you guys have?",
        "what trailers do you carry?",
    ):
        assert service._should_route_to_graph(session, message) is False


def test_business_overview_smalltalk_prompt_guides_llm(monkeypatch):
    session = service._new_session("overview-response")
    captured = {}

    class _OverviewLLM:
        def __init__(self, **_kwargs):
            pass

        def invoke(self, messages):
            captured["system"] = messages[0].content

            class _Response:
                content = "LLM overview response"

            return _Response()

    monkeypatch.setattr(service, "ChatOpenAI", _OverviewLLM)

    response = service._main_smalltalk_response(session, "what services do you guys offer?")

    assert response == "LLM overview response"
    assert "utility, dump, equipment" in captured["system"]
    assert "bumper pull" in captured["system"]
    assert "gooseneck" in captured["system"]
    assert "financing" in captured["system"]
    assert "trade-ins" in captured["system"]
    assert "service or spare parts" in captured["system"]
    assert "Do not mention rentals" in captured["system"]
