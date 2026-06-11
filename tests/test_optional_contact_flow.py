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


def test_stock_lookup_before_results_stays_in_contact_flow(monkeypatch):
    session_id = "00000000-0000-0000-0000-000000001019"
    service.reset_session(session_id)
    search_calls = []

    monkeypatch.setattr(service, "create_or_get_soft_lead", lambda **kwargs: "00000000-0000-0000-0000-000000009019")
    monkeypatch.setattr(service, "update_lead_contact", lambda **kwargs: "00000000-0000-0000-0000-000000009019")
    monkeypatch.setattr(
        service,
        "search_trailers",
        lambda *_args, **_kwargs: search_calls.append(True) or {},
    )

    response = service.handle_chat(_req(session_id, "do you have stock 13066?"))

    assert "Before we get started" in response.assistant_text
    assert search_calls == []
    assert service._get_session(session_id)["initial_contact_request_asked"] is True


def test_year_make_price_lookup_answers_before_contact_or_graph(monkeypatch):
    session_id = "00000000-0000-0000-0000-000000001021"
    service.reset_session(session_id)

    monkeypatch.setattr(service, "create_or_get_soft_lead", lambda **kwargs: "00000000-0000-0000-0000-000000009021")
    monkeypatch.setattr(service, "update_lead_contact", lambda **kwargs: "00000000-0000-0000-0000-000000009021")
    monkeypatch.setattr(service, "_persist", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        service,
        "search_trailers",
        lambda *_args, **_kwargs: {
            "reply": "We do not currently show 2026 Aluma in the inventory data, but this is the closest alternative.",
            "entity_type": "YEAR_MAKE_SEARCH",
            "confidence": 0.0,
            "top_matches": [{"title": "2025 Aluma Utility", "stock_number": "12345"}],
            "extraction": {"year": 2026, "possible_make": "Aluma", "user_wants_price": True},
            "no_exact_reason": "no_exact_inventory_match_for_requested_identifiers",
        },
    )
    monkeypatch.setattr(
        service,
        "_should_route_to_graph",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("direct lookup should not route to graph")),
    )

    response = service.handle_chat(_req(session_id, "2026 Aluma price?"))

    assert response.assistant_text.startswith("We do not currently show 2026 Aluma")
    assert "Before we get started" not in response.assistant_text
    assert service._get_session(session_id)["initial_contact_request_asked"] is False


def test_pure_make_lookup_still_uses_initial_contact_flow(monkeypatch):
    session_id = "00000000-0000-0000-0000-000000001020"
    service.reset_session(session_id)

    monkeypatch.setattr(service, "create_or_get_soft_lead", lambda **kwargs: "00000000-0000-0000-0000-000000009020")
    monkeypatch.setattr(service, "update_lead_contact", lambda **kwargs: "00000000-0000-0000-0000-000000009020")
    monkeypatch.setattr(service, "should_attempt_chat_lookup", lambda *_args, **_kwargs: False)

    response = service.handle_chat(_req(session_id, "Diamond C trailer"))

    assert "Before we get started" in response.assistant_text
    assert service._get_session(session_id)["pending_initial_user_message"] == "Diamond C trailer"


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


def test_inventory_lookup_is_blocked_during_active_qna(monkeypatch):
    session_id = "00000000-0000-0000-0000-000000001116"
    service.reset_session(session_id)
    session = service._get_session(session_id)
    session["initial_contact_request_asked"] = True
    session["has_shown_search_results"] = True
    session["awaiting_slot"] = "haul_weight_lbs"
    session["trailer_category"] = "Dump"
    session["slots_collected"] = {"haul_material": "gravel"}
    search_calls = []

    monkeypatch.setattr(service, "create_or_get_soft_lead", lambda **kwargs: "00000000-0000-0000-0000-000000009116")
    monkeypatch.setattr(service, "update_lead_contact", lambda **kwargs: "00000000-0000-0000-0000-000000009116")
    monkeypatch.setattr(
        service,
        "search_trailers",
        lambda *_args, **_kwargs: search_calls.append(True) or {},
    )
    monkeypatch.setattr(service, "_is_confused_user_turn", lambda *_args, **_kwargs: (False, 0))
    monkeypatch.setattr(service, "_persist", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        service,
        "_invoke_graph",
        lambda *_args, **_kwargs: {
            "assistant_text": "What's the rough haul weight per load?",
            "mind_decision": {"action": "respond"},
            "trailer_category": "Dump",
            "slots_collected": {"haul_material": "gravel", "haul_weight_lbs": "5000 pounds"},
            "slots_skipped": [],
            "metadata_filters_collected": {"payload_lbs": "5000 pounds"},
            "requested_non_metadata_features": [],
            "active_search_request_text": "",
            "make_category_options": [],
            "awaiting_slot": None,
            "pending_questions": [],
            "asked_questions": [],
            "already_shown_listing_urls": [],
            "last_listings": [],
            "tool_events": [],
            "pending_contact_action": None,
        },
    )

    response = service.handle_chat(_req(session_id, "5000 pounds"))

    assert search_calls == []
    assert response.assistant_text == "What's the rough haul weight per load?"


def test_inventory_lookup_works_after_results_for_ordinal_reference(monkeypatch):
    session_id = "00000000-0000-0000-0000-000000001117"
    service.reset_session(session_id)
    session = service._get_session(session_id)
    session["initial_contact_request_asked"] = True
    session["has_shown_search_results"] = True
    session["last_listings"] = [
        {"title": "Trailer A", "stock_number": "11111", "price": "$1"},
        {"title": "Trailer B", "stock_number": "22222", "price": "$2"},
    ]
    session["already_shown_listing_urls"] = ["https://example.test/a", "https://example.test/b"]
    search_calls = []

    monkeypatch.setattr(service, "create_or_get_soft_lead", lambda **kwargs: "00000000-0000-0000-0000-000000009117")
    monkeypatch.setattr(service, "update_lead_contact", lambda **kwargs: "00000000-0000-0000-0000-000000009117")
    monkeypatch.setattr(
        service,
        "search_trailers",
        lambda *_args, **_kwargs: search_calls.append(True) or {
            "reply": "Yes, that trailer is listed at $1.",
            "entity_type": "STOCK_SEARCH",
            "confidence": 1.0,
            "top_matches": [{"title": "Trailer A", "stock_number": "11111", "price": "$1"}],
            "extraction": {},
        },
    )
    monkeypatch.setattr(service, "_persist", lambda *_args, **_kwargs: None)

    response = service.handle_chat(_req(session_id, "what is the price of the second one?"))

    assert search_calls == [True]
    assert response.assistant_text == "Yes, that trailer is listed at $1."


def test_inventory_lookup_is_inactive_after_category_switch_qna_reset(monkeypatch):
    session_id = "00000000-0000-0000-0000-000000001118"
    service.reset_session(session_id)
    session = service._get_session(session_id)
    session["initial_contact_request_asked"] = True
    session["has_shown_search_results"] = False
    session["awaiting_slot"] = "haul_item"
    session["trailer_category"] = "Utility"
    search_calls = []

    monkeypatch.setattr(service, "create_or_get_soft_lead", lambda **kwargs: "00000000-0000-0000-0000-000000009118")
    monkeypatch.setattr(service, "update_lead_contact", lambda **kwargs: "00000000-0000-0000-0000-000000009118")
    monkeypatch.setattr(
        service,
        "search_trailers",
        lambda *_args, **_kwargs: search_calls.append(True) or {},
    )
    monkeypatch.setattr(service, "_is_confused_user_turn", lambda *_args, **_kwargs: (False, 0))
    monkeypatch.setattr(service, "_persist", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        service,
        "_invoke_graph",
        lambda *_args, **_kwargs: {
            "assistant_text": "What will you be hauling on the utility trailer?",
            "mind_decision": {"action": "respond"},
            "trailer_category": "Utility",
            "slots_collected": {},
            "slots_skipped": [],
            "metadata_filters_collected": {},
            "requested_non_metadata_features": [],
            "active_search_request_text": "",
            "make_category_options": [],
            "awaiting_slot": "haul_item",
            "pending_questions": [{"slot": "haul_item", "question": "What will you be hauling on the utility trailer?", "required": True}],
            "asked_questions": ["haul_item"],
            "already_shown_listing_urls": [],
            "last_listings": [],
            "tool_events": [],
            "pending_contact_action": None,
        },
    )

    response = service.handle_chat(_req(session_id, "2026 Aluma price?"))

    assert search_calls == []
    assert response.assistant_text == "What will you be hauling on the utility trailer?"


def test_year_make_lookup_is_blocked_once_new_search_flow_has_started(monkeypatch):
    session_id = "00000000-0000-0000-0000-000000001119"
    service.reset_session(session_id)
    session = service._get_session(session_id)
    session["initial_contact_request_asked"] = True
    session["trailer_category"] = "Utility"
    session["slots_collected"] = {"haul_item": "golf cart"}
    session["metadata_filters_collected"] = {"length_ft": "12 ft"}
    session["has_shown_search_results"] = False
    search_calls = []

    monkeypatch.setattr(service, "create_or_get_soft_lead", lambda **kwargs: "00000000-0000-0000-0000-000000009119")
    monkeypatch.setattr(service, "update_lead_contact", lambda **kwargs: "00000000-0000-0000-0000-000000009119")
    monkeypatch.setattr(
        service,
        "search_trailers",
        lambda *_args, **_kwargs: search_calls.append(True) or {},
    )
    monkeypatch.setattr(service, "_is_confused_user_turn", lambda *_args, **_kwargs: (False, 0))
    monkeypatch.setattr(service, "_persist", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        service,
        "_invoke_graph",
        lambda *_args, **_kwargs: {
            "assistant_text": "What kind of budget are you targeting?",
            "mind_decision": {"action": "respond"},
            "trailer_category": "Utility",
            "slots_collected": {"haul_item": "golf cart"},
            "slots_skipped": [],
            "metadata_filters_collected": {"length_ft": "12 ft"},
            "requested_non_metadata_features": [],
            "active_search_request_text": "",
            "make_category_options": [],
            "awaiting_slot": "max_price",
            "pending_questions": [{"slot": "max_price", "question": "What kind of budget are you targeting?", "required": True}],
            "asked_questions": ["max_price"],
            "already_shown_listing_urls": [],
            "last_listings": [],
            "tool_events": [],
            "pending_contact_action": None,
        },
    )

    response = service.handle_chat(_req(session_id, "2026 Aluma price?"))

    assert search_calls == []
    assert response.assistant_text == "What kind of budget are you targeting?"


def test_unsupported_business_action_routing_uses_structured_classifier(monkeypatch):
    session_id = "00000000-0000-0000-0000-000000001120"
    service.reset_session(session_id)
    session = service._get_session(session_id)
    session["initial_contact_request_asked"] = True

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(service, "_is_catalogue_overview_turn", lambda *_args, **_kwargs: False)

    class _UnsupportedActionRouter:
        def invoke(self, _messages):
            return service.UnsupportedBusinessActionRoutingDecision(
                should_route_graph=True,
                reason="customer requested a business action",
            )

    monkeypatch.setattr(service, "_unsupported_business_action_router_llm", lambda: _UnsupportedActionRouter())

    assert service._should_route_to_graph(session, "Please arrange the paperwork for me") is True


def test_contact_info_question_with_buying_intent_uses_faq_tool_not_generic_category(monkeypatch):
    session_id = "00000000-0000-0000-0000-000000001121"
    service.reset_session(session_id)
    session = service._get_session(session_id)
    session["initial_contact_request_asked"] = True
    session["customer_full_name"] = "Ibrahim"
    session["customer_phone"] = "03304388550"
    session["contact_status"] = "sufficient"
    email_calls = []

    monkeypatch.setattr(service, "create_or_get_soft_lead", lambda **kwargs: "00000000-0000-0000-0000-000000009121")
    monkeypatch.setattr(service, "update_lead_contact", lambda **kwargs: "00000000-0000-0000-0000-000000009121")
    monkeypatch.setattr(service, "_persist", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(service, "_is_confused_user_turn", lambda *_args, **_kwargs: (False, 0))
    monkeypatch.setattr(service, "_is_catalogue_overview_turn", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(service, "_is_unsupported_business_action_turn", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        graph,
        "_classify_non_recommendation_turn",
        lambda **_kwargs: graph.NonRecommendationTurnDecision(
            turn_type="contact_or_store_info",
            action="send_non_sales_faq_email",
            faq_category="contact_human",
            faq_summary="Customer asked how to contact TrailerPlace.",
            assistant_text="You can reach our team at 979-532-1486.",
            should_store_freeform_fields=True,
            confidence="high",
            reason="contact_question_with_buying_intent",
        ),
    )
    monkeypatch.setattr(
        graph,
        "send_non_sales_faq_email",
        lambda **kwargs: email_calls.append(kwargs) or {"status": "sent"},
    )

    response = service.handle_chat(
        _req(session_id, "I want to buy a trailer but I want to know first how to contact you guys")
    )

    assert email_calls
    assert email_calls[0]["faq_category"] == "contact_human"
    assert "What type of trailer are you looking for?" not in response.assistant_text
    assert "979-532-1486" in response.assistant_text


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


def test_recommendation_results_include_llm_interest_followup_without_contact_ask(monkeypatch):
    state = {
        "trailer_category": "Utility",
        "slots_collected": {},
        "metadata_filters_collected": {},
        "user_message": "show utility trailers",
        "already_shown_listing_urls": [],
        "tool_events": [],
        "customer_email": None,
        "customer_phone": None,
    }

    def _search(**_kwargs):
        return graph.PineconeListingSearchResult(
            listings=[{"title": "Trailer A", "url": "https://example.test/a", "length": "12 ft"}],
            query_text="show utility trailers",
            metadata_filter=None,
        )

    monkeypatch.setattr(graph, "search_pinecone_listing_result", _search)
    monkeypatch.setenv("WHY_IT_FITS_LLM_ENABLED", "0")
    monkeypatch.setattr(graph, "_pinecone_match_framing_text", lambda **_kwargs: ("", {}))
    monkeypatch.setattr(
        graph,
        "_result_interest_followup_text",
        lambda **_kwargs: "Do any of these trailers stand out to you?",
    )

    out = graph._pinecone_search_node(state)

    assert "Do any of these trailers stand out to you?" in out["assistant_text"]
    assert "phone number or email address" not in out["assistant_text"]


def test_recommendation_results_skip_followup_when_llm_interest_prompt_unavailable(monkeypatch):
    state = {
        "trailer_category": "Utility",
        "slots_collected": {},
        "metadata_filters_collected": {},
        "user_message": "show utility trailers",
        "already_shown_listing_urls": [],
        "tool_events": [],
    }

    monkeypatch.setattr(
        graph,
        "search_pinecone_listing_result",
        lambda **_kwargs: graph.PineconeListingSearchResult(
            listings=[{"title": "Trailer A", "url": "https://example.test/a", "length": "12 ft"}],
            query_text="show utility trailers",
            metadata_filter=None,
        ),
    )
    monkeypatch.setenv("WHY_IT_FITS_LLM_ENABLED", "0")
    monkeypatch.setattr(graph, "_pinecone_match_framing_text", lambda **_kwargs: ("", {}))
    monkeypatch.setattr(graph, "_result_interest_followup_text", lambda **_kwargs: "")

    out = graph._pinecone_search_node(state)

    assert "Trailer #1: [Trailer A](https://example.test/a)" in out["assistant_text"]
    assert "phone number or email address" not in out["assistant_text"]
    assert "stand out to you" not in out["assistant_text"]


def test_pinecone_search_uses_active_request_after_short_qualification(monkeypatch):
    state = {
        "trailer_category": "Livestock",
        "slots_collected": {"trailer_length_ft": "12"},
        "metadata_filters_collected": {"length_ft": "12"},
        "user_message": "12",
        "active_search_request_text": (
            "I am looking for a livestock trailer with offroad wheels and swinging gates"
            " | Current requirements: length 12"
        ),
        "already_shown_listing_urls": [],
        "tool_events": [],
    }
    captured = {}

    def _search(**kwargs):
        captured["search_user_message"] = kwargs["user_message"]
        return graph.PineconeListingSearchResult(
            listings=[{"title": "Trailer A", "url": "https://example.test/a", "length": "12 ft"}],
            query_text=kwargs["user_message"],
            metadata_filter={"category": {"$eq": "Livestock"}},
        )

    def _framing(**kwargs):
        captured["framing_user_message"] = kwargs["user_message"]
        captured["latest_user_message"] = kwargs["latest_user_message"]
        return "", {}

    def _format(_listings, **kwargs):
        captured["format_user_message"] = kwargs["user_message"]
        return "formatted cards"

    monkeypatch.setattr(graph, "search_pinecone_listing_result", _search)
    monkeypatch.setattr(graph, "_pinecone_match_framing_text", _framing)
    monkeypatch.setattr(graph, "format_listing_results", _format)
    monkeypatch.setattr(graph, "_result_interest_followup_text", lambda **_kwargs: "")

    out = graph._pinecone_search_node(state)

    assert out["assistant_text"] == "formatted cards"
    for key in ("search_user_message", "framing_user_message", "format_user_message"):
        assert "offroad wheels" in captured[key]
        assert "swinging gates" in captured[key]
        assert "length 12" in captured[key]
        assert captured[key] != "12"
    assert captured["latest_user_message"] == "12"


def test_pinecone_results_prepend_llm_match_framing_and_hide_evidence(monkeypatch):
    state = {
        "trailer_category": "Livestock",
        "slots_collected": {},
        "metadata_filters_collected": {},
        "user_message": "show me livestock trailers with sliding gates",
        "already_shown_listing_urls": [],
        "tool_events": [],
    }
    captured = {}

    monkeypatch.setattr(
        graph,
        "search_pinecone_listing_result",
        lambda **_kwargs: graph.PineconeListingSearchResult(
            listings=[
                {
                    "title": "Sliding Gate Trailer",
                    "url": "https://example.test/sliding",
                    "length": "16 ft",
                    "match_evidence_text": "Details: includes sliding gate and livestock divider.",
                }
            ],
            query_text="show me livestock trailers with sliding gates | Category: Livestock",
            metadata_filter={"category": {"$eq": "Livestock"}},
            rerank_debug={"applied": True},
            make_debug={"applied": True},
        ),
    )

    def _framing(**kwargs):
        captured["evidence"] = kwargs["search_result"].listings[0]["match_evidence_text"]
        return (
            "I found 1 option that fully matches your request, with strong alternatives ready if you want to compare.",
            {"overall_match_level": "full", "full_match_count": 1, "source": "llm"},
        )

    monkeypatch.setenv("WHY_IT_FITS_LLM_ENABLED", "0")
    monkeypatch.setattr(graph, "_pinecone_match_framing_text", _framing)
    monkeypatch.setattr(graph, "_result_interest_followup_text", lambda **_kwargs: "")

    out = graph._pinecone_search_node(state)

    assert out["assistant_text"].startswith("I found 1 option that fully matches your request")
    assert "Trailer #1: [Sliding Gate Trailer](https://example.test/sliding)" in out["assistant_text"]
    assert captured["evidence"] == "Details: includes sliding gate and livestock divider."
    assert "match_evidence_text" not in out["last_listings"][0]


def test_pinecone_results_use_llm_no_exact_framing(monkeypatch):
    state = {
        "trailer_category": "Livestock",
        "slots_collected": {},
        "metadata_filters_collected": {},
        "user_message": "show me livestock trailers with sliding gates",
        "already_shown_listing_urls": [],
        "tool_events": [],
    }
    monkeypatch.setattr(
        graph,
        "search_pinecone_listing_result",
        lambda **_kwargs: graph.PineconeListingSearchResult(
            listings=[
                {
                    "title": "Close Alternative Trailer",
                    "url": "https://example.test/alt",
                    "length": "16 ft",
                    "match_evidence_text": "Details: livestock trailer with rear gate.",
                }
            ],
            query_text="show me livestock trailers with sliding gates | Category: Livestock",
            metadata_filter={"category": {"$eq": "Livestock"}},
        ),
    )
    monkeypatch.setenv("WHY_IT_FITS_LLM_ENABLED", "0")
    monkeypatch.setattr(
        graph,
        "_pinecone_match_framing_text",
        lambda **_kwargs: (
            "The exact sliding-gate combination is not clearly shown, so I selected the closest available livestock options.",
            {"overall_match_level": "no_exact", "full_match_count": 0, "source": "llm"},
        ),
    )
    monkeypatch.setattr(graph, "_result_interest_followup_text", lambda **_kwargs: "")

    out = graph._pinecone_search_node(state)

    assert out["assistant_text"].startswith("The exact sliding-gate combination is not clearly shown")
    assert "Trailer #1: [Close Alternative Trailer](https://example.test/alt)" in out["assistant_text"]
    assert out["tool_events"][-1]["match_analysis"]["overall_match_level"] == "no_exact"


def test_pinecone_match_framing_disabled_uses_neutral_fallback(monkeypatch):
    state = {
        "trailer_category": "Utility",
        "slots_collected": {},
        "metadata_filters_collected": {},
        "user_message": "show me utility trailers with mesh sides",
        "already_shown_listing_urls": [],
        "tool_events": [],
    }
    monkeypatch.setattr(
        graph,
        "search_pinecone_listing_result",
        lambda **_kwargs: graph.PineconeListingSearchResult(
            listings=[
                {
                    "title": "Utility Trailer",
                    "url": "https://example.test/utility",
                    "length": "12 ft",
                    "match_evidence_text": "Details: utility trailer.",
                }
            ],
            query_text="show me utility trailers with mesh sides | Category: Utility",
            metadata_filter={"category": {"$eq": "Utility"}},
        ),
    )
    monkeypatch.setenv("PINECONE_MATCH_FRAMING_LLM_ENABLED", "0")
    monkeypatch.setenv("WHY_IT_FITS_LLM_ENABLED", "0")
    monkeypatch.setattr(graph, "_result_interest_followup_text", lambda **_kwargs: "")

    out = graph._pinecone_search_node(state)

    assert out["assistant_text"].startswith("Here are the strongest available options")
    assert "fully match" not in out["assistant_text"].lower()
    assert "Trailer #1: [Utility Trailer](https://example.test/utility)" in out["assistant_text"]


class _PineconeValidationLLM:
    def __init__(self, decision: graph.PineconeMatchFramingDecision, captured: dict | None = None):
        self.decision = decision
        self.captured = captured

    def invoke(self, messages):
        if self.captured is not None:
            self.captured["system"] = messages[0].content
            self.captured["human"] = messages[1].content
        return self.decision


class _PineconeIntroLLM:
    def __init__(self, intro_text: str, captured: dict | None = None):
        self.decision = graph.PineconeSalesIntroDecision(intro_text=intro_text)
        self.captured = captured

    def invoke(self, messages):
        if self.captured is not None:
            self.captured["intro_system"] = messages[0].content
            self.captured["intro_human"] = messages[1].content
        return self.decision


def test_pinecone_validation_reorders_full_match_before_alternatives(monkeypatch):
    state = {
        "trailer_category": "Livestock",
        "slots_collected": {"trailer_length_ft": "12"},
        "metadata_filters_collected": {"length_ft": "12"},
        "user_message": "I need a 12 ft livestock trailer with a swing slide gate",
        "already_shown_listing_urls": [],
        "tool_events": [],
    }
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("WHY_IT_FITS_LLM_ENABLED", "0")
    monkeypatch.setattr(graph, "_result_interest_followup_text", lambda **_kwargs: "")
    monkeypatch.setattr(graph, "_extract_requested_non_metadata_features", lambda **_kwargs: ["swing slide gate"])
    monkeypatch.setattr(
        graph,
        "search_pinecone_listing_result",
        lambda **_kwargs: graph.PineconeListingSearchResult(
            listings=[
                {
                    "title": "Close Livestock Trailer",
                    "url": "https://example.test/alt",
                    "category": "Livestock",
                    "length": "16 ft",
                    "match_evidence_text": "Livestock trailer with rear gate.",
                },
                {
                    "title": "12 Ft Swing Slide Livestock Trailer",
                    "url": "https://example.test/full",
                    "category": "Livestock",
                    "length": "12 ft",
                    "match_evidence_text": "12 ft livestock trailer with swing slide gate.",
                },
            ],
            query_text="I need a 12 ft livestock trailer with a swing slide gate | Category: Livestock",
            metadata_filter={"category": {"$eq": "Livestock"}},
            rerank_debug={"applied": True, "required_length_ft": 12.0},
        ),
    )
    monkeypatch.setattr(
        graph,
        "_pinecone_match_audit_llm",
        lambda: _PineconeValidationLLM(
            graph.PineconeMatchFramingDecision(
                intro_text="I found 1 option that fully matches your swing slide gate request, followed by another strong livestock option to compare.",
                overall_match_level="mixed",
                requested_non_metadata_features=["swing slide gate"],
                per_listing_match=[
                    graph.PineconeListingMatchDecision(
                        position=1,
                        match_level="alternative",
                        confirmed_requirements=["livestock"],
                        missing_or_unconfirmed_requirements=["12 ft", "swing slide gate"],
                        sales_blurb="This livestock trailer is a strong option to compare, with practical cattle-hauling utility and confirmed specs above.",
                    ),
                    graph.PineconeListingMatchDecision(
                        position=2,
                        match_level="full",
                        confirmed_requirements=["12 ft", "livestock", "swing slide gate"],
                        sales_blurb="This one lines up especially well, with the swing slide gate clearly shown in the listing details.",
                    ),
                ],
            )
        ),
    )
    monkeypatch.setattr(
        graph,
        "_pinecone_sales_intro_llm",
        lambda: _PineconeIntroLLM(
            "I found 1 option that fully matches your swing slide gate request, followed by another strong livestock option to compare."
        ),
    )

    out = graph._pinecone_search_node(state)

    assert out["assistant_text"].startswith("I found 1 option that fully matches")
    assert out["assistant_text"].index("Trailer #1: [12 Ft Swing Slide Livestock Trailer]") < out["assistant_text"].index(
        "Trailer #2: [Close Livestock Trailer]"
    )
    assert out["tool_events"][-1]["match_analysis"]["full_match_count"] == 1
    assert out["tool_events"][-1]["match_analysis"]["alternative_count"] == 1
    assert out["last_listings"][0]["url"] == "https://example.test/full"
    assert "match_evidence_text" not in out["last_listings"][0]


def test_pinecone_validation_prompt_requires_strict_feature_concept_matching(monkeypatch):
    captured = {}
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr(graph, "_extract_requested_non_metadata_features", lambda **_kwargs: ["sliding gates"])
    monkeypatch.setattr(
        graph,
        "_pinecone_match_audit_llm",
        lambda: _PineconeValidationLLM(
            graph.PineconeMatchFramingDecision(
                intro_text="The exact sliding-gate setup is not clearly shown, so I selected the strongest livestock option to compare.",
                overall_match_level="no_exact",
                requested_non_metadata_features=["sliding gates"],
                per_listing_match=[
                    graph.PineconeListingMatchDecision(
                        position=1,
                        match_level="alternative",
                        confirmed_requirements=["livestock", "butterfly gates"],
                        missing_or_unconfirmed_requirements=["sliding gates"],
                    )
                ],
            ),
            captured,
        ),
    )
    monkeypatch.setattr(
        graph,
        "_pinecone_sales_intro_llm",
        lambda: _PineconeIntroLLM(
            "The exact sliding-gate setup is not clearly shown, so I selected the strongest livestock option to compare."
        ),
    )
    result = graph.PineconeListingSearchResult(
        listings=[
            {
                "title": "Livestock Trailer W/ Butterfly Gates",
                "url": "https://example.test/butterfly",
                "category": "Livestock",
                "match_evidence_text": "Livestock trailer with butterfly gates.",
            }
        ],
        query_text="livestock trailer with sliding gates",
        metadata_filter={"category": {"$eq": "Livestock"}},
    )

    graph._pinecone_match_framing_text(
        user_message="livestock trailer with sliding gates",
        category="Livestock",
        slots={},
        metadata_filters={},
        search_result=result,
    )

    system_prompt = captured["system"].lower()
    assert "match requested non-metadata features by exact feature concept" in system_prompt
    assert "length, width, payload capacity, and gvwr alone do not make a full match" in system_prompt
    assert "butterfly gate is not the same requested feature as sliding gate" in system_prompt
    assert "missing_or_unconfirmed_requirements must include requested features" in system_prompt


def test_pinecone_validation_no_exact_does_not_claim_requested_feature(monkeypatch):
    state = {
        "trailer_category": "Livestock",
        "slots_collected": {"trailer_length_ft": "12"},
        "metadata_filters_collected": {"length_ft": "12"},
        "user_message": "I need a 12 ft livestock trailer with a swing slide gate",
        "already_shown_listing_urls": [],
        "tool_events": [],
    }
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("WHY_IT_FITS_LLM_ENABLED", "0")
    monkeypatch.setattr(graph, "_result_interest_followup_text", lambda **_kwargs: "")
    monkeypatch.setattr(graph, "_extract_requested_non_metadata_features", lambda **_kwargs: ["swing slide gate"])
    monkeypatch.setattr(
        graph,
        "search_pinecone_listing_result",
        lambda **_kwargs: graph.PineconeListingSearchResult(
            listings=[
                {
                    "title": "12 Ft Livestock Trailer",
                    "url": "https://example.test/alt",
                    "category": "Livestock",
                    "length": "12 ft",
                    "match_evidence_text": "12 ft livestock trailer with rear gate.",
                }
            ],
            query_text="I need a 12 ft livestock trailer with a swing slide gate | Category: Livestock",
            metadata_filter={"category": {"$eq": "Livestock"}},
        ),
    )
    monkeypatch.setattr(
        graph,
        "_pinecone_match_audit_llm",
        lambda: _PineconeValidationLLM(
            graph.PineconeMatchFramingDecision(
                intro_text="The exact swing slide gate combination is not clearly shown right now, so I selected the strongest livestock option worth comparing.",
                overall_match_level="no_exact",
                requested_non_metadata_features=["swing slide gate"],
                per_listing_match=[
                    graph.PineconeListingMatchDecision(
                        position=1,
                        match_level="alternative",
                        confirmed_requirements=["12 ft", "livestock"],
                        missing_or_unconfirmed_requirements=["swing slide gate"],
                        sales_blurb="This livestock trailer is a strong option to compare, with practical hauling utility and confirmed specs above.",
                    )
                ],
            )
        ),
    )
    monkeypatch.setattr(
        graph,
        "_pinecone_sales_intro_llm",
        lambda: _PineconeIntroLLM(
            "The exact swing slide gate combination is not clearly shown right now, so I selected the strongest livestock option worth comparing."
        ),
    )

    out = graph._pinecone_search_node(state)

    assert out["assistant_text"].startswith("The exact swing slide gate combination is not clearly shown")
    card_text = out["assistant_text"].split("Trailer #1:", 1)[1]
    assert "features a convenient swing slide gate" not in card_text.lower()
    assert "strong option to compare" in card_text
    assert "partial match" not in card_text.lower()
    assert "close alternative" not in card_text.lower()
    assert out["tool_events"][-1]["match_analysis"]["overall_match_level"] == "no_exact"


def test_pinecone_validation_blocks_full_match_intro_when_no_full_matches(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr(graph, "_extract_requested_non_metadata_features", lambda **_kwargs: ["swing slide gate"])
    monkeypatch.setattr(
        graph,
        "_pinecone_match_audit_llm",
        lambda: _PineconeValidationLLM(
            graph.PineconeMatchFramingDecision(
                intro_text="All of these fully match your swing slide gate request.",
                requested_non_metadata_features=["swing slide gate"],
                per_listing_match=[
                    graph.PineconeListingMatchDecision(
                        position=1,
                        match_level="alternative",
                        confirmed_requirements=["livestock"],
                        missing_or_unconfirmed_requirements=["swing slide gate"],
                    )
                ],
            )
        ),
    )
    monkeypatch.setattr(
        graph,
        "_pinecone_sales_intro_llm",
        lambda: _PineconeIntroLLM("All of these fully match your swing slide gate request."),
    )
    result = graph.PineconeListingSearchResult(
        listings=[
            {
                "title": "Livestock Trailer",
                "url": "https://example.test/alt",
                "category": "Livestock",
                "match_evidence_text": "Livestock trailer with rear gate.",
            }
        ],
        query_text="livestock trailer with swing slide gate",
        metadata_filter={"category": {"$eq": "Livestock"}},
    )

    intro, analysis = graph._pinecone_match_framing_text(
        user_message="livestock trailer with swing slide gate",
        category="Livestock",
        slots={},
        metadata_filters={},
        search_result=result,
    )

    assert intro == "Here are the strongest available options I found based on your search."
    assert analysis["source"] == "neutral_fallback_invalid_match_claim"
    assert analysis["full_match_count"] == 0


def test_pinecone_validation_blocks_strong_match_intro_when_no_full_matches(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr(
        graph,
        "_extract_requested_non_metadata_features",
        lambda **_kwargs: ["offroad wheels", "sliding gates"],
    )
    monkeypatch.setattr(
        graph,
        "_pinecone_match_audit_llm",
        lambda: _PineconeValidationLLM(
            graph.PineconeMatchFramingDecision(
                intro_text="I found a few strong matches for your requested combination and included additional relevant trailers worth comparing.",
                requested_non_metadata_features=["offroad wheels", "sliding gates"],
                per_listing_match=[
                    graph.PineconeListingMatchDecision(
                        position=1,
                        match_level="alternative",
                        confirmed_requirements=["12 ft", "livestock"],
                        missing_or_unconfirmed_requirements=["offroad wheels", "sliding gates"],
                    )
                ],
            )
        ),
    )
    monkeypatch.setattr(
        graph,
        "_pinecone_sales_intro_llm",
        lambda: _PineconeIntroLLM(
            "I found a few strong matches for your requested combination and included additional relevant trailers worth comparing."
        ),
    )
    result = graph.PineconeListingSearchResult(
        listings=[
            {
                "title": "12 Ft Livestock Trailer",
                "url": "https://example.test/alt",
                "category": "Livestock",
                "length": "12 ft",
                "match_evidence_text": "12 ft livestock trailer with rear gate.",
            }
        ],
        query_text="12 ft livestock trailer with offroad wheels and sliding gates",
        metadata_filter={"category": {"$eq": "Livestock"}},
    )

    intro, analysis = graph._pinecone_match_framing_text(
        user_message="12 ft livestock trailer with offroad wheels and sliding gates",
        category="Livestock",
        slots={"trailer_length_ft": "12"},
        metadata_filters={"length_ft": "12"},
        search_result=result,
    )

    assert intro == "Here are the strongest available options I found based on your search."
    assert analysis["source"] == "neutral_fallback_invalid_match_claim"
    assert analysis["full_match_count"] == 0


def test_pinecone_validation_demotes_full_when_requested_features_not_confirmed(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr(
        graph,
        "_extract_requested_non_metadata_features",
        lambda **_kwargs: ["offroad wheels", "sliding gates"],
    )
    monkeypatch.setattr(
        graph,
        "_pinecone_match_audit_llm",
        lambda: _PineconeValidationLLM(
            graph.PineconeMatchFramingDecision(
                intro_text="An exact match is not currently shown, but I selected the strongest available livestock option to compare.",
                requested_non_metadata_features=["offroad wheels", "sliding gates"],
                per_listing_match=[
                    graph.PineconeListingMatchDecision(
                        position=1,
                        match_level="full",
                        confirmed_requirements=["12 ft", "livestock"],
                        missing_or_unconfirmed_requirements=[],
                    )
                ],
            )
        ),
    )
    monkeypatch.setattr(
        graph,
        "_pinecone_sales_intro_llm",
        lambda: _PineconeIntroLLM(
            "An exact match is not currently shown, but I selected the strongest available livestock option to compare."
        ),
    )
    result = graph.PineconeListingSearchResult(
        listings=[
            {
                "title": "12 Ft Livestock Trailer",
                "url": "https://example.test/alt",
                "category": "Livestock",
                "length": "12 ft",
                "match_evidence_text": "12 ft livestock trailer with rear gate.",
            }
        ],
        query_text="12 ft livestock trailer with offroad wheels and sliding gates",
        metadata_filter={"category": {"$eq": "Livestock"}},
    )

    _intro, analysis = graph._pinecone_match_framing_text(
        user_message="12 ft livestock trailer with offroad wheels and sliding gates",
        category="Livestock",
        slots={"trailer_length_ft": "12"},
        metadata_filters={"length_ft": "12"},
        search_result=result,
    )

    listing_match = analysis["per_listing_match"][0]
    assert analysis["full_match_count"] == 0
    assert analysis["overall_match_level"] == "partial_only"
    assert listing_match["match_level"] == "partial"
    assert "offroad wheels" in listing_match["missing_or_unconfirmed_requirements"]
    assert "sliding gates" in listing_match["missing_or_unconfirmed_requirements"]


def test_requested_feature_extractor_is_authoritative_for_hallucinated_gate_requirement(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr(graph, "_extract_requested_non_metadata_features", lambda **_kwargs: [])
    monkeypatch.setattr(
        graph,
        "_pinecone_match_audit_llm",
        lambda: _PineconeValidationLLM(
            graph.PineconeMatchFramingDecision(
                intro_text="While the exact requested combination is not currently shown, I have some excellent trailers available for you.",
                overall_match_level="partial_only",
                requested_non_metadata_features=["sliding gates"],
                per_listing_match=[
                    graph.PineconeListingMatchDecision(
                        position=1,
                        match_level="partial",
                        confirmed_requirements=["Length: 24 ft 0 in", "Hitch Type: Gooseneck"],
                        missing_or_unconfirmed_requirements=["sliding gates"],
                    )
                ],
            )
        ),
    )
    monkeypatch.setattr(
        graph,
        "_pinecone_sales_intro_llm",
        lambda: _PineconeIntroLLM(
            "Confirmed livestock fits are shown first, followed by relevant options worth comparing."
        ),
    )
    result = graph.PineconeListingSearchResult(
        listings=[
            {
                "title": "24 Ft Cattle Trailer W/ Swing Slide Gate",
                "url": "https://example.test/livestock-24",
                "category": "Livestock",
                "length": "24 ft",
                "hitch_type": "Gooseneck",
                "match_evidence_text": "24 ft cattle trailer with swing slide gate.",
            }
        ],
        query_text="I am also looking for a 24' cattle trailer | Current requirements: length 24 | Category: Livestock",
        metadata_filter={"category": {"$eq": "Livestock"}},
    )

    intro, analysis = graph._pinecone_match_framing_text(
        user_message="I am also looking for a 24' cattle trailer | Current requirements: length 24 | Category: Livestock",
        latest_user_message="I am also looking for a 24' cattle trailer",
        category="Livestock",
        slots={"trailer_length_ft": "24 feet"},
        metadata_filters={"length_ft": "24"},
        search_result=result,
    )

    assert analysis["requested_non_metadata_features"] == []
    assert analysis["per_listing_match"][0]["missing_or_unconfirmed_requirements"] == []
    assert analysis["full_match_count"] == 1
    assert analysis["overall_match_level"] == "full"
    assert intro.startswith("Confirmed livestock fits are shown first")


def test_requested_feature_extractor_failure_falls_back_to_empty_requested_features(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    def _boom(**_kwargs):
        raise RuntimeError("extractor unavailable")

    monkeypatch.setattr(graph, "_extract_requested_non_metadata_features", _boom)
    monkeypatch.setattr(
        graph,
        "_pinecone_match_audit_llm",
        lambda: _PineconeValidationLLM(
            graph.PineconeMatchFramingDecision(
                requested_non_metadata_features=["sliding gates"],
                per_listing_match=[
                    graph.PineconeListingMatchDecision(
                        position=1,
                        match_level="partial",
                        confirmed_requirements=["Length: 24 ft 0 in"],
                        missing_or_unconfirmed_requirements=["sliding gates"],
                    )
                ],
            )
        ),
    )
    monkeypatch.setattr(
        graph,
        "_pinecone_sales_intro_llm",
        lambda: _PineconeIntroLLM("Confirmed livestock fits are shown first, followed by relevant options worth comparing."),
    )
    result = graph.PineconeListingSearchResult(
        listings=[
            {
                "title": "24 Ft Cattle Trailer",
                "url": "https://example.test/livestock-24",
                "category": "Livestock",
                "length": "24 ft",
                "match_evidence_text": "24 ft cattle trailer with rear gate.",
            }
        ],
        query_text="I am also looking for a 24' cattle trailer",
        metadata_filter={"category": {"$eq": "Livestock"}},
    )

    _intro, analysis = graph._pinecone_match_framing_text(
        user_message="I am also looking for a 24' cattle trailer",
        latest_user_message="I am also looking for a 24' cattle trailer",
        category="Livestock",
        slots={"trailer_length_ft": "24 feet"},
        metadata_filters={"length_ft": "24"},
        search_result=result,
    )

    assert analysis["requested_non_metadata_features"] == []
    assert analysis["per_listing_match"][0]["missing_or_unconfirmed_requirements"] == []
    assert analysis["full_match_count"] == 1


def test_requested_feature_extractor_preserves_explicit_feature_request(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr(
        graph,
        "_extract_requested_non_metadata_features",
        lambda **_kwargs: ["swing slide gate"],
    )
    monkeypatch.setattr(
        graph,
        "_pinecone_match_audit_llm",
        lambda: _PineconeValidationLLM(
            graph.PineconeMatchFramingDecision(
                overall_match_level="mixed",
                requested_non_metadata_features=["swing slide gate"],
                per_listing_match=[
                    graph.PineconeListingMatchDecision(
                        position=1,
                        match_level="full",
                        confirmed_requirements=["livestock", "swing slide gate"],
                        missing_or_unconfirmed_requirements=[],
                    )
                ],
            )
        ),
    )
    monkeypatch.setattr(
        graph,
        "_pinecone_sales_intro_llm",
        lambda: _PineconeIntroLLM(
            "Confirmed fits are shown first, followed by relevant options worth comparing."
        ),
    )
    result = graph.PineconeListingSearchResult(
        listings=[
            {
                "title": "24 Ft Cattle Trailer W/ Swing Slide Gate",
                "url": "https://example.test/livestock-24",
                "category": "Livestock",
                "match_evidence_text": "24 ft cattle trailer with swing slide gate.",
            }
        ],
        query_text="I am also looking for a 24' cattle trailer with a swing slide gate",
        metadata_filter={"category": {"$eq": "Livestock"}},
    )

    _intro, analysis = graph._pinecone_match_framing_text(
        user_message="I am also looking for a 24' cattle trailer with a swing slide gate",
        latest_user_message="I am also looking for a 24' cattle trailer with a swing slide gate",
        category="Livestock",
        slots={"trailer_length_ft": "24 feet"},
        metadata_filters={"length_ft": "24"},
        search_result=result,
    )

    assert analysis["requested_non_metadata_features"] == ["swing slide gate"]
    assert analysis["full_match_count"] == 1
    assert analysis["per_listing_match"][0]["match_level"] == "full"


def test_pinecone_validation_blocks_negative_structured_mismatch_intro(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr(
        graph,
        "_pinecone_match_audit_llm",
        lambda: _PineconeValidationLLM(
            graph.PineconeMatchFramingDecision(
                intro_text="These are useful options, but each exceeds your 12 ft requirement.",
                requested_non_metadata_features=[],
                per_listing_match=[
                    graph.PineconeListingMatchDecision(
                        position=1,
                        match_level="alternative",
                        confirmed_requirements=["livestock"],
                        missing_or_unconfirmed_requirements=["12 ft"],
                    )
                ],
            )
        ),
    )
    monkeypatch.setattr(
        graph,
        "_pinecone_sales_intro_llm",
        lambda: _PineconeIntroLLM("These are useful options, but each exceeds your 12 ft requirement."),
    )
    result = graph.PineconeListingSearchResult(
        listings=[
            {
                "title": "Livestock Trailer",
                "url": "https://example.test/alt",
                "category": "Livestock",
                "length": "16 ft",
                "match_evidence_text": "Livestock trailer with rear gate.",
            }
        ],
        query_text="12 ft livestock trailer",
        metadata_filter={"category": {"$eq": "Livestock"}},
    )

    intro, analysis = graph._pinecone_match_framing_text(
        user_message="12 ft livestock trailer",
        category="Livestock",
        slots={"trailer_length_ft": "12"},
        metadata_filters={"length_ft": "12"},
        search_result=result,
    )

    assert intro == "Here are the strongest available options I found based on your search."
    assert analysis["source"] == "neutral_fallback_invalid_match_claim"


def test_result_interest_followup_rejects_meet_needs_for_alternatives(monkeypatch):
    class _BadFollowupLLM:
        def invoke(self, _messages):
            return type(
                "_Response",
                (),
                {"content": "Does any of these trailers meet your needs?"},
            )()

    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr(graph, "_result_interest_followup_llm", lambda: _BadFollowupLLM())

    text = graph._result_interest_followup_text(
        user_message="I need a 12 ft livestock trailer with a swing slide gate",
        category="Livestock",
        slots={"trailer_length_ft": "12"},
        listings=[
            {
                "title": "Livestock Trailer",
                "url": "https://example.test/alt",
                "length": "16 ft",
                "match_validation": {"match_level": "alternative"},
            }
        ],
    )

    assert text == ""


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


def test_active_recommendation_qna_stays_in_graph_even_if_overview_classifier_fires(monkeypatch):
    session = service._new_session("overview-during-qna")
    session["trailer_category"] = None
    session["awaiting_slot"] = "generic_category_choice"
    session["pending_questions"] = [{"slot": "trailer_length_ft", "question": "What trailer length would you prefer?"}]
    session["messages"] = [
        {"role": "user", "content": "I want to know what trailers do you guys offer?"},
        {"role": "assistant", "content": "What type of trailer are you looking for?"},
        {"role": "user", "content": "what are the options?"},
    ]

    class _OverviewClassifier:
        def invoke(self, messages):
            assert "active qualification question" in messages[0].content
            return service.CatalogueOverviewDecision(
                is_catalogue_overview=True,
                reason="User asks for broad trailer options, not a specific recommendation.",
            )

    monkeypatch.setattr(service, "_catalogue_overview_llm", lambda: _OverviewClassifier())

    assert service._should_route_to_graph(session, "what are the options?") is True


def test_active_qna_answer_stays_in_graph_even_if_overview_classifier_fires(monkeypatch):
    session = service._new_session("active-answer-during-qna")
    session["trailer_category"] = "Dump"
    session["awaiting_slot"] = "haul_material"
    session["pending_questions"] = [
        {"slot": "haul_weight_lbs", "question": "What's the rough haul weight per load?"}
    ]
    session["messages"] = [
        {"role": "user", "content": "I need a dump trailer"},
        {"role": "assistant", "content": "What material will you be hauling (dirt, gravel, debris, etc.)?"},
        {"role": "user", "content": "construction debris"},
    ]

    class _OverviewClassifier:
        def invoke(self, _messages):
            return service.CatalogueOverviewDecision(
                is_catalogue_overview=True,
                reason="Incorrect broad overview classification.",
            )

    monkeypatch.setattr(service, "_catalogue_overview_llm", lambda: _OverviewClassifier())

    assert service._should_route_to_graph(session, "construction debris") is True


def test_specific_recommendation_request_still_routes_with_catalogue_classifier(monkeypatch):
    session = service._new_session("specific-search-route")

    class _OverviewClassifier:
        def invoke(self, _messages):
            return service.CatalogueOverviewDecision(
                is_catalogue_overview=False,
                reason="User gives a specific trailer shopping request.",
            )

    monkeypatch.setattr(service, "_catalogue_overview_llm", lambda: _OverviewClassifier())

    assert service._should_route_to_graph(session, "show me 12 ft livestock trailers") is True


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
