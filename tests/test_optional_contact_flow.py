from __future__ import annotations

import json

from src.chatbot import graph, service
from src.models import ChatRequest


def _req(session_id: str, message: str) -> ChatRequest:
    return ChatRequest(session_id=session_id, message=message)


def _decision(action: str, remaining_message: str | None = None) -> service.ContactPromptReplyDecision:
    return service.ContactPromptReplyDecision(action=action, remaining_message=remaining_message)


def test_partial_contact_gets_one_followup_then_resumes_saved_request(monkeypatch):
    session_id = "00000000-0000-0000-0000-000000001198"
    service.reset_session(session_id)
    session = service._get_session(session_id)
    session.update(
        initial_contact_request_asked=True,
        awaiting_initial_contact_reply=True,
        pending_initial_user_message="I am looking for a dump trailer",
    )
    monkeypatch.setattr(service, "create_or_get_soft_lead", lambda **_kwargs: "lead")
    monkeypatch.setattr(service, "update_lead_contact", lambda **_kwargs: "lead")
    monkeypatch.setattr(
        service,
        "_extract_contact",
        lambda text, *_args: {
            "full_name": "Ibrahim" if "Ibrahim" in text else None,
            "email": None,
            "phone": None,
            "name_confidence": "high" if "Ibrahim" in text else "none",
        },
    )
    monkeypatch.setattr(
        service,
        "_classify_contact_prompt_reply",
        lambda _session, text: _decision(
            "acknowledge_contact_details" if "Ibrahim" in text else "resume_saved_request"
        ),
    )
    monkeypatch.setattr(service, "_contact_prompt_bridge_text", lambda **_kwargs: "Okay, let's continue.")
    monkeypatch.setattr(service, "_is_confused_user_turn", lambda *_args: (False, 0))
    monkeypatch.setattr(service, "_should_route_to_graph", lambda *_args: True)
    monkeypatch.setattr(
        service,
        "_invoke_graph",
        lambda _session, message, _shown, **_kwargs: {
            "assistant_text": f"Continuing: {message}",
            "tool_events": [],
            "last_listings": [],
        },
    )

    first = service.handle_chat(_req(session_id, "my name is Ibrahim"))
    second = service.handle_chat(_req(session_id, "I'd rather not share that"))

    assert "email address or phone number" in first.assistant_text
    assert "Before we get started" not in first.assistant_text
    assert second.assistant_text.endswith("Continuing: I am looking for a dump trailer")
    assert service._get_session(session_id)["customer_full_name"] == "Ibrahim"


def test_contact_plus_message_continues_with_clean_routing_context(monkeypatch):
    session_id = "00000000-0000-0000-0000-000000001099"
    service.reset_session(session_id)
    session = service._get_session(session_id)
    session["initial_contact_request_asked"] = True
    captured = {}
    monkeypatch.setattr(service, "create_or_get_soft_lead", lambda **_kwargs: "lead")
    monkeypatch.setattr(service, "update_lead_contact", lambda **_kwargs: "lead")
    monkeypatch.setattr(service, "_extract_contact", lambda *_args, **_kwargs: {
        "full_name": "Frank Garvey", "email": "frank@example.com", "phone": "4092677859",
        "name_confidence": "high",
    })
    monkeypatch.setattr(
        service, "_classify_contact_prompt_reply",
        lambda *_args, **_kwargs: _decision("acknowledge_and_continue", "Do you offer financing?"),
    )
    monkeypatch.setattr(service, "_is_confused_user_turn", lambda *_args: (False, 0))
    monkeypatch.setattr(service, "_should_route_to_graph", lambda *_args: True)
    def invoke(context, message, _shown, **_kwargs):
        captured.update(message=message, messages=context["messages"])
        return {"assistant_text": "Yes, financing is available.", "tool_events": [], "last_listings": []}
    monkeypatch.setattr(service, "_invoke_graph", invoke)

    original = "Frank Garvey, frank@example.com, 4092677859. Do you offer financing?"
    response = service.handle_chat(_req(session_id, original))

    assert response.assistant_text == f"{service._CONTACT_CONTINUE_ACK}\n\nYes, financing is available."
    assert captured["message"] == "Do you offer financing?"
    assert captured["messages"][-1]["content"] == "Do you offer financing?"
    assert service._get_session(session_id)["messages"][0]["content"] == original


def test_contact_refusal_policy_and_authoritative_category_resolution(monkeypatch):
    session = service._new_session("policy-category")
    session["contact_status"] = "contact_declined"

    # The contact-response policy is now enforced purely by the deterministic
    # regex guard in _enforce_contact_response_policy (the old validator/rewriter
    # LLMs were removed as dead code). A prompt that asks for email/phone while
    # contact is declined is rewritten to the active question.
    cleaned = service._enforce_contact_response_policy(
        session,
        "Please provide your email or phone number.",
        "What will you be hauling on the utility trailer?",
    )
    assert cleaned == "What will you be hauling on the utility trailer?"
    service._sync_contact_status(session)
    assert session["contact_status"] == "contact_declined"
    session["pending_contact_action"] = {"type": "faq"}
    assert service._enforce_contact_response_policy(
        session, "Please provide your email."
    ) == "Please provide your email."
    session["customer_full_name"] = "Ibrahim"
    session["customer_email"] = "ibrahim@example.com"
    service._sync_contact_status(session)
    assert session["contact_status"] == "contact_available"

    class _Mind:
        def invoke(self, _messages):
            return graph.MindDecision(
                action="send_non_sales_faq_email",
                assistant_text="We offer financing.",
                faq_category="financing",
            )

    monkeypatch.setattr(graph, "_mind_llm", lambda: _Mind())
    out = graph._mind_node({
        "user_message": "Do you finance utility trailers?",
        "messages": [],
        "slots_collected": {},
        "metadata_filters_collected": {},
    })
    decision = out["mind_decision"]
    assert decision["trailer_category"] == "Utility"
    assert decision["category_resolution_kind"] == "explicit"
    assert decision["category_confidence"] == "high"
    assert decision["action"] == "send_non_sales_faq_email"


def test_meaningful_initial_message_resumes_after_contact_refusal(monkeypatch):
    session_id = "00000000-0000-0000-0000-000000001100"
    service.reset_session(session_id)

    class _Preserver:
        def invoke(self, _messages):
            return service.InitialMessagePreservationDecision(
                has_meaningful_non_contact_intent=True
            )

    monkeypatch.setattr(service, "_initial_message_preservation_llm", lambda: _Preserver())
    monkeypatch.setattr(service, "create_or_get_soft_lead", lambda **_kwargs: "lead")
    monkeypatch.setattr(service, "update_lead_contact", lambda **_kwargs: "lead")
    monkeypatch.setattr(service, "_extract_contact", lambda *_args, **_kwargs: {
        "full_name": "Ibrahim", "email": None, "phone": None, "name_confidence": "high"
    })
    monkeypatch.setattr(
        service, "_classify_contact_prompt_reply",
        lambda *_args, **_kwargs: _decision("decline_contact_details"),
    )
    monkeypatch.setattr(service, "_contact_prompt_bridge_text", _bridge)
    monkeypatch.setattr(service, "_is_confused_user_turn", lambda *_args: (False, 0))
    monkeypatch.setattr(service, "_should_route_to_graph", lambda *_args: True)
    monkeypatch.setattr(service, "_enforce_contact_response_policy", lambda _s, text, *_a: text)
    monkeypatch.setattr(service, "_invoke_graph", lambda _s, message, _shown, **_kwargs: {
        "assistant_text": f"Answering: {message}", "tool_events": [], "last_listings": []
    })

    original = "Hello, what are the use cases for trailers? My name is Ibrahim"
    service.handle_chat(_req(session_id, original))
    response = service.handle_chat(_req(session_id, "no"))

    assert response.assistant_text == f"Bridge[resume_saved_request]\n\nAnswering: {original}"
    assert response.contact_status == "contact_declined"


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


def test_simple_catalogue_question_resumes_after_contact(monkeypatch):
    session_id = "00000000-0000-0000-0000-000000001022"
    service.reset_session(session_id)
    invoked = []

    monkeypatch.setattr(service, "create_or_get_soft_lead", lambda **kwargs: "00000000-0000-0000-0000-000000009022")
    monkeypatch.setattr(service, "update_lead_contact", lambda **kwargs: "00000000-0000-0000-0000-000000009022")
    monkeypatch.setattr(service, "_classify_contact_prompt_reply", lambda *_args, **_kwargs: _decision("resume_saved_request"))
    monkeypatch.setattr(service, "_contact_prompt_bridge_text", _bridge)
    monkeypatch.setattr(service, "_is_confused_user_turn", lambda *_args, **_kwargs: (False, 0))
    monkeypatch.setattr(service, "_should_route_to_graph", lambda _session, message: invoked.append(message) or True)
    monkeypatch.setattr(
        service,
        "_invoke_graph",
        lambda _session, message, _shown, **_kwargs: {
            "assistant_text": "TrailerPlace carries utility, dump, equipment, enclosed, flatbed, car hauler, livestock, tilt, roll-off, and more.",
            "tool_events": [],
            "last_listings": [],
        },
    )

    first = service.handle_chat(_req(session_id, "Hello. What type of trailers do you guys have?"))
    second = service.handle_chat(_req(session_id, "my name is Ibrahim and phone is 03304388550"))

    assert "Before we get started" in first.assistant_text
    assert second.assistant_text.startswith("Bridge[resume_saved_request]\n\nTrailerPlace carries")
    assert invoked[-1] == "Hello. What type of trailers do you guys have?"
    assert service._get_session(session_id)["pending_initial_user_message"] is None


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
        lambda _session, message, _shown, **_kwargs: {
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
    monkeypatch.setattr(
        service,
        "_extract_contact",
        lambda message, *_args, **_kwargs: {
            "full_name": None,
            "email": None,
            "phone": "25" if "25ft" in message else None,
            "name_confidence": "none",
        },
    )
    monkeypatch.setattr(service, "_is_confused_user_turn", lambda *_args, **_kwargs: (False, 0))
    monkeypatch.setattr(service, "_should_route_to_graph", lambda _session, message: invoked.append(message) or True)
    monkeypatch.setattr(
        service,
        "_invoke_graph",
        lambda _session, message, _shown, **_kwargs: {
            "assistant_text": f"Continuing with: {message}",
            "tool_events": [],
            "last_listings": [],
        },
    )

    service.handle_chat(_req(session_id, "I need a 6x12 utility trailer"))
    response = service.handle_chat(_req(session_id, "The trailer should be a 25ft livestock"))

    assert response.assistant_text == "Continuing with: The trailer should be a 25ft livestock"
    assert invoked[-1] == "The trailer should be a 25ft livestock"
    assert service._get_session(session_id)["customer_phone"] is None
    assert service._get_session(session_id)["pending_initial_user_message"] is None


def test_contact_ignored_requirement_refines_saved_request(monkeypatch):
    session_id = "00000000-0000-0000-0000-000000001105"
    service.reset_session(session_id)
    invoked = []
    monkeypatch.setattr(service, "create_or_get_soft_lead", lambda **_kwargs: "lead")
    monkeypatch.setattr(service, "update_lead_contact", lambda **_kwargs: "lead")
    monkeypatch.setattr(
        service,
        "_classify_contact_prompt_reply",
        lambda *_args, **_kwargs: _decision("resume_saved_request_with_update"),
    )
    monkeypatch.setattr(service, "_contact_prompt_bridge_text", _bridge)
    monkeypatch.setattr(service, "_is_confused_user_turn", lambda *_args, **_kwargs: (False, 0))
    monkeypatch.setattr(service, "_should_route_to_graph", lambda _session, message: invoked.append(message) or True)
    monkeypatch.setattr(
        service,
        "_invoke_graph",
        lambda _session, message, _shown, **_kwargs: {
            "assistant_text": f"Continuing with: {message}",
            "tool_events": [],
            "last_listings": [],
        },
    )

    service.handle_chat(_req(session_id, "Hello. I am looking for a livestock trailer"))
    service.handle_chat(_req(session_id, "it should be of 20ft length"))

    assert invoked[-1] == (
        "Hello. I am looking for a livestock trailer"
        " | Additional requirement: it should be of 20ft length"
    )


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
        lambda _session, message, _shown, **_kwargs: {
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
        lambda _session, message, _shown, **_kwargs: {
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

    def _invoke_graph(session, message, _shown, **_kwargs):
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


def test_contact_resume_and_selective_category_filter_carryover(monkeypatch):
    session_id = "00000000-0000-0000-0000-000000001024"
    service.reset_session(session_id)
    resumed_messages = []
    original_request = (
        "I need a 20 ft livestock trailer with a 7 ft width, "
        "7000 lbs payload, gooseneck hitch, and sliding gates"
    )

    monkeypatch.setattr(
        service,
        "create_or_get_soft_lead",
        lambda **kwargs: "00000000-0000-0000-0000-000000009024",
    )
    monkeypatch.setattr(
        service,
        "update_lead_contact",
        lambda **kwargs: "00000000-0000-0000-0000-000000009024",
    )
    monkeypatch.setattr(
        service,
        "_classify_contact_prompt_reply",
        lambda *_args, **_kwargs: _decision("resume_saved_request"),
    )
    monkeypatch.setattr(service, "_contact_prompt_bridge_text", _bridge)
    monkeypatch.setattr(service, "_is_confused_user_turn", lambda *_args, **_kwargs: (False, 0))
    monkeypatch.setattr(service, "_should_route_to_graph", lambda *_args, **_kwargs: True)

    def _capture_resumed_request(_session, message, _shown, **_kwargs):
        resumed_messages.append(message)
        return {
            "assistant_text": "Continuing your trailer search.",
            "tool_events": [],
            "last_listings": [],
        }

    monkeypatch.setattr(service, "_invoke_graph", _capture_resumed_request)

    service.handle_chat(_req(session_id, original_request))
    service.handle_chat(_req(session_id, "Ibrahim, 03304388550"))

    assert resumed_messages[-1] == original_request

    class _StaleHistoryExtractor:
        def invoke(self, _messages):
            return graph.FieldExtractionAdjudicationDecision(
                metadata_filters_update={
                    "length_ft": "20 ft",
                    "width_ft": "7 ft",
                    "payload_lbs": "7000 lbs",
                    "hitch_type": "Gooseneck",
                },
                slots_collected_update={"trailer_length_ft": "20 ft"},
                requested_non_metadata_features=["sliding gates"],
                confidence="high",
            )

    class _SelectiveConfirmation:
        def invoke(self, _messages):
            return graph.CategoryFilterConfirmationDecision(
                keep_fields=["length_ft"],
                discard_fields=["payload_lbs"],
                updates={"hitch_type": "Bumper Pull"},
                resolved=True,
                confidence="high",
            )

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        graph,
        "_field_extraction_adjudicator_llm",
        lambda: _StaleHistoryExtractor(),
    )
    monkeypatch.setattr(
        graph,
        "_category_filter_confirmation_llm",
        lambda: _SelectiveConfirmation(),
    )
    monkeypatch.setattr(
        graph,
        "classify_haul_requirements",
        lambda **kwargs: graph.HaulClassificationDecision(),
    )
    monkeypatch.setattr(
        graph,
        "classify_no_preference",
        lambda **kwargs: graph.PreferenceNullDecision(),
    )

    state = {
        "session_id": session_id,
        "user_message": "I need a utility trailer",
        "active_search_request_text": original_request,
        "messages": [
            {"role": "user", "content": original_request},
            {"role": "assistant", "content": "Here are matching livestock trailers."},
            {"role": "user", "content": "I need a utility trailer"},
        ],
        "mind_decision": {
            "action": "respond",
            "trailer_category": "Utility",
            "slots_collected_update": {},
            "metadata_filters_update": {},
        },
        "trailer_category": "Livestock",
        "slots_collected": {"trailer_length_ft": "20 ft"},
        "slots_skipped": [],
        "metadata_filters_collected": {
            "length_ft": "20 ft",
            "width_ft": "7 ft",
            "payload_lbs": "7000 lbs",
            "hitch_type": "Gooseneck",
        },
        "defaulted_metadata_filters": [],
        "requested_non_metadata_features": ["sliding gates"],
        "pending_questions": [],
        "asked_questions": [],
        "already_shown_listing_urls": ["https://example.test/livestock"],
        "last_listings": [{"title": "Livestock result"}],
        "tool_events": [],
        "has_shown_search_results": True,
    }

    changed = graph._apply_mind_node(state)

    assert changed["metadata_filters_collected"] == {}
    assert changed["requested_non_metadata_features"] == []
    assert set(changed["pending_category_change"]["carry_filters"]) == {
        "length_ft",
        "width_ft",
        "payload_lbs",
        "hitch_type",
    }

    changed["user_message"] = (
        "Keep the length, discard the payload, and change the hitch to bumper pull"
    )
    changed["mind_decision"] = {"action": "respond", "trailer_category": "Utility"}
    selective = graph._apply_mind_node(changed)

    assert selective["metadata_filters_collected"] == {
        "length_ft": "20 ft",
        "hitch_type": "Bumper Pull",
    }
    assert selective["pending_category_change"]["carry_filters"] == {"width_ft": "7 ft"}
    assert "width" in selective["assistant_text"].lower()

    selective["user_message"] = "no"
    selective["mind_decision"] = {"action": "respond", "trailer_category": "Utility"}
    final = graph._apply_mind_node(selective)

    assert final["pending_category_change"] is None
    assert final["metadata_filters_collected"] == {
        "length_ft": "20 ft",
        "hitch_type": "Bumper Pull",
    }
    assert final["requested_non_metadata_features"] == []


def test_category_clarification_state_survives_after_contact_resume(monkeypatch):
    session_id = "00000000-0000-0000-0000-000000001022"
    service.reset_session(session_id)

    monkeypatch.setattr(service, "create_or_get_soft_lead", lambda **kwargs: "00000000-0000-0000-0000-000000009022")
    monkeypatch.setattr(service, "update_lead_contact", lambda **kwargs: "00000000-0000-0000-0000-000000009022")
    monkeypatch.setattr(service, "_classify_contact_prompt_reply", lambda *_args, **_kwargs: _decision("resume_saved_request"))
    monkeypatch.setattr(service, "_contact_prompt_bridge_text", _bridge)
    monkeypatch.setattr(service, "_is_confused_user_turn", lambda *_args, **_kwargs: (False, 0))

    def _invoke_graph(session, message, _shown, **_kwargs):
        if message == "I am looking for an office trailer":
            return {
                "assistant_text": "Will this be for fiber/telecom work specifically, or a more general office trailer?",
                "tool_events": [],
                "last_listings": [],
                "trailer_category": None,
                "category_needs_clarification": True,
                "category_clarification_key": "office_trailer_use",
                "awaiting_slot": "category_clarification",
                "pending_questions": [],
            }
        assert session["category_needs_clarification"] is True
        assert session["category_clarification_key"] == "office_trailer_use"
        assert session["awaiting_slot"] == "category_clarification"
        return {
            "assistant_text": "What will you be using the enclosed trailer for?",
            "tool_events": [],
            "last_listings": [],
            "trailer_category": "Enclosed",
            "category_needs_clarification": False,
            "category_clarification_key": None,
            "awaiting_slot": "use_case",
            "pending_questions": [{"slot": "use_case", "question": "What will you be using the enclosed trailer for?"}],
        }

    monkeypatch.setattr(service, "_should_route_to_graph", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(service, "_invoke_graph", _invoke_graph)

    first = service.handle_chat(_req(session_id, "I am looking for an office trailer"))
    second = service.handle_chat(_req(session_id, "my name is Ibrahim and phone is 03304388550"))
    third = service.handle_chat(_req(session_id, "it'll be for office only"))

    assert "Before we get started" in first.assistant_text
    assert "Will this be for fiber/telecom work specifically" in second.assistant_text
    assert third.assistant_text == "What will you be using the enclosed trailer for?"
    assert service._get_session(session_id)["trailer_category"] == "Enclosed"
    assert service._get_session(session_id)["category_needs_clarification"] is False
    assert service._get_session(session_id)["category_clarification_key"] is None


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
    )

    system_prompt = captured["system"]
    assert "do not ask any trailer-search or qualification question" in system_prompt
    assert "Never ask what type" in system_prompt
    assert "Vary the wording naturally" in system_prompt
    assert "whenever you're ready" in system_prompt
    assert "never thank them for sharing contact details" in system_prompt
    assert "without guilt or pressure" in system_prompt
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


def test_contact_extraction_prompt_covers_standalone_contact_values(monkeypatch):
    captured = {}

    class _CaptureLLM:
        def invoke(self, messages):
            captured["system"] = messages[0].content
            captured["human"] = messages[1].content
            return service.ContactExtraction()

    monkeypatch.setattr(service, "_contact_llm", lambda: _CaptureLLM())
    service._extract_contact(
        "ibrahim",
        {
            "customer_full_name": None,
            "customer_email": "mk@gmail.com",
            "customer_phone": None,
        },
    )

    prompt = captured["system"].lower()
    assert "standalone alphabetic personal name" in prompt
    assert "standalone phone number" in prompt
    assert "standalone email address" in prompt
    assert "03304388550" in prompt
    assert "ibrahim@example.com" in prompt
    assert "email='mk@gmail.com'" in captured["human"]


def test_contact_reply_prompt_treats_standalone_values_as_contact_replies(monkeypatch):
    captured = {}

    class _CaptureLLM:
        def invoke(self, messages):
            captured["system"] = messages[0].content
            return service.ContactPromptReplyDecision(action="resume_saved_request")

    monkeypatch.setattr(service, "_contact_prompt_reply_llm", lambda: _CaptureLLM())
    session = service._new_session("00000000-0000-0000-0000-000000001130")
    session["pending_initial_user_message"] = "I need a utility trailer"

    service._classify_contact_prompt_reply(session, "ibrahim")

    prompt = captured["system"].lower()
    assert "immediately preceding contact request" in prompt
    assert "'ibrahim'" in prompt
    assert "'03304388550'" in prompt
    assert "'ibrahim@example.com'" in prompt


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
        lambda _session, message, _shown, **_kwargs: {
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
    assert (response.thinking_context or {}).get("tool_events") == [
        {"tool": "send_interested_listing_email", "result": {"status": "sent"}}
    ]
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


def test_validated_inventory_lookup_overrides_post_results_qna(monkeypatch):
    session_id = "00000000-0000-0000-0000-000000001118"
    service.reset_session(session_id)
    session = service._get_session(session_id)
    session.update(
        {
            "initial_contact_request_asked": True,
            "has_shown_search_results": True,
            "trailer_category": "Dump",
            "awaiting_slot": "haul_weight_lbs",
            "pending_questions": [{"slot": "haul_weight_lbs"}],
            "last_listings": [{"title": "Old Dump"}],
        }
    )
    validated = object()
    captured = {}
    monkeypatch.setattr(service, "is_potential_direct_inventory_lookup", lambda _text: True)
    monkeypatch.setattr(service, "validated_direct_inventory_extraction", lambda _text: validated)
    monkeypatch.setattr(
        service,
        "search_trailers",
        lambda *_args, **kwargs: captured.update(kwargs) or {
            "reply": "I found matching Diamond C FMAX trailers.",
            "entity_type": "MODEL_SEARCH",
            "confidence": 1.0,
            "top_matches": [{"title": "Diamond C FMAX"}],
            "extraction": {"should_lookup": True},
        },
    )
    monkeypatch.setattr(service, "_persist", lambda *_args, **_kwargs: None)

    response = service._inventory_lookup_response(
        session,
        _req(session_id, "Is Diamond C FMAX available?"),
        "Is Diamond C FMAX available?",
    )

    assert response is not None
    assert captured["extraction"] is validated
    assert session["awaiting_slot"] == "haul_weight_lbs"
    assert session["pending_questions"] == [{"slot": "haul_weight_lbs"}]


def test_inventory_batch_becomes_reference_set_and_second_interest_bypasses_graph(monkeypatch):
    session_id = "00000000-0000-0000-0000-000000001129"
    service.reset_session(session_id)
    session = service._get_session(session_id)
    session.update(
        {
            "initial_contact_request_asked": True,
            "has_shown_search_results": True,
            "trailer_category": "Dump",
            "metadata_filters_collected": {"hitch_type": "Gooseneck"},
            "customer_full_name": "Ibrahim",
            "customer_email": "ibrahim@example.com",
            "contact_status": "contact_available",
            "last_listings": [{"title": "Old Dump", "url": "https://example.test/dump"}],
            "already_shown_listing_urls": ["https://example.test/dump"],
        }
    )
    fmax = [
        {"title": "FMAX One", "url": "https://example.test/fmax-1"},
        {"title": "FMAX Two", "url": "https://example.test/fmax-2"},
    ]
    monkeypatch.setattr(service, "is_potential_direct_inventory_lookup", lambda _text: True)
    monkeypatch.setattr(service, "validated_direct_inventory_extraction", lambda _text: object())
    monkeypatch.setattr(
        service,
        "search_trailers",
        lambda *_args, **_kwargs: {
            "reply": "Trailer #1: FMAX One\nTrailer #2: FMAX Two",
            "entity_type": "MODEL_SEARCH",
            "confidence": 1.0,
            "top_matches": fmax,
            "extraction": {},
        },
    )
    monkeypatch.setattr(service, "_persist", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        service,
        "_resolve_listing_reference",
        lambda _session, message: (
            service.ListingReferenceDecision()
            if "availability" in message
            else service.ListingReferenceDecision(
                is_listing_selection=True,
                has_explicit_listing_reference=True,
                reference_intent="interest",
                selected_index=2,
                selected_title="FMAX Two",
                selected_url="https://example.test/fmax-2",
                confidence="high",
            )
        ),
    )
    sent = []
    monkeypatch.setattr(
        service,
        "send_interested_listing_email",
        lambda **kwargs: sent.append(kwargs) or {"status": "queued"},
    )
    monkeypatch.setattr(
        service,
        "_persist_email_transcript_snapshot",
        lambda *_args, **_kwargs: None,
    )

    first_request = _req(session_id, "I want to know the availability of Diamond C FMAX")
    first = service._inventory_lookup_response(session, first_request, first_request.message)
    assert first is not None
    assert session["last_listings"] == fmax
    assert set(session["already_shown_listing_urls"]) == {
        "https://example.test/dump",
        "https://example.test/fmax-1",
        "https://example.test/fmax-2",
    }

    second_request = _req(session_id, "I like the 2nd one")
    second = service._inventory_lookup_response(session, second_request, second_request.message)
    assert second is not None
    assert sent[0]["item_name"] == "FMAX Two"
    assert second.thinking_context["tool_events"] == [
        {
            "tool": "send_interested_listing_email",
            "result": {"status": "queued"},
        }
    ]
    assert session["trailer_category"] == "Dump"
    assert session["metadata_filters_collected"] == {"hitch_type": "Gooseneck"}


def test_new_category_request_is_not_a_listing_selection(monkeypatch):
    class _NoReferenceGate:
        def invoke(self, _messages):
            return service.ListingReferenceIntentDecision(
                reason="New category shopping request has no explicit listing reference."
            )

    class _WrongSelectionLLM:
        def invoke(self, _messages):
            return service.ListingReferenceDecision(
                is_listing_selection=True,
                has_explicit_listing_reference=False,
                reference_intent="interest",
                selected_index=3,
                selected_title="Equipment Three",
                selected_url="https://example.test/equipment-3",
                confidence="medium",
                reason="User is shopping for another trailer category, not selecting a shown item.",
            )

    monkeypatch.setattr(
        service, "_listing_reference_intent_llm", lambda: _NoReferenceGate()
    )
    monkeypatch.setattr(service, "_listing_reference_llm", lambda: _WrongSelectionLLM())
    decision = service._resolve_listing_reference(
        {
            "trailer_category": "Equipment",
            "messages": [{"role": "assistant", "content": "Do any of these trailers interest you?"}],
            "last_listings": [
                {"title": "Equipment One", "url": "https://example.test/equipment-1"},
                {"title": "Equipment Two", "url": "https://example.test/equipment-2"},
                {"title": "Equipment Three", "url": "https://example.test/equipment-3"},
            ],
        },
        "I am looking for a car hauler as well",
    )

    assert decision.is_listing_selection is False
    assert decision.reference_intent == "none"
    assert decision.selected_index is None


def test_listing_reference_gate_rejects_unrelated_turn_after_interest(monkeypatch):
    class _Gate:
        def invoke(self, messages):
            latest = json.loads(messages[-1].content)["latest_message"]
            explicit = latest == "I am interested in the first trailer."
            return service.ListingReferenceIntentDecision(
                has_explicit_listing_reference=explicit,
                reference_kind="ordinal" if explicit else "none",
                reference_intent="interest" if explicit else "none",
                confidence="high",
                reason="current message only",
            )

    class _Resolver:
        calls = 0

        def invoke(self, _messages):
            self.calls += 1
            return service.ListingReferenceDecision(
                is_listing_selection=True,
                has_explicit_listing_reference=True,
                reference_intent="interest",
                selected_index=1,
                selected_title="Trailer One",
                selected_url="https://example.test/one",
                confidence="high",
            )

    resolver = _Resolver()
    monkeypatch.setattr(service, "_listing_reference_intent_llm", lambda: _Gate())
    monkeypatch.setattr(service, "_listing_reference_llm", lambda: resolver)
    session = {
        "trailer_category": "Equipment",
        "messages": [
            {"role": "assistant", "content": "Your interest in Trailer One has been logged."}
        ],
        "last_listings": [
            {"title": "Trailer One", "url": "https://example.test/one"}
        ],
    }

    selected = service._resolve_listing_reference(
        session, "I am interested in the first trailer."
    )
    unrelated = service._resolve_listing_reference(
        session, "What are equipment trailers commonly used for?"
    )

    assert selected.is_listing_selection is True
    assert unrelated.is_listing_selection is False
    assert resolver.calls == 1


def test_inventory_results_send_silent_notification_with_completed_turn(monkeypatch):
    session_id = "00000000-0000-0000-0000-000000001127"
    service.reset_session(session_id)
    session = service._get_session(session_id)
    session.update(
        {
            "initial_contact_request_asked": True,
            "has_shown_search_results": True,
            "customer_full_name": "Test User",
            "customer_phone": "979-555-1111",
            "contact_status": "contact_available",
            "last_listings": [
                {"title": "Utility A", "stock_number": "11111"},
                {"title": "Utility B", "stock_number": "22222"},
            ],
        }
    )
    calls = []

    monkeypatch.setattr(
        service,
        "search_trailers",
        lambda *_args, **_kwargs: {
            "reply": "Trailer #1: Utility A",
            "entity_type": "GENERAL",
            "confidence": 1.0,
            "top_matches": [{"title": "Utility A"}],
            "extraction": {},
        },
    )
    monkeypatch.setattr(service, "_persist", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        service,
        "_persist_email_transcript_snapshot",
        lambda current: calls.append(("snapshot", current["messages"][-1]["content"])),
    )
    monkeypatch.setattr(
        service,
        "send_trailer_results_shown_email",
        lambda **kwargs: calls.append(("email", kwargs)) or {"status": "sent"},
    )

    request = _req(session_id, "what is the price of the second one?")
    session["messages"].append({"role": "user", "content": request.message})
    response = service._inventory_lookup_response(session, request, request.message)

    assert response is not None
    assert response.assistant_text == "Trailer #1: Utility A"
    assert calls[0] == ("snapshot", "Trailer #1: Utility A")
    assert calls[1][0] == "email"


def test_pinecone_results_defer_until_full_contact_and_send_latest_silently(monkeypatch):
    session_id = "00000000-0000-0000-0000-000000001128"
    service.reset_session(session_id)
    session = service._get_session(session_id)
    session["initial_contact_request_asked"] = True
    calls = []

    monkeypatch.setattr(service, "create_or_get_soft_lead", lambda **kwargs: "lead")
    monkeypatch.setattr(service, "update_lead_contact", lambda **kwargs: "lead")
    monkeypatch.setattr(service, "_is_confused_user_turn", lambda *_args, **_kwargs: (False, 0))
    monkeypatch.setattr(service, "_should_route_to_graph", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(service, "_persist", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(service, "_persist_email_transcript_snapshot", lambda *_args: None)
    monkeypatch.setattr(
        service,
        "send_trailer_results_shown_email",
        lambda **kwargs: calls.append(kwargs) or {"status": "sent"},
    )
    result_number = {"value": 0}

    def _graph(*_args, **_kwargs):
        result_number["value"] += 1
        number = result_number["value"]
        return {
            "assistant_text": f"Result batch {number}",
            "mind_decision": {"action": "pinecone_search"},
            "tool_events": [{"tool": "pinecone_search", "result_count": 1}],
            "last_listings": [{"title": f"Trailer {number}"}],
        }

    monkeypatch.setattr(service, "_invoke_graph", _graph)

    first = service.handle_chat(_req(session_id, "show utility trailers"))
    second = service.handle_chat(_req(session_id, "show more"))

    assert first.assistant_text == "Result batch 1"
    assert second.assistant_text == "Result batch 2"
    assert calls == []
    assert session["pending_results_notification"] is True

    monkeypatch.setattr(service, "_should_route_to_graph", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(service, "_main_smalltalk_response", lambda *_args, **_kwargs: "Contact saved.")
    third = service.handle_chat(_req(session_id, "I am Alex and my phone is 979-555-2222"))

    assert "saved" in third.assistant_text.lower()
    assert "email" not in third.assistant_text.lower()
    assert len(calls) == 1
    assert calls[0]["full_name"] == "Alex"
    assert session["pending_results_notification"] is False


def test_results_email_failure_does_not_block_pinecone_response(monkeypatch):
    session_id = "00000000-0000-0000-0000-000000001129"
    service.reset_session(session_id)
    session = service._get_session(session_id)
    session.update(
        {
            "initial_contact_request_asked": True,
            "customer_full_name": "Test User",
            "customer_phone": "979-555-1111",
            "contact_status": "contact_available",
        }
    )

    monkeypatch.setattr(service, "create_or_get_soft_lead", lambda **kwargs: "lead")
    monkeypatch.setattr(service, "_is_confused_user_turn", lambda *_args, **_kwargs: (False, 0))
    monkeypatch.setattr(service, "_should_route_to_graph", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(service, "_persist", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(service, "_persist_email_transcript_snapshot", lambda *_args: None)
    monkeypatch.setattr(
        service,
        "send_trailer_results_shown_email",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("email unavailable")),
    )
    monkeypatch.setattr(
        service,
        "_invoke_graph",
        lambda *_args, **_kwargs: {
            "assistant_text": "Trailer #1: Dump A",
            "mind_decision": {"action": "pinecone_search"},
            "tool_events": [{"tool": "pinecone_search", "result_count": 1}],
            "last_listings": [{"title": "Dump A"}],
        },
    )

    response = service.handle_chat(_req(session_id, "show dump trailers"))

    assert response.assistant_text == "Trailer #1: Dump A"


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
    monkeypatch.setattr(
        service, "_route_decision",
        lambda *_args, **_kwargs: service.RoutingDecision(route="escalation", reason="business action"),
    )

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
    monkeypatch.setattr(service, "_route_decision", lambda *_args, **_kwargs: service.RoutingDecision(route="other"))
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
        lambda _session, message, _shown, **_kwargs: {
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
        lambda _session, message, _shown, **_kwargs: {
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
    assert out["tool_events"][-1]["overall_match_level"] == "no_exact"


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


def test_auditor_input_hides_dimensions_and_weights(monkeypatch):
    captured = {}
    listing = {
        "title": "2026 24 ft Livestock Trailer - 15181",
        "category": "Livestock",
        "price": "$23,995",
        "length": "24 ft",
        "payload_capacity": "10,170 lbs",
        "match_evidence_text": "Length: 24 ft | Payload Capacity: 10,170 lbs | Sliding gate included",
    }
    result = graph.PineconeListingSearchResult(
        listings=[listing],
        query_text="20 ft livestock trailer with 5,000 lbs payload and sliding gate",
        metadata_filter={"length_ft_num": {"$gte": 20}, "category": {"$eq": "Livestock"}},
        rerank_debug={"required_length_ft": 20, "required_payload_lbs": 5000},
    )
    monkeypatch.setattr(
        graph,
        "_pinecone_match_audit_llm",
        lambda: _PineconeValidationLLM(
            graph.PineconeMatchFramingDecision(
                intro_text="Confirmed feature fit.",
                per_listing_match=[graph.PineconeListingMatchDecision(position=1, match_level="full")],
            ),
            captured,
        ),
    )

    graph._pinecone_match_audit(
        user_message="I need a 20 ft livestock trailer with 5,000 lbs payload and sliding gate",
        latest_user_message=None,
        category="Livestock",
        slots={"trailer_length_ft": "20", "payload_need": "5000 lbs"},
        metadata_filters={"length_ft": "20", "payload_lbs": "5000", "color": "Black"},
        search_result=result,
        facts=graph._pinecone_listing_facts([listing]),
        requested_non_metadata_features=["sliding gate"],
    )

    human = captured["human"].lower()
    for hidden in ("20 ft", "24 ft", "5,000 lbs", "10,170 lbs", "length_ft", "payload_lbs", "rerank_debug"):
        assert hidden not in human
    assert "sliding gate" in human
    assert "$23,995" in human
    assert "2026" in human


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
    out = graph._pinecone_search_node(state)

    assert out["assistant_text"].startswith("I found 1 option that fully matches")
    assert out["assistant_text"].index("Trailer #1: [12 Ft Swing Slide Livestock Trailer]") < out["assistant_text"].index(
        "Trailer #2: [Close Livestock Trailer]"
    )
    assert out["tool_events"][-1]["full_match_count"] == 1
    assert out["tool_events"][-1]["alternative_count"] == 1
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
    out = graph._pinecone_search_node(state)

    assert out["assistant_text"].startswith("The exact swing slide gate combination is not clearly shown")
    card_text = out["assistant_text"].split("Trailer #1:", 1)[1]
    assert "features a convenient swing slide gate" not in card_text.lower()
    assert "strong option to compare" in card_text
    assert "partial match" not in card_text.lower()
    assert "close alternative" not in card_text.lower()
    assert out["tool_events"][-1]["overall_match_level"] == "no_exact"


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
    assert intro == "Here are the strongest available options I found based on your search."


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


def test_result_interest_followup_is_deterministic():
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

    assert text == "Do any of these trailers interest you?"


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

    monkeypatch.setattr(service, "_route_decision", lambda *_a, **_k: service.RoutingDecision(route="catalogue_overview"))

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

    monkeypatch.setattr(service, "_route_decision", lambda *_a, **_k: service.RoutingDecision(route="catalogue_overview"))

    assert service._should_route_to_graph(session, "construction debris") is True


def test_specific_recommendation_request_still_routes_with_catalogue_classifier(monkeypatch):
    session = service._new_session("specific-search-route")

    monkeypatch.setattr(service, "_route_decision", lambda *_a, **_k: service.RoutingDecision(route="other"))

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

    # Smalltalk now builds its client through the central make_llm factory
    # (E2) rather than constructing ChatOpenAI directly.
    monkeypatch.setattr(service, "make_llm", lambda *_args, **_kwargs: _OverviewLLM())

    response = service._main_smalltalk_response(session, "what services do you guys offer?")

    assert response == "LLM overview response"
    assert "utility, dump, equipment" in captured["system"]
    assert "bumper pull" in captured["system"]
    assert "gooseneck" in captured["system"]
    assert "financing" in captured["system"]
    assert "trade-ins" in captured["system"]
    assert "service or spare parts" in captured["system"]
    assert "Do not mention rentals" in captured["system"]


def test_missing_contact_phrase_names_only_absent_fields():
    assert (
        service._missing_contact_phrase({})
        == "your name and either an email address or phone number"
    )
    assert (
        service._missing_contact_phrase({"customer_full_name": "Ibrahim"})
        == "either an email address or phone number"
    )
    assert service._missing_contact_phrase({"customer_email": "a@b.com"}) == "your name"
    assert service._missing_contact_phrase({"customer_phone": "555"}) == "your name"
    assert (
        service._missing_contact_phrase(
            {"customer_full_name": "Ibrahim", "customer_email": "a@b.com"}
        )
        == ""
    )
