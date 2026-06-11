from src.chatbot.tools import email_tools


def test_interested_listing_email_promotes_lead_to_hard(monkeypatch):
    calls = []
    monkeypatch.setattr(email_tools, "send_ticket_notification", lambda **kwargs: calls.append(("send", kwargs)))
    monkeypatch.setattr(
        email_tools,
        "promote_lead_to_hard",
        lambda session_id: calls.append(("promote", session_id)),
    )
    monkeypatch.setattr(
        email_tools,
        "update_lead_item_of_interest",
        lambda session_id, item: calls.append(("interest", session_id, item)),
    )

    result = email_tools.send_interested_listing_email(
        session_id="session-123",
        full_name="Test User",
        email="test@example.com",
        phone="979-555-1111",
        item_name="Trailer A",
    )

    assert result["status"] == "sent"
    assert ("promote", "session-123") in calls
    assert ("interest", "session-123", "Trailer A") in calls


def test_non_sales_faq_email_promotes_lead_to_hard(monkeypatch):
    calls = []
    monkeypatch.setattr(email_tools, "send_faq_email_sync", lambda **kwargs: calls.append(("send", kwargs)))
    monkeypatch.setattr(
        email_tools,
        "promote_lead_to_hard",
        lambda session_id: calls.append(("promote", session_id)),
    )
    monkeypatch.setattr(
        email_tools,
        "update_lead_item_of_interest",
        lambda session_id, item: calls.append(("interest", session_id, item)),
    )

    result = email_tools.send_non_sales_faq_email(
        session_id="session-456",
        full_name="Test User",
        email=None,
        phone="979-555-1111",
        faq_category="financing",
        summary="Customer asked about financing.",
        user_message="Can you help me with financing?",
    )

    assert result["status"] == "sent"
    assert ("promote", "session-456") in calls
    assert ("interest", "session-456", "Finance Query") in calls
    send_call = calls[0][1]
    assert "Customer asked about financing." in send_call["summary_line"]
    assert "Last user message: Can you help me with financing?" in send_call["summary_line"]


def test_non_sales_faq_email_sets_item_of_interest_by_category(monkeypatch):
    calls = []
    monkeypatch.setattr(email_tools, "send_faq_email_sync", lambda **kwargs: calls.append(("send", kwargs)))
    monkeypatch.setattr(
        email_tools,
        "promote_lead_to_hard",
        lambda session_id: calls.append(("promote", session_id)),
    )
    monkeypatch.setattr(
        email_tools,
        "update_lead_item_of_interest",
        lambda session_id, item: calls.append(("interest", session_id, item)),
    )

    email_tools.send_non_sales_faq_email(
        session_id="session-parts",
        full_name="Test User",
        email=None,
        phone="979-555-1111",
        faq_category="service_parts",
    )
    email_tools.send_non_sales_faq_email(
        session_id="session-human",
        full_name="Test User",
        email=None,
        phone="979-555-1111",
        faq_category="contact_human",
    )

    assert ("interest", "session-parts", "Spare Parts Query") in calls
    assert (
        "interest",
        "session-human",
        "Wants to talk to a sales representative",
    ) in calls


def test_escalation_alert_email_uses_alert_subject_without_session_id(monkeypatch):
    calls = []
    monkeypatch.setattr(email_tools, "send_faq_email_sync", lambda **kwargs: calls.append(("send", kwargs)))

    result = email_tools.send_escalation_alert_email(
        full_name="Test User",
        email="test@example.com",
        phone="979-555-1111",
        summary="Customer asked for a quote.",
        user_message="Can you email me a quote?",
        context_summary="Category: Dump",
    )

    assert result["status"] == "sent"
    send_call = calls[0][1]
    assert send_call["subject"] == "Escalation Alert"
    assert "[Escalation Alert] Customer asked for a quote." in send_call["summary_line"]
    assert "Last user message: Can you email me a quote?" in send_call["summary_line"]
    assert "Context: Category: Dump" in send_call["summary_line"]
    assert "session" not in result["body_preview"].lower()
