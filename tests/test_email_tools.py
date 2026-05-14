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
    send_call = calls[0][1]
    assert "Customer asked about financing." in send_call["summary_line"]
    assert "Last user message: Can you help me with financing?" in send_call["summary_line"]
