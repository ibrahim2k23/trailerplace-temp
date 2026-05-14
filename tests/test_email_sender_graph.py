from __future__ import annotations

import src.email_sender as email_sender


class _Resp:
    def __init__(self, status_code=200, text="", payload=None):
        self.status_code = status_code
        self.text = text
        self._payload = payload or {}

    def json(self):
        return self._payload


def test_graph_sender_posts_token_then_sendmail(monkeypatch):
    monkeypatch.setenv("EMAIL_BACKEND", "graph")
    monkeypatch.setenv("TENANT_ID", "tenant-1")
    monkeypatch.setenv("CLIENT_ID", "client-1")
    monkeypatch.setenv("CLIENT_SECRET", "secret-1")
    monkeypatch.setenv("SENDER_EMAIL", "leads@trailerplace.com")
    monkeypatch.setenv("RECIPIENT_EMAIL", "sales@trailerplace.com")
    monkeypatch.delenv("EMAIL_TO", raising=False)

    calls = []

    def _fake_post(url, **kwargs):
        calls.append((url, kwargs))
        if "oauth2/v2.0/token" in url:
            return _Resp(payload={"access_token": "token-123"})
        if "graph.microsoft.com/v1.0/users/" in url and url.endswith("/sendMail"):
            return _Resp(status_code=202)
        return _Resp(status_code=404, text="not found")

    monkeypatch.setattr(email_sender.requests, "post", _fake_post)

    email_sender.send_faq_email_sync(
        full_name="Test User",
        email="test@example.com",
        phone="979-555-1111",
        subject="FAQ inquiry: contact_human",
        summary_line="Customer appears confused.",
    )

    assert len(calls) == 2
    assert "oauth2/v2.0/token" in calls[0][0]
    assert calls[0][1]["data"]["grant_type"] == "client_credentials"
    assert calls[1][0].endswith("/users/leads@trailerplace.com/sendMail")
    assert calls[1][1]["headers"]["Authorization"] == "Bearer token-123"
    assert calls[1][1]["json"]["message"]["toRecipients"][0]["emailAddress"]["address"] == "sales@trailerplace.com"


def test_send_ticket_notification_uses_recipient_email_fallback(monkeypatch):
    monkeypatch.setenv("EMAIL_BACKEND", "smtp")
    monkeypatch.delenv("EMAIL_TO", raising=False)
    monkeypatch.setenv("RECIPIENT_EMAIL", "sales@trailerplace.com")

    sent = []

    class _FakeSender:
        def send_plain_text(self, to_address, subject, body):
            sent.append((to_address, subject, body))

    monkeypatch.setattr(email_sender, "get_email_sender", lambda: _FakeSender())

    email_sender.send_ticket_notification(
        full_name="Test User",
        email="test@example.com",
        phone="979-555-1111",
        item_name="Trailer A",
    )

    assert sent
    assert sent[0][0] == "sales@trailerplace.com"
    assert sent[0][1] == email_sender.TICKET_EMAIL_SUBJECT

