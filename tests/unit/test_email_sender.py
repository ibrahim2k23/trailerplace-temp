from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.tools import email_sender


def _settings(**over):
    base = {
        "email_backend": "smtp",
        "tenant_id": "tid",
        "client_id": "cid",
        "client_secret": "secret",
        "sender_email": "sender@trailerplace.com",
        "recipient_email": "leads@trailerplace.com",
        "smtp_host": "smtp.example.com",
        "smtp_port": 587,
        "smtp_user": "smtpuser",
        "smtp_password": "smtppass",
        "smtp_from": "from@trailerplace.com",
        "email_to": "smtp-leads@trailerplace.com",
    }
    base.update(over)
    return SimpleNamespace(**base)


def test_email_body_exact_format():
    body = email_sender.render_email_body(
        name="John Doe",
        email="john@example.com",
        phone=None,
        reason="Escalation",
        description="Wants a quote call.",
    )
    assert body == (
        "Full Name: John Doe\n"
        "Email: john@example.com\n"
        "Phone Number: Not provided\n"
        "\n"
        "[Escalation] Wants a quote call."
    )


def test_email_body_all_not_provided():
    body = email_sender.render_email_body(name=None, email=None, phone=None, reason="Results Shown to User", description="Pinecone search — 3 results — Dump")
    assert body.startswith("Full Name: Not provided\nEmail: Not provided\nPhone Number: Not provided\n\n")
    assert body.endswith("[Results Shown to User] Pinecone search — 3 results — Dump")


def test_subject_format_with_and_without_name():
    assert email_sender.render_subject(reason="FAQ – financing", name="Jane", session_id="sess-1") == "TrailerPlace Lead — FAQ – financing — Jane"
    assert email_sender.render_subject(reason="Escalation", name=None, session_id="sess-1") == "TrailerPlace Lead — Escalation — sess-1"


def test_graph_backend_token_and_sendmail(monkeypatch):
    monkeypatch.setattr(email_sender, "settings", _settings(email_backend="graph"))
    calls = []

    class _Resp:
        def __init__(self, status_code, payload=None):
            self.status_code = status_code
            self._payload = payload or {}
            self.text = ""

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_post(url, **kwargs):
        calls.append({"url": url, **kwargs})
        if "oauth2/v2.0/token" in url:
            return _Resp(200, {"access_token": "abc123"})
        return _Resp(202)

    monkeypatch.setattr(email_sender.requests, "post", fake_post)
    assert email_sender.send_email("subj", "body") is True

    token_call = calls[0]
    assert token_call["data"]["grant_type"] == "client_credentials"
    assert token_call["data"]["client_id"] == "cid"
    assert token_call["data"]["scope"] == "https://graph.microsoft.com/.default"

    send_call = calls[1]
    assert send_call["url"].endswith("/users/sender@trailerplace.com/sendMail")
    assert send_call["headers"]["Authorization"] == "Bearer abc123"
    message = send_call["json"]["message"]
    assert message["subject"] == "subj"
    assert message["body"] == {"contentType": "Text", "content": "body"}
    assert message["toRecipients"] == [{"emailAddress": {"address": "leads@trailerplace.com"}}]


def test_graph_backend_non_202_returns_false(monkeypatch):
    monkeypatch.setattr(email_sender, "settings", _settings(email_backend="graph"))

    class _Resp:
        status_code = 500
        text = "boom"

        def raise_for_status(self):
            return None

        def json(self):
            return {"access_token": "abc"}

    monkeypatch.setattr(email_sender.requests, "post", lambda url, **kw: _Resp())
    assert email_sender.send_email("s", "b") is False


def test_smtp_backend_sends_to_email_to(monkeypatch):
    monkeypatch.setattr(email_sender, "settings", _settings(email_backend="smtp"))
    captured = {}

    class _SMTP:
        def __init__(self, host, port, timeout=None):
            captured["host"] = host
            captured["port"] = port

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def starttls(self):
            captured["starttls"] = True

        def login(self, user, password):
            captured["login"] = (user, password)

        def sendmail(self, sender, recipients, message):
            captured["sender"] = sender
            captured["recipients"] = recipients
            captured["message"] = message

    monkeypatch.setattr(email_sender.smtplib, "SMTP", _SMTP)
    assert email_sender.send_email("subj", "body") is True
    assert captured["host"] == "smtp.example.com"
    assert captured["recipients"] == ["smtp-leads@trailerplace.com"]
    assert captured["login"] == ("smtpuser", "smtppass")


def test_failure_returns_false_without_raising(monkeypatch):
    monkeypatch.setattr(email_sender, "settings", _settings(email_backend="graph"))

    def _boom(*a, **kw):
        raise RuntimeError("network down")

    monkeypatch.setattr(email_sender.requests, "post", _boom)
    assert email_sender.send_email("s", "b") is False
