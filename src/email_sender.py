"""
Pluggable outbound email: SMTP (default) or Microsoft Graph (stub).

Switch backends with env ``EMAIL_BACKEND`` (``smtp`` | ``graph``).
Graph: implement ``GraphEmailSender.send_plain_text`` using client credentials
and ``POST https://graph.microsoft.com/v1.0/users/{sender}/sendMail``.
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
import smtplib
import ssl
from abc import ABC, abstractmethod
from email.message import EmailMessage
from typing import Optional, Protocol, runtime_checkable
import requests

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

TICKET_EMAIL_SUBJECT = "ticket notification"

_faq_email_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="faq_email")


@runtime_checkable
class EmailSender(Protocol):
    def send_plain_text(self, to_address: str, subject: str, body: str) -> None: ...


class SmtpEmailSender:
    """Send mail via STARTTLS SMTP (e.g. Gmail). Uses SMTP_* and optional SMTP_FROM from env."""

    def __init__(self) -> None:
        self._host = (os.getenv("SMTP_HOST") or "smtp.gmail.com").strip()
        self._port = int((os.getenv("SMTP_PORT") or "587").strip())
        self._user = (os.getenv("SMTP_USER") or "").strip()
        self._password = "".join((os.getenv("SMTP_PASSWORD") or "").split())
        self._from = (os.getenv("SMTP_FROM") or "").strip() or self._user

    def send_plain_text(self, to_address: str, subject: str, body: str) -> None:
        if not self._user or not self._password:
            raise RuntimeError("SMTP_USER and SMTP_PASSWORD must be set for SMTP email backend")
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = self._from
        msg["To"] = to_address
        msg.set_content(body)
        context = ssl.create_default_context()
        with smtplib.SMTP(self._host, self._port, timeout=30) as server:
            server.ehlo()
            server.starttls(context=context)
            server.ehlo()
            server.login(self._user, self._password)
            server.send_message(msg)


class GraphEmailSender(ABC):
    """
    Placeholder for Microsoft Graph application mail.

    Implement send_plain_text using e.g. requests + client_credentials token,
    then POST /v1.0/users/{sender_id}/sendMail with a JSON message payload.
    """

    @abstractmethod
    def send_plain_text(self, to_address: str, subject: str, body: str) -> None:
        raise NotImplementedError(
            "Implement MS Graph in GraphEmailSender: obtain token for "
            "https://graph.microsoft.com/.default, then sendMail for the configured sender."
        )


class _GraphEmailSenderImpl(GraphEmailSender):
    """Send mail via Microsoft Graph using app-only OAuth2 client credentials."""

    def __init__(self) -> None:
        self._tenant_id = (os.getenv("TENANT_ID") or "").strip()
        self._client_id = (os.getenv("CLIENT_ID") or "").strip()
        self._client_secret = (os.getenv("CLIENT_SECRET") or "").strip()
        self._sender_email = (os.getenv("SENDER_EMAIL") or os.getenv("SMTP_FROM") or "").strip()

    def _access_token(self) -> str:
        if not self._tenant_id or not self._client_id or not self._client_secret:
            raise RuntimeError(
                "TENANT_ID, CLIENT_ID, and CLIENT_SECRET must be set for EMAIL_BACKEND=graph"
            )
        token_url = f"https://login.microsoftonline.com/{self._tenant_id}/oauth2/v2.0/token"
        response = requests.post(
            token_url,
            data={
                "client_id": self._client_id,
                "client_secret": self._client_secret,
                "scope": "https://graph.microsoft.com/.default",
                "grant_type": "client_credentials",
            },
            timeout=30,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Graph token request failed: {response.status_code} {response.text[:400]}"
            )
        payload = response.json()
        token = str(payload.get("access_token") or "").strip()
        if not token:
            raise RuntimeError("Graph token response did not include access_token")
        return token

    def send_plain_text(self, to_address: str, subject: str, body: str) -> None:
        sender = self._sender_email
        if not sender:
            raise RuntimeError("SENDER_EMAIL must be set for EMAIL_BACKEND=graph")
        token = self._access_token()
        send_url = f"https://graph.microsoft.com/v1.0/users/{sender}/sendMail"
        payload = {
            "message": {
                "subject": subject,
                "body": {"contentType": "Text", "content": body},
                "toRecipients": [
                    {
                        "emailAddress": {
                            "address": to_address,
                        }
                    }
                ],
            },
            "saveToSentItems": "true",
        }
        response = requests.post(
            send_url,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Graph sendMail failed: {response.status_code} {response.text[:400]}"
            )


def get_email_sender() -> EmailSender:
    backend = (os.getenv("EMAIL_BACKEND") or "smtp").strip().lower()
    if backend == "graph":
        return _GraphEmailSenderImpl()
    if backend != "smtp":
        logger.warning("Unknown EMAIL_BACKEND=%r — using smtp", backend)
    return SmtpEmailSender()


def send_ticket_notification(
    *,
    full_name: str,
    email: Optional[str],
    phone: str,
    item_name: str,
    details: Optional[str] = None,
) -> None:
    """
    Send lead notification to EMAIL_TO. Subject is fixed (ticket notification).

    Raises on missing configuration or send failure so the agent tool can surface errors.
    """
    to_addr = (os.getenv("EMAIL_TO") or os.getenv("RECIPIENT_EMAIL") or "").strip()
    if not to_addr:
        raise RuntimeError("EMAIL_TO or RECIPIENT_EMAIL is not set in the environment")

    email_line = (email or "").strip() or "Not provided"
    details_block = (details or "").strip()
    body = (
        f"Full Name: {full_name}\n"
        f"Email: {email_line}\n"
        f"Phone Number: {phone}\n"
        f'\nThe user is interested in "{item_name}"\n'
    )
    if details_block:
        body = f"{body}\n{details_block}\n"
    sender = get_email_sender()
    sender.send_plain_text(to_addr, TICKET_EMAIL_SUBJECT, body)
    logger.info("Ticket notification email sent to configured EMAIL_TO")


def send_faq_email_sync(
    *,
    full_name: str,
    email: Optional[str],
    phone: str,
    subject: str,
    summary_line: str,
) -> None:
    """Send FAQ / non-sales inquiry email to EMAIL_TO (blocking)."""
    to_addr = (os.getenv("EMAIL_TO") or os.getenv("RECIPIENT_EMAIL") or "").strip()
    if not to_addr:
        raise RuntimeError("EMAIL_TO or RECIPIENT_EMAIL is not set in the environment")
    subj = (subject or "").strip() or "FAQ inquiry"
    email_line = (email or "").strip() or "Not provided"
    summary = (summary_line or "").strip() or "(no summary)"
    body = (
        f"Name: {full_name}\n"
        f"Email: {email_line}\n"
        f"Phone Number: {phone}\n"
        f"\n{summary}\n"
    )
    sender = get_email_sender()
    sender.send_plain_text(to_addr, subj, body)
    logger.info("FAQ email sent to configured EMAIL_TO subject=%r", subj)


def enqueue_faq_email_notification(
    *,
    full_name: str,
    email: Optional[str],
    phone: str,
    subject: str,
    summary_line: str,
) -> None:
    """Queue FAQ email send in a background thread (non-blocking for chat)."""

    def _run() -> None:
        try:
            send_faq_email_sync(
                full_name=full_name,
                email=email,
                phone=phone,
                subject=subject,
                summary_line=summary_line,
            )
        except Exception:
            logger.exception("Background FAQ email send failed")

    _faq_email_pool.submit(_run)
    logger.info(
        "FAQ_EMAIL_ENQUEUED | subject=%r | summary_len=%s",
        (subject or "")[:120],
        len(summary_line or ""),
    )
