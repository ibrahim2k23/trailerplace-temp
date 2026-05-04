"""
Send a test email using SMTP_* and EMAIL_TO from .env.
Requires: python-dotenv (already in this project). smtplib/email are stdlib.
"""
import os
import smtplib
import ssl
from email.message import EmailMessage

from dotenv import load_dotenv

load_dotenv()

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
# Gmail app passwords are often pasted with spaces; strip them for auth.
SMTP_PASSWORD = "".join(
    (os.getenv("SMTP_PASSWORD") or "").split()
)  # remove all whitespace; Gmail app passwords are 16 chars
SMTP_FROM = os.getenv("SMTP_FROM") or SMTP_USER
EMAIL_TO = os.getenv("EMAIL_TO")


def main() -> None:
    missing = [
        name
        for name, val in [
            ("SMTP_USER", SMTP_USER),
            ("SMTP_PASSWORD", SMTP_PASSWORD),
            ("EMAIL_TO", EMAIL_TO),
        ]
        if not val
    ]
    if missing:
        print(f"Missing env: {', '.join(missing)}")
        raise SystemExit(1)

    msg = EmailMessage()
    msg["Subject"] = "Trailer Place SMTP test"
    msg["From"] = SMTP_FROM
    msg["To"] = EMAIL_TO
    msg.set_content(
        "This is a test message sent with smtplib.\n"
        "If you received it, SMTP settings in .env are working.\n"
    )

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            server.ehlo()
            server.starttls(context=context)
            server.ehlo()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
    except smtplib.SMTPAuthenticationError as e:
        print("SMTP authentication failed (Gmail 535 = bad username/password for AUTH).")
        print(f"  SMTP_USER={SMTP_USER!r}")
        print(
            f"  password length after stripping whitespace: {len(SMTP_PASSWORD)} "
            "(Gmail app passwords are 16 characters)"
        )
        print(
            "  Check: 2-Step Verification on, App Password for Mail, "
            "SMTP_USER = full gmail address, regenerate secret if unsure."
        )
        raise SystemExit(str(e)) from e
    except OSError as e:
        print(f"Could not reach {SMTP_HOST}:{SMTP_PORT}")
        raise SystemExit(str(e)) from e

    print(f"Sent test email from {SMTP_FROM!r} to {EMAIL_TO!r} via {SMTP_HOST}:{SMTP_PORT}")


if __name__ == "__main__":
    main()
