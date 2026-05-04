"""
LLM-only onboarding: collect full name, phone, and optionally email from natural language.

Uses OpenAI tool calling (no LangGraph). When ``submit_customer_contact`` succeeds
validation, returns ``CustomerContact`` so the app can start ``TrailerAgent``.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

from dotenv import load_dotenv
from openai import OpenAI

from src.contact_validation import validate_customer_profile
from src.models import CustomerContact

load_dotenv()

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = (os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip()

# Shown to the user immediately after a successful submit (deterministic; no second model round).
ONBOARDING_SUCCESS_MESSAGE = (
    "Thank you for providing your contact details. How can we help you today?"
)

SUBMIT_CUSTOMER_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_customer_contact",
        "description": (
            "When the customer's full name and phone are known and pass validation (count every digit in the phone; "
            "at least 10 digits is valid), you MUST call this tool immediately in that assistant turn. "
            "If they already gave name + phone in their message, your turn MUST be ONLY this tool call — no assistant "
            "text, no question for email, no other reply. "
            "Include `email` only if they already provided a valid-looking address in the conversation; otherwise omit "
            "email or use empty string. If they gave an email, it must include @ and a domain with a dot."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "full_name": {"type": "string", "description": "Customer's full name"},
                "email": {"type": "string", "description": "Customer email if provided; omit or leave empty if declined"},
                "phone": {"type": "string", "description": "Customer's phone (any common format)"},
            },
            "required": ["full_name", "phone"],
        },
    },
}

ONBOARDING_SYSTEM_PROMPT = """You are a friendly assistant for TrailerPlace (Wharton, TX). Collect **full name** and **phone number** to start the sales chat. Email is optional: only include it if the customer already shared a valid-looking address; never delay the tool call to ask for email when name and phone are already sufficient.

For your **first** assistant message only, start with exactly:
Thank you for contacting TrailerPlace. Before we get started, could you please provide your name,email and phone number

Strict rules:
- If the customer's **latest message** already contains a usable full name and a phone with **10 or more digits** (count every digit; leading zeros count), your response **must** be **only** a `submit_customer_contact` tool call in that turn: **no** assistant visible text, **no** question about email, **no** acknowledgment before the tool. Include `email` in the tool args only if they also gave a valid email in that message or earlier turns; otherwise omit `email` or use "".
- If name or phone is missing or ambiguous, ask **one** short follow-up for what is missing. Never ask for email instead of calling the tool when name + phone are already valid.
- Do **not** discuss trailers, inventory, or pricing until contact is saved.
- Do **not** read back fields or ask "Is that correct?" before calling the tool.
- After the tool succeeds, the app will show a fixed thank-you line — do **not** rely on you for that message; never send a second assistant reply in the same user turn after a successful tool (the run ends after success)."""


def _assistant_to_dict(msg: Any) -> dict[str, Any]:
    d: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
    tcs = getattr(msg, "tool_calls", None) or None
    if tcs:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments or "{}",
                },
            }
            for tc in tcs
        ]
    return d


def run_contact_onboarding_turn(
    *,
    api_messages: list[dict[str, Any]],
    user_message: str,
    client: Optional[OpenAI] = None,
    model: Optional[str] = None,
) -> tuple[list[dict[str, Any]], str, Optional[CustomerContact]]:
    """
    Append the user turn, run the onboarding model (with tool loop), return:
    (updated_api_messages_without_system, assistant_text_for_ui, customer_if_just_completed).

    ``api_messages`` is persisted OpenAI history (user / assistant / tool only — no system).
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")

    openai = client or OpenAI(api_key=OPENAI_API_KEY)
    model_name = model or OPENAI_MODEL

    work: list[dict[str, Any]] = [m.copy() for m in api_messages]
    work.append({"role": "user", "content": user_message.strip()})

    system = {"role": "system", "content": ONBOARDING_SYSTEM_PROMPT}
    customer_out: Optional[CustomerContact] = None
    last_assistant_text = ""

    for _ in range(8):
        messages = [system] + work
        response = openai.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=[SUBMIT_CUSTOMER_TOOL],
            tool_choice="auto",
        )
        msg = response.choices[0].message
        last_assistant_text = (msg.content or "").strip()

        if not msg.tool_calls:
            work.append(_assistant_to_dict(msg))
            return (
                work,
                last_assistant_text or "Thanks — you're all set.",
                customer_out,
            )

        work.append(_assistant_to_dict(msg))

        for tc in msg.tool_calls:
            if tc.function.name != "submit_customer_contact":
                tool_body = json.dumps({"ok": False, "error": f"unknown_tool:{tc.function.name}"})
                work.append({"role": "tool", "tool_call_id": tc.id, "content": tool_body})
                continue
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            fn = str(args.get("full_name") or "").strip()
            em = str(args.get("email") or "").strip()
            ph = str(args.get("phone") or "").strip()
            errs = validate_customer_profile(fn, em, ph)
            if errs:
                tool_body = json.dumps({"ok": False, "validation_errors": errs})
            else:
                customer_out = CustomerContact(
                    full_name=fn,
                    email=em if em else None,
                    phone=ph,
                )
                tool_body = json.dumps({"ok": True, "message": "contact_saved"})
                logger.info(
                    "Onboarding: submit_customer_contact accepted (email=%s)",
                    em if em else "(none)",
                )
            work.append({"role": "tool", "tool_call_id": tc.id, "content": tool_body})

        if customer_out is not None:
            work.append({"role": "assistant", "content": ONBOARDING_SUCCESS_MESSAGE})
            return (work, ONBOARDING_SUCCESS_MESSAGE, customer_out)

    work.append(
        {
            "role": "assistant",
            "content": "Sorry — something got stuck. Please try again with your name and phone (and email if you like) in one message.",
        }
    )
    return work, work[-1]["content"], customer_out
