from __future__ import annotations

import os
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


# The single most safety-critical facts in a dealership reply are the phone
# number and website. The LLM is only asked to "preserve" them, and generative
# rewriting can transpose digits or drop the number. We therefore treat these as
# constants and deterministically verify/repair the generated text against them
# rather than trusting the model to reproduce them bit-for-bit.
TRAILERPLACE_PHONE = "979-532-1486"
TRAILERPLACE_URL = "https://trailerplace.com"

# Matches any US-style phone token (e.g. 979-532-1486, 979.532.1486, 9795321486).
_PHONE_TOKEN_RE = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")
# Matches trailerplace.com with an optional scheme/path so a mangled URL variant
# can be normalized back to the canonical form.
_URL_TOKEN_RE = re.compile(r"(?:https?://)?(?:www\.)?trailerplace\.com\S*", re.IGNORECASE)


def _enforce_contact_facts(text: str, fallback: str) -> str:
    """Guarantee the canonical phone/website appear correctly in the reply.

    Only enforced when the deterministic ``fallback`` already carried the fact,
    so we never inject the dealership phone into a reply where it does not belong
    (e.g. a reply that merely echoed a customer-supplied number).
    """
    result = text

    if TRAILERPLACE_PHONE in fallback:
        phones = _PHONE_TOKEN_RE.findall(result)
        if phones:
            # Repair any transposed/incorrect number to the canonical one.
            result = _PHONE_TOKEN_RE.sub(TRAILERPLACE_PHONE, result)
        elif result:
            result = f"{result} You can reach our team at {TRAILERPLACE_PHONE}."

    if "trailerplace.com" in fallback.lower():
        if _URL_TOKEN_RE.search(result):
            result = _URL_TOKEN_RE.sub(TRAILERPLACE_URL, result)

    return result


EDITABLE_CUSTOMER_RESPONSE_GUIDANCE = """
CUSTOMER RESPONSE GUIDANCE:

FAQ scenarios:
- contact_human:
  "You can reach our team at 979-532-1486. Happy to keep helping with your trailer search too!"
- financing:
  "We offer financing. Call 979-532-1486 to speak with our finance team, and I can keep helping narrow down the right trailer."
- trade_in:
  "Our sales team handles trade-in appraisals. Call 979-532-1486."
- service_parts:
  "Our service and parts team can help. Reach them at 979-532-1486."
- store_info:
  "We're located in Wharton, TX. Call 979-532-1486 or visit https://trailerplace.com. We also offer financing and delivery."

Non-FAQ scenarios:
- generic team request:
  "Thanks, I shared that request with the team so they can help you with it."
- escalation:
  "I've passed your query to our team. In the meantime, I can keep helping you narrow down the right trailer."
- listing interest with a selected item:
  "Your interest in the selected trailer has been logged. Our team can follow up. In the meantime, feel free to visit https://trailerplace.com or call 979-532-1486."
- listing interest without a selected item:
  "Your interest has been logged. Our team can follow up. In the meantime, feel free to visit https://trailerplace.com or call 979-532-1486."
- safe listing-interest fallback:
  "Great, I shared your interest in that trailer with the team. They can follow up with you."

Treat these as editable response templates and factual guidance. Adapt them naturally to the supplied scenario,
customer message, selected trailer, and next question. Never expose the internal notification email.
""".strip()


def compose_email_tool_reply(
    *,
    email_purpose: str,
    latest_message: str,
    conversation_context: str = "",
    base_reply: str = "",
    next_question: str = "",
    fallback: str,
) -> str:
    clean_latest = " ".join(str(latest_message or "").lower().split()).rstrip("?.!")
    clean_next = " ".join(str(next_question or "").lower().split()).rstrip("?.!")
    if clean_next and clean_next == clean_latest:
        next_question = ""
    if not os.getenv("OPENAI_API_KEY"):
        return fallback
    try:
        response = ChatOpenAI(
            model=(os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip(),
            temperature=0,
        ).invoke(
            [
                SystemMessage(
                    content=(
                        "Write a brief, natural customer-facing reply after an internal notification tool succeeded. "
                        "Never mention an email, email delivery, tool call, and never say 'email sent'. "
                        "Describe only the customer-relevant outcome, such as sharing a request with the team, "
                        "logging listing interest, or directly answering an FAQ. "
                        "Address the customer's latest question using base_reply when supplied. "
                        "If next_question is supplied, end with that exact question unchanged. "
                        "If next_question is empty, do not ask any question. "
                        "When next_question is supplied, connect to it with a smooth, natural transition. "
                        "The response must not sound like an acknowledgement followed by a separately pasted question. "
                        "First answer or acknowledge the customer's counter-question or request, then use a brief "
                        "contextual bridge and naturally rephrase the still-unanswered active question. Preserve its "
                        "meaning and expected answer while making it sound conversational rather than fixed. "
                        "Never repeat or rephrase the customer's latest question as your own question. "
                        "Use the fallback response as style and content guidance, but rewrite it naturally; "
                        "preserve useful facts such as phone numbers, website links, and the email purpose. "
                        "Plain text only.\n\n"
                        "The following are the responses which are intended for the customer-facing reply. Use them as strict guidance, but adapt them naturally to the scenario, "
                        f"{EDITABLE_CUSTOMER_RESPONSE_GUIDANCE}"
                    )
                ),
                HumanMessage(
                    content=(
                        f"Email purpose: {email_purpose}\n"
                        f"Latest customer message: {latest_message}\n"
                        f"Base reply: {base_reply}\n"
                        f"Previous customer-facing response for guidance: {fallback}\n"
                        f"Next exact question: {next_question}\n"
                        f"Recent context: {conversation_context}"
                    )
                ),
            ]
        )
        text = str(response.content or "").strip()
        if not str(next_question or "").strip() and "?" in text:
            text = " ".join(
                part.strip()
                for part in re.split(r"(?<=[.!?])\s+", text)
                if part.strip() and not part.strip().endswith("?")
            ).strip()
        text = _enforce_contact_facts(text, fallback)
        return text or fallback
    except Exception:
        return fallback
