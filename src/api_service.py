"""
HTTP-facing chat orchestration: onboarding + LangGraph agent (server-side sessions).
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from src.agent_lg import TrailerAgentLG
from src.contact_onboarding import run_contact_onboarding_turn
from src.conversation_store import persistence_enabled, save_turn, ensure_session_lead_bundle
from src.models import CustomerContact

logger = logging.getLogger(__name__)

_sessions_lock = threading.Lock()
_agents: dict[str, TrailerAgentLG] = {}


def reset_server_session(session_id: str) -> None:
    """Drop in-memory LangGraph agent for a chat session (e.g. new conversation)."""
    sid = (session_id or "").strip()
    if not sid:
        return
    with _sessions_lock:
        _agents.pop(sid, None)


def _prior_messages_from_onboarding(api_msgs: list) -> list[dict[str, str]]:
    """Strip tool turns; keep user/assistant text for optional UI context."""
    out: list[dict[str, str]] = []
    for m in api_msgs or []:
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        c = m.get("content")
        if isinstance(c, str) and c.strip():
            out.append({"role": role, "content": c.strip()})
    return out


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    sales_phase: Literal["onboarding", "main"] = "onboarding"
    message: str = Field(..., min_length=1)
    onboarding_api_messages: list[dict[str, Any]] = Field(default_factory=list)
    customer_full_name: Optional[str] = None
    customer_email: Optional[str] = None
    customer_phone: Optional[str] = None


class ChatResponse(BaseModel):
    assistant_text: str
    listings: list[dict[str, Any]] = Field(default_factory=list)
    sales_phase: Literal["onboarding", "main"]
    onboarding_api_messages: list[dict[str, Any]] = Field(default_factory=list)
    customer_full_name: Optional[str] = None
    customer_email: Optional[str] = None
    customer_phone: Optional[str] = None
    main_prior_messages: Optional[list[dict[str, str]]] = None


def run_chat(req: ChatRequest) -> ChatResponse:
    if req.sales_phase == "onboarding":
        new_hist, text, customer_done = run_contact_onboarding_turn(
            api_messages=list(req.onboarding_api_messages),
            user_message=req.message,
        )
        out = ChatResponse(
            assistant_text=text,
            listings=[],
            sales_phase="onboarding",
            onboarding_api_messages=new_hist,
        )
        if customer_done is not None:
            out.sales_phase = "main"
            out.customer_full_name = customer_done.full_name
            out.customer_email = customer_done.email
            out.customer_phone = customer_done.phone
            out.main_prior_messages = _prior_messages_from_onboarding(new_hist)
            reset_server_session(req.session_id)
            if persistence_enabled():
                try:
                    save_turn(
                        req.session_id,
                        req.message,
                        text,
                        None,
                        None,
                        None,
                        customer_contact=customer_done,
                    )
                except Exception:
                    logger.exception(
                        "save_turn failed (onboarding complete) session_id=%s", req.session_id
                    )
        return out

    fn = (req.customer_full_name or "").strip()
    ph = (req.customer_phone or "").strip()
    if not fn or not ph:
        raise ValueError("customer_full_name and customer_phone are required when sales_phase is main")

    raw_email = (req.customer_email or "").strip()
    cust = CustomerContact(
        full_name=fn,
        email=raw_email if raw_email else None,
        phone=ph,
    )

    with _sessions_lock:
        agent = _agents.get(req.session_id)
        if agent is None:
            agent = TrailerAgentLG(customer=cust)
            _agents[req.session_id] = agent

    if persistence_enabled():
        try:
            ensure_session_lead_bundle(req.session_id, cust)
        except Exception:
            logger.exception("ensure_session_lead_bundle failed session_id=%s", req.session_id)

    reply, listings = agent.chat(req.message, session_id=req.session_id)

    if persistence_enabled():
        try:
            save_turn(
                req.session_id,
                req.message,
                reply,
                None,
                None,
                None,
                customer_contact=cust,
            )
        except Exception:
            logger.exception("save_turn failed (main) session_id=%s", req.session_id)

    return ChatResponse(
        assistant_text=reply,
        listings=listings or [],
        sales_phase="main",
        onboarding_api_messages=list(req.onboarding_api_messages),
        customer_full_name=cust.full_name,
        customer_email=cust.email,
        customer_phone=cust.phone,
    )
