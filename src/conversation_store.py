"""
Append chat turns to Postgres (`chatbot_conversations` + `chatbot_leads`).
Writes from Streamlit may use a thread pool; FastAPI may call sync helpers directly.
"""
from __future__ import annotations

import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Optional

from dotenv import load_dotenv
from sqlalchemy import select

load_dotenv()

logger = logging.getLogger(__name__)

_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tp_conv")

from src.db import get_engine, get_session_factory  # noqa: E402
from src.db_models import ChatbotConversation, ChatbotLead  # noqa: E402
from src.models import CustomerContact  # noqa: E402


def persistence_enabled() -> bool:
    if (os.getenv("TRAILERPLACE_PERSIST_CHATS") or "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return False
    return get_engine() is not None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _parse_session_uuid(session_id: str) -> Optional[uuid.UUID]:
    try:
        return uuid.UUID(str(session_id).strip())
    except (ValueError, AttributeError):
        return None


def _ensure_soft_lead_and_conversation_row(
    session: Any,
    sid: uuid.UUID,
    contact: CustomerContact,
) -> ChatbotConversation:
    row = session.execute(
        select(ChatbotConversation).where(ChatbotConversation.session_id == sid)
    ).scalar_one_or_none()
    if row is not None:
        return row

    lead = ChatbotLead(
        psid=None,
        name=(contact.full_name or "Unknown")[:255],
        phone_number=(contact.phone or "")[:64],
        email=(contact.email or None),
        lead_type="soft",
        item_of_interest="General inquiry",
    )
    session.add(lead)
    session.flush()

    conv = ChatbotConversation(
        session_id=sid,
        lead_id=lead.lead_id,
        conversation=[],
    )
    session.add(conv)
    session.flush()
    return conv


def ensure_session_lead_bundle(session_id: str, contact: CustomerContact) -> None:
    """
    Ensure a soft lead + chatbot_conversations row exists before a main-phase agent turn
    (so interest logging can update the lead even on the first message).
    """
    if not session_id or not persistence_enabled():
        return
    sid = _parse_session_uuid(session_id)
    if sid is None:
        return
    sf = get_session_factory()
    if sf is None:
        return
    db = sf()
    try:
        _ensure_soft_lead_and_conversation_row(db, sid, contact)
        db.commit()
    except Exception:
        logger.exception("ensure_session_lead_bundle failed session_id=%s", session_id)
        db.rollback()
    finally:
        db.close()


def upsert_hard_lead_for_interest(session_id: str, item_name: str) -> None:
    """
    After a successful product-interest email, mark the session's lead as hard
    and set item_of_interest.
    """
    if not session_id or not persistence_enabled():
        return
    sid = _parse_session_uuid(session_id)
    if sid is None:
        logger.warning("upsert_hard_lead_for_interest: invalid session_id=%r", session_id)
        return
    sf = get_session_factory()
    if sf is None:
        return
    db = sf()
    try:
        row = db.execute(
            select(ChatbotConversation).where(ChatbotConversation.session_id == sid)
        ).scalar_one_or_none()
        if row is None:
            raise RuntimeError(f"No chatbot_conversations row for session_id={session_id}")
        lead = db.get(ChatbotLead, row.lead_id)
        if lead is None:
            raise RuntimeError("Lead row missing for conversation")
        lead.lead_type = "hard"
        lead.item_of_interest = (item_name or "")[:8000]
        row.updated_at = datetime.now(timezone.utc)
        db.commit()
    except Exception:
        logger.exception("upsert_hard_lead_for_interest failed session_id=%s", session_id)
        db.rollback()
        raise
    finally:
        db.close()


def save_turn(
    session_id: str,
    user_text: str,
    assistant_text: str,
    tool_call: Any,
    tool_result: Any,
    search_runs: Any = None,
    *,
    customer_contact: Optional[CustomerContact] = None,
) -> None:
    if not session_id or not persistence_enabled():
        return
    if customer_contact is None:
        logger.debug(
            "save_turn skipped (no customer_contact yet) session_id=%s", session_id
        )
        return
    sid = _parse_session_uuid(session_id)
    if sid is None:
        logger.warning("save_turn: invalid session_id=%r", session_id)
        return

    turn: dict[str, Any] = {
        "at": _utc_now_iso(),
        "user": user_text,
        "assistant": assistant_text,
    }
    if tool_call is not None:
        turn["tool_call"] = tool_call
    if tool_result is not None:
        turn["tool_call_result"] = tool_result
    if search_runs is not None:
        turn["search_runs"] = search_runs

    sf = get_session_factory()
    if sf is None:
        return
    db = sf()
    try:
        conv_row = _ensure_soft_lead_and_conversation_row(db, sid, customer_contact)
        hist = list(conv_row.conversation or [])
        hist.append(turn)
        conv_row.conversation = hist
        conv_row.updated_at = datetime.now(timezone.utc)
        db.commit()
    except Exception:
        logger.exception("save_turn failed session_id=%s", session_id)
        db.rollback()
    finally:
        db.close()


def enqueue_save_turn(
    session_id: str,
    user_text: str,
    assistant_text: str,
    tool_call: Any,
    tool_result: Any,
    search_runs: Any = None,
    *,
    customer_contact: Optional[CustomerContact] = None,
) -> None:
    if not session_id or not persistence_enabled():
        return
    _pool.submit(
        save_turn,
        session_id,
        user_text,
        assistant_text,
        tool_call,
        tool_result,
        search_runs,
        customer_contact=customer_contact,
    )


def get_messages_for_session(session_id: str) -> Optional[list]:
    if not persistence_enabled():
        return None
    sid = _parse_session_uuid(session_id)
    if sid is None:
        return None
    sf = get_session_factory()
    if sf is None:
        return None
    db = sf()
    try:
        row = db.execute(
            select(ChatbotConversation).where(ChatbotConversation.session_id == sid)
        ).scalar_one_or_none()
        if not row:
            return None
        return list(row.conversation or [])
    except Exception:
        logger.exception("get_messages_for_session failed session_id=%s", session_id)
        return None
    finally:
        db.close()


def save_user_feedback(
    session_id: str, turn_index: int, feedback_text: str, feedback_at: str
) -> None:
    if not session_id or not persistence_enabled():
        return
    sid = _parse_session_uuid(session_id)
    if sid is None:
        return
    sf = get_session_factory()
    if sf is None:
        return
    db = sf()
    try:
        row = db.execute(
            select(ChatbotConversation).where(ChatbotConversation.session_id == sid)
        ).scalar_one_or_none()
        if not row:
            logger.warning("save_user_feedback: no row session_id=%s", session_id)
            return
        messages = list(row.conversation or [])
        if turn_index < 0 or turn_index >= len(messages):
            logger.warning(
                "save_user_feedback: bad turn_index %s len=%s",
                turn_index,
                len(messages),
            )
            return
        turn = messages[turn_index]
        if not isinstance(turn, dict):
            return
        if feedback_text:
            turn["user_feedback"] = {"text": feedback_text, "at": feedback_at}
        else:
            turn.pop("user_feedback", None)
        row.conversation = messages
        row.updated_at = datetime.now(timezone.utc)
        db.commit()
    except Exception:
        logger.exception("save_user_feedback failed session_id=%s", session_id)
        db.rollback()
    finally:
        db.close()


def enqueue_save_user_feedback(
    session_id: str, turn_index: int, feedback_text: str, feedback_at: str
) -> None:
    if not session_id or not persistence_enabled():
        return
    _pool.submit(save_user_feedback, session_id, turn_index, feedback_text, feedback_at)
