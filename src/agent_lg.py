"""
agent_lg.py – LangGraph implementation of the TrailerPlace conversational agent.

Architecture
────────────
                  ┌──────────────────────┐
   user input ──► │  master_router_node  │
                  └──────────┬───────────┘
                             │ trailer_type set?
                    ┌────────▼────────┐
                    │ specialist_node │◄── re-entered if category changes
                    └────────┬────────┘
                             │ search_results for this category?
                    ┌────────▼─────────────┐
                    │ recommendation_node  │
                    └──────────────────────┘

Session state key: `trailer_type`
  – Set by master_router_node via set_trailer_type() tool call.
  – Once set, triggers fetch_trailer_fields() which populates
    required_slots / optional_slots in state.
  – specialist_node uses those lists to drive one-question-at-a-time
    slot collection before calling search_trailers.

Tools exposed to the LLM
─────────────────────────
  faq_tool                 – Master / recommendation: scripted FAQ + async email.
  set_trailer_type         – Master router saves the resolved category.
  fetch_trailer_fields     – Specialist fetches required/optional slots.
  record_slot_answer       – Specialist records one qualification slot answer.
  search_trailers          – Specialist triggers Pinecone search.
  log_product_interest     – Recommendation node logs customer interest.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from src.email_sender import enqueue_faq_email_notification, send_ticket_notification
from src.models import CustomerContact, TrailerFilter, TrailerListing
from src.normalizer import (
    normalize_category,
    normalize_color,
    normalize_hitch,
    normalize_make,
    normalize_subcategory,
    HITCH_MAP,
)
from src.state import SessionState
from src.trailer_fields import get_trailer_fields_as_dict, list_all_categories
from src.shown_listings_store import merge_shown_urls_for_show_more, sanitize_already_shown_urls
from src.conversation_store import upsert_hard_lead_for_interest

# Re-use the search/rerank helpers from the original agent
from src.agent import (
    _apply_category_make_priority,
    _build_pinecone_filter,
    _canonical_listing_key_from_match,
    _coerce_required_length_ft,
    _coerce_required_payload_lbs,
    _dedupe_matches,
    _extract_length_ft_from_text,
    _extract_weight_lbs_from_text,
    _infer_hitch_type_from_text,
    _log_product_fetch_event,
    _match_summary_for_log,
    _metadata_to_listing,
    SEARCH_MAX_RECOMMENDATIONS,
    SEARCH_TOP_K,
    SEARCH_TOP_K_MORE,
    SEARCH_TOP_K_MORE_MAX,
)

# Canonical trailer categories only (never hitch terms like Gooseneck / Bumper Pull).
_ALLOWED_TRAILER_TYPES: frozenset[str] = frozenset(list_all_categories())

# While `trailer_type` is Aluminum and `base_category` is not yet recorded, these
# `set_trailer_type` targets are almost always a mistaken "aluminum style" answer.
# Categories like **Dump** are intentionally excluded so customers can hard-pivot.
_ALUMINUM_AMBIGUOUS_CATEGORY_PIVOTS: frozenset[str] = frozenset(
    {
        "Utility",
        "Equipment",
        "Enclosed",
        "Car Hauler",
        "Flatbed",
        "Tilt",
        "Livestock",
        "Fiber",
        "Race Trailer",
        "Welding",
        "Roll Off",
        "Diesel Tank",
    }
)


def _pinecone_subcategory_from_aluminum_base_slot(value: Any) -> Optional[str]:
    """Map Aluminum `base_category` slot text to Pinecone metadata subcategory string."""
    raw = str(value or "").strip()
    if not raw:
        return None
    cat = normalize_category(raw)
    if cat and cat != "Unknown":
        sub = normalize_subcategory(cat)
        return sub if sub else cat
    return normalize_subcategory(raw)


def _validate_set_trailer_type_arg(raw: str) -> tuple[Optional[str], Optional[dict[str, Any]]]:
    """
    If *raw* is a valid trailer category, return (canonical_name, None).
    Otherwise return (None, error_dict) for the tool message — do not mutate trailer_type.
    """
    if not (raw or "").strip():
        return None, {"ok": False, "error": "missing_trailer_type"}
    canonical = normalize_category(raw)
    if canonical in _ALLOWED_TRAILER_TYPES:
        return canonical, None
    key = (raw or "").strip().lower()
    hitch_guess = HITCH_MAP.get(key) or _infer_hitch_type_from_text(raw)
    err: dict[str, Any] = {
        "ok": False,
        "error": "invalid_trailer_category",
        "received": raw,
        "normalized_attempt": canonical,
        "allowed_categories": sorted(_ALLOWED_TRAILER_TYPES),
    }
    if hitch_guess:
        err["detected_hitch_type"] = hitch_guess
        err["message"] = (
            f"'{raw}' is a hitch preference ({hitch_guess}), not a trailer category. "
            "Do NOT call set_trailer_type for hitch wording. Keep the current trailer category "
            f"and use search_trailers(..., hitch_type={hitch_guess!r}, ...) when searching."
        )
    else:
        err["message"] = (
            f"'{raw}' is not a recognized trailer category. "
            f"Use set_trailer_type with one of: {sorted(_ALLOWED_TRAILER_TYPES)}."
        )
    return None, err


def _infer_hitch_from_recent_user_messages(
    state: SessionState,
    *,
    max_human_turns: int = 12,
) -> Optional[str]:
    """Walk recent user messages newest-first; return first clear hitch preference."""
    seen = 0
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            text = str(msg.content or "")
            seen += 1
        elif isinstance(msg, dict) and msg.get("role") == "user":
            text = str(msg.get("content") or "")
            seen += 1
        else:
            continue
        h = _infer_hitch_type_from_text(text)
        if h:
            return h
        if seen >= max_human_turns:
            break
    return None


load_dotenv()

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Lightweight haul detection
# ─────────────────────────────────────────────────────────────────────────────

# Slot names that represent haul-weight questions.  When the haul item is
# detected as lightweight these slots are stripped from required_slots so the
# specialist never asks for them.
WEIGHT_SLOT_NAMES: frozenset[str] = frozenset({
    "haul_weight_lbs",
    "total_weight",
})


def _strip_weight_slots_from_spec_dict(spec: dict[str, Any]) -> dict[str, Any]:
    """
    Return a copy of get_trailer_fields_as_dict output with weight-related slots
    and their questions removed (Utility lightweight path).
    """
    out = dict(spec)
    out["required_slots"] = [
        s for s in out.get("required_slots", []) if s not in WEIGHT_SLOT_NAMES
    ]
    q = dict(out.get("questions") or {})
    for w in WEIGHT_SLOT_NAMES:
        q.pop(w, None)
    out["questions"] = q
    return out


def _sanitize_fetch_tool_json_for_utility_lightweight(content: str) -> Optional[str]:
    """
    If *content* is a fetch_trailer_fields JSON payload for Utility that still lists
    weight slots, return stripped JSON string; otherwise return None (caller keeps original).
    """
    text = (content or "").strip()
    if not text.startswith("{"):
        return None
    try:
        spec = json.loads(text)
    except json.JSONDecodeError:
        return None
    if spec.get("category") != "Utility":
        return None
    req = list(spec.get("required_slots") or [])
    if not any(s in WEIGHT_SLOT_NAMES for s in req):
        return None
    stripped = _strip_weight_slots_from_spec_dict(spec)
    return json.dumps(stripped, ensure_ascii=True)


# Keys merged from last_search_args when search_trailers(..., more_results=true).
_LAST_SEARCH_MERGE_KEYS: tuple[str, ...] = (
    "hitch_type",
    "required_payload_lbs",
    "required_length_ft",
    "required_gvwr_lbs",
    "condition",
    "price_min",
    "price_max",
    "make",
    "color",
    "category_subcategory",
    "subcategory",
)

# Canonical lightweight item keywords (plain lowercase, no regex).
# Checked via substring/word-boundary match in is_lightweight_haul().
_LIGHTWEIGHT_KEYWORDS: tuple[str, ...] = (
    "golf cart", "golf carts", "golf equipment",
    "atv", "atvs", "utv", "utvs", "side-by-side", "side by side",
    "dirt bike", "dirt bikes", "motorcycle", "motorcycles",
    "lawn mower", "lawn mowers", "zero turn", "zero-turn",
    "riding mower", "push mower",
    "gardening tools", "landscaping tools", "lawn equipment", "lawn tools",
    "small generator", "small equipment",
    "canoe", "canoes", "kayak", "kayaks", "small watercraft",
    "bicycle", "bicycles", "e-bike", "e-bikes", "ebike", "ebikes",
    "small furniture", "camping gear", "light cargo", "hobby equipment",
)


def is_lightweight_haul(text: str) -> bool:
    """
    Return True when *text* mentions an item that is inherently lightweight
    (under ~1,500 lbs without needing a scale).

    Uses the same patterns as agent.py:_is_clear_light_cargo_text plus a fast
    keyword pre-check so callers can skip importing agent.py if needed.
    """
    if not (text or "").strip():
        return False
    t = text.lower()
    # Fast keyword scan first
    for kw in _LIGHTWEIGHT_KEYWORDS:
        if kw in t:
            return True
    # Fall back to the regex-based check in agent.py for edge cases
    from src.agent import _is_clear_light_cargo_text

    return _is_clear_light_cargo_text(t)


def _strip_weight_slots(required: list[str], haul_text: str) -> tuple[list[str], bool]:
    """
    Legacy helper (keyword fallback). Prefer Utility-only classifier in TrailerAgentLG.
    """
    if not is_lightweight_haul(haul_text):
        return required, False
    filtered = [s for s in required if s not in WEIGHT_SLOT_NAMES]
    return filtered, True


OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "trailerplace-listings")


# ─────────────────────────────────────────────────────────────────────────────
# Tool schemas (OpenAI function format)
# ─────────────────────────────────────────────────────────────────────────────

SET_TRAILER_TYPE_TOOL = {
    "type": "function",
    "function": {
        "name": "set_trailer_type",
        "description": (
            "Use this tool as soon as you have identified the customer's trailer category. "
            "This saves the trailer type in the session so the specialist can take over. "
            "Call this BEFORE asking any qualification questions. "
            "Supported categories: " + ", ".join(list_all_categories())
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "trailer_type": {
                    "type": "string",
                    "enum": sorted(list_all_categories()),
                    "description": (
                        "The resolved canonical trailer category. "
                        "Never use Gooseneck or Bumper Pull here — those are hitch_type on search_trailers."
                    ),
                }
            },
            "required": ["trailer_type"],
        },
    },
}

FETCH_TRAILER_FIELDS_TOOL = {
    "type": "function",
    "function": {
        "name": "fetch_trailer_fields",
        "description": (
            "Fetch the required and optional qualification slots for the current trailer type. "
            "Call this once at the start of the specialist node to know which questions to ask. "
            "Returns a JSON object with required_slots, optional_slots, questions, and notes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "trailer_type": {
                    "type": "string",
                    "description": "The trailer category to fetch fields for.",
                }
            },
            "required": ["trailer_type"],
        },
    },
}

SEARCH_TRAILERS_TOOL = {
    "type": "function",
    "function": {
        "name": "search_trailers",
        "description": (
            "Search the trailer inventory once all required qualification slots are collected. "
            "Provide a rich natural-language query. Add filter fields only when the customer "
            "has clearly specified them."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Detailed natural language query: include haul item, weight, trailer type, "
                        "and any feature tags from the conversation."
                    ),
                },
                "condition": {"type": "string", "enum": ["New", "Pre-Owned"]},
                "price_min": {"type": "number"},
                "price_max": {"type": "number"},
                "category_subcategory": {
                    "type": "string",
                    "description": (
                        "Resolved trailer category. Use one of: Equipment, Car Hauler, Utility, "
                        "Dump, Tilt, Enclosed, Livestock, Roll Off, Diesel Tank, Flatbed, "
                        "Fiber, Race Trailer, Welding, Aluminum. "
                        "For **Aluminum**, keep this as **Aluminum**; put the customer's style "
                        "(Utility, Equipment, …) in **subcategory**."
                    ),
                },
                "subcategory": {
                    "type": "string",
                    "description": (
                        "Pinecone subcategory — required style filter when category is **Aluminum** "
                        "(Utility, Equipment, Enclosed, …). The system may also infer this from "
                        "the `base_category` slot."
                    ),
                },
                "make": {"type": "string"},
                "color": {"type": "string"},
                "hitch_type": {
                    "type": "string",
                    "enum": ["Bumper Pull", "Gooseneck"],
                    "description": "Set when the customer clearly specifies a hitch preference.",
                },
                "required_payload_lbs": {"type": "number"},
                "required_length_ft": {"type": "number"},
                "required_gvwr_lbs": {"type": "number"},
                "more_results": {
                    "type": "boolean",
                    "description": "True when customer asks for more options from the same search.",
                },
            },
            "required": ["query"],
        },
    },
}

RECORD_SLOT_ANSWER_TOOL = {
    "type": "function",
    "function": {
        "name": "record_slot_answer",
        "description": (
            "Record the customer's answer to a required or optional qualification slot. "
            "Call immediately after the customer answers a question, before asking the next "
            "question or calling search_trailers."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "slot": {
                    "type": "string",
                    "description": "Slot name from required_slots or optional_slots.",
                },
                "value": {
                    "description": "The customer's answer (string or number).",
                },
            },
            "required": ["slot", "value"],
        },
    },
}

LOG_INTEREST_TOOL = {
    "type": "function",
    "function": {
        "name": "log_product_interest",
        "description": (
            "Call when the customer clearly expresses interest in a specific trailer listing. "
            "Use the exact listing title from search results. "
            "After this succeeds, tell the customer their query has been logged and the team will follow up."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "item_name": {
                    "type": "string",
                    "description": "Full listing title of the trailer the customer is interested in.",
                }
            },
            "required": ["item_name"],
        },
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Node prompts
# ─────────────────────────────────────────────────────────────────────────────

_SITE_URL = (os.getenv("TRAILERPLACE_WEBSITE", "https://www.trailerplace.com") or "").strip()
if not _SITE_URL.startswith("http"):
    _SITE_URL = "https://www.trailerplace.com"

# FAQ tool: enum + canonical reply_text templates (used when model reply is empty/off-script).
FAQ_TOOL_TYPES: tuple[str, ...] = (
    "contact_or_human",
    "financing",
    "trade_in",
    "service_parts",
    "store_info",
)
FAQ_TYPES_SET: frozenset[str] = frozenset(FAQ_TOOL_TYPES)
FAQ_CANONICAL_REPLY_TEXT: dict[str, str] = {
    "contact_or_human": (
        "You can reach our team at 979-532-1486. Happy to keep helping with your trailer search too!"
    ),
    "financing": (
        "We offer financing — call 979-532-1486 to speak with our finance team. "
        "I can also keep helping you narrow down the right trailer."
    ),
    "trade_in": "Our sales team handles trade-in appraisals — call 979-532-1486.",
    "service_parts": "Our service and parts team can help — reach them at 979-532-1486.",
    "store_info": (
        f"We're located in Wharton, TX. Give us a call at 979-532-1486 or visit {_SITE_URL}. "
        "We offer financing and delivery."
    ),
}
FAQ_CANONICAL_EMAIL_SUMMARY: dict[str, str] = {
    "contact_or_human": "Customer asked how to contact a representative.",
    "financing": "Customer asked about financing options.",
    "trade_in": "Customer asked about trade-in options.",
    "service_parts": "Customer asked about service or parts.",
    "store_info": "Customer asked for store information (location/hours/website).",
}

FAQ_TOOL = {
    "type": "function",
    "function": {
        "name": "faq_tool",
        "description": (
            "Use for non-sales questions: contact/human rep, financing, trade-in, service/parts, "
            "or store info (location/hours/website). Set reply_text to the customer-facing text "
            "you intend to show (ideally the canonical script for the chosen faq_type). "
            "You MUST call this tool before answering any FAQ question — do not answer FAQ in plain text only."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "faq_type": {
                    "type": "string",
                    "enum": list(FAQ_TOOL_TYPES),
                    "description": "Which FAQ category applies.",
                },
                "title": {
                    "type": "string",
                    "description": "Short email subject for the internal notification.",
                },
                "user_summary": {
                    "type": "string",
                    "description": "One-line summary of what the customer asked.",
                },
                "reply_text": {
                    "type": "string",
                    "description": "Customer-facing reply you plan to send (non-empty).",
                },
            },
            "required": ["faq_type", "title", "user_summary", "reply_text"],
        },
    },
}


def _normalize_ws(s: str) -> str:
    return " ".join((s or "").split())


def _faq_reply_matches_canonical(faq_type: str, reply_text: str) -> bool:
    exp = FAQ_CANONICAL_REPLY_TEXT.get(faq_type)
    if not exp:
        return False
    return _normalize_ws(exp) == _normalize_ws(reply_text)


def _effective_faq_reply_text(faq_type: str, reply_text: str) -> str:
    """Prefer canonical script when model reply is missing or does not match canonical."""
    rt = (reply_text or "").strip()
    if rt and _faq_reply_matches_canonical(faq_type, rt):
        return rt
    return (FAQ_CANONICAL_REPLY_TEXT.get(faq_type) or rt).strip()


def _effective_faq_email_summary(faq_type: str, user_summary: str) -> str:
    """Use deterministic one-line email summaries to avoid noisy raw question echoes."""
    fallback = (FAQ_CANONICAL_EMAIL_SUMMARY.get(faq_type) or "").strip()
    raw = (user_summary or "").strip()
    return fallback or raw


def _patch_final_assistant_with_faq_reply_text(
    messages: list[Any], reply_text: str
) -> None:
    """Backward-compatible alias for tests; prefer _ensure_faq_assistant_reply."""
    _ensure_faq_assistant_reply(messages, reply_text)


def _ensure_faq_assistant_reply(messages: list[Any], reply_text: str) -> None:
    """
    Ensure the user sees reply_text: patch the last plain AIMessage, or append one if missing.
    """
    text = (reply_text or "").strip()
    if not text:
        return
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, AIMessage):
            tcs = getattr(msg, "tool_calls", None) or []
            if not tcs:
                messages[i] = AIMessage(content=text)
                return
    messages.append(AIMessage(content=text))
    logger.info("FAQ_FALLBACK_REPLY_USED | reason=no_plain_assistant_message_appended")


MASTER_ROUTER_PROMPT = f"""You are a friendly trailer sales assistant for TrailerPlace, Wharton TX (979-532-1486).

## YOUR ROLE
You are the entry point of this conversation. Your sole job is to:
1. Greet the customer warmly (use their first name if known).
2. For **contact / human rep / financing / trade-in / service or parts / store info**, you **must** call **`faq_tool` first** (before any assistant text answering that FAQ). Use a clear `title`, one-line `user_summary`, and `reply_text`; the customer will see the canonical script for that FAQ type if needed.
3. Identify the trailer **category** the customer needs, then call `set_trailer_type` immediately (sales flow).

## AVAILABLE TOOLS
- **faq_tool**: Non-sales FAQ only. Enum `faq_type`: contact_or_human, financing, trade_in, service_parts, store_info.
- **set_trailer_type**: Call this as soon as you know the trailer category. This hands the customer off to the category specialist. Do NOT ask qualification questions yourself — the specialist handles that.

## CATEGORY IDENTIFICATION RULES
Map what the customer says to one of these categories:
Equipment, Car Hauler, Utility, Dump, Tilt, Enclosed, Livestock, Flatbed, Roll Off, Diesel Tank, Fiber, Race Trailer, Welding, Aluminum

**Hitch vs category (critical):** Words like **gooseneck**, **bumper pull**, **tag-along** describe a **hitch type**, not a trailer category. Never call `set_trailer_type` with those values. If the customer only mentions hitch style, keep the category you already identified (or ask which trailer category they need) — the specialist will set `hitch_type` on `search_trailers`.

Key disambiguation:
- "aluminum" / "lightweight" / "won't rust" → Aluminum
- "toy hauler" / "trailer without sides" → Car Hauler
- "office trailer" / "cooldown trailer" → Ask: "Will this be for fiber/telecom work specifically → then "Fiber", or a more general office trailer? → Enclosed" 
- "lowboy" / "low profile" / "skid steer" / "mini ex" / "mini excuvator" / "tractor" / "DeckOver"→ Equipment
- "box trailer"/ "cargo" / "V-nose" → Enclosed
- "landscape" / "lawnmower" / "ATV trailer" → Utility
- "splicing trailer" / "fiber optic trailer" → Fiber
- "enclosed car hauler" → Race Trailer
- "dumpster" / "roll-off" → Roll Off
- "fuel tank" / "tank trailer" → Diesel Tank
- "hotshot" / "step deck" / "platform trailer" → Flatbed
- "scissor lift" / "hoist" / "dump trailer" / "telescopic" / "front lift" → Dump
- "full tilt"/"gravity dampened tilt" / "hydraulic dampened tilt" → Tilt
- "Galyean" / "Star trailer" / "Calico trailer" / goats / hogs / cattle → Livestock

If the customer hasn't said enough to determine a category, ask ONE question: "What will you be using the trailer for?" or "What are you looking to haul?"

## IMPORTANT
- Do NOT ask weight, size, hitch, or any other qualification questions here. Just identify intent and category.
- Once you know the category, call `set_trailer_type` — the specialist takes over from there.
- **Hitch vs category change:** Never call `set_trailer_type` with hitch-only words (gooseneck, bumper pull, tag-along). **Do** call `set_trailer_type` when the customer names a **different trailer category** than the session (e.g. switching from Dump to Utility) — see CURRENT SESSION when shown.
- If the customer already has an active category and their new message only refines **hitch**, length, budget, or the same category need, **do not** change `trailer_type`.
- If CURRENT SESSION already shows a trailer category and the customer only describes **what they are hauling** (e.g. an ATV, lumber, gravel) without naming a **different** trailer category, **do not** call `set_trailer_type` again — category is already set; the specialist will qualify them.
"""

_SPECIALIST_PROMPT_TEMPLATE = """You are the {trailer_type} Trailer Specialist for TrailerPlace (Wharton TX, 979-532-1486).

## YOUR ROLE
The customer is looking for a **{trailer_type}** trailer. Your job is to:
1. First, call `fetch_trailer_fields` with trailer_type="{trailer_type}" to get the required and optional slots.
2. Walk through **required slots in order** — ask ONE question per turn.
3. After each answer, call `record_slot_answer` with that slot and value before asking the next question.
4. Evaluate optional slots only after all required slots appear under **SLOTS COLLECTED SO FAR** — ask an optional question only if it matters for this customer.
5. Only when every required slot is recorded below, call `search_trailers`. **Early calls are rejected by the system.**

## AVAILABLE TOOLS
- **fetch_trailer_fields**: Call first to learn required_slots, optional_slots, and questions for this trailer type.
- **record_slot_answer**: Call immediately after the customer answers each qualification question (required or optional).
- **search_trailers**: Call only after all required slots are listed under **SLOTS COLLECTED SO FAR**. Combine slot values into `query` and use filter fields when known.
- **set_trailer_type**: Only when the customer switches to a **different trailer category** than **{trailer_type}** (not for aluminum **style** words like Utility/Equipment while `base_category` is still empty — use `record_slot_answer` for those).

## SLOT COLLECTION RULES
- Trust **SLOTS COLLECTED SO FAR** — ask only slots not yet marked collected.
- **Utility + lightweight haul**: When `trailer_type` is Utility and the system has classified the haul as lightweight, weight slots will **not** appear below — do not ask about weight; the system sets payload silently for search.
- For non-Utility categories, always collect every required slot shown below (including weight when listed).

## CATEGORY PIVOT
If the customer names a **different trailer category** than **{trailer_type}**, call `set_trailer_type` first, then `fetch_trailer_fields`, then collect **all** required slots for the new category via `record_slot_answer` before `search_trailers`.
- **Never** call `set_trailer_type` for hitch-only wording — use `hitch_type` on `search_trailers`.
- **Aluminum exception:** While **{trailer_type}** is **Aluminum** and the aluminum **style** (`base_category`) slot is not yet collected, words like **Utility** / **Equipment** / **Enclosed** are **sub-types** for search — record them with **`record_slot_answer(slot='base_category', ...)`**. Do **not** call `set_trailer_type` for those style answers until `base_category` is recorded (the system blocks mistaken pivots). For a true category change (e.g. they want **Dump** instead), `set_trailer_type` is appropriate.

## SEARCH CALL RULES
- Normalize units: tons/kg → lbs; m/cm/in → ft.
- Set `hitch_type` when the customer clearly prefers Bumper Pull or Gooseneck.
- Set `required_payload_lbs`, `required_length_ft`, `required_gvwr_lbs` from collected slots when applicable.
- For **Aluminum**: keep **`category_subcategory` = `Aluminum`** and set **`subcategory`** to the style from **`base_category`** (Utility, Equipment, …). Put haul context in **`query`**; **`payload_need`** is weight-only for **`required_payload_lbs`**.
- Build a rich `query` from all collected slots.

## SLOTS COLLECTED SO FAR
{slots_summary}
"""

RECOMMENDATION_PROMPT = f"""You are the Recommendation Specialist for TrailerPlace (Wharton TX, 979-532-1486).
Search results have been retrieved. Your job is to present them clearly and move toward a sale.

## RECOMMENDATIONS (same search context)
Use **log_product_interest** when the customer clearly picks a specific unit after you recommend trailers.

## SHOW MORE AND FILTER UPDATES
- If the customer asks for **more results** ("show me more", "any others?", "what else?"), call `search_trailers` with **more_results=true** immediately — **do not** ask clarifying questions first.
- Reuse the values from **PREVIOUS SEARCH FILTERS** below. Override a filter field **only** when the customer's **latest message** explicitly states a new constraint (e.g. at least 20 ft, under $15k, gooseneck only, payload 2500 lbs).
- If they add a new constraint **without** saying "more", still call `search_trailers` with **more_results=true** and update only the mentioned fields; keep all others from PREVIOUS SEARCH FILTERS.
- Never ask the customer to restate constraints already captured below.

## AVAILABLE TOOLS
- **faq_tool**: Non-sales FAQ (contact/human, financing, trade-in, service/parts, store info). **Call this tool before answering** those questions in plain text.
- **log_product_interest**: Call when the customer clearly expresses interest in a specific unit. Use the exact full listing title from the results.
- **search_trailers**: For more inventory (`more_results=true`) or updated constraints as above.
- **set_trailer_type**: When the customer switches to a **different trailer category** — then the specialist will re-qualify on the next turn.
- **record_slot_answer**: Rarely needed here; use if recording a slot answer before searching again.

## PRESENTATION FORMAT
For each trailer, use this exact layout:
1. **Title as Markdown link**: [Full listing title](listing URL)
2. **Bullet block** (max 6 bullets): • Label: value — only include non-empty fields; pick the ones most relevant to the customer's stated needs (Price, Length, GVWR, Payload Capacity, Hitch Type, Color).
3. **Why-it-fits**: One or two sentences grounding this trailer in what the customer said. Mandatory — never skip this.

Number the trailers 1., 2., 3. if multiple. For a single trailer, numbering is optional.

End with exactly ONE warm closing question (vary the wording each turn).

## RULES
- Never invent specs — only use values from the search results.
- Always mention financing and delivery are available.
- If results aren't a perfect match, say so honestly and show the closest available option.
- After log_product_interest succeeds, direct the customer to: {_SITE_URL}
- Never claim a human has already contacted them — only say "your query has been logged, the team will follow up soon."

## FAQ (non-sales)
For contact/human rep, financing, trade-in, service/parts, or store info, you **must** call **`faq_tool` first** before answering in text. Pass `title`, `user_summary`, and `reply_text`. After the tool succeeds, your visible reply must match the tool result (canonical wording is applied when needed). Then offer to keep helping with trailers if relevant.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Pinecone helper (thin wrapper around agent.py search)
# ─────────────────────────────────────────────────────────────────────────────


def _log_lg_pinecone_attempt(
    query: str,
    pinecone_filter: Optional[dict],
    relaxed: bool,
    tk: int,
    mlist: list[dict],
) -> None:
    """Structured + JSON log for each Pinecone query (parity with TrailerAgent._search)."""
    phase = "relaxed" if relaxed else "strict"
    logger.info(
        "PINECONE_FILTER | phase=%s | top_k=%s | filter=%s",
        phase,
        tk,
        json.dumps(pinecone_filter, ensure_ascii=True) if pinecone_filter else "None",
    )
    attempt: dict[str, Any] = {
        "source": "langgraph",
        "phase": phase,
        "query": query,
        "top_k": tk,
        "pinecone_filter": pinecone_filter,
        "match_count": len(mlist),
        "matches": [_match_summary_for_log(x) for x in mlist],
    }
    _log_product_fetch_event(attempt)


def _run_search(
    query: str,
    trailer_filter: TrailerFilter,
    top_k: int = SEARCH_TOP_K,
    *,
    exclude_keys: Optional[set[str]] = None,
    exclude_urls: Optional[set[str]] = None,
) -> list[TrailerListing]:
    """
    Execute a Pinecone vector search and return TrailerListing objects (pre-rerank).

    When exclude_keys / exclude_urls are set (show-more path), fetch a larger top_k and
    drop matches already stored for this session in shown_listings/{session_id}.json.
    """
    from openai import OpenAI
    from pinecone import Pinecone

    ex_k = set(exclude_keys) if exclude_keys else set()
    ex_u = set(exclude_urls) if exclude_urls else set()
    have_ex = bool(ex_k or ex_u)

    def _effective_top_k(k: int) -> int:
        if not have_ex:
            return k
        pool = len(ex_k) + len(ex_u)
        return min(
            SEARCH_TOP_K_MORE_MAX,
            max(SEARCH_TOP_K_MORE, pool + SEARCH_MAX_RECOMMENDATIONS * 2, k),
        )

    def _match_excluded(match: dict) -> bool:
        md = match.get("metadata", {}) or {}
        url = str(md.get("url", "") or "").strip().lower()
        if ex_u and url and url in ex_u:
            return True
        if ex_k and _canonical_listing_key_from_match(match) in ex_k:
            return True
        return False

    def _filter_excluded(raw: list[dict]) -> list[dict]:
        if not have_ex:
            return raw
        return [m for m in raw if not _match_excluded(m)]

    oai = OpenAI(api_key=OPENAI_API_KEY)
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    vec = oai.embeddings.create(model=embedding_model, input=[query]).data[0].embedding

    pc = Pinecone(api_key=PINECONE_API_KEY).Index(INDEX_NAME)
    pf = _build_pinecone_filter(trailer_filter)
    k0 = _effective_top_k(top_k)

    relaxed_f: dict = {}
    f_arg: Optional[dict] = None
    if pf:
        relaxed_f = {k: v for k, v in pf.items() if k in ("condition", "price", "hitch_type")}
        f_arg = relaxed_f if relaxed_f else None

    matches: list[dict] = []
    if not have_ex:
        result = pc.query(vector=vec, top_k=k0, include_metadata=True, filter=pf)
        matches = result.get("matches", [])
        _log_lg_pinecone_attempt(query, pf, False, k0, matches)
        if not matches and pf:
            result = pc.query(
                vector=vec, top_k=k0, include_metadata=True, filter=f_arg or None
            )
            matches = result.get("matches", [])
            _log_lg_pinecone_attempt(query, f_arg, True, k0, matches)
    else:
        steps: list[tuple[Optional[dict], bool, int]] = [
            (pf, False, k0),
        ]
        if pf:
            steps.append((f_arg, True, k0))
        steps.append((pf, False, SEARCH_TOP_K_MORE_MAX))
        if pf:
            steps.append((f_arg, True, SEARCH_TOP_K_MORE_MAX))
        for filt, is_relaxed, tk in steps:
            raw = pc.query(
                vector=vec, top_k=tk, include_metadata=True, filter=filt
            ).get("matches", [])
            _log_lg_pinecone_attempt(query, filt, is_relaxed, tk, raw)
            filtered = _filter_excluded(raw)
            dropped_by_url = sum(
                1
                for m in raw
                if str((m.get("metadata") or {}).get("url") or "").strip().lower() in ex_u
            )
            logger.info(
                "LG_PINECONE_EXCLUDE | raw=%s after=%s exclude_keys=%s exclude_urls=%s "
                "dropped_by_url=%s tk=%s relaxed=%s",
                len(raw),
                len(filtered),
                len(ex_k),
                len(ex_u),
                dropped_by_url,
                tk,
                is_relaxed,
            )
            if filtered:
                matches = filtered
                break

    deduped, _ = _dedupe_matches(matches)
    return [_metadata_to_listing(m) for m in deduped]


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph agent class
# ─────────────────────────────────────────────────────────────────────────────

class TrailerAgentLG:
    """
    LangGraph-based conversational trailer recommendation agent.

    Usage
    -----
    agent = TrailerAgentLG(customer=CustomerContact(...))
    reply, listings = agent.chat("I need a dump trailer for hauling gravel")
    """

    def __init__(
        self,
        customer: Optional[CustomerContact] = None,
    ):
        self._customer = customer
        self._llm = ChatOpenAI(
            model=OPENAI_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=0,
        )
        self._lightweight_cache: dict[str, bool] = {}
        self._state: SessionState = self._initial_state()
        self._graph = self._build_graph()

    # ── State initialisation ─────────────────────────────────────────────────

    def _initial_state(self) -> SessionState:
        return SessionState(
            messages=[],
            intent=None,
            trailer_type=None,
            slots_collected={},
            slots_asked=[],
            utility_lightweight_decided=None,
            required_slots=[],
            optional_slots=[],
            search_results=[],
            search_results_for_category=None,
            api_listings_this_turn=[],
            last_search_args=None,
            recommendation_entry_due=False,
            is_interested=False,
            interested_item=None,
            customer_full_name=self._customer.full_name if self._customer else None,
            customer_email=self._customer.email if self._customer else None,
            customer_phone=self._customer.phone if self._customer else None,
            session_id=None,
            next_node=None,
            client_shown_urls=[],
        )

    # ── Graph construction ───────────────────────────────────────────────────

    def _build_graph(self) -> Any:
        builder = StateGraph(SessionState)
        builder.add_node("master_router_node", self._master_router_node)
        builder.add_node("specialist_node", self._specialist_node)
        builder.add_node("recommendation_node", self._recommendation_node)

        builder.set_entry_point("master_router_node")

        builder.add_conditional_edges(
            "master_router_node",
            self._route_after_master,
            {
                "specialist_node": "specialist_node",
                "recommendation_node": "recommendation_node",
                END: END,
            },
        )
        builder.add_conditional_edges(
            "specialist_node",
            self._route_after_specialist,
            {
                "master_router_node": "master_router_node",
                "recommendation_node": "recommendation_node",
                END: END,
            },
        )
        builder.add_edge("recommendation_node", END)

        return builder.compile()

    # ── Routing conditions ────────────────────────────────────────────────────

    @staticmethod
    def _route_after_master(state: SessionState) -> str:
        if state.get("next_node") == END:
            return END
        tt = state.get("trailer_type")
        if not tt:
            return END
        sr = state.get("search_results") or []
        sfc = state.get("search_results_for_category")
        if sr and sfc == tt:
            return "recommendation_node"
        return "specialist_node"

    @staticmethod
    def _route_after_specialist(state: SessionState) -> str:
        if not state.get("trailer_type"):
            return "master_router_node"
        sr = state.get("search_results") or []
        sfc = state.get("search_results_for_category")
        tt = state.get("trailer_type")
        # Enter recommendation whenever we have live search results for this category — not only
        # the same invoke as search_trailers (follow-up turns express interest without re-search).
        if sr and sfc == tt:
            return "recommendation_node"
        return END

    def _classify_lightweight_haul(self, haul_text: str) -> bool:
        """
        Mini-classifier for Utility trailers: True if haul is lightweight (~under 1500 lbs).
        Uses a short LLM JSON reply with fallback to keyword/heuristic checks.
        """
        key = (haul_text or "").strip().lower()
        if not key:
            return False
        if key in self._lightweight_cache:
            return self._lightweight_cache[key]

        fallback = is_lightweight_haul(haul_text)
        result_bool = fallback
        try:
            mini = ChatOpenAI(
                model=OPENAI_MODEL,
                api_key=OPENAI_API_KEY,
                temperature=0,
            )
            resp = mini.invoke(
                [
                    {
                        "role": "system",
                        "content": (
                            "You classify whether the haul item for a UTILITY trailer is lightweight "
                            "(combined load roughly under 1500 lbs). "
                            "If the customer's description mentions any of these items or phrases "
                            "(substring match, case-insensitive), you MUST classify it as lightweight — "
                            "i.e. your JSON must be "
                            '{"lightweight": true}. Phrases: '
                            + ", ".join(_LIGHTWEIGHT_KEYWORDS)
                            + ". Otherwise judge whether the combined load is roughly under 1500 lbs. "
                            "Reply with strict JSON only, no markdown: "
                            '{"lightweight": true} or {"lightweight": false}'
                        ),
                    },
                    {"role": "user", "content": haul_text},
                ]
            )
            raw = str(getattr(resp, "content", "") or "").strip()
            if raw.startswith("```"):
                raw = raw.replace("```json", "").replace("```", "").strip()
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                m = re.search(r"\{[^{}]*\}", raw)
                if m:
                    parsed = json.loads(m.group(0))
                else:
                    raise
            result_bool = bool(parsed.get("lightweight"))
        except Exception:
            result_bool = fallback

        self._lightweight_cache[key] = result_bool
        logger.info(
            "LIGHTWEIGHT_CLASSIFIER | trailer_type=Utility | text=%s | result=%s",
            key[:240],
            result_bool,
        )
        return result_bool

    def _run_set_trailer_type_tool(
        self, raw_type: str, state: SessionState
    ) -> tuple[str, dict[str, Any]]:
        """Shared handler for set_trailer_type from specialist or recommendation node."""
        canonical, err = _validate_set_trailer_type_arg(raw_type)
        if err:
            logger.info(
                "TRAILER_TYPE_REJECTED | raw=%s | reason=%s",
                raw_type,
                err.get("error"),
            )
            return json.dumps(err, ensure_ascii=True), {}

        slots = state.get("slots_collected") or {}
        if (
            state.get("trailer_type") == "Aluminum"
            and "base_category" not in slots
            and canonical is not None
            and canonical != "Aluminum"
            and canonical in _ALUMINUM_AMBIGUOUS_CATEGORY_PIVOTS
        ):
            err_guard: dict[str, Any] = {
                "ok": False,
                "error": "aluminum_base_category_slot_required",
                "received": raw_type,
                "normalized": canonical,
                "message": (
                    f"'{canonical}' matches an aluminum inventory style, not a new top-level category "
                    "while the aluminum style question is still unanswered. Keep trailer_type as "
                    "Aluminum and record the answer with record_slot_answer(slot='base_category', value=...). "
                    "Use set_trailer_type only for a real category change (e.g. Dump) or after base_category "
                    "has been collected."
                ),
            }
            logger.info(
                "TRAILER_TYPE_REJECTED | aluminum_guard | missing_base_category | proposed=%s",
                canonical,
            )
            return json.dumps(err_guard, ensure_ascii=True), {}

        extra: dict[str, Any] = {}
        prev_tt = state.get("trailer_type")
        extra["trailer_type"] = canonical
        extra["slots_collected"] = {}
        extra["slots_asked"] = []
        extra["utility_lightweight_decided"] = None
        spec = get_trailer_fields_as_dict(canonical)
        extra["required_slots"] = spec["required_slots"]
        extra["optional_slots"] = spec["optional_slots"]

        if prev_tt and prev_tt != canonical:
            extra["search_results"] = []
            extra["search_results_for_category"] = None
            extra["recommendation_entry_due"] = False
            extra["is_interested"] = False
            extra["interested_item"] = None
            extra["last_search_args"] = None
            logger.info(
                "TRAILER_TYPE_PIVOT | from=%s to=%s | cleared_search",
                prev_tt,
                canonical,
            )
        else:
            logger.info("TRAILER_TYPE_UPDATE | trailer_type=%s", canonical)

        return json.dumps(
            {"ok": True, "trailer_type": canonical, "note": "category updated"}
        ), extra

    def _execute_faq_tool(
        self, args: dict[str, Any], state: SessionState
    ) -> tuple[str, dict[str, Any]]:
        """Validate faq_tool args (structural), enqueue async FAQ email, return JSON for the tool message."""
        extra: dict[str, Any] = {}
        faq_type = str(args.get("faq_type") or "").strip()
        title = str(args.get("title") or "").strip()
        user_summary = str(args.get("user_summary") or "").strip()
        reply_text = str(args.get("reply_text") or "").strip()

        _max_title = 200
        _max_summary = 2000
        _max_reply = 12000

        logger.info(
            "FAQ_TOOL_ATTEMPT | faq_type=%s | title_len=%s | reply_len=%s",
            faq_type or "(none)",
            len(title),
            len(reply_text),
        )

        def _reject(reason: str) -> tuple[str, dict[str, Any]]:
            logger.info("FAQ_TOOL_REJECTED | reason=%s | faq_type=%s", reason, faq_type or "(none)")
            return json.dumps({"ok": False, "error": reason}), extra

        if faq_type not in FAQ_TYPES_SET:
            logger.info(
                "FAQ_TOOL_REJECTED | reason=invalid_faq_type | faq_type=%s",
                faq_type or "(none)",
            )
            return (
                json.dumps({"ok": False, "error": "invalid_faq_type", "faq_type": faq_type}),
                extra,
            )
        if not title or len(title) > _max_title:
            return _reject("invalid_title")
        if not user_summary or len(user_summary) > _max_summary:
            return _reject("invalid_user_summary")
        if not reply_text or len(reply_text) > _max_reply:
            return _reject("invalid_reply_text")

        effective = _effective_faq_reply_text(faq_type, reply_text)
        if not effective:
            return _reject("invalid_effective_reply")

        if not _faq_reply_matches_canonical(faq_type, reply_text):
            logger.info(
                "FAQ_FALLBACK_REPLY_USED | reason=non_canonical_model_reply | faq_type=%s",
                faq_type,
            )
        effective_summary = _effective_faq_email_summary(faq_type, user_summary)
        if effective_summary != user_summary:
            logger.info(
                "FAQ_SUMMARY_NORMALIZED | faq_type=%s | used_deterministic_summary=true",
                faq_type,
            )

        full_name = (
            (self._customer.full_name if self._customer else None)
            or state.get("customer_full_name")
            or "Unknown"
        )
        email = (self._customer.email if self._customer else None) or state.get("customer_email")
        phone = (
            (self._customer.phone if self._customer else None)
            or state.get("customer_phone")
            or "Not provided"
        )

        enqueue_faq_email_notification(
            full_name=str(full_name),
            email=str(email) if email else None,
            phone=str(phone),
            subject=title,
            summary_line=f"[{faq_type}] {effective_summary}",
        )
        logger.info("FAQ_TOOL_SUCCESS_ENQUEUED | faq_type=%s | title=%s", faq_type, title[:120])
        return (
            json.dumps({"ok": True, "effective_reply_text": effective}),
            extra,
        )

    # ── Node: master_router_node ──────────────────────────────────────────────

    def _master_router_node(self, state: SessionState) -> dict:
        """Greet, route intent, identify trailer category."""
        logger.info(
            "LANGGRAPH_NODE | node=master_router_node | trailer_type=%s",
            state.get("trailer_type"),
        )
        system = MASTER_ROUTER_PROMPT
        if self._customer:
            first_name = (self._customer.full_name or "").split()[0]
            system += (
                f"\n\n## VERIFIED CUSTOMER\n"
                f"Full Name: {self._customer.full_name}\n"
                f"Email: {self._customer.email or '(not provided)'}\n"
                f"Phone: {self._customer.phone}\n"
                f"Greet them by their first name: {first_name}. Do NOT ask for contact info again."
            )

        cur_tt = state.get("trailer_type")
        if cur_tt:
            system += (
                f"\n\n## CURRENT SESSION\n"
                f"Trailer category already set: **{cur_tt}**.\n"
                "- If the customer names a **different trailer category** (e.g. Utility vs Dump), call "
                "`set_trailer_type` immediately with the new category. That clears prior search context "
                "so the specialist can collect fresh qualification slots.\n"
                "- Do **not** call `set_trailer_type` for **hitch-only** wording (gooseneck, bumper pull, "
                "tag-along); those are hitch preferences, not categories.\n"
                "- If their message only refines hitch, length, budget, or the same category need, "
                "**do not** change `trailer_type`."
            )
            if cur_tt == "Aluminum":
                system += (
                    "\n- **Aluminum:** If the customer says **utility**, **equipment**, **enclosed**, etc., "
                    "that is an aluminum **style** (sub-type), **not** a category change — the specialist "
                    "must use **`record_slot_answer(slot='base_category', ...)`**. Do **not** call "
                    "`set_trailer_type('Utility')` / similar for those answers while the style slot is still "
                    "open. `set_trailer_type` is for a genuinely different category (e.g. **Dump**)."
                )

        messages = self._messages_for_llm(state, system)
        tools = [FAQ_TOOL, SET_TRAILER_TYPE_TOOL]
        llm_with_tools = self._llm.bind_tools(tools)

        updates: dict[str, Any] = {}
        all_new_messages: list[BaseMessage] = []
        current_messages = messages
        shadow: dict[str, Any] = dict(state)
        faq_reply_override: Optional[str] = None

        for _ in range(4):
            response = llm_with_tools.invoke(current_messages)
            all_new_messages.append(response)

            if not (hasattr(response, "tool_calls") and response.tool_calls):
                break

            tool_messages: list[ToolMessage] = []
            for tc in response.tool_calls:
                fn = str(tc.get("name", "")).strip()
                args = dict(tc.get("args") or {})
                tid = str(tc.get("id", "") or "")

                if fn == "faq_tool":
                    body, extra = self._execute_faq_tool(args, shadow)
                    updates.update(extra)
                    shadow.update(extra)
                    try:
                        data = json.loads(body)
                        if data.get("ok"):
                            eff = str(data.get("effective_reply_text") or "").strip()
                            faq_reply_override = eff or str(args.get("reply_text") or "").strip()
                    except json.JSONDecodeError:
                        pass
                    updates["next_node"] = END
                    tool_messages.append(ToolMessage(content=body, tool_call_id=tid))
                elif fn == "set_trailer_type":
                    raw_type = str(args.get("trailer_type", "")).strip()
                    result_json, extra = self._run_set_trailer_type_tool(raw_type, shadow)
                    updates.update(extra)
                    shadow.update(extra)
                    tool_messages.append(ToolMessage(content=result_json, tool_call_id=tid))
                else:
                    tool_messages.append(
                        ToolMessage(
                            content=json.dumps({"ok": False, "error": f"unknown_tool:{fn}"}),
                            tool_call_id=tid,
                        )
                    )

            all_new_messages.extend(tool_messages)
            current_messages = list(messages) + all_new_messages

        updates["messages"] = all_new_messages
        if faq_reply_override:
            _ensure_faq_assistant_reply(all_new_messages, faq_reply_override)
        return updates

    # ── Node: specialist_node ─────────────────────────────────────────────────

    def _specialist_node(self, state: SessionState) -> dict:
        """Collect qualification slots then trigger search."""
        trailer_type = state.get("trailer_type", "Unknown")
        logger.info(
            "LANGGRAPH_NODE | node=specialist_node | trailer_type=%s",
            trailer_type,
        )
        slots = state.get("slots_collected", {}) or {}
        required = list(state.get("required_slots", []))
        optional = state.get("optional_slots", [])

        haul_item_text = (
            str(slots.get("haul_item") or "")
            + " " + str(slots.get("vehicle_type") or "")
            + " " + str(slots.get("haul_material") or "")
        )
        for _msg in reversed(state.get("messages", [])):
            if isinstance(_msg, HumanMessage):
                haul_item_text += " " + str(_msg.content or "")
                break
            if isinstance(_msg, dict) and _msg.get("role") == "user":
                haul_item_text += " " + str(_msg.get("content") or "")
                break

        lightweight_updates: dict[str, Any] = {}
        if trailer_type == "Utility":
            is_light = self._classify_lightweight_haul(haul_item_text)
            lightweight_updates["utility_lightweight_decided"] = is_light
            if is_light:
                required = [s for s in required if s not in WEIGHT_SLOT_NAMES]
                lightweight_updates["required_slots"] = required
                logger.info("LIGHTWEIGHT_DETECTED | utility | weight slots stripped")

        shadow: dict[str, Any] = dict(state)
        shadow.update(lightweight_updates)
        sc = dict(shadow.get("slots_collected") or {})

        slots_summary_lines = []
        for slot in required + optional:
            val = sc.get(slot)
            if val is not None:
                slots_summary_lines.append(f"  {slot}: {val} ✓")
            else:
                slots_summary_lines.append(f"  {slot}: (not yet collected)")
        slots_summary = "\n".join(slots_summary_lines) if slots_summary_lines else "  (none yet)"

        system = _SPECIALIST_PROMPT_TEMPLATE.format(
            trailer_type=trailer_type,
            slots_summary=slots_summary,
        )
        if trailer_type == "Utility" and shadow.get("utility_lightweight_decided") is True:
            req_display = ", ".join(required) if required else "(none)"
            system += (
                "\n\n## AUTHORITATIVE REQUIRED SLOTS (Utility lightweight)\n"
                f"The only required qualification slots for search are: **{req_display}**. "
                "Do **not** ask about haul weight or total weight; payload is set silently.\n"
                "Ignore any earlier `fetch_trailer_fields` tool message that still lists "
                "**haul_weight_lbs** — that requirement was superseded when the haul was "
                "classified as lightweight."
            )

        messages = self._messages_for_specialist_llm(shadow, system)
        tools = [
            FETCH_TRAILER_FIELDS_TOOL,
            RECORD_SLOT_ANSWER_TOOL,
            SEARCH_TRAILERS_TOOL,
            SET_TRAILER_TYPE_TOOL,
        ]
        llm_with_tools = self._llm.bind_tools(tools)

        updates: dict = {"messages": [], **lightweight_updates}
        all_new_messages: list[BaseMessage] = []

        current_messages = messages
        max_iterations = 6
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            response = llm_with_tools.invoke(current_messages)
            all_new_messages.append(response)

            if not (hasattr(response, "tool_calls") and response.tool_calls):
                break

            tool_messages = []
            for tc in response.tool_calls:
                fn = tc["name"]
                args = tc.get("args") or {}
                tool_result, extra_updates = self._execute_specialist_tool(
                    fn, args, shadow
                )
                updates.update(extra_updates)
                shadow.update(extra_updates)
                tool_messages.append(
                    ToolMessage(content=tool_result, tool_call_id=tc["id"])
                )
                if fn == "search_trailers" and updates.get("search_results"):
                    all_new_messages.extend(tool_messages)
                    updates["messages"] = all_new_messages
                    updates["recommendation_entry_due"] = True
                    return updates

            all_new_messages.extend(tool_messages)
            current_messages = list(messages) + all_new_messages

        updates["messages"] = all_new_messages
        updates["recommendation_entry_due"] = False
        return updates

    def _execute_specialist_tool(
        self, fn: str, args: dict, state: SessionState
    ) -> tuple[str, dict]:
        """Execute a tool call from the specialist node. Returns (tool_result_str, state_updates)."""
        extra: dict = {}

        if fn == "record_slot_answer":
            slot = str(args.get("slot", "")).strip()
            value = args.get("value")
            req = list(state.get("required_slots") or [])
            opt = list(state.get("optional_slots") or [])
            if slot not in req and slot not in opt:
                return (
                    json.dumps(
                        {
                            "ok": False,
                            "error": "invalid_slot",
                            "slot": slot,
                            "message": "Use a slot name from required_slots or optional_slots.",
                        }
                    ),
                    {},
                )
            prev_sc = dict(state.get("slots_collected") or {})
            prev_sc[slot] = value
            extra["slots_collected"] = prev_sc
            asked = list(state.get("slots_asked") or [])
            if slot not in asked:
                asked.append(slot)
            extra["slots_asked"] = asked
            logger.info("SLOT_RECORDED | slot=%s | value=%s", slot, value)
            return json.dumps({"ok": True, "slot": slot, "value": value}), extra

        if fn == "fetch_trailer_fields":
            trailer_type = str(args.get("trailer_type", state.get("trailer_type", ""))).strip()
            spec = get_trailer_fields_as_dict(trailer_type)
            if trailer_type == "Utility" and state.get("utility_lightweight_decided") is True:
                spec = _strip_weight_slots_from_spec_dict(spec)
            extra["required_slots"] = spec["required_slots"]
            extra["optional_slots"] = spec["optional_slots"]
            logger.info("FETCH_FIELDS | trailer_type=%s | required=%s", trailer_type, spec["required_slots"])
            return json.dumps(spec), extra

        if fn == "set_trailer_type":
            raw_type = str(args.get("trailer_type", "")).strip()
            return self._run_set_trailer_type_tool(raw_type, state)

        if fn == "search_trailers":
            return self._execute_search_tool(args, state, extra)

        return json.dumps({"ok": False, "error": f"unknown_tool:{fn}"}), extra

    def _execute_search_tool(
        self, args: dict, state: SessionState, extra: dict
    ) -> tuple[str, dict]:
        """Run a Pinecone search, update state with results."""
        args = dict(args or {})
        more_results = bool(args.get("more_results", False))

        if more_results:
            prev = state.get("last_search_args") or {}
            merged_keys: list[str] = []
            for k in _LAST_SEARCH_MERGE_KEYS:
                av = args.get(k)
                if av in (None, "") and prev.get(k) is not None:
                    args[k] = prev[k]
                    merged_keys.append(k)
            q = str(args.get("query") or "").strip()
            if not q and prev.get("query"):
                args["query"] = prev["query"]
                merged_keys.append("query")
            logger.info("SHOW_MORE_REUSE | merged_keys=%s", merged_keys)

        if not more_results:
            required = list(state.get("required_slots") or [])
            if (
                state.get("trailer_type") == "Utility"
                and state.get("utility_lightweight_decided") is True
            ):
                required = [s for s in required if s not in WEIGHT_SLOT_NAMES]
            collected = state.get("slots_collected") or {}
            missing = [s for s in required if s not in collected]
            if missing:
                logger.info("SEARCH_REJECTED_MISSING_SLOTS | missing=%s", missing)
                return (
                    json.dumps(
                        {
                            "ok": False,
                            "error": "missing_required_slots",
                            "missing": missing,
                            "message": (
                                f"Ask the customer about: {missing[0]} before calling search_trailers."
                            ),
                        }
                    ),
                    {},
                )

        query = str(args.get("query", "")).strip()
        required_payload_lbs = _coerce_required_payload_lbs(args.get("required_payload_lbs"))
        required_length_ft = _coerce_required_length_ft(args.get("required_length_ft"))
        required_gvwr_lbs = _coerce_required_payload_lbs(args.get("required_gvwr_lbs"))

        exclude_urls: Optional[set[str]] = None
        if more_results:
            sid = (state.get("session_id") or "").strip()
            client = list(state.get("client_shown_urls") or [])
            exclude_urls = merge_shown_urls_for_show_more(sid, client)
            logger.info(
                "LG_SHOW_MORE | session_id=%s exclude_urls=%s client_urls=%s",
                sid or "(none)",
                len(exclude_urls),
                len(client),
            )
            if not sid and not client:
                logger.warning(
                    "search_trailers more_results=True but no session_id and no client_shown_urls"
                )

        if required_payload_lbs is None:
            required_payload_lbs = _extract_weight_lbs_from_text(query)
        if required_length_ft is None:
            required_length_ft = _extract_length_ft_from_text(query)

        if (
            state.get("trailer_type") == "Utility"
            and state.get("utility_lightweight_decided") is True
            and required_payload_lbs is None
        ):
            required_payload_lbs = 1000.0

        if state.get("trailer_type") == "Aluminum":
            collected = state.get("slots_collected") or {}
            if not args.get("subcategory"):
                sub_slot = _pinecone_subcategory_from_aluminum_base_slot(collected.get("base_category"))
                if sub_slot:
                    args["subcategory"] = sub_slot
            if required_payload_lbs is None:
                required_payload_lbs = _coerce_required_payload_lbs(collected.get("payload_need"))
            args["category_subcategory"] = "Aluminum"

        hitch_type = args.get("hitch_type")
        if not hitch_type:
            hitch_type = _infer_hitch_type_from_text(query)
        if not hitch_type:
            hitch_type = _infer_hitch_from_recent_user_messages(state)
        if hitch_type:
            hitch_type = normalize_hitch(hitch_type) or hitch_type
        logger.info("LG_HITCH | hitch_type=%s", hitch_type)

        cat_sub = args.get("category_subcategory") or state.get("trailer_type")

        extra["last_search_args"] = {
            "query": query,
            "condition": args.get("condition"),
            "price_min": args.get("price_min"),
            "price_max": args.get("price_max"),
            "category_subcategory": cat_sub,
            "subcategory": args.get("subcategory"),
            "make": args.get("make"),
            "color": args.get("color"),
            "hitch_type": hitch_type,
            "required_payload_lbs": required_payload_lbs,
            "required_length_ft": required_length_ft,
            "required_gvwr_lbs": required_gvwr_lbs,
        }

        trailer_filter = TrailerFilter(
            condition=args.get("condition"),
            price_min=args.get("price_min"),
            price_max=args.get("price_max"),
            category_subcategory=cat_sub,
            subcategory=args.get("subcategory"),
            make=args.get("make"),
            color=args.get("color"),
            hitch_type=hitch_type,
            required_length_ft=required_length_ft,
            required_gvwr_lbs=required_gvwr_lbs,
        )

        listings = _run_search(
            query,
            trailer_filter,
            top_k=SEARCH_TOP_K,
            exclude_urls=exclude_urls,
        )
        logger.info(
            "LG_RERANK_INPUT | listing_count=%s | payload_lbs=%s | length_ft=%s | gvwr_lbs=%s | more_results=%s",
            len(listings),
            required_payload_lbs,
            required_length_ft,
            required_gvwr_lbs,
            more_results,
        )

        # Rerank using the staticmethod-equivalent helper from agent.py
        from src.agent import TrailerAgent as _TA

        _dummy = _TA.__new__(_TA)
        reranked, _ = _dummy._rerank_by_fit(
            listings,
            required_payload_lbs=required_payload_lbs,
            required_length_ft=required_length_ft,
            required_gvwr_lbs=required_gvwr_lbs,
            desired_count=SEARCH_MAX_RECOMMENDATIONS,
        )
        category_for_make = cat_sub or state.get("trailer_type")
        reranked, _mp_dbg = _apply_category_make_priority(reranked, category_for_make)
        selected = reranked[:SEARCH_MAX_RECOMMENDATIONS]

        if not selected:
            extra["search_results"] = []
            extra["search_results_for_category"] = None
            extra["api_listings_this_turn"] = []
            extra["recommendation_entry_due"] = False
            return "No trailers found matching those criteria.", extra

        result_dicts = []
        for i, lst in enumerate(selected, 1):
            result_dicts.append({
                "rank": i,
                "title": lst.title,
                "condition": lst.condition,
                "price": lst.price_display or (f"${lst.price:,.0f}" if lst.price else "Call for price"),
                "category": lst.category_subcategory,
                "make": lst.make,
                "color": lst.color,
                "hitch_type": lst.hitch_type,
                "year": lst.year,
                "length": lst.length,
                "width": lst.width,
                "axles": lst.axles,
                "gvwr": lst.gvwr,
                "payload_capacity": lst.payload_capacity,
                "material": lst.trailer_material,
                "floor": lst.floor,
                "url": lst.url,
                "relevance_score": lst.score,
            })

        extra["search_results"] = result_dicts
        extra["api_listings_this_turn"] = list(result_dicts)
        tt = state.get("trailer_type")
        extra["search_results_for_category"] = tt
        logger.info("SEARCH_COMPLETE | trailer_type=%s | result_count=%s", tt, len(result_dicts))
        return json.dumps(result_dicts, indent=2), extra

    # ── Node: recommendation_node ─────────────────────────────────────────────

    def _recommendation_node(self, state: SessionState) -> dict:
        """Present search results, handle interest logging."""
        search_results = state.get("search_results", [])
        logger.info(
            "LANGGRAPH_NODE | node=recommendation_node | trailer_type=%s | search_result_count=%s",
            state.get("trailer_type"),
            len(search_results) if isinstance(search_results, list) else 0,
        )
        results_json = json.dumps(search_results, indent=2) if search_results else "[]"
        prev_filters_json = json.dumps(
            state.get("last_search_args") or {}, indent=2, ensure_ascii=True
        )

        system = (
            RECOMMENDATION_PROMPT
            + "\n\n## PREVIOUS SEARCH FILTERS\n"
            "Reuse these values when calling `search_trailers` unless the customer overrides "
            "them in their latest message.\n```json\n"
            f"{prev_filters_json}\n```\n"
            + f"\n\n## CURRENT SEARCH RESULTS\n```json\n{results_json}\n```\n"
            + f"\nMax trailers to present: {SEARCH_MAX_RECOMMENDATIONS}"
        )

        messages = self._messages_for_llm(state, system)
        tools = [
            FAQ_TOOL,
            LOG_INTEREST_TOOL,
            SEARCH_TRAILERS_TOOL,
            SET_TRAILER_TYPE_TOOL,
            RECORD_SLOT_ANSWER_TOOL,
        ]
        llm_with_tools = self._llm.bind_tools(tools)

        shadow: dict[str, Any] = dict(state)
        updates: dict = {"messages": []}
        all_new_messages: list[BaseMessage] = []
        current_messages = messages
        max_iterations = 4
        faq_reply_override: Optional[str] = None

        for _ in range(max_iterations):
            response = llm_with_tools.invoke(current_messages)
            all_new_messages.append(response)

            if not (hasattr(response, "tool_calls") and response.tool_calls):
                break

            tool_messages = []
            for tc in response.tool_calls:
                fn = tc["name"]
                args = tc.get("args") or {}
                if fn == "faq_tool":
                    body, extra = self._execute_faq_tool(args, shadow)
                    updates.update(extra)
                    shadow.update(extra)
                    try:
                        data = json.loads(body)
                        if data.get("ok"):
                            eff = str(data.get("effective_reply_text") or "").strip()
                            faq_reply_override = eff or str(args.get("reply_text") or "").strip()
                    except json.JSONDecodeError:
                        pass
                    tool_messages.append(ToolMessage(content=body, tool_call_id=tc["id"]))
                elif fn == "log_product_interest":
                    item_name = str(args.get("item_name", "")).strip()
                    result = self._execute_log_interest(
                        item_name, session_id=state.get("session_id")
                    )
                    updates["is_interested"] = True
                    updates["interested_item"] = item_name
                    tool_messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
                elif fn == "search_trailers":
                    faq_reply_override = None
                    result, extra = self._execute_search_tool(args, shadow, {})
                    shadow.update(extra)
                    updates.update(extra)
                    tool_messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
                elif fn == "set_trailer_type":
                    raw_type = str(args.get("trailer_type", "")).strip()
                    result, extra = self._run_set_trailer_type_tool(raw_type, shadow)
                    shadow.update(extra)
                    updates.update(extra)
                    tool_messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
                elif fn == "record_slot_answer":
                    tool_result, extra_updates = self._execute_specialist_tool(
                        fn, args, shadow
                    )
                    shadow.update(extra_updates)
                    updates.update(extra_updates)
                    tool_messages.append(
                        ToolMessage(content=tool_result, tool_call_id=tc["id"])
                    )
                else:
                    tool_messages.append(
                        ToolMessage(
                            content=json.dumps({"ok": False, "error": f"unknown:{fn}"}),
                            tool_call_id=tc["id"],
                        )
                    )
            all_new_messages.extend(tool_messages)
            current_messages = list(messages) + all_new_messages

        updates["messages"] = all_new_messages
        if faq_reply_override:
            _ensure_faq_assistant_reply(all_new_messages, faq_reply_override)
        updates["recommendation_entry_due"] = False
        return updates

    def _execute_log_interest(
        self, item_name: str, *, session_id: Optional[str] = None
    ) -> str:
        if not item_name:
            return json.dumps({"ok": False, "error": "item_name is required"})
        if self._customer is None:
            return json.dumps({"ok": False, "error": "No customer contact configured."})
        try:
            send_ticket_notification(
                full_name=self._customer.full_name,
                email=self._customer.email,
                phone=self._customer.phone,
                item_name=item_name,
            )
            sid = (session_id or "").strip() or (self._state.get("session_id") or "").strip()
            if sid:
                upsert_hard_lead_for_interest(sid, item_name)
            logger.info("INTEREST_LOGGED | item=%s", item_name)
            return json.dumps({"ok": True, "message": "interest_logged"})
        except Exception as exc:
            logger.exception("log_product_interest failed")
            return json.dumps({"ok": False, "error": str(exc)})

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _messages_for_specialist_llm(state: SessionState, system_prompt: str) -> list[dict]:
        """
        Convert state messages for ChatOpenAI; rewrite stale fetch_trailer_fields
        ToolMessage JSON when Utility lightweight so history matches required_slots.
        """
        lw = state.get("utility_lightweight_decided") is True
        out: list[dict] = [{"role": "system", "content": system_prompt}]
        sanitized_count = 0
        for msg in state.get("messages", []):
            if isinstance(msg, HumanMessage):
                out.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    out.append(msg)
                else:
                    out.append({"role": "assistant", "content": msg.content or ""})
            elif isinstance(msg, ToolMessage):
                content = str(msg.content or "")
                if lw:
                    new_json = _sanitize_fetch_tool_json_for_utility_lightweight(content)
                    if new_json is not None:
                        content = new_json
                        sanitized_count += 1
                        msg = ToolMessage(content=content, tool_call_id=msg.tool_call_id)
                out.append(msg)
            else:
                role = getattr(msg, "role", None) or str(msg.get("role", "user"))
                content = getattr(msg, "content", None) or str(msg.get("content", ""))
                out.append({"role": role, "content": content})
        if sanitized_count:
            logger.info(
                "FETCH_TOOL_HISTORY_SANITIZED | count=%s",
                sanitized_count,
            )
        return out

    @staticmethod
    def _messages_for_llm(state: SessionState, system_prompt: str) -> list[dict]:
        """Convert state messages to the format expected by ChatOpenAI."""
        out: list[dict] = [{"role": "system", "content": system_prompt}]
        for msg in state.get("messages", []):
            if isinstance(msg, HumanMessage):
                out.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                # Include tool_calls if present
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    out.append(msg)  # Pass the full AIMessage for tool call continuity
                else:
                    out.append({"role": "assistant", "content": msg.content or ""})
            elif isinstance(msg, ToolMessage):
                out.append(msg)
            else:
                role = getattr(msg, "role", None) or str(msg.get("role", "user"))
                content = getattr(msg, "content", None) or str(msg.get("content", ""))
                out.append({"role": role, "content": content})
        return out

    # ── Public interface ──────────────────────────────────────────────────────

    def chat(
        self,
        user_message: str,
        session_id: Optional[str] = None,
        *,
        client_shown_urls: Optional[list[str]] = None,
    ) -> tuple[str, list[dict]]:
        """
        Process a user message.

        Returns
        -------
        (assistant_text, api_listings)
            assistant_text – The final text reply to show the user.
            api_listings   – Listings for the HTTP/UI for this turn only (empty unless
            ``search_trailers`` ran this turn and returned matches). ``search_results``
            in graph state may still hold prior results for prompting.
        """
        self._state["session_id"] = session_id
        self._state["client_shown_urls"] = sanitize_already_shown_urls(client_shown_urls)
        self._state["messages"] = list(self._state["messages"]) + [HumanMessage(content=user_message)]
        # Fresh HTTP/UI payload each turn — do not leak prior search_results to the API.
        self._state["api_listings_this_turn"] = []
        self._state["recommendation_entry_due"] = False
        self._state["next_node"] = None

        # Run the graph; it returns the final state
        result_state = self._graph.invoke(self._state)
        self._state = result_state

        # Extract the last non-empty plain assistant text message (skip tool-call-only AIMessages)
        reply = ""
        for msg in reversed(result_state.get("messages", [])):
            if isinstance(msg, AIMessage):
                if getattr(msg, "tool_calls", None):
                    continue
                c = str(msg.content or "").strip()
                if c:
                    reply = c
                    break
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                c = str(msg.get("content", "") or "").strip()
                if c:
                    reply = c
                    break

        if not reply:
            logger.warning(
                "EMPTY_CHAT_REPLY_FALLBACK | session_id=%s",
                result_state.get("session_id"),
            )
            reply = (
                "Sorry—I didn't catch that. Tell me what trailer you're looking for, "
                "or call TrailerPlace at 979-532-1486."
            )

        return reply, list(result_state.get("api_listings_this_turn") or [])

    @property
    def trailer_type(self) -> Optional[str]:
        """Currently identified trailer type (persisted in session state)."""
        return self._state.get("trailer_type")

    @property
    def slots(self) -> dict:
        """All qualification slots collected so far."""
        return dict(self._state.get("slots_collected", {}))

    def reset(self) -> None:
        """Reset the agent to a fresh state (new session)."""
        self._state = self._initial_state()
