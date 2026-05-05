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
                             │ search_results available?
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
  set_trailer_type         – Master router saves the resolved category.
  fetch_trailer_fields     – Specialist fetches required/optional slots.
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

from src.email_sender import send_ticket_notification
from src.models import CustomerContact, TrailerFilter, TrailerListing
from src.normalizer import (
    normalize_category,
    normalize_color,
    normalize_hitch,
    normalize_make,
    HITCH_MAP,
)
from src.state import SessionState
from src.trailer_fields import get_trailer_fields_as_dict, list_all_categories
from src.shown_listings_store import load_shown_urls
from src.conversation_store import upsert_hard_lead_for_interest

# Re-use the search/rerank helpers from the original agent
from src.agent import (
    _build_pinecone_filter,
    _canonical_listing_key_from_match,
    _coerce_required_length_ft,
    _coerce_required_payload_lbs,
    _dedupe_matches,
    _extract_length_ft_from_text,
    _extract_weight_lbs_from_text,
    _infer_hitch_type_from_text,
    _is_clear_light_cargo_text,
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
    return _is_clear_light_cargo_text(t)


def _strip_weight_slots(required: list[str], haul_text: str) -> tuple[list[str], bool]:
    """
    If *haul_text* describes a lightweight item, remove weight-related slot names
    from *required* and return (filtered_list, lightweight_detected).
    The original list is not mutated.
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
                        "Fiber, Race Trailer, Welding, Aluminum."
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

MASTER_ROUTER_PROMPT = f"""You are a friendly trailer sales assistant for TrailerPlace, Wharton TX (979-532-1486).

## YOUR ROLE
You are the entry point of this conversation. Your sole job is to:
1. Greet the customer warmly (use their first name if known).
2. Handle non-sales intents (financing, trade-in, service, store info) by directing them to 979-532-1486.
3. Identify the trailer **category** the customer needs, then call `set_trailer_type` immediately.

## AVAILABLE TOOLS
- **set_trailer_type**: Call this as soon as you know the trailer category. This hands the customer off to the category specialist. Do NOT ask qualification questions yourself — the specialist handles that.

## INTENT ROUTING (non-sales)
- financing    → "Let me connect you with our finance team — call 979-532-1486. I can also help narrow down the trailer type first if that's helpful."
- trade_in     → "Our sales team handles trade-in appraisals — give us a call at 979-532-1486."
- service/parts → "Our service and parts team can help — reach them at 979-532-1486."
- store_info   → "We're in Wharton, TX. Call us at 979-532-1486. We offer financing and delivery."
- human_handoff → "You can reach our team at 979-532-1486. Happy to keep helping here too."

## CATEGORY IDENTIFICATION RULES
Map what the customer says to one of these categories:
Equipment, Car Hauler, Utility, Dump, Tilt, Enclosed, Livestock, Flatbed, Roll Off, Diesel Tank, Fiber, Race Trailer, Welding, Aluminum

**Hitch vs category (critical):** Words like **gooseneck**, **bumper pull**, **tag-along** describe a **hitch type**, not a trailer category. Never call `set_trailer_type` with those values. If the customer only mentions hitch style, keep the category you already identified (or ask which trailer category they need) — the specialist will set `hitch_type` on `search_trailers`.

Key disambiguation:
- "aluminum" / "lightweight" / "won't rust" → MODIFIER, not a category. Ask: "What type of trailer are you wanting in aluminum — utility, equipment, enclosed, or something else?" Then call set_trailer_type with "Aluminum".
- "toy hauler" → Ask: "Will you be hauling a vehicle on an open deck, or looking for a camper-style toy hauler?"
- "office trailer" / "cooldown trailer" → Ask: "Will this be for fiber/telecom work specifically, or a more general office trailer?" → then Fiber or Enclosed
- "lowboy" / "low profile" → Equipment
- "box trailer" / "V-nose" → Enclosed
- "skid steer" / "mini ex" / "tractor" → Equipment
- "landscape" / "lawnmower" / "ATV trailer" → Utility
- "splicing trailer" / "fiber optic trailer" → Fiber
- "enclosed car hauler" → Race Trailer
- "dumpster" / "roll-off" → Roll Off
- "fuel tank" / "tank trailer" → Diesel Tank
- "hotshot" / "step deck" / "platform trailer" → Flatbed
- "Galyean" / "Star trailer" / cattle → Livestock
- "Calico trailer" / goats / hogs → Livestock

If the customer hasn't said enough to determine a category, ask ONE question: "What will you be using the trailer for?" or "What are you looking to haul?"

## IMPORTANT
- Do NOT ask weight, size, hitch, or any other qualification questions here. Just identify intent and category.
- Once you know the category, call `set_trailer_type` — the specialist takes over from there.
- **Hitch vs category change:** Never call `set_trailer_type` with hitch-only words (gooseneck, bumper pull, tag-along). **Do** call `set_trailer_type` when the customer names a **different trailer category** than the session (e.g. switching from Dump to Utility) — see CURRENT SESSION when shown.
- If the customer already has an active category and their new message only refines **hitch**, length, budget, or the same category need, **do not** change `trailer_type`.
"""

_SPECIALIST_PROMPT_TEMPLATE = """You are the {trailer_type} Trailer Specialist for TrailerPlace (Wharton TX, 979-532-1486).

## YOUR ROLE
The customer is looking for a **{trailer_type}** trailer. Your job is to:
1. First, call `fetch_trailer_fields` with trailer_type="{trailer_type}" to get the required and optional slots.
2. Ask ONE question at a time to fill the required slots.
3. Evaluate whether to ask optional questions (ask only if relevant to what the customer has said).
4. Once required slots are filled, call `search_trailers` immediately.

## AVAILABLE TOOLS
- **fetch_trailer_fields**: Call this first to learn what questions to ask for this trailer type. Returns required_slots, optional_slots, and the exact question to ask for each slot.
- **search_trailers**: Call this once all required slots are collected. Provide a rich query combining all slot values. Use filter fields (hitch_type, required_payload_lbs, etc.) when the customer has clearly specified them.

## SLOT COLLECTION RULES
- Ask **ONE question at a time** in the order returned by fetch_trailer_fields.
- Before asking a question, scan the conversation — if the customer already answered it, skip it silently.
- **Lightweight weight exception (CRITICAL)**: If the haul item is any of the following, you must NEVER ask about haul weight — not even once. Silently set payload=1000 lbs when calling search_trailers:
  golf cart/carts, golf equipment, ATV/ATVs, UTV/UTVs, side-by-side,
  dirt bike/bikes, motorcycle/motorcycles, lawn mower/mowers, zero-turn,
  riding mower, push mower, gardening tools, landscaping tools,
  small generator, canoe/canoes, kayak/kayaks, bicycle/bicycles, e-bike/e-bikes,
  small furniture, camping gear, small equipment, light cargo, hobby equipment.
  This rule is silent — never acknowledge to the customer that you skipped it.
- For optional slots: only ask if (a) the conversation suggests it matters to the customer, OR (b) you've asked all required slots and still have room for one more.

## CATEGORY PIVOT
If the customer's **latest message** indicates a different **trailer category** than **{trailer_type}** (e.g. they were looking at Dump results but now want Utility), call `set_trailer_type` with the new category **first**, then `fetch_trailer_fields` for that type. Do **not** call `search_trailers` for the new need until required slots for the **new** category are collected.
- **Never** call `set_trailer_type` for hitch-only wording (gooseneck, bumper pull, tag-along). Those belong in `search_trailers` as `hitch_type` ("Gooseneck" or "Bumper Pull"), not as `trailer_type`.

## SHOW MORE (same category / same need)
If the customer wants **more listings** for the same search (e.g. "show me more", "any others?", "what else do you have?"), call `search_trailers` immediately with **more_results=true** and the same filters/query intent as before. **Do not** only promise to search in plain text — you must invoke the tool so new inventory is fetched.

## SEARCH CALL RULES
- Normalize units before calling: convert tons/kg → lbs, convert m/cm/in → ft.
- Set `hitch_type` filter whenever the customer clearly prefers Bumper Pull or Gooseneck.
- Set `required_payload_lbs`, `required_length_ft`, `required_gvwr_lbs` from collected slots.
- Build a rich `query` string that captures: haul item, weight, use case, trailer type, and any features mentioned.

## SLOTS COLLECTED SO FAR
{slots_summary}
"""

RECOMMENDATION_PROMPT = f"""You are the Recommendation Specialist for TrailerPlace (Wharton TX, 979-532-1486).
Search results have been retrieved. Your job is to present them clearly and move toward a sale.

## AVAILABLE TOOLS
- **log_product_interest**: Call this when the customer clearly expresses interest in a specific unit (says a stock number, "the first one", "the red one", a partial title, etc.). Use the exact full listing title from the results. After the tool succeeds, tell them their query has been logged and the team will follow up soon.
- **search_trailers**: Call this if the customer wants to see more options ("show me more", "any others?") with more_results=true. Keep the same filter fields and intent as the last search; only set more_results when continuing the same search.

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
        self._state: SessionState = self._initial_state()
        self._graph = self._build_graph()

    # ── State initialisation ─────────────────────────────────────────────────

    def _initial_state(self) -> SessionState:
        return SessionState(
            messages=[],
            intent=None,
            trailer_type=None,
            slots_collected={},
            required_slots=[],
            optional_slots=[],
            search_results=[],
            search_results_for_category=None,
            api_listings_this_turn=[],
            recommendation_entry_due=False,
            is_interested=False,
            interested_item=None,
            customer_full_name=self._customer.full_name if self._customer else None,
            customer_email=self._customer.email if self._customer else None,
            customer_phone=self._customer.phone if self._customer else None,
            session_id=None,
            next_node=None,
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
        if state.get("trailer_type"):
            return "specialist_node"
        return END

    @staticmethod
    def _route_after_specialist(state: SessionState) -> str:
        if not state.get("trailer_type"):
            return "master_router_node"
        sr = state.get("search_results") or []
        sfc = state.get("search_results_for_category")
        tt = state.get("trailer_type")
        if sr and sfc == tt and state.get("recommendation_entry_due"):
            return "recommendation_node"
        return END

    # ── Node: master_router_node ──────────────────────────────────────────────

    def _master_router_node(self, state: SessionState) -> dict:
        """Greet, route intent, identify trailer category."""
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

        messages = self._messages_for_llm(state, system)
        tools = [SET_TRAILER_TYPE_TOOL]

        llm_with_tools = self._llm.bind_tools(tools)
        response = llm_with_tools.invoke(messages)

        updates: dict = {"messages": [response]}

        prev_tt = state.get("trailer_type")

        # Handle tool calls
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_messages = []
            for tc in response.tool_calls:
                fn = tc["name"]
                args = tc["args"]
                if fn == "set_trailer_type":
                    raw_type = str(args.get("trailer_type", "")).strip()
                    canonical, err = _validate_set_trailer_type_arg(raw_type)
                    if err:
                        logger.info(
                            "TRAILER_TYPE_REJECTED | master | raw=%s | reason=%s",
                            raw_type,
                            err.get("error"),
                        )
                        tool_messages.append(
                            ToolMessage(
                                content=json.dumps(err, ensure_ascii=True),
                                tool_call_id=tc["id"],
                            )
                        )
                        continue
                    if canonical is None:
                        continue
                    updates["trailer_type"] = canonical
                    # Pre-populate slot lists from trailer_fields
                    spec = get_trailer_fields_as_dict(canonical)
                    updates["required_slots"] = spec["required_slots"]
                    updates["optional_slots"] = spec["optional_slots"]
                    # Category pivot: drop stale search results from a previous category
                    if prev_tt and prev_tt != canonical:
                        updates["search_results"] = []
                        updates["search_results_for_category"] = None
                        updates["slots_collected"] = {}
                        updates["is_interested"] = False
                        updates["interested_item"] = None
                        updates["recommendation_entry_due"] = False
                        logger.info(
                            "TRAILER_TYPE_PIVOT_MASTER | from=%s to=%s | cleared_search",
                            prev_tt,
                            canonical,
                        )
                    logger.info("TRAILER_TYPE_SET | trailer_type=%s", canonical)
                    tool_messages.append(
                        ToolMessage(
                            content=json.dumps({"ok": True, "trailer_type": canonical}),
                            tool_call_id=tc["id"],
                        )
                    )
                else:
                    tool_messages.append(
                        ToolMessage(
                            content=json.dumps({"ok": False, "error": f"unknown_tool:{fn}"}),
                            tool_call_id=tc["id"],
                        )
                    )
            updates["messages"] = updates["messages"] + tool_messages

        return updates

    # ── Node: specialist_node ─────────────────────────────────────────────────

    def _specialist_node(self, state: SessionState) -> dict:
        """Collect qualification slots then trigger search."""
        trailer_type = state.get("trailer_type", "Unknown")
        slots = state.get("slots_collected", {})
        required = state.get("required_slots", [])
        optional = state.get("optional_slots", [])

        # ── Lightweight haul detection ──────────────────────────────────────────
        # If the haul item has already been collected and is lightweight, strip
        # weight-related slots from required so the LLM never asks for them.
        haul_item_text = (
            str(slots.get("haul_item") or "")
            + " " + str(slots.get("vehicle_type") or "")
            + " " + str(slots.get("haul_material") or "")
        )
        # Only the latest user message (current turn) — avoids livestock / listing text
        # in assistant history falsely triggering lightweight detection.
        for _msg in reversed(state.get("messages", [])):
            if isinstance(_msg, HumanMessage):
                haul_item_text += " " + str(_msg.content or "")
                break
            if isinstance(_msg, dict) and _msg.get("role") == "user":
                haul_item_text += " " + str(_msg.get("content") or "")
                break
        required, _is_light = _strip_weight_slots(required, haul_item_text)
        if _is_light:
            logger.info("LIGHTWEIGHT_DETECTED | weight slots stripped from required for trailer_type=%s", trailer_type)

        slots_summary_lines = []
        for slot in required + optional:
            val = slots.get(slot)
            if val is not None:
                slots_summary_lines.append(f"  {slot}: {val} ✓")
            else:
                slots_summary_lines.append(f"  {slot}: (not yet collected)")
        slots_summary = "\n".join(slots_summary_lines) if slots_summary_lines else "  (none yet)"

        system = _SPECIALIST_PROMPT_TEMPLATE.format(
            trailer_type=trailer_type,
            slots_summary=slots_summary,
        )

        messages = self._messages_for_llm(state, system)
        tools = [FETCH_TRAILER_FIELDS_TOOL, SEARCH_TRAILERS_TOOL, SET_TRAILER_TYPE_TOOL]
        llm_with_tools = self._llm.bind_tools(tools)

        # Agentic loop within the node: keep calling until no more tool calls
        updates: dict = {"messages": []}
        all_new_messages: list[BaseMessage] = []

        current_messages = messages
        max_iterations = 6
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            response = llm_with_tools.invoke(current_messages)
            all_new_messages.append(response)

            if not (hasattr(response, "tool_calls") and response.tool_calls):
                # No tool calls — final text response
                break

            tool_messages = []
            for tc in response.tool_calls:
                fn = tc["name"]
                args = tc["args"]
                tool_result, extra_updates = self._execute_specialist_tool(fn, args, state)
                updates.update(extra_updates)
                tool_messages.append(
                    ToolMessage(content=tool_result, tool_call_id=tc["id"])
                )
                # If search results were populated, exit the loop after tool response
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

        if fn == "fetch_trailer_fields":
            trailer_type = str(args.get("trailer_type", state.get("trailer_type", ""))).strip()
            spec = get_trailer_fields_as_dict(trailer_type)
            extra["required_slots"] = spec["required_slots"]
            extra["optional_slots"] = spec["optional_slots"]
            logger.info("FETCH_FIELDS | trailer_type=%s | required=%s", trailer_type, spec["required_slots"])
            return json.dumps(spec), extra

        if fn == "set_trailer_type":
            # Customer changed their mind
            raw_type = str(args.get("trailer_type", "")).strip()
            canonical, err = _validate_set_trailer_type_arg(raw_type)
            if err:
                logger.info(
                    "TRAILER_TYPE_REJECTED | specialist | raw=%s | reason=%s",
                    raw_type,
                    err.get("error"),
                )
                return json.dumps(err, ensure_ascii=True), {}
            prev_tt = state.get("trailer_type")
            extra["trailer_type"] = canonical
            extra["slots_collected"] = {}
            spec = get_trailer_fields_as_dict(canonical)
            extra["required_slots"] = spec["required_slots"]
            extra["optional_slots"] = spec["optional_slots"]
            if prev_tt and prev_tt != canonical:
                extra["search_results"] = []
                extra["search_results_for_category"] = None
                extra["recommendation_entry_due"] = False
                extra["is_interested"] = False
                extra["interested_item"] = None
                logger.info(
                    "TRAILER_TYPE_PIVOT_SPECIALIST | from=%s to=%s | cleared_search",
                    prev_tt,
                    canonical,
                )
            else:
                logger.info("TRAILER_TYPE_SPECIALIST | trailer_type=%s", canonical)
            return json.dumps({"ok": True, "trailer_type": canonical, "note": "category updated"}), extra

        if fn == "search_trailers":
            return self._execute_search_tool(args, state, extra)

        return json.dumps({"ok": False, "error": f"unknown_tool:{fn}"}), extra

    def _execute_search_tool(
        self, args: dict, state: SessionState, extra: dict
    ) -> tuple[str, dict]:
        """Run a Pinecone search, update state with results."""
        from src.agent import TrailerAgent  # reuse rerank logic

        query = str(args.get("query", "")).strip()
        more_results = bool(args.get("more_results", False))
        required_payload_lbs = _coerce_required_payload_lbs(args.get("required_payload_lbs"))
        required_length_ft = _coerce_required_length_ft(args.get("required_length_ft"))
        required_gvwr_lbs = _coerce_required_payload_lbs(args.get("required_gvwr_lbs"))

        exclude_urls: Optional[set[str]] = None
        if more_results:
            sid = (state.get("session_id") or "").strip()
            if sid:
                try:
                    exclude_urls = load_shown_urls(sid)
                except ValueError:
                    logger.warning("Invalid session_id for shown listings: %r", sid)
                    exclude_urls = set()
                logger.info(
                    "LG_SHOW_MORE | session_id=%s exclude_urls=%s",
                    sid,
                    len(exclude_urls),
                )
            else:
                logger.warning(
                    "search_trailers more_results=True but no session_id; cannot exclude shown listings"
                )

        # Infer from query/history if not explicitly provided
        if required_payload_lbs is None:
            required_payload_lbs = _extract_weight_lbs_from_text(query)
        if required_length_ft is None:
            required_length_ft = _extract_length_ft_from_text(query)

        # Lightweight cargo detection (query + recent user messages only)
        light = _is_clear_light_cargo_text(query)
        if not light:
            for msg in reversed(state.get("messages", [])):
                if isinstance(msg, HumanMessage):
                    if _is_clear_light_cargo_text(str(msg.content or "")):
                        light = True
                    break
                if isinstance(msg, dict) and msg.get("role") == "user":
                    if _is_clear_light_cargo_text(str(msg.get("content") or "")):
                        light = True
                    break
        if required_payload_lbs is None and light:
            required_payload_lbs = 1000.0

        hitch_type = args.get("hitch_type")
        if not hitch_type:
            hitch_type = _infer_hitch_type_from_text(query)
        if not hitch_type:
            hitch_type = _infer_hitch_from_recent_user_messages(state)
        if hitch_type:
            hitch_type = normalize_hitch(hitch_type) or hitch_type
        logger.info("LG_HITCH | hitch_type=%s", hitch_type)

        trailer_filter = TrailerFilter(
            condition=args.get("condition"),
            price_min=args.get("price_min"),
            price_max=args.get("price_max"),
            category_subcategory=args.get("category_subcategory") or state.get("trailer_type"),
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
        results_json = json.dumps(search_results, indent=2) if search_results else "[]"

        system = (
            RECOMMENDATION_PROMPT
            + f"\n\n## CURRENT SEARCH RESULTS\n```json\n{results_json}\n```\n"
            + f"\nMax trailers to present: {SEARCH_MAX_RECOMMENDATIONS}"
        )

        messages = self._messages_for_llm(state, system)
        tools = [LOG_INTEREST_TOOL, SEARCH_TRAILERS_TOOL]
        llm_with_tools = self._llm.bind_tools(tools)

        updates: dict = {"messages": []}
        all_new_messages: list[BaseMessage] = []
        current_messages = messages
        max_iterations = 4

        for _ in range(max_iterations):
            response = llm_with_tools.invoke(current_messages)
            all_new_messages.append(response)

            if not (hasattr(response, "tool_calls") and response.tool_calls):
                break

            tool_messages = []
            for tc in response.tool_calls:
                fn = tc["name"]
                args = tc["args"]
                if fn == "log_product_interest":
                    item_name = str(args.get("item_name", "")).strip()
                    result = self._execute_log_interest(item_name)
                    updates["is_interested"] = True
                    updates["interested_item"] = item_name
                    tool_messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
                elif fn == "search_trailers":
                    result, extra = self._execute_search_tool(args, state, {})
                    updates.update(extra)
                    tool_messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
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
        updates["recommendation_entry_due"] = False
        return updates

    def _execute_log_interest(self, item_name: str) -> str:
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
            sid = (self._state.get("session_id") or "").strip()
            if sid:
                upsert_hard_lead_for_interest(sid, item_name)
            logger.info("INTEREST_LOGGED | item=%s", item_name)
            return json.dumps({"ok": True, "message": "interest_logged"})
        except Exception as exc:
            logger.exception("log_product_interest failed")
            return json.dumps({"ok": False, "error": str(exc)})

    # ── Helpers ───────────────────────────────────────────────────────────────

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
        self._state["messages"] = list(self._state["messages"]) + [HumanMessage(content=user_message)]
        # Fresh HTTP/UI payload each turn — do not leak prior search_results to the API.
        self._state["api_listings_this_turn"] = []
        self._state["recommendation_entry_due"] = False

        # Run the graph; it returns the final state
        result_state = self._graph.invoke(self._state)
        self._state = result_state

        # Extract the last assistant text message
        reply = ""
        for msg in reversed(result_state.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.content:
                reply = str(msg.content)
                break
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                reply = str(msg.get("content", ""))
                break

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
