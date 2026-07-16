from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.config import settings
from src.graph.apply_analysis import lookup_requested
from src.graph.contact_gate import contact_gate_pending
from src.graph.nodes.analyze import make_analyze_node
from src.graph.nodes.apply_analysis import apply_analysis_node
from src.graph.nodes.email_actions import email_actions_node
from src.graph.nodes.inventory_lookup import inventory_lookup_node
from src.graph.nodes.qualification import qualification_node
from src.graph.nodes.respond import make_respond_node
from src.graph.nodes.search import search_node
from src.graph.state import SessionState
from src.llm.client import LLMClient, OpenAILLMClient


# The customer asking, in so many words, to see (more) inventory. Always worth a search,
# even when nothing about their requirements changed.
_SHOW_RESULTS_INTENTS = {"skip_all_show_results", "show_more_results", "recommendation_request"}

# Turns that are conversation, not shopping. These never search, however the extractor
# happened to fill the slots — a question about a listing, or "which trailer suits a
# tractor?", is not a request for a fresh set of results. Anything such a message did change
# stays flagged, and lands on the next turn that IS a search.
#
# Handing over contact details is deliberately NOT on this list. The opening gate defers the
# customer's actual request by a turn, so the moment they clear the gate is the moment we owe
# them the results they asked for. It cannot cause a spurious search either: a contact reply
# on its own changes nothing a search is built from, so search_pending stays false.
_NON_SEARCH_INTENTS = {
    "general_question",
    "category_exploration",
    "faq",
    "team_request_escalation",
    "listing_interest",
    "inventory_lookup",
    "smalltalk_other",
}

# Kept off the blocklist above so the opening gate can deliver the request it deferred — but
# only then. See should_search.
_CONTACT_INTENTS = {"contact_info_provided", "contact_declined"}


def should_search(state: dict) -> bool:
    """Whether this turn earns a Pinecone query.

    Search is a tool call, not a turn type: it runs when every question has been asked
    (answered, skipped or waved off) AND there is actually something new to look up —
    either the customer just changed what they're after, or they asked to see more. It
    must NOT run again simply because qualification finished several turns ago: "I like
    the second one" and "here's my email" are conversation, not a new query.
    """
    if not state.get("qualification_complete"):
        return False
    if state.get("pending_category_change") or state.get("pending_category_suggestion") or state.get("pending_brand_categories"):
        return False
    turn = state.get("turn")
    if turn:
        if turn.intent in _SHOW_RESULTS_INTENTS:
            return True
        if turn.intent in _NON_SEARCH_INTENTS or turn.is_category_info_only:
            return False
        if turn.intent in _CONTACT_INTENTS and state.get("shown_urls"):
            # Handing over contact details searches ONLY when it clears the opening gate and
            # the results they asked for are still owed. Once results are on screen, an email
            # address is just an email address — it is not a new query.
            return False
    return bool(state.get("search_pending"))


def _route(state: dict) -> str:
    turn = state.get("turn")
    if lookup_requested(turn):
        return "inventory_lookup"
    # The opening contact ask comes before anything else. What they asked for is already
    # recorded in state — it just waits a turn while we ask who we're talking to.
    if contact_gate_pending(state):
        return "respond"
    if state.get("clarification_key"):
        return "respond"
    # A category move (or a suggested one) always pauses to ask before anything else:
    # the new category's questions are unanswered, so there is nothing to search on yet.
    if state.get("pending_category_change") or state.get("pending_category_suggestion"):
        return "respond"
    # A brand named before any category: ask which of that make's categories they want
    # before qualification would ask its generic "what type of trailer?".
    if state.get("pending_brand_categories"):
        return "respond"
    # Qualification is the gate for everything else. It re-checks the current category's
    # required slots every turn, so a category change re-opens the questions instead of
    # falling through to a stale qualification_complete=True.
    return "qualification"


def _after_qualification(state: dict) -> str:
    return "search" if should_search(state) else "respond"


def build_graph(client: LLMClient | None = None):
    # Tests and callers can still inject one fake/client for the entire graph.
    # Production uses a dedicated model for analysis and keeps the customer-
    # facing response model on OPENAI_MODEL.
    if client is None:
        analyze_client = OpenAILLMClient(
            model=settings.analyze_model,
            reasoning_effort=settings.analyze_reasoning_effort,
        )
        respond_client = OpenAILLMClient(model=settings.openai_model)
    else:
        analyze_client = respond_client = client
    graph = StateGraph(SessionState)
    graph.add_node("analyze", make_analyze_node(analyze_client))
    graph.add_node("apply_analysis", apply_analysis_node)
    graph.add_node("email_actions", email_actions_node)
    graph.add_node("qualification", qualification_node)
    graph.add_node("search", search_node)
    graph.add_node("inventory_lookup", inventory_lookup_node)
    graph.add_node("respond", make_respond_node(respond_client))
    graph.set_entry_point("analyze")
    graph.add_edge("analyze", "apply_analysis")
    graph.add_edge("apply_analysis", "email_actions")
    graph.add_conditional_edges(
        "email_actions",
        _route,
        {
            "inventory_lookup": "inventory_lookup",
            "respond": "respond",
            "search": "search",
            "qualification": "qualification",
        },
    )
    graph.add_node("email_actions_after_results", email_actions_node)
    graph.add_edge("inventory_lookup", "email_actions_after_results")
    graph.add_edge("search", "email_actions_after_results")
    graph.add_edge("email_actions_after_results", "respond")
    graph.add_conditional_edges("qualification", _after_qualification, {"search": "search", "respond": "respond"})
    graph.add_edge("respond", END)
    return graph.compile()
