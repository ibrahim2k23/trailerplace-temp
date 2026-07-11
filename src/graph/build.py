from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.config import settings
from src.graph.nodes.analyze import make_analyze_node
from src.graph.nodes.apply_analysis import apply_analysis_node
from src.graph.nodes.email_actions import email_actions_node
from src.graph.nodes.inventory_lookup import inventory_lookup_node
from src.graph.nodes.qualification import qualification_node
from src.graph.nodes.respond import make_respond_node
from src.graph.nodes.search import search_node
from src.graph.state import SessionState
from src.llm.client import LLMClient, OpenAILLMClient


def _route(state: dict) -> str:
    turn = state.get("turn")
    if turn and turn.intent == "inventory_lookup" and turn.inventory_lookup.is_lookup and turn.inventory_lookup.confidence != "low":
        return "inventory_lookup"
    if state.get("clarification_key"):
        return "respond"
    if state.get("pending_category_change") or state.get("pending_category_suggestion"):
        return "respond"
    if state.get("qualification_complete") or (turn and turn.intent == "skip_all_show_results"):
        return "search"
    return "qualification"


def _after_qualification(state: dict) -> str:
    return "search" if state.get("qualification_complete") else "respond"


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
