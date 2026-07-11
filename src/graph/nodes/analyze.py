from __future__ import annotations

from src.llm.analyze import analyze_turn
from src.llm.client import LLMClient
from src.tracing import add_turn_metadata


def make_analyze_node(client: LLMClient):
    def analyze_node(state: dict) -> dict:
        analysis = analyze_turn(client, state)
        state["turn"] = analysis
        # Intent/category only exist once Analyze has run, so the trace opened in
        # /chat is annotated here rather than at open time (M9 §2).
        add_turn_metadata(intent=analysis.intent, category=analysis.category_mentioned or state.get("category"))
        return state

    return analyze_node
