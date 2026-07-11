from __future__ import annotations

from src.graph.apply_analysis import apply_analysis_to_state


def apply_analysis_node(state: dict) -> dict:
    return apply_analysis_to_state(state)
