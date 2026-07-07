"""State-return assembly for the mind-application node.

``_apply_mind_node`` rebuilds and returns the full ``ChatbotState`` dict in a
dozen different early-return branches. Each branch historically wrote an inline
``{**state, ...}`` literal, which made it easy to drop a key when copying one
branch to another. ``build_state_return`` is the single choke point every one of
those branches now flows through: it always starts from the incoming ``state``
and layers the branch-specific overrides on top, so the merge semantics are
identical everywhere and defined in exactly one place.
"""

from __future__ import annotations

from typing import Any

from src.chatbot.state import ChatbotState


def build_state_return(state: ChatbotState, **overrides: Any) -> ChatbotState:
    """Return ``state`` with ``overrides`` applied.

    Semantically identical to ``{**state, **overrides}`` — the value is that
    every ``_apply_mind_node`` exit path is assembled the same way through one
    function, rather than via ad-hoc inline dict literals scattered across the
    node.
    """

    return {**state, **overrides}
