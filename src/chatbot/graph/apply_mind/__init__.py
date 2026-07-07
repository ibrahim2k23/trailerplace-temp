"""Decomposed pieces of the mind-application node (``_apply_mind_node``).

This subpackage holds logic extracted from the very large ``_apply_mind_node``
function in ``src.chatbot.graph``. The first extraction is the single
``build_state_return`` state-assembly helper; further phase decomposition builds
on top of it.
"""

from __future__ import annotations

from src.chatbot.graph.apply_mind.state_return import build_state_return

__all__ = ["build_state_return"]
