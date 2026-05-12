from __future__ import annotations

import os
from typing import Any


def thinking_agent_enabled() -> bool:
    return (os.getenv("THINKING_AGENT_ENABLED") or "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def thinking_agent_background() -> bool:
    return (os.getenv("THINKING_AGENT_BACKGROUND") or "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def generate_thinking_flow(payload: dict[str, Any]) -> dict[str, Any]:
    _ = payload
    return {"status": "disabled", "thinking_markdown": ""}


def log_thinking_flow(session_id: str, payload: dict[str, Any], result: dict[str, Any]) -> None:
    _ = (session_id, payload, result)
