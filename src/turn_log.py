"""Structured per-turn JSON logs (M9 §4) — also the cost report's data source (M9 §3).

One line of JSON per /chat turn on the `trailerplace.turn` logger. When
TURN_LOG_PATH is set, those lines also land in a dedicated append-only JSONL file
that `scripts/cost_report.py` reads; the same records go to the normal log stream
either way, so an operator sees them without extra configuration.

Kept deliberately flat and machine-readable: the cost audit asserts on
`llm_calls`, and ops greps on `session_id`/`latency_ms`.
"""
from __future__ import annotations

import json
import logging
from typing import Any

# Read through the module so a test can swap the frozen Settings object wholesale.
from src import config

TURN_LOGGER_NAME = "trailerplace.turn"

logger = logging.getLogger(TURN_LOGGER_NAME)

_file_handler_installed = False


class _JsonLineFormatter(logging.Formatter):
    """Emits the record's `turn` payload as bare JSON — no level/timestamp prefix.

    The cost report parses this file with json.loads per line, so anything that is
    not the payload would have to be stripped back out.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = getattr(record, "turn", None)
        if payload is None:
            return record.getMessage()
        return json.dumps(payload, default=str)


def _ensure_file_handler() -> None:
    global _file_handler_installed
    if _file_handler_installed or not config.settings.turn_log_path:
        return
    handler = logging.FileHandler(config.settings.turn_log_path, encoding="utf-8")
    handler.setFormatter(_JsonLineFormatter())
    logger.addHandler(handler)
    # The dedicated file is a machine feed; the record still propagates to root
    # for humans. Setting the level here keeps it independent of root's level.
    logger.setLevel(logging.INFO)
    _file_handler_installed = True


def reset_for_tests() -> None:
    """Drop the file handler so a test can point TURN_LOG_PATH somewhere new."""
    global _file_handler_installed
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    _file_handler_installed = False


def tools_fired(turn_outcome: dict[str, Any]) -> list[str]:
    """Which tools ran this turn — drives both ops logs and the cost assertion.

    `search` implies one embedding call; `inventory_lookup` implies none (the
    Excel matcher uses neither an LLM nor an embedding — Locked Decision).
    """
    tools: list[str] = []
    if turn_outcome.get("search_ran"):
        tools.append("search")
    if turn_outcome.get("inventory_lookup_ran"):
        tools.append("inventory_lookup")
    if turn_outcome.get("emails_sent"):
        tools.append("email")
    return tools


def log_turn(
    *,
    session_id: str,
    turn_id: str,
    intent: str | None,
    category: str | None,
    latency_ms: float,
    turn_outcome: dict[str, Any],
    usage: Any = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Emit (and return) one turn record."""
    _ensure_file_handler()
    record: dict[str, Any] = {
        "event": "chat_turn",
        "session_id": session_id,
        "turn_id": turn_id,
        "intent": intent,
        "category": category,
        "latency_ms": round(float(latency_ms), 2),
        "tools_fired": tools_fired(turn_outcome or {}),
        "emails_sent": list((turn_outcome or {}).get("emails_sent") or []),
        "llm_calls": usage.as_dict() if usage is not None else None,
    }
    if error:
        record["error"] = error
    logger.info("chat_turn", extra={"turn": record})
    return record
