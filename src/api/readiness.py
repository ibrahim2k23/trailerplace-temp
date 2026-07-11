"""Process readiness for the /health probe (M8 §6).

app.py's `_wait_for_backend_ready()` polls GET /health until it sees HTTP 2xx with
`{"status": "ok"}`, retrying on anything else. So /health must report "ok" only after
the graph is compiled and the database is reachable (or persistence is off) — otherwise
the frontend proceeds and the first real turn pays the graph-build cost inside its
own timeout budget.
"""
from __future__ import annotations

_state: dict[str, object] = {"graph": False, "db": False, "error": None}


def mark_graph_ready() -> None:
    _state["graph"] = True


def mark_db_ready() -> None:
    _state["db"] = True


def mark_failed(reason: str) -> None:
    _state["error"] = reason


def reset() -> None:
    """Test hook: return to the pre-startup state."""
    _state.update({"graph": False, "db": False, "error": None})


def is_ready() -> bool:
    return bool(_state["graph"]) and bool(_state["db"]) and _state["error"] is None


def readiness_status() -> dict[str, object]:
    if _state["error"]:
        return {"status": "error", "detail": _state["error"]}
    if is_ready():
        return {"status": "ok"}
    pending = [name for name in ("graph", "db") if not _state[name]]
    return {"status": "starting", "detail": f"waiting on: {', '.join(pending)}"}
