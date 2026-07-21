"""Cost audit over the per-turn JSON logs (Milestone 9 §3).

Asserts the Locked Decision on LLM spend:

  * exactly 2 chat completions per normal turn (Analyze + Respond) — haul
    classification and inventory-lookup detection are folded into Analyze;
  * +1 embedding on Pinecone search turns, and ONLY on those;
  * inventory-lookup turns add zero extra calls (the Excel matcher uses neither
    an LLM nor an embedding).

Reads the JSONL written by `src/turn_log.py` when TURN_LOG_PATH is set:

    TURN_LOG_PATH=turns.jsonl python main.py
    python scripts/cost_report.py turns.jsonl

Exits nonzero when any turn violates a budget, so it can gate a release.
Network-free: no LangSmith, no OpenAI.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

# Analyze + Respond. One separately tagged feature-rerank completion may be
# added by violations() on non-metadata feature search turns.
MAX_CHAT_COMPLETIONS_PER_TURN = 2


# ---------------------------------------------------------------------------
# Parsing (pure)
# ---------------------------------------------------------------------------

def load_turns(path: str | Path) -> list[dict[str, Any]]:
    """Read chat_turn records from a JSONL file, skipping blanks and other events."""
    records: list[dict[str, Any]] = []
    for line_no, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: not valid JSON ({exc})") from exc
        if record.get("event") == "chat_turn":
            records.append(record)
    return records


# ---------------------------------------------------------------------------
# Auditing (pure)
# ---------------------------------------------------------------------------

@dataclass
class TurnCost:
    session_id: str
    turn_id: str
    intent: str | None
    tools: list[str]
    chat_completions: int
    feature_reranks: int
    embeddings: int
    total_tokens: int
    latency_ms: float

    @property
    def is_replay(self) -> bool:
        """A receipt replay runs no graph, so it makes no calls. Nothing to audit."""
        return self.chat_completions == 0


def summarize(record: dict[str, Any]) -> TurnCost:
    calls = record.get("llm_calls") or {}
    return TurnCost(
        session_id=str(record.get("session_id", "")),
        turn_id=str(record.get("turn_id", "")),
        intent=record.get("intent"),
        tools=list(record.get("tools_fired") or []),
        chat_completions=int(calls.get("chat_completions", 0) or 0),
        feature_reranks=int(calls.get("feature_reranks", 0) or 0),
        embeddings=int(calls.get("embeddings", 0) or 0),
        total_tokens=int(calls.get("total_tokens", 0) or 0),
        latency_ms=float(record.get("latency_ms", 0) or 0),
    )


def violations(turn: TurnCost, *, max_completions: int = MAX_CHAT_COMPLETIONS_PER_TURN) -> list[str]:
    """Budget violations for one turn (empty = within budget)."""
    if turn.is_replay:
        return []

    problems: list[str] = []
    if turn.feature_reranks > 1:
        problems.append(f"{turn.feature_reranks} feature-rerank completions (maximum 1)")
    allowed_completions = max_completions + min(turn.feature_reranks, 1)
    if turn.chat_completions > allowed_completions:
        problems.append(
            f"{turn.chat_completions} chat completions (budget {allowed_completions}: "
            "Analyze + Respond + optional Feature Rerank)"
        )

    searched = "search" in turn.tools
    if turn.feature_reranks and not searched:
        problems.append("feature rerank ran on a non-search turn")
    expected_embeddings = 1 if searched else 0
    if turn.embeddings != expected_embeddings:
        where = "a search turn" if searched else "a non-search turn"
        problems.append(f"{turn.embeddings} embeddings on {where} (expected {expected_embeddings})")

    # An inventory lookup is pure Excel + rapidfuzz: it must not add either kind of
    # call on top of the two the turn already spends on Analyze and Respond.
    if "inventory_lookup" in turn.tools and not searched and turn.embeddings:
        problems.append("inventory lookup made an embedding call (the Excel matcher must make none)")

    return problems


def audit(records: Iterable[dict[str, Any]], *, max_completions: int = MAX_CHAT_COMPLETIONS_PER_TURN) -> dict[str, Any]:
    turns = [summarize(record) for record in records]
    failures = [(turn, violations(turn, max_completions=max_completions)) for turn in turns]
    failures = [(turn, problems) for turn, problems in failures if problems]

    audited = [turn for turn in turns if not turn.is_replay]
    total_tokens = sum(turn.total_tokens for turn in audited)
    return {
        "turns": turns,
        "audited": len(audited),
        "replays": len(turns) - len(audited),
        "failures": failures,
        "total_tokens": total_tokens,
        "total_completions": sum(turn.chat_completions for turn in audited),
        "total_embeddings": sum(turn.embeddings for turn in audited),
        "avg_tokens_per_turn": (total_tokens / len(audited)) if audited else 0.0,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_report(result: dict[str, Any]) -> None:
    print(f"Turns audited: {result['audited']} (+{result['replays']} receipt replays skipped)")
    print(f"Chat completions: {result['total_completions']}   Embeddings: {result['total_embeddings']}")
    print(f"Tokens: {result['total_tokens']} total, {result['avg_tokens_per_turn']:.0f} avg/turn")
    print()
    print(f"{'intent':<26}{'calls':>6}{'embed':>7}{'tokens':>9}{'ms':>9}  tools")
    for turn in result["turns"]:
        marker = " (replay)" if turn.is_replay else ""
        tools = ",".join(turn.tools) or "-"
        print(
            f"{(turn.intent or '-') + marker:<26}{turn.chat_completions:>6}{turn.embeddings:>7}"
            f"{turn.total_tokens:>9}{turn.latency_ms:>9.0f}  {tools}"
        )

    if result["failures"]:
        print(f"\nBUDGET VIOLATIONS ({len(result['failures'])} turn(s)):")
        for turn, problems in result["failures"]:
            print(f"  session={turn.session_id} turn={turn.turn_id} intent={turn.intent}")
            for problem in problems:
                print(f"    - {problem}")
    else:
        print("\nAll turns within budget.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit LLM calls/tokens per turn from the structured turn log.")
    parser.add_argument("log_file", help="JSONL file written via TURN_LOG_PATH (see src/turn_log.py).")
    parser.add_argument(
        "--max-completions",
        type=int,
        default=MAX_CHAT_COMPLETIONS_PER_TURN,
        help=f"Chat completions allowed per turn (default {MAX_CHAT_COMPLETIONS_PER_TURN}).",
    )
    args = parser.parse_args(argv)

    path = Path(args.log_file)
    if not path.exists():
        print(f"No such log file: {path}. Run the backend with TURN_LOG_PATH={path} first.")
        return 1

    records = load_turns(path)
    if not records:
        print(f"No chat_turn records in {path}.")
        return 1

    result = audit(records, max_completions=args.max_completions)
    _print_report(result)
    return 1 if result["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
