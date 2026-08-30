"""Resilient three-stage live conversation test for TrailerPlace.

This is intentionally not part of pytest: it calls the running backend, which in
turn calls the real LLM and Pinecone services and may emit the configured test
emails.  A failed turn is recorded and the suite continues.

Examples (PowerShell)::

    uv run python scripts/live_llm_conversation_test.py
    uv run python scripts/live_llm_conversation_test.py --stage 1 --category Dump
    uv run python scripts/live_llm_conversation_test.py --list

The backend must already be running (``python main.py``).  Every flow gets a new
UUID session, and the name/email are included in both the first message and the
first request's contact fields.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.domain.categories import CANONICAL_CATEGORIES  # noqa: E402
from src.domain.trailer_fields import get_trailer_fields  # noqa: E402


DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_NAME = "Ibrahim"
DEFAULT_EMAIL = "ibrahim@esided.ai"
ERROR_ASSISTANT_FRAGMENT = "something went wrong on our end"


SLOT_ANSWERS: dict[str, str] = {
    "base_category": "an equipment trailer",
    "payload_need": "roughly 7,000 to 9,000 pounds",
    "haul_item": "a skid steer, toolboxes, and assorted farm equipment",
    "haul_material": "gravel, broken concrete, and yard debris",
    "haul_weight_lbs": "probably 7,000 to 9,000 pounds, roughly",
    "haul_length_ft": "somewhere from 18 to 22 feet",
    "hitch_type": "bumper pull or gooseneck; either is acceptable",
    "vehicle_type": "a midsize SUV or a full-size pickup",
    "vehicle_length_ft": "approximately 16 to 19 feet long",
    "use_case": "general cargo deliveries and an occasional mobile workspace",
    "cargo_size": "roughly 16 by 7 by 7 feet, but I can be flexible",
    "trailer_length_ft": "about 18 to 22 feet",
    "package_scope": "the trailer and a few bins together",
    "bin_size": "medium bins, perhaps 15 to 20 yards",
    "fuel_type": "mainly diesel, possibly gasoline occasionally",
    "tank_capacity": "approximately 450 to 650 gallons",
    "fiber_use_case": "mostly field splicing, with some office work",
    # Volunteer-only slots: no category asks these, so they never appear as a generated
    # question. They are here so a scenario that names one gets sensible wording.
    "axle_capacity_lbs": "7,000 lb axles would suit me",
    "total_axle_capacity_lbs": "about 14,000 pounds across the axles in total",
    "axle_count": "two axles",
}

# Volunteered in EVERY category's stage-1 and stage-2 flow, right before results are asked
# for. Axle capacity is never a scripted question, so without this the axle path would be
# exercised in the one hand-written stage-3 conversation and nowhere else.
AXLE_VOLUNTEER_MESSAGE = (
    "One more thing - I want 7,000 lb axles, two of them."
)

VAGUE_FOLLOWUPS = (
    "I am flexible on the exact details; something fairly standard should work.",
    "I do not know the exact measurement. Use the closest practical option.",
)

POST_RESULTS_MESSAGES = (
    "I am interested in the first trailer. Please log my interest for the sales team.",
    "Do you offer financing, and can you email the team about my question?",
    "What other trailer categories do you carry, and what are they generally used for?",
)


@dataclass
class TurnResult:
    number: int
    user: str
    assistant: str = ""
    listings: int = 0
    returned_name: str | None = None
    returned_email: str | None = None
    elapsed_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


@dataclass
class FlowResult:
    stage: int
    name: str
    session_id: str
    turns: list[TurnResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.errors and all(not turn.errors for turn in self.turns)


def _contact_intro(name: str, email: str) -> str:
    return f"My name is {name} and my email is {email}."


def _slot_sentence(slot: str) -> str:
    answer = SLOT_ANSWERS.get(slot, "no strict preference; a standard option is fine")
    return f"For {slot.replace('_', ' ')}, {answer}."


def _required_answers(category: str) -> list[str]:
    return [_slot_sentence(slot) for slot in get_trailer_fields(category).required]


def stage_one_messages(category: str, name: str, email: str) -> list[str]:
    """Natural one-answer-per-turn qualification for one category."""
    return [
        f"{_contact_intro(name, email)} I am looking for a {category} trailer.",
        # Second, not last: stage 1 stops at the first results, and several categories
        # qualify in three turns - an axle turn at the end simply never ran.
        AXLE_VOLUNTEER_MESSAGE,
        *_required_answers(category),
        "I have no other strict preferences. Please show me the best matching results.",
    ]


def stage_two_messages(category: str, name: str, email: str) -> list[str]:
    """Dense first turn, vague follow-ups, results, interest, FAQ, exploration."""
    all_info = " ".join(_required_answers(category))
    return [
        # Stage 2 puts everything in the first turn, axles included: the count and the rating
        # arrive in one phrase, which must fill both facts and ask nothing back.
        f"{_contact_intro(name, email)} I need a {category} trailer. {all_info} "
        "It should have 2-7,000# axles.",
        *VAGUE_FOLLOWUPS,
        "Please use what I have given you and show me the matching results now.",
        *POST_RESULTS_MESSAGES,
    ]


def stage_three_messages(name: str, email: str) -> list[str]:
    """One deliberately messy, long conversation spanning the main behaviors."""
    return [
        (
            f"{_contact_intro(name, email)} Before shopping, what is the difference between "
            "utility, equipment, dump, and enclosed trailers?"
        ),
        "Do you offer financing and trade-ins? Please email the team about these questions too.",
        (
            "I think I need a Dump trailer for gravel and broken concrete, around 6 to 8 tons "
            "per load; I am flexible on the dump mechanism."
        ),
        "That is approximate. Pick a practical standard option and show me results.",
        "I am interested in the second trailer; please log that interest.",
        "Can you tell me what dump trailers are generally best used for?",
        "Actually, change the category to Enclosed, but keep any details that still make sense.",
        "Keep only the general use information; drop the dump-specific requirements.",
        (
            "It is for cargo and sometimes a mobile workshop, about 16 by 7 by maybe 7 feet, "
            "and AC or cabinets would be nice but are not mandatory."
        ),
        "I am not sure about the interior finish. Either is fine.",
        "Show the closest enclosed options even if my answers are vague.",
        "Show me more options, and explain the main benefit of an enclosed trailer.",
        "I like the first one. Please have someone contact me with a formal quote.",
        "What other categories work for hauling a skid steer?",
        "Switch me to Equipment: skid steer, about 8,000 pounds, about 18 feet, either hitch.",
        "I do not know the width; use a normal suitable width and show results.",
        "Is the first result available, and can you log my interest in it too?",
        # Ambiguous on purpose: this must be QUESTIONED (per axle or total?), not guessed.
        "Also it needs 14,000 lbs of axle capacity.",
        "Per axle, please.",
        # Outside the 1-4 we stock: must be refused, with a way out offered.
        "Make it seven axles.",
        # The recovery - and the count question must not be asked again afterwards.
        "Sorry, two axles.",
        "Finally, what are your store hours and service or parts options? Please notify the team.",
    ]


class LiveClient:
    def __init__(self, base_url: str, name: str, email: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.name = name
        self.email = email
        self.timeout = timeout
        self.http = requests.Session()

    def health_error(self) -> str | None:
        try:
            response = self.http.get(f"{self.base_url}/health", timeout=min(self.timeout, 10))
            response.raise_for_status()
            body = response.json()
            if body.get("status") != "ok":
                return f"backend is not ready: {body!r}"
        except Exception as exc:  # noqa: BLE001 - present a useful preflight error
            return f"cannot reach healthy backend at {self.base_url}: {exc}"
        return None

    def send(self, session_id: str, message: str, number: int) -> TurnResult:
        turn = TurnResult(number=number, user=message)
        payload: dict[str, Any] = {
            "session_id": session_id,
            "turn_id": str(uuid.uuid4()),
            "message": message,
        }
        if number == 1:
            payload["customer_full_name"] = self.name
            payload["customer_email"] = self.email
        started = time.perf_counter()
        try:
            response = self.http.post(f"{self.base_url}/chat", json=payload, timeout=self.timeout)
            response.raise_for_status()
            body = response.json()
            turn.assistant = str(body.get("assistant_text") or "")
            turn.listings = len(body.get("listings") or [])
            turn.returned_name = body.get("customer_full_name")
            turn.returned_email = body.get("customer_email")
            if not turn.assistant.strip():
                turn.errors.append("assistant returned an empty response")
            if ERROR_ASSISTANT_FRAGMENT in turn.assistant.lower():
                turn.errors.append("backend returned its contained graph-error apology")
            if number == 1 and turn.returned_name != self.name:
                turn.errors.append(
                    f"first response did not retain customer name (got {turn.returned_name!r})"
                )
            if number == 1 and turn.returned_email != self.email:
                turn.errors.append(
                    f"first response did not retain customer email (got {turn.returned_email!r})"
                )
        except Exception as exc:  # noqa: BLE001 - resilience is the point of this runner
            turn.errors.append(f"request failed: {type(exc).__name__}: {exc}")
        finally:
            turn.elapsed_seconds = round(time.perf_counter() - started, 3)
        return turn


def run_flow(
    client: LiveClient,
    *,
    stage: int,
    name: str,
    messages: Iterable[str],
    stop_after_first_results: bool = False,
) -> FlowResult:
    session_id = str(uuid.uuid4())
    result = FlowResult(stage=stage, name=name, session_id=session_id)
    saw_listings = False
    for number, message in enumerate(messages, 1):
        turn = client.send(session_id, message, number)
        reply_low = turn.assistant.lower()
        message_low = message.lower()
        if "interested" in message_low and not any(
            phrase in reply_low for phrase in ("interest", "logged", "shared", "passed")
        ):
            turn.errors.append("interest request was not acknowledged as logged/shared")
        if "financing" in message_low and not any(
            phrase in reply_low for phrase in ("financing", "finance team")
        ):
            turn.errors.append("financing FAQ was not answered in the response")
        if "service or parts" in message_low and not any(
            phrase in reply_low for phrase in ("service", "parts")
        ):
            turn.errors.append("service/parts FAQ was not answered in the response")
        result.turns.append(turn)
        saw_listings = saw_listings or turn.listings > 0
        marker = "ERROR" if turn.errors else "OK"
        print(
            f"  [{marker}] turn {number:02d} ({turn.elapsed_seconds:.1f}s, "
            f"listings={turn.listings})\n    USER: {message}\n    BOT:  {turn.assistant or '<no reply>'}"
        )
        for error in turn.errors:
            print(f"    ISSUE: {error}")
        if stop_after_first_results and saw_listings:
            break
    if not saw_listings:
        result.errors.append("flow never returned any listing cards")
        print("    FLOW ISSUE: flow never returned any listing cards")
    return result


def build_flows(
    stages: set[int], categories: list[str], name: str, email: str
) -> list[tuple[int, str, list[str], bool]]:
    flows: list[tuple[int, str, list[str], bool]] = []
    if 1 in stages:
        flows.extend(
            (1, f"Stage 1 / {category}", stage_one_messages(category, name, email), True)
            for category in categories
        )
    if 2 in stages:
        flows.extend(
            (2, f"Stage 2 / {category}", stage_two_messages(category, name, email), False)
            for category in categories
        )
    if 3 in stages:
        flows.append((3, "Stage 3 / mixed long conversation", stage_three_messages(name, email), False))
    return flows


def _write_report(path: Path, results: list[FlowResult], started_at: str, base_url: str) -> None:
    total_messages = sum(len(flow.turns) for flow in results)
    failed_turns = sum(bool(turn.errors) for flow in results for turn in flow.turns)
    report = {
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "summary": {
            "flows_completed": len(results),
            "flows_passed": sum(flow.passed for flow in results),
            "flows_with_issues": sum(not flow.passed for flow in results),
            "total_messages_sent": total_messages,
            "turns_with_errors": failed_turns,
        },
        "flows": [asdict(flow) | {"passed": flow.passed} for flow in results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def _parse_stages(values: list[int] | None) -> set[int]:
    stages = set(values or [1, 2, 3])
    invalid = stages - {1, 2, 3}
    if invalid:
        raise ValueError(f"invalid stage(s): {sorted(invalid)}")
    return stages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--email", default=DEFAULT_EMAIL)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--stage", type=int, action="append", help="stage to run; repeat for several")
    parser.add_argument("--category", action="append", help="category for Stages 1/2; repeat for several")
    parser.add_argument("--output", type=Path, default=ROOT / "logs" / "live_llm_conversation_report.json")
    parser.add_argument("--list", action="store_true", help="list flows/messages without making requests")
    args = parser.parse_args(argv)

    try:
        stages = _parse_stages(args.stage)
    except ValueError as exc:
        parser.error(str(exc))
    categories = args.category or list(CANONICAL_CATEGORIES)
    unknown = [category for category in categories if category not in CANONICAL_CATEGORIES]
    if unknown:
        parser.error(f"unknown category: {', '.join(unknown)}")

    flows = build_flows(stages, categories, args.name, args.email)
    if args.list:
        for stage, name, messages, _ in flows:
            print(f"Stage {stage}: {name} ({len(messages)} planned messages)")
        print(f"Planned flows: {len(flows)}")
        print(f"Maximum planned messages: {sum(len(flow[2]) for flow in flows)}")
        return 0

    client = LiveClient(args.base_url, args.name, args.email, args.timeout)
    health_error = client.health_error()
    if health_error:
        print(f"PRECHECK FAILED: {health_error}", file=sys.stderr)
        return 2

    started_at = datetime.now(timezone.utc).isoformat()
    results: list[FlowResult] = []
    for index, (stage, name, messages, stop_on_results) in enumerate(flows, 1):
        print(f"\n=== FLOW {index}/{len(flows)}: {name} ===")
        result = run_flow(
            client,
            stage=stage,
            name=name,
            messages=messages,
            stop_after_first_results=stop_on_results,
        )
        results.append(result)

    _write_report(args.output, results, started_at, args.base_url)
    total_messages = sum(len(flow.turns) for flow in results)
    failed_flows = sum(not flow.passed for flow in results)
    print("\n=== FINAL SUMMARY ===")
    print(f"Flows completed: {len(results)}")
    print(f"Total messages sent: {total_messages}")
    print(f"Flows with issues: {failed_flows}")
    print(f"Report: {args.output.resolve()}")
    return 1 if failed_flows else 0


if __name__ == "__main__":
    raise SystemExit(main())
