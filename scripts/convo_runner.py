"""Live scripted-conversation harness (Milestone 5, consolidated into a tagged
regression suite in Milestone 9).

Usage:
    python scripts/convo_runner.py --suite regression        # the M9 release gate
    python scripts/convo_runner.py --suite regression --repeat 2
    python scripts/convo_runner.py --suite adversarial --list  # no network
    python scripts/convo_runner.py scripts/scenarios/ --all
    python scripts/convo_runner.py scripts/scenarios/contact-full.yaml

Every scenario declares `tags: [...]`; `--suite <tag>` selects by tag. The M5-M7
scenarios plus the five M9 adversarial ones all carry `regression`.

Requires a running backend (`python main.py`) with a real OPENAI_API_KEY, and
DEBUG_STATE_ENDPOINT=1 set on that backend process so `expect_state` assertions
can read the raw session snapshot via GET /session/{id}/state.

Real API calls, real tokens. Not run by the pytest suite — see
tests/unit/test_convo_runner_parsing.py for the network-free parsing/assertion
tests that guard this module's logic in CI.
"""
from __future__ import annotations

import argparse
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
SCENARIOS_DIR = ROOT / "scripts" / "scenarios"

# The M9 release gate. Every M5-M7 scenario plus the five adversarial ones carry it.
REGRESSION_SUITE = "regression"


# ---------------------------------------------------------------------------
# Loading (pure, no network)
# ---------------------------------------------------------------------------

def load_scenario(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "name" not in data or "turns" not in data:
        raise ValueError(f"{path}: scenario must define 'name' and 'turns'")
    data.setdefault("_path", str(path))
    data.setdefault("tags", [])
    return data


def load_scenarios(target: str | Path) -> list[dict[str, Any]]:
    target = Path(target)
    if target.is_dir():
        paths = sorted(target.glob("*.yaml"))
    else:
        paths = [target]
    return [load_scenario(path) for path in paths]


def filter_by_tag(scenarios: list[dict[str, Any]], tag: str) -> list[dict[str, Any]]:
    """Scenarios carrying `tag`, in stable name order (M9 §1)."""
    selected = [s for s in scenarios if tag in (s.get("tags") or [])]
    return sorted(selected, key=lambda s: s["name"])


def load_suite(tag: str, scenarios_dir: str | Path = SCENARIOS_DIR) -> list[dict[str, Any]]:
    return filter_by_tag(load_scenarios(scenarios_dir), tag)


# ---------------------------------------------------------------------------
# Assertion evaluation (pure, no network)
# ---------------------------------------------------------------------------

def _check_expect_state(expected: dict[str, Any], state: dict[str, Any]) -> list[str]:
    """Evaluate one turn's expect_state block against the raw session snapshot.

    Keys are top-level SessionState fields (category, pending_question_repeats,
    contact_declined, ...) matched by equality. Two suffix conventions extend
    that for the common nested cases:
      - "<list_field>_contains": value must be a member of state[list_field]
        (e.g. skipped_slots_contains: "haul_material").
      - "slot:<name>": value must equal state["slots"][name] (individual
        qualification-slot values, e.g. "slot:trailer_length_ft": 15.0).
    """
    failures: list[str] = []
    for key, expected_value in expected.items():
        if key.startswith("slot:"):
            slot_name = key[len("slot:") :]
            actual_value = (state.get("slots") or {}).get(slot_name)
            if actual_value != expected_value:
                failures.append(f"expect_state.{key}: expected {expected_value!r}, got {actual_value!r}")
            continue
        if key.endswith("_contains"):
            list_key = key[: -len("_contains")]
            actual_list = state.get(list_key) or []
            if expected_value not in actual_list:
                failures.append(f"expect_state.{key}: {expected_value!r} not in {list_key}={actual_list!r}")
            continue
        actual_value = state.get(key)
        if actual_value != expected_value:
            failures.append(f"expect_state.{key}: expected {expected_value!r}, got {actual_value!r}")
    return failures


def evaluate_assertions(
    turn: dict[str, Any],
    *,
    reply_text: str,
    state: dict[str, Any],
    listings: list[Any],
) -> list[str]:
    """Return a list of failure messages for one turn's assertions (empty = pass)."""
    failures: list[str] = []

    expect_state = turn.get("expect_state")
    if expect_state:
        failures.extend(_check_expect_state(expect_state, state))

    contains_any = turn.get("expect_reply_contains_any")
    if contains_any:
        low = (reply_text or "").lower()
        if not any(str(term).lower() in low for term in contains_any):
            failures.append(f"expect_reply_contains_any: none of {contains_any!r} found in reply {reply_text!r}")

    not_contains = turn.get("expect_reply_not_contains")
    if not_contains:
        low = (reply_text or "").lower()
        hit = [term for term in not_contains if str(term).lower() in low]
        if hit:
            failures.append(f"expect_reply_not_contains: found forbidden {hit!r} in reply {reply_text!r}")

    if "expect_listings" in turn:
        expected_bool = bool(turn["expect_listings"])
        actual_bool = bool(listings)
        if actual_bool != expected_bool:
            failures.append(f"expect_listings: expected {expected_bool}, got {actual_bool}")

    if "expect_emails_sent" in turn:
        expected_emails = turn["expect_emails_sent"]
        actual_emails = state.get("emails_sent")
        if expected_emails is None:
            if actual_emails:
                failures.append(f"expect_emails_sent: expected none, got {actual_emails!r}")
        elif list(expected_emails) != list(actual_emails or []):
            failures.append(f"expect_emails_sent: expected {expected_emails!r}, got {actual_emails!r}")

    return failures


# ---------------------------------------------------------------------------
# Execution (network)
# ---------------------------------------------------------------------------

@dataclass
class TurnResult:
    user_message: str
    reply_text: str
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures


@dataclass
class ScenarioResult:
    name: str
    session_id: str
    turns: list[TurnResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(turn.passed for turn in self.turns)


def _fetch_state(base_url: str, session_id: str) -> dict[str, Any]:
    response = requests.get(f"{base_url}/session/{session_id}/state", timeout=30)
    response.raise_for_status()
    body = response.json()
    return body.get("state") or {}


def run_scenario(base_url: str, scenario: dict[str, Any], session_id: str | None = None) -> ScenarioResult:
    session_id = session_id or str(uuid.uuid4())
    result = ScenarioResult(name=scenario["name"], session_id=session_id)
    for turn in scenario.get("turns", []):
        user_message = turn.get("user", "")
        response = requests.post(
            f"{base_url}/chat",
            json={"session_id": session_id, "turn_id": str(uuid.uuid4()), "message": user_message},
            timeout=180,
        )
        response.raise_for_status()
        body = response.json()
        reply_text = body.get("assistant_text", "")
        listings = body.get("listings") or []
        state = _fetch_state(base_url, session_id)
        failures = evaluate_assertions(turn, reply_text=reply_text, state=state, listings=listings)
        result.turns.append(TurnResult(user_message=user_message, reply_text=reply_text, failures=failures))
    return result


def _print_result(result: ScenarioResult) -> None:
    status = "PASS" if result.passed else "FAIL"
    print(f"[{status}] {result.name} (session={result.session_id})")
    for idx, turn in enumerate(result.turns, 1):
        marker = "  ok" if turn.passed else "  XX"
        print(f"{marker} turn {idx}: user={turn.user_message!r}")
        print(f"       reply: {turn.reply_text!r}")
        for failure in turn.failures:
            print(f"       FAILURE: {failure}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run live scripted-conversation scenarios against a running backend.")
    parser.add_argument("target", nargs="?", help="Scenario YAML file or a directory of them. Omit when using --suite.")
    parser.add_argument("--suite", help=f"Run every scenario tagged with this tag (e.g. {REGRESSION_SUITE}).")
    parser.add_argument("--all", action="store_true", help="Run every *.yaml file when target is a directory (default).")
    parser.add_argument("--list", action="store_true", help="Print the selected scenarios and exit. Makes no network calls.")
    parser.add_argument("--repeat", type=int, default=1, help="Run the selection N times; all passes must pass (M9 gate uses 2).")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"Backend base URL (default {DEFAULT_BASE_URL}).")
    args = parser.parse_args(argv)

    if args.suite:
        scenarios = load_suite(args.suite)
        label = f"suite {args.suite!r}"
    elif args.target:
        scenarios = load_scenarios(args.target)
        label = str(args.target)
    else:
        parser.error("give a target path or --suite <tag>")

    if not scenarios:
        print(f"No scenarios found for {label}")
        return 1

    if args.list:
        print(f"{len(scenarios)} scenario(s) in {label}:")
        for scenario in scenarios:
            tags = ",".join(scenario.get("tags") or []) or "-"
            print(f"  {scenario['name']:<45} [{tags}]")
        return 0

    exit_code = 0
    for run_index in range(1, max(1, args.repeat) + 1):
        if args.repeat > 1:
            print(f"\n===== pass {run_index}/{args.repeat} — {label} =====")
        results = [run_scenario(args.base_url, scenario) for scenario in scenarios]
        for result in results:
            _print_result(result)
        failed = [r for r in results if not r.passed]
        print(f"\n{len(results) - len(failed)}/{len(results)} scenarios passed.")
        if failed:
            print("FAILED: " + ", ".join(r.name for r in failed))
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
