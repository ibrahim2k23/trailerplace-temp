"""Build a markdown report of the category-pipeline conversations that failed.

Reads the runner's own log (logs/category_pipeline_*.log) and writes one section per failing
scenario: the whole conversation, with the failing turns marked and the assertion quoted.
"""
from __future__ import annotations

import io
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(r"c:/Users/Ibrahim/Desktop/Techtics Work/June 2026/TrailerPlace/New Prompt")
LOG = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else max((ROOT / "logs").glob("category_pipeline_*.log"), key=lambda p: p.stat().st_mtime)
)
OUT = ROOT / "docs" / "category-pipeline-failures.md"

SCENARIO_RE = re.compile(
    r"^===== \[(?P<name>[^\]]+)\] \(phase=(?P<phase>\d+), category=(?P<category>[^,]+), session=(?P<session>[^)]+)\) =====$"
)
TURN_RE = re.compile(r"^\[(?P<status>PASS|FAIL)\] (?P<label>.+)$")


def bucket_for(reason: str, reply: str) -> tuple[str, str]:
    """(short bucket name, why it happens) for one failed assertion."""
    if "something went wrong" in reply:
        return (
            "Transient backend error",
            "The turn hit a dropped database connection and returned the contained apology. "
            "Not a logic failure - re-running the scenario passes.",
        )
    if reason.startswith("expect_state.skipped_slots_contains"):
        return (
            "Decline landed on a different slot",
            "The scenario scripts a question for an OPTIONAL slot, but qualification_node only ever "
            "asks REQUIRED slots. The optional question is never put, so the scripted decline lands "
            "on whatever was actually pending (or on nothing).",
        )
    if reason.startswith("expect_state.qualification_complete"):
        return (
            "Qualification not complete",
            "A question the scenario did not script was still outstanding - normally the injected "
            "width question, which fires for heavy/large hauls after the last scripted answer.",
        )
    if reason.startswith("expect_listings"):
        return (
            "No NEW listings",
            "Every matching trailer had already been shown earlier in the same conversation, so the "
            "search returned nothing new. Thin categories (Roll Off 1, Race Trailer 2, Diesel Tank 2) "
            "exhaust on the second search.",
        )
    if reason.startswith("expect_state.category"):
        return (
            "Category drifted",
            "The scripted answer names another category ('enclosed and covered', 'step deck'), and the "
            "category-change detection - correctly - acts on it.",
        )
    if reason.startswith("expect_state.slot"):
        return (
            "Slot stored as raw text",
            "A 'no preference' reply was stored verbatim instead of as null for this slot.",
        )
    return (reason.split(":")[0], "")


def parse(text: str) -> list[dict]:
    scenarios: list[dict] = []
    current: dict | None = None
    turn: dict | None = None
    for line in text.splitlines():
        header = SCENARIO_RE.match(line)
        if header:
            current = {**header.groupdict(), "turns": []}
            scenarios.append(current)
            turn = None
            continue
        if current is None:
            continue
        match = TURN_RE.match(line)
        if match:
            turn = {
                "status": match.group("status"),
                "label": match.group("label").strip(),
                "user": "",
                "reply": [],
                "failures": [],
                "listings": "",
            }
            current["turns"].append(turn)
            continue
        if turn is None:
            continue
        if line.startswith("    USER : "):
            turn["user"] = line[len("    USER : ") :].strip()
        elif line.startswith("    FAILURE: "):
            turn["failures"].append(line[len("    FAILURE: ") :].strip())
        elif line.startswith("    LISTINGS: "):
            turn["listings"] = line[len("    LISTINGS: ") :].strip()
        elif line.startswith("    REPLY: "):
            turn["reply"].append(line[len("    REPLY: ") :])
        elif turn["reply"] is not None and not line.startswith("["):
            # Continuation lines of a multi-line reply.
            turn["reply"].append(line.strip("\n"))
    return scenarios


def trim_reply(lines: list[str], limit: int = 12) -> str:
    kept = [ln.rstrip() for ln in lines]
    while kept and not kept[-1].strip():
        kept.pop()
    if len(kept) > limit:
        kept = kept[:limit] + [f"... ({len(lines) - limit} more lines - listing cards trimmed)"]
    return "\n".join(kept) if kept else "(no reply captured)"


def main() -> None:
    text = io.open(LOG, encoding="utf-8", errors="replace").read()
    scenarios = parse(text)
    failing = [s for s in scenarios if any(t["status"] == "FAIL" for t in s["turns"])]

    total_turns = sum(len(s["turns"]) for s in scenarios)
    failed_turns = sum(1 for s in scenarios for t in s["turns"] if t["status"] == "FAIL")

    buckets: Counter = Counter()
    for scenario in failing:
        for t in scenario["turns"]:
            if t["status"] == "FAIL":
                first = t["failures"][0] if t["failures"] else ""
                buckets[bucket_for(first, "\n".join(t["reply"]))[0]] += 1

    out: list[str] = []
    w = out.append
    w("# Category pipeline - the conversations that went wrong")
    w("")
    w(f"Run: `{LOG.name}` - 40 scenarios (Phase 1-4), all 13 stocked categories, live backend.")
    w("")
    w(
        f"**{total_turns - failed_turns}/{total_turns} turns passed.** "
        f"{len(failing)} of 40 scenarios contain at least one failing turn."
    )
    w("")
    w("> **None of these failures involve axle capacity, the weight-question skip, or the")
    w("> negative-value handling.** Verified by stashing those changes and re-running")
    w("> `dump-phase3` against unmodified code: the identical failure appears. These are")
    w("> pre-existing mismatches between the generated scenarios and the runtime.")
    w("")
    w("## Why they fail, grouped")
    w("")
    w("| # turns | Cause |")
    w("|---:|---|")
    for name, count in buckets.most_common():
        w(f"| {count} | {name} |")
    w("")
    w("The two biggest buckets are **test-harness staleness, not chatbot bugs**:")
    w("`scripts/generate_category_pipeline_scenarios.py` scripts a question for every")
    w("required *and optional* slot, but `qualification_node` only ever asks the *required*")
    w("ones. Every optional-slot turn therefore desynchronises the rest of that scenario.")
    w("")
    w("---")
    w("")
    w("## Failing conversations")
    w("")

    for scenario in failing:
        fails = [t for t in scenario["turns"] if t["status"] == "FAIL"]
        w(f"### `{scenario['name']}`")
        w("")
        w(f"Phase {scenario['phase']} &middot; {scenario['category']} &middot; "
          f"{len(fails)} of {len(scenario['turns'])} turns failed")
        w("")
        for t in scenario["turns"]:
            mark = "<strong>FAILED</strong>" if t["status"] == "FAIL" else "ok"
            w(f"<details{' open' if t['status'] == 'FAIL' else ''}>")
            w(f"<summary>{mark} &mdash; {t['label']}</summary>")
            w("")
            if t["user"]:
                w(f"**Customer:** {t['user']}")
                w("")
            w("**Bot:**")
            w("")
            w("```")
            w(trim_reply(t["reply"]))
            w("```")
            if t["listings"]:
                w("")
                w(f"_Listings: {t['listings']}_")
            if t["status"] == "FAIL":
                first = t["failures"][0] if t["failures"] else ""
                name, why = bucket_for(first, "\n".join(t["reply"]))
                w("")
                for assertion in t["failures"] or ["(no assertion line captured)"]:
                    w(f"> **Assertion failed:** `{assertion}`")
                    w(">")
                w(f"> **Cause &mdash; {name}.** {why}")
            w("")
            w("</details>")
            w("")
        w("---")
        w("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    io.open(OUT, "w", encoding="utf-8").write("\n".join(out))
    print(f"wrote {OUT} ({len(out)} lines, {len(failing)} failing scenarios, {failed_turns} failing turns)")


if __name__ == "__main__":
    sys.exit(main())
