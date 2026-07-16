from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import convo_runner  # noqa: E402
from convo_runner import (  # noqa: E402
    REGRESSION_SUITE,
    evaluate_assertions,
    filter_by_tag,
    load_scenario,
    load_scenarios,
)

SCENARIOS_DIR = SCRIPTS_DIR / "scenarios"


def test_load_scenario_round_trips(tmp_path):
    path = tmp_path / "sample.yaml"
    path.write_text(
        "name: sample\nturns:\n  - user: \"hi\"\n    expect_state: {category: null}\n",
        encoding="utf-8",
    )
    scenario = load_scenario(path)
    assert scenario["name"] == "sample"
    assert scenario["turns"][0]["user"] == "hi"
    assert scenario["turns"][0]["expect_state"] == {"category": None}


def test_load_scenario_requires_name_and_turns(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("turns: []\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_scenario(path)


def test_load_scenarios_from_directory(tmp_path):
    (tmp_path / "a.yaml").write_text("name: a\nturns: []\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("name: b\nturns: []\n", encoding="utf-8")
    scenarios = load_scenarios(tmp_path)
    assert [s["name"] for s in scenarios] == ["a", "b"]


def test_expect_state_equality_pass_and_fail():
    turn = {"expect_state": {"category": "Dump"}}
    assert evaluate_assertions(turn, reply_text="", state={"category": "Dump"}, listings=[]) == []
    failures = evaluate_assertions(turn, reply_text="", state={"category": "Utility"}, listings=[])
    assert failures and "category" in failures[0]


def test_expect_state_contains_suffix():
    turn = {"expect_state": {"skipped_slots_contains": "haul_material"}}
    ok_state = {"skipped_slots": ["haul_material", "hitch_type"]}
    bad_state = {"skipped_slots": ["hitch_type"]}
    assert evaluate_assertions(turn, reply_text="", state=ok_state, listings=[]) == []
    failures = evaluate_assertions(turn, reply_text="", state=bad_state, listings=[])
    assert failures and "skipped_slots_contains" in failures[0]


def test_expect_state_slot_prefix():
    turn = {"expect_state": {"slot:trailer_length_ft": 15.0}}
    ok_state = {"slots": {"trailer_length_ft": 15.0}}
    bad_state = {"slots": {"trailer_length_ft": 20.0}}
    assert evaluate_assertions(turn, reply_text="", state=ok_state, listings=[]) == []
    failures = evaluate_assertions(turn, reply_text="", state=bad_state, listings=[])
    assert failures and "slot:trailer_length_ft" in failures[0]


def test_expect_reply_contains_any_and_not_contains():
    contains_turn = {"expect_reply_contains_any": ["axle", "hitch"]}
    assert evaluate_assertions(contains_turn, reply_text="Let's talk about your AXLE setup.", state={}, listings=[]) == []
    failures = evaluate_assertions(contains_turn, reply_text="Nothing relevant here.", state={}, listings=[])
    assert failures

    not_contains_turn = {"expect_reply_not_contains": ["email", "phone number"]}
    assert evaluate_assertions(not_contains_turn, reply_text="Sure, tell me more.", state={}, listings=[]) == []
    failures = evaluate_assertions(not_contains_turn, reply_text="What's your email?", state={}, listings=[])
    assert failures


def test_expect_listings_bool():
    turn_true = {"expect_listings": True}
    turn_false = {"expect_listings": False}
    assert evaluate_assertions(turn_true, reply_text="", state={}, listings=[{"title": "x"}]) == []
    assert evaluate_assertions(turn_false, reply_text="", state={}, listings=[]) == []
    assert evaluate_assertions(turn_true, reply_text="", state={}, listings=[]) != []
    assert evaluate_assertions(turn_false, reply_text="", state={}, listings=[{"title": "x"}]) != []


def test_expect_emails_sent_none_noop_and_mismatch():
    none_turn = {"expect_emails_sent": None}
    assert evaluate_assertions(none_turn, reply_text="", state={}, listings=[]) == []
    assert evaluate_assertions(none_turn, reply_text="", state={"emails_sent": ["FAQ - financing"]}, listings=[]) != []

    list_turn = {"expect_emails_sent": ["Escalation"]}
    assert evaluate_assertions(list_turn, reply_text="", state={"emails_sent": ["Escalation"]}, listings=[]) == []
    assert evaluate_assertions(list_turn, reply_text="", state={"emails_sent": []}, listings=[]) != []


def test_no_assertions_means_pass():
    assert evaluate_assertions({"user": "hi"}, reply_text="anything", state={}, listings=[]) == []


@pytest.mark.parametrize("path", sorted(SCENARIOS_DIR.glob("*.yaml")), ids=lambda p: p.stem)
def test_every_real_scenario_file_parses(path):
    scenario = load_scenario(path)
    assert scenario["name"] == path.stem
    assert isinstance(scenario["turns"], list) and scenario["turns"]
    for turn in scenario["turns"]:
        assert "user" in turn


def test_expected_scenario_set_is_complete():
    expected = {
        # M5
        "contact-full", "contact-partial-name", "contact-decline", "contact-ignore-asks-trailer",
        "category-direct", "category-info-vs-select", "category-exploration", "feature-request-no-category",
        "prefill-from-first-message", "interruption-repeat-once", "explicit-skip", "skip-all-show-results",
        "haul-item-lock", "clarification-term", "brand-only", "brand-plus-category", "brand-typo",
        "gooseneck-not-category", "defaults-applied", "loose-answers", "lightweight-utility-skips-weight",
        # M6
        "search-happy-path", "refine-length-change", "more-options-dedupe", "category-change-keep-some",
        "reference-second-listing", "brand-filter-search", "brand-filter-no-match", "inventory-stock-lookup",
        "inventory-first-message-contact-deferred", "inventory-weight-not-stock", "inventory-model-typo",
        "inventory-ambiguous-clarify", "inventory-mid-qna",
        # M7
        "faq-financing-with-contact", "escalate-quote-no-contact-then-provide", "escalate-decline-contact",
        "listing-interest-second-one", "faq-plus-escalate-one-message", "inventory-lookup-then-interest",
        "results-shown-alert", "unanswered-question-alert", "results-alert-stashed-then-sent",
        # M9 adversarial
        "adversarial-contradictory-sizes", "adversarial-category-change-twice",
        "adversarial-paragraph-requirements", "adversarial-faq-midqual-then-answer",
        "adversarial-gibberish-input",
        # Regressions replayed from live logs
        "category-change-keep-drop-asked",
        "category-change-default-not-carried", "category-change-show-me-instead",
        "vague-show-more-types-no-category", "category-change-wrong-answer-reasks",
    }
    actual = {path.stem for path in SCENARIOS_DIR.glob("*.yaml")}
    assert actual == expected


# ---------------------------------------------------------------------------
# M9: tagged regression suite
# ---------------------------------------------------------------------------

def test_every_scenario_declares_tags():
    """A scenario with no tags can never be selected by --suite, so it would rot."""
    for path in sorted(SCENARIOS_DIR.glob("*.yaml")):
        scenario = load_scenario(path)
        assert scenario["tags"], f"{path.stem} declares no tags"


def test_regression_suite_covers_every_scenario():
    """The M9 release gate must run everything — a scenario left out is a blind spot."""
    scenarios = load_scenarios(SCENARIOS_DIR)
    regression = filter_by_tag(scenarios, REGRESSION_SUITE)
    assert {s["name"] for s in regression} == {s["name"] for s in scenarios}
    assert len(regression) == 53  # 21 (M5) + 13 (M6) + 9 (M7) + 5 (M9 adversarial) + 5 (live-log regressions)


def test_adversarial_suite_is_the_five_new_scenarios():
    adversarial = filter_by_tag(load_scenarios(SCENARIOS_DIR), "adversarial")
    assert [s["name"] for s in adversarial] == [
        "adversarial-category-change-twice",
        "adversarial-contradictory-sizes",
        "adversarial-faq-midqual-then-answer",
        "adversarial-gibberish-input",
        "adversarial-paragraph-requirements",
    ]


def test_filter_by_tag_is_sorted_and_excludes_untagged(tmp_path):
    (tmp_path / "b.yaml").write_text("name: b\ntags: [regression]\nturns: [{user: hi}]\n", encoding="utf-8")
    (tmp_path / "a.yaml").write_text("name: a\ntags: [regression, m5]\nturns: [{user: hi}]\n", encoding="utf-8")
    (tmp_path / "c.yaml").write_text("name: c\ntags: [other]\nturns: [{user: hi}]\n", encoding="utf-8")
    (tmp_path / "d.yaml").write_text("name: d\nturns: [{user: hi}]\n", encoding="utf-8")

    assert [s["name"] for s in filter_by_tag(load_scenarios(tmp_path), "regression")] == ["a", "b"]
    assert [s["name"] for s in filter_by_tag(load_scenarios(tmp_path), "other")] == ["c"]
    assert filter_by_tag(load_scenarios(tmp_path), "nope") == []


def test_load_scenario_defaults_tags_to_empty(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text("name: s\nturns: [{user: hi}]\n", encoding="utf-8")
    assert load_scenario(path)["tags"] == []


def test_cli_list_selects_a_suite_without_network(capsys):
    """--list must not touch the backend; run_scenario is never called."""
    exit_code = convo_runner.main(["--suite", "adversarial", "--list"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "5 scenario(s) in suite 'adversarial'" in out
    assert "adversarial-gibberish-input" in out


def test_cli_requires_target_or_suite():
    with pytest.raises(SystemExit):
        convo_runner.main([])


def test_cli_unknown_suite_exits_nonzero(capsys):
    assert convo_runner.main(["--suite", "does-not-exist", "--list"]) == 1
    assert "No scenarios found" in capsys.readouterr().out
