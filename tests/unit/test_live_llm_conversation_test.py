from __future__ import annotations

from scripts.live_llm_conversation_test import (
    build_flows,
    stage_one_messages,
    stage_three_messages,
    stage_two_messages,
)
from src.domain.categories import CANONICAL_CATEGORIES


def test_default_suite_has_fresh_flow_for_every_category_and_one_mixed_flow() -> None:
    flows = build_flows({1, 2, 3}, list(CANONICAL_CATEGORIES), "Test User", "test@example.com")
    assert len(flows) == len(CANONICAL_CATEGORIES) * 2 + 1
    assert sum(stage == 1 for stage, *_ in flows) == len(CANONICAL_CATEGORIES)
    assert sum(stage == 2 for stage, *_ in flows) == len(CANONICAL_CATEGORIES)
    assert sum(stage == 3 for stage, *_ in flows) == 1


def test_every_flow_starts_with_name_and_email() -> None:
    name = "Test User"
    email = "test@example.com"
    for category in CANONICAL_CATEGORIES:
        for messages in (
            stage_one_messages(category, name, email),
            stage_two_messages(category, name, email),
        ):
            assert name in messages[0]
            assert email in messages[0]
    mixed = stage_three_messages(name, email)
    assert name in mixed[0]
    assert email in mixed[0]


def test_stage_two_and_three_include_required_behavior_mix() -> None:
    stage_two = " ".join(stage_two_messages("Dump", "A", "a@example.com")).lower()
    assert "interested" in stage_two
    assert "financing" in stage_two
    assert "other trailer categories" in stage_two

    stage_three = " ".join(stage_three_messages("A", "a@example.com")).lower()
    for expected in ("financing", "interested", "change the category", "vague", "store hours"):
        assert expected in stage_three
