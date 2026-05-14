from src.chatbot.mini_preference_classifier import (
    PreferenceNullDecision,
    classify_no_preference,
)


def test_preference_classifier_returns_llm_decision(monkeypatch):
    expected = PreferenceNullDecision(
        has_no_preference=True,
        target_slots=["payload_need"],
        target_metadata_filters=["payload_lbs"],
        reason="User said they do not care about payload.",
        confidence="high",
    )

    class FakeLLM:
        def invoke(self, messages):
            assert messages
            return expected

    monkeypatch.setattr(
        "src.chatbot.mini_preference_classifier._preference_classifier_llm",
        lambda: FakeLLM(),
    )

    result = classify_no_preference(
        category="Aluminum",
        user_message="no preference regarding that",
        awaiting_slot="payload_need",
        pending_questions=[],
        slots_collected={},
        metadata_filters_collected={},
        allowed_category_slots=["base_category", "payload_need"],
    )

    assert result == expected


def test_preference_classifier_fallback_is_conservative(monkeypatch):
    class FailingLLM:
        def invoke(self, messages):
            raise RuntimeError("boom")

    monkeypatch.setattr(
        "src.chatbot.mini_preference_classifier._preference_classifier_llm",
        lambda: FailingLLM(),
    )

    result = classify_no_preference(
        category="Equipment",
        user_message="no preference",
        awaiting_slot="haul_weight_lbs",
    )

    assert result.has_no_preference is False
    assert result.confidence == "low"
    assert result.target_slots == []
    assert result.target_metadata_filters == []


def test_preference_classifier_ignores_plain_search_request(monkeypatch):
    class BadLLM:
        def invoke(self, messages):
            raise AssertionError("LLM should not be called without no-preference language")

    monkeypatch.setattr(
        "src.chatbot.mini_preference_classifier._preference_classifier_llm",
        lambda: BadLLM(),
    )

    result = classify_no_preference(
        category="Utility",
        user_message="I am looking for a 12 ft utility trailer",
        allowed_category_slots=["haul_item", "haul_weight_lbs"],
    )

    assert result.has_no_preference is False
    assert result.target_slots == []
    assert result.confidence == "low"


def test_preference_classifier_requires_active_question(monkeypatch):
    class BadLLM:
        def invoke(self, messages):
            raise AssertionError("LLM should not be called without an active question")

    monkeypatch.setattr(
        "src.chatbot.mini_preference_classifier._preference_classifier_llm",
        lambda: BadLLM(),
    )

    result = classify_no_preference(
        category="Utility",
        user_message="I have no idea",
        awaiting_slot=None,
        pending_questions=[],
        allowed_category_slots=["haul_item", "haul_weight_lbs"],
    )

    assert result.has_no_preference is False
    assert result.target_slots == []
    assert result.confidence == "low"


def test_preference_classifier_runs_for_no_idea_answer_to_active_question(monkeypatch):
    expected = PreferenceNullDecision(
        has_no_preference=True,
        target_slots=["haul_weight_lbs"],
        reason="User does not know the weight.",
        confidence="high",
    )

    class FakeLLM:
        def invoke(self, messages):
            assert messages
            return expected

    monkeypatch.setattr(
        "src.chatbot.mini_preference_classifier._preference_classifier_llm",
        lambda: FakeLLM(),
    )

    result = classify_no_preference(
        category="Utility",
        user_message="I have no idea",
        awaiting_slot="haul_weight_lbs",
        pending_questions=[],
        allowed_category_slots=["haul_item", "haul_weight_lbs"],
    )

    assert result == expected
