from src.chatbot.mini_llm_classifier import fallback_haul_classification


def test_fallback_classifies_lightweight_utility_keyword():
    result = fallback_haul_classification(
        category="Utility",
        user_message="I need a utility trailer for a golf cart",
    )

    assert result.is_lightweight_utility_load is True
    assert result.needs_width_question is False
    assert result.confidence == "high"
    assert result.matched_item == "golf cart"


def test_fallback_classifies_heavy_duty_width_for_non_utility_non_enclosed():
    result = fallback_haul_classification(
        category="Equipment",
        user_message="I need to haul a skid steer",
    )

    assert result.is_lightweight_utility_load is False
    assert result.needs_width_question is True
    assert result.confidence == "high"
    assert result.matched_item == "skid steer"


def test_fallback_does_not_add_width_for_utility_or_enclosed():
    utility = fallback_haul_classification(
        category="Utility",
        user_message="I need to haul a skid steer",
    )
    enclosed = fallback_haul_classification(
        category="Enclosed",
        user_message="I need to haul a skid steer",
    )

    assert utility.needs_width_question is False
    assert enclosed.needs_width_question is False


def test_fallback_does_not_add_width_for_flatbed():
    result = fallback_haul_classification(
        category="Flatbed",
        user_message="I need a flatbed trailer to haul a skid steer",
    )

    assert result.needs_width_question is False
