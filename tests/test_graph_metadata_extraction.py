from src.chatbot import graph


def _state(message: str, *, category: str) -> dict:
    return {
        "session_id": "s1",
        "user_message": message,
        "messages": [{"role": "user", "content": message}],
        "mind_decision": {
            "action": "respond",
            "trailer_category": category,
            "slots_collected_update": {},
            "metadata_filters_update": {},
        },
        "slots_collected": {},
        "metadata_filters_collected": {},
        "pending_questions": [],
        "asked_questions": [],
        "already_shown_listing_urls": [],
        "last_listings": [],
        "tool_events": [],
        "has_shown_search_results": False,
    }


def _use_fallback_extractor(monkeypatch):
    monkeypatch.setattr(
        graph,
        "_extract_filter_decision",
        lambda state, category: graph._fallback_filter_extraction(state),
    )
    monkeypatch.setattr(
        graph,
        "classify_haul_requirements",
        lambda **kwargs: graph.HaulClassificationDecision(),
    )
    monkeypatch.setattr(
        graph,
        "classify_no_preference",
        lambda **kwargs: graph.PreferenceNullDecision(),
    )


def _mock_haul_classifier(monkeypatch, **values):
    monkeypatch.setattr(
        graph,
        "classify_haul_requirements",
        lambda **kwargs: graph.HaulClassificationDecision(**values),
    )


def _mock_preference_classifier(monkeypatch, **values):
    monkeypatch.setattr(
        graph,
        "classify_no_preference",
        lambda **kwargs: graph.PreferenceNullDecision(**values),
    )


def _mock_extractor(monkeypatch, **values):
    monkeypatch.setattr(
        graph,
        "_extract_filter_decision",
        lambda state, category: graph.FilterExtractionDecision(**values),
    )
    monkeypatch.setattr(
        graph,
        "classify_haul_requirements",
        lambda **kwargs: graph.HaulClassificationDecision(),
    )
    monkeypatch.setattr(
        graph,
        "classify_no_preference",
        lambda **kwargs: graph.PreferenceNullDecision(),
    )


def test_livestock_length_in_message_satisfies_required_slot(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    out = graph._apply_mind_node(_state("I want a 12 feet livestock trailer", category="Livestock"))

    assert out["slots_collected"]["trailer_length_ft"] == "12 feet"
    assert out["metadata_filters_collected"]["length_ft"] == "12 feet"
    assert out["mind_decision"]["action"] == "pinecone_search"
    assert out["pending_questions"] == []


def test_livestock_width_is_metadata_filter_not_category_slot(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    out = graph._apply_mind_node(
        _state("I want a 12 feet livestock trailer, 6 feet wide", category="Livestock")
    )

    assert out["slots_collected"] == {"trailer_length_ft": "12 feet"}
    assert out["metadata_filters_collected"]["length_ft"] == "12 feet"
    assert out["metadata_filters_collected"]["width_ft"] == "6 feet"
    assert "width_ft" not in out["slots_collected"]
    assert "trailer_width_ft" not in out["slots_collected"]
    assert out["mind_decision"]["action"] == "pinecone_search"


def test_dimension_shorthand_uses_width_by_length(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    out = graph._apply_mind_node(_state("I am looking for a 6x12 livestock trailer", category="Livestock"))

    assert out["slots_collected"]["trailer_length_ft"] == "12"
    assert out["metadata_filters_collected"]["length_ft"] == "12"
    assert out["metadata_filters_collected"]["width_ft"] == "6"
    assert out["mind_decision"]["action"] == "pinecone_search"


def test_three_part_dimension_shorthand_uses_width_length_and_ignores_height(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    out = graph._apply_mind_node(_state("I am looking for a 6x12x5 livestock trailer", category="Livestock"))

    assert out["slots_collected"]["trailer_length_ft"] == "12"
    assert out["metadata_filters_collected"]["length_ft"] == "12"
    assert out["metadata_filters_collected"]["width_ft"] == "6"
    assert "height_ft" not in out["metadata_filters_collected"]
    assert out["mind_decision"]["action"] == "pinecone_search"


def test_equipment_required_length_and_weight_can_be_filled_from_filters(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    state = _state("I need an equipment trailer for a 3000 lb tractor 12 feet long", category="Equipment")
    state["mind_decision"]["slots_collected_update"] = {"haul_item": "tractor"}

    out = graph._apply_mind_node(state)

    assert out["slots_collected"]["haul_item"] == "tractor"
    assert out["slots_collected"]["haul_weight_lbs"] == "3000 lb"
    assert out["slots_collected"]["haul_length_ft"] == "12 feet"
    assert out["metadata_filters_collected"]["payload_lbs"] == "3000 lb"
    assert out["mind_decision"]["action"] == "pinecone_search"


def test_width_update_does_not_overwrite_existing_length(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    state = _state("construction equipment. the width should be atleast 6 ft", category="Utility")
    state["slots_collected"] = {
        "trailer_size": "12 ft",
        "haul_weight_lbs": "3000 lbs",
        "haul_item": "construction equipment",
    }
    state["metadata_filters_collected"] = {
        "length_ft": "12 ft",
        "payload_lbs": "3000 lbs",
    }

    out = graph._apply_mind_node(state)

    assert out["metadata_filters_collected"]["length_ft"] == "12 ft"
    assert out["metadata_filters_collected"]["width_ft"] == "6 ft"
    assert out["slots_collected"]["trailer_size"] == "12 ft"
    assert out["mind_decision"]["action"] == "pinecone_search"


def test_category_change_clears_old_filters_and_slots(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    state = _state("Now I need a 14 ft livestock trailer", category="Livestock")
    state["trailer_category"] = "Utility"
    state["has_shown_search_results"] = True
    state["slots_collected"] = {
        "haul_item": "equipment",
        "haul_weight_lbs": "3000 lbs",
        "trailer_size": "12 ft",
    }
    state["metadata_filters_collected"] = {
        "length_ft": "12 ft",
        "payload_lbs": "3000 lbs",
        "width_ft": "6 ft",
    }
    state["pending_questions"] = [{"slot": "haul_item", "question": "Old?", "required": True}]
    state["asked_questions"] = ["haul_item"]
    state["already_shown_listing_urls"] = ["https://example.test/old"]
    state["last_listings"] = [{"title": "old"}]

    out = graph._apply_mind_node(state)

    assert out["trailer_category"] == "Livestock"
    assert out["slots_collected"] == {"trailer_length_ft": "14 ft"}
    assert out["metadata_filters_collected"] == {"length_ft": "14 ft"}
    assert out["pending_questions"] == []
    assert out["asked_questions"] == []
    assert out["already_shown_listing_urls"] == []
    assert out["last_listings"] == []
    assert out["mind_decision"]["action"] == "pinecone_search"


def test_utility_lightweight_item_skips_weight_and_adds_payload_default(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    _mock_haul_classifier(
        monkeypatch,
        is_lightweight_utility_load=True,
        matched_item="golf cart",
        reason="Lightweight utility load.",
        confidence="high",
    )
    state = _state("I need a utility trailer for a golf cart", category="Utility")

    out = graph._apply_mind_node(state)

    assert out["slots_collected"] == {"haul_item": "golf cart"}
    assert out["metadata_filters_collected"]["payload_lbs"] == "1500 lbs"
    assert out["pending_questions"] == []
    assert out["mind_decision"]["action"] == "pinecone_search"


def test_utility_lightweight_removes_pending_weight_question(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    _mock_haul_classifier(
        monkeypatch,
        is_lightweight_utility_load=True,
        matched_item="golf cart",
        reason="Lightweight utility load.",
        confidence="high",
    )
    state = _state("It is a golf cart", category="Utility")
    state["slots_collected"] = {"haul_item": "golf cart"}
    state["pending_questions"] = [
        {
            "slot": "haul_weight_lbs",
            "question": "What's the rough total weight of your load?",
            "required": True,
        }
    ]

    out = graph._apply_mind_node(state)

    assert out["pending_questions"] == []
    assert out["awaiting_slot"] is None
    assert out["metadata_filters_collected"]["payload_lbs"] == "1500 lbs"
    assert out["mind_decision"]["action"] == "pinecone_search"


def test_utility_lightweight_user_weight_overrides_payload_default(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    _mock_haul_classifier(
        monkeypatch,
        is_lightweight_utility_load=True,
        matched_item="golf cart",
        reason="Lightweight utility load.",
        confidence="high",
    )
    state = _state("I need a utility trailer for a golf cart 900 lbs", category="Utility")
    state["mind_decision"]["slots_collected_update"] = {"haul_item": "golf cart"}

    out = graph._apply_mind_node(state)

    assert out["metadata_filters_collected"]["payload_lbs"] == "900 lbs"
    assert out["slots_collected"]["haul_weight_lbs"] == "900 lbs"
    assert out["mind_decision"]["action"] == "pinecone_search"


def test_utility_length_without_haul_item_asks_haul_question(monkeypatch):
    _mock_extractor(monkeypatch, length_ft="12")
    _mock_haul_classifier(
        monkeypatch,
        is_lightweight_utility_load=True,
        needs_width_question=False,
        matched_item=None,
        confidence="high",
    )

    out = graph._apply_mind_node(_state("I am looking for a 12 ft utility trailer", category="Utility"))

    assert out["mind_decision"]["action"] == "respond"
    assert out["assistant_text"] == "What will you be hauling on the utility trailer?"
    assert out["awaiting_slot"] == "haul_item"
    assert out["metadata_filters_collected"] == {"length_ft": "12"}
    assert "payload_lbs" not in out["metadata_filters_collected"]


def test_utility_generic_request_asks_haul_question(monkeypatch):
    _mock_extractor(monkeypatch)
    _mock_haul_classifier(
        monkeypatch,
        is_lightweight_utility_load=True,
        needs_width_question=False,
        matched_item=None,
        confidence="high",
    )

    out = graph._apply_mind_node(_state("I am looking for a utility trailer", category="Utility"))

    assert out["mind_decision"]["action"] == "respond"
    assert out["assistant_text"] == "What will you be hauling on the utility trailer?"
    assert out["awaiting_slot"] == "haul_item"
    assert out["metadata_filters_collected"] == {}


def test_no_preference_only_skips_current_question(monkeypatch):
    _mock_extractor(monkeypatch)
    _mock_preference_classifier(
        monkeypatch,
        has_no_preference=True,
        target_slots=["haul_item", "haul_weight_lbs"],
        confidence="high",
    )
    state = _state("no preference", category="Utility")
    state["awaiting_slot"] = "haul_weight_lbs"
    state["slots_collected"] = {"haul_item": "golf cart"}

    out = graph._apply_mind_node(state)

    assert out["slots_skipped"] == ["haul_weight_lbs"]
    assert out["slots_collected"] == {"haul_item": "golf cart"}
    assert out["mind_decision"]["action"] == "pinecone_search"


def test_utility_unknown_item_still_asks_weight(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    state = _state("I need a utility trailer for construction equipment", category="Utility")
    state["mind_decision"]["slots_collected_update"] = {"haul_item": "construction equipment"}

    out = graph._apply_mind_node(state)

    assert out["awaiting_slot"] == "haul_weight_lbs"
    assert out["assistant_text"] == "What's the rough total weight of your load?"
    assert out["mind_decision"]["action"] == "respond"


def test_heavy_equipment_adds_dynamic_width_question(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    _mock_haul_classifier(
        monkeypatch,
        needs_width_question=True,
        matched_item="skid steer",
        reason="Heavy-duty equipment.",
        confidence="high",
    )
    state = _state("I need an equipment trailer for a skid steer", category="Equipment")
    state["slots_collected"] = {
        "haul_item": "skid steer",
        "haul_weight_lbs": "7000 lbs",
        "haul_length_ft": "12 ft",
    }

    out = graph._apply_mind_node(state)

    assert out["awaiting_slot"] == "item_or_trailer_width_ft"
    assert out["assistant_text"] == "About how wide is the item, or what trailer width do you need?"
    assert out["mind_decision"]["action"] == "respond"


def test_classifier_matched_item_fills_haul_item_for_non_utility(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    _mock_haul_classifier(
        monkeypatch,
        needs_width_question=True,
        matched_item="car",
        reason="The user said they are hauling a car.",
        confidence="high",
    )
    state = _state("I am looking for a tilt trailer to haul my car", category="Tilt")

    out = graph._apply_mind_node(state)

    assert out["slots_collected"]["haul_item"] == "car"
    assert out["awaiting_slot"] == "haul_weight_lbs"
    assert out["assistant_text"] == "What's the approximate weight of the load?"
    assert out["mind_decision"]["action"] == "respond"


def test_heavy_equipment_width_filter_fills_dynamic_width_slot(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    _mock_haul_classifier(
        monkeypatch,
        needs_width_question=True,
        matched_item="skid steer",
        reason="Heavy-duty equipment.",
        confidence="high",
    )
    state = _state("I need an equipment trailer for a skid steer 6 ft wide", category="Equipment")
    state["slots_collected"] = {
        "haul_item": "skid steer",
        "haul_weight_lbs": "7000 lbs",
        "haul_length_ft": "12 ft",
    }

    out = graph._apply_mind_node(state)

    assert out["slots_collected"]["item_or_trailer_width_ft"] == "6 ft"
    assert out["metadata_filters_collected"]["width_ft"] == "6 ft"
    assert out["pending_questions"] == []
    assert out["mind_decision"]["action"] == "pinecone_search"


def test_enclosed_does_not_add_dynamic_width_question(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    _mock_haul_classifier(
        monkeypatch,
        needs_width_question=True,
        matched_item="skid steer",
        reason="Heavy-duty equipment.",
        confidence="high",
    )
    state = _state("I need an enclosed trailer for a skid steer", category="Enclosed")
    state["slots_collected"] = {
        "use_case": "hauling a skid steer",
        "cargo_size": "12 ft by 6 ft",
    }

    out = graph._apply_mind_node(state)

    assert "item_or_trailer_width_ft" not in out["slots_collected"]
    assert out["mind_decision"]["action"] == "pinecone_search"


def test_utility_does_not_add_dynamic_width_question(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    _mock_haul_classifier(
        monkeypatch,
        needs_width_question=True,
        matched_item="skid steer",
        reason="Heavy-duty equipment.",
        confidence="high",
    )
    state = _state("I need a utility trailer for a skid steer", category="Utility")
    state["slots_collected"] = {
        "haul_item": "skid steer",
        "haul_weight_lbs": "7000 lbs",
    }

    out = graph._apply_mind_node(state)

    assert "item_or_trailer_width_ft" not in out["slots_collected"]
    assert out["mind_decision"]["action"] == "pinecone_search"


def test_livestock_does_not_add_dynamic_width_question(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    _mock_haul_classifier(
        monkeypatch,
        needs_width_question=True,
        matched_item="livestock trailer",
        reason="Classifier confused trailer category for haul item.",
        confidence="high",
    )
    state = _state("I am looking for a 12 feet livestock trailer", category="Livestock")

    out = graph._apply_mind_node(state)

    assert out["slots_collected"]["trailer_length_ft"] == "12 feet"
    assert "item_or_trailer_width_ft" not in out["slots_collected"]
    assert out["pending_questions"] == []
    assert out["mind_decision"]["action"] == "pinecone_search"


def test_invalid_hitch_metadata_filter_is_rejected(monkeypatch):
    _mock_extractor(monkeypatch, hitch_type="Tilt")
    state = _state("around 77 inches", category="Tilt")
    state["slots_collected"] = {"haul_item": "a car", "haul_weight_lbs": "3000 lbs"}

    out = graph._apply_mind_node(state)

    assert "hitch_type" not in out["metadata_filters_collected"]


def test_valid_hitch_metadata_filters_are_normalized(monkeypatch):
    _mock_extractor(monkeypatch, hitch_type="gooseneck")
    state = _state("gooseneck please", category="Equipment")

    out = graph._apply_mind_node(state)

    assert out["metadata_filters_collected"]["hitch_type"] == "Gooseneck"

    _mock_extractor(monkeypatch, hitch_type="bumper pull")
    state = _state("bumper pull please", category="Equipment")
    out = graph._apply_mind_node(state)

    assert out["metadata_filters_collected"]["hitch_type"] == "Bumper Pull"


def test_subcategory_rejected_without_explicit_request(monkeypatch):
    _mock_extractor(monkeypatch, subcategory="Tilt")
    state = _state("I want a tilt trailer", category="Tilt")
    state["slots_collected"] = {"haul_item": "a car", "haul_weight_lbs": "3000 lbs"}

    out = graph._apply_mind_node(state)

    assert "subcategory" not in out["metadata_filters_collected"]


def test_subcategory_stored_when_explicitly_requested(monkeypatch):
    _mock_extractor(monkeypatch, subcategory="deckover")
    state = _state("filter by subcategory deckover", category="Aluminum")

    out = graph._apply_mind_node(state)

    assert out["metadata_filters_collected"]["subcategory"] == "Deckover"


def test_aluminum_base_category_maps_to_subcategory_filter(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    state = _state("utility trailer", category="Aluminum")
    state["awaiting_slot"] = "base_category"

    out = graph._apply_mind_node(state)

    assert out["slots_collected"]["base_category"] == "utility trailer"
    assert out["metadata_filters_collected"]["subcategory"] == "Utility"
    assert out["awaiting_slot"] == "payload_need"


def test_aluminum_base_category_none_leaves_subcategory_unset(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    _mock_preference_classifier(
        monkeypatch,
        has_no_preference=True,
        target_slots=["base_category"],
        target_metadata_filters=["subcategory"],
        reason="User has no base category preference.",
        confidence="high",
    )
    state = _state("no preference", category="Aluminum")
    state["awaiting_slot"] = "base_category"
    state["metadata_filters_collected"] = {"subcategory": "Utility"}

    out = graph._apply_mind_node(state)

    assert "base_category" not in out["slots_collected"]
    assert "base_category" in out["slots_skipped"]
    assert "subcategory" not in out["metadata_filters_collected"]
    assert out["awaiting_slot"] == "payload_need"


def test_aluminum_does_not_add_dynamic_width_question(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    _mock_haul_classifier(
        monkeypatch,
        needs_width_question=True,
        matched_item="10 feet aluminum trailer",
        reason="Classifier confused trailer phrase for haul item.",
        confidence="high",
    )
    state = _state("I am looking for a 10 feet aluminum trailer", category="Aluminum")

    out = graph._apply_mind_node(state)

    assert "item_or_trailer_width_ft" not in out["slots_collected"]
    assert all(q.get("slot") != "item_or_trailer_width_ft" for q in out["pending_questions"])
    assert out["awaiting_slot"] == "base_category"


def test_no_preference_skips_aluminum_payload_and_searches(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    _mock_preference_classifier(
        monkeypatch,
        has_no_preference=True,
        target_slots=["payload_need"],
        target_metadata_filters=["payload_lbs"],
        reason="User has no payload preference.",
        confidence="high",
    )
    state = _state("no preference regarding that", category="Aluminum")
    state["awaiting_slot"] = "payload_need"
    state["slots_skipped"] = ["base_category"]
    state["metadata_filters_collected"] = {"length_ft": "10", "payload_lbs": "3000 lbs"}

    out = graph._apply_mind_node(state)

    assert "payload_need" not in out["slots_collected"]
    assert set(out["slots_skipped"]) == {"base_category", "payload_need"}
    assert out["metadata_filters_collected"] == {"length_ft": "10"}
    assert out["pending_questions"] == []
    assert out["mind_decision"]["action"] == "pinecone_search"


def test_no_preference_skips_stale_width_question(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    _mock_preference_classifier(
        monkeypatch,
        has_no_preference=True,
        target_slots=["item_or_trailer_width_ft"],
        target_metadata_filters=["width_ft"],
        reason="User has no width preference.",
        confidence="high",
    )
    state = _state("no preference", category="Equipment")
    state["slots_collected"] = {
        "haul_item": "skid steer",
        "haul_weight_lbs": "7000 lbs",
        "haul_length_ft": "12 ft",
    }
    state["awaiting_slot"] = "item_or_trailer_width_ft"
    state["pending_questions"] = [
        {
            "slot": "item_or_trailer_width_ft",
            "question": "About how wide is the item, or what trailer width do you need?",
            "required": True,
        }
    ]
    state["metadata_filters_collected"] = {"width_ft": "6 ft"}

    out = graph._apply_mind_node(state)

    assert "item_or_trailer_width_ft" in out["slots_skipped"]
    assert "width_ft" not in out["metadata_filters_collected"]
    assert out["pending_questions"] == []
    assert out["mind_decision"]["action"] == "pinecone_search"


def test_no_preference_skips_equipment_weight(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    _mock_preference_classifier(
        monkeypatch,
        has_no_preference=True,
        target_slots=["haul_weight_lbs"],
        target_metadata_filters=["payload_lbs"],
        reason="User has no weight preference.",
        confidence="high",
    )
    state = _state("no preference", category="Equipment")
    state["slots_collected"] = {
        "haul_item": "tractor",
        "haul_length_ft": "12 ft",
    }
    state["awaiting_slot"] = "haul_weight_lbs"
    state["metadata_filters_collected"] = {"payload_lbs": "3000 lbs"}

    out = graph._apply_mind_node(state)

    assert "haul_weight_lbs" in out["slots_skipped"]
    assert "payload_lbs" not in out["metadata_filters_collected"]
    assert out["pending_questions"] == []
    assert out["mind_decision"]["action"] == "pinecone_search"
