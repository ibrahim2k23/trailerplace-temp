import inspect

from src.chatbot import graph, make_resolver, mini_llm_classifier, prompts, service


def test_roll_off_bin_yards_map_directly_to_trailer_length_feet():
    assert graph._canonicalize_adjudicated_metadata(
        key="length_ft",
        value="15 yd",
        category="Roll Off",
    ) == ("length_ft", "15 ft")


def test_all_existing_llm_prompts_forbid_category_as_implicit_haul_item():
    prompt_text = " ".join(
        [
            prompts.MIND_SYSTEM_PROMPT,
            inspect.getsource(graph._extract_field_updates),
            inspect.getsource(graph._adjudicate_active_question_turn),
            inspect.getsource(mini_llm_classifier.classify_haul_requirements),
        ]
    ).lower()

    assert prompt_text.count("critical haul-item rule") == 3
    assert "a trailer category names the requested trailer type, not its cargo" in prompt_text
    assert "i want an equipment trailer" in prompt_text
    assert "i need to haul equipment" in prompt_text


def _state(message: str, *, category: str) -> dict:
    return {
        "session_id": "s1",
        "user_message": message,
        "active_search_request_text": "",
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
        "_extract_field_updates",
        lambda **kwargs: graph._legacy_field_updates_from_filter_extraction(
            state=kwargs["state"],
            category=kwargs["category"],
            awaiting_slot=kwargs["awaiting_slot"],
            apply_slot_updates=kwargs["apply_slot_updates"],
        ),
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
    monkeypatch.setattr(
        graph,
        "resolve_make_from_text",
        lambda text, *, use_llm_fallback=True: make_resolver.resolve_make_from_text(
            text,
            use_llm_fallback=False,
        ),
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


def _mock_pre_generic_classifier(monkeypatch, **values):
    monkeypatch.setattr(
        graph,
        "_classify_non_recommendation_turn",
        lambda **kwargs: graph.NonRecommendationTurnDecision(**values),
    )


def _mock_extractor(monkeypatch, **values):
    monkeypatch.setattr(
        graph,
        "_extract_filter_decision",
        lambda state, category: graph.FilterExtractionDecision(**values),
    )
    monkeypatch.setattr(
        graph,
        "_extract_field_updates",
        lambda **kwargs: graph._legacy_field_updates_from_filter_extraction(
            state=kwargs["state"],
            category=kwargs["category"],
            awaiting_slot=kwargs["awaiting_slot"],
            apply_slot_updates=kwargs["apply_slot_updates"],
        ),
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


def _mock_field_updates(monkeypatch, **values):
    payload = {
        "metadata_filters_update": {},
        "slots_collected_update": {},
        "requested_non_metadata_features": [],
        "confidence": "high",
    }
    payload.update(values)
    if payload.get("confidence") == "low":
        payload["metadata_filters_update"] = {}
        payload["slots_collected_update"] = {}
        payload["requested_non_metadata_features"] = []

    def _extract(**kwargs):
        category = kwargs["category"]
        metadata = {}
        for key, value in payload["metadata_filters_update"].items():
            sanitized = graph._canonicalize_adjudicated_metadata(
                key=key,
                value=value,
                category=category,
            )
            if sanitized:
                clean_key, clean_value = sanitized
                metadata[clean_key] = clean_value
        slots = graph._canonicalize_adjudicated_slots(
            raw=payload["slots_collected_update"],
            allowed_category_slots=graph._category_slots(category),
            metadata_updates=metadata,
        )
        slots, features = graph._remove_generic_haul_use_feature_duplicates(
            slots,
            payload["requested_non_metadata_features"],
        )
        return graph.FieldExtractionAdjudicationDecision(
            metadata_filters_update=metadata,
            slots_collected_update=slots,
            requested_non_metadata_features=features,
            confidence=payload["confidence"],
            clarification_needed=payload.get("clarification_needed"),
            reason=payload.get("reason", ""),
        )

    monkeypatch.setattr(
        graph,
        "_extract_field_updates",
        _extract,
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


def _mock_make_resolution(monkeypatch, mapping: dict[tuple[str, bool], make_resolver.MakeResolution]):
    def _resolve_make(text, *, use_llm_fallback=True):
        key = (text, use_llm_fallback)
        if key in mapping:
            return mapping[key]
        return make_resolver.resolve_make_from_text(text, use_llm_fallback=False)

    monkeypatch.setattr(graph, "resolve_make_from_text", _resolve_make)


def test_livestock_length_in_message_satisfies_required_slot(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    out = graph._apply_mind_node(_state("I want a 12 feet livestock trailer", category="Livestock"))

    assert out["slots_collected"]["trailer_length_ft"] == "12 feet"
    assert out["metadata_filters_collected"]["length_ft"] == "12 feet"
    assert out["mind_decision"]["action"] == "pinecone_search"
    assert out["pending_questions"] == []


def test_active_search_request_preserves_original_features_after_length_reply(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    state = _state("12", category="Livestock")
    state["trailer_category"] = "Livestock"
    state["active_search_request_text"] = (
        "I am looking for a livestock trailer with offroad wheels and swinging gates"
    )
    state["awaiting_slot"] = "trailer_length_ft"
    state["messages"] = [
        {
            "role": "user",
            "content": "I am looking for a livestock trailer with offroad wheels and swinging gates",
        },
        {"role": "assistant", "content": "What length trailer are you looking for?"},
        {"role": "user", "content": "12"},
    ]
    state["mind_decision"]["action"] = "pinecone_search"

    out = graph._apply_mind_node(state)

    active = out["active_search_request_text"]
    assert "offroad wheels" in active
    assert "swinging gates" in active
    assert "length 12" in active
    assert not active.startswith("12 |")
    assert out["mind_decision"]["action"] == "pinecone_search"


def test_active_search_request_merges_freeform_feature_from_followup(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    state = _state("it should be a 12 ft trailer with offroad wheels", category="Livestock")
    state["trailer_category"] = "Livestock"
    state["active_search_request_text"] = "i am looking for a trailer livestock"
    state["awaiting_slot"] = "trailer_length_ft"
    state["messages"] = [
        {"role": "user", "content": "Ibrahim here. 03304388550"},
        {"role": "assistant", "content": "Thanks for sharing your contact details."},
        {"role": "user", "content": "i am looking for a trailer livestock"},
        {"role": "assistant", "content": "What length trailer are you looking for?"},
        {"role": "user", "content": "it should be a 12 ft trailer with offroad wheels"},
    ]
    state["mind_decision"]["action"] = "pinecone_search"

    out = graph._apply_mind_node(state)

    active = out["active_search_request_text"]
    assert "offroad wheels" in active
    assert "length 12" in active
    assert "03304388550" not in active
    assert active.startswith("i am looking for a trailer livestock")


def test_active_search_request_edits_length_without_duplicate(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    state = _state("make it 14 ft instead", category="Livestock")
    state["trailer_category"] = "Livestock"
    state["active_search_request_text"] = (
        "I am looking for a livestock trailer with swinging gates"
        " | Current requirements: length 12"
    )
    state["slots_collected"] = {"trailer_length_ft": "12"}
    state["metadata_filters_collected"] = {"length_ft": "12"}
    state["has_shown_search_results"] = True
    state["mind_decision"]["action"] = "pinecone_search"

    out = graph._apply_mind_node(state)

    active = out["active_search_request_text"]
    assert "swinging gates" in active
    assert "length 14 ft" in active
    assert "length 12" not in active
    assert out["metadata_filters_collected"]["length_ft"] == "14 ft"


def test_active_search_request_category_reset_drops_old_freeform_features(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    state = _state("show me dump trailers instead", category="Dump")
    state["trailer_category"] = "Livestock"
    state["active_search_request_text"] = (
        "i am looking for a trailer livestock; offroad wheels | Current requirements: length 12"
    )
    state["metadata_filters_collected"] = {"length_ft": "12"}
    state["messages"] = [
        {"role": "user", "content": "i am looking for a trailer livestock"},
        {"role": "assistant", "content": "What length trailer are you looking for?"},
        {"role": "user", "content": "it should be a 12 ft trailer with offroad wheels"},
        {"role": "user", "content": "show me dump trailers instead"},
    ]
    state["has_shown_search_results"] = True
    state["mind_decision"]["action"] = "pinecone_search"

    out = graph._apply_mind_node(state)

    active = out["active_search_request_text"]
    assert active.startswith("show me dump trailers instead")
    assert "offroad wheels" not in active
    assert "livestock" not in active.lower()


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


def test_generic_livestock_6x12_does_not_apply_inferred_make(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    _mock_make_resolution(
        monkeypatch,
        {
            (
                "I am looking for a 6x12 livestock trailer",
                True,
            ): make_resolver.MakeResolution("Calico Trailers", "high", "llm", "Inferred from ranked makes."),
        },
    )

    out = graph._apply_mind_node(_state("I am looking for a 6x12 livestock trailer", category=None))

    assert out["trailer_category"] == "Livestock"
    assert out["metadata_filters_collected"]["length_ft"] == "12"
    assert out["metadata_filters_collected"]["width_ft"] == "6"
    assert "make" not in out["metadata_filters_collected"]
    assert out["mind_decision"]["action"] == "pinecone_search"


def test_generic_livestock_12ft_does_not_apply_inferred_make(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    _mock_make_resolution(
        monkeypatch,
        {
            (
                "I am looking for a 12 ft livestock trailer",
                True,
            ): make_resolver.MakeResolution("Calico Trailers", "high", "llm", "Inferred from ranked makes."),
        },
    )

    out = graph._apply_mind_node(_state("I am looking for a 12 ft livestock trailer", category=None))

    assert out["trailer_category"] == "Livestock"
    assert out["metadata_filters_collected"]["length_ft"] == "12 ft"
    assert "make" not in out["metadata_filters_collected"]
    assert out["mind_decision"]["action"] == "pinecone_search"


def test_make_only_request_with_multiple_categories_asks_user_to_choose(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    state = _state("I am looking for a 6x12 Iron Bull trailer", category=None)

    out = graph._apply_mind_node(state)

    assert out["metadata_filters_collected"]["make"] == "Iron Bull Trailers"
    assert out["metadata_filters_collected"]["width_ft"] == "6"
    assert out["metadata_filters_collected"]["length_ft"] == "12"
    assert out["awaiting_slot"] == "make_category_choice"
    assert "Flatbed" in out["make_category_options"]
    assert "Which category" in out["assistant_text"]
    assert out["mind_decision"]["action"] == "respond"


def test_make_resolution_uses_llm_for_rd_trailer(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    llm_enabled_calls = []

    def _resolve_make(text, *, use_llm_fallback=True):
        if use_llm_fallback:
            llm_enabled_calls.append(text)
            return make_resolver.MakeResolution("RD TRAILERS", "high", "llm", "Resolved RD as make.")
        return make_resolver.MakeResolution()

    monkeypatch.setattr(graph, "resolve_make_from_text", _resolve_make)
    monkeypatch.setattr(graph, "categories_for_make", lambda make: ["Dump", "Utility"])
    state = _state("I am looking for an RD trailer", category=None)

    out = graph._apply_mind_node(state)

    assert out["metadata_filters_collected"]["make"] == "RD TRAILERS"
    assert out["awaiting_slot"] == "make_category_choice"
    assert "Which category" in out["assistant_text"]
    assert llm_enabled_calls == ["I am looking for an RD trailer"]


def test_same_category_generic_query_keeps_existing_make(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    _mock_make_resolution(
        monkeypatch,
        {
            (
                "I am looking for a 6x12 livestock trailer",
                True,
            ): make_resolver.MakeResolution("Calico Trailers", "high", "llm", "Inferred from ranked makes."),
        },
    )
    state = _state("I am looking for a 6x12 livestock trailer", category="Livestock")
    state["trailer_category"] = "Livestock"
    state["metadata_filters_collected"] = {"make": "Iron Bull Trailers"}

    out = graph._apply_mind_node(state)

    assert out["trailer_category"] == "Livestock"
    assert out["metadata_filters_collected"]["make"] == "Iron Bull Trailers"
    assert out["metadata_filters_collected"]["length_ft"] == "12"
    assert out["metadata_filters_collected"]["width_ft"] == "6"


def test_make_switch_clears_previous_qualifications_and_resets_active_request(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    state = _state("show me Diamond C instead", category="Equipment")
    state["trailer_category"] = "Equipment"
    state["slots_collected"] = {"haul_length_ft": "12"}
    state["metadata_filters_collected"] = {"make": "Aluma", "length_ft": "12"}
    state["active_search_request_text"] = (
        "I need an Aluma equipment trailer | Current requirements: make Aluma; length 12"
    )
    state["has_shown_search_results"] = True
    state["mind_decision"]["action"] = "pinecone_search"

    out = graph._apply_mind_node(state)

    assert out["metadata_filters_collected"] == {"make": "Diamond C"}
    assert out["slots_collected"] == {}
    assert out["already_shown_listing_urls"] == []
    assert out["last_listings"] == []
    assert out["active_search_request_text"].startswith("show me Diamond C instead")
    assert "Aluma" not in out["active_search_request_text"]
    assert "length 12" not in out["active_search_request_text"]


def test_category_change_clears_existing_make_before_generic_livestock_search(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    _mock_make_resolution(
        monkeypatch,
        {
            (
                "I am looking for a 12 ft livestock trailer",
                True,
            ): make_resolver.MakeResolution("Calico Trailers", "high", "llm", "Inferred from ranked makes."),
        },
    )
    state = _state("I am looking for a 12 ft livestock trailer", category="Livestock")
    state["trailer_category"] = "Utility"
    state["slots_collected"] = {"haul_length_ft": "14"}
    state["metadata_filters_collected"] = {"make": "Iron Bull Trailers", "length_ft": "14"}
    state["has_shown_search_results"] = True

    out = graph._apply_mind_node(state)

    assert out["trailer_category"] == "Livestock"
    assert out["slots_collected"] == {"trailer_length_ft": "12 ft"}
    assert out["metadata_filters_collected"]["length_ft"] == "12 ft"
    assert "make" not in out["metadata_filters_collected"]


def test_make_category_no_preference_keeps_existing_length_and_asks_payload_only(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    state = _state("any type", category=None)
    state["metadata_filters_collected"] = {
        "make": "Iron Bull Trailers",
        "width_ft": "6",
        "length_ft": "12",
    }
    state["awaiting_slot"] = "make_category_choice"
    state["make_category_options"] = ["Dump", "Equipment", "Flatbed", "Roll Off", "Tilt", "Utility"]

    out = graph._apply_mind_node(state)

    assert out["trailer_category"] is None
    assert out["metadata_filters_collected"]["make"] == "Iron Bull Trailers"
    assert out["metadata_filters_collected"]["width_ft"] == "6"
    assert out["metadata_filters_collected"]["length_ft"] == "12"
    assert out["awaiting_slot"] == "haul_weight_lbs"
    assert out["pending_questions"] == []
    assert "payload" in out["assistant_text"].lower() or "weight" in out["assistant_text"].lower()
    assert "make_category_choice" in out["slots_skipped"]


def test_make_category_doesnt_matter_with_make_reuse_asks_payload(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    state = _state("the category doesn't matter. I am looking for any iron bull trailer", category=None)
    state["metadata_filters_collected"] = {
        "make": "Iron Bull Trailers",
        "width_ft": "7",
        "length_ft": "14",
    }
    state["awaiting_slot"] = "make_category_choice"
    state["make_category_options"] = ["Dump", "Equipment", "Flatbed", "Roll Off", "Tilt", "Utility"]

    out = graph._apply_mind_node(state)

    assert out["trailer_category"] is None
    assert out["metadata_filters_collected"]["make"] == "Iron Bull Trailers"
    assert out["metadata_filters_collected"]["length_ft"] == "14"
    assert out["awaiting_slot"] == "haul_weight_lbs"
    assert out["assistant_text"] == "What payload or weight capacity do you need?"
    assert "Which category should I look at" not in out["assistant_text"]


def test_make_category_doesnt_matter_without_length_asks_length(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    state = _state("the category doesn't matter. I am looking for any iron bull trailer", category=None)
    state["metadata_filters_collected"] = {"make": "Iron Bull Trailers"}
    state["awaiting_slot"] = "make_category_choice"
    state["make_category_options"] = ["Dump", "Equipment", "Flatbed", "Roll Off", "Tilt", "Utility"]

    out = graph._apply_mind_node(state)

    assert out["trailer_category"] is None
    assert out["metadata_filters_collected"]["make"] == "Iron Bull Trailers"
    assert out["awaiting_slot"] == "trailer_length_ft"
    assert out["assistant_text"] == "What trailer length would you prefer?"
    assert "Which category should I look at" not in out["assistant_text"]


def test_make_category_no_idea_is_treated_as_no_preference(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    state = _state("no idea", category=None)
    state["metadata_filters_collected"] = {
        "make": "Iron Bull Trailers",
        "width_ft": "6",
        "length_ft": "12",
    }
    state["awaiting_slot"] = "make_category_choice"
    state["make_category_options"] = ["Dump", "Equipment", "Flatbed", "Roll Off", "Tilt", "Utility"]

    out = graph._apply_mind_node(state)

    assert out["trailer_category"] is None
    assert out["awaiting_slot"] == "haul_weight_lbs"
    assert "Which category should I use" not in out["assistant_text"]


def test_make_category_no_type_in_mind_is_treated_as_no_preference(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    state = _state("honestly, I have no type in mind", category=None)
    state["metadata_filters_collected"] = {
        "make": "Iron Bull Trailers",
        "width_ft": "6",
        "length_ft": "12",
    }
    state["awaiting_slot"] = "make_category_choice"
    state["make_category_options"] = ["Dump", "Equipment", "Flatbed", "Roll Off", "Tilt", "Utility"]

    out = graph._apply_mind_node(state)

    assert out["trailer_category"] is None
    assert out["awaiting_slot"] == "haul_weight_lbs"
    assert "Which category should I use" not in out["assistant_text"]


def test_make_category_no_preference_uses_mini_classifier(monkeypatch):
    _use_fallback_extractor(monkeypatch)

    def _preference_classifier(**kwargs):
        if kwargs["awaiting_slot"] == "make_category_choice":
            assert "Which category should I use" in kwargs["active_question"]
            return graph.PreferenceNullDecision(
                has_no_preference=True,
                target_slots=["make_category_choice"],
                reason="User is flexible on category.",
                confidence="high",
            )
        return graph.PreferenceNullDecision()

    monkeypatch.setattr(graph, "classify_no_preference", _preference_classifier)
    state = _state("I'm flexible on the category", category=None)
    state["metadata_filters_collected"] = {
        "make": "Iron Bull Trailers",
        "width_ft": "6",
        "length_ft": "12",
    }
    state["awaiting_slot"] = "make_category_choice"
    state["make_category_options"] = ["Dump", "Equipment", "Flatbed", "Roll Off", "Tilt", "Utility"]

    out = graph._apply_mind_node(state)

    assert out["trailer_category"] is None
    assert out["awaiting_slot"] == "haul_weight_lbs"
    assert "Which category should I use" not in out["assistant_text"]


def test_make_category_no_preference_searches_when_size_and_payload_known(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    state = _state("any type", category=None)
    state["metadata_filters_collected"] = {
        "make": "Iron Bull Trailers",
        "width_ft": "6",
        "length_ft": "12",
        "payload_lbs": "2000 pounds",
    }
    state["awaiting_slot"] = "make_category_choice"
    state["make_category_options"] = ["Dump", "Equipment", "Flatbed", "Roll Off", "Tilt", "Utility"]

    out = graph._apply_mind_node(state)

    assert out["trailer_category"] is None
    assert out["pending_questions"] == []
    assert out["awaiting_slot"] is None
    assert out["mind_decision"]["action"] == "pinecone_search"


def test_make_category_choice_preserves_payload_hitch_and_budget(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    state = _state("I want an Iron Bull gooseneck trailer under 10000 with 2000 pounds payload", category=None)

    out = graph._apply_mind_node(state)

    assert out["awaiting_slot"] == "make_category_choice"
    assert out["metadata_filters_collected"]["make"] == "Iron Bull Trailers"
    assert out["metadata_filters_collected"]["hitch_type"] == "Gooseneck"
    assert out["metadata_filters_collected"]["max_price"] == "10000"
    assert out["metadata_filters_collected"]["payload_lbs"] == "2000 pounds"


def test_make_and_category_request_sets_both_without_choice(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    state = _state("I need a Diamond C flatbed trailer", category=None)

    out = graph._apply_mind_node(state)

    assert out["trailer_category"] == "Flatbed"
    assert out["metadata_filters_collected"]["make"] == "Diamond C"
    assert out["awaiting_slot"] == "haul_item"


def test_gooseneck_hitch_does_not_set_make_filter(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    state = _state("I need a gooseneck flatbed trailer", category=None)

    out = graph._apply_mind_node(state)

    assert out["trailer_category"] == "Flatbed"
    assert "make" not in out["metadata_filters_collected"]
    assert out["metadata_filters_collected"]["hitch_type"] == "Gooseneck"


def test_make_category_no_preference_falls_back_to_length_capacity(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    state = _state("I do not know", category=None)
    state["metadata_filters_collected"] = {"make": "Diamond C"}
    state["awaiting_slot"] = "make_category_choice"
    state["make_category_options"] = ["Car Hauler", "Dump", "Equipment", "Flatbed"]

    out = graph._apply_mind_node(state)

    assert out["trailer_category"] is None
    assert out["metadata_filters_collected"]["make"] == "Diamond C"
    assert out["awaiting_slot"] == "trailer_length_ft"
    assert out["pending_questions"][0]["slot"] == "haul_weight_lbs"
    assert "make_category_choice" in out["slots_skipped"]


def test_make_category_declined_then_dimension_asks_payload_not_category(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    state = _state("a 8x18", category=None)
    state["metadata_filters_collected"] = {"make": "Diamond C"}
    state["slots_skipped"] = ["make_category_choice"]

    out = graph._apply_mind_node(state)

    assert out["trailer_category"] is None
    assert out["metadata_filters_collected"]["make"] == "Diamond C"
    assert out["metadata_filters_collected"]["width_ft"] == "8"
    assert out["metadata_filters_collected"]["length_ft"] == "18"
    assert out["awaiting_slot"] == "haul_weight_lbs"
    assert out["assistant_text"] == "What payload or weight capacity do you need?"


def test_make_category_declined_searches_when_length_and_payload_known(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    state = _state("3000 pounds", category=None)
    state["metadata_filters_collected"] = {
        "make": "Diamond C",
        "length_ft": "18",
        "width_ft": "8",
    }
    state["slots_skipped"] = ["make_category_choice"]

    out = graph._apply_mind_node(state)

    assert out["metadata_filters_collected"]["payload_lbs"] == "3000 pounds"
    assert out["mind_decision"]["action"] == "pinecone_search"


def test_dimension_shorthand_uses_width_by_length(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    out = graph._apply_mind_node(_state("I am looking for a 6x12 livestock trailer", category="Livestock"))

    assert out["slots_collected"]["trailer_length_ft"] == "12"
    assert out["metadata_filters_collected"]["length_ft"] == "12"
    assert out["metadata_filters_collected"]["width_ft"] == "6"
    assert out["mind_decision"]["action"] == "pinecone_search"


def test_three_part_dimension_shorthand_uses_width_length_and_stores_height(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    out = graph._apply_mind_node(_state("I am looking for a 6x12x5 livestock trailer", category="Livestock"))

    assert out["slots_collected"]["trailer_length_ft"] == "12"
    assert out["metadata_filters_collected"]["length_ft"] == "12"
    assert out["metadata_filters_collected"]["width_ft"] == "6"
    assert out["metadata_filters_collected"]["height_ft"] == "5"
    assert out["mind_decision"]["action"] == "pinecone_search"


def test_side_wall_wording_maps_to_height(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    out = graph._apply_mind_node(_state("I need a dump trailer with 3 inch sides", category="Dump"))

    assert out["metadata_filters_collected"]["height_ft"] == "0.25 ft"


def test_equipment_required_length_and_weight_can_be_filled_from_filters(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    state = _state("I need an equipment trailer for a 3000 lb tractor 12 feet long", category="Equipment")
    state["mind_decision"]["slots_collected_update"] = {"haul_item": "tractor"}

    out = graph._apply_mind_node(state)

    assert out["slots_collected"]["haul_item"] == "tractor"
    assert out["slots_collected"]["haul_weight_lbs"] == "3000 lb"
    assert out["slots_collected"]["haul_length_ft"] == "12 feet"
    assert out["metadata_filters_collected"]["payload_lbs"] == "3000 lb"
    assert out["awaiting_slot"] == "hitch_type"
    assert out["assistant_text"] == "Do you prefer a bumper pull or gooseneck hitch?"
    assert out["mind_decision"]["action"] == "respond"


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
    state["active_search_request_text"] = "utility trailer with sliding gates | Current requirements: length 12 ft"
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
    assert out["active_search_request_text"].startswith("Now I need a 14 ft livestock trailer")
    assert "sliding gates" not in out["active_search_request_text"].lower()
    assert out["mind_decision"]["action"] == "pinecone_search"


def test_category_change_resets_active_question_attempts(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    state = _state("I am looking for a dump trailer", category="Dump")
    state["trailer_category"] = "Equipment"
    state["awaiting_slot"] = "haul_weight_lbs"
    state["active_question_attempts"] = {"haul_weight_lbs": 1}
    state["active_question_unanswered_count"] = 1
    state["active_question_tracker"] = {
        "slot": "haul_weight_lbs",
        "question": "What's the rough total weight of the load?",
        "unanswered_count": 1,
    }

    out = graph._apply_mind_node(state)

    assert out["trailer_category"] == "Dump"
    assert out["active_question_attempts"] == {}
    assert out["repeated_unanswered_question_escalation"] is False


def test_category_change_confirms_only_old_common_filters(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    state = _state("I am looking for a 20 ft livestock trailer", category="Livestock")
    state["trailer_category"] = "Utility"
    state["slots_collected"] = {"haul_item": "mower"}
    state["metadata_filters_collected"] = {
        "length_ft": "12 ft",
        "width_ft": "7 ft",
        "payload_lbs": "5000 lbs",
        "hitch_type": "gooseneck",
        "color": "black",
    }

    changed = graph._apply_mind_node(state)

    assert changed["trailer_category"] == "Livestock"
    assert changed["metadata_filters_collected"] == {"length_ft": "20 ft"}
    assert changed["pending_category_change"]["carry_filters"] == {
        "width_ft": "7 ft",
        "payload_lbs": "5000 lbs",
        "hitch_type": "gooseneck",
    }
    assert "length" not in changed["assistant_text"].split("previous", 1)[-1]

    changed["user_message"] = "yes"
    changed["mind_decision"] = {"action": "respond", "trailer_category": "Livestock"}
    kept = graph._apply_mind_node(changed)

    assert kept["pending_category_change"] is None
    assert kept["metadata_filters_collected"]["length_ft"] == "20 ft"
    assert kept["metadata_filters_collected"]["width_ft"] == "7 ft"
    assert kept["metadata_filters_collected"]["payload_lbs"] == "5000 lbs"
    assert kept["metadata_filters_collected"]["hitch_type"] == "gooseneck"
    assert "color" not in kept["metadata_filters_collected"]


def test_llm_field_updates_accept_24_ft_cattle_with_requested_feature(monkeypatch):
    _mock_field_updates(
        monkeypatch,
        metadata_filters_update={"length_ft": "24 ft"},
        slots_collected_update={"trailer_length_ft": "24 ft"},
        requested_non_metadata_features=["sliding gates"],
    )
    state = _state("I am looking for a 24 ft cattle trailer with sliding gates", category="Livestock")

    out = graph._apply_mind_node(state)

    assert out["trailer_category"] == "Livestock"
    assert out["metadata_filters_collected"]["length_ft"] == "24 ft"
    assert out["slots_collected"]["trailer_length_ft"] == "24 ft"
    assert out["requested_non_metadata_features"] == ["sliding gates"]
    assert out["mind_decision"]["action"] == "pinecone_search"


def test_llm_field_updates_accept_288_inches_as_livestock_length(monkeypatch):
    _mock_field_updates(
        monkeypatch,
        metadata_filters_update={"length_ft": "288 inch"},
        slots_collected_update={"trailer_length_ft": "288 inch"},
    )
    state = _state("I am looking for a 288 inch trailer to haul cattle", category="Livestock")

    out = graph._apply_mind_node(state)

    assert out["metadata_filters_collected"]["length_ft"] == "24 ft"
    assert out["slots_collected"]["trailer_length_ft"] == "24 ft"
    assert out["mind_decision"]["action"] == "pinecone_search"


def test_llm_field_updates_extract_multiple_constraints_in_one_turn(monkeypatch):
    _mock_field_updates(
        monkeypatch,
        metadata_filters_update={
            "length_ft": "24 ft",
            "color": "Black",
            "hitch_type": "gooseneck",
            "max_price": "30000",
        },
        slots_collected_update={"trailer_length_ft": "24 ft"},
        requested_non_metadata_features=["sliding gates"],
    )
    state = _state("I need a 24 ft black gooseneck cattle trailer under 30000 with sliding gates", category="Livestock")

    out = graph._apply_mind_node(state)

    assert out["metadata_filters_collected"]["length_ft"] == "24 ft"
    assert out["metadata_filters_collected"]["color"] == "black"
    assert out["metadata_filters_collected"]["hitch_type"] == "Gooseneck"
    assert out["metadata_filters_collected"]["max_price"] == "30000"
    assert out["slots_collected"]["trailer_length_ft"] == "24 ft"
    assert out["requested_non_metadata_features"] == ["sliding gates"]


def test_make_still_uses_existing_resolver_when_field_updates_do_not_extract_make(monkeypatch):
    _mock_field_updates(monkeypatch)
    state = _state("I need a Diamond C flatbed trailer", category=None)

    out = graph._apply_mind_node(state)

    assert out["trailer_category"] == "Flatbed"
    assert out["metadata_filters_collected"]["make"] == "Diamond C"


def test_non_aluminum_subcategory_candidate_is_discarded(monkeypatch):
    _mock_field_updates(
        monkeypatch,
        metadata_filters_update={"subcategory": "equipment", "length_ft": "24 ft"},
        slots_collected_update={"trailer_length_ft": "24 ft"},
    )
    state = _state("I need a 24 ft cattle trailer", category="Livestock")

    out = graph._apply_mind_node(state)

    assert "subcategory" not in out["metadata_filters_collected"]
    assert out["metadata_filters_collected"]["length_ft"] == "24 ft"


def test_aluminum_subcategory_candidate_is_allowed(monkeypatch):
    _mock_field_updates(
        monkeypatch,
        metadata_filters_update={"subcategory": "Utility", "payload_lbs": "3000 lbs"},
        slots_collected_update={"base_category": "Utility", "payload_need": "3000 lbs"},
    )
    state = _state("I need an aluminum utility trailer for 3000 lbs", category="Aluminum")

    out = graph._apply_mind_node(state)

    assert out["metadata_filters_collected"]["subcategory"] == "Utility"
    assert out["metadata_filters_collected"]["payload_lbs"] == "3000 lbs"


def test_llm_field_updates_width_only_does_not_become_length(monkeypatch):
    _mock_field_updates(
        monkeypatch,
        metadata_filters_update={"width_ft": "6 feet"},
    )
    state = _state("6 feet wide", category="Livestock")
    state["slots_collected"] = {"trailer_length_ft": "24 ft"}
    state["metadata_filters_collected"] = {"length_ft": "24 ft"}

    out = graph._apply_mind_node(state)

    assert out["metadata_filters_collected"]["length_ft"] == "24 ft"
    assert out["metadata_filters_collected"]["width_ft"] == "6 feet"
    assert out["slots_collected"]["trailer_length_ft"] == "24 ft"


def test_llm_field_updates_awaited_weight_answer_sets_payload(monkeypatch):
    _mock_field_updates(
        monkeypatch,
        metadata_filters_update={"payload_lbs": "5000 pounds"},
        slots_collected_update={"haul_weight_lbs": "5000 pounds"},
    )
    state = _state("5000 pounds", category="Dump")
    state["awaiting_slot"] = "haul_weight_lbs"
    state["slots_collected"] = {"haul_material": "gravel"}

    out = graph._apply_mind_node(state)

    assert out["metadata_filters_collected"]["payload_lbs"] == "5000 pounds"
    assert out["slots_collected"]["haul_weight_lbs"] == "5000 pounds"


def test_category_change_clears_old_requested_non_metadata_features(monkeypatch):
    _mock_field_updates(
        monkeypatch,
        metadata_filters_update={"length_ft": "24 ft"},
        slots_collected_update={"trailer_length_ft": "24 ft"},
        requested_non_metadata_features=[],
    )
    state = _state("Now I need a 24 ft livestock trailer", category="Livestock")
    state["trailer_category"] = "Dump"
    state["has_shown_search_results"] = True
    state["requested_non_metadata_features"] = ["sliding gates"]
    state["slots_collected"] = {"haul_material": "gravel", "haul_weight_lbs": "5000 lbs"}
    state["metadata_filters_collected"] = {"payload_lbs": "5000 lbs"}

    out = graph._apply_mind_node(state)

    assert out["trailer_category"] == "Livestock"
    assert out["requested_non_metadata_features"] == []
    assert out["metadata_filters_collected"] == {"length_ft": "24 ft"}


def test_low_confidence_field_updates_do_not_store_new_values(monkeypatch):
    _mock_field_updates(
        monkeypatch,
        metadata_filters_update={"length_ft": "24 ft"},
        slots_collected_update={"trailer_length_ft": "24 ft"},
        confidence="low",
        clarification_needed="What length trailer are you looking for?",
    )
    state = _state("maybe something big", category="Livestock")

    out = graph._apply_mind_node(state)

    assert out["metadata_filters_collected"] == {}
    assert out["slots_collected"] == {}
    assert out["assistant_text"] == "What length trailer are you looking for?"


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


def test_main_llm_resolves_category_without_inventing_make(monkeypatch):
    class _FakeMindLLM:
        def invoke(self, _messages):
            return graph.MindDecision(
                action="ask_next_question",
                trailer_category="Livestock",
            )

    monkeypatch.setattr(graph, "_mind_llm", lambda: _FakeMindLLM())

    message = "suggest me a trailer to load live stock"
    planned = graph._mind_node(_state(message, category=None))
    make = make_resolver.resolve_make_from_text(message)

    assert planned["mind_decision"]["trailer_category"] == "Livestock"
    assert make.make is None


def test_generic_6x12_trailer_request_extracts_size_and_asks_category(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    _mock_pre_generic_classifier(
        monkeypatch,
        action="ask_trailer_category",
        confidence="high",
        reason="Generic size-only trailer request needs category.",
    )
    state = _state("I am looking for a 6x12 trailer", category=None)
    state["mind_decision"]["action"] = "ask_next_question"

    out = graph._apply_mind_node(state)

    assert out["trailer_category"] is None
    assert out["metadata_filters_collected"]["width_ft"] == "6"
    assert out["metadata_filters_collected"]["length_ft"] == "12"
    assert out["assistant_text"] == "What type of trailer are you looking for?"
    assert out["awaiting_slot"] == "generic_category_choice"
    assert out["mind_decision"]["action"] == "respond"


def test_mind_owns_catalogue_category_question_and_followup(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    catalogue_reply = (
        "We carry Utility, Dump, Equipment, Enclosed, and other trailer types, "
        "each suited to different hauling needs. Which type interests you?"
    )
    _mock_pre_generic_classifier(
        monkeypatch,
        action="respond",
        assistant_text="This opposing classifier response must not replace the mind.",
        confidence="high",
    )
    state = _state("Which trailers do you have and what are they used for?", category=None)
    state["mind_decision"] = {
        **state["mind_decision"],
        "action": "ask_trailer_category",
        "assistant_text": catalogue_reply,
    }

    first = graph._apply_mind_node(state)

    assert first["assistant_text"] == catalogue_reply
    assert first["awaiting_slot"] == "generic_category_choice"

    second = graph._apply_mind_node({
        **first,
        "user_message": "Dump",
        "mind_decision": {
            **first["mind_decision"],
            "action": "respond",
            "assistant_text": "",
            "trailer_category": "Dump",
            "category_resolution_kind": "explicit",
            "category_confidence": "high",
        },
    })

    assert second["trailer_category"] == "Dump"
    assert second["awaiting_slot"] != "generic_category_choice"


def test_substantive_mind_response_is_preserved_when_requirements_complete(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    advice = (
        "A Dump trailer is versatile for those materials, while a Flatbed can be "
        "better for securing furniture and pipes. Which would you like to explore?"
    )
    state = _state(
        "Which trailer type is best for hauling wood, tires, furniture, and pipes?",
        category="Dump",
    )
    state["slots_collected"] = {
        "haul_material": "wood, tires, furniture, pipes",
        "haul_weight_lbs": "5000 lbs",
    }
    state["has_shown_search_results"] = True
    state["mind_decision"]["assistant_text"] = advice
    state["mind_decision"]["category_recommendations"] = [
        {"category": "Dump"},
        {"category": "Flatbed"},
    ]

    out = graph._apply_mind_node(state)

    assert out["mind_decision"]["action"] == "respond"
    assert out["assistant_text"] == advice


def test_final_active_answer_still_searches_despite_mind_acknowledgement(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    monkeypatch.setattr(
        graph,
        "_adjudicate_active_question_turn",
        lambda **kwargs: graph.QuestionTurnDecision(
            answered_active_question=True,
            active_slot_value="5000 lbs",
            confidence="high",
        ),
    )
    state = _state("5000 pounds", category="Dump")
    state["slots_collected"] = {"haul_material": "wood"}
    state["awaiting_slot"] = "haul_weight_lbs"
    state["mind_decision"]["assistant_text"] = "Thanks, that gives me what I need."

    out = graph._apply_mind_node(state)

    assert out["mind_decision"]["action"] == "pinecone_search"


def test_pre_results_qualification_still_uses_pending_question(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    explanation = "A tandem axle generally provides better stability for heavier loads."
    state = _state("Why would I need tandem axles?", category="Dump")
    state["mind_decision"]["assistant_text"] = explanation

    out = graph._apply_mind_node(state)

    assert out["mind_decision"]["action"] == "respond"
    assert out["assistant_text"] == "What material will you be hauling (dirt, gravel, debris, etc.)?"
    assert out["awaiting_slot"] == "haul_material"


def test_explicit_pinecone_action_still_searches_when_complete(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    state = _state("Show me trailers", category="Dump")
    state["slots_collected"] = {
        "haul_material": "wood",
        "haul_weight_lbs": "5000 lbs",
    }
    state["mind_decision"]["action"] = "pinecone_search"

    out = graph._apply_mind_node(state)

    assert out["mind_decision"]["action"] == "pinecone_search"


def test_generic_category_question_allows_medium_confidence_gate(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    _mock_pre_generic_classifier(
        monkeypatch,
        action="ask_trailer_category",
        confidence="medium",
        reason="Medium confidence is enough to ask category.",
    )
    state = _state("I am looking for a 6x12 trailer", category=None)
    state["mind_decision"]["action"] = "ask_next_question"

    out = graph._apply_mind_node(state)

    assert out["assistant_text"] == "What type of trailer are you looking for?"
    assert out.get("awaiting_slot") == "generic_category_choice"


def test_recommendation_request_uses_structured_response_not_generic_category(monkeypatch):
    _mock_field_updates(
        monkeypatch,
        slots_collected_update={"generic_haul_use": "hauling heavy vehicles"},
    )
    _mock_pre_generic_classifier(
        monkeypatch,
        action="respond",
        assistant_text="For heavy vehicles, I would compare equipment and car hauler options.",
        confidence="high",
        reason="User asked for a recommendation, not a category clarification.",
    )
    state = _state(
        "I am looking for a trailer to haul heavy vehicles. Can you recommend a type?",
        category=None,
    )
    state["mind_decision"]["action"] = "ask_next_question"

    out = graph._apply_mind_node(state)

    assert out["assistant_text"] == "For heavy vehicles, I would compare equipment and car hauler options."
    assert out.get("awaiting_slot") != "generic_category_choice"


def test_recommendation_stores_multiple_category_suggestions(monkeypatch):
    class _FakeMindLLM:
        def invoke(self, _messages):
            return graph.MindDecision(
                action="respond",
                trailer_category="Equipment",
                category_resolution_kind="recommendation",
                category_confidence="high",
                category_recommendations=[
                    {"category": "Equipment", "confidence": "high", "reasoning": "Handles heavy loads."},
                    {"category": "Car Hauler", "confidence": "medium", "reasoning": "Built for vehicles."},
                ],
                recommended_category="Equipment",
            )

    monkeypatch.setattr(graph, "_mind_llm", lambda: _FakeMindLLM())

    out = graph._mind_node(_state("recommend a trailer for heavy vehicles", category=None))

    pending = out["pending_category_suggestion"]
    assert pending["recommended_category"] == "Equipment"
    assert [item["category"] for item in pending["categories"]] == ["Equipment", "Car Hauler"]
    assert out["awaiting_slot"] == "category_suggestion_confirmation"
    assert "recommend one" in out["mind_decision"]["assistant_text"]


def test_generic_haul_use_stores_when_category_unknown(monkeypatch):
    _mock_field_updates(
        monkeypatch,
        metadata_filters_update={"length_ft": "12 ft"},
        slots_collected_update={"generic_haul_use": "debris"},
    )
    _mock_pre_generic_classifier(
        monkeypatch,
        action="ask_trailer_category",
        confidence="high",
        reason="Generic constrained request needs category.",
    )
    state = _state("I am looking for a 12ft trailer to haul some debris", category=None)
    state["mind_decision"]["action"] = "ask_next_question"

    out = graph._apply_mind_node(state)

    assert out["trailer_category"] is None
    assert out["metadata_filters_collected"]["length_ft"] == "12 ft"
    assert out["slots_collected"]["generic_haul_use"] == "debris"
    assert out["requested_non_metadata_features"] == []
    assert out["assistant_text"] == "What type of trailer are you looking for?"
    assert out["awaiting_slot"] == "generic_category_choice"


def test_generic_haul_use_maps_to_dump_haul_material(monkeypatch):
    _mock_field_updates(monkeypatch)
    state = _state("a dump trailer", category=None)
    state["awaiting_slot"] = "generic_category_choice"
    state["slots_collected"] = {"generic_haul_use": "debris"}
    state["mind_decision"]["action"] = "ask_next_question"

    out = graph._apply_mind_node(state)

    assert out["trailer_category"] == "Dump"
    assert out["slots_collected"]["haul_material"] == "debris"
    assert "generic_haul_use" not in out["slots_collected"]
    assert out["awaiting_slot"] == "haul_weight_lbs"


def test_office_trailer_asks_specific_clarification_not_generic_question(monkeypatch):
    _use_fallback_extractor(monkeypatch)

    planned = graph._mind_node(_state("I am looking for an office trailer", category=None))
    out = graph._apply_mind_node(planned)

    assert out["trailer_category"] is None
    assert out["assistant_text"] == "Will this be for fiber/telecom work specifically, or a more general office trailer?"
    assert out["awaiting_slot"] == "category_clarification"
    assert out["category_needs_clarification"] is True
    assert out["category_clarification_key"] == "office_trailer_use"
    assert out["slots_collected"] == {}


def test_office_trailer_blocks_direct_enclosed_guess_until_clarified(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    state = _state("I am loking for an office trailer", category=None)
    state["mind_decision"]["action"] = "respond"
    state["mind_decision"]["trailer_category"] = "Enclosed"

    out = graph._apply_mind_node(state)

    assert out["trailer_category"] is None
    assert out["assistant_text"] == "Will this be for fiber/telecom work specifically, or a more general office trailer?"
    assert out["awaiting_slot"] == "category_clarification"
    assert out["category_needs_clarification"] is True
    assert out["category_clarification_key"] == "office_trailer_use"


def test_office_trailer_clarification_answer_resolves_to_fiber(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    state = _state("fiber", category=None)
    state["awaiting_slot"] = "category_clarification"
    state["category_needs_clarification"] = True
    state["category_clarification_key"] = "office_trailer_use"
    state["messages"] = [
        {"role": "user", "content": "I am looking for an office trailer"},
        {
            "role": "assistant",
            "content": "Will this be for fiber/telecom work specifically, or a more general office trailer?",
        },
        {"role": "user", "content": "fiber"},
    ]

    out = graph._apply_mind_node(state)

    assert out["trailer_category"] == "Fiber"
    assert out["category_needs_clarification"] is False
    assert out["category_clarification_key"] is None
    assert out["awaiting_slot"] != "category_clarification"


def test_office_trailer_clarification_answer_resolves_to_enclosed(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    state = _state("general office", category=None)
    state["awaiting_slot"] = "category_clarification"
    state["category_needs_clarification"] = True
    state["category_clarification_key"] = "office_trailer_use"
    state["messages"] = [
        {"role": "user", "content": "I am looking for an office trailer"},
        {
            "role": "assistant",
            "content": "Will this be for fiber/telecom work specifically, or a more general office trailer?",
        },
        {"role": "user", "content": "general office"},
    ]

    out = graph._apply_mind_node(state)

    assert out["trailer_category"] == "Enclosed"
    assert out["category_needs_clarification"] is False
    assert out["category_clarification_key"] is None


def test_office_trailer_clarification_repeats_when_answer_unclear(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    state = _state("not sure yet", category=None)
    state["awaiting_slot"] = "category_clarification"
    state["category_needs_clarification"] = True
    state["category_clarification_key"] = "office_trailer_use"
    state["messages"] = [
        {"role": "user", "content": "I am looking for an office trailer"},
        {
            "role": "assistant",
            "content": "Will this be for fiber/telecom work specifically, or a more general office trailer?",
        },
        {"role": "user", "content": "not sure yet"},
    ]

    out = graph._apply_mind_node(state)

    assert out["trailer_category"] is None
    assert out["assistant_text"] == "No problem.\n\nWill this be for fiber/telecom work specifically, or a more general office trailer?"
    assert out["awaiting_slot"] == "category_clarification"
    assert out["category_needs_clarification"] is True


def test_office_trailer_clarification_llm_maps_office_work_to_enclosed(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class _FakeClarifier:
        def invoke(self, _messages):
            return graph.OfficeTrailerClarificationDecision(
                answered_clarification=True,
                resolved_category="Enclosed",
                confidence="high",
                reason="office_work_means_general_office",
            )

    monkeypatch.setattr(graph, "_office_trailer_clarification_llm", lambda: _FakeClarifier())
    state = _state("it'll be for office work", category=None)
    state["awaiting_slot"] = "category_clarification"
    state["category_needs_clarification"] = True
    state["category_clarification_key"] = "office_trailer_use"
    state["messages"] = [
        {"role": "user", "content": "I am looking for an office trailer"},
        {
            "role": "assistant",
            "content": "Will this be for fiber/telecom work specifically, or a more general office trailer?",
        },
        {"role": "user", "content": "it'll be for office work"},
    ]

    out = graph._apply_mind_node(state)

    assert out["trailer_category"] == "Enclosed"
    assert out["category_needs_clarification"] is False
    assert out["category_clarification_key"] is None


def test_office_trailer_clarification_llm_maps_fiber_splicing_to_fiber(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class _FakeClarifier:
        def invoke(self, _messages):
            return graph.OfficeTrailerClarificationDecision(
                answered_clarification=True,
                resolved_category="Fiber",
                confidence="high",
                reason="fiber_splicing_means_fiber",
            )

    monkeypatch.setattr(graph, "_office_trailer_clarification_llm", lambda: _FakeClarifier())
    state = _state("for fiber splicing", category=None)
    state["awaiting_slot"] = "category_clarification"
    state["category_needs_clarification"] = True
    state["category_clarification_key"] = "office_trailer_use"
    state["messages"] = [
        {"role": "user", "content": "I am looking for an office trailer"},
        {
            "role": "assistant",
            "content": "Will this be for fiber/telecom work specifically, or a more general office trailer?",
        },
        {"role": "user", "content": "for fiber splicing"},
    ]

    out = graph._apply_mind_node(state)

    assert out["trailer_category"] == "Fiber"
    assert out["category_needs_clarification"] is False
    assert out["category_clarification_key"] is None


def test_office_trailer_clarification_can_trigger_faq_and_preserve_question(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class _FakeClarifier:
        def invoke(self, _messages):
            return graph.OfficeTrailerClarificationDecision(
                answered_clarification=False,
                email_action="send_non_sales_faq_email",
                faq_category="contact_human",
                faq_summary="Customer asked how to contact TrailerPlace during office trailer clarification.",
                reply_to_user="You can reach our team at 979-532-1486.",
                confidence="high",
                reason="contact_question_during_clarification",
            )

    monkeypatch.setattr(graph, "_office_trailer_clarification_llm", lambda: _FakeClarifier())
    state = _state("how do I contact you guys?", category=None)
    state["awaiting_slot"] = "category_clarification"
    state["category_needs_clarification"] = True
    state["category_clarification_key"] = "office_trailer_use"
    state["messages"] = [
        {"role": "user", "content": "I am looking for an office trailer"},
        {
            "role": "assistant",
            "content": "Will this be for fiber/telecom work specifically, or a more general office trailer?",
        },
        {"role": "user", "content": "how do I contact you guys?"},
    ]

    out = graph._apply_mind_node(state)

    assert out["mind_decision"]["action"] == "send_non_sales_faq_email"
    assert out["mind_decision"]["faq_category"] == "contact_human"
    assert out["awaiting_slot"] == "category_clarification"
    assert out["category_needs_clarification"] is True
    assert out["category_clarification_key"] == "office_trailer_use"


def test_qna_email_action_uses_llm_wording_and_resumes_active_question(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    email_decision = graph.QuestionTurnDecision(
        answered_active_question=False,
        no_preference_for_active_question=True,
        email_action="send_non_sales_faq_email",
        faq_category="financing",
        reply_to_user="Our finance team can help with financing.",
        confidence="high",
    )
    monkeypatch.setattr(
        graph,
        "_adjudicate_active_question_turn",
        lambda **kwargs: email_decision,
    )
    monkeypatch.setattr(
        graph,
        "_reconcile_active_question_turn",
        lambda **kwargs: email_decision,
    )
    state = _state("I want to finance a trailer", category="Utility")
    state.update(
        {
            "awaiting_slot": "haul_weight_lbs",
            "slots_collected": {"haul_item": "furniture"},
            "customer_email": "ibrahim@esided.ai",
            "customer_full_name": "Ibrahim",
        }
    )

    applied = graph._apply_mind_node(state)

    assert applied["awaiting_slot"] == "haul_weight_lbs"
    assert "haul_weight_lbs" not in applied["slots_skipped"]
    monkeypatch.setattr(graph, "_persist_email_transcript_snapshot", lambda _state: None)
    monkeypatch.setattr(
        graph,
        "send_non_sales_faq_email",
        lambda **kwargs: {"status": "queued"},
    )

    def _compose(**kwargs):
        assert kwargs["next_question"] == "What's the rough total weight of your load?"
        return "Our finance team can help. About how heavy is everything you plan to haul?"

    monkeypatch.setattr(graph, "compose_email_tool_reply", _compose)
    completed = graph._faq_email_node(applied)

    assert completed["assistant_text"] == (
        "Our finance team can help. About how heavy is everything you plan to haul?"
    )


def test_one_time_automatic_search_per_category_cycle(monkeypatch):
    _mock_field_updates(monkeypatch)
    monkeypatch.setattr(
        graph,
        "_reconcile_category_transition",
        lambda **kwargs: graph.CategoryTransitionDecision(
            final_category="Dump",
            approve_category_change=False,
            explicit_category_switch=False,
            confidence="high",
            reason="Informational Utility question, not a category switch.",
        ),
    )

    informational = _state("What is the use case of Utility trailers?", category="Utility")
    informational.update(
        {
            "trailer_category": "Dump",
            "slots_collected": {
                "haul_material": "debris",
                "haul_weight_lbs": "5000 lbs",
            },
            "has_shown_search_results": True,
            "already_shown_listing_urls": ["https://example.test/first"],
        }
    )
    informational["mind_decision"].update(
        {
            "action": "ask_trailer_category",
            "assistant_text": "Utility trailers are versatile general-hauling trailers.",
        }
    )
    info_out = graph._apply_mind_node(informational)
    assert info_out["mind_decision"]["action"] == "respond"
    assert info_out["assistant_text"] == "Utility trailers are versatile general-hauling trailers."

    show_more = _state("show more results", category="Dump")
    show_more.update(
        {
            "slots_collected": {
                "haul_material": "debris",
                "haul_weight_lbs": "5000 lbs",
            },
            "has_shown_search_results": True,
            "already_shown_listing_urls": ["https://example.test/first"],
        }
    )
    show_more["mind_decision"]["action"] = "pinecone_search"
    more_out = graph._apply_mind_node(show_more)
    assert more_out["mind_decision"]["action"] == "pinecone_search"
    assert more_out["already_shown_listing_urls"] == ["https://example.test/first"]

    final_answer = _state("5000 lbs", category="Dump")
    final_answer.update(
        {
            "slots_collected": {"haul_material": "debris"},
            "awaiting_slot": "haul_weight_lbs",
            "has_shown_search_results": False,
        }
    )
    monkeypatch.setattr(
        graph,
        "_adjudicate_active_question_turn",
        lambda **kwargs: graph.QuestionTurnDecision(
            answered_active_question=True,
            active_slot_value="5000 lbs",
            confidence="high",
        ),
    )
    monkeypatch.setattr(
        graph,
        "_reconcile_active_question_turn",
        lambda **kwargs: graph.QuestionTurnDecision(
            answered_active_question=True,
            active_slot_value="5000 lbs",
            confidence="high",
        ),
    )
    first_results_out = graph._apply_mind_node(final_answer)
    assert first_results_out["mind_decision"]["action"] == "pinecone_search"


def test_initial_turn_completion_searches_despite_substantive_mind_response(monkeypatch):
    _mock_field_updates(
        monkeypatch,
        slots_collected_update={"trailer_length_ft": "20 ft"},
        metadata_filters_update={"length_ft": "20 ft"},
    )
    state = _state("I am looking for a 20ft livestock trailer", category="Livestock")
    state["mind_decision"].update(
        {
            "action": "respond",
            "assistant_text": "What is your preferred payload capacity?",
        }
    )

    out = graph._apply_mind_node(state)

    assert out["trailer_category"] == "Livestock"
    assert out["slots_collected"]["trailer_length_ft"] == "20 ft"
    assert out["mind_decision"]["action"] == "pinecone_search"


def test_generic_haul_use_maps_to_utility_haul_item(monkeypatch):
    _mock_field_updates(monkeypatch)
    state = _state("a utility trailer", category=None)
    state["awaiting_slot"] = "generic_category_choice"
    state["slots_collected"] = {"generic_haul_use": "mower"}
    state["mind_decision"]["action"] = "ask_next_question"

    out = graph._apply_mind_node(state)

    assert out["trailer_category"] == "Utility"
    assert out["slots_collected"]["haul_item"] == "mower"
    assert "generic_haul_use" not in out["slots_collected"]
    assert out["awaiting_slot"] == "haul_weight_lbs"


def test_generic_heavy_items_maps_to_flatbed_haul_item(monkeypatch):
    _mock_field_updates(monkeypatch)
    state = _state("a flatbed trailer", category=None)
    state["awaiting_slot"] = "generic_category_choice"
    state["slots_collected"] = {"generic_haul_use": "some heavy items"}
    state["mind_decision"]["action"] = "ask_next_question"

    out = graph._apply_mind_node(state)

    assert out["trailer_category"] == "Flatbed"
    assert out["slots_collected"]["haul_item"] == "some heavy items"
    assert "generic_haul_use" not in out["slots_collected"]
    assert out["awaiting_slot"] == "haul_weight_lbs"


def test_generic_haul_use_duplicate_feature_removed(monkeypatch):
    _mock_field_updates(
        monkeypatch,
        slots_collected_update={"generic_haul_use": "debris"},
        requested_non_metadata_features=["haul debris"],
    )
    state = _state("I need a trailer to haul debris", category=None)

    extraction = graph._extract_field_updates(
        state=state,
        category=None,
        awaiting_slot=None,
        apply_slot_updates=True,
    )

    assert extraction.slots_collected_update["generic_haul_use"] == "debris"
    assert extraction.requested_non_metadata_features == []


def test_contact_question_with_trailer_intent_uses_faq_tool_before_generic_category(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    monkeypatch.setattr(
        graph,
        "_classify_non_recommendation_turn",
        lambda **_kwargs: graph.NonRecommendationTurnDecision(
            turn_type="contact_or_store_info",
            action="send_non_sales_faq_email",
            faq_category="contact_human",
            faq_summary="Customer asked how to contact TrailerPlace.",
            assistant_text="You can reach our team at 979-532-1486.",
            should_store_freeform_fields=True,
            confidence="high",
            reason="contact_question_with_trailer_intent",
        ),
    )
    state = _state("I want to buy a trailer but I want to know first how to contact you guys", category=None)

    out = graph._apply_mind_node(state)

    assert out["mind_decision"]["action"] == "send_non_sales_faq_email"
    assert out["mind_decision"]["faq_category"] == "contact_human"
    assert out["awaiting_slot"] is None
    assert out["assistant_text"] == "You can reach our team at 979-532-1486."


def test_seller_contact_request_uses_escalation_before_generic_category(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    monkeypatch.setattr(
        graph,
        "_classify_non_recommendation_turn",
        lambda **_kwargs: graph.NonRecommendationTurnDecision(
            turn_type="unsupported_business_action",
            action="send_escalation_alert_email",
            escalation_summary="Customer wants to sell trailers to TrailerPlace.",
            assistant_text="I can send that request to our team.",
            should_store_freeform_fields=False,
            confidence="high",
            reason="seller_request",
        ),
    )
    state = _state("I have multiple trailers and want to sell them to you; contact me", category=None)

    out = graph._apply_mind_node(state)

    assert out["mind_decision"]["action"] == "send_escalation_alert_email"
    assert out["mind_decision"]["escalation_summary"] == "Customer wants to sell trailers to TrailerPlace."
    assert out["awaiting_slot"] is None


def test_mixed_contact_question_preserves_explicit_trailer_metadata(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    monkeypatch.setattr(
        graph,
        "_classify_non_recommendation_turn",
        lambda **_kwargs: graph.NonRecommendationTurnDecision(
            turn_type="contact_or_store_info",
            action="send_non_sales_faq_email",
            faq_category="contact_human",
            faq_summary="Customer asked how to contact TrailerPlace.",
            should_store_freeform_fields=True,
            confidence="high",
            reason="mixed_contact_and_size",
        ),
    )
    state = _state("I need a 6x12 trailer and want to know how to contact you guys", category=None)

    out = graph._apply_mind_node(state)

    assert out["mind_decision"]["action"] == "send_non_sales_faq_email"
    assert out["metadata_filters_collected"]["width_ft"] == "6"
    assert out["metadata_filters_collected"]["length_ft"] == "12"
    assert out["awaiting_slot"] is None


def test_llm_field_extraction_normalizes_compact_size_order(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class _FakeExtractorLLM:
        def invoke(self, _messages):
            return graph.FieldExtractionAdjudicationDecision(
                metadata_filters_update={"length_ft": "6", "width_ft": "12"},
                slots_collected_update={},
                requested_non_metadata_features=[],
                confidence="high",
            )

    monkeypatch.setattr(graph, "_field_extraction_adjudicator_llm", lambda: _FakeExtractorLLM())

    decision = graph._extract_field_updates(
        state=_state("I am looking for a 6x12 livestock trailer", category="Livestock"),
        category="Livestock",
        awaiting_slot=None,
        apply_slot_updates=True,
    )

    assert decision.metadata_filters_update["width_ft"] == "6"
    assert decision.metadata_filters_update["length_ft"] == "12"


def test_field_extraction_decision_normalizes_empty_rejected_candidates_object():
    decision = graph.FieldExtractionAdjudicationDecision.model_validate(
        {"rejected_candidates": {}}
    )

    assert decision.rejected_candidates == []


def test_mind_decision_normalizes_string_category_recommendations():
    decision = graph.MindDecision.model_validate(
        {"category_recommendations": ["Utility", "Dump", "Equipment"]}
    )

    assert [item["category"] for item in decision.category_recommendations] == [
        "Utility", "Dump", "Equipment"
    ]


def test_llm_field_extraction_accepts_natural_14_footer(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class _FakeExtractorLLM:
        def invoke(self, _messages):
            return graph.FieldExtractionAdjudicationDecision(
                metadata_filters_update={
                    "length_ft": "14 footer",
                    "width_ft": "72 inches",
                    "height_ft": "100 cm",
                    "payload_lbs": "1000 kg",
                },
                confidence="high",
            )

    monkeypatch.setattr(graph, "_field_extraction_adjudicator_llm", lambda: _FakeExtractorLLM())

    decision = graph._extract_field_updates(
        state=_state("I want a 14 footer", category=None),
        category=None,
        awaiting_slot=None,
        apply_slot_updates=True,
    )

    assert decision.metadata_filters_update == {
        "length_ft": "14 ft",
        "width_ft": "6 ft",
        "height_ft": "3.28084 ft",
        "payload_lbs": "2204.62 lbs",
    }


def test_category_recommendation_preserves_length_and_bridge_cannot_copy_question(monkeypatch):
    monkeypatch.setattr(
        graph,
        "_extract_field_updates",
        lambda **_kwargs: graph.FieldExtractionAdjudicationDecision(
            metadata_filters_update={"length_ft": "14 ft"},
            confidence="high",
        ),
    )
    state = _state("I am looking for a 14 footer", category=None)
    state["pending_category_suggestion"] = {
        "categories": [{"category": "Livestock"}],
        "status": "awaiting_choice_or_recommendation",
    }

    out = graph._apply_mind_node(state)

    assert out["metadata_filters_collected"] == {"length_ft": "14 ft"}

    captured = {}

    class _BridgeLLM:
        def invoke(self, messages):
            captured["human"] = messages[1].content
            return type("_Response", (), {"content": "No problem—I’ll keep helping."})()

    monkeypatch.setattr(service, "_contact_prompt_bridge_llm", lambda: _BridgeLLM())
    service._contact_prompt_bridge_text(
        action="decline_contact_details",
        latest_message="no",
        saved_request="I am looking for a 14 footer",
    )

    assert "Trailer/search response" not in captured["human"]
    assert "What type of trailer" not in captured["human"]


def test_generic_category_no_preference_reuses_length_and_asks_payload(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    _mock_preference_classifier(
        monkeypatch,
        has_no_preference=True,
        target_slots=["generic_category_choice"],
        reason="User has no category preference.",
        confidence="high",
    )
    state = _state("any type", category=None)
    state["awaiting_slot"] = "generic_category_choice"
    state["metadata_filters_collected"] = {"width_ft": "6", "length_ft": "12"}
    state["messages"] = [
        {"role": "assistant", "content": "What type of trailer are you looking for?"},
        {"role": "user", "content": "any type"},
    ]

    out = graph._apply_mind_node(state)

    assert out["trailer_category"] is None
    assert out["metadata_filters_collected"]["length_ft"] == "12"
    assert out["awaiting_slot"] == "haul_weight_lbs"
    assert out["assistant_text"] == "What payload or weight capacity do you need?"
    assert "generic_category_choice" in out["slots_skipped"]


def test_generic_category_no_preference_without_length_asks_length_first(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    _mock_preference_classifier(
        monkeypatch,
        has_no_preference=True,
        target_slots=["generic_category_choice"],
        reason="User has no category preference.",
        confidence="high",
    )
    state = _state("no preference", category=None)
    state["awaiting_slot"] = "generic_category_choice"
    state["messages"] = [
        {"role": "assistant", "content": "What type of trailer are you looking for?"},
        {"role": "user", "content": "no preference"},
    ]

    out = graph._apply_mind_node(state)

    assert out["awaiting_slot"] == "trailer_length_ft"
    assert out["assistant_text"] == "What trailer length would you prefer?"


def test_generic_no_category_searches_when_length_and_payload_known(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    state = _state("3000 pounds payload", category=None)
    state["slots_skipped"] = ["generic_category_choice"]
    state["metadata_filters_collected"] = {"length_ft": "12"}

    out = graph._apply_mind_node(state)

    assert out["trailer_category"] is None
    assert out["metadata_filters_collected"]["payload_lbs"] == "3000 pounds"
    assert out["mind_decision"]["action"] == "pinecone_search"


def test_generic_category_followup_preserves_metadata_and_reasks_category(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    _mock_pre_generic_classifier(
        monkeypatch,
        action="ask_trailer_category",
        confidence="high",
        reason="Still needs category after preserving metadata.",
    )
    state = _state("it must be a bumper pull", category=None)
    state["awaiting_slot"] = "generic_category_choice"

    out = graph._apply_mind_node(state)

    assert out["metadata_filters_collected"]["hitch_type"] == "Bumper Pull"
    assert out["awaiting_slot"] == "generic_category_choice"
    assert out["assistant_text"] == "What type of trailer are you looking for?"


def test_categoryless_followups_update_supported_metadata(monkeypatch):
    cases = [
        ("under 10000", "max_price", "10000"),
        ("make it 14 ft", "length_ft", "14 ft"),
        ("6 feet wide", "width_ft", "6 feet"),
        ("around 3000 pounds payload", "payload_lbs", "3000 pounds"),
        ("black", "color", "black"),
        ("Diamond C", "make", "Diamond C"),
    ]
    for message, key, expected in cases:
        _use_fallback_extractor(monkeypatch)
        state = _state(message, category=None)
        state["awaiting_slot"] = "generic_category_choice"

        out = graph._apply_mind_node(state)

        assert out["metadata_filters_collected"][key] == expected


def test_broad_catalogue_request_redirects_to_website(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    state = _state("list me all your products", category=None)
    state["mind_decision"]["action"] = "pinecone_search"

    out = graph._apply_mind_node(state)

    assert out["mind_decision"]["action"] == "respond"
    assert "[TrailerPlace](https://trailerplace.com)" in out["assistant_text"]
    assert out["pending_questions"] == []


def test_catalogue_word_redirects_to_website(monkeypatch):
    _use_fallback_extractor(monkeypatch)

    out = graph._apply_mind_node(_state("I want to see a catalogue", category=None))

    assert out["mind_decision"]["action"] == "respond"
    assert "[TrailerPlace](https://trailerplace.com)" in out["assistant_text"]


def test_broad_catalogue_request_with_budget_does_not_redirect(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    _mock_pre_generic_classifier(
        monkeypatch,
        action="ask_trailer_category",
        confidence="high",
        reason="Budget narrows inventory but category is still needed.",
    )
    state = _state("list all your products under 10000", category=None)
    state["mind_decision"]["action"] = "pinecone_search"

    out = graph._apply_mind_node(state)

    assert "[TrailerPlace](https://trailerplace.com)" not in out["assistant_text"]
    assert out["metadata_filters_collected"]["max_price"] == "10000"
    assert out["assistant_text"] == "What type of trailer are you looking for?"
    assert out["awaiting_slot"] == "generic_category_choice"


def test_browse_after_generic_type_question_redirects_when_unconstrained(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    state = _state("I just want to browse the inventory", category=None)
    state["messages"] = [
        {"role": "assistant", "content": "What type of trailer are you looking for?"},
        {"role": "user", "content": "I just want to browse the inventory"},
    ]

    out = graph._apply_mind_node(state)

    assert out["mind_decision"]["action"] == "respond"
    assert "[TrailerPlace](https://trailerplace.com)" in out["assistant_text"]
    assert out["awaiting_slot"] is None


def test_show_more_after_results_still_searches(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    state = _state("show me more options", category="Utility")
    state["has_shown_search_results"] = True
    state["last_listings"] = [{"title": "Trailer A", "url": "https://example.com/a"}]
    state["slots_collected"] = {"haul_item": "mower", "haul_weight_lbs": "900 lbs"}
    state["mind_decision"]["action"] = "pinecone_search"

    out = graph._apply_mind_node(state)

    assert out["mind_decision"]["action"] == "pinecone_search"
    assert "[TrailerPlace](https://trailerplace.com)" not in out["assistant_text"]


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


def test_classifier_trailer_description_does_not_fill_utility_haul_item(monkeypatch):
    _mock_extractor(monkeypatch, length_ft="12")
    _mock_haul_classifier(
        monkeypatch,
        is_lightweight_utility_load=True,
        needs_width_question=False,
        matched_item="12 ft trailer",
        confidence="high",
    )

    out = graph._apply_mind_node(_state("I am looking for a 12 ft utility trailer", category="Utility"))

    assert out["mind_decision"]["action"] == "respond"
    assert out["assistant_text"] == "What will you be hauling on the utility trailer?"
    assert out["awaiting_slot"] == "haul_item"
    assert "haul_item" not in out["slots_collected"]
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

    assert out["awaiting_slot"] == "hitch_type"
    assert out["assistant_text"] == "Do you prefer a bumper pull or gooseneck hitch?"
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
    assert out["awaiting_slot"] == "hitch_type"
    assert out["assistant_text"] == "Do you prefer a bumper pull or gooseneck hitch?"
    assert out["mind_decision"]["action"] == "respond"


def test_equipment_requires_hitch_question_before_search(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    _mock_haul_classifier(
        monkeypatch,
        matched_item="tractor",
        reason="Standard equipment.",
        confidence="high",
    )
    state = _state("I need an equipment trailer", category="Equipment")
    state["slots_collected"] = {
        "haul_item": "tractor",
        "haul_weight_lbs": "3000 lbs",
        "haul_length_ft": "12 ft",
    }
    state["metadata_filters_collected"] = {
        "payload_lbs": "3000 lbs",
        "length_ft": "12 ft",
    }

    out = graph._apply_mind_node(state)

    assert out["awaiting_slot"] == "hitch_type"
    assert out["assistant_text"] == "Do you prefer a bumper pull or gooseneck hitch?"
    assert out["mind_decision"]["action"] == "respond"


def test_flatbed_does_not_add_dynamic_width_question_and_defaults_width(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    _mock_haul_classifier(
        monkeypatch,
        needs_width_question=True,
        matched_item="tractor",
        reason="Heavy-duty equipment.",
        confidence="high",
    )
    state = _state("ready to search", category="Flatbed")
    state["slots_collected"] = {
        "haul_item": "tractor",
        "haul_weight_lbs": "7000 lbs",
    }

    out = graph._apply_mind_node(state)

    assert out["metadata_filters_collected"]["width_ft"] == "8 ft"
    assert "item_or_trailer_width_ft" not in out["slots_collected"]
    assert all(q.get("slot") != "item_or_trailer_width_ft" for q in out["pending_questions"])
    assert out["mind_decision"]["action"] == "pinecone_search"


def test_flatbed_default_width_is_metadata_only(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    state = _state("ready to search", category="Flatbed")
    state["slots_collected"] = {
        "haul_item": "hay",
        "haul_weight_lbs": "5000 lbs",
    }

    out = graph._apply_mind_node(state)

    assert out["metadata_filters_collected"]["width_ft"] == "8 ft"
    assert "item_or_trailer_width_ft" not in out["slots_collected"]
    assert "width_ft" not in out["slots_collected"]
    assert out["mind_decision"]["action"] == "pinecone_search"


def test_flatbed_explicit_width_overrides_default_width(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    state = _state("make it 7 ft wide", category="Flatbed")
    state["slots_collected"] = {
        "haul_item": "hay",
        "haul_weight_lbs": "5000 lbs",
    }

    out = graph._apply_mind_node(state)

    assert out["metadata_filters_collected"]["width_ft"] == "7 ft"
    assert "item_or_trailer_width_ft" not in out["slots_collected"]
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


def test_dump_does_not_add_dynamic_width_question(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    _mock_haul_classifier(
        monkeypatch,
        needs_width_question=True,
        matched_item="skid steer",
        reason="Heavy-duty equipment.",
        confidence="high",
    )
    state = _state("I need a dump trailer for a skid steer", category="Dump")
    state["slots_collected"] = {
        "payload_capacity": "7000 lbs",
        "bin_size": "12 ft",
    }

    out = graph._apply_mind_node(state)

    assert "item_or_trailer_width_ft" not in out["slots_collected"]
    assert all(q.get("slot") != "item_or_trailer_width_ft" for q in out["pending_questions"])
    assert out["awaiting_slot"] == "haul_material"
    assert out["assistant_text"] == "What material will you be hauling (dirt, gravel, debris, etc.)?"
    assert out["mind_decision"]["action"] == "respond"


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


def test_hitch_filter_is_not_overwritten_without_explicit_latest_message(monkeypatch):
    _mock_extractor(monkeypatch, payload_lbs="1600", hitch_type="gooseneck")
    state = _state("1600 lbs", category="Equipment")
    state["slots_collected"] = {
        "haul_item": "a car",
        "haul_length_ft": "12",
        "item_or_trailer_width_ft": "6",
        "hitch_type": "Bumper Pull",
    }
    state["metadata_filters_collected"] = {
        "length_ft": "12",
        "width_ft": "6",
        "hitch_type": "Bumper Pull",
    }

    out = graph._apply_mind_node(state)

    assert out["metadata_filters_collected"]["hitch_type"] == "Bumper Pull"
    assert out["metadata_filters_collected"]["payload_lbs"] == "1600"


def test_hitch_filter_changes_when_latest_message_is_explicit(monkeypatch):
    _mock_extractor(monkeypatch, hitch_type="gooseneck")
    state = _state("make it a gooseneck hitch", category="Equipment")
    state["metadata_filters_collected"] = {"hitch_type": "Bumper Pull"}

    out = graph._apply_mind_node(state)

    assert out["metadata_filters_collected"]["hitch_type"] == "Gooseneck"


def test_payload_filter_is_not_overwritten_without_weight_evidence(monkeypatch):
    _mock_extractor(monkeypatch, payload_lbs="9000")
    state = _state("make it black", category="Equipment")
    state["slots_collected"] = {
        "haul_item": "tractor",
        "haul_weight_lbs": "3000 lbs",
        "haul_length_ft": "12 ft",
    }
    state["metadata_filters_collected"] = {"payload_lbs": "3000 lbs"}

    out = graph._apply_mind_node(state)

    assert out["metadata_filters_collected"]["payload_lbs"] == "3000 lbs"


def test_width_filter_is_not_overwritten_without_width_evidence(monkeypatch):
    _mock_extractor(monkeypatch, width_ft="8")
    state = _state("1600 lbs", category="Equipment")
    state["slots_collected"] = {
        "haul_item": "tractor",
        "haul_length_ft": "12 ft",
    }
    state["metadata_filters_collected"] = {"width_ft": "6"}

    out = graph._apply_mind_node(state)

    assert out["metadata_filters_collected"]["width_ft"] == "6"


def test_color_filter_is_not_overwritten_without_color_evidence(monkeypatch):
    _mock_extractor(monkeypatch, color="red")
    state = _state("1600 lbs", category="Equipment")
    state["metadata_filters_collected"] = {"color": "black"}

    out = graph._apply_mind_node(state)

    assert out["metadata_filters_collected"]["color"] == "black"


def test_bare_length_answer_is_accepted_for_active_length_question(monkeypatch):
    _mock_extractor(monkeypatch, length_ft="12 ft")
    state = _state("12 ft", category="Equipment")
    state["slots_collected"] = {
        "haul_item": "tractor",
        "haul_weight_lbs": "3000 lbs",
    }
    state["awaiting_slot"] = "haul_length_ft"

    out = graph._apply_mind_node(state)

    assert out["slots_collected"]["haul_length_ft"] == "12 ft"
    assert out["metadata_filters_collected"]["length_ft"] == "12 ft"


def test_cargo_size_current_length_overrides_reconciler_and_searches(monkeypatch):
    _mock_field_updates(
        monkeypatch,
        slots_collected_update={"cargo_size": "18 ft"},
        metadata_filters_update={"length_ft": "18 ft", "width_ft": "8 ft"},
    )
    monkeypatch.setattr(
        graph,
        "_adjudicate_active_question_turn",
        lambda **kwargs: graph.QuestionTurnDecision(
            answered_active_question=False,
            reply_to_user="What's the rough cargo size?",
            rephrased_question="What dimensions do you need?",
            retry_slot="cargo_size",
            confidence="low",
        ),
    )
    monkeypatch.setattr(
        graph,
        "_reconcile_active_question_turn",
        lambda **kwargs: graph.QuestionTurnDecision(
            answered_active_question=False,
            reply_to_user="What's the rough cargo size?",
            rephrased_question="What dimensions do you need?",
            retry_slot="cargo_size",
            confidence="low",
        ),
    )
    state = _state(
        "About 18 by 8 feet; height is not particularly important.",
        category="Enclosed",
    )
    state["slots_collected"] = {"use_case": "general cargo"}
    state["awaiting_slot"] = "cargo_size"

    out = graph._apply_mind_node(state)

    assert out["slots_collected"]["cargo_size"] == "18 ft × 8 ft"
    assert out["metadata_filters_collected"]["length_ft"] == "18 ft"
    assert out["metadata_filters_collected"]["width_ft"] == "8 ft"
    assert out["awaiting_slot"] is None
    assert out["mind_decision"]["action"] == "pinecone_search"


def test_mixed_active_answer_and_explicit_hitch_does_not_invent_weight(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    state = _state("a car. The trailer should be bumper pull", category="Equipment")
    state["awaiting_slot"] = "haul_item"

    out = graph._apply_mind_node(state)

    assert out["slots_collected"]["haul_item"] == "a car"
    assert out["metadata_filters_collected"]["hitch_type"] == "Bumper Pull"
    assert "payload_lbs" not in out["metadata_filters_collected"]
    assert out["awaiting_slot"] == "haul_weight_lbs"


def test_active_counterquestion_replies_and_repeats_same_question(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    monkeypatch.setattr(
        graph,
        "_adjudicate_active_question_turn",
        lambda **kwargs: graph.QuestionTurnDecision(
            answered_active_question=True,
            active_slot_value="financing",
            no_preference_for_active_question=True,
            counter_question_topic="other",
            reply_to_user=(
                "We do offer financing options. To narrow down the right dump trailer, "
                "What kind of material do you expect to haul?"
            ),
            rephrased_question="What kind of material do you expect to haul?",
            retry_slot="haul_material",
            confidence="high",
            reason="counter_question_not_answer",
        ),
    )
    state = _state("do you offer financing?", category="Dump")
    state["awaiting_slot"] = "haul_material"
    state["messages"] = [
        {"role": "assistant", "content": "What material will you be hauling (dirt, gravel, debris, etc.)?"},
        {"role": "user", "content": "do you offer financing?"},
    ]

    out = graph._apply_mind_node(state)

    assert out["awaiting_slot"] == "haul_material"
    assert out["assistant_text"] == (
        "We do offer financing options. To narrow down the right dump trailer, "
        "What kind of material do you expect to haul?"
    )
    assert out["mind_decision"]["action"] == "respond"
    assert "haul_material" not in out["slots_skipped"]
    assert out["active_question_attempts"]["haul_material"] == 1

    state["active_question_attempts"] = out["active_question_attempts"]
    out = graph._apply_mind_node(state)

    assert out["repeated_unanswered_question_escalation"] is True
    assert "haul_material" in out["slots_skipped"]
    assert out["awaiting_slot"] == "haul_weight_lbs"
    assert "What's the rough haul weight per load?" in out["assistant_text"]
    assert "What kind of material do you expect to haul?" not in out["assistant_text"]


def test_active_question_immediate_search_bypasses_current_question(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    monkeypatch.setattr(
        graph,
        "_adjudicate_active_question_turn",
        lambda **kwargs: graph.QuestionTurnDecision(
            search_now_requested=True,
            confidence="high",
            reason="user_requested_results",
        ),
    )
    state = _state("Just show me the trailers", category="Dump")
    state["awaiting_slot"] = "haul_material"

    out = graph._apply_mind_node(state)

    assert out["mind_decision"]["action"] == "pinecone_search"
    assert out["awaiting_slot"] is None
    assert "haul_material" not in out["slots_collected"]


def test_active_extraction_overrides_wrong_adjudicator_and_asks_next_question(monkeypatch):
    _mock_field_updates(
        monkeypatch,
        slots_collected_update={"haul_material": "random things"},
    )
    monkeypatch.setattr(
        graph,
        "_adjudicate_active_question_turn",
        lambda **kwargs: graph.QuestionTurnDecision(
            answered_active_question=False,
            search_now_requested=True,
            skip_remaining_questions=True,
            reply_to_user="What material will you be hauling?",
            rephrased_question="What material will you be hauling?",
            retry_slot="haul_material",
            confidence="low",
        ),
    )
    monkeypatch.setattr(
        graph,
        "_reconcile_active_question_turn",
        lambda **kwargs: graph.QuestionTurnDecision(
            answered_active_question=True,
            active_slot_value="random things",
            search_now_requested=False,
            skip_remaining_questions=False,
            confidence="high",
            reason="llm_reconciled_free_text_answer",
        ),
    )
    state = _state("random things", category="Dump")
    state["awaiting_slot"] = "haul_material"

    out = graph._apply_mind_node(state)

    assert out["slots_collected"]["haul_material"] == "random things"
    assert out["awaiting_slot"] == "haul_weight_lbs"
    assert out["mind_decision"]["action"] == "respond"
    assert out["assistant_text"] == "What's the rough haul weight per load?"


def test_active_haul_answer_reconciles_before_category_change(monkeypatch):
    _mock_field_updates(
        monkeypatch,
        slots_collected_update={"haul_item": "assorted equipment"},
    )
    monkeypatch.setattr(
        graph,
        "_reconcile_category_transition",
        lambda **kwargs: graph.CategoryTransitionDecision(
            final_category="Utility",
            approve_category_change=False,
            explicit_category_switch=False,
            confidence="high",
            reason="Equipment describes cargo, not a trailer-category switch.",
        ),
    )
    monkeypatch.setattr(
        graph,
        "_adjudicate_active_question_turn",
        lambda **kwargs: graph.QuestionTurnDecision(
            answered_active_question=True,
            active_slot_value="assorted equipment",
            confidence="high",
        ),
    )
    monkeypatch.setattr(
        graph,
        "_reconcile_active_question_turn",
        lambda **kwargs: graph.QuestionTurnDecision(
            answered_active_question=True,
            active_slot_value="assorted equipment",
            confidence="high",
        ),
    )
    state = _state("Mostly assorted equipment.", category="Utility")
    state["trailer_category"] = "Utility"
    state["awaiting_slot"] = "haul_item"
    state["mind_decision"]["trailer_category"] = "Equipment"

    out = graph._apply_mind_node(state)

    assert out["trailer_category"] == "Utility"
    assert out["slots_collected"]["haul_item"] == "assorted equipment"
    assert out["awaiting_slot"] == "haul_weight_lbs"


def test_explicit_category_switch_is_applied_after_reconciliation(monkeypatch):
    _mock_field_updates(monkeypatch)
    monkeypatch.setattr(
        graph,
        "_reconcile_category_transition",
        lambda **kwargs: graph.CategoryTransitionDecision(
            final_category="Equipment",
            approve_category_change=True,
            explicit_category_switch=True,
            confidence="high",
            reason="The user explicitly requested Equipment instead.",
        ),
    )
    state = _state("Actually switch me to an Equipment trailer instead.", category="Utility")
    state["trailer_category"] = "Utility"
    state["awaiting_slot"] = "haul_item"
    state["mind_decision"]["trailer_category"] = "Equipment"

    out = graph._apply_mind_node(state)

    assert out["trailer_category"] == "Equipment"


def test_category_switch_discards_stale_mind_question(monkeypatch):
    _mock_field_updates(
        monkeypatch,
        slots_collected_update={"vehicle_type": "a car"},
    )
    _mock_haul_classifier(monkeypatch)
    monkeypatch.setattr(
        graph,
        "_reconcile_category_transition",
        lambda **kwargs: graph.CategoryTransitionDecision(
            final_category="Car Hauler",
            approve_category_change=True,
            explicit_category_switch=True,
            confidence="high",
            reason="Explicit switch.",
        ),
    )
    state = _state("I am looking for a car hauler as well", category="Car Hauler")
    state["trailer_category"] = "Equipment"
    state["slots_collected"] = {"haul_item": "skid steer"}
    state["has_shown_search_results"] = True
    state["mind_decision"]["assistant_text"] = (
        "What is the rough total weight of the skid steer?"
    )

    out = graph._apply_mind_node(state)

    assert out["trailer_category"] == "Car Hauler"
    assert out["slots_collected"] == {"vehicle_type": "a car"}
    assert out["awaiting_slot"] == "haul_weight_lbs"
    assert out["assistant_text"] == "What's the approximate weight of the vehicle?"
    assert "skid steer" not in out["assistant_text"].lower()


def test_either_hitch_is_advisory_no_preference_to_reconciler(monkeypatch):
    _mock_field_updates(monkeypatch)
    monkeypatch.setattr(
        graph,
        "_adjudicate_active_question_turn",
        lambda **kwargs: graph.QuestionTurnDecision(
            answered_active_question=True,
            active_slot_value="Bumper Pull",
            confidence="medium",
        ),
    )
    monkeypatch.setattr(
        graph,
        "classify_no_preference",
        lambda **kwargs: graph.PreferenceNullDecision(
            has_no_preference=True,
            target_slots=["hitch_type"],
            confidence="high",
            reason="Either allowed hitch is acceptable.",
        ),
    )

    def _reconcile(**kwargs):
        assert kwargs["no_preference_decision"].has_no_preference is True
        return graph.QuestionTurnDecision(
            no_preference_for_active_question=True,
            confidence="high",
            reason="Either A or B means no preference.",
        )

    monkeypatch.setattr(graph, "_reconcile_active_question_turn", _reconcile)
    state = _state("Either bumper pull or gooseneck is fine.", category="Equipment")
    state["trailer_category"] = "Equipment"
    state["slots_collected"] = {
        "haul_item": "skid steer",
        "haul_weight_lbs": "5000 lbs",
        "haul_length_ft": "16 ft",
    }
    state["awaiting_slot"] = "hitch_type"

    out = graph._apply_mind_node(state)

    assert "hitch_type" in out["slots_skipped"]
    assert "hitch_type" not in out["slots_collected"]
    assert "hitch_type" not in out["metadata_filters_collected"]


def test_make_candidate_requires_llm_verification(monkeypatch):
    monkeypatch.setattr(
        graph,
        "resolve_make_from_text",
        lambda *args, **kwargs: make_resolver.MakeResolution(
            make="Cargo Craft",
            confidence="high",
            match_type="partial",
        ),
    )
    monkeypatch.setattr(
        graph,
        "_verify_make_candidate",
        lambda **kwargs: graph.MakeVerificationDecision(
            approve_make=False,
            verified_make=None,
            confidence="high",
            reason="General cargo is not an explicit manufacturer request.",
        ),
    )
    metadata = {}

    category, options, question = graph._apply_make_resolution(
        latest_message="general cargo and occasional work use",
        category="Enclosed",
        metadata_filters=metadata,
        recent_messages=[],
    )

    assert category == "Enclosed"
    assert options == []
    assert question is None
    assert "make" not in metadata


def test_explicit_make_candidate_is_persisted_after_llm_verification(monkeypatch):
    monkeypatch.setattr(
        graph,
        "resolve_make_from_text",
        lambda *args, **kwargs: make_resolver.MakeResolution(
            make="Cargo Craft",
            confidence="high",
            match_type="exact",
        ),
    )
    monkeypatch.setattr(
        graph,
        "_verify_make_candidate",
        lambda **kwargs: graph.MakeVerificationDecision(
            approve_make=True,
            verified_make="Cargo Craft",
            explicit_evidence="I want Cargo Craft",
            confidence="high",
        ),
    )
    metadata = {}

    category, _, _ = graph._apply_make_resolution(
        latest_message="I want a Cargo Craft enclosed trailer",
        category="Enclosed",
        metadata_filters=metadata,
        recent_messages=[],
    )

    assert category == "Enclosed"
    assert metadata["make"] == "Cargo Craft"


def test_active_question_skip_remaining_marks_questions_skipped_and_searches(monkeypatch):
    _use_fallback_extractor(monkeypatch)
    monkeypatch.setattr(
        graph,
        "_adjudicate_active_question_turn",
        lambda **kwargs: graph.QuestionTurnDecision(
            search_now_requested=True,
            skip_remaining_questions=True,
            email_action="send_escalation_alert_email",
            escalation_summary="Incorrect escalation proposal.",
            confidence="high",
            reason="user_refused_more_questions",
        ),
    )
    state = _state("I don't want more questions; show me what you have", category="Dump")
    state["awaiting_slot"] = "haul_material"
    state["mind_decision"]["action"] = "send_escalation_alert_email"

    out = graph._apply_mind_node(state)

    assert out["mind_decision"]["action"] == "pinecone_search"
    assert out["awaiting_slot"] is None
    assert {"haul_material", "haul_weight_lbs"}.issubset(set(out["slots_skipped"]))
    assert out["pending_questions"] == []


def test_active_question_followup_prefers_awaiting_slot_over_next_pending_question():
    state = {
        "trailer_category": "Equipment",
        "awaiting_slot": "haul_weight_lbs",
        "pending_questions": [
            {
                "slot": "haul_length_ft",
                "question": "About how long is the load (or what deck length do you need)?",
            }
        ],
    }

    assert graph._active_question_followup(state) == "What's the rough total weight of the load?"


def test_retry_fallback_preserves_direct_counterquestion_answer(monkeypatch):
    class _InvalidRepair:
        def invoke(self, _messages):
            return graph.QuestionTurnDecision()

    monkeypatch.setattr(graph, "_question_turn_adjudicator_llm", lambda: _InvalidRepair())
    result = graph._repair_question_retry(
        state={"messages": []},
        decision=graph.QuestionTurnDecision(
            counter_question_topic="hitch_types",
            reply_to_user="What's the rough total weight of the load?",
        ),
        active_slot="haul_weight_lbs",
        active_question="What's the rough total weight of the load?",
        latest_message="Which hitch types do you have?",
        direct_answer_fallback=(
            "Available hitch configurations include Bumper Pull and Gooseneck. "
            "Do you have a preference?"
        ),
    )

    assert result.reply_to_user == (
        "Available hitch configurations include Bumper Pull and Gooseneck. "
        "What's the rough total weight of the load?"
    )


def test_retry_repair_preserves_faq_email_action(monkeypatch):
    class _ValidRepair:
        def invoke(self, _messages):
            return graph.QuestionTurnDecision(
                reply_to_user=(
                    "Yes, financing is available. "
                    "What material will you be hauling?"
                ),
                rephrased_question="What material will you be hauling?",
                retry_slot="haul_material",
            )

    monkeypatch.setattr(graph, "_question_turn_adjudicator_llm", lambda: _ValidRepair())
    result = graph._repair_question_retry(
        state={"messages": []},
        decision=graph.QuestionTurnDecision(
            email_action="send_non_sales_faq_email",
            faq_category="financing",
            faq_summary="Customer asked about financing.",
            reply_to_user=(
                "Yes, financing is available. "
                "What material will you be hauling?"
            ),
        ),
        active_slot="haul_material",
        active_question="What material will you be hauling?",
        latest_message="Do you offer financing?",
    )

    assert result.email_action == "send_non_sales_faq_email"
    assert result.faq_category == "financing"
    assert result.faq_summary == "Customer asked about financing."
    assert result.retry_slot == "haul_material"


def test_retry_validation_accepts_semantically_matching_different_wording():
    decision = graph.QuestionTurnDecision(
        reply_to_user=(
            "TrailerPlace offers Bumper Pull and Gooseneck configurations. "
            "Now, could you let me know the rough total weight of the load?"
        ),
        rephrased_question="What is the approximate weight of the load you plan to haul?",
        retry_slot="haul_weight_lbs",
    )

    assert graph._valid_question_retry(decision, "haul_weight_lbs") is True


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

    assert out["slots_collected"]["base_category"] == "utility"
    assert out["metadata_filters_collected"]["subcategory"] == "Utility"
    assert out["awaiting_slot"] == "payload_need"


def test_aluminum_category_and_base_category_guardrails():
    implicit = {
        "trailer_category": "Utility",
        "category_resolution_kind": "explicit",
        "slots_collected_update": {},
    }
    graph._apply_aluminum_category_guardrail(
        latest_message="I need a utility aluminum trailer",
        current_category=None,
        awaiting_slot=None,
        decision=implicit,
    )
    assert implicit["trailer_category"] == "Aluminum"
    assert implicit["slots_collected_update"]["base_category"] == "Utility"

    qna = {"trailer_category": "Utility"}
    graph._apply_aluminum_category_guardrail(
        latest_message="utility",
        current_category="Aluminum",
        awaiting_slot="base_category",
        decision=qna,
    )
    assert qna["trailer_category"] == "Aluminum"

    slots = {"base_category": "Utility"}
    filters = {}
    graph._apply_aluminum_base_category_filter("Aluminum", slots, filters)
    assert filters["subcategory"] == "Utility"

    explicit_switch = {"trailer_category": "Utility"}
    graph._apply_aluminum_category_guardrail(
        latest_message="not aluminum, I want utility",
        current_category="Aluminum",
        awaiting_slot="base_category",
        decision=explicit_switch,
    )
    assert explicit_switch["trailer_category"] == "Utility"


def test_llm_field_mapping_normalizes_aluminum_payload_alias():
    normalized = graph._normalize_llm_field_mappings(
        {
            "slots_collected_update": {
                "base_category": "Utility",
                "payload_lbs": "5000 pounds",
            },
            "metadata_filters_update": {},
        },
        "Aluminum",
    )

    assert normalized["slots_collected_update"] == {
        "base_category": "Utility",
        "payload_need": "5000 pounds",
    }
    assert normalized["metadata_filters_update"] == {
        "payload_lbs": "5000 pounds",
    }

    capacity_alias = graph._normalize_llm_field_mappings(
        {
            "slots_collected_update": {
                "base_category": "Utility",
                "payload_capacity": "5000 pounds",
            },
            "metadata_filters_update": {},
        },
        "Aluminum",
    )
    assert capacity_alias["slots_collected_update"] == {
        "base_category": "Utility",
        "payload_need": "5000 pounds",
    }
    assert capacity_alias["metadata_filters_update"] == {
        "payload_lbs": "5000 pounds",
    }


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
            "question": "About how wide is the load, or what trailer width do you need?",
            "required": True,
        }
    ]
    state["metadata_filters_collected"] = {"width_ft": "6 ft"}

    out = graph._apply_mind_node(state)

    assert "item_or_trailer_width_ft" in out["slots_skipped"]
    assert "width_ft" not in out["metadata_filters_collected"]
    assert out["pending_questions"] == []
    assert out["awaiting_slot"] == "hitch_type"
    assert out["assistant_text"] == "Do you prefer a bumper pull or gooseneck hitch?"
    assert out["mind_decision"]["action"] == "respond"


def test_no_fixed_size_answer_skips_active_width_question(monkeypatch):
    _use_fallback_extractor(monkeypatch)

    def _preference_classifier(**kwargs):
        assert kwargs["active_question"] == "About how wide is the load, or what trailer width do you need?"
        assert kwargs["user_message"] == "no fixed size regarding it"
        assert kwargs["awaiting_slot"] == "item_or_trailer_width_ft"
        return graph.PreferenceNullDecision(
            has_no_preference=True,
            target_slots=["item_or_trailer_width_ft"],
            target_metadata_filters=["width_ft"],
            reason="User has no fixed width preference.",
            confidence="high",
        )

    monkeypatch.setattr(graph, "classify_no_preference", _preference_classifier)
    state = _state("no fixed size regarding it", category="Equipment")
    state["messages"] = [
        {
            "role": "assistant",
            "content": "About how wide is the load, or what trailer width do you need?",
        },
        {"role": "user", "content": "no fixed size regarding it"},
    ]
    state["slots_collected"] = {
        "haul_item": "a car",
        "haul_weight_lbs": "3000 lbs",
        "haul_length_ft": "12 ft",
    }
    state["awaiting_slot"] = "item_or_trailer_width_ft"
    state["pending_questions"] = [
        {
            "slot": "item_or_trailer_width_ft",
            "question": "About how wide is the load, or what trailer width do you need?",
            "required": True,
        }
    ]
    state["metadata_filters_collected"] = {"width_ft": "6 ft"}

    out = graph._apply_mind_node(state)

    assert "item_or_trailer_width_ft" in out["slots_skipped"]
    assert "width_ft" not in out["metadata_filters_collected"]
    assert out["pending_questions"] == []
    assert out["awaiting_slot"] == "hitch_type"
    assert out["assistant_text"] == "Do you prefer a bumper pull or gooseneck hitch?"
    assert out["mind_decision"]["action"] == "respond"


def test_active_extraction_advances_and_skipped_width_cannot_reappear(monkeypatch):
    _mock_field_updates(
        monkeypatch,
        slots_collected_update={"haul_length_ft": "18 ft"},
        metadata_filters_update={"length_ft": "18 ft"},
    )
    _mock_haul_classifier(monkeypatch)
    _mock_preference_classifier(monkeypatch)
    _mock_pre_generic_classifier(monkeypatch)
    unresolved = graph.QuestionTurnDecision(
        answered_active_question=False,
        reply_to_user="What is the rough total weight?",
        rephrased_question="What is the rough total weight?",
        retry_slot="haul_length_ft",
        confidence="low",
    )
    monkeypatch.setattr(
        graph,
        "_adjudicate_active_question_turn",
        lambda **kwargs: unresolved,
    )
    monkeypatch.setattr(
        graph,
        "_reconcile_active_question_turn",
        lambda **kwargs: unresolved,
    )
    state = _state("Maybe eighteen to twenty-four feet should do.", category="Equipment")
    state["slots_collected"] = {
        "haul_item": "tractor",
        "haul_weight_lbs": "8000 lbs",
    }
    state["metadata_filters_collected"] = {"payload_lbs": "8000 lbs"}
    state["awaiting_slot"] = "haul_length_ft"
    state["pending_questions"] = [{
        "slot": "haul_length_ft",
        "question": "About how long is the load (or what deck length do you need)?",
        "required": True,
    }]
    state["mind_decision"]["assistant_text"] = "What is the rough total weight?"

    out = graph._apply_mind_node(state)

    assert out["slots_collected"]["haul_length_ft"] == "18 ft"
    assert out["metadata_filters_collected"]["length_ft"] == "18 ft"
    assert out["awaiting_slot"] == "hitch_type"
    assert out["assistant_text"] == "Do you prefer a bumper pull or gooseneck hitch?"

    for _category in ("Car Hauler", "Tilt"):
        slots = {"item_or_trailer_width_ft": "flexible"}
        filters = {"width_ft": "flexible"}
        graph._enforce_skipped_slot_invariants(
            slots=slots,
            metadata_filters=filters,
            slots_skipped={"item_or_trailer_width_ft"},
        )
        assert slots == {}
        assert filters == {}


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
    assert out["awaiting_slot"] == "hitch_type"
    assert out["assistant_text"] == "Do you prefer a bumper pull or gooseneck hitch?"
    assert out["mind_decision"]["action"] == "respond"
