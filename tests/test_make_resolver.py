from src.chatbot.make_inventory import categories_for_make, make_filter_values
from src.chatbot import make_resolver
from src.chatbot.make_resolver import MakeResolution, _LLMMakeResolution, resolve_make_from_text


class _FakeMakeLLM:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        if self.error:
            raise self.error
        return self.result


def test_resolves_make_typo_without_llm():
    result = resolve_make_from_text("Do you have a Dimond C 6x12?", use_llm_fallback=False)

    assert result.make == "Diamond C"
    assert result.match_type in {"alias", "fuzzy"}


def test_gooseneck_hitch_does_not_resolve_as_make():
    result = resolve_make_from_text("I need a gooseneck flatbed trailer", use_llm_fallback=False)

    assert result.make is None


def test_gooseneck_brand_context_can_resolve_as_make():
    result = resolve_make_from_text("Do you have Gooseneck brand trailers?", use_llm_fallback=False)

    assert result.make == "Gooseneck"


def test_inventory_lookup_normalizes_make_and_filter_variants():
    assert "Flatbed" in categories_for_make("Diamond C")
    assert set(make_filter_values("Diamond C")) >= {"Diamond C", "Diamond C Trailers"}


def test_llm_runs_before_deterministic_make_match(monkeypatch):
    fake_llm = _FakeMakeLLM(
        _LLMMakeResolution(
            make="RD TRAILERS",
            confidence="high",
            reason="The user means the RD brand.",
        )
    )
    monkeypatch.setattr(make_resolver, "known_makes", lambda: ["Diamond C", "RD TRAILERS"])
    monkeypatch.setattr(make_resolver, "_llm", lambda: fake_llm)

    result = resolve_make_from_text("I want a Diamond C trailer")

    assert result == MakeResolution("RD TRAILERS", "high", "llm", "The user means the RD brand.")
    assert fake_llm.messages is not None


def test_llm_resolves_rd_trailer_when_deterministic_is_uncertain(monkeypatch):
    fake_llm = _FakeMakeLLM(
        _LLMMakeResolution(
            make="RD TRAILERS",
            confidence="high",
            reason="RD refers to a known make.",
        )
    )
    monkeypatch.setattr(make_resolver, "known_makes", lambda: ["RD TRAILERS"])
    monkeypatch.setattr(make_resolver, "_llm", lambda: fake_llm)

    result = resolve_make_from_text("I am looking for an RD trailer")

    assert result.make == "RD TRAILERS"
    assert result.match_type == "llm"


def test_llm_uncertain_result_falls_back_to_deterministic(monkeypatch):
    fake_llm = _FakeMakeLLM(_LLMMakeResolution(make=None, confidence="none", reason="Unclear."))
    monkeypatch.setattr(make_resolver, "known_makes", lambda: ["Diamond C"])
    monkeypatch.setattr(make_resolver, "_llm", lambda: fake_llm)

    result = resolve_make_from_text("Do you have a Dimond C 6x12?")

    assert result.make == "Diamond C"
    assert result.match_type in {"alias", "fuzzy"}


def test_llm_gooseneck_hitch_resolution_is_rejected(monkeypatch):
    fake_llm = _FakeMakeLLM(
        _LLMMakeResolution(
            make="Gooseneck",
            confidence="high",
            reason="The word gooseneck was present.",
        )
    )
    monkeypatch.setattr(make_resolver, "known_makes", lambda: ["Gooseneck"])
    monkeypatch.setattr(make_resolver, "_llm", lambda: fake_llm)

    result = resolve_make_from_text("I need a gooseneck flatbed trailer")

    assert result.make is None
    assert result.reason == "gooseneck_without_brand_context"
