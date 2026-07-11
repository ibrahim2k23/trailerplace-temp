"""Milestone 9 §2 — LangSmith wiring. No network: tracing stays disabled or is faked."""
from __future__ import annotations

import os

import pytest

from src import tracing
from tests.conftest import replace_settings


@pytest.fixture(autouse=True)
def _reset_configured(monkeypatch):
    monkeypatch.setattr(tracing, "_configured", False)
    yield
    monkeypatch.setattr(tracing, "_configured", False)


def _set(monkeypatch, **values):
    """Settings is a frozen dataclass — swap the whole object, not a field."""
    replace_settings(monkeypatch, **values)


def test_tracing_disabled_without_a_key(monkeypatch):
    _set(monkeypatch, langsmith_tracing=True, langsmith_api_key="")
    assert tracing.tracing_enabled() is False


def test_tracing_disabled_when_flag_off(monkeypatch):
    _set(monkeypatch, langsmith_tracing=False, langsmith_api_key="lsv2_key")
    assert tracing.tracing_enabled() is False


def test_tracing_enabled_needs_both(monkeypatch):
    _set(monkeypatch, langsmith_tracing=True, langsmith_api_key="lsv2_key")
    assert tracing.tracing_enabled() is True


def test_configure_forces_env_off_when_disabled(monkeypatch):
    # A stale LANGSMITH_TRACING=true in the ambient env must not survive: without a
    # key the SDK would try to ship traces and log errors on every call.
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    _set(monkeypatch, langsmith_tracing=False, langsmith_api_key="")
    assert tracing.configure_langsmith() is False
    assert os.environ["LANGSMITH_TRACING"] == "false"


def test_configure_exports_settings_to_env(monkeypatch):
    monkeypatch.delenv("LANGSMITH_ENDPOINT", raising=False)
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    _set(
        monkeypatch,
        langsmith_tracing=True,
        langsmith_api_key="lsv2_key",
        langsmith_endpoint="https://api.smith.langchain.com",
        langsmith_project="TrailerPlace",
    )
    assert tracing.configure_langsmith() is True
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_API_KEY"] == "lsv2_key"
    assert os.environ["LANGSMITH_ENDPOINT"] == "https://api.smith.langchain.com"
    assert os.environ["LANGSMITH_PROJECT"] == "TrailerPlace"


def test_configure_is_idempotent(monkeypatch):
    _set(monkeypatch, langsmith_tracing=True, langsmith_api_key="lsv2_key", langsmith_project="P1")
    tracing.configure_langsmith()
    # A second call must not re-export changed settings; the SDK reads env once.
    _set(monkeypatch, langsmith_project="P2")
    tracing.configure_langsmith()
    assert os.environ["LANGSMITH_PROJECT"] == "P1"


def test_trace_turn_is_a_noop_when_disabled(monkeypatch):
    _set(monkeypatch, langsmith_tracing=False, langsmith_api_key="")
    with tracing.trace_turn("session-1", turn_id="t1"):
        pass  # must not raise, must not import/contact langsmith


def test_add_turn_metadata_is_a_noop_when_disabled(monkeypatch):
    _set(monkeypatch, langsmith_tracing=False, langsmith_api_key="")
    tracing.add_turn_metadata(intent="faq", category="Dump")  # no run tree, no raise


def test_trace_turn_tags_the_session_and_drops_none_metadata(monkeypatch):
    _set(monkeypatch, langsmith_tracing=True, langsmith_api_key="lsv2_key")
    captured = {}

    class _Ctx:
        def __enter__(self):
            return None

        def __exit__(self, *exc):
            return False

    def fake_tracing_context(*, tags=None, metadata=None, **kwargs):
        captured["tags"] = tags
        captured["metadata"] = metadata
        return _Ctx()

    monkeypatch.setattr("langsmith.run_helpers.tracing_context", fake_tracing_context)
    with tracing.trace_turn("session-1", turn_id="t1", category=None):
        pass

    assert captured["tags"] == ["session:session-1"]
    assert captured["metadata"] == {"session_id": "session-1", "turn_id": "t1"}
    assert "category" not in captured["metadata"]


def test_add_turn_metadata_updates_the_current_run(monkeypatch):
    _set(monkeypatch, langsmith_tracing=True, langsmith_api_key="lsv2_key")

    class _Run:
        def __init__(self):
            self.extra = {}

    run = _Run()
    monkeypatch.setattr("langsmith.run_helpers.get_current_run_tree", lambda: run)
    tracing.add_turn_metadata(intent="faq", category=None)
    assert run.extra["metadata"] == {"intent": "faq"}


def test_add_turn_metadata_without_a_run(monkeypatch):
    _set(monkeypatch, langsmith_tracing=True, langsmith_api_key="lsv2_key")
    monkeypatch.setattr("langsmith.run_helpers.get_current_run_tree", lambda: None)
    tracing.add_turn_metadata(intent="faq")  # no raise


def test_report_llm_usage_is_a_noop_when_disabled(monkeypatch):
    _set(monkeypatch, langsmith_tracing=False, langsmith_api_key="")
    tracing.report_llm_usage("gpt-4o-mini", 100, 20)  # no run tree, no raise


def test_report_llm_usage_sets_usage_metadata_and_model(monkeypatch):
    _set(monkeypatch, langsmith_tracing=True, langsmith_api_key="lsv2_key")
    captured = {}

    class _Run:
        def set(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("langsmith.run_helpers.get_current_run_tree", lambda: _Run())
    tracing.report_llm_usage("gpt-4o-mini", 1000, 250)

    assert captured["usage_metadata"] == {
        "input_tokens": 1000,
        "output_tokens": 250,
        "total_tokens": 1250,
    }
    # ls_model_name is the key LangSmith prices from — without it, Cost stays blank.
    assert captured["metadata"]["ls_model_name"] == "gpt-4o-mini"


def test_report_llm_usage_without_a_run(monkeypatch):
    _set(monkeypatch, langsmith_tracing=True, langsmith_api_key="lsv2_key")
    monkeypatch.setattr("langsmith.run_helpers.get_current_run_tree", lambda: None)
    tracing.report_llm_usage("gpt-4o-mini", 1, 1)  # no raise


def test_traceable_or_passthrough_returns_a_working_decorator():
    @tracing.traceable_or_passthrough("test.fn")
    def add(a, b):
        return a + b

    assert add(2, 3) == 5
    assert add.__name__ == "add"  # functools.wraps preserved


def test_decorated_function_is_not_traced_when_disabled(monkeypatch):
    """The whole point of the runtime gate: pytest must never build a traced fn."""
    _set(monkeypatch, langsmith_tracing=False, langsmith_api_key="")
    calls = []
    monkeypatch.setattr("langsmith.traceable", lambda **kw: calls.append(kw))

    @tracing.traceable_or_passthrough("test.fn")
    def add(a, b):
        return a + b

    assert add(2, 3) == 5
    assert calls == []


def test_decorated_function_is_traced_once_when_enabled(monkeypatch):
    _set(monkeypatch, langsmith_tracing=True, langsmith_api_key="lsv2_key")
    built = []

    def fake_traceable(**kwargs):
        built.append(kwargs["name"])

        def wrap(func):
            def inner(*a, **kw):
                return func(*a, **kw) * 10

            return inner

        return wrap

    monkeypatch.setattr("langsmith.traceable", fake_traceable)

    @tracing.traceable_or_passthrough("test.fn")
    def add(a, b):
        return a + b

    assert add(2, 3) == 50
    assert add(1, 1) == 20
    assert built == ["test.fn"]  # traced variant built once, then cached
