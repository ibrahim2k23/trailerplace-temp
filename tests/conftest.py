from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _never_touch_a_real_database(request, monkeypatch) -> None:
    """No unpaid-for test may reach the configured Postgres (cross-milestone rule 5).

    `persistence_enabled()` is `TRAILERPLACE_PERSIST_CHATS and db.database_enabled()`,
    and a developer `.env` satisfies both. So `apply_analysis`'s calls to
    `update_lead_contact` / `update_lead_item_of_interest` would run
    `ensure_persistence_schema()` -> `create_all()` against a *production* database on
    every unit test — several seconds each, and a schema write nobody asked for.

    DB-marked tests opt back in: they are the ones that own a throwaway database via
    TEST_DATABASE_URL, and they skip when it is unset.
    """
    if request.node.get_closest_marker("db"):
        return
    from src import db

    monkeypatch.setattr(db, "database_enabled", lambda: False)


@pytest.fixture(autouse=True, scope="session")
def _disable_langsmith_tracing() -> None:
    """No test may ship a trace to LangSmith (cross-milestone rule 5: no network).

    The developer `.env` normally has LANGSMITH_TRACING=true, and `traceable` reads
    the env var, so without this the suite would bill real traces just by calling a
    decorated function. src.tracing gates on config.settings, so both are cleared.
    """
    from src import config

    os.environ["LANGSMITH_TRACING"] = "false"
    config.settings = dataclasses.replace(config.settings, langsmith_tracing=False, langsmith_api_key="")


def replace_settings(monkeypatch: pytest.MonkeyPatch, **values: Any) -> None:
    """Swap fields on the frozen Settings singleton for one test."""
    from src import config

    monkeypatch.setattr(config, "settings", dataclasses.replace(config.settings, **values))


class FakeLLM:
    def __init__(self, outputs: list[Any] | None = None) -> None:
        self.outputs = list(outputs or [])
        self.calls: list[dict[str, Any]] = []

    def structured(self, *, system: str, messages: list[dict], schema: type) -> Any:
        self.calls.append({"system": system, "messages": messages, "schema": schema})
        if not self.outputs:
            raise AssertionError("FakeLLM has no queued outputs")
        output = self.outputs.pop(0)
        if not isinstance(output, schema):
            raise AssertionError(f"FakeLLM output {type(output).__name__} does not match {schema.__name__}")
        return output


class FakeEmailSender:
    """Capturing stand-in for src.tools.email_sender used by M7 tests.

    Wire it either as a direct monkeypatch of email_sender.send_email (persistence
    off) or as an outbox handler (persistence on). `succeed=False` simulates a
    delivery failure so retry/failed-status paths can be exercised.
    """

    def __init__(self, succeed: bool = True) -> None:
        self.succeed = succeed
        self.sent: list[dict[str, Any]] = []

    def send_email(self, subject: str, body: str) -> bool:
        self.sent.append({"subject": subject, "body": body})
        return self.succeed

    def handler(self, *, subject: str, body: str) -> None:
        self.sent.append({"subject": subject, "body": body})
        if not self.succeed:
            raise RuntimeError("FakeEmailSender simulated failure")
