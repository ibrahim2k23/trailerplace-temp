from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _LiveCallBlocked:
    """Offline stand-in for the real ``ChatOpenAI`` client.

    After the E2 refactor every conversational/classifier LLM is constructed
    through ``src.chatbot.llm.make_llm`` -> ``ChatOpenAI``. Tests that mock their
    LLM (by patching the relevant factory function) never reach this class. A
    test that instead relies on a real model call reaches here and is *skipped*
    rather than making a live network request that would fail (no/invalid key)
    or return non-deterministic output. Set ``RUN_LIVE=1`` to exercise those
    tests against the real API.
    """

    def __init__(self, *args, **kwargs):
        pass

    def with_structured_output(self, *args, **kwargs):
        return self

    def bind(self, *args, **kwargs):
        return self

    def invoke(self, *args, **kwargs):
        # pytest.skip raises Skipped (a BaseException), so it propagates through
        # the app's ``except Exception`` fallbacks and cleanly skips the test.
        pytest.skip("un-mocked live LLM call (set RUN_LIVE=1 to run)")


@pytest.fixture(autouse=True)
def _block_live_llm(monkeypatch):
    """Auto-skip any test that makes an un-mocked live LLM call, unless RUN_LIVE."""
    if os.getenv("RUN_LIVE"):
        return
    import src.chatbot.llm as _llm

    monkeypatch.setattr(_llm, "ChatOpenAI", _LiveCallBlocked)
