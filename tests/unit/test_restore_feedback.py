"""Milestone 8 §3 — feedback survives the round trip back to the frontend."""
from __future__ import annotations

from src.conversation_store import _apply_feedback_to_messages


def _messages(pairs: int) -> list[dict]:
    messages: list[dict] = []
    for i in range(pairs):
        messages.append({"role": "user", "content": f"q{i}"})
        messages.append({"role": "assistant", "content": f"a{i}"})
    return messages


def test_feedback_lands_on_the_matching_assistant_message():
    conversation = [
        {"user": "q0", "chatbot": "a0", "feedback": "great"},
        {"user": "q1", "chatbot": "a1"},
        {"user": "q2", "chatbot": "a2"},
        {"user": "q3", "chatbot": "a3", "feedback": "wrong trailer"},
    ]
    restored = _apply_feedback_to_messages(_messages(4), conversation)

    # turn_idx N maps to message 2N+1 — the same math app.py uses in reverse.
    assert restored[1]["user_feedback"] == "great"
    assert restored[7]["user_feedback"] == "wrong trailer"
    assert restored[3]["user_feedback"] is None
    assert restored[5]["user_feedback"] is None


def test_every_restored_message_has_the_keys_app_py_reads():
    restored = _apply_feedback_to_messages([{"role": "user", "content": "hi"}], None)
    assert restored[0]["listings"] is None
    assert restored[0]["user_feedback"] is None


def test_feedback_past_the_end_is_ignored():
    conversation = [{"user": "q0", "chatbot": None, "feedback": "note"}]
    # A trailing user message with no assistant reply yet: nothing to attach to.
    restored = _apply_feedback_to_messages([{"role": "user", "content": "q0"}], conversation)
    assert restored[0]["user_feedback"] is None


def test_empty_feedback_does_not_overwrite():
    conversation = [{"user": "q0", "chatbot": "a0", "feedback": ""}]
    restored = _apply_feedback_to_messages(_messages(1), conversation)
    assert restored[1]["user_feedback"] is None


def test_original_messages_are_not_mutated():
    messages = _messages(1)
    _apply_feedback_to_messages(messages, [{"user": "q0", "chatbot": "a0", "feedback": "x"}])
    assert "user_feedback" not in messages[1]
