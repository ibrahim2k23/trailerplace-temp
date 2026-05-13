from src import conversation_store
from src.chatbot import service


def test_merge_existing_feedback_preserves_feedback_when_incoming_lacks_it():
    existing = [
        {
            "user": "hello",
            "chatbot": "hi",
            "user_feedback": "Helpful response",
            "feedback_at": "2026-05-13T10:00:00+00:00",
        }
    ]
    incoming = [{"user": "hello", "chatbot": "hi again"}]

    merged = conversation_store._merge_existing_feedback(existing, incoming)

    assert merged == [
        {
            "user": "hello",
            "chatbot": "hi again",
            "feedback": "Helpful response",
        }
    ]


def test_merge_existing_feedback_keeps_incoming_feedback_when_present():
    existing = [
        {
            "user": "hello",
            "chatbot": "hi",
            "feedback": "Old feedback",
        }
    ]
    incoming = [
        {
            "user": "hello",
            "chatbot": "hi again",
            "feedback": "New feedback",
        }
    ]

    merged = conversation_store._merge_existing_feedback(existing, incoming)

    assert merged[0]["feedback"] == "New feedback"


def test_conversation_payload_uses_compact_blob_shape():
    session = {
        "messages": [
            {"role": "user", "content": "I need a utility trailer"},
            {
                "role": "assistant",
                "content": "What will you be hauling?",
                "user_feedback": "Already answered this",
                "tool_events": [{"tool": "pinecone_search", "result_count": 0}],
                "slots_collected": {"haul_item": "tractor"},
                "trailer_category": "Utility",
                "metadata_filters_collected": {"width_ft": "6"},
            },
        ]
    }

    payload = service._conversation_payload(session)

    assert payload == [
        {
            "user": "I need a utility trailer",
            "chatbot": "What will you be hauling?",
            "feedback": "Already answered this",
            "trailer_category": "Utility",
            "metadata_filters_collected": {"width_ft": "6"},
        }
    ]
