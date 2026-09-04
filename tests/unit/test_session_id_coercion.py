"""Session ids reach Postgres as UUIDs, whatever the channel sent.

Streamlit sends a uuid4 string; Facebook Messenger sends a PSID like
"9876543210987654". The chatbot_* tables are keyed by a real UUID column, so the
PSID has to be mapped onto one - deterministically, or a serverless cold start
would lose the customer's conversation.
"""
import uuid

import pytest

from src.conversation_store import SESSION_ID_NAMESPACE, as_session_uuid

PSID = "9876543210987654"


def test_a_streamlit_uuid_passes_through_unchanged():
    # Existing rows are keyed by exactly this value; remapping them would orphan them.
    sid = str(uuid.uuid4())
    assert str(as_session_uuid(sid)) == sid


def test_a_uuid_object_is_returned_as_is():
    sid = uuid.uuid4()
    assert as_session_uuid(sid) is sid


def test_a_psid_becomes_a_uuid_instead_of_raising():
    # Before the fix this raised ValueError inside durable_turn on the first message.
    with pytest.raises(ValueError):
        uuid.UUID(PSID)
    assert isinstance(as_session_uuid(PSID), uuid.UUID)


def test_the_same_psid_always_maps_to_the_same_uuid():
    """The whole point: this is what survives a cold start."""
    assert as_session_uuid(PSID) == as_session_uuid(PSID)
    assert as_session_uuid(PSID) == uuid.uuid5(SESSION_ID_NAMESPACE, PSID)


def test_different_psids_do_not_collide():
    assert as_session_uuid(PSID) != as_session_uuid("9876543210987655")


def test_the_namespace_is_pinned():
    """A regenerated namespace remaps every PSID, so it must never drift."""
    assert SESSION_ID_NAMESPACE == uuid.UUID("e893cdad-ec15-5fbd-80ae-7ef40a19fa55")
    assert str(as_session_uuid(PSID)) == "139a1793-928b-5757-964b-70874d3f6719"
