from src.chatbot.service import _create_or_get_conversation
from src.db_models import ChatbotConversation
from sqlalchemy.dialects import postgresql


class _Session:
    def __init__(self):
        self.statement = None
        self.row = object()

    def execute(self, statement):
        self.statement = statement

    def get(self, model, identity):
        assert model is ChatbotConversation
        assert identity == "session-id"
        return self.row


def test_conversation_creation_uses_on_conflict_and_reloads_row():
    session = _Session()

    row = _create_or_get_conversation(
        session,
        session_id="session-id",
        lead_id="lead-id",
        conversation=[],
    )

    sql = str(session.statement.compile(dialect=postgresql.dialect()))
    assert "ON CONFLICT (session_id) DO NOTHING" in sql
    assert row is session.row
