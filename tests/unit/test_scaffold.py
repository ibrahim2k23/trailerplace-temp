from __future__ import annotations

from fastapi.testclient import TestClient

from src.api import routes
from src.api.app import create_app
from src.config import Settings
from src.models import TrailerListing
from src.shown_listings_store import (
    accumulate_shown_urls_from_chat_messages,
    add_shown_keys_and_urls,
    add_shown_urls,
)
from tests.conftest import FakeLLM


def test_health(monkeypatch):
    # Since M8 /health reports ok only after the lifespan compiles the graph and
    # clears the DB probe, so the client must be entered as a context manager.
    monkeypatch.setattr("src.conversation_store.persistence_enabled", lambda: False)
    monkeypatch.setattr("src.db.database_enabled", lambda: False)
    routes.set_graph_client(FakeLLM([]))
    with TestClient(create_app()) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_app_py_imports():
    import src.conversation_store as conversation_store
    import src.log_setup as log_setup
    import src.models as models
    import src.shown_listings_store as shown_listings_store
    import src.thinking_agent as thinking_agent

    for module, names in {
        log_setup: ["configure_trailerplace_logging"],
        conversation_store: ["enqueue_save_user_feedback", "persistence_enabled"],
        shown_listings_store: [
            "accumulate_shown_urls_from_chat_messages",
            "add_shown_keys_and_urls",
            "add_shown_urls",
        ],
        models: ["TrailerListing"],
        thinking_agent: [
            "generate_thinking_flow",
            "log_thinking_flow",
            "thinking_agent_background",
            "thinking_agent_enabled",
        ],
    }.items():
        for name in names:
            assert getattr(module, name)


def test_settings_loads(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "custom-model")
    monkeypatch.setenv("SEARCH_TOP_K", "7")
    monkeypatch.setenv("TRAILERPLACE_PERSIST_CHATS", "true")
    settings = Settings.from_env()
    assert settings.openai_model == "custom-model"
    assert settings.search_top_k == 7
    assert settings.trailerplace_persist_chats is True
    assert settings.openai_embedding_model == "text-embedding-3-small"
    assert settings.chatbot_api_port == 8000


def test_trailer_listing_lenient():
    TrailerListing(title="x", url="y", price="Call for price")
    TrailerListing(title="x", url="y", price=8400.0)


def test_shown_listings_store():
    add_shown_urls("s1", ["https://a.test"])
    add_shown_keys_and_urls("s1", ["k"], ["https://b.test"])
    listing = TrailerListing(title="x", url="https://c.test")
    urls = accumulate_shown_urls_from_chat_messages(
        [
            {"role": "assistant", "listings": None},
            {"role": "assistant", "listings": [{"url": "https://a.test"}, listing]},
            {"role": "assistant", "listings": [{"title": "missing"}]},
        ]
    )
    assert urls == ["https://a.test", "https://c.test"]
