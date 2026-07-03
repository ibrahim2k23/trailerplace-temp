from __future__ import annotations

import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI

from src.chatbot.inventory_matcher import (
    TrailerSearchRequest,
    TrailerSearchResponse,
    search_trailers,
)
from src.chatbot.service import handle_chat, reset_session
from src.log_setup import configure_trailerplace_logging
from src.conversation_store import restore_session
from src.models import ChatRequest, ChatResponse, ResetSessionRequest, SessionRestoreResponse

load_dotenv()
configure_trailerplace_logging()

app = FastAPI(title="TrailerPlace LangGraph Chatbot")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    return handle_chat(request)


@app.get("/session/{session_id}", response_model=SessionRestoreResponse)
def get_session(session_id: str) -> SessionRestoreResponse:
    return SessionRestoreResponse.model_validate(restore_session(session_id))


@app.post("/trailer-search", response_model=TrailerSearchResponse)
def trailer_search(request: TrailerSearchRequest) -> TrailerSearchResponse:
    result = search_trailers(request.message)
    return TrailerSearchResponse(
        reply=str(result.get("reply") or ""),
        entity_type=str(result.get("entity_type") or "UNKNOWN_SEARCH"),
        confidence=float(result.get("confidence") or 0.0),
        best_match=result.get("best_match"),
        top_matches=result.get("top_matches") or [],
        extraction=result.get("extraction") or {},
    )


@app.post("/session/reset")
def reset(request: ResetSessionRequest) -> dict[str, str]:
    reset_session(request.session_id)
    return {"status": "ok"}


if __name__ == "__main__":
    port = int((os.getenv("CHATBOT_API_PORT") or "8000").strip())
    uvicorn.run("main:app", host="127.0.0.1", port=port, reload=False)
