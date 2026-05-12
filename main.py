from __future__ import annotations

import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI

from src.chatbot.service import handle_chat, reset_session
from src.log_setup import configure_trailerplace_logging
from src.models import ChatRequest, ChatResponse, ResetSessionRequest

load_dotenv()
configure_trailerplace_logging()

app = FastAPI(title="TrailerPlace LangGraph Chatbot")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    return handle_chat(request)


@app.post("/session/reset")
def reset(request: ResetSessionRequest) -> dict[str, str]:
    reset_session(request.session_id)
    return {"status": "ok"}


if __name__ == "__main__":
    port = int((os.getenv("CHATBOT_API_PORT") or "8000").strip())
    uvicorn.run("main:app", host="127.0.0.1", port=port, reload=False)
