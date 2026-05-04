"""
TrailerPlace chat API (FastAPI).

Run from the project folder (next to this file):
    python main.py

Then start Streamlit (separate process). Set CHATBOT_API_URL if the API is not on http://127.0.0.1:8000.

HTTP listen port: CHATBOT_API_PORT (default 8000). Do not use generic PORT here — it often holds Postgres (5432) in .env.
"""
from __future__ import annotations

import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

from src.api_service import ChatRequest, ChatResponse, reset_server_session, run_chat  # noqa: E402
from src.log_setup import configure_trailerplace_logging  # noqa: E402

configure_trailerplace_logging()

_DEFAULT_ORIGINS = "http://127.0.0.1:8501,http://localhost:8501"
_origins_raw = (os.getenv("TRAILERPLACE_CORS_ORIGINS") or _DEFAULT_ORIGINS).strip()
_cors_origins = [o.strip() for o in _origins_raw.split(",") if o.strip()]

app = FastAPI(title="TrailerPlace Chat API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(body: ChatRequest) -> ChatResponse:
    try:
        return run_chat(body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class ResetSessionBody(BaseModel):
    session_id: str = Field(..., min_length=1)


@app.post("/session/reset")
def session_reset(body: ResetSessionBody) -> dict[str, bool]:
    reset_server_session(body.session_id)
    return {"ok": True}


def main() -> None:
    host = (os.getenv("CHATBOT_API_HOST") or "0.0.0.0").strip()
    # Never fall back to PORT — many .env files set PORT=5432 for Postgres.
    port = int((os.getenv("CHATBOT_API_PORT") or "8000").strip())
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
