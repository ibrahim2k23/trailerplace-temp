from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


def _bool(value: str | None, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int(value: str | None, default: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _float(value: str | None, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    analyze_model: str = "gpt-5-mini"
    analyze_reasoning_effort: str = "minimal"
    openai_embedding_model: str = "text-embedding-3-small"
    pinecone_api_key: str = ""
    pinecone_index_name: str = "trailerplace-listings"
    search_top_k: int = 50
    search_max_recommendations: int = 5
    rerank_enabled: bool = True
    rerank_warn_ratio: float = 1.35
    rerank_extreme_ratio: float = 1.9
    rerank_length_weight: float = 8.0
    rerank_missing_dim_penalty: float = 0.35
    rerank_verbose_logs: bool = False
    make_rerank_verbose_logs: bool = False
    show_only_llm_mentioned_cards: bool = True
    trailerplace_website: str = ""
    host: str = ""
    pguser: str = ""
    password: str = ""
    database: str = ""
    port: int = 0
    trailerplace_persist_chats: bool = True
    email_backend: str = "smtp"
    tenant_id: str = ""
    client_id: str = ""
    client_secret: str = ""
    sender_email: str = ""
    recipient_email: str = ""
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = ""
    email_to: str = ""
    chatbot_api_port: int = 8000
    langsmith_tracing: bool = False
    langsmith_endpoint: str = ""
    langsmith_api_key: str = ""
    langsmith_project: str = ""
    debug_state_endpoint: bool = False
    db_auto_create: bool = False
    inventory_lookup_limit: int = 5
    # Must stay below app.py's 180 s client timeout on POST /chat.
    chat_timeout_seconds: float = 150.0
    chat_max_message_chars: int = 4000
    # Append-only JSONL of per-turn records; scripts/cost_report.py reads it (M9).
    turn_log_path: str = ""

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            analyze_model=os.getenv("ANALYZE_MODEL", "gpt-5-mini"),
            analyze_reasoning_effort=os.getenv("ANALYZE_REASONING_EFFORT", "minimal"),
            openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            pinecone_api_key=os.getenv("PINECONE_API_KEY", ""),
            pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "trailerplace-listings"),
            search_top_k=_int(os.getenv("SEARCH_TOP_K"), 50),
            search_max_recommendations=_int(os.getenv("SEARCH_MAX_RECOMMENDATIONS"), 6),
            rerank_enabled=_bool(os.getenv("RERANK_ENABLED"), True),
            rerank_warn_ratio=_float(os.getenv("RERANK_WARN_RATIO"), 1.35),
            rerank_extreme_ratio=_float(os.getenv("RERANK_EXTREME_RATIO"), 1.9),
            rerank_length_weight=_float(os.getenv("RERANK_LENGTH_WEIGHT"), 8.0),
            rerank_missing_dim_penalty=_float(os.getenv("RERANK_MISSING_DIM_PENALTY"), 0.35),
            rerank_verbose_logs=_bool(os.getenv("RERANK_VERBOSE_LOGS")),
            make_rerank_verbose_logs=_bool(os.getenv("MAKE_RERANK_VERBOSE_LOGS")),
            show_only_llm_mentioned_cards=_bool(os.getenv("SHOW_ONLY_LLM_MENTIONED_CARDS"), True),
            trailerplace_website=os.getenv("TRAILERPLACE_WEBSITE", ""),
            host=os.getenv("HOST", ""),
            pguser=os.getenv("PGUSER", ""),
            password=os.getenv("PASSWORD", ""),
            database=os.getenv("DATABASE", ""),
            port=_int(os.getenv("PORT"), 0),
            trailerplace_persist_chats=_bool(os.getenv("TRAILERPLACE_PERSIST_CHATS"), True),
            email_backend=os.getenv("EMAIL_BACKEND", "smtp"),
            tenant_id=os.getenv("TENANT_ID", ""),
            client_id=os.getenv("CLIENT_ID", ""),
            client_secret=os.getenv("CLIENT_SECRET", ""),
            sender_email=os.getenv("SENDER_EMAIL", ""),
            recipient_email=os.getenv("RECIPIENT_EMAIL", ""),
            smtp_host=os.getenv("SMTP_HOST", ""),
            smtp_port=_int(os.getenv("SMTP_PORT"), 587),
            smtp_user=os.getenv("SMTP_USER", ""),
            smtp_password=os.getenv("SMTP_PASSWORD", ""),
            smtp_from=os.getenv("SMTP_FROM", ""),
            email_to=os.getenv("EMAIL_TO", ""),
            chatbot_api_port=_int(os.getenv("CHATBOT_API_PORT"), 8000),
            langsmith_tracing=_bool(os.getenv("LANGSMITH_TRACING")),
            langsmith_endpoint=os.getenv("LANGSMITH_ENDPOINT", ""),
            langsmith_api_key=os.getenv("LANGSMITH_API_KEY", ""),
            langsmith_project=os.getenv("LANGSMITH_PROJECT", ""),
            debug_state_endpoint=_bool(os.getenv("DEBUG_STATE_ENDPOINT")),
            db_auto_create=_bool(os.getenv("DB_AUTO_CREATE")),
            inventory_lookup_limit=_int(os.getenv("INVENTORY_LOOKUP_LIMIT"), 5),
            chat_timeout_seconds=_float(os.getenv("CHAT_TIMEOUT_SECONDS"), 150.0),
            chat_max_message_chars=_int(os.getenv("CHAT_MAX_MESSAGE_CHARS"), 4000),
            turn_log_path=os.getenv("TURN_LOG_PATH", ""),
        )


settings = Settings.from_env()
