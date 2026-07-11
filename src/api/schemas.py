from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str
    turn_id: str | None = None
    sales_phase: str | None = None
    message: str = ""
    onboarding_api_messages: list[dict[str, Any]] = Field(default_factory=list)
    customer_full_name: str | None = None
    customer_email: str | None = None
    customer_phone: str | None = None
    already_shown_listing_urls: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    assistant_text: str
    sales_phase: str = "main"
    onboarding_api_messages: list[dict[str, Any]] = Field(default_factory=list)
    listings: list[dict[str, Any]] = Field(default_factory=list)
    thinking_context: dict[str, Any] | None = None
    customer_full_name: str | None = None
    customer_email: str | None = None
    customer_phone: str | None = None
    main_prior_messages: list[dict[str, Any]] | None = None


class SessionResponse(BaseModel):
    exists: bool
    closed: bool = False
    messages: list[dict[str, Any]] = Field(default_factory=list)
    sales_phase: str = "main"
    onboarding_api_messages: list[dict[str, Any]] = Field(default_factory=list)
    customer_full_name: str | None = None
    customer_email: str | None = None
    customer_phone: str | None = None


class ResetRequest(BaseModel):
    session_id: str


class DebugStateResponse(BaseModel):
    exists: bool
    state: dict[str, Any] | None = None
