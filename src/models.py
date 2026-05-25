from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class TrailerListing(BaseModel):
    listing_id: str
    title: str
    condition: str = "New"
    price: Optional[float | str] = None
    price_display: Optional[str] = None
    payments_from: Optional[str] = None
    category_subcategory: str = ""
    make: str = ""
    color: str = ""
    hitch_type: Optional[str] = None
    year: Optional[str | int] = None
    length: Optional[str] = None
    width: Optional[str] = None
    axles: Optional[str] = None
    gvwr: Optional[str] = None
    payload_capacity: Optional[str] = None
    trailer_material: Optional[str] = None
    floor: Optional[str] = None
    url: str = ""
    score: Optional[float] = None


class ChatRequest(BaseModel):
    session_id: str
    sales_phase: str = "onboarding"
    message: str
    onboarding_api_messages: list[dict[str, Any]] = Field(default_factory=list)
    customer_full_name: Optional[str] = None
    customer_email: Optional[str] = None
    customer_phone: Optional[str] = None
    already_shown_listing_urls: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    assistant_text: str
    sales_phase: str
    onboarding_api_messages: list[dict[str, Any]] = Field(default_factory=list)
    customer_full_name: Optional[str] = None
    customer_email: Optional[str] = None
    customer_phone: Optional[str] = None
    contact_status: Optional[str] = None
    main_prior_messages: list[dict[str, Any]] = Field(default_factory=list)
    listings: list[dict[str, Any]] = Field(default_factory=list)
    thinking_context: Optional[dict[str, Any]] = None


class ResetSessionRequest(BaseModel):
    session_id: str
