from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ContactInfo(StrictBaseModel):
    name: str | None = Field(description="Customer name if provided, otherwise null.")
    email: str | None = Field(description="Customer email if provided, otherwise null.")
    phone: str | None = Field(description="Customer phone if provided, otherwise null.")


class SlotAnswer(StrictBaseModel):
    slot_name: str = Field(description="One current-category slot name exactly as listed in the prompt.")
    raw_answer: str = Field(description="The user's raw answer for that slot.")


class EmailTrigger(StrictBaseModel):
    kind: Literal["faq", "escalation", "team_request", "listing_interest"] = Field(description="Email-worthy request kind.")
    faq_key: Literal["contact_human", "financing", "trade_in", "service_parts", "store_info"] | None = Field(description="Required for FAQ triggers, otherwise null.")
    listing_reference: int | None = Field(description="1-based shown-listing index for listing interest, otherwise null.")
    description: str = Field(description="One-line summary for the email body.")


class ExtractedFields(StrictBaseModel):
    trailer_length_ft: float | None = Field(description="Trailer length already converted to feet by the model, or null.")
    trailer_width_ft: float | None = Field(description="Trailer width already converted to feet by the model, or null.")
    trailer_height_ft: float | None = Field(description="Trailer height already converted to feet by the model, or null.")
    payload_lbs: float | None = Field(description="Payload already converted to pounds by the model, or null.")
    hitch_type: list[Literal["Bumper Pull", "Gooseneck"]] | None = Field(description="A single clear hitch preference, or null (including when the customer says either/any/no preference).")
    haul_item: str | None = Field(description="Cargo or item to haul, in the user's words.")
    brand_preference: str | None = Field(description="Canonical known make, or unknown brand verbatim.")
    non_metadata_features: list[str] = Field(description="Preferences not directly searchable as metadata.")
    numeric_no_preference: list[str] = Field(description="Slot names where the user gave no numeric preference.")


class HaulClassification(StrictBaseModel):
    is_lightweight_utility_load: bool = Field(description="True only for Utility cargo at or below ~1500 lbs (a WEIGHT judgment). Signals the load is light, so the weight question is skipped.")
    needs_width_question: bool = Field(description="True for large, wide, heavy-duty, or vehicle cargo (a SIZE judgment). Signals the trailer must be wide enough, so the width question is asked.")
    haul_item_matched: str | None = Field(description="Specific cargo grounded in user words, or null.")


class InventoryLookup(StrictBaseModel):
    is_lookup: bool = Field(description="True when the message references specific inventory identifiers.")
    year: int | None = Field(description="Model year mentioned, never a stock number.")
    make: str | None = Field(description="Canonical make if mentioned for lookup.")
    model_text: str | None = Field(description="Model code or phrase exactly as the user typed it.")
    stock_number: str | None = Field(description="Explicit stock/unit/id number only, never weight/length/price/year/phone.")
    wants: Literal["price", "availability", "details", "general"] | None = Field(description="What the user wants about the inventory item.")
    confidence: Literal["low", "medium", "high"] = Field(description="Lookup confidence; low must fail closed.")


class TurnAnalysis(StrictBaseModel):
    intent: Literal[
        "general_question", "category_exploration", "category_selection",
        "feature_request_no_category", "recommendation_request",
        "qualification_answer", "skip_current", "skip_all_show_results", "show_more_results",
        "requirement_change", "drop_requirements", "category_change",
        "listing_interest", "faq", "team_request_escalation",
        "inventory_lookup", "contact_info_provided", "contact_declined", "smalltalk_other",
    ] = Field(description="Dominant user intent for routing.")
    email_triggers: list[EmailTrigger] = Field(description="Every email-worthy request in message order.")
    haul_classification: HaulClassification = Field(description="Cargo weight/size classification for width logic.")
    inventory_lookup: InventoryLookup = Field(description="Specific-inventory lookup identifiers.")
    category_mentioned: str | None = Field(description="Canonical category mentioned or null.")
    is_category_info_only: bool = Field(description="True for category information questions, not selection.")
    extracted: ExtractedFields = Field(description="Structured extracted fields from the latest message.")
    slot_answers: list[SlotAnswer] = Field(description="List of slot answer pairs; never a dict.")
    contact: ContactInfo = Field(description="Contact information extracted from the message.")
    listing_reference: int | None = Field(description="1-based shown-listing reference, if any.")
    dropped_fields: list[str] = Field(description="Fields the user asked to drop.")
    keep_fields_answer: Literal["all", "none", "some"] | None = Field(description="Category-change keep/drop answer.")
    kept_fields: list[str] = Field(description="Specific fields the user wants to keep.")
    category_confirm_answer: Literal["yes", "no"] | None = Field(
        description="Only when a category-switch suggestion is pending: 'yes' to switch to the suggested category, 'no' to stay on the current one. Null otherwise."
    )
    answered_current_question: bool = Field(description="Whether latest message answered the pending question.")
    user_question_to_answer: str | None = Field(description="Interruption question to answer, verbatim.")


class ReplyOutput(StrictBaseModel):
    assistant_text: str = Field(description="Customer-facing assistant reply.")
    cited_listing_urls: list[str] = Field(description="Exact URLs for every listing mentioned in the reply.")
