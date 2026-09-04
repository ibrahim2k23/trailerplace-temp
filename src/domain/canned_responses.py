FAQ_CANNED_RESPONSES = {
    "contact_human": "You can reach our team at 979-532-1486. Happy to keep helping with your trailer search too!",
    "financing": "We offer financing. Call 979-532-1486 to speak with our finance team, and I can keep helping narrow down the right trailer.",
    "trade_in": "Our sales team handles trade-in appraisals. Call 979-532-1486.",
    "service_parts": "Our service and parts team can help. Reach them at 979-532-1486.",
    "store_info": "We're located in Wharton, TX. Call 979-532-1486 or visit https://trailerplace.com. We also offer financing and delivery.",
}

# Every one of these fires on a turn we could NOT settle ourselves, so every one of them names
# the number. Leaving it out made the phone line a coin flip exactly where it matters most: the
# canned text arrives as an order, the model treats it as the answer, and the sales-rep line it
# would otherwise have written gets displaced - so the customer we just failed to help was told
# only that "the team" had been notified, with no way to reach anyone. The FAQ responses above
# always carried the number; these did not.
NON_FAQ_CANNED_RESPONSES = {
    "generic_team_request": "Thanks, I shared that request with the team so they can help you with it. If you'd rather not wait, our sales team is on 979-532-1486.",
    # NEVER offer to keep helping them shop here. This text used to end "In the meantime, I can
    # keep helping you narrow down the right trailer" - and because canned text arrives at the
    # respond model as an ORDER, a customer who opened with "I have a complaint against you guys"
    # was answered with a 13-item trailer catalogue and "which type do you want to go with?".
    # An escalation is someone telling us something went wrong, or asking something we cannot
    # answer; the only appropriate reply is that it is recorded, and how a person reaches them.
    "escalation": "I'm sorry to hear that. I've noted it and passed it to our team - they'll reach out to you, and you can also reach them directly on 979-532-1486.",
    "listing_interest_selected": "Your interest in the selected trailer has been logged. Our team can follow up. In the meantime, feel free to visit https://trailerplace.com or call 979-532-1486.",
    "listing_interest_unselected": "Your interest has been logged. Our team can follow up. In the meantime, feel free to visit https://trailerplace.com or call 979-532-1486.",
    "listing_interest_fallback": "Great, I shared your interest in that trailer with the team. They can follow up with you, or you can reach them on 979-532-1486.",
}

CANNED_RESPONSES = {**FAQ_CANNED_RESPONSES, **NON_FAQ_CANNED_RESPONSES}
