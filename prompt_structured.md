# Multi-Agent Trailer Chatbot — System Design Prompt

Design a multi-agent AI chatbot for a trailer dealership. It helps customers select trailers, answers trailer/dealership questions, collects lead info, and drives the sales process.

## Architecture Principles
- Multi-agent: each agent is independent, does one well-defined job, takes **structured input**, returns **structured output**.
- Structured over free-form: consistent, machine-readable schemas that downstream agents consume without re-parsing or a second LLM to interpret.
- Minimize context passed between agents — share only what downstream needs.
- LLM-centric reasoning: no deterministic rules, hard-coded guards, regex, keyword filters, or rule-based routing unless explicitly requested. Structured outputs exist to aid reasoning.(Use less LLMS but not that less that the quality of chatbot degrades)
- Model: GPT-4o Mini is the default unless stated otherwise. Optimize for cost.
- Priority order: (1) low inference cost, (2) efficient context, (3) modular agents, (4) structured inter-agent communication, (5) easy extensibility.

## Codebase Structure
- Highly modularized, especially for LangGraph. Each helper function and each LLM call gets its own file. Optimize for readability.

---

## Data (authoritative unless stated otherwise)

**Trailer Categories** — each: name, description (optional), mapping terms (synonyms/abbreviations/phrases pointing to the category). Mapping terms help infer category from non-exact wording.
- Source: `@categories` (names + synonym mappings). Injected at runtime via `category_prompt_block()`, which lists each canonical category with its mapping terms.
- **Guard: Gooseneck and Bumper Pull are strictly hitch types, never categories.** Never infer, recommend, or return either as a category.
- **Category clarification rules:** some trigger terms are ambiguous across categories. When such a term appears, ask a clarification question **first** and resolve using the provided option set before mapping.

**Category Questions** — each category has an ordered list of dealership-defined qualification questions. Treat as data, not hard-coded logic; new categories/questions must not require architecture changes.
- Source: `@trailer_fields` / `@trailer questions`.

**Ingestion** — `@ingest.py` defines how Excel data is ingested.

---

## Tools

### 1. Escalation Alert Tool
Use when the user requests something the chatbot can't do directly (schedule a call/meeting, request a quote, draft/send email, any action needing a human).
- Recognize the request needs a human → invoke tool → it sends an internal escalation email to the dealership.
- Continue the conversation naturally afterward unless told otherwise.

### 2. Email Tool
Sends internal notification emails. Politely tell the user their request was forwarded, then continue helping.

FAQ canned responses:
- **contact_human:** "You can reach our team at 979-532-1486. Happy to keep helping with your trailer search too!"
- **financing:** "We offer financing. Call 979-532-1486 to speak with our finance team, and I can keep helping narrow down the right trailer."
- **trade_in:** "Our sales team handles trade-in appraisals. Call 979-532-1486."
- **service_parts:** "Our service and parts team can help. Reach them at 979-532-1486."
- **store_info:** "We're located in Wharton, TX. Call 979-532-1486 or visit https://trailerplace.com. We also offer financing and delivery."

Non-FAQ canned responses:
- **generic team request:** "Thanks, I shared that request with the team so they can help you with it."
- **escalation:** "I've passed your query to our team. In the meantime, I can keep helping you narrow down the right trailer."
- **listing interest, item selected:** "Your interest in the selected trailer has been logged. Our team can follow up. In the meantime, feel free to visit https://trailerplace.com or call 979-532-1486."
- **listing interest, no item selected:** "Your interest has been logged. Our team can follow up. In the meantime, feel free to visit https://trailerplace.com or call 979-532-1486."
- **safe listing-interest fallback:** "Great, I shared your interest in that trailer with the team. They can follow up with you."

Email body format:
```
Full Name:
Email:
Phone Number:

[Reason for Mail] One line description
```

### 3. Log Interest Tool
Logs interest in a specific displayed listing (see "After Pinecone Results").

### 4. Pinecone Search Tool
Retrieves trailer recommendations from the vector DB after requirements are collected. **The only deterministic tool invocation.**
- Invoke only when: every required qualification question for the selected category is answered, OR remaining questions are explicitly skipped and none are left.
- Never invoke before qualification is complete.
- Pass collected requirements as structured input; return structured results. No extra deterministic validation before/after unless later specified.

### Contact-Info Gate (Escalation, Non-FAQ Email, and Log Interest if it needs follow-up)
Minimum to send: **Name + (Email OR Phone)**.
- If missing when an email tool is triggered, ask again for name/email/phone.
  - Name + email → proceed. Name + phone → proceed.
  - Email/phone but no name → ask once for name.
  - Name but no contact method → ask once for email or phone.
  - User declines/ignores → do **not** invoke the tool; continue the conversation politely. Do not keep asking.
- Core rule: only invoke email tools when there's enough info for a human to follow up.

---

## Contact Collection (start of conversation)
Goal: enough info for follow-up with minimal friction. Aim for **Name + (Email OR Phone)**; all three preferred, not required.

On first message, politely request name, email, phone (any order). Outcomes:
- Name + email + phone → accept, don't ask again.
- Name + email → accept, don't ask for phone.
- Name + phone → accept, don't ask for email.
- Email/phone, no name → ask once for name.
- Name, no contact method → ask once for email or phone.

If the user declines, ignores, changes subject, or starts asking about trailers: immediately continue with their request; don't insist or interrupt.

Rules: ask for contact info **only once** during initial interaction. Never pressure. If any field is declined/ignored, consider collection complete and continue. Willingness to share must never affect assistance quality.

---

## Conversation Flow — Category & Qualification

The chatbot is a trailer lead specialist: identify category, collect requirements, answer trailer questions anytime, recommend trailers.

### Step 1 — Interpret intent (every message)
Interpret by intent, not by assuming the user is answering the current question. Intents:
1. **General trailer question** (towing, payload, dimensions, axles, features, use cases) → answer naturally; if mid-qualification, resume the pending question afterward (per repetition rules).
2. **Category exploration** ("options for hauling heavy equipment?") → recommend relevant categories, briefly explain each, ask which to explore.
3. **Category selection** ("I'm looking for a utility trailer") → map category, load its questions, begin the flow.
4. **Feature-based request without category** ("15-foot, tandem axles, bumper pull") → extract/store features, ask which category; don't start category questions until category is known.
5. **Recommendation request, unclear category** → suggest best-fit categories, ask which to explore.

### Step 2 — Category mapping
On selection/clear implication: map via names + mapping terms, load qualification questions, begin collecting. Distinguish **information request** ("What is a utility trailer?" — answer only) from **selection** ("I want a utility trailer" — start qualification).

### Step 3 — Lead-specialist behavior
Naturally guide toward finding a trailer. After answering info questions, transition into selection when appropriate. Be helpful and conversational, not repetitive or pushy.

### Step 4 — Pre-fill known variables
Each question maps to one variable. Before asking, check if it was mentioned anywhere earlier; if so store it, mark answered, skip, move on. Never ask for already-provided info.

### Step 5 — Question flow
After category known: ask one question at a time, store each answer to its variable, continue until all answered or skipped.

### Step 6 — Interruptions, counter-questions, non-answers
The flow is interruptible. The user may ask a general/clarifying/counter question, change topic, or reply without answering. When this happens:
1. Pause qualification.
2. Answer naturally.
3. Re-ask the current question **once**.
- If answered after the repeat → store, continue.
- If still unanswered after one repeat → mark **skipped**, continue.
- Never ask the same question beyond this single follow-up.

### Step 7 — Explicit skip
"Skip this" / "Next" / "I don't know" / "I'd rather not answer" → mark variable skipped, continue.

### Step 8 — Skip remaining
"Just show me what you have" / "Show me the trailers" / "No more questions" / "Give me recommendations" → stop asking, mark all remaining skipped, invoke Pinecone with collected variables, present results.

### Principles
Conversational not linear; interruptible anytime; interpret intent first; answer trailer questions before resuming; resume exactly where paused; never re-ask collected info; respect skips; Pinecone only after qualification completes or on early explicit request.

---

## After Pinecone Results
Conversation stays active and stateful. Determine intent after results.

1. **Search again / update requirements** — user changes a requirement (length, width, height, hitch, payload, brand): update that variable, keep the rest, re-run Pinecone, show new results.
2. **Listing interest** ("I like the second one", "tell me more about that Iron Bull", "I want this one") — identify the referenced listing (use stored listing data, especially URL), invoke Log Interest with that listing + available customer details, continue naturally.
3. **Store shown listings** — store every shown listing's URL in state to avoid repeats, support references ("the first one"), and let Log Interest resolve the listing. The Pinecone fetch/rank/filter layer handles avoiding previously-shown/unavailable listings; don't reimplement in the prompt.
4. **Repeated searches & category changes:**
   - *Requirement change, same category:* update only that variable, keep the rest; re-run Pinecone if the category flow is complete.
   - *Dropping requirements* ("forget all that", "just search by length"): remove what's dropped, keep the rest; continue flow or run Pinecone depending on completion.
   - *Category change* ("show me dump trailers instead") — pause and ask which collected features to keep:
     1. Map the new category.
     2. Identify transferable requirements (length, width, height, payload, hitch, axle, intended use, budget, brand).
     3. Ask which to keep.
     4. Keep all/some/none per the answer.
     5. Store kept values against the new category's variables.
     6. Ask remaining new-category questions.
     7. Run Pinecone once the new flow completes/skips.
   - Keep-all → reuse all transferable values. Drop-all → clear and start fresh. Keep-some → preserve those, ask the rest.
   - **Core rule:** on category change, never auto-reuse all previous requirements — ask first.
5. **Principles after results:** results aren't the end; keep helping refine/compare/select; detect new-search vs. details vs. compare vs. human follow-up; preserve context across searches; update only what the user changes.

---

## Global Behavior
Treat the conversation as holistic, not a fixed sequence. At any point (before selection, mid-qualification, after results, after multiple searches) the user may switch topics or ask anything. Always determine current intent first.

Possible intents anytime: general trailer question; specific category; feature/towing/payload questions; hauling a material/equipment; dealership questions; tool-requiring requests; change requirements; change category; compare; search again; express listing interest; resume qualification.

**State** persists across the session. Interruptions must **not** reset: selected category, collected variables, skipped variables, prior Pinecone searches, shown listings, or the current pending question. After an interruption, resume from the right point unless the latest message redirects.

**Tools** (Escalation, Non-FAQ Email, Pinecone, Log Interest) may be needed anytime — decide by current intent, not conversation stage.

Core principle: feel natural and human. Switch fluidly between answering, collecting, refining, explaining categories, invoking tools, showing recommendations, and continuing prior threads — without losing context or forcing a restart.

---

## Field Extraction & Category Defaults
Extract structured info from every message into state (to skip answered questions, prepare searches, preserve preferences). State is the single source of truth for downstream agents/tools.

### Searchable filter fields
Trailer Length, Trailer Width, Trailer Height, Payload Capacity, Hitch Type, Haul Item.
Brand Name is extracted from every message and stored as **Brand Preference**, but is **not** an active Pinecone metadata filter for now (see Brand Handling; expandable later).

### Extraction rules
- `AxB` → width x length. `AxBxC` → width x length x height.
- "16 by 8" → 16 ft length, 8 ft width.
- Convert length/width/height → **ft**; payload → **lbs**.
- Hitch type is only **bumper pull** or **gooseneck**.
- Extract on every message, even for fields not explicitly asked.

### Category defaults (plug-and-play)
Admins may set defaults per category for length, width, height, payload, hitch. On category selection: apply defaults, store to variables, treat as answered, skip matching questions — unless the user later changes them. User-provided values override defaults.
Example — Utility Trailer: Default Width 83 in, Default Hitch Bumper Pull → store both, don't ask.

### Loose / no-preference answers
Numeric fields (length/width/height/payload):
- Clear number → store it.
- Range → store the **smallest** acceptable value ("15–18 ft" → 15 ft).
- Loose answer that can't be numeric ("no preference", "flexible", "not sure") → store `null`.

Hitch type: clear preference → store; multiple acceptable → store as options; loose no-preference for example no idea, either is fine etcx` → `null`.

Haul Item: store whatever the user says to haul, as provided; don't over-normalize or discard vague descriptions (e.g. "random things", "furniture, wood, and other stuff"). Keep it usable for search.

Brand Name: store preference if mentioned, else `null`.

### Non-Metadata Features
Preferences that aren't searchable filter fields (ramp, axle capacity, drive-over fenders, electric brakes, hydraulic jack, tarp, spare tire, winch, LED lights, toolbox, dovetail, mesh sides, color) → store under **Non-Metadata Features**. Preserved in state but not directly searchable now.

### Updating state
State always reflects latest intent. Change one field → update only it. Drop requests ("forget all that", "just use the length") → remove dropped fields, keep the rest.

### Extraction principles
Extract every message; apply category defaults on selection; user values override defaults; ranges → smallest value; loose numeric → null; store haul items as given; unsupported prefs → Non-Metadata Features; never ask for a variable already extracted/defaulted/skipped/no-preference.

---

## Category Mapping Priority & Haul-Item Lock
- **Before a category is chosen:** may infer category from mapping terms ("haul livestock" → livestock's category, begin its flow).
- **Explicit category wins over haul-item terms:** "tilt trailer to haul a tractor" → Category = Tilt, Haul Item = Tractor (don't switch to a tractor-mapped category).
- **Haul item never overrides a selected category (deterministic lock):** once a category is set, extracting the haul-item answer must not remap the category, even if the item appears in another category's mapping terms.
- **Category change requires explicit user intent:** only change on clear requests ("show me utility trailers instead", "switch to dump", "I don't want tilt anymore"), then run the category-change flow.

---

## Brand Handling
Recognize brands (makes) separately from categories. For now Brand Name is **not** a searchable filter in main extraction (expandable later). The known-make list is injected at runtime via `make_prompt_block()` — each canonical make listed with the categories it's available in (from the inventory workbook).
- **Guard: Gooseneck and Bumper Pull are strictly hitch types, never makes/brands.** Never infer, recommend, or return either as a make.
- The make→categories mapping is what powers the brand-only flow below ("show that brand's available categories").
- Known brand mentioned → store as **Brand Preference**, not a category.
- **Brand + category** ("Diamond C utility trailer") → store Brand Preference, map Category, begin that category's flow, don't re-ask category.
- **Brand only** ("Diamond C trailer") → store Brand Preference, don't assume it's the category; show that brand's available categories (if available) and ask which to explore. E.g. "We have Diamond C options across these trailer categories. Which type would you like to look into?"
- **Never infer category from brand alone** ("I want Diamond C" must not map to any category) unless the category is stated or implied by approved mapping terms.
- Core: brand recognition and category mapping are separate; a brand narrows search later but never replaces category selection unless both are clearly stated.

---

## Pinecone Search Module (`@pinecone_search`)
Contains:
- Pinecone inventory search logic + OpenAI embedding generation for queries.
- Metadata filter building for category, make, hitch type, subcategory, length.
- Listing cleanup/normalization from match metadata.
- Fit-based reranking (length, payload/GVWR, width, height) and category/brand preference reranking.
- Make/brand alias normalization.
- Dedupe of already-shown listing URLs.
- Debug metadata for reranking and make-preference decisions.
- Public search functions returning listing recommendations.

### Slot → metadata normalization
Bin size = length. Each category has different variables but all resolve to length/width/payload/etc. Normalize via:
```python
_SLOT_METADATA_FILTER_MAP = {
    "base_category": ("subcategory",),
    "bin_size": ("length_ft",),
    "cargo_size": ("length_ft", "width_ft"),
    "haul_length_ft": ("length_ft",),
    "haul_weight_lbs": ("payload_lbs",),
    "item_or_trailer_width_ft": ("width_ft",),
    "payload_need": ("payload_lbs",),
    "trailer_length_ft": ("length_ft",),
    "trailer_size": ("length_ft", "width_ft"),
    "vehicle_length_ft": ("length_ft",),
}
```
Roll Off special case:
```python
if key in {"length_ft", "width_ft", "height_ft"}:
    if key == "length_ft" and normalize_category(category) == "Roll Off":
        return key, _normalize_roll_off_bin_size_as_length(value)
    return key, _normalize_length_or_width_value(value)
```
