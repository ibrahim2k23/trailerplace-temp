# TrailerPlace Chatbot — Functional and Technical Overview

## Instructions for a new Codex or model session

Read this file before planning or changing chatbot behavior. It is the functional handoff for the
project: it explains what the chatbot is for, how each conversation flow should work, which state
must be preserved, when tools may run, and how the LLMs cooperate.

When implementing or reviewing a change:

1. Treat the rules marked as intended/non-negotiable in this document as the desired behavior.
2. Verify the current implementation in the referenced source files before editing; this document
   explains the contract but is not executable code.
3. Preserve unrelated flows. A fix to qualification, listing references, inventory lookup,
   contact handling, or email actions must not silently alter the others.
4. Keep LLM semantic judgment separate from deterministic state/tool control.
5. Check `PROMPT_AUDIT.md` for confirmed defects and known current deviations.
6. Check `trailer_fields.py` for the authoritative per-category required questions.
7. Check `src/chatbot/prompts.py` for the Mind's current policy wording.
8. Use focused tests for the changed behavior; live tests are opt-in and call real LLMs.
9. If code and this document disagree, determine whether the code contains a known defect before
   changing this document to match it. Do not turn an accidental implementation bug into policy.

Primary source locations:

- Runtime orchestration and contact/listing/inventory routing: `src/chatbot/service.py`
- LangGraph qualification, reconciliation, search, and email nodes: `src/chatbot/graph.py`
- Mind policy: `src/chatbot/prompts.py`
- Category questions and field guidance: `trailer_fields.py`
- Category mapping: `src/chatbot/categories.py`
- Inventory lookup: `src/chatbot/inventory_matcher.py`
- Pinecone recommendation search: `src/chatbot/tools/pinecone_search.py`
- Persistence: `src/conversation_store.py`
- Confirmed prompt/behavior defects: `PROMPT_AUDIT.md`

---

## 1. Why the chatbot exists

The TrailerPlace chatbot is a conversational sales assistant for helping a customer find a
suitable trailer from TrailerPlace inventory.

Its core responsibilities are:

1. Understand what the customer wants to haul or accomplish.
2. Resolve or recommend a trailer category.
3. Ask only the qualification questions required for that category.
4. Preserve useful requirements across turns and server restarts.
5. Search real inventory without inventing listings, prices, or specifications.
6. Explain which results satisfy the customer's requirements and which are alternatives.
7. Handle follow-up questions about displayed trailers.
8. Notify TrailerPlace when the customer:
   - is interested in a specific listing;
   - asks about financing, trade-ins, service, parts, store information, or human contact;
   - requests a business action that the chatbot cannot perform.

The chatbot is not intended to complete reservations, holds, quotes, invoices, contracts,
appointments, calls, reminders, delivery scheduling, custom work, or other real-world business
commitments. Those requests are escalated to the TrailerPlace team.

The runtime consists of:

- a Streamlit frontend;
- a FastAPI backend;
- a LangGraph recommendation and qualification graph;
- PostgreSQL-backed session and lead persistence;
- Pinecone recommendation search;
- a local structured-inventory matcher;
- email notification tools.

---

## 2. Governing design principles

### 2.1 State is authoritative

The durable session state—not an LLM's memory—is the source of truth. Important persisted fields
include:

- selected category;
- collected and skipped qualification slots;
- Pinecone metadata filters;
- requested non-metadata features;
- current active question and pending questions;
- contact information and contact status;
- latest displayed listing batch;
- all previously displayed listing URLs;
- pending email actions;
- current category cycle and whether its first results have been shown.

An LLM may propose an update, but Python decides whether it is valid and applies it to state.

### 2.2 LLM judgment and deterministic control have different jobs

LLMs are used for semantic tasks:

- understanding intent;
- interpreting vague natural language;
- deciding whether a reply answers an active question;
- recommending categories;
- extracting requirements;
- resolving conversational listing references;
- writing natural replies.

Deterministic code is used for control and safety:

- API validation;
- persistence and idempotency;
- category-cycle resets;
- tool routing;
- required-slot completion checks;
- canonical units and filter keys;
- listing identity validation;
- shown-listing exclusion;
- email queuing;
- fallback behavior.

### 2.3 Never invent inventory or customer requirements

- Listings, prices, URLs, stock data, and specifications must come from inventory data.
- A missing or vague numerical value must not be guessed.
- A non-metadata feature must be explicitly requested by the user.
- A category or make must not be inferred merely from an incidental word.
- Search result language must distinguish confirmed matches from partial or alternative matches.

### 2.4 Contact is optional for trailer help

The initial contact request may ask for name, email, and phone, but the customer may decline and
still use the recommendation flow.

If only part of the contact information is supplied:

- name without email/phone → ask once for email or phone;
- email/phone without name → ask once for name;
- if the user declines or ignores the follow-up → continue with the saved trailer request.

An email tool requires:

- a name; and
- either an email address or a phone number.

If those details are missing when an email action is requested, the action is persisted and the
service asks only for the missing information. Once contact becomes available, the pending action
is delivered and the interrupted qualification flow resumes.

### 2.5 Ask one qualification question at a time

The active slot, displayed question, and queued questions must remain synchronized. A customer
should not be asked again for a field that has been answered or skipped.

If a customer fails to answer the same qualification question twice, the field may be skipped and
the flow proceeds to the next question or search. Counter-questions and email-triggering turns do
not count as failed answers.

---

## 3. Category rules

### 3.1 Canonical categories

The canonical inventory categories are:

- Aluminum
- Car Hauler
- Equipment
- Enclosed
- Utility
- Fiber
- Race Trailer
- Roll Off
- Diesel Tank
- Flatbed
- Dump
- Tilt
- Livestock

`Bumper Pull` and `Gooseneck` are hitch types, not trailer categories.

### 3.2 Category resolution

Category resolution uses three signals:

1. Deterministic synonym matching from `src/chatbot/categories.py`.
2. The Mind LLM's structured category proposal.
3. Category-transition reconciliation before an existing category can change.

A directly named category or configured synonym should resolve as `explicit` with high
confidence. A category inferred from a use case is a recommendation, not an explicit selection.

Examples:

- “I want a dump trailer” → `Dump`, explicit.
- “I need to haul gravel” → recommend `Dump`; do not silently select it.
- “I need something for cars or heavy equipment” → recommend suitable categories and ask the
  customer to choose.
- “I want a utility trailer to haul equipment” → `Utility`; “equipment” is cargo in this
  sentence, not a category switch.

### 3.3 Category changes

While a category is active, it should change only when the customer explicitly switches trailer
types, such as:

- “Actually, switch to Dump.”
- “I want an Equipment trailer instead.”

Incidental category-like cargo words must not change the category. The category-transition LLM
receives the persisted category, proposed category, active slot, latest message, recent context,
and canonical mappings. Medium/high-confidence approval and explicit switch intent are required.
On failure or low confidence, the old category is preserved.

When a category change is approved:

- category-specific slots and skipped fields are cleared;
- category-specific filters and requested features are cleared;
- reusable common filters may require customer confirmation;
- the latest result set and shown-result state for that category cycle are reset;
- required qualification begins for the new category.

### 3.4 Aluminum

Aluminum is a primary category with an underlying `base_category`.

- “Aluminum utility trailer” → category `Aluminum`, base category `Utility`.
- “Aluminum trailer” → ask which underlying type is wanted.
- A base-category answer must not replace `Aluminum` as the primary category.
- Switch away from Aluminum only when the user explicitly rejects or replaces it.

### 3.5 Office/cooldown ambiguity

“Office trailer” and “cooldown trailer” are ambiguous:

- fiber, telecom, or splicing use → `Fiber`;
- general office/jobsite use → `Enclosed`.

The chatbot asks a clarification question before resolving the category.

### 3.6 Known current category limitation

The deterministic resolver currently returns the first synonym match by dictionary order.
Therefore, a sentence containing both a named trailer type and a cargo term can produce a
misleading deterministic hint. The intended rule is:

1. explicit trailer naming term outranks cargo/use terms;
2. within the same tier, the earliest and most specific phrase wins;
3. ambiguous multi-category messages are sent to reconciliation.

This intended ranking has not yet been fully implemented in `categories.py`.

---

## 4. Qualification and field-extraction rules

### 4.1 Category field definitions

Required fields are defined in `trailer_fields.py`:

| Category | Required qualification |
|---|---|
| Aluminum | underlying base category, payload need |
| Car Hauler | vehicle type, vehicle weight, vehicle length |
| Equipment | haul item, haul weight, haul/deck length, hitch type |
| Enclosed | use case, cargo size |
| Utility | haul item, haul weight |
| Fiber | fiber use case |
| Race Trailer | vehicle type, trailer length |
| Roll Off | package scope, bin size |
| Diesel Tank | fuel type, tank capacity |
| Flatbed | haul item, haul weight |
| Dump | haul material, haul weight |
| Tilt | haul item, haul weight |
| Livestock | trailer length only |

Optional questions are asked only when they materially improve matching.

### 4.2 Field types

#### Free-text haul/use fields

Examples include:

- `haul_item`
- `haul_material`
- `vehicle_type`
- `use_case`
- `fiber_use_case`
- `equipment_list`

Store any substantive relevant answer, even if broad:

- “random construction debris”
- “ordinary cars”
- “assorted equipment and supplies”
- “general cargo and occasional work use”

Do not store the answer only when the customer:

- explicitly refuses or skips;
- says they do not know;
- asks a counter-question;
- clearly answers a different field.

A trailer category is not automatically its haul item. “I want an Equipment trailer” does not
mean the customer will haul “Equipment” unless that is explicitly given as cargo.

#### Numeric, measurement, weight, capacity, price, and crew fields

- Accept digits, number words, approximate values, and ranges.
- Convert dimensions to feet and weights to pounds.
- For a range, store the smallest stated value:
  - `15–18 ft` → `15 ft`
  - `5,000–10,000 lbs` → `5,000 lbs`
- A vague answer with no usable numeric value is no preference:
  - “whatever is standard”
  - “somewhere in the normal range”
  - “I am flexible”
- Never invent a number from words such as “several,” “medium,” or “normal.”

#### Width

Width requires an explicit numeric measurement. “Flexible,” “standard,” “whatever fits,” and
similar answers skip the width slot and must not create a `width_ft` filter.

#### Enclosed cargo size

A usable length alone completes `cargo_size`; width and height are optional.

- “About 18 feet long” → `cargo_size="18 ft"`, `length_ft="18 ft"`.
- “18 by 8 feet; height does not matter” →
  `cargo_size="18 ft × 8 ft"`, `length_ft="18 ft"`, `width_ft="8 ft"`.

A current-message extraction safeguard prevents reconciliation from repeating this question when
a usable length was extracted.

#### Hitch and fixed-choice fields

Store a value only when the user selects a recognized choice.

- “Gooseneck” → store `Gooseneck`.
- “Either bumper pull or gooseneck is fine” → no preference; store neither.
- “Whatever is standard” → no preference.

#### Roll Off bin size

Preserve the bin-size answer in `bin_size`. For Pinecone compatibility, map the same numeric value
directly to `length_ft`:

- `15 yd` → `bin_size="15 yd"`, `length_ft="15 ft"`;
- `15–20 yd` → use `15`, not `45`.

This is an inventory-search mapping, not a physical yards-to-feet conversion.

### 4.3 Metadata and non-metadata requirements

Pinecone metadata filters are structured inventory fields:

- category
- length
- width
- height
- payload
- maximum price
- hitch type
- subcategory
- make
- color

Non-metadata features are free-text equipment/configuration requirements not reliably represented
as structured columns, such as:

- sliding gate;
- wireless tarp remote;
- spreader gate;
- slide-in/MAX ramps;
- winch plate;
- recessed D-rings;
- insulated roof;
- finished walls;
- drive-over fenders;
- scissor hoist;
- generator tray.

Only explicit positive requirements should be stored. Do not extract features from:

- negation: “I do not need a tarp”;
- no preference;
- informational questions;
- hypothetical comparisons;
- listing descriptions supplied by the assistant.

Known current limitation: live testing found that the feature extractor can convert negated
features into `"no tarp"` requirements and can invent abstract features from a general
informational question. Those cases are documented in `PROMPT_AUDIT.md`.

### 4.4 Active-question decision trichotomy

Every reply to an active qualification question must be classified as one of:

1. answered active question;
2. no preference/skip for the active field;
3. counter-question or answer to another field.

The no-preference classifier advises the final active-turn reconciler. It does not directly mutate
state. The question adjudicator, field extractor, and reconciler combine their evidence before
state changes are applied.

Counter-question behavior:

1. answer the customer's question;
2. preserve the active slot;
3. naturally resume the same question;
4. do not increment its unanswered count;
5. do not skip or search merely because a counter-question occurred.

Known current limitation: live tests still show disagreement between the no-preference classifier
and adjudicator for “Either A or B,” explicit single-field skips, and nonnumeric phrases such as
“several thousand pounds.” See `PROMPT_AUDIT.md`.

### 4.5 Dynamic width question

The haul-classifier LLM can add `item_or_trailer_width_ft` when the cargo is unusually wide,
large, heavy-duty, or vehicle-like.

It must not add dynamic width for:

- Utility;
- Enclosed;
- Flatbed.

For other categories it may add width when justified by the haul item. A vague width answer is
handled by the normal no-preference rule.

---

## 5. Tool calls and when they are used

### 5.1 Pinecone recommendation search

Purpose: find category-based trailer recommendations using the accumulated qualification state.

Inputs include:

- resolved category;
- collected slots;
- metadata filters;
- active search request text;
- all previously shown URLs.

Search is permitted when:

- the category is resolved; and
- all currently required fields are answered or skipped; and
- first results have not yet been shown in the active category cycle; or the user explicitly asks
  to search/show/refresh/more.

After results have already been shown:

- informational questions are answered without another search;
- a filter update is stored but does not search unless the user also asks to refresh results;
- “show more” searches and excludes every previously shown URL;
- an approved category switch starts a new category cycle and restores one-time automatic search.

The Pinecone tool:

1. builds metadata filters and semantic query text;
2. retrieves inventory candidates;
3. reranks by required dimensions, payload, make/category priorities, and other fit signals;
4. excludes `already_shown_listing_urls`;
5. audits structured and non-metadata requirements;
6. labels listings as full, partial, alternative, or unknown;
7. formats customer-facing listing cards deterministically.

`last_listings` is replaced with the latest displayed batch. `already_shown_listing_urls` is an
accumulated cross-batch set and is never replaced when a new batch is displayed.

### 5.2 Direct inventory lookup

Purpose: answer explicit availability, price, or detail questions about a recognizable inventory
identity without running the category qualification flow.

The current direct-lookup contract requires:

- explicit availability/price/details intent; and
- make + model, or year + make; and
- medium/high LLM confidence after validation.

Examples:

- “Is the Diamond C FMAX available?” → direct lookup.
- “Do you have a 2026 Diamond C?” → direct lookup.
- “Show me Diamond C trailers” → ordinary shopping/recommendation flow unless sufficient direct
  lookup identifiers and intent are present.

Although extraction supports a `stock` field, the current validation prompt explicitly rejects
stock-only direct lookups. Stock/model references to already displayed listings can still be
handled by listing-reference resolution.

The inventory lookup uses the structured local inventory dataset rather than Pinecone. Its returned
batch replaces `last_listings`, but its URLs are added to the accumulated shown-URL exclusion set.
It does not change the current recommendation category merely because the looked-up models belong
to another category.

### 5.3 Listing-reference resolution

Purpose: interpret references to the latest displayed result batch.

This is a two-stage LLM flow:

1. A strict latest-message-only intent gate verifies an explicit reference:
   - ordinal/number: “second,” “#2,” “last”;
   - title/model/stock identifier;
   - demonstrative: “that one,” “the gray one.”
2. A resolver maps the reference to an exact one-based index in current `last_listings`.

Python then verifies that index, title, and URL all match the current batch.

For a confident interest reference:

- skip Mind, field extraction, category changes, and qualification updates;
- send or defer the interested-listing notification.

For a detail reference:

- answer from the resolved listing/inventory context;
- do not send an interest notification.

For an ambiguous reference:

- ask the customer to choose from concise numbered current titles;
- never resolve against an older batch.

Generic messages about financing, quotes, reservations, categories, or trailer uses are not listing
references even if the previous turn logged interest.

### 5.4 Interested-listing email

Trigger: explicit interest in a specific displayed listing.

Effects:

- sends the customer identity, selected listing, and transcript to TrailerPlace;
- promotes the lead to a hard lead;
- stores the item of interest;
- returns a natural confirmation.

If contact is missing, the action is queued and the chatbot asks only for missing contact details.

### 5.5 Non-sales FAQ email

Trigger categories:

- human contact;
- financing;
- trade-in;
- service/parts;
- store information.

The chatbot also answers the customer directly and includes the TrailerPlace phone number. The
email is an internal notification, not the customer-facing answer.

Product-information questions such as “Which trailer types do you carry?” or “What hitch types do
you have?” are answered directly and do not trigger this tool.

### 5.6 Escalation-alert email

Trigger: the customer asks TrailerPlace or the chatbot to perform an unsupported real-world action:

- reserve or hold a trailer;
- create/send a quote, invoice, contract, application, or paperwork;
- call, text, or email later;
- schedule a call, meeting, appointment, delivery, pickup, inspection, or installation;
- send a reminder;
- promise a future time or make a custom arrangement.

Do not escalate:

- “What does it cost?”
- “What are your hours?”
- “Do you offer financing?”
- “How do reservations work?”
- “Stop asking questions.”
- “Show me results.”
- ordinary trailer advice or informational questions.

### 5.7 Results-shown notification

After a non-empty result batch is shown and sufficient contact exists, a silent internal
notification may be sent to TrailerPlace. This is separate from listing-interest email and should
not change the customer conversation.

### 5.8 Email delivery and outbox

In durable mode, email tool calls are captured as events inside the same database transaction as
the chat turn. They are written to an outbox and delivered after the conversation state commits.
This prevents an email from being sent for a turn whose state failed to save and supports safe
retry behavior.

---

## 6. End-to-end flows

### 6.1 Request lifecycle

```text
Streamlit
   │ POST /chat (session_id + turn_id + message)
   ▼
FastAPI
   ▼
Durable turn / idempotency check
   ▼
Restore persisted session
   ▼
Contact extraction and pending-action delivery
   ▼
Latest-listing reference gate / direct inventory lookup
   ▼
Initial optional-contact flow, when applicable
   ▼
Service routing: graph or smalltalk
   ▼
Mind → extraction/adjudication/reconciliation → action
   ├── respond / ask question
   ├── Pinecone search
   ├── interest email
   ├── FAQ email
   └── escalation email
   ▼
Persist state + turn + outbox atomically
   ▼
Deliver outbox and return response
```

### 6.2 New conversation and contact

1. Create or restore a session using `session_id`.
2. Extract name/email/phone from request fields and message text.
3. Save the original trailer request if the initial response must ask for contact.
4. Ask once for missing initial contact information.
5. Classify the next reply as contact details, decline, question, update, or new request.
6. Resume the saved trailer request after contact or refusal.
7. Contact refusal never blocks recommendation/search.

### 6.3 Generic trailer request

Example: “I need a 6×12 trailer.”

1. Extract 6 ft width and 12 ft length.
2. Do not search without a category.
3. Mind returns `ask_trailer_category` with complete natural wording.
4. Persist `awaiting_slot="generic_category_choice"`.
5. The next category answer enters that category's qualification flow.

If the customer has no category preference, generic length and payload questions can be used before
searching broad inventory.

### 6.4 Use-case recommendation

Example: “What is best for hauling cars and furniture?”

1. Do not silently select a category.
2. Mind recommends two or three appropriate canonical categories.
3. Store a pending recommendation.
4. Ask the customer to confirm a category.
5. On confirmation, begin that category's qualification.

### 6.5 Category qualification

1. Resolve category.
2. Extract every usable field already present in the message.
3. Load required fields from `trailer_fields.py`.
4. Run haul classification for lightweight Utility handling and optional dynamic width.
5. Ask the first missing required question.
6. On each answer:
   - run no-preference classification;
   - run active-question adjudication;
   - run general field extraction;
   - reconcile the results;
   - update slots, skipped fields, filters, and requested features.
7. If another required field is missing, ask it.
8. When complete, automatically show first results once for the category cycle.

If the customer supplies all remaining required fields in one reply, every usable field should be
extracted and the graph should search immediately rather than asking stale queued questions.

### 6.6 Counter-question during qualification

1. Detect that the user is asking about the active field rather than answering it.
2. Answer the question.
3. Preserve the active slot and its unanswered count.
4. Rephrase/resume the same qualification question naturally.
5. If the counter-question triggers an email action, send/defer it and then resume the same active
   question.

### 6.7 Search and post-results conversation

1. Search and display the first result batch.
2. Persist that batch as `last_listings`.
3. Add URLs to accumulated exclusions.
4. Mark first results shown for the current category cycle.
5. On later turns:
   - explicit listing reference → listing-reference flow;
   - direct make/model availability request → inventory lookup;
   - “show more” → Pinecone with exclusions;
   - filter-only update → store it without automatic refresh;
   - informational question → answer only;
   - explicit new category → reconcile and start a new cycle.

### 6.8 Session sleep and restore

Each successful durable turn stores a full state snapshot and conversation. The frontend keeps the
session ID in the URL/browser session storage.

When Streamlit reloads:

1. wait for FastAPI health;
2. read the browser/session ID;
3. call `GET /session/{session_id}`;
4. load persisted messages before rendering the conversation;
5. continue with the same category, slots, filters, pending question, shown listings, and pending
   actions.

`turn_id` makes retries idempotent: the same turn returns its prior receipt rather than executing
tools twice.

---

## 7. LLM inventory and responsibilities

Unless overridden by its environment variable, most conversational LLMs use `gpt-4o-mini` at
temperature 0. The Pinecone match auditor defaults to `gpt-5-mini` with minimal reasoning and
strict JSON schema.

### 7.1 Service-layer LLMs

| LLM | Purpose | Output/use |
|---|---|---|
| Contact extractor | Extract name, email, phone, and name confidence | Merged with regex extraction; accepted only when evidence is plausible |
| Contact-prompt reply classifier | Decide whether a post-contact reply supplies/declines contact, asks why, updates the saved request, or starts a new request | Controls whether the saved initial request is resumed |
| Contact bridge generator | Write a natural acknowledgement when returning from contact collection | Prepended to the real trailer response |
| Initial-message preservation classifier | Decide whether the opening message contains non-contact intent worth saving | Intended to prevent loss of the original request |
| Catalogue-overview classifier | Detect unconstrained “show everything” requests | Routes broad catalogue browsing to website guidance |
| Unsupported-business-action router | Detect requests that must reach the graph's escalation rules | Prevents unsupported actions from falling into smalltalk |
| Main route classifier/generator | Decide graph relevance and generate smalltalk answers when graph routing is unnecessary | Service-level conversation routing |
| Confusion detector | Detect repeated/confused customer turns | May trigger escalation outside active qualification |
| Listing-reference intent gate | Using only the latest message, detect an explicit ordinal, identifier, or demonstrative listing reference | Prevents prior interest context from biasing selection |
| Listing-reference resolver | Resolve an approved reference to the exact current listing | Python validates index/title/URL before action |

`ContactPolicyDecision` and `ContactPolicyRewrite` LLM constructors remain defined, but the active
post-decline contact guard is regex-based rather than LLM-based.

### 7.2 Graph-layer LLMs

| LLM | Purpose | How it cooperates |
|---|---|---|
| Mind/planner | Propose category, action, response, field updates, recommendations, FAQ/escalation intent, and selected listing | First semantic planner; proposals are normalized and reconciled before execution |
| Filter extractor | Extract structured dimensions, payload, price, hitch, make, color, subcategory, and slot values | Supplies candidates, not final authority |
| Requested-feature extractor | Extract explicit non-metadata equipment/configuration requirements | Passed through normalization and later match auditing |
| Field-extraction adjudicator | Compare extraction candidates to message/context and reject unsupported values | Prevents unrelated values from entering state |
| Non-recommendation turn classifier | Distinguish FAQ, escalation, catalogue, ordinary response, active answer, and shopping flow | Can override only confident FAQ/escalation cases; does not own ordinary category wording |
| Active-question adjudicator | Decide answer/no-preference/counter-question/search-now/skip-all and produce active-slot updates | Core interpretation of an answer to the current question |
| No-preference classifier | Independently judge whether the active field should be unrestricted | Advisory input to active-turn reconciliation |
| Active-turn reconciler | Combine adjudicator, extractor, no-preference result, active slot, and context into the final turn decision | Resolves disagreements before mutation |
| Haul classifier | Identify actual haul item, lightweight Utility cargo, and dynamic-width need | Alters effective required questions |
| Category-transition reconciler | Approve or reject a proposed change to an existing category | Protects category from incidental cargo words |
| Make verifier | Verify a deterministic make candidate is explicitly intended as a manufacturer | Rejects collisions such as “general cargo” → Cargo Craft |
| Category-filter confirmation LLM | Interpret keep/discard/change answers for filters carried across category changes | Applies per-field confirmation |
| Office-trailer clarification LLM | Resolve Fiber vs Enclosed and detect email action during the clarification | Keeps ambiguous office terminology safe |
| Pinecone match auditor | Verify every returned listing against structured and non-metadata requirements and write match framing | Does not retrieve; audits retrieved facts |

### 7.3 Inventory-lookup LLMs

| LLM | Purpose |
|---|---|
| Trailer-query extractor | Extract year, make, model, stock number, requested features, price/availability/details intent, and direct-lookup confidence |
| Inventory response generator | Write the customer-facing introduction/answer around deterministic inventory matches |
| Inventory feature-framing LLM | Explain whether a direct lookup's feature configuration is full, partial, or unavailable |

Deterministic inventory matching and scoring remain authoritative. The LLM does not invent rows.

### 7.4 Email reply composer

After an email tool succeeds, this LLM writes a natural customer-facing response. It receives:

- the email purpose;
- latest message;
- planner reply;
- recent context;
- fallback text;
- the still-active question, when one exists.

It must not mention internal email/tool mechanics. When an active question exists, it should
naturally resume it after addressing the customer's request.

### 7.5 How the LLMs work together on an active Q&A turn

```text
Mind proposal
     │
     ├── general field extractor
     ├── requested-feature extractor
     ├── active-question adjudicator
     ├── no-preference classifier
     └── haul classifier
              │
              ▼
       active-turn reconciler
              │
              ├── category-transition reconciler, if category differs
              ├── make verifier, if a deterministic make candidate exists
              └── deterministic state invariants/required-slot completion
                              │
                              ▼
                     response, next question,
                     search, or email tool
```

No single classifier should independently overwrite category, skip state, or search state. The
reconciler and deterministic invariants exist because small LLMs can disagree.

---

## 8. Routing precedence

The intended practical precedence is:

1. Restore and validate session/turn.
2. Extract contact and deliver eligible pending actions.
3. Resolve explicit references to the latest listing batch.
4. Validate direct inventory lookup.
5. Complete initial optional-contact handling.
6. Handle active qualification reply.
7. Preserve explicit category/field information from the turn.
8. Supported FAQ email action.
9. Unsupported business-action escalation.
10. Broad catalogue redirect.
11. Category recommendation/selection.
12. Qualification question.
13. First automatic search or explicit subsequent search.
14. Informational response/smalltalk.

The code currently has several service-level classifiers plus graph-level routing. Where their
definitions overlap, Python call order decides which one runs first. The long-term intended design
is one documented routing contract with sub-classifiers acting only as evidence, not independent
competing routers.

---

## 9. Persistence and data ownership

### PostgreSQL

Stores:

- soft/hard leads and contact details;
- conversation transcript;
- complete session state snapshot;
- idempotent turn receipts;
- email outbox events.

Conversation creation uses PostgreSQL `ON CONFLICT DO NOTHING` and reloads the existing row, so
concurrent first requests for one session do not crash with a uniqueness error.

### Pinecone

Stores embedded trailer inventory for recommendation retrieval. Metadata is generated during the
Excel ingest pipeline and used for structured filtering/reranking.

### Local inventory dataset

Used by direct make/model/year inventory lookup. It supports deterministic matching and details
without relying on semantic Pinecone recommendation flow.

### Streamlit/browser

Owns display state and carries the session ID. It does not own the authoritative conversation
state; it restores that state from FastAPI/PostgreSQL.

---

## 10. Non-negotiable behavioral checklist

Before accepting a behavior change, verify:

- A named category is not overwritten by its cargo.
- A category switch is explicit and reconciled.
- Free-text haul/use answers are retained unless explicitly skipped or unrelated.
- Vague numeric answers do not create invented numbers.
- Ranges use their smallest value.
- “Either A or B” does not select A.
- Skipped width creates no width filter.
- Positive non-metadata features are preserved; negated/informational features are not.
- Counter-questions preserve and resume the active slot.
- Email interruptions preserve qualification state.
- First completed qualification searches once per category cycle.
- Post-results informational questions do not trigger more results.
- “Show more” retains filters and excludes all shown URLs.
- Inventory lookup replaces only `last_listings`, not accumulated shown URLs or category state.
- Ordinal listing references use only the latest displayed batch.
- Field extraction is bypassed on confident listing-selection turns.
- Contact refusal never blocks trailer help.
- Missing contact defers an email tool without losing the requested action.
- Every durable turn is restorable and idempotent.
- Customer-facing listing facts always come from inventory data.
