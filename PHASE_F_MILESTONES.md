# Phase F Milestones

Working plan for tackling **Stage F** of the refactor (see `~/.claude/plans/refactor-prompt-gentle-globe.md` → "Stage F — Deferred"). Stages A–E are complete and committed on branch `v4.4`; F is the deferred bucket: items that need a **product decision** or are **standalone bigger bets**, not mechanical consolidation.

Guiding rules (carried over from A–E):
- One milestone per commit minimum; each verified before moving on.
- Verification vector is the live conversation harness: `RUN_LIVE=1 TRAILERPLACE_PERSIST_CHATS=0 LANGCHAIN_TRACING_V2=false uv run pytest tests/test_live_conversations.py` (expect 6/6, ~3 min). Offline pytest suite is stale — do not gate on it.
- Feature-gating discipline: missing OPENAI/DB/SMTP keys must never crash.
- Surgical edits; comment the *why* on anything touched.

Legend — **Type**: `ENG` = pure engineering (no decision needed) · `DECISION` = needs a product/architecture call first · `PROJECT` = multi-session design effort.
Ordering below is by recommended execution order (risk-adjusted, dependency-aware), not by plan number.

---

## Milestone F1 — Outbox retry scheduler + `claimed_at` (Type: ENG) ⭐ do first

**Why first:** most self-contained, highest value-to-risk, zero change to conversation behavior. Pure reliability hardening of the durable email path.

**Problem (from audit §1F):** `ChatbotOutbox.claimed_at` is never set; the outbox drain is best-effort with no retry scheduler, so a failed post-commit send is lost.

**Scope:**
1. On drain start, stamp `claimed_at` (claim rows so concurrent drainers don't double-send).
2. Add a bounded retry: unclaimed/failed rows past a backoff window get re-attempted; cap attempts, record last error.
3. Reclaim stale claims (`claimed_at` older than a lease window) so a crashed drainer's rows aren't orphaned.

**Touch points:** `src/conversation_store.py` (outbox drain), `src/db_models.py` (`ChatbotOutbox` — confirm/add `claimed_at`, `attempts`, `last_error`), possibly an Alembic migration if columns are added.

**Decisions needed:** none (pick sane defaults: lease ~5 min, max ~5 attempts, exponential backoff — call these out in the commit).

**Verification:** unit test for claim/reclaim/backoff logic (offline, no live send); manual: force a send failure, confirm row is retried not lost. Live harness unaffected but run once as a guard.

**Estimated size:** small–medium, 1 commit (+1 migration commit if schema changes).

---

## Milestone F2 — Unify the two listing-card renderers (Type: ENG)

**Why second:** mechanical but customer-facing, so it needs careful before/after output diffing.

**Problem (audit §1C/§1F):** two layouts for the same concept — `formatting.format_listing_results` vs `inventory_matcher._format_listing_block`. Divergent field sets/wording.

**Scope:** extract one shared card component (fields, ordering, why-line/blurb fallback) into `formatting.py`; have `inventory_matcher` call it. Preserve each call site's current *inputs*; only unify the *rendering*.

**Touch points:** `src/chatbot/formatting.py`, `src/chatbot/inventory_matcher.py` (`_format_listing_block` ~983–1020).

**Decisions needed:** if the two layouts differ in *visible* output, pick the canonical one — **flag any user-visible change** for review before committing.

**Verification:** `tests/test_listing_formatting.py` + a golden snapshot of both call paths' output before/after; live harness (search conversation renders cards) 6/6.

**Estimated size:** medium, 1 commit.

---

## Milestone F3 — gpt-5-mini for planner (G1) + reconciler (G8) (Type: DECISION → then ENG)

**Why here:** the config plumbing is already prepared (central `llm.py` factory + per-role env vars from Stage E2); the remaining work is an **evaluation**, not code.

**Problem (audit §1G):** the two highest-stakes decisions run on the weakest model (gpt-4o-mini) with the longest prompts. Auditor already proves gpt-5-mini + json_schema works.

**Decision required from product/owner before executing:**
- Define "better": what does the A/B measure? (correct action selection, category-transition accuracy, fewer coercion retries, reconciler agreement with human judgment on a labeled set?)
- Cost tolerance: gpt-5-mini on every turn's planner + reconciler is a material spend increase — acceptable?

**Scope once decided:**
1. ✅ **DONE (enabling plumbing).** `_mind_llm` now takes `model_env="MIND_MODEL"`; `_active_turn_reconciler_llm` already had `ACTIVE_TURN_RECONCILER_MODEL`. Both flip to gpt-5-mini via config alone; default unchanged (OPENAI_MODEL → gpt-4o-mini). Verified offline: default resolves gpt-4o-mini, env override resolves gpt-5-mini for both.
2. ❌ **BLOCKED — cannot use `method="json_schema"` as-is.** `MindDecision` and `QuestionTurnDecision` carry free-form `dict[str, Any]` payloads (`slots_collected_update`, `metadata_filters_update`, plus `category_recommendations: list[dict]`). OpenAI strict json_schema requires every object property enumerated with `additionalProperties:false` — free-form dicts are not representable, so strict mode rejects these schemas at invoke (this is why only the fully-enumerated auditor uses json_schema today). Enabling it needs a **typed-slot schema redesign** (fixed slot key set / typed slot objects), which is behavior-affecting → belongs behind this milestone's decision gate, not a mechanical step. Reconsider as part of the A/B work.
3. Run the A/B on a captured conversation set; flip the default (set `MIND_MODEL`/`ACTIVE_TURN_RECONCILER_MODEL=gpt-5-mini`) only if the metric clears the bar.

**Verification:** the A/B itself + live harness 6/6 on the new model. (Enabling plumbing already verified offline; no live run needed since default behavior is unchanged.)

**Estimated size:** small code / large eval effort. Steps 2–3 gated on the decision above; step 1 done.

---

> **UPDATE (F3 partial — DONE):** The **reconciler (G8)** now runs on **gpt-5-mini** by
> default (`_active_turn_reconciler_llm`): dedicated resolver skips `OPENAI_MODEL`,
> temperature left unset for reasoning models, `function_calling` kept (json_schema still
> blocked). Override with `ACTIVE_TURN_RECONCILER_MODEL`. Verified live: full active-Q&A
> flow incl. counter-question classification, 0 errors. The **planner (G1)** stays on
> gpt-4o-mini (flip via `MIND_MODEL`); a broader A/B is still the open piece.

## Milestone F4 — In-memory session locking (or drop the in-memory path) (Type: DECISION → then ENG) — ✅ CLOSED (option A)

**Closed: no change needed.** Decided against dropping the in-memory path. Finding: in
production (persistence enabled) every turn already goes through the durable path, which
serializes per-session via the Postgres advisory lock in `durable_turn` *and* writes
`_sessions` under `_lock` — so the race can't occur in production. The in-memory branch
(`handle_chat` early-return when `not persistence_enabled()`) only runs DB-less, i.e. in
tests and local dev (the whole suite + live harness run `TRAILERPLACE_PERSIST_CHATS=0`).
Deleting it would break all of that to fix a race that production doesn't have. Left as-is.

<details><summary>Original F4 write-up</summary>

**Why here:** correctness-relevant but the *right* fix depends on deployment topology — an architecture call.

**Problem (audit §1F reliability):** session dicts are mutated **outside** the `_sessions` lock on the in-memory path; only the durable path serializes via a Postgres advisory lock. Concurrent requests to the same session can race in memory.

**Decision required:**
- Is a DB always present in every deployment target? If yes → **drop the in-memory path** (simplest, removes the whole class of races). If no → **add per-session locking** to the in-memory path mirroring the advisory-lock discipline.

**Scope once decided:** either (a) delete the in-memory branch and require the durable wrapper, or (b) wrap in-memory session read-modify-write in a per-session `threading.Lock`.

**Touch points:** `src/chatbot/service.py` (`_sessions`, `handle_chat` / `_handle_chat_in_memory`).

**Verification:** concurrency test firing two simultaneous requests at one session id; live harness 6/6.

**Estimated size:** small (drop path) or medium (locking). Gated on the decision.

</details>

---

## Milestone F5 — Confusion-detection redesign (Type: DECISION → then ENG) — ⏸️ DEFERRED

**Deferred (not now).** Not a bug — current 0–100 confusion score works; this is a
fragility cleanup. Revisit if/when escalation misfires become a real problem. When picked
up, decide: switch to repeat-counting + pick N (escalate after N repeats).

**Why here:** changes user-facing escalation behavior, so it needs sign-off; lower urgency than the above.

**Problem (audit §1C/§1E):** the S9 confusion score is a calibrated 0–100 magic number double-encoded in both prompt (threshold 85) and Python. Brittle and hard to reason about.

**Decision required:** approve replacing the 0–100 score with an **ordinal enum** (e.g. `none|mild|high`) + **deterministic repeat-counting** (N identical/unresolved turns → escalate), and agree on N and the enum→action mapping.

**Scope once decided:** S9 prompt returns ordinal signals only; Python owns the threshold + repeat counter (single encoding). Remove the 85 from the prompt.

**Touch points:** `src/chatbot/service.py` (S9 confusion detector + its Python re-check).

**Verification:** `tests/test_chat_service_confusion_escalation.py` updated to the new contract; live harness 6/6.

**Estimated size:** medium. Gated on the decision.

---

## Milestone F6 — Feature ontology for the match auditor (Type: PROJECT) — ❌ WON'T-DO

**Closed 2026-07-08.** A feature ontology needs both sides populated: a canonical feature vocabulary *and* each listing tagged with the feature ids it actually has. **There is no knowledge base of per-listing features** — the inventory (Excel) doesn't carry structured feature data, so the deterministic set-membership check the ontology enables has nothing to validate against. Building it would just relocate the fuzzy match from the user's text to the listing's text without gaining ground truth. Not worth doing unless/until per-listing feature data exists.

**Original problem (audit §2.1/§1D):** requested features are free-text strings; the auditor matches them fuzzily against listing text. A canonical feature-id ontology would make matching deterministic and auditable — but only given per-listing feature ids, which we don't have.

---

## Milestone F7 — Rolling summarization for very long threads (Type: PROJECT, likely WON'T-DO)

**Why last / candidate to drop:** the plan itself deferred this because the **durable snapshot already preserves the semantically-important memory** (slots/filters). Summarization adds per-turn LLM cost for marginal gain here.

**Scope if pursued:** a rolling summary of old turns injected into LLM contexts once history exceeds a cap, replacing the ad-hoc `[-8:]`/`[-6:]` windows (already unified into `compact_recent_messages` in Stage D4).

**Decision required:** confirm there's an actual failure mode (very long threads degrading answers) worth the cost — otherwise **close as won't-do**.

**Estimated size:** large. Recommend leaving closed unless a concrete long-thread problem appears.

---

## Suggested sequencing

```
F1 (outbox retry)      ── do now, no blockers ─────────────► commit
F2 (card renderer)     ── do now, watch visible output ────► commit
        │
        ▼  (need your decisions before these)
F3 (gpt-5-mini A/B)    ── decision: metric + cost
F4 (session locking)   ── decision: is DB always present?
F5 (confusion redesign)── decision: ordinal enum + N
        │
        ▼  (own projects, own plans)
F6 (feature ontology)  ── WON'T-DO (no per-listing feature knowledge base)
F7 (summarization)     ── likely won't-do
```

**Immediate next action:** F1. Everything above F3 is unblocked engineering; F3–F5 are parked pending the three decisions; F6–F7 are separate projects.

## Open decisions checklist (for the owner)
- [x] **F3:** ~~reconciler on gpt-5-mini~~ **DONE** (reconciler flipped, verified live). Planner still on gpt-4o-mini pending a broader A/B — open question: is the per-turn cost of also flipping the planner worth it?
- [x] **F4:** ~~Is a Postgres DB guaranteed in every deployment?~~ **Closed (option A)** — production already race-safe via the durable advisory lock; in-memory path kept for tests/dev.
- [~] **F5:** Deferred — revisit only if escalation misfires become a real problem.
- [x] **F6:** ~~Commit to building a canonical feature ontology?~~ **Closed won't-do** — no per-listing feature knowledge base to validate against.
- [ ] **F7:** Is there a real long-thread degradation worth summarization's cost, or close as won't-do?
