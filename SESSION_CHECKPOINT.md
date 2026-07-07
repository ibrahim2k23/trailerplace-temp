# Session Checkpoint — TrailerPlace Chatbot Refactor

**Branch:** `v4.4`  ·  **Plan file:** `~/.claude/plans/refactor-prompt-gentle-globe.md`
**Scope:** Phase-3 implementation of the audit/refactor plan (task.md). Prompt/flow correctness, hallucination guardrails, code health.

---

## TL;DR
Stages A (subset), B, C, D, and E1/E2/E3 are done, verified, and committed across 8 commits. Every change was verified with **deterministic** unit/offline tests (the integration suite is unreliable here — see DB note). Only **E7** (splitting the 1,392-line `_apply_mind_node`) remains from the plan, deliberately deferred.

---

## Commits made this session (oldest → newest)
| Commit | What |
|---|---|
| `0ac8dc7` | Stage A (hygiene: auditor log fix, dead constants) + **B** (P0 tiered category resolver + advisory hint) + **C** core (C1 auditor count, C3 phone injection, C4 haul retry, C6 catalogue single-source) + **D** (D2 contact-reply fallback, D5 temperatures) |
| `200df2a` | C2 (blurb validation parity), C5 (feature negation/informational filter), D3 (counter-question counter — *later reverted*) |
| `a589ccb` | **Revert D3** — counter-questions/email-triggers/no-answer all DO increment; skip after 2 unanswered (per clarified intent) |
| `835e4a5` | Freeze qualification counter while collecting contact for a deferred email (trigger turn increments once; contact-collection turns don't) |
| `7cd92ae` | D1 (deterministic turn-1 save gate, −1 LLM call) + D4 (pruned planner context, `constants.py` window helpers) |
| `2600eba` | Stage E: E3 (dedup planner prompt blocks) + E1 (single-source dynamic-width exclusions) |
| `6411c31` | Stage E2: create `src/chatbot/llm.py` (`make_llm`/`safe_invoke`/`resolve_model`) + migrate 5 representative factories |
| `cbeaf0e` | Stage E2: migrate ALL remaining LLM factories to `make_llm` (ChatOpenAI now imported only in `llm.py`; −76 lines) |

---

## What each change does (for quick recall)
- **B (P0):** `categories.py` now splits synonyms into `_NAMING_TERMS` vs `_CARGO_TERMS`; an explicitly named type outranks a cargo word. "tilt trailer to haul a tractor" → **Tilt** (was Equipment). The deterministic hint is now advisory in `graph.py:_mind_node` (naming-tier wins on disagreement; else keep LLM at medium confidence).
- **C1:** auditor Scenario C is count-relative (`len(listings)`), not magic `6`; absolute "exactly" wording softened (auditor is dimension-blind).
- **C2:** `_safe_sales_blurb` now also rejects "strong/best/closest match" overclaims on non-full listings.
- **C3:** `email_reply.py` deterministically verifies/repairs the dealership phone + website (constants + regex).
- **C4:** haul-classifier retry reframed so `matched_item=null` is correct; deterministic flag-clear when no cargo.
- **C5:** deterministic post-filter drops negated ("no tarp") and ungrounded-informational features.
- **C6:** advertised catalogue derived from `CANONICAL_CATEGORIES` (was 10 of 13); Welding spec annotated non-canonical.
- **D1:** turn-1 save gate is deterministic (greeting-aware), no LLM. Opener wording intentionally unchanged (test-pinned).
- **D4:** `constants.py` + `compact_recent_messages`/`compact_listings`; planner context pruned (dropped `listing_model_fields`, projected listings, 500-char cap).
- **Contact counter-freeze:** `suppress_active_question_progress` flag threaded service→graph.
- **E1:** `DYNAMIC_WIDTH_EXCLUDED_CATEGORIES` in `constants.py` drives the classifier prompt, its fallback, and the graph gate (fixes prompt/code 3-vs-6 mismatch).
- **E2:** `src/chatbot/llm.py` central factory; all ~26 LLM sites migrated behavior-preserving.

## New files
- `src/chatbot/constants.py` — window sizing, `compact_recent_messages`, `compact_listings`, `DYNAMIC_WIDTH_EXCLUDED_CATEGORIES`.
- `src/chatbot/llm.py` — `resolve_model`, `make_llm`, `safe_invoke`.

---

## Remaining tasks
1. **E7 (deferred, largest):** split `_apply_mind_node` (~1,392 lines) into a `graph/apply_mind/` package + a single `build_state_return()` helper. Pure refactor of the most complex function — do **only** with a working behavioral baseline.
2. **Stale-string test refresh:** a handful of assertions drifted from code (e.g. `test_result_interest_followup_is_deterministic` expects `"Want to compare any of these side by side?"`, code returns `"Do any of these trailers interest you?"`).
3. **Gate live-LLM tests behind an env flag** so CI/offline runs are clean (`test_live_prompt_audit_issues.py`, `test_live_vague_qna_conversations.py` already are; some `test_graph_metadata_extraction`/`test_optional_contact_flow` cases make real calls).
4. **Optional E-items not done:** E5 (merge routing classifiers S5/S6/S7 into one call), E6 (parallelize reconciler inputs, merge no-preference into adjudicator, fold feature extractor into search), E8 (async email on in-memory path, cache `_valid_make_map`, single Excel load).

## Open decisions (waiting on you)
- **Turn-1 acknowledgement:** change the opener to acknowledge the request ("Happy to help you find the right trailer…")? This needs updating a test that pins the current wording. (Left unchanged for now.)

---

## ⚠️ Environment note (important for tomorrow)
Your `.env` sets `TRAILERPLACE_PERSIST_CHATS=1` against a **real Postgres**, so integration tests hit the DB and crash on `ForeignKeyViolation` (tests mock `create_or_get_soft_lead` to return a fake `lead_id` without inserting the lead). **Always run tests with persistence off:**

```
TRAILERPLACE_PERSIST_CHATS=0 uv run pytest -q
```

Baseline with that flag: **74 failed / 254 passed / 66 skipped** (317s). The 74 are pre-existing flaky-live-LLM + stale-string tests, NOT regressions from this session. (With persistence on it's 154 failed / 594s.)

---

## Verified facts to trust
- All deterministic offline tests pass (77/77 in the retrieval/formatting/classifier set).
- `make_llm` reproduces exact model resolution: site-env → `OPENAI_MODEL` → default.
- P0 category cases 7/7 correct; C3 phone-repair 4/4; C5 negation/grounding all correct.

## Assumptions flagged (from the plan's "flag, don't decide" rule)
- B: naming-vs-cargo term classification (no terms added/removed).
- C6: Welding spec annotated, not deleted.
- D1: kept the turn-1 contact ask; opener wording unchanged.
- Audit's "dead code" claims were largely wrong (legacy extraction stack, contact-policy LLMs, `GraphEmailSender`, `_result_interest_followup_text` are all tested/structural) — NOT deleted.

---

## 👉 Exact request to give tomorrow
Pick one:

**To continue the refactor (recommended — start E7 with a real baseline):**
> First run `TRAILERPLACE_PERSIST_CHATS=0 uv run pytest -q` and save the failing-test list as the baseline. Then refresh the stale-string test assertions so the deterministic suite is clean, and gate the live-LLM tests behind a `RUN_LIVE=1` env flag. Once the offline suite is green-except-live, take on E7: split `_apply_mind_node` into a `graph/apply_mind/` package with a single `build_state_return()` helper, preserving call order exactly, and verify with `TRAILERPLACE_PERSIST_CHATS=0 uv run pytest -q`.

**Or, to do the cheaper optimization items first:**
> With `TRAILERPLACE_PERSIST_CHATS=0`, take on E5 (merge the S5/S6/S7 routing classifiers into one structured router call) and E6 (parallelize the reconciler's independent LLM inputs and fold the feature extractor into the search node), verifying each against the offline baseline.
