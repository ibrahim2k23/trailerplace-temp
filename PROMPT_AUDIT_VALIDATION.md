# PROMPT_AUDIT.md — Validation Against `langgraph_rules_vs_excel.md`

This cross-checks each audit finding against the behavioral spec
(`langgraph_rules_vs_excel.md`, which describes the Excel workbook + intended chatbot
behavior). Each finding is marked:

- ✅ **VALIDATED** — the spec confirms the audited behavior is a real deviation/problem.
- ⚠️ **PARTIAL / NUANCED** — the spec supports part of the concern but qualifies it.
- ❌ **CONTRADICTED** — the spec shows the flagged behavior is actually *intended*; the finding needs correction.
- ➖ **NEUTRAL (out of spec scope)** — internal engineering/prompt-hygiene concern the spec doesn't govern; the finding stands on its own but isn't a spec violation.

The spec is written in business language and does **not** cover internal implementation
(temperature, structured-output method, context assembly, guardrail architecture), so several
findings land NEUTRAL — that does not mean they are wrong, only that the spec neither confirms
nor denies them.

---

## Findings that the spec STRENGTHENS

### #1 & #2 — Category resolver / LLM override (Tilt→Equipment) — ✅ VALIDATED (strengthened)
The spec's whole Section 1.1 premise is *"first understand what kind of trailer the customer
is asking about."* The customer naming "tilt trailer" is exactly that. Decisive evidence in
the mapping table (spec §1.2, lines 88, 102, 84):

| Category | Spec wording | Code `_SYNONYMS` |
|---|---|---|
| Equipment | "**tractor trailer**", "**skid steer trailer**", "mini excavator trailer" | bare `"tractor"`, `"skid steer"`, `"mini ex"` |
| Utility | "**motorcycle trailer**", "**ATV trailer**", "**lawnmower trailer**" | bare `"motorcycle"`, `"atv"`, `"lawnmower"` |
| Dump | "scissor lift **dump**", "telescopic **dump**", "front lift **dump**" | bare `"scissor lift"`, `"telescopic"`, `"front lift"` |

The spec's terms are **phrases that name the trailer type**; the code stores **bare cargo
words** as category signals. So "haul **tractor**" trips the Equipment synonym even though the
customer named Tilt — a direct consequence of the code being broader than the spec. This
confirms both the resolver ranking bug (#2) and that the override propagating it (#1) violates
the spec's "understand the trailer type first" intent. **The recommended naming-term vs
cargo-term split is essentially "match the spec's phrasing."**

### #3 — Catalogue ≠ supported categories; Welding unreachable — ✅ VALIDATED
The spec **explicitly lists Welding** as a category (§1.2, lines 103–104: *"Welding — common
wording: welding trailer, welder trailer"*), and also lists Fiber, Race Trailer, and Diesel
Tank. The code omits Welding from `CANONICAL_CATEGORIES` (so "welding trailer" can't resolve —
a straight spec violation) and the `KNOWLEDGE` prompt omits Fiber/Race/Diesel from what it
advertises. Spec-confirmed.

### #8 — Contact-reply fallback drops the saved request — ✅ VALIDATED
Spec §1.1: *"instead of starting the whole conversation over"*; §2.1: early info is saved and
*"once the customer names the category, the chatbot carries that information into the correct
flow."* Dropping the saved request on an LLM error violates that carry-forward intent.

### #18 — Turn-1 contact detour deprioritizes the request — ✅ VALIDATED (severity likely understated)
This is the strongest spec catch. The spec's **canonical example conversation** (§6, lines
400–405) is:
```
Customer: I need a utility trailer.
Bot:      What will you be hauling on the utility trailer?
```
The bot answers the trailer request **immediately**, with **no contact request**. The spec's
only "contact-first" rule (§2.3, lines 231–243) is scoped explicitly to **email actions**
(interest / FAQ / escalation) — *not* the opening turn. The implemented behavior (respond to a
specific first request with a generic "can I get your name, email, phone?" and defer the
request) is **not in the spec and contradicts the documented example flow.** Recommend raising
this from P2 toward P1.

---

## Findings the spec PARTIALLY supports

### #6 — Auditor "exact match on count==6" + dimension-blind — ⚠️ PARTIAL
- The over-claim concern is validated: the spec treats **length** as a primary search detail
  (§ "How The Search Works", lines 384–389: *"leans most on … length, if clearly given"*), so
  a customer-facing "exact match for exactly what you asked" that ignores length conflicts with
  the spec's emphasis.
- **But** the dimension-blindness has spec rationale: the spec justifies leaning on few fields
  *"because not every trailer record has every field filled out well enough to rely on it"*
  (lines 381–383). So stripping dimensions from the *auditor* is defensible; the real issue is
  narrower — the **absolute wording** ("exact match") plus the **magic `==6`** trigger. Keep the
  finding, but reframe: fix the over-claim wording and the hard-coded count, not the
  dimension-blindness itself.

### #15 — Haul-classifier retry pressures item invention — ⚠️ PARTIAL
Spec §1.1 / §2.1 favor natural handling and no-preference over forcing answers, which supports
the concern, but the spec doesn't describe the retry mechanism. Valid, lightly spec-backed.

### #17 — Confusion thresholds / escalation calibration — ⚠️ PARTIAL
Spec §2.3 explicitly endorses confusion→handoff, but only when the customer *"seems stuck,
confused, or frustrated **repeatedly**"* (line 260). "Repeatedly" supports the audit's concern
about premature single-signal escalation. The specific numeric thresholds are out of spec scope.

---

## Finding the spec CONTRADICTS (needs correction)

### #20 — "Make→category inference is a problem" — ❌ CONTRADICTED (correct the finding)
The spec **wants** certain brand→category mappings. §1.2 (line 94) lists **"Galyean, Star
trailer, Calico trailer"** as Livestock wording, and line 108 gives *"cattle leads to
Livestock."* So resolving "Galyean" → Livestock is **correct per spec**, not a hallucination.
My example in #20 (Galyean → Livestock treated as a bug) is wrong.

The genuine issue is the **opposite and still worth flagging**: the planner prompt says *"do
NOT infer category from make"* (`PRODUCT_INFO_RULES`) while the spec and the code's
`make_inventory`/`make_category_options` **do** map brands to categories. That is a
**prompt-vs-spec contradiction**, and the fix is to align the prompt with the spec (brands like
Galyean *are* category signals; general brands are for within-category reranking per spec §2.2),
not to suppress make→category inference. Rewrite #20 accordingly.

---

## Findings NEUTRAL to the spec (valid engineering, spec silent)

These are real prompt/reliability concerns, but the business spec neither confirms nor denies
them. They should be judged on engineering merit, not spec compliance:

| # | Finding | Note vs spec |
|---|---|---|
| #4 | Width-exclusion list: prompt `{Utility,Enclosed,Flatbed}` vs code `{+livestock,aluminum,dump}` | Spec only fixes **Flatbed=8 ft default** (§2.2/§5) and "width in heavy-duty cases"; exact per-category exclusion isn't specified. Internal inconsistency stands; notably the **classifier prompt is closer to spec** than the code's wider list. |
| #5 | Planner context bloat | Implementation detail; spec silent. |
| #7 | Duplicated planner prompt blocks | Spec silent. |
| #9 | Overlapping routing classifiers | Spec §2.4 **endorses** separate helpers, so their existence is intended; only the hidden-precedence concern is the audit's point — soften accordingly. |
| #10 | LLM contact-policy validate+rewrite guardrail | Spec silent (contact-declined policy isn't described). |
| #11 | `function_calling` weak enums | Implementation detail. |
| #12 | Nonzero temperature on customer text | Implementation detail. |
| #13 | FAQ rewrite may corrupt phone number | Spec silent on the reply text / phone number, but the reliability concern is legitimate. |
| #14 | Feature matching by enumerated examples | Spec silent; it only says search leans on category/make/hitch/length. |
| #16 | Prohibition-heavy prompting | Prompt-engineering craft; spec silent. |
| #19 | Answer/no-pref/counter trichotomy split across 3 sites | Spec §2.1 supports simple no-preference handling; the multi-site inconsistency is internal. |
| #21 | Dead `sides_gate_storage` slot / static follow-up / unused email ctx | Spec's Utility flow (§1.1, §6) never mentions sides/gate/storage, so removing the dead slot is spec-consistent. |

---

## Validation summary

| Verdict | Count | Findings |
|---|---|---|
| ✅ Validated (spec confirms) | 4 | #1, #2, #3, #8, #18 (5 incl. the paired #1/#2) |
| ⚠️ Partial / nuanced | 3 | #6, #15, #17 |
| ❌ Contradicted (correct it) | 1 | #20 |
| ➖ Neutral (spec silent; still valid) | 13 | #4, #5, #7, #9, #10, #11, #12, #13, #14, #16, #19, #21 |

### Net conclusions
1. **The two P0s (#1/#2) and #3, #8, #18 are confirmed real spec violations** — with the spec's
   own category wording ("tractor **trailer**", not "tractor") and its opening-turn example
   giving independent evidence. #18 (contact-first detour) is arguably under-rated.
2. **#20 must be corrected** — brand→category mapping is spec-intended (Galyean→Livestock); the
   real defect is the prompt rule that contradicts the spec.
3. **#6 should be reframed** — dimension-blindness is spec-justified by imperfect inventory; the
   fixable part is the absolute "exact match" wording and the magic `==6` trigger.
4. **The remaining 13 are legitimate but out of the spec's scope** — engineering/reliability
   improvements, not spec compliance issues. #9 should be softened (spec endorses multiple helpers).
5. **No audit finding was proven false except #20's framing.** The qualification flows the audit
   did *not* flag (Utility/Equipment/Dump/Livestock question order, no-preference handling,
   lightweight-utility weight skip, width-question wording) all match the spec — good corroboration
   that the audit focused on the right problem areas.
