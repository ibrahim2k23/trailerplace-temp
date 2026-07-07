# Refactor Prompt — Trailer Suggestion Chatbot

You are refactoring an existing production chatbot codebase. The chatbot answers user queries and suggests trailers. It uses two LLMs: **GPT-4o-mini** (handling user query responses) and **GPT-5-mini** (a second role in the pipeline — clarify its exact function from the code, e.g. reasoning/routing/extraction). I'll share the codebase with you. Do not rewrite anything yet — follow the phases below in order.

## Known problems to solve
1. **Hallucinations** — the bot sometimes states things about trailers/inventory/specs that aren't grounded in real data.
2. **Unnatural conversation flow** — responses feel scripted, disjointed, or don't track context/turn history well.
3. **Prompt sprawl** — many prompts spread across the codebase, likely inconsistent in structure, tone, and instructions.
4. **General code health** — needs modern best practices, cleaner separation of concerns, and performance optimization.

## Phase 1 — Audit (do this first, output only, no code changes)
Read through the full codebase and produce a written audit covering:
- **Architecture map**: how a user message flows end-to-end — entry point → any retrieval/data lookup → GPT-4o-mini call → GPT-5-mini call (if in the loop) → response formatting → output. Note where each LLM sits in this chain.
- **Prompt inventory**: list every system/user prompt template in the codebase, where it lives, and what it's trying to do.
- **Rules inventory**: separately list every business rule, guardrail, or constraint in the codebase — whether it lives inside a prompt (as an instruction), in code (as conditionals/validation logic), or in config/data files. For each rule, note: what it enforces, where it's defined, and whether it's enforced consistently in one place or duplicated/contradicted across multiple locations (e.g. a rule stated in a prompt but not actually validated in code, or two prompts stating the same rule slightly differently).
- **Hallucination root causes**: for each place the bot generates trailer facts/specs/availability, identify whether it's grounded in retrieved data or if the model is free-generating. Flag any prompt that lacks explicit "only use provided data" constraints, missing citations back to source data, or lacks a fallback for "I don't have that information."
- **Conversational flow issues**: check whether conversation history/state is passed correctly, whether turns get truncated, whether the persona/tone is defined once vs. redefined inconsistently, and whether there's abrupt topic-switching logic that breaks flow.
- **Code quality issues**: dead code, duplicated logic, inconsistent error handling, missing type hints/validation, tight coupling between prompt text and business logic, lack of tests, poor logging/observability, inefficient or duplicate LLM calls.
- **Model allocation sanity check**: confirm whether GPT-4o-mini vs GPT-5-mini split of responsibilities is actually the most efficient/accurate split, or whether tasks should be reassigned between them.

Present this audit as a structured document before touching any code.

## Phase 2 — Refactor Plan
Based on the audit, propose a plan with:
- A **new prompt architecture**: a single source of truth for persona/tone/rules (system prompt), with task-specific prompts layered on top rather than duplicated. Include concrete grounding techniques to cut hallucinations — e.g. structured tool/function-calling for trailer data lookup instead of relying on the model to "know" specs, explicit "if not in the data provided, say you don't know" instructions, and citation-style grounding where the response references which retrieved record it came from.
- **Conversation flow fixes**: recommend how conversation state/history should be managed (windowing strategy, summarization for long threads, consistent handling of topic changes) so responses feel continuous rather than reset each turn.
- **Code structure**: propose a clean module layout (e.g. separating prompt templates, LLM client wrappers, data retrieval, conversation state, and response post-processing into distinct modules), with rationale.
- **Model usage**: recommend the ideal division of labor between GPT-4o-mini and GPT-5-mini given their respective strengths, including whether temperature/other params should change per call type.
- **Rules consolidation**: propose where each rule should live going forward (enforced in code vs. stated in a prompt vs. both), and how to eliminate duplication/contradiction so there's one source of truth per rule instead of it being re-stated or re-implemented in multiple places.
- **Optimizations**: flag opportunities to reduce redundant LLM calls, cache repeated lookups,reduce the number of LLMs and reduce latency.

Wait for my go-ahead after presenting this plan before writing full implementation code.

## Phase 3 — Implementation
Once I approve the plan:
- Refactor incrementally, module by module, not as one giant diff.
- Rewrite prompts to be explicit, structured (clear role/instructions/constraints/output format sections), and grounded in retrieved data rather than open-ended generation.
- Add guardrails: fallback responses for missing data, input validation, and error handling around both LLM calls.
- Add inline comments explaining *why* a prompt or piece of logic is structured a certain way, not just what it does.
- Note any place where you had to make an assumption about intended behavior, and flag it for my review rather than silently deciding.

## Output format
For each phase, use clear headers and keep code blocks scoped to one file/module at a time so changes are easy to review.



## Extra Info:
overview.md contains all of the code information, PROMPT_AUDIT.md contains all the potentials things that are wrong right now.