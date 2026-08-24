# Handoff prompt — route Concession trailers out of Enclosed

Paste everything below the line into a fresh Claude Code session in this repo.

---

`Concession` is now a canonical trailer category in this repo (`src/domain/categories.py`),
with the naming terms `concession`, `concession trailer`, `food trailer`, `food trailers`, and
its own qualification spec in `src/domain/trailer_fields.py`. The Respond prompt has a
guardrail that tells the customer we do not stock a category when it has no listings.

The problem: **no listing in the catalogue carries `category = "Concession"`, so the guardrail
tells customers we have no concession trailers while we actually have three.** They are filed
under `Enclosed` and identifiable only by the word "Concession" in their title:

- `2026 Haulmark Transport 8.5' x 16' w/ Concession Window and Elec - 46863`
- `2026 Cargo Craft Trailers 8.5'x22' Inclu. 8' Porch Concession Trailer - 71152`
- `2026 Cargo Craft Trailers Concession Trailer 8.5' x 16' - 70864`

Change ingest so the Concession category is populated correctly:

1. **A listing whose source category is already `Concession` is accepted as-is.** It must
   survive ingest as `Concession` and must not be rewritten to `Enclosed` or dropped as a
   non-canonical category.
2. **A listing whose source category is `Enclosed` but whose title or model names a concession
   trailer is stored as `Concession`.** Match on the title and the model/subcategory fields.
   Reclassification applies only to rows coming in as `Enclosed` — never re-file a Dump or
   Flatbed row because of a stray word.

Where to work:

- `src/search/ingest.py` is the workbook → `trailer_listings` path; the category is normalised
  there before the row is written.
- `src/domain/brands.py::_display_category` and `load_make_inventory` decide which categories
  are advertised; `stocked_categories()` is what the prompts read.
- `src/domain/categories.py` already holds `CANONICAL_CATEGORIES` and the term lists — reuse
  the Concession naming terms rather than writing a second copy of that vocabulary.

Watch out for:

- **Be conservative about what counts as a concession trailer.** "Concession Window" on a
  cargo trailer is a fitted option, not necessarily a concession unit — decide deliberately
  whether unit `46863` should move, and say which way you went and why. Getting this wrong in
  the greedy direction silently empties the Enclosed category.
- **`load_make_inventory()` is `@lru_cache(maxsize=1)`**, so the advertised category list is
  resolved once per process. After re-ingesting, the API must be restarted before the prompt
  stops saying we have no concession trailers.
- The three units above are the only current matches, so `Concession` will be a **3-listing
  category** — thin, like Roll Off (1) and Race Trailer (2).
- `tests/unit/test_stock_guardrails.py` asserts that Concession is canonical **and currently
  unstocked**. Once ingest gives it stock, `test_concession_is_canonical_but_not_advertised`
  becomes wrong and must be updated — it is not a spurious failure, it is that test doing its
  job. Keep the "recognised but unstocked" path covered with a different category or a
  monkeypatched catalogue, since the guardrail itself still needs testing.

**DO NOT RUN THE INGEST.** `src/search/ingest.py` writes to whatever `HOST`/`DATABASE` in
`.env` point at, and that is currently the live Azure Postgres the bot serves from. It is
destructive on every path: `upsert_batch` overwrites rows, `prune_missing` DELETEs anything
missing from the workbook on a NORMAL run (not just `--force`), and `--force` wipes the whole
`trailer_listings` table before reloading. There is no dry-run flag and no separate ingest
target. Write the classification change and its unit tests, then STOP and hand the run back
to the repo owner to execute against a database of their choosing.

Verify your change WITHOUT touching the database:

- Unit-test the classification function directly against the three titles above plus
  counter-examples (a plain Enclosed cargo trailer, a Dump row containing a stray word).
- `build_record()` is pure — feed it a `pandas.Series` built from workbook-shaped fields and
  assert the category it produces. No session, no commit.

Once the owner has ingested, the end-to-end check is: restart the API (see the `lru_cache`
note above), then ask the bot "do you have any food trailers?" — it should show the concession
units instead of the not-in-stock message.
