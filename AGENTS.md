# AGENTS.md — repository conventions

This is a **teaching repository**. The notebooks are the product; `src/ragkit/` exists to serve them.
That inverts some normal instincts, so read this before editing.

## Layout

```
src/ragkit/          Importable toolkit. Shared, tested, boring.
foundation/          Tier 1 curriculum: RAG from scratch, in memory then in Postgres.
intermediate/        Registry reuse, model comparison, embedding dimensions.
advanced-techniques/ Reranking, query expansion, hybrid search, chunking, citations.
evaluation-lab/      Ground truth, metrics, baselines, dashboards.
scripts/             Maintenance and CI gates (nb_lint, check_links, preflight, reset_db).
tests/               Pytest suite. Imports ragkit; never redefines it.
```

Notebook numbering is **one global sequence across tiers** (00 → 12). Many markdown files link into
it. **Append; never renumber.**

## The teaching-copy rule

A learning project dies if every cell becomes `from ragkit import x`. It also dies if the same
function exists in eight drifting copies — which is exactly what happened here before September 2026.
The resolution is three tiers:

- **Tier 1 — Teach it.** An algorithm named in a notebook's learning objective is written out by hand,
  inline, in the notebook that owns that lesson. It appears inline **exactly once in the repository**.
- **Tier 2 — Use it.** Every *later* notebook imports that same function from `ragkit`. It does not
  re-derive it. This is where the duplicate copies went to die.
- **Tier 3 — Never teach.** Connections, DDL, registry upserts, experiment rows, env loading. Always
  imported, never shown. This was never the lesson; it was noise that made drift invisible.

Every Tier-1 cell must open with the marker comment:

```python
# TEACHING COPY — library version: ragkit.metrics.ndcg_at_k
```

`tests/test_teaching_parity.py` extracts each marked cell, executes it, and asserts the inline
implementation agrees with the library version on shared fixtures. **Intentional duplication is
therefore a tested invariant, not a drift liability.** Adding a Tier-1 copy without the marker, or
whose behavior diverges from the library, fails CI.

## The canonical alias invariant

An embedding model has two distinct names, and conflating them caused the worst bug in this repo's
history:

- **`ollama_tag`** — what you pull and pass to Ollama (`nomic-embed-text`, `all-minilm`).
- **`model_alias`** — the canonical registry key and table-name stem (`nomic_embed_text`).

`ragkit.models.canonical_alias()` is the **only** function permitted to produce an alias, and
`ragkit.registry.table_name_for()` the **only** one permitted to produce a table name. Never build
either by hand.

Four layers enforce this, because convention alone already failed once:

1. `canonical_alias()` emits a legal unquoted SQL identifier by construction.
2. It is called on every registry read path *and* every write path.
3. A DB `CHECK (model_alias ~ '^[a-z][a-z0-9_]*$')` constraint rejects anything else.
4. `scripts/nb_lint.py` bans `.replace(".", "_")` in notebook source.

**`experiments.embedding_model_alias` has a foreign key to `embedding_registry.model_alias`.** So an
alias change that touches only one of those tables produces mid-notebook FK violations. Alias changes,
registry seeding, and a DB reset ship in the same commit — never separately.

## Dimensions are never hardcoded

`ragkit.db.ensure_embedding_table()` reads `(table_name, dimension)` from the registry and emits the
**only** `vector(N)` string in the codebase. `nb_lint` bans the literal `vector(` from notebooks.
The catalog spans 384 / 768 / 1024 dimensions on purpose, so a dimension assumption fails loudly
rather than silently returning wrong neighbors.

## Verifying a change

| Check | Command |
|---|---|
| No services required | `RAG_FAKE_MODELS=1 pytest -m "not postgres"` |
| Teaching copies agree with library | `pytest tests/test_teaching_parity.py` |
| Notebook conventions | `python scripts/nb_lint.py` |
| Markdown links resolve | `python scripts/check_links.py` |
| Catalog matches real models | `python scripts/preflight.py` |
| Everything | `pytest` (needs PostgreSQL) |

`RAG_FAKE_MODELS=1` swaps in `ragkit.testing.FakeOllamaClient`, whose embeddings are derived from
content hash and whose dimensions come from the catalog. This lets the full notebook suite execute in
CI with **no Ollama server and no model downloads**. Use it.

### Local PostgreSQL

Port 5432 is frequently occupied by other projects. This repo defaults to **5433** and reads
connection settings from the environment, so it can never reach into a neighbouring database:

```bash
docker run -d --name rag-wiki-pgvector \
  -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=rag_db \
  -p 127.0.0.1:5433:5432 pgvector/pgvector:pg16
```

Data here is regenerable — the corpus streams from `wikimedia/wikipedia`. When schema or aliases
change, `python scripts/reset_db.py --yes` is the expected answer, not a migration.

## Things that are deliberately not here

- **No `docs/development/`.** Point-in-time phase reports and release notes were deleted in September
  2026; they described a January snapshot and read as current status. `git log` retains them.
- **No jupytext pairing.** It doubles what a learner browsing the repo sees. Structural notebook edits
  go through `scripts/nb_apply.py` (idempotent `nbformat` passes, reviewed via `git diff`).
- **No committed notebook outputs.** They bloat diffs and enshrine numbers from whichever era produced
  them.
