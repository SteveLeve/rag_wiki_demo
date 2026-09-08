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

## Ground truth belongs to one embedding model

`evaluation_groundtruth.relevant_chunk_ids` are **row ids in `embeddings_<alias>`** — not global
document ids. There is no shared `chunks` table; each model owns its own table with its own sequence,
so id 42 is a different chunk in every one of them.

That is why the table carries `embedding_model_alias`. `evaluation-lab/01` writes the alias it sampled
its chunks from, and every consumer filters on it:

```sql
FROM evaluation_groundtruth
WHERE quality_rating = 'good'
  AND embedding_model_alias = %s
```

Drop the filter and a notebook scores one model's retrieval against another model's ids. Nothing
raises — precision just comes out near zero and reads as a bad retriever. Before this column existed,
the whole advanced tier evaluated against an empty table and reported it as a result.

If a notebook loads zero questions, the alias is the first thing to check: it must match the alias
`evaluation-lab/01` ran under (the catalog default, `nomic_embed_text`).

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

## The teaching-copy register

`scripts/nb_lint.py` enforces this table. A notebook that is not the listed owner may not define
the helper at all — it imports it — and the owner must carry a `# TEACHING COPY` marker naming the
ragkit function it mirrors. `tests/test_teaching_parity.py` then executes the marked copies and
compares them to the library on shared fixtures.

Which helper is taught where. A helper is hand-written inline **exactly once**,
in the notebook whose learning objective names it; everywhere else it is imported.

| Helper | Taught in | Library home |
|---|---|---|
| `chunk_text` | `foundation/01` | `ragkit.chunking` |
| `cosine_similarity` | `foundation/01` | `ragkit.retrieval` |
| `PostgreSQLVectorDB` | `foundation/02` | `ragkit.store.VectorStore` |
| `precision_at_k`, `recall_at_k`, `mean_reciprocal_rank`, `ndcg_at_k` | `evaluation-lab/02` | `ragkit.metrics` |
| `bm25_search_postgresql`, `reciprocal_rank_fusion` | `advanced-techniques/07-hybrid-search` | `ragkit.retrieval` |
| `start_experiment`, `complete_experiment`, `save_metrics`, `compare_experiments` | `foundation/00-registry-and-tracking-utilities` | `ragkit.experiment` |
| connections, DDL, registry upserts | nowhere — Tier 3 | `ragkit.db`, `ragkit.registry` |

## Adjudications

Decisions taken while consolidating duplicated implementations, recorded so they
are not silently re-litigated.

**NDCG's ideal DCG is computed over the full relevant set, not the retrieved set.**
Six notebook copies of `ndcg_at_k` existed. An AST comparison showed only three
were semantically distinct, differing solely in `math.log2` versus `np.log2` — but
all six computed the ideal as `sorted(retrieved_relevance, reverse=True)`, the best
ordering *of what was retrieved*. That makes NDCG blind to recall: retrieving 1
relevant chunk out of 10 and ranking it first scored a perfect 1.000, because one
hit at rank 1 is the only achievable ordering of one hit. `ragkit.metrics`
normalises against `min(len(relevant), k)` instead, scoring that case 0.339.
The majority implementation was the wrong one. **Stored NDCG values from before
September 2026 are not comparable to values after it.**

**`precision_at_k` divides by `k`, not by the number of results returned.**
Returning fewer than `k` results is itself a failure to fill the slots, and the
metric should reflect it rather than grading on a curve.

**`compute_config_hash` truncates to 12 hex characters.** 48 bits is ample for
deduplicating experiment configs and short enough to read in a dashboard column.
The value is stored in `experiments.config_hash`, so widening it would invalidate
every existing row.

**`model_name` *is* the Ollama tag.** An earlier draft added a separate
`ollama_tag` column; it was removed as a second place for one fact to drift.
`table_name` is a generated column (`'embeddings_' || model_alias`), so the alias
and its table cannot disagree.

**`embedding_registry` keeps its v1 column ordinals, with v2 columns appended.**
Notebooks and tests read rows via `SELECT *` with positional indexing, so
inserting a column mid-table silently shifts every one of those reads.

## Known sharp edges

- **pgvector values come back as text.** psycopg2 has no adapter for the type, so
  `SELECT embedding` returns `'[0.1,0.2,...]'` and `len()` counts characters — a
  768-dimensional vector measures about 9,400. Always ask the database:
  `SELECT vector_dims(embedding)`.
- **Never pass a Python list of numpy floats to a vector column.** psycopg2 renders
  a list as a Postgres `ARRAY[...]`, and under numpy 2 each element reprs as
  `np.float64(0.0)` — producing invalid SQL. Pass `str([...])` of plain floats.
- **A name assigned anywhere in a function is local for that whole function.**
  Notebooks bind `db`, `config` and `conn` freely, so importing a *module* under
  one of those names and using it earlier in the same scope is an
  `UnboundLocalError`. Import the function directly instead.
- **`input()` blocks papermill forever.** `scripts/nb_lint.py` bans it.
- **A failed statement poisons the whole transaction.** Postgres answers every
  later query with "current transaction is aborted", so a `try/except` that
  prints and continues turns one real error into hundreds of useless ones and
  hides the cause. Any `except` around a query must `conn.rollback()` first.
- **Bind vector parameters with an explicit cast.** psycopg2 sends a Python list
  as `numeric[]`, and `embedding <=> %s` then fails with *operator does not
  exist: vector <=> numeric[]*. Write `%s::vector` every time.

## Things that are deliberately not here

- **No `docs/development/`.** Point-in-time phase reports and release notes were deleted in September
  2026; they described a January snapshot and read as current status. `git log` retains them.
- **No jupytext pairing.** It doubles what a learner browsing the repo sees. Structural notebook edits
  go through `scripts/nb_apply.py` (idempotent `nbformat` passes, reviewed via `git diff`).
- **No committed notebook outputs.** They bloat diffs and enshrine numbers from whichever era produced
  them.
