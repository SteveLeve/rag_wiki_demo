# CLAUDE.md

See **[AGENTS.md](./AGENTS.md)** for this repository's conventions.

The three rules most likely to trip you up:

1. **The teaching-copy rule.** Some duplication here is deliberate and tested. Before deleting an
   inline helper as "redundant", check for a `# TEACHING COPY` marker — those are the curriculum.
2. **Never build an embedding alias or table name by hand.** Use `ragkit.models.canonical_alias()`
   and `ragkit.registry.table_name_for()`. A foreign key makes partial alias fixes worse than none.
3. **Never hardcode a vector dimension.** Read it from the registry.

Verify with `RAG_FAKE_MODELS=1 pytest -m "not postgres"` — no Ollama or Postgres needed.
