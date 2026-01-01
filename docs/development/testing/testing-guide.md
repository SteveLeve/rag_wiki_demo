# Testing Guide - Advanced RAG Learning System

## What We've Built (Phase 1 Complete ✅)

We've successfully created the complete infrastructure for an advanced RAG learning system:

- **7 git commits** with detailed documentation
- **22 files created** (20 notebooks + 2 documentation files)
- **Foundation reorganized** with registry integration
- **Ready for execution** and testing

## Current System Structure

```
rag_wiki_demo/
├── foundation/
│   ├── 00-setup-postgres-schema.ipynb          [Ready to test ✅]
│   ├── 00-registry-and-tracking-utilities.ipynb [Ready to test ✅]
│   ├── 00-load-or-generate-pattern.ipynb       [Ready to test ✅]
│   ├── 01-basic-rag-in-memory.ipynb            [Ready to test ✅]
│   └── 02-rag-postgresql-persistent.ipynb      [Ready to test ✅]
│
├── intermediate/                                [Not yet created]
│   ├── 03-loading-and-reusing-embeddings.ipynb [TODO]
│   └── 04-comparing-embedding-models.ipynb     [TODO]
│
├── advanced-techniques/                         [Stubs ready]
│   ├── INDEX.ipynb                             [Ready to read ✅]
│   ├── 05-reranking.ipynb                      [Structure only]
│   ├── 06-query-expansion.ipynb                [Structure only]
│   ├── 07-hybrid-search.ipynb                  [Structure only]
│   ├── 08-semantic-chunking-and-metadata.ipynb [Structure only]
│   ├── 09-citation-tracking.ipynb              [Structure only]
│   └── 10-combined-advanced-rag.ipynb          [Structure only]
│
├── evaluation-lab/                              [Stubs ready]
│   ├── INDEX.ipynb                             [Ready to read ✅]
│   ├── 01-create-ground-truth-human-in-loop.ipynb [Structure only]
│   ├── 02-evaluation-metrics-framework.ipynb   [Structure only]
│   ├── 03-baseline-and-comparison.ipynb        [Structure only]
│   └── 04-experiment-dashboard.ipynb           [Structure only]
│
├── LEARNING_ROADMAP.md                          [Ready to read ✅]
├── EVALUATION_GUIDE.md                          [Ready to read ✅]
└── IMPLEMENTATION_PROGRESS.md                   [Ready to read ✅]
```

---

## Recommended Testing Sequence

### Phase 1: Foundation Testing (Now!)

Test the complete foundation layer to ensure everything works:

#### Test 1: PostgreSQL Setup ⚙️
**Notebook:** `foundation/00-setup-postgres-schema.ipynb`

**Prerequisites:**
- PostgreSQL with pgvector running (see POSTGRESQL_SETUP.md)
- Database: `rag_db`
- User/password configured

**What to verify:**
1. All 4 tables created successfully:
   - `embedding_registry`
   - `evaluation_groundtruth`
   - `experiments`
   - `evaluation_results`
2. All 6 indexes created
3. No errors in DDL execution
4. Final verification query shows all tables

**Expected outcome:** All tables visible in verification query

---

#### Test 2: In-Memory RAG Basics 📚
**Notebook:** `foundation/01-basic-rag-in-memory.ipynb`

**Prerequisites:**
- Ollama installed and running
- Models downloaded:
  ```bash
  ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
  ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
  ```
- Python packages: `ollama`, `datasets`, `jupyter`

**What to verify:**
1. Dataset loads successfully (10MB Wikipedia)
2. Embeddings generate without errors
3. Similarity search returns relevant results
4. Question answering produces coherent responses
5. Sample questions work correctly

**Expected outcome:** All test questions answered with relevant context

**Time estimate:** 5-10 minutes (including embedding generation)

---

#### Test 3: PostgreSQL Persistent Storage + Registry 🗄️
**Notebook:** `foundation/02-rag-postgresql-persistent.ipynb`

**Prerequisites:**
- Test 1 completed (schema created)
- Ollama models available
- PostgreSQL connection configured

**What to verify:**
1. PostgreSQL connection successful
2. Table created with pgvector extension
3. Embeddings stored persistently
4. **Registry integration works:**
   - `register_embedding()` function executes
   - Metadata stored correctly
   - Registry ID returned
   - Discovery message printed
5. Similarity search via pgvector works
6. Questions answered correctly
7. Preservation mode works (skips regeneration on second run)

**Expected outcome:** 
- Embeddings persisted and registered
- Registry entry visible with: `SELECT * FROM embedding_registry;`
- Message: "✓ Registered embeddings in registry (ID: X)"

**Time estimate:** 10-15 minutes (first run with generation)

**Second run test:** Should skip embedding generation and reuse existing

---

#### Test 4: Registry Utilities 🛠️
**Notebook:** `foundation/00-registry-and-tracking-utilities.ipynb`

**Prerequisites:**
- Test 3 completed (embeddings registered)
- PostgreSQL schema exists

**What to verify:**
1. All utility functions defined successfully
2. `list_available_embeddings()` returns DataFrame with registered model
3. `get_embedding_metadata()` returns correct dimension and metadata
4. `compute_config_hash()` produces consistent hashes
5. Experiment tracking functions available

**Expected outcome:** All functions execute without errors

**Time estimate:** 2-3 minutes

---

#### Test 5: Load-or-Generate Pattern 🔄
**Notebook:** `foundation/00-load-or-generate-pattern.ipynb`

**Prerequisites:**
- Test 3 completed (embeddings in registry)

**What to verify:**
1. Pattern documentation clear
2. Function structure understandable
3. Interactive prompts explained
4. Example code demonstrates registry checking

**Expected outcome:** Clear understanding of pattern for use in advanced notebooks

**Time estimate:** 2-3 minutes (reading only)

---

### Phase 2: Documentation Review 📖

Read through documentation to understand the system:

#### Review 1: Learning Roadmap
**File:** `LEARNING_ROADMAP.md`

**What to verify:**
- Three paths clearly explained (A: 1-2hr, B: 3-4hr, C: 6-8hr)
- Success criteria actionable
- Commands helpful
- Troubleshooting relevant

#### Review 2: Evaluation Guide
**File:** `EVALUATION_GUIDE.md`

**What to verify:**
- Ground-truth creation process clear
- Metrics explained with formulas
- Code examples understandable
- Evaluation checklist complete

#### Review 3: Implementation Progress
**File:** `IMPLEMENTATION_PROGRESS.md`

**What to verify:**
- Current status accurate
- Next steps clear
- Pending tasks identified

---

### Phase 3: Advanced Notebooks Review 👀

Review stub notebooks to understand structure:

#### Review 1: Advanced Techniques INDEX
**Notebook:** `advanced-techniques/INDEX.ipynb`

**What to verify:**
- All 6 techniques described
- Complexity levels clear
- Recommended learning order makes sense
- Trade-off table helpful

#### Review 2: Individual Technique Stubs
**Notebooks:** `advanced-techniques/05-10.ipynb`

**What to verify:**
- Consistent structure across all notebooks
- Configuration sections clear
- TODO sections indicate what needs implementation
- Evaluation and tracking sections present

#### Review 3: Evaluation Lab INDEX
**Notebook:** `evaluation-lab/INDEX.ipynb`

**What to verify:**
- Evaluation workflow clear
- Metrics interpretation table helpful
- Tips actionable

---

## Success Criteria for Phase 1 Testing ✅

Foundation testing complete when:

- [ ] PostgreSQL schema created successfully (4 tables + 6 indexes)
- [ ] In-memory RAG works (embeddings + questions answered)
- [ ] PostgreSQL RAG works (persistent storage)
- [ ] **Registry integration works (embeddings registered with metadata)**
- [ ] Utility functions accessible
- [ ] Load-or-generate pattern understood
- [ ] Documentation reviewed and clear
- [ ] Advanced notebook structure understood

---

## Common Issues & Solutions

### Issue 1: PostgreSQL Connection Failed
**Symptom:** `psycopg2.OperationalError: could not connect`

**Solution:**
```bash
# Check PostgreSQL is running
docker ps | grep pgvector

# If not running, start it:
docker run -d \
  --name pgvector-demo \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=rag_db \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

### Issue 2: Ollama Models Not Found
**Symptom:** `Error: model not found`

**Solution:**
```bash
# Check models available
ollama list

# Download if missing
ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
```

### Issue 3: Registry Table Not Found
**Symptom:** `relation "embedding_registry" does not exist`

**Solution:**
Run `foundation/00-setup-postgres-schema.ipynb` first to create tables.

### Issue 4: Embedding Generation Too Slow
**Symptom:** Taking longer than expected

**Solution:**
- Reduce `TARGET_SIZE_MB` to 5 or 10
- Be patient - first run takes 5-10 minutes
- Second run should skip generation (preservation mode)

---

## What to Test Next (After Phase 1)

Once foundation testing passes:

1. **Implement intermediate notebooks** (03-04)
   - Create registry discovery teaching notebook
   - Create model comparison notebook
   - Test load-or-generate pattern in practice

2. **Implement evaluation lab** (01-04)
   - Synthetic test generation
   - Interactive curation interface
   - Metrics computation
   - Comparison and dashboard

3. **Implement advanced techniques** (05-10)
   - Fill TODO sections with working code
   - Test each technique individually
   - Integration notebook

---

## Current Git State

**Branch:** main  
**Last commit:** 69984ca (docs: Mark foundation reorganization task as complete)  
**Commits in Phase 1:** 7 total  
**Files added:** 22  
**Status:** Clean working directory, ready for testing

---

## Quick Test Commands

```bash
# Check PostgreSQL is running
docker ps | grep pgvector

# Check Ollama models
ollama list

# Verify registry tables exist (after running 00-setup-postgres-schema.ipynb)
psql -U postgres -d rag_db -c "SELECT tablename FROM pg_tables WHERE schemaname='public';"

# Check registered embeddings (after running 02-rag-postgresql-persistent.ipynb)
psql -U postgres -d rag_db -c "SELECT model_alias, dimension, embedding_count, created_at FROM embedding_registry;"

# View recent commits
git log --oneline -5
```

---

## Feedback Requested

After testing, please provide feedback on:

1. **Clarity:** Are notebook instructions clear?
2. **Errors:** Any errors encountered during execution?
3. **Performance:** How long did embedding generation take?
4. **Registry:** Did registry integration work as expected?
5. **Documentation:** Is LEARNING_ROADMAP.md helpful?
6. **Structure:** Is the numbered progression intuitive?

---

*Last updated: After completing Phase 1 foundation reorganization*  
*Next milestone: Intermediate notebooks + Evaluation lab implementation*
