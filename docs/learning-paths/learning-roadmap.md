# Learning Roadmap: Advanced RAG System Development

This document guides you through the progressive learning path for building and understanding advanced RAG systems.

## Overview

The project is organized into 4 learning levels, each building on the previous:

1. **Foundation** (1-2 hours) — RAG fundamentals and persistent storage setup
2. **Intermediate** (1-2 hours) — Reusing embeddings and comparing models
3. **Advanced Techniques** (4-6 hours) — Individual techniques with isolation and integration
4. **Evaluation Lab** (2-4 hours) — Creating test sets and measuring system quality

## Learning Paths

Choose based on your goals and available time:

### Path A: Quick Learning (1-2 hours)
**Goal:** Understand RAG fundamentals without infrastructure

1. Read: [readme.md](../../README.md) - Architecture section
2. Run: `foundation/01-basic-rag-in-memory.ipynb` — In-memory RAG with Simple Wikipedia
3. Test: Run sample queries to see basic RAG in action
4. Next: Explore [quick-reference.md](../user-guides/quick-reference.md) for concepts

**Time estimate:** 1-2 hours  
**Hardware:** 8GB RAM sufficient

---

### Path B: Practical Experimentation (3-4 hours)
**Goal:** Set up persistent storage and run multiple experiments

1. Setup: Run `foundation/00-setup-postgres-schema.ipynb` (one-time)
2. Run: `foundation/02-rag-postgresql-persistent.ipynb` — Generate embeddings once
3. Reuse: `intermediate/03-loading-and-reusing-embeddings.ipynb` — Load without regeneration
4. Compare: `intermediate/04-comparing-embedding-models.ipynb` — Test 2+ models
5. Analyze: Create your own analysis notebook from template
6. Next: Try one advanced technique notebook

**Time estimate:** 3-4 hours (50 min generation + analysis)  
**Hardware:** 8GB+ RAM, ~500MB disk for PostgreSQL

**Key learning:**
- How persistent storage saves time
- Registry system for discovering embeddings
- Flexible load-or-generate pattern
- Side-by-side model comparison

---

### Path C: Deep Mastery (6-8 hours)
**Goal:** Understand advanced techniques and integrated RAG systems

**Prerequisites:**
- Complete Path B (persistent storage working)
- Have embeddings registered in the registry

**Steps:**

1. **Infrastructure Setup** (10 min)
   - Verify `foundation/00-setup-postgres-schema.ipynb` completed
   - Check `embedding_registry` table with available embeddings

2. **Technique Exploration** (2-3 hours, choose 3-4 techniques)
   
   Each technique notebook (~30-45 min each):
   - `advanced-techniques/05-reranking.ipynb`
   - `advanced-techniques/06-query-expansion.ipynb`
   - `advanced-techniques/07-hybrid-search.ipynb`
   - `advanced-techniques/08-semantic-chunking-and-metadata.ipynb`
   - `advanced-techniques/09-citation-tracking.ipynb`
   
   **For each notebook:**
   1. Read the overview section
   2. Load embeddings using registry discovery
   3. Implement the technique
   4. Measure impact on retrieval metrics
   5. Take notes on trade-offs (quality vs. latency)

3. **Integration** (1-2 hours)
   - `advanced-techniques/10-combined-advanced-rag.ipynb` — Stack multiple techniques
   - Enable/disable techniques with configuration
   - Compare: Single techniques vs. combination

4. **Evaluation** (1-2 hours)
   - `evaluation-lab/01-create-ground-truth-human-in-loop.ipynb` — Build test set
   - `evaluation-lab/02-evaluation-metrics-framework.ipynb` — Measure system quality
   - `evaluation-lab/03-baseline-and-comparison.ipynb` — Compare configurations
   - `evaluation-lab/04-experiment-dashboard.ipynb` — Analyze results

**Key learning:**
- How to implement advanced RAG techniques
- Impact of each technique on quality and performance
- Human-in-the-loop evaluation curation
- Comprehensive metrics and benchmarking
- Experiment tracking for reproducibility

---

## Technique Selection Guide

Not sure which advanced techniques to explore first? Use this guide:

| Goal | Techniques | Time | Difficulty |
|------|-----------|------|------------|
| Better answer quality | Reranking, Query Expansion | 1-2 hours | Medium |
| Handle specific terminology | Hybrid Search | 1-2 hours | Medium |
| Transparent answers | Citation Tracking | 1 hour | Low |
| Optimized retrieval | Semantic Chunking | 1.5 hours | Medium |
| Multiple techniques combined | All of above | 3+ hours | Hard |

### Technique Dependencies

```
Foundation
  ↓
Intermediate
  ├─→ Reranking
  ├─→ Query Expansion
  ├─→ Hybrid Search
  ├─→ Semantic Chunking
  └─→ Citation Tracking
       ↓
  Combined RAG (stacks above)
       ↓
  Evaluation Lab
       ├─→ Ground Truth Creation
       ├─→ Metrics Framework
       ├─→ Baseline Comparison
       └─→ Experiment Dashboard
```

---

## Typical Workflow

### For Exploratory Research (Path C)

1. **Week 1:** Complete Foundation & Intermediate
2. **Week 2:** Explore 2-3 technique notebooks
3. **Week 3:** Integrate techniques, build ground-truth test set
4. **Week 4:** Systematic evaluation, comparison, optimization

### For Production Preparation

1. Complete Path B (persistent storage working)
2. Run technique notebooks in order (05 → 10)
3. Create ground-truth evaluation set with domain experts
4. Benchmark each configuration variant
5. Select optimal configuration based on metrics
6. Deploy with chosen configuration

---

## Key Concepts by Level

### Foundation Level
- Vector similarity search (cosine distance)
- Retrieval-Augmented Generation pipeline
- Chunking strategies
- Embedding models
- PostgreSQL + pgvector

### Intermediate Level
- Embedding registry & discovery
- Load-or-generate pattern
- Comparing embedding models
- Configuration management
- Experiment tracking basics

### Advanced Techniques Level
- Reranking (cross-encoder models)
- Query expansion (LLM-based query generation)
- Hybrid search (vector + BM25 fusion)
- Semantic chunking (meaning-aware splitting)
- Citation tracking (provenance for answers)
- Technique composition (stacking improvements)

### Evaluation Lab Level
- Ground-truth dataset creation
- Human-in-the-loop curation
- Retrieval metrics (Precision@K, Recall@K, MRR, NDCG)
- Generation metrics (BLEU, ROUGE, LLM-as-judge)
- Experiment tracking & reproducibility
- Comparative analysis dashboards

---

## Estimated Time Investment

| Level | Components | Total Time |
|-------|-----------|-----------|
| Foundation | 01, 02 | 1-2 hours |
| Intermediate | 03, 04 | 1-2 hours |
| Advanced Techniques | 05-10 | 4-6 hours |
| Evaluation Lab | 01-04 | 2-4 hours |
| **Total (all paths)** | **All** | **8-14 hours** |

*Times assume prior RAG familiarity. Add 2-3 hours if new to vector databases.*

---

## Success Criteria

### Foundation Level ✓
- [ ] In-memory RAG works with sample queries
- [ ] PostgreSQL running and connected
- [ ] Schema tables created (registry, experiments, etc.)
- [ ] Can describe RAG pipeline

### Intermediate Level ✓
- [ ] Embeddings persisted in PostgreSQL
- [ ] Registry shows available models
- [ ] Can load different embedding models
- [ ] Compared 2+ models on same queries

### Advanced Techniques Level ✓
- [ ] Implemented 3+ techniques
- [ ] Understand trade-offs of each
- [ ] Combined techniques show improvement
- [ ] Can measure impact via metrics

### Evaluation Lab Level ✓
- [ ] Created ground-truth test set (50+ questions)
- [ ] Computed retrieval metrics
- [ ] Computed generation metrics
- [ ] Compared 2+ configurations systematically

---

## Helpful Commands

### Start PostgreSQL
```bash
docker run -d --name pgvector-rag \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=rag_db \
  -p 5432:5432 \
  -v pgvector_data:/var/lib/postgresql/data \
  pgvector/pgvector:pg16
```

### Check Status
```bash
docker ps | grep pgvector-rag
```

### View Embeddings Registry
```bash
# In a notebook:
available = list_available_embeddings(db)
print(available)
```

### View Experiments
```bash
recent = list_experiments(db, limit=10)
print(recent)
```

### Re-run an Experiment
```bash
# Find configuration
exp = get_experiment(db, experiment_id=42)
config = exp['config']

# Re-run with same config
experiment_id = start_experiment(..., config=config)
```

---

## Troubleshooting

**"Embeddings not found in registry"**
→ Run `foundation/02-rag-postgresql-persistent.ipynb` first to generate embeddings

**"PostgreSQL connection refused"**
→ Start Docker container: `docker start pgvector-rag`

**"Slow retrieval queries"**
→ Indexes created automatically. Check: `SELECT * FROM pg_indexes WHERE tablename LIKE 'embeddings%'`

**"Want to start fresh"**
→ `docker volume rm pgvector_data && rm -rf data/experiment_results`

---

## Next Steps

1. **Choose your learning path** (A, B, or C above)
2. **Start with foundation notebooks** (foundation/01 or 02)
3. **Work through notebooks in order** - they build on each other
4. **Create your own experiment notebooks** - copy structure from templates
5. **Share results** - compare metrics with others

For detailed information:
- Technique-specific guides → See notebook docstrings
- Evaluation methodology → See [evaluation-guide.md](../development/testing/evaluation-guide.md)
- Configuration reference → See [quick-reference.md](../user-guides/quick-reference.md)

Happy learning! 🚀
