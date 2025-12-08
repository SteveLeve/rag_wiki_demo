# Advanced Techniques Layer: Beyond Basic RAG

Welcome to the advanced techniques layer! Here you'll explore sophisticated RAG enhancements that improve retrieval quality, ranking, and generation.

## Overview

The advanced layer contains 6 progressive techniques (05-10) that build on each other:

1. **Reranking** - Refine retrieval results with a second-pass ranking model
2. **Query Expansion** - Generate multiple queries to broaden retrieval
3. **Hybrid Search** - Combine dense and sparse retrieval methods
4. **Semantic Chunking & Metadata** - Structure documents for better retrieval
5. **Citation Tracking** - Track which chunks generated which answers
6. **Combined Advanced RAG** - Integrate multiple techniques together

## Prerequisites

Before starting this layer, you should have completed:

- ✅ `foundation/01-basic-rag-in-memory.ipynb` - Understand core RAG concepts
- ✅ `foundation/02-rag-postgresql-persistent.ipynb` - PostgreSQL integration
- ✅ `intermediate/03-loading-and-reusing-embeddings.ipynb` - Registry discovery
- ✅ `intermediate/04-comparing-embedding-models.ipynb` - Model evaluation

**Key skills needed:**
- Embedding generation and similarity search
- PostgreSQL with pgvector
- Registry discovery patterns (load_or_generate)
- Metrics computation and interpretation

## Architecture Pattern

All notebooks in this layer follow a consistent pattern:

```python
# 1. Load embeddings from registry (no regeneration!)
embeddings = load_or_generate('model', chunks, fallback_to_generate=False)

# 2. Implement the technique
results_before = basic_retrieval(query)
results_after = advanced_technique(query)

# 3. Compare performance
metrics = evaluate_improvement(results_before, results_after)

# 4. Track in experiments table
register_experiment(
    technique='technique_name',
    config=config_dict,
    metrics=metrics
)
```

## Notebooks

### 05: Reranking

**What you'll learn:**
- Use a dedicated reranking model to refine retrieval results
- Score chunks with both embedding similarity and contextual relevance
- Combine multiple ranking signals
- Measure improvement in retrieval quality

**Improvement area:** Retrieval precision (top results more relevant)

**Key pattern:**
```python
# First pass: broad retrieval with embeddings
candidates = embedding_retrieval(query, top_n=50)

# Second pass: rerank with specialized model
reranked = rerank_model.score(query, candidates)
top_5 = reranked[:5]
```

**Expected improvement:** Precision@5 +10-20%

**Time estimate:** 30-40 minutes

---

### 06: Query Expansion

**What you'll learn:**
- Generate multiple reformulations of user queries
- Search with multiple query versions
- Combine results to capture diverse relevant chunks
- Balance coverage vs. redundancy

**Improvement area:** Retrieval recall (fewer relevant chunks missed)

**Key pattern:**
```python
# Original query
query = "What is photosynthesis?"

# Generate reformulations
expanded = [
    query,
    "How do plants convert sunlight to energy?",
    "Explain plant energy production",
    "Light reactions and Calvin cycle"
]

# Search with all, deduplicate
results = []
for q in expanded:
    results.extend(similarity_search(q, top_n=10))
results = deduplicate(results)
```

**Expected improvement:** Recall +15-30%

**Time estimate:** 30-40 minutes

---

### 07: Hybrid Search

**What you'll learn:**
- Combine dense (embedding-based) and sparse (keyword-based) retrieval
- Use BM25 for keyword matching
- Fuse results from both methods
- Handle different document types effectively

**Improvement area:** Retrieval robustness (handles both semantic and exact matches)

**Key pattern:**
```python
# Dense retrieval (semantic similarity)
dense_results = embedding_search(query, top_n=10)

# Sparse retrieval (keyword matching)
sparse_results = bm25_search(query, top_n=10)

# Fuse results (reciprocal rank fusion)
fused = reciprocal_rank_fusion(dense_results, sparse_results)
```

**Expected improvement:** Handles +30-50% more query types effectively

**Time estimate:** 40-50 minutes

---

### 08: Semantic Chunking and Metadata

**What you'll learn:**
- Structure documents with semantic boundaries (not just size)
- Attach metadata (source, date, confidence) to chunks
- Filter retrieval by metadata
- Preserve document structure for better context

**Improvement area:** Retrieval context quality (more coherent passages)

**Key pattern:**
```python
# Semantic chunking (respect paragraph/sentence boundaries)
chunks = semantic_chunk(document, max_tokens=300)

# Attach metadata
for i, chunk in enumerate(chunks):
    chunk.metadata = {
        'source': document.source,
        'chunk_number': i,
        'confidence': calculate_relevance(chunk)
    }

# Filter during retrieval
results = search(query, min_confidence=0.7)
```

**Expected improvement:** Context coherence score +25-35%

**Time estimate:** 40-50 minutes

---

### 09: Citation Tracking

**What you'll learn:**
- Track which source chunks contributed to each answer
- Build provenance chains (answer ← context ← source)
- Enable fact verification and source highlighting
- Measure answer grounding quality

**Improvement area:** Answer trustworthiness (can verify claims)

**Key pattern:**
```python
# Track chunk-to-answer provenance
citations = []

for chunk in retrieved_chunks:
    # Use chunk in context
    context = build_context(retrieved_chunks)
    
    # Generate answer
    answer = generate_answer(query, context)
    
    # Track which chunks were used
    citations.append({
        'chunk_id': chunk.id,
        'chunk_text': chunk.text,
        'answer_segment': relevant_segment(answer, chunk),
        'confidence': similarity(answer_segment, chunk)
    })
```

**Expected improvement:** Answer traceability: 100% of claims verified

**Time estimate:** 35-45 minutes

---

### 10: Combined Advanced RAG

**What you'll learn:**
- Integrate techniques 05-09 into a unified system
- Stack techniques effectively (order matters!)
- Balance improvements vs. computational cost
- Design optimal pipeline for your use case

**Key pattern:**
```
Input Query
    ↓
[06] Query Expansion → multiple queries
    ↓
[07] Hybrid Search → dense + sparse results
    ↓
[08] Semantic Chunking → filter by metadata
    ↓
[05] Reranking → top results only
    ↓
[09] Citation Tracking → track provenance
    ↓
Generate + explain → answer with sources
```

**Expected improvement:** +35-50% total quality improvement, 100% traceable

**Time estimate:** 50-60 minutes

---

## Learning Progression

**Recommended path:**

```
Foundation complete
        ↓
[05] Reranking (easiest, immediate improvement)
        ↓
[06] Query Expansion (moderate, addresses recall)
        ↓
[07] Hybrid Search (more complex, handles edge cases)
        ↓
[08] Semantic Chunking (requires restructuring)
        ↓
[09] Citation Tracking (adds traceability)
        ↓
[10] Combined RAG (integrates everything)
```

**Alternative: Focused path**
- Only do [05] if you care about precision
- Only do [06] if you care about recall
- Only do [07] if you have keyword-heavy use cases
- Jump to [10] if you want everything

## Key Patterns Across All Notebooks

### Pattern 1: Registry-First Design
```python
# ALWAYS check registry first - never regenerate!
embeddings = load_or_generate(
    model='target_model',
    chunks=chunks,
    fallback_to_generate=False  # Fail fast if not registered
)
```

### Pattern 2: Experiment Tracking
```python
# Track every technique run
exp_id = start_experiment(
    technique='05_reranking',
    model='bge_base_en_v1.5',
    config={'reranker': 'ms_marco', 'top_n': 5}
)

# Run technique...
metrics = evaluate_results(results)

# Save results
complete_experiment(exp_id, metrics)
```

### Pattern 3: Metrics Comparison
```python
# Always compare before vs. after
baseline = evaluate_on_testset(basic_retrieval)
improved = evaluate_on_testset(advanced_technique)

improvement = calculate_improvement(baseline, improved)
print(f"Improvement: {improvement['precision_delta']:+.3f}")
```

## Common Implementation Tasks

Each notebook will have TODO sections for:

1. **Data loading** - Load registered embeddings
2. **Technique implementation** - Core algorithm code
3. **Evaluation** - Compute metrics
4. **Experiment tracking** - Save results to registry
5. **Visualization** - Show before/after comparison

## Troubleshooting

**Q: "Model not found in registry"**
- Run `intermediate/04` to register multiple models
- Or run `foundation/02` to create initial embeddings

**Q: "Technique doesn't improve results"**
- Check if your testset is representative
- Try different hyperparameters
- Combine with other techniques

**Q: "Performance is too slow"**
- Reduce chunk count for testing
- Use smaller models for development
- Optimize database queries

## What You'll Build

After completing this layer, you'll have:

✅ Advanced RAG system with multiple improvement techniques  
✅ Experiment tracking for continuous improvement  
✅ Metrics showing specific improvements from each technique  
✅ Production-ready code patterns  
✅ Understanding of RAG optimization trade-offs  

## Next Steps

1. **Evaluation Lab** - Measure overall system quality (evaluation-lab/01-04)
2. **Custom Notebooks** - Implement techniques specific to your domain
3. **Production Deployment** - Use these patterns in real applications

## Resources

- `LEARNING_ROADMAP.md` - Full progression map
- `EVALUATION_GUIDE.md` - How to measure improvements
- `foundation/00-registry-and-tracking-utilities.ipynb` - Available functions
- `intermediate/README.md` - What intermediate notebooks teach

---

**Difficulty:** ⭐⭐⭐ Advanced  
**Time commitment:** 4-6 hours total (all techniques)  
**Recommended approach:** 1-2 techniques per session for understanding
