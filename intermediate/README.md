# Intermediate Layer: Registry Discovery and Model Comparison

Welcome to the intermediate layer! After completing the foundation notebooks, you're ready to explore practical patterns for managing embeddings and comparing their performance.

## Overview

The intermediate layer teaches two critical concepts:
1. **Registry discovery** - How to find and reuse pre-computed embeddings
2. **Model comparison** - How to evaluate trade-offs between embedding models

These notebooks prepare you for advanced techniques by establishing patterns you'll use throughout the system.

## Notebooks

### 03: Loading and Reusing Embeddings from Registry

**What you'll learn:**
- Discover embeddings registered in PostgreSQL using `list_available_embeddings()`
- Retrieve metadata about models using `get_embedding_metadata()`
- Use the `load_or_generate()` pattern to efficiently reuse existing embeddings
- Build an interactive discovery interface

**Prerequisites:**
- Completion of `foundation/02-rag-postgresql-persistent.ipynb`
- PostgreSQL running with at least one registered embedding model
- Python packages: psycopg2-binary, ollama

**Key concepts:**
```python
# Find what embeddings are available
available = list_available_embeddings()

# Get details about a model
metadata = get_embedding_metadata('bge_base_en_v1.5')

# Load or generate (no regeneration if exists!)
embeddings = load_or_generate(
    model='new_model',
    chunks=chunks,
    fallback_to_generate=True
)
```

**Time estimate:** 10-15 minutes

**Next steps:**
- Use this pattern in advanced techniques (05-10)
- Reference it when implementing evaluation notebooks (01-04)

---

### 04: Comparing Embedding Models

**What you'll learn:**
- Load multiple embedding models from the registry
- Run identical queries on different models
- Compute evaluation metrics (Precision@K, Recall@K, MRR, NDCG)
- Analyze quality vs. speed trade-offs
- Make informed decisions about model selection

**Prerequisites:**
- Completion of `intermediate/03-loading-and-reusing-embeddings.ipynb`
- Access to 2+ embedding models registered in PostgreSQL
- Python packages: psycopg2-binary, ollama, numpy, pandas

**Key concepts:**
```python
# Load multiple models
model1 = load_embedding_model('bge_base_en_v1.5')
model2 = load_embedding_model('all_minilm_l6_v2')

# Compare on same queries
for query in test_queries:
    results1 = model1.query(query)
    results2 = model2.query(query)
    
    # Compute metrics
    metrics = compute_metrics(results1, results2)
    print(f"Model 1 Precision@5: {metrics['p@5']:.3f}")
    print(f"Model 2 Precision@5: {metrics['p@5']:.3f}")
```

**Time estimate:** 15-20 minutes

**Next steps:**
- Reference these metrics in advanced techniques
- Use this methodology in evaluation-lab/03

---

## Learning Path

This layer bridges foundation and advanced concepts:

```
Foundation (foundation/01-02)
        ↓
   Intermediate (03-04)
        ↓
Advanced Techniques (advanced-techniques/05-10)
        ↓
Evaluation Lab (evaluation-lab/01-04)
```

**Recommended progression:**
1. Complete `foundation/02` (register some embeddings)
2. Run `intermediate/03` (discover what you registered)
3. Run `intermediate/04` (if you have multiple models)
4. Move to `advanced-techniques/05` (apply these patterns)

## Key Skills You'll Build

✅ **Registry discovery** - Find and reuse embeddings without regeneration  
✅ **Model comparison** - Evaluate embedding quality objectively  
✅ **Performance optimization** - Avoid expensive operations  
✅ **Metrics interpretation** - Understand what the numbers mean  

## Common Patterns

### Pattern 1: Efficient Reuse
```python
# Check if embeddings exist before regenerating
if embedding_exists('my_model', 'my_dataset'):
    embeddings = load_embedding('my_model', 'my_dataset')
else:
    embeddings = generate_and_register('my_model', chunks)
```

### Pattern 2: Model Comparison
```python
# Compare models on the same test set
models = ['model1', 'model2', 'model3']
results = {}

for model in models:
    embedding_fn = get_embedding_function(model)
    scores = evaluate_on_testset(embedding_fn, testset)
    results[model] = scores

# Summarize findings
compare_results(results)
```

## Troubleshooting

**Q: "No embeddings found in registry"**
- Solution: Run `foundation/02-rag-postgresql-persistent.ipynb` first to register embeddings

**Q: "Model not found in Ollama"**
- Solution: `ollama pull <model-name>` in terminal first

**Q: "Registry table doesn't exist"**
- Solution: Run `foundation/00-setup-postgres-schema.ipynb` to create tables

## Next Steps

Once you complete these intermediate notebooks:

1. **Advanced Techniques** - Apply these patterns to advanced RAG methods
2. **Evaluation Lab** - Use metrics from notebook 04 in evaluation framework
3. **Custom Implementation** - Create your own registry discovery interface

## Resources

- `foundation/README.md` - What foundation notebooks do
- `foundation/00-registry-and-tracking-utilities.ipynb` - Available utility functions
- `LEARNING_ROADMAP.md` - Full learning progression
- `EVALUATION_GUIDE.md` - How to measure RAG quality

---

**Last updated:** After intermediate notebook creation  
**Difficulty:** ⭐⭐ Intermediate  
**Time commitment:** 25-35 minutes total
