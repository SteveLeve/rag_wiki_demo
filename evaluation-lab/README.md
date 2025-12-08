# Evaluation Lab: Measuring RAG Quality

Welcome to the evaluation lab! Here you'll learn how to rigorously measure your RAG system's quality through ground-truth development, metrics, and systematic comparison.

## Overview

The evaluation lab is organized as a progression from basic measurement to comprehensive dashboards:

1. **Ground-Truth Creation** - Curate high-quality test sets
2. **Metrics Framework** - Compute retrieval and generation metrics
3. **Baseline & Comparison** - Benchmark against baselines
4. **Experiment Dashboard** - Visualize results and trends
5. **Supplemental Analysis** (optional) - Deep-dive embedding analysis

## Why Evaluation Matters

You can't improve what you don't measure. This layer teaches systematic evaluation so you can:

- ✅ Know if your technique actually helps
- ✅ Quantify improvement in concrete terms
- ✅ Compare techniques objectively
- ✅ Detect regressions early
- ✅ Build confidence in production systems

## Prerequisites

Before starting, you should have:

- ✅ Completed `foundation/01-02` - Basic RAG working
- ✅ Completed `intermediate/03-04` - Registry patterns
- ⚠️ Optional: Completed some `advanced-techniques/05-X` - New techniques to evaluate

**Not required but helpful:**
- Familiarity with metrics like Precision, Recall, MRR
- Understanding of test set design
- Basic SQL and PostgreSQL

## Architecture Overview

The evaluation lab integrates with your PostgreSQL registry:

```
Your RAG System
        ↓
Retrieve + Generate Results
        ↓
Evaluate Against Ground-Truth → evaluation_groundtruth table
        ↓
Compute Metrics → evaluation_results table
        ↓
Compare to Baselines → experiments table
        ↓
Visualize in Dashboard
```

## Notebooks

### 01: Create Ground-Truth (Human-in-Loop)

**What you'll learn:**
- Design representative test sets
- Use interactive prompts to curate relevant chunks
- Rate answer quality with human judgment
- Create reproducible evaluation sets

**Key concepts:**

A ground-truth test set has:
- **Questions** - Diverse, representative queries
- **Relevant chunks** - Which passages should the retriever find
- **Quality ratings** - How good is each answer (1-5 scale)

**Data structure:**
```python
ground_truth = {
    'id': 'gt_001',
    'question': 'What is photosynthesis?',
    'relevant_chunk_ids': ['chunk_42', 'chunk_157'],
    'expected_answer': 'Process by which plants...',
    'quality_rating': 5,  # Human judgment
    'domain': 'biology'
}
```

**Workflow:**
```
1. Generate candidate questions (from your data)
2. For each question:
   a. Ask RAG to retrieve chunks
   b. Show human the retrieved chunks
   c. Ask: "Are these relevant?" (Yes/No for each)
   d. Ask: "Rate answer quality" (1-5)
3. Save curated set to evaluation_groundtruth table
```

**Time estimate:** 60-90 minutes (includes human curation time)

**Deliverable:** 20-50 high-quality test questions with ratings

---

### 02: Evaluation Metrics Framework

**What you'll learn:**
- Compute retrieval metrics (Precision, Recall, MRR, NDCG)
- Compute generation metrics (BLEU, ROUGE, exact match)
- Interpret what each metric means
- Know which metrics matter for your use case

**Key metrics:**

**Retrieval metrics** (did we find relevant chunks?):
- **Precision@K** - "Of top K results, how many are relevant?" (0-1)
- **Recall@K** - "Of all relevant chunks, how many did we find?" (0-1)
- **MRR** - "How high was the first relevant result?" (0-1)
- **NDCG** - "How well were results ranked?" (0-1)

**Generation metrics** (did we generate good answers?):
- **BLEU** - "How similar is answer to expected?" (0-1)
- **ROUGE** - "How much content overlap?" (0-1)
- **Exact Match** - "Is it exactly right?" (0-1)
- **Semantic similarity** - "Is meaning preserved?" (0-1)

**Code pattern:**
```python
def evaluate_retrieval(retrieved_chunks, relevant_chunk_ids):
    """Compute retrieval metrics."""
    precision = len(set(retrieved) & set(relevant)) / len(retrieved)
    recall = len(set(retrieved) & set(relevant)) / len(relevant)
    mrr = compute_mrr(retrieved, relevant)
    ndcg = compute_ndcg(retrieved, relevant)
    return {'precision': precision, 'recall': recall, 'mrr': mrr, 'ndcg': ndcg}

def evaluate_generation(generated_answer, expected_answer):
    """Compute generation metrics."""
    bleu = bleu_score(generated_answer, expected_answer)
    rouge = rouge_score(generated_answer, expected_answer)
    exact_match = (generated_answer == expected_answer)
    return {'bleu': bleu, 'rouge': rouge, 'exact_match': exact_match}
```

**Time estimate:** 20-30 minutes

**Deliverable:** Metrics framework in PostgreSQL with computation functions

---

### 03: Baseline and Comparison

**What you'll learn:**
- Establish baselines (simple retrieval without advanced techniques)
- Compare techniques objectively
- Quantify improvement margins
- Identify diminishing returns

**Workflow:**

```
Step 1: Baseline
  Run ground-truth queries with basic RAG → baseline_metrics

Step 2: Technique A
  Run with technique 05 (reranking) → technique_a_metrics
  
Step 3: Technique B
  Run with technique 06 (query expansion) → technique_b_metrics

Step 4: Compare
  print("Reranking improves Precision@5 by +0.12")
  print("Query expansion improves Recall by +0.15")
```

**Code pattern:**
```python
# Run baseline
baseline = run_rag(testset, technique=None)
baseline_metrics = evaluate_retrieval(baseline)

# Run technique
improved = run_rag(testset, technique='reranking')
improved_metrics = evaluate_retrieval(improved)

# Compare
delta = {
    k: improved_metrics[k] - baseline_metrics[k]
    for k in baseline_metrics.keys()
}
print(f"Improvement: {delta}")
```

**Time estimate:** 45-60 minutes (includes RAG execution time)

**Deliverable:** Comparison showing which techniques help most

---

### 04: Experiment Dashboard

**What you'll learn:**
- Visualize metrics over time
- Track technique improvements
- Identify best configurations
- Share results with stakeholders

**Dashboard components:**

1. **Metrics leaderboard** - Techniques ranked by quality
2. **Improvement over time** - How has the system improved?
3. **Trade-off plots** - Speed vs. Quality analysis
4. **Query difficulty heatmap** - Which questions are hardest?
5. **Technique comparison** - Head-to-head results

**Visualizations:**
```python
# Leaderboard
techniques = ['baseline', 'reranking', 'query_expansion', 'hybrid', 'combined']
precision_scores = [0.65, 0.77, 0.72, 0.78, 0.82]
plot_bar(techniques, precision_scores, title='Precision@5 Leaderboard')

# Trade-off
speed = [0.1, 2.5, 0.8, 3.2, 4.5]  # seconds
quality = [0.65, 0.77, 0.72, 0.78, 0.82]
plot_scatter(speed, quality, labels=techniques, title='Speed vs Quality')
```

**Time estimate:** 30-40 minutes

**Deliverable:** Interactive dashboard for exploring results

---

### 05: Supplemental Embedding Analysis (Optional)

**What you'll learn:**
- Analyze embedding quality directly
- Visualize embedding space
- Understand why certain queries work better
- Debug embedding model choices

**Content:**
- Embedding space visualization (t-SNE, UMAP)
- Similarity distribution analysis
- Model-specific insights
- Chunk clustering patterns

**When to use:**
- Advanced users wanting deep understanding
- Debugging why a technique doesn't work
- Choosing between embedding models
- Research and publication

**Time estimate:** 20-30 minutes

**Note:** This is optional - most users won't need it

---

## Integration with Registry

All notebooks use the PostgreSQL registry:

```
foundation/00-setup-postgres-schema.ipynb creates:
├── embedding_registry        ← Which models are available
├── evaluation_groundtruth    ← Your test set
├── experiments              ← Technique configurations
└── evaluation_results       ← Metrics from each run

Evaluation lab populates:
├── evaluation_groundtruth   ← Add test questions
├── experiments              ← Log technique runs
└── evaluation_results       ← Record metrics
```

## Data Collection Workflow

```
Day 1: Create ground-truth
  - curate_test_set() → 30 questions
  - save to evaluation_groundtruth

Day 2: Measure baseline
  - run_rag(testset, technique=None)
  - compute_metrics()
  - save to evaluation_results

Day 3: Test technique A
  - run_rag(testset, technique='reranking')
  - compute_metrics()
  - save to evaluation_results

Day 4: Test technique B
  - run_rag(testset, technique='query_expansion')
  - compute_metrics()
  - save to evaluation_results

Day 5: Compare and visualize
  - Compare all results in dashboard
  - Identify best technique
  - Document learnings
```

## Interpreting Metrics

### Metric meanings:

| Metric | Range | Good | Interpretation |
|--------|-------|------|-----------------|
| Precision@5 | 0-1 | >0.7 | Most top-5 results are relevant |
| Recall | 0-1 | >0.8 | Found most relevant chunks |
| MRR | 0-1 | >0.5 | First relevant result is high |
| NDCG | 0-1 | >0.7 | Results are well-ranked |
| BLEU | 0-1 | >0.4 | Answer has similar content |
| ROUGE | 0-1 | >0.5 | Answer overlaps with expected |

### When metrics disagree:

```
High Precision, Low Recall
→ System is conservative (few false positives)
→ Missing some relevant information
→ Solution: Query expansion

High Recall, Low Precision
→ System is liberal (many false positives)
→ Retrieving noise
→ Solution: Reranking

High Retrieval, Low Generation
→ Good chunks but bad answers
→ Solution: Better prompt, more context, or better LLM

Low Retrieval, High Generation
→ Lucky - good answers despite poor retrieval
→ Solution: Don't rely on it, it's not sustainable
```

## Quick Setup

To get started quickly:

```python
# 1. Load utilities
from foundation.utils import *

# 2. Create test set (or load existing)
testset = load_or_create_groundtruth(size=20)

# 3. Run baseline
baseline_results = evaluate_rag(testset, technique=None)

# 4. Run your technique
technique_results = evaluate_rag(testset, technique='your_technique')

# 5. Compare
print(compare_results(baseline_results, technique_results))
```

## Common Questions

**Q: How many test questions do I need?**
- Minimum: 10 (quick validation)
- Good: 30-50 (representative)
- Comprehensive: 100+ (statistical significance)

**Q: Should I reuse test set across techniques?**
- Yes! Same test set for fair comparison
- Save it in evaluation_groundtruth table
- New techniques use same questions

**Q: How often should I evaluate?**
- After each major technique
- When changing models
- Before deploying to production
- Monthly for monitoring

**Q: What if my metrics don't improve?**
- Check test set quality (maybe it's too easy/hard)
- Check metric computation (bugs?)
- Technique might not help your data
- Try different hyperparameters
- Combine with other techniques

## Expected Timeline

```
Session 1: Create ground-truth          (1-2 hours)
Session 2: Baseline measurement         (30-45 min)
Session 3: Technique A evaluation      (45-60 min)
Session 4: Technique B evaluation      (45-60 min)
Session 5: Comparison & dashboard      (30-45 min)
──────────────────────────────────────────────
Total: 3-5 hours for complete evaluation
```

## Next Steps After This Layer

1. **Continue experimenting** - Test more advanced-techniques/05-10
2. **Iterate on techniques** - Use metrics to guide optimization
3. **Production readiness** - Document best configuration
4. **Monitor over time** - Track metrics in production

## Resources

- `LEARNING_ROADMAP.md` - Full progression
- `EVALUATION_GUIDE.md` - Detailed metrics explanations
- `foundation/00-registry-and-tracking-utilities.ipynb` - Available functions
- `advanced-techniques/README.md` - What techniques to evaluate

---

**Difficulty:** ⭐⭐ Intermediate  
**Time commitment:** 3-5 hours total (all notebooks)  
**Most valuable first:** Notebook 01 (ground-truth) → then 02 → then 03 → 04
