# Evaluation Guide: Building and Using Test Sets

This guide covers how to create ground-truth evaluation datasets, understand metrics, and systematically compare RAG configurations.

## Overview

Evaluating a RAG system requires:

1. **Ground-Truth Test Set** — Curated questions with known-good answers
2. **Retrieval Metrics** — Measure if the right documents are retrieved
3. **Generation Metrics** — Measure if the answer is good quality
4. **Experiment Tracking** — Compare configurations reproducibly

## Part 1: Creating Ground-Truth Test Sets

### Why Ground-Truth Matters

A ground-truth test set enables you to:
- Objectively measure retrieval quality
- Compare different embedding models
- Evaluate the impact of each technique
- Validate that changes improve the system
- Catch regressions before deployment

### Ground-Truth Dataset Structure

Each test case includes:

```python
{
    "question": "What is photosynthesis?",
    "relevant_chunk_ids": [42, 87, 105],  # IDs of correct chunks
    "quality_rating": "good",              # "good", "bad", "ambiguous", "rejected"
    "source_type": "llm_generated",        # How it was created
    "human_notes": "Clear and specific question"
}
```

Stored in: `evaluation_groundtruth` PostgreSQL table

### Creating Test Sets: Human-in-the-Loop Workflow

Run: `evaluation-lab/01-create-ground-truth-human-in-loop.ipynb`

**Phase 1: Synthetic Test Generation**

Two generation methods (use one or both):

#### Option A: LLM-Generated Questions
```python
for chunk in dataset_sample:
    prompt = f"""
    Given this Wikipedia article excerpt:
    
    {chunk}
    
    Generate 3 factual questions this excerpt could answer.
    Return as JSON list: [question1, question2, question3]
    """
    
    questions = llm.generate(prompt)
    candidates.extend([(q, 'llm_generated', chunk_id) for q in questions])
```

**Pros:** Creative, diverse questions; good coverage  
**Cons:** May generate unanswerable questions

#### Option B: Template-Based Questions
```python
templates = [
    "What is {noun}?",
    "How does {verb} work?",
    "Where is {place}?",
    "Why do {subject} {verb}?",
]

for chunk in dataset_sample:
    entities = extract_entities(chunk)  # nouns, verbs, places
    for template in templates:
        if applicable_entities(template, entities):
            question = fill_template(template, entities)
            candidates.append((question, 'template_based', chunk_id))
```

**Pros:** Consistent, reproducible  
**Cons:** More formulaic, less variety

#### Blended Approach (Recommended)
Generate with both methods, deduplicate on question text, present mixed set to human curator.

### Curation Interface

Interactive prompt for each synthetic question:

```
Question: "What is photosynthesis?"
Source Chunk Preview: "Photosynthesis is the process by which plants..."

Rate this question:
[g]ood   — Clear, answerable from retrieved chunks
[b]ad    — Unanswerable or misleading  
[a]mbiguous — Could be interpreted multiple ways
[r]eject — Don't include in test set
[n]otes  — Add curator notes
[s]kip   — Come back later

Your input: 
```

### Best Practices for Curation

1. **Aim for 50-100 good test cases** minimum for meaningful metrics
2. **Aim for diversity:**
   - Different topics (if using diverse dataset)
   - Different question types (what, how, why, where, when)
   - Different difficulty levels (factual vs. synthetic)

3. **Be consistent:** Curators should agree on ratings
   - Low inter-curator agreement → questions are genuinely ambiguous
   - High agreement → rating scheme is working

4. **Document decisions:** Use "notes" field to explain borderline cases
   - Helps with future curation
   - Makes metrics interpretation easier

5. **Save frequently:** Curation is persisted to DB after each rating
   - No risk of losing work
   - Can pause and resume

### Quality Levels Explained

| Rating | Meaning | Usage |
|--------|---------|-------|
| **good** | Clear question, answerable from dataset | Use in main evaluation |
| **bad** | Unanswerable or misleading | Filter out completely |
| **ambiguous** | Could have multiple interpretations | Optional: separate analysis |
| **rejected** | Curator rejected for any reason | Filter out completely |

---

## Part 2: Retrieval Metrics

Measure how well your system retrieves relevant documents.

### Precision@K

**Definition:** Of the top K retrieved chunks, what % are relevant?

$$\text{Precision@K} = \frac{\text{# relevant chunks in top K}}{\text{K}}$$

**Interpretation:**
- 0.8 = 80% of top results are relevant (4 out of 5)
- Higher is better
- Measures "does system avoid noise"

**When to use:** When false positives are expensive (noisy context hurts generation)

**Code:**
```python
def precision_at_k(retrieved_ids: list, relevant_ids: list, k: int) -> float:
    top_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    if k == 0:
        return 0.0
    return len(top_k & relevant) / k
```

### Recall@K

**Definition:** Of all relevant chunks, what % appear in top K?

$$\text{Recall@K} = \frac{\text{# relevant chunks in top K}}{\text{# total relevant chunks}}$$

**Interpretation:**
- 0.6 = 60% of all correct chunks are in top results
- Higher is better
- Measures "does system find all relevant documents"

**When to use:** When missing information is problematic (need comprehensive answers)

**Code:**
```python
def recall_at_k(retrieved_ids: list, relevant_ids: list, k: int) -> float:
    top_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    if len(relevant) == 0:
        return 1.0  # Perfect recall if no relevant docs
    return len(top_k & relevant) / len(relevant)
```

### Mean Reciprocal Rank (MRR)

**Definition:** Average rank of the first relevant result

$$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$

Where rank_i is position of first relevant result for query i.

**Interpretation:**
- 1.0 = Perfect (first result always relevant)
- 0.5 = First relevant result is position 2 on average
- Higher is better
- Favors early relevant results

**When to use:** When first match matters (user satisfaction)

**Code:**
```python
def mrr(ranked_results: list, relevant_ids: set) -> float:
    for rank, item_id in enumerate(ranked_results, 1):
        if item_id in relevant_ids:
            return 1.0 / rank
    return 0.0  # No relevant result found
```

### Normalized Discounted Cumulative Gain (NDCG)

**Definition:** Weighted ranking considering all positions

$$\text{NDCG@K} = \frac{\text{DCG@K}}{\text{Ideal DCG@K}}$$

Where DCG = $\sum_{i=1}^{k} \frac{2^{\text{relevance}_i} - 1}{\log_2(i+1)}$

**Interpretation:**
- 1.0 = Perfect ranking
- 0.7 = Good ranking (some errors)
- Higher is better
- Accounts for partial relevance (some docs more relevant than others)

**When to use:** Comprehensive ranking evaluation

**Code:** See evaluation-lab/02 notebook for implementation

---

## Part 3: Generation Metrics

Measure quality of generated answers (not just retrieval).

### BLEU Score

**Definition:** Overlap of n-grams between generated and reference text

$$\text{BLEU} = \text{BP} \times \exp\left(\sum_{n=1}^{4} w_n \log p_n\right)$$

Where BP is brevity penalty, p_n is precision of n-grams.

**Interpretation:**
- 0-1 scale (0 = no overlap, 1 = perfect match)
- Typical RAG: 0.15-0.30 (not word-for-word match)
- Higher is better
- **Limitation:** Penalizes paraphrasing

**When to use:** When you have reference answers to compare against

**Requires:** Ground-truth reference answers (not just questions)

### ROUGE Score

**Definition:** Recall of n-grams (reverse of BLEU)

$$\text{ROUGE-L} = F_{\text{lcs}} = \frac{(1 + \beta^2) P_{\text{lcs}} R_{\text{lcs}}}{\beta^2 P_{\text{lcs}} + R_{\text{lcs}}}$$

Where lcs is longest common subsequence.

**Interpretation:**
- 0-1 scale
- Better handles paraphrasing than BLEU
- Higher is better

**When to use:** Evaluating answer similarity to reference

**Requires:** Ground-truth reference answers

### LLM-as-Judge

**Definition:** Use another LLM to score relevance/quality

```python
prompt = f"""
Question: {question}
Reference Answer: {reference_answer}
Generated Answer: {generated_answer}

Rate the generated answer on:
1. Relevance: Does it address the question?
2. Factuality: Is it consistent with the reference?
3. Completeness: Does it cover key points?
4. Clarity: Is it well-written and understandable?

Score overall: 1 (poor) to 5 (excellent)
Reasoning: [explain your rating]
"""

score = llm.evaluate(prompt)
```

**Interpretation:**
- 1-5 scale
- 1-2 = Poor quality
- 3 = Acceptable
- 4-5 = Good to excellent

**Advantages:**
- Flexible (can define custom criteria)
- Handles paraphrasing well
- No reference answers needed
- Human-interpretable reasoning

**Disadvantages:**
- Expensive (each question = 1 LLM call)
- LLM bias affects scores
- Slower than automatic metrics

**When to use:** When automatic metrics insufficient; final quality verification

---

## Part 4: Experiment Tracking & Comparison

### What Gets Tracked

Each notebook run creates an experiment record with:

```python
EXPERIMENT_CONFIG = {
    'reranking_enabled': True,
    'reranking_model': 'cross-encoder-ms-marco-miniLM-L-6-v2',
    'query_expansion_enabled': True,
    'query_expansion_variants': 3,
    'top_n': 5,
    'similarity_threshold': 0.5,
    'chunk_size': 1000,
}

experiment_id = start_experiment(
    db,
    experiment_name='RAG with reranking + query expansion',
    notebook_path='advanced-techniques/10-combined-advanced-rag.ipynb',
    embedding_model_alias='bge_base_en_v1_5',
    config=EXPERIMENT_CONFIG,
    techniques=['reranking', 'query_expansion']
)
```

Stored in: `experiments` table with config_hash for reproducibility

### Computing & Storing Metrics

At end of evaluation notebook:

```python
metrics = {
    'precision_at_3': 0.82,
    'precision_at_5': 0.78,
    'recall_at_5': 0.71,
    'mrr': 0.65,
    'ndcg_at_10': 0.78,
    'bleu': {'value': 0.24, 'details': {}},
    'llm_judge_score': {'value': 4.1, 'details': {'reasoning': '...'}}
}

save_metrics(db, experiment_id, metrics, export_to_file=True)
```

Stored in:
- `evaluation_results` table (queryable)
- `data/experiment_results/` (shareable JSON files)

### Comparing Experiments

```python
# Get specific experiments
exp1 = get_experiment(db, experiment_id=42)
exp2 = get_experiment(db, experiment_id=43)

# Side-by-side comparison
comparison = compare_experiments(db, [42, 43], 
                                metric_names=['precision_at_5', 'recall_at_5', 'mrr'])
print(comparison)
# Output:
#   id  experiment_name  embedding_model  precision_at_5  recall_at_5   mrr
#   42  Baseline         bge_base_en      0.78            0.71         0.65
#   43  With reranking   bge_base_en      0.85            0.79         0.72
```

### Reproducibility: Re-running Experiments

```python
# Find experiment with specific configuration
past_exp = get_experiment(db, experiment_id=42)
original_config = past_exp['config']

# Re-run with identical configuration
experiment_id = start_experiment(
    db,
    experiment_name='Repeat of exp #42',
    config=original_config,
    techniques=past_exp['techniques']
)

# Should get very similar (or identical) metrics
```

---

## Part 5: Common Evaluation Patterns

### Pattern 1: Baseline Establishment

**Goal:** Establish single-technique baseline for comparison

```python
# Baseline: Simple vector search, no techniques
baseline_config = {
    'reranking_enabled': False,
    'query_expansion_enabled': False,
    'hybrid_search_enabled': False,
    'top_n': 5,
}

exp_baseline = start_experiment(db, 'Baseline (vector search)', config=baseline_config)
# ... run evaluation on test set ...
save_metrics(db, exp_baseline, baseline_metrics)

# Improvement 1: Add reranking
config_with_reranking = {**baseline_config, 'reranking_enabled': True}
exp_rerank = start_experiment(db, 'Baseline + reranking', config=config_with_reranking)
# ... run evaluation ...
save_metrics(db, exp_rerank, reranking_metrics)

# Compare impact
print(compare_experiments(db, [exp_baseline, exp_rerank]))
```

### Pattern 2: Parameter Tuning

**Goal:** Find optimal threshold for a technique

```python
results = []

for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    config = {
        'similarity_threshold': threshold,
        'query_expansion_enabled': True,
    }
    
    exp_id = start_experiment(db, f'QE with threshold={threshold}', config=config)
    metrics = evaluate_on_groundtruth(db, groundtruth_set)
    save_metrics(db, exp_id, metrics)
    
    results.append((threshold, metrics['mrr']))

# Plot: threshold vs. MRR
import matplotlib.pyplot as plt
thresholds, mrrs = zip(*results)
plt.plot(thresholds, mrrs, marker='o')
plt.xlabel('Similarity Threshold')
plt.ylabel('MRR')
plt.title('Parameter Tuning: Query Expansion')
plt.show()
```

### Pattern 3: Embedding Model Comparison

**Goal:** Test different embedding models on same ground-truth

```python
models = ['bge_base_en_v1_5', 'bge_small_en_v1_5', 'all_minilm_l6_v2']

comparison_results = {}

for model_alias in models:
    # Verify model is available in registry
    metadata = get_embedding_metadata(db, model_alias)
    
    config = {
        'embedding_model': model_alias,
        'top_n': 5,
    }
    
    exp_id = start_experiment(
        db,
        f'Embedding model: {model_alias}',
        embedding_model_alias=model_alias,
        config=config
    )
    
    metrics = evaluate_on_groundtruth(db, groundtruth_set)
    save_metrics(db, exp_id, metrics)
    comparison_results[model_alias] = metrics

# Compare all models
df = compare_experiments(db, [exp.id for exp in experiments], 
                         metric_names=['precision_at_5', 'recall_at_5', 'mrr'])
print(df)
```

---

## Part 6: Interpreting Results

### Good System Characteristics

| Metric | Benchmark | Interpretation |
|--------|-----------|-----------------|
| Precision@5 | > 0.70 | ~3.5 out of 5 results relevant |
| Recall@5 | > 0.60 | Finding 60%+ of all relevant chunks |
| MRR | > 0.60 | First relevant result in top 2-3 |
| NDCG@10 | > 0.70 | Good ranking with minimal errors |
| LLM Judge | > 3.5/5 | Acceptable answer quality |

### Trade-offs to Expect

- **More techniques = better metrics but slower queries**
  - Each technique adds latency
  - Diminishing returns after 2-3 techniques

- **Reranking improves quality but hurts latency**
  - Precision@5 may increase 10-15%
  - Adds 100-500ms per query

- **Query expansion improves recall but adds noise**
  - Recall@K may increase 5-10%
  - Precision may decrease slightly

- **Larger top_n = better recall but worse generation**
  - Recall@K increases (retrieving more)
  - More context confuses LLM (quality down)
  - Optimal typically 3-5 for generation

### Debugging Poor Results

| Symptom | Likely Cause | Solution |
|---------|-------------|----------|
| Low Precision@5 | Poor embedding model | Try different model (see comparison notebook) |
| Low Recall@K | Insufficient top_n | Increase top_n parameter |
| Low MRR | Relevant docs ranked low | Try reranking or query expansion |
| Low NDCG | Ranking inconsistent | Debug similarity scores, check chunking |
| Low LLM judge score | Bad context or prompt | Check retrieved chunks, refine prompt |

---

## Checklist: Complete Evaluation

- [ ] Ground-truth test set created (50+ questions)
  - [ ] ~70% from "good" ratings
  - [ ] Coverage of different topics/question types
  
- [ ] Retrieval metrics computed on test set
  - [ ] Precision@3, Precision@5
  - [ ] Recall@5, Recall@10
  - [ ] MRR, NDCG@10
  
- [ ] Generation metrics computed
  - [ ] LLM-as-judge scores (prefer this)
  - [ ] Or BLEU/ROUGE if reference answers available
  
- [ ] Baseline established
  - [ ] Simple vector search baseline metrics documented
  
- [ ] At least 3 configurations compared
  - [ ] Baseline vs. +technique vs. +techniques
  - [ ] Results show improvement path
  
- [ ] Experiment tracking complete
  - [ ] Config hashes for reproducibility
  - [ ] Metrics exported to file + DB
  - [ ] Notes on findings documented

---

## Next Steps

1. **Run ground-truth curation:** `evaluation-lab/01-create-ground-truth-human-in-loop.ipynb`
2. **Compute metrics:** `evaluation-lab/02-evaluation-metrics-framework.ipynb`
3. **Compare configurations:** `evaluation-lab/03-baseline-and-comparison.ipynb`
4. **Visualize results:** `evaluation-lab/04-experiment-dashboard.ipynb`

For more details on specific techniques and their evaluation, see individual technique notebooks in `advanced-techniques/`.
