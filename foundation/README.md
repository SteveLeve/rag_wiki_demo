# Foundation Layer: RAG Learning Path

This directory contains the foundation notebooks for learning RAG systems. Start here!

---

## 📚 Choose Your Starting Point

### **Option 1: Learn RAG Basics (Simple, In-Memory)**
👉 **Start with:** `01-basic-rag-in-memory.ipynb`

**What it covers:**
- RAG fundamentals without complexity
- Load Wikipedia dataset
- Generate embeddings with BGE model
- Implement similarity-based retrieval
- Generate answers with Llama model
- Best practices for chunking and prompting

**Requirements:**
- Ollama (with 2 models downloaded)
- Python packages: `ollama`, `datasets`, `jupyter`
- **No database needed** ✓

**Time:** ~30-45 minutes (including embedding generation)

**Good for:**
- First-time RAG learners
- Understanding core concepts
- Quick prototyping
- Environments without database access

---

### **Option 2: Build Production Systems (PostgreSQL + Registry)**
👉 **Prerequisites First:**
1. Run `00-setup-postgres-schema.ipynb` (creates database schema) - **5 minutes**
2. Then: `02-rag-postgresql-persistent.ipynb` (learn persistent storage)

**What it covers:**
- All of Option 1, PLUS:
- PostgreSQL + pgvector setup
- Persistent embedding storage
- Embedding registry for reuse
- PRESERVE_EXISTING_EMBEDDINGS pattern (avoid 50+ min regeneration)
- Multi-model storage and comparison

**Requirements:**
- Everything from Option 1, PLUS:
- PostgreSQL running (Docker: `pgvector/pgvector:pg16`)
- `psycopg2-binary` Python package

**Time:** ~1 hour total (5 min setup + 45 min notebook)

**Good for:**
- Learning production RAG patterns
- Building reusable embedding datasets
- Comparing multiple embedding models
- Foundation for intermediate/advanced notebooks

---

## 🔧 Infrastructure Notebooks (00-prefix)

These are **reference utilities**, not meant to be run standalone:

| Notebook | Purpose | Used By |
|----------|---------|---------|
| `00-setup-postgres-schema.ipynb` | Create database schema (4 tables, 6 indexes) | **Run once before foundation/02** |
| `00-registry-and-tracking-utilities.ipynb` | 20+ utility functions for discovery, tracking, metrics | Copy functions into your notebooks |
| `00-load-or-generate-pattern.ipynb` | Smart pattern: check registry, load if exists, generate if not | Reference for advanced notebooks |

**When to use infrastructure notebooks:**
- Running `00-setup-postgres-schema.ipynb`: Before your first `foundation/02` run
- Referencing others: Copy functions into your own notebooks (see examples in foundation/02)

---

## 📊 Schema Overview

When you run `00-setup-postgres-schema.ipynb`, it creates 4 tables:

```
embedding_registry
├─ Catalog of embedding models (384 / 768 / 1024-dim)
├─ Stores: dimension, chunk count, dataset source
└─ Used by: foundation/02, intermediate/03-04, advanced/05-10

evaluation_groundtruth
├─ Curated test questions (human reviewed)
├─ Stores: questions, relevant chunks, quality ratings
└─ Used by: evaluation-lab/01 (create), evaluation-lab/02-04 (measure)

experiments
├─ Track each technique/configuration run
├─ Stores: config hash, techniques applied, status
└─ Used by: All advanced notebooks

evaluation_results
├─ Metrics computed for each experiment
├─ Stores: Precision@K, Recall, NDCG, etc.
└─ Used by: evaluation-lab/02-04 (analyze)
```

---

## 🎓 Learning Progression

**Path A: Simple Only (1-2 hours)**
```
01-basic-rag-in-memory.ipynb
└─ Learn RAG fundamentals
└─ No database, just Python
└─ Good for: Understanding concepts
```

**Path B: Simple + Persistent (3-4 hours)**
```
00-setup-postgres-schema.ipynb (5 min)
    ↓
01-basic-rag-in-memory.ipynb (30 min)
    ↓
02-rag-postgresql-persistent.ipynb (45 min)
    ↓
You can now move to: intermediate/03-04
└─ Learn registry pattern
└─ Compare embedding models
└─ Prepare for advanced techniques
```

**Path C: Full Learning (6-8+ hours)**
```
[Path B] ← Complete this first
    ↓
intermediate/03-loading-and-reusing-embeddings.ipynb
    ↓
intermediate/04-comparing-embedding-models.ipynb
    ↓
advanced-techniques/05-10 (Reranking, Query Expansion, Hybrid Search, etc.)
    ↓
evaluation-lab/01-04 (Create test set, measure metrics, compare, visualize)
```

---

## ✅ Quick Checklist

### Before foundation/01
- [ ] Ollama installed
- [ ] Models downloaded: `ollama pull nomic-embed-text`
- [ ] Models downloaded: `ollama pull llama3.2:3b`
- [ ] Python packages installed: `pip install ollama datasets jupyter`

### Before foundation/02
- [ ] Complete everything from "Before foundation/01"
- [ ] PostgreSQL running (Docker recommended)
- [ ] `psycopg2-binary` installed: `pip install psycopg2-binary`
- [ ] `00-setup-postgres-schema.ipynb` run successfully

### Troubleshooting

**"ModuleNotFoundError: No module named 'ollama'"**
```bash
pip install ollama
```

**"Can't connect to Ollama on localhost:11434"**
- Make sure Ollama is running (might need to start the app)
- On Mac/Linux: `ollama serve` in a terminal

**"Failed to connect to PostgreSQL"**
- Start PostgreSQL container:
```bash
docker run -d --name pgvector-rag \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=rag_db \
  -p 5432:5432 \
  -v pgvector_data:/var/lib/postgresql/data \
  pgvector/pgvector:pg16
```

**"ModuleNotFoundError: No module named 'psycopg2'"**
```bash
pip install psycopg2-binary
```

---

## 📖 Core Concepts

### The RAG Pipeline (3 phases)

```
[USER QUESTION]
    ↓
📚 INDEXING (foundation/01-02)
   ├─ Load dataset (Wikipedia)
   ├─ Split into chunks
   ├─ Generate embeddings (BGE model)
   └─ Store in vector DB
    ↓
🔍 RETRIEVAL (retrieve function)
   ├─ Convert query to embedding
   ├─ Find similar chunks (cosine similarity)
   └─ Return top-k results
    ↓
💬 GENERATION (ask_question function)
   ├─ Build prompt with retrieved chunks
   ├─ Send to LLM (Llama model)
   └─ Stream answer back
    ↓
[GROUNDED ANSWER]
```

### Why Foundation/02 is Important

Foundation/01 loses embeddings on restart → Foundation/02 solves this by:
1. Storing embeddings in PostgreSQL (persistent)
2. Registering them in `embedding_registry` (discoverable)
3. Using load-or-generate pattern (reuse without regeneration)

**Time saved:** 50+ minutes per technique experiment!

---

## 🚀 Next Steps

1. **Pick a path** above (A, B, or C)
2. **Run foundation/01** to learn the basics
3. **Optionally run foundation/02** to learn persistent storage
4. **Move to intermediate/** to learn registry pattern
5. **Explore advanced-techniques/** for production-grade RAG patterns

---

## 📚 Files in This Directory

```
foundation/
├── README.md ← You are here
├── 00-setup-postgres-schema.ipynb (Infrastructure: Create DB schema)
├── 00-registry-and-tracking-utilities.ipynb (Reference: 20+ utility functions)
├── 00-load-or-generate-pattern.ipynb (Reference: Smart loading pattern)
├── 01-basic-rag-in-memory.ipynb (Tutorial: In-memory RAG with examples)
└── 02-rag-postgresql-persistent.ipynb (Tutorial: PostgreSQL + registry with examples)
```

---

## 🤔 FAQ

**Q: Do I need PostgreSQL to learn RAG?**
A: No! Start with foundation/01 (in-memory). Foundation/02 adds persistent storage if you want to experiment with multiple techniques.

**Q: Can I skip foundation/01 and go straight to foundation/02?**
A: You can, but foundation/01 is simpler and helps you understand the concepts. Foundation/02 is the same system with database persistence added.

**Q: How long does embedding generation take?**
A: ~1-2 minutes per 100 chunks with the BGE model. Once stored in PostgreSQL, you can reuse them across experiments (instant loading).

**Q: Why are there "00-" notebooks?**
A: They're infrastructure or utility libraries. The "00-setup" must run once to create the database. The others are reference implementations you copy functions from.

**Q: What's the registry pattern?**
A: Instead of regenerating embeddings for each notebook, you register them once in foundation/02, then other notebooks discover and reuse them. This saves 50+ minutes per experiment!

**Q: Should I delete the original notebooks?**
A: Yes! Their content has been moved to foundation/01-02. We kept git history for reference.
