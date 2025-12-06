# Project Documentation Index

## Getting Started

### I'm new to this project
Start here → **[README.md](./README.md)**
- Overview of the RAG system
- 5-minute quick start
- Architecture diagram
- Dataset specifications

### I want to set up PostgreSQL
Start here → **[POSTGRESQL_SETUP.md](./POSTGRESQL_SETUP.md)**
- Step-by-step Docker setup
- Connection configuration
- Data persistence strategies
- Troubleshooting guide

### I'm not sure which storage backend to use
Start here → **[QUICK_REFERENCE.md](./QUICK_REFERENCE.md)**
- Decision tree for backend choice
- Quick comparison table
- Common commands
- Example configurations

### I want to see working code examples
Start here → **[embedding-analysis-template.ipynb](./embedding-analysis-template.ipynb)**
- Load pre-computed embeddings
- Run retrieval quality analysis
- Compare embedding models
- Statistical analysis examples

## Project Files Overview

### Notebooks

| File | Purpose | Time to Run |
|------|---------|------------|
| `wikipedia-rag-tutorial-simple.ipynb` | **Simple tutorial** - Learn RAG fundamentals with in-memory storage | 50+ min (embeddings regenerate each run) |
| `wikipedia-rag-tutorial-advanced.ipynb` | **Advanced tutorial** - PostgreSQL for persistent embeddings | 50+ min first run, then 2-5 min for analysis |
| `embedding-analysis-template.ipynb` | **Experiment template** - Analyze stored embeddings, run comparisons | 2-5 min |

### Documentation

| File | Purpose | Read Time |
|------|---------|-----------|
| `README.md` | Project overview, quick start, architecture | 10 min |
| `POSTGRESQL_SETUP.md` | PostgreSQL/pgvector detailed setup guide | 10 min |
| `QUICK_REFERENCE.md` | Quick lookup guide, decision trees, commands | 5 min |
| `ENHANCEMENT_SUMMARY.md` | Summary of PostgreSQL integration features | 5 min |

### Data Files

| File | Purpose | Size |
|------|---------|------|
| `wikipedia_dataset_10mb.json` | Cached dataset (optional, saves download time) | ~10MB |
| `wikipedia_vectorize_export.json` | Example export format for other platforms | ~20MB |

---

## Workflow Paths

### Path A: Quick Learning (In-Memory)
```
1. Read: README.md (5 min)
2. Run: wikipedia-rag-tutorial-simple.ipynb
3. Done! Understand RAG basics
Time: ~1 hour total
```

### Path B: Single Deep Dive (PostgreSQL)
```
1. Read: README.md (5 min)
2. Read: POSTGRESQL_SETUP.md (10 min)
3. Start PostgreSQL with Docker (1 min)
4. Run: wikipedia-rag-tutorial-advanced.ipynb
5. Create analysis notebook from template
Time: ~2 hours total
```

### Path C: Model Comparison (PostgreSQL + Multiple Experiments)
```
1. Read: README.md (5 min)
2. Read: QUICK_REFERENCE.md (5 min)
3. Start PostgreSQL with Docker (1 min)
4. Run: wikipedia-rag-tutorial-advanced.ipynb with Model A
5. Modify & run: wikipedia-rag-tutorial-advanced.ipynb with Model B
6. Create: analysis notebook comparing both models
Time: ~3 hours total
```

---

## Key Concepts

### Storage Backends

**In-Memory** (`'memory'`)
- Perfect for: Learning, quick testing
- Setup: None needed
- Data persistence: Lost on restart

**JSON** (`'json'`)
- Perfect for: Archiving small datasets
- Setup: None needed
- Data persistence: Saved to file

**PostgreSQL** (`'postgresql'`)
- Perfect for: Multi-experiment workflows
- Setup: Docker (2 minutes)
- Data persistence: Durable, queryable

See **[QUICK_REFERENCE.md](./QUICK_REFERENCE.md)** for detailed comparison.

### Table Naming for Multiple Models

Embeddings are stored in tables named: `embeddings_{model_alias}`

Examples:
- `embeddings_bge_base_en_v1_5` (from `EMBEDDING_MODEL_ALIAS = 'bge_base_en_v1.5'`)
- `embeddings_all_minilm_l6_v2` (from `EMBEDDING_MODEL_ALIAS = 'all_minilm_l6_v2'`)

This lets you store and compare multiple models without conflicts.

### pgvector for Similarity Search

pgvector provides:
- HNSW indexing for fast vector search (~1ms queries)
- Native cosine similarity operators
- Automatic index creation
- Scalability to millions of vectors

---

## Common Tasks

### "I want to start fresh"
```bash
# Reset everything
docker stop pgvector-rag
docker volume rm pgvector_data
rm wikipedia_dataset_10mb.json
```

### "I want to compare two embedding models"
1. See **[QUICK_REFERENCE.md](./QUICK_REFERENCE.md)** - "Example 4: Compare Two Models"
2. See **[embedding-analysis-template.ipynb](./embedding-analysis-template.ipynb)** - Section "2. Compare Embedding Models"

### "I want to understand my retrieval quality"
1. See **[embedding-analysis-template.ipynb](./embedding-analysis-template.ipynb)** - Section "1. Analyze Query Performance"

### "I'm seeing slow queries"
1. See **[POSTGRESQL_SETUP.md](./POSTGRESQL_SETUP.md)** - "Troubleshooting" section
2. Check if pgvector index was created (it's created automatically)

### "I want to backup my embeddings"
1. See **[POSTGRESQL_SETUP.md](./POSTGRESQL_SETUP.md)** - "Data Persistence" section

### "I want to move to production"
1. See **[POSTGRESQL_SETUP.md](./POSTGRESQL_SETUP.md)** - "Advanced: Using Other Vector Databases" section
2. Also see **README.md** - "Production Deployment" section

---

## Configuration Quick Reference

### Minimal Configuration (In-Memory)
```python
STORAGE_BACKEND = 'memory'
TARGET_SIZE_MB = 10
```

### Recommended Configuration (PostgreSQL)
```python
STORAGE_BACKEND = 'postgresql'
TARGET_SIZE_MB = 10
EMBEDDING_MODEL_ALIAS = 'bge_base_en_v1.5'

POSTGRES_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'rag_db',
    'user': 'postgres',
    'password': 'postgres',
}
```

### Full Configuration (All Options)
See notebooks for all available options:
- **Simple**: `wikipedia-rag-tutorial-simple.ipynb` - Configuration cell
- **Advanced**: `wikipedia-rag-tutorial-advanced.ipynb` - Configuration cell

---

## Learning Path for RAG Concepts

1. **RAG Fundamentals** (30 min)
   - Read: README.md - "🏗️ Architecture" section
   - Run: wikipedia-rag-tutorial-simple.ipynb cells 1-10 (data loading)

2. **Embeddings** (20 min)
   - Run: wikipedia-rag-tutorial-simple.ipynb cells 11-15 (embedding generation)
   - Understand: Vector similarity and cosine distance

3. **Retrieval** (15 min)
   - Run: wikipedia-rag-tutorial-simple.ipynb cell 18 (retrieve function)
   - Test: Different query types

4. **Generation** (15 min)
   - Run: wikipedia-rag-tutorial-simple.ipynb cell 20 (ask_question function)
   - Understand: Prompt construction and context feeding

5. **Optimization** (30 min)
   - Use: embedding-analysis-template.ipynb for experiments
   - Compare: Different configurations and models

---

## Troubleshooting Decision Tree

```
Does it work locally with 'memory' backend?
├─ NO
│  └─ Check: Ollama is running, models are downloaded
│     Run in terminal: ollama list
│
└─ YES
   │
   ├─ Want to use PostgreSQL?
   │  ├─ NO → You're done! Enjoy development.
   │  │
   │  └─ YES
   │     ├─ Docker installed?
   │     │  ├─ NO → Install Docker Desktop
   │     │  └─ YES → See: POSTGRESQL_SETUP.md
   │     │
   │     └─ Getting connection error?
   │        └─ See: POSTGRESQL_SETUP.md - "Troubleshooting"
```

---

## File Structure

```
rag_wiki_demo/
├── README.md                              # Start here
├── ENHANCEMENT_SUMMARY.md                 # What's new
├── POSTGRESQL_SETUP.md                    # Detailed PostgreSQL guide
├── QUICK_REFERENCE.md                     # Quick lookups
├── wikipedia-rag-tutorial-simple.ipynb    # Simple version (in-memory)
├── wikipedia-rag-tutorial-advanced.ipynb  # Advanced version (PostgreSQL)
├── embedding-analysis-template.ipynb      # Experiment template
├── wikipedia_dataset_10mb.json            # Cached data (generated)
└── wikipedia_vectorize_export.json        # Export example (generated)
```

---

## Getting Help

| Issue | Solution |
|-------|----------|
| "I don't understand RAG" | Read README.md → "RAG System Fundamentals" |
| "How do I set up PostgreSQL?" | Read POSTGRESQL_SETUP.md |
| "Which storage backend should I use?" | Read QUICK_REFERENCE.md → "Storage Backend Decision Tree" |
| "How do I load existing embeddings?" | See embedding-analysis-template.ipynb → cell 8 |
| "I get a connection error" | Read POSTGRESQL_SETUP.md → "Troubleshooting" |
| "What are my next steps?" | See this file → "Workflow Paths" |

---

## Quick Start (TL;DR)

```bash
# 1. In-memory (simplest)
jupyter notebook wikipedia-rag-tutorial-simple.ipynb
# Run all cells

# 2. With PostgreSQL (recommended for experiments)
docker run -d --name pgvector-rag \
  -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=rag_db \
  -p 5432:5432 -v pgvector_data:/var/lib/postgresql/data \
  pgvector/pgvector:pg16

pip install psycopg2-binary

jupyter notebook wikipedia-rag-tutorial-advanced.ipynb
# Run all cells

# 3. Analyze (create new notebook from template)
jupyter notebook embedding-analysis-template.ipynb
```

---

**Last Updated**: December 2025
**PostgreSQL Integration**: ✅ Complete
