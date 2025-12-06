# Quick Reference: Storage Backends

## Choosing a Storage Backend

### In-Memory (`'memory'`)
```python
STORAGE_BACKEND = 'memory'
```
✅ **Use when:**
- Quick prototyping and testing
- Small datasets (< 10MB)
- You don't need to save embeddings

⏱️ **Time to first query:** < 1 second (embeddings are not saved)

### JSON File (`'json'`)
```python
STORAGE_BACKEND = 'json'
```
✅ **Use when:**
- Want to save embeddings locally
- Sharing datasets via file
- Small datasets where you load everything into memory

⏱️ **Time to first query:** Depends on file size, all data loaded on notebook start

### PostgreSQL + pgvector (`'postgresql'`)
```python
STORAGE_BACKEND = 'postgresql'
```
✅ **Use when:**
- Want durable, searchable storage
- Running multiple experiments/notebooks
- Comparing different embedding models
- Planning to deploy to production

⏱️ **Time to first query:** < 1 second (no data loading needed)

---

## Storage Backend Decision Tree

```
Do you need embeddings to survive
notebook restarts?
├─ NO → Use 'memory' (simplest)
└─ YES
    ├─ Plan to run experiments
    │  on the same embeddings?
    │  ├─ NO → Use 'json' (portable)
    │  └─ YES
    │      ├─ Single embedding model → Use 'postgresql'
    │      └─ Multiple models → Use 'postgresql' (separate tables)
    └─ Want to eventually move
       to production?
       └─ YES → Use 'postgresql' (easier migration)
```

---

## Quick Commands

### Start PostgreSQL
```bash
docker run -d --name pgvector-rag \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=rag_db \
  -p 5432:5432 \
  -v pgvector_data:/var/lib/postgresql/data \
  pgvector/pgvector:pg16
```

### Stop PostgreSQL (data preserved)
```bash
docker stop pgvector-rag
```

### Restart PostgreSQL
```bash
docker start pgvector-rag
```

### Remove PostgreSQL (destroys data)
```bash
docker rm pgvector-rag
docker volume rm pgvector_data
```

### View PostgreSQL logs
```bash
docker logs pgvector-rag
```

---

## Notebook Configuration Examples

### Example 1: Quick Testing (In-Memory)
```python
# In wikipedia-rag-tutorial-simple.ipynb
TARGET_SIZE_MB = 10
SAVE_LOCALLY = False
```

### Example 2: Archive Local Dataset (JSON)
```python
# In wikipedia-rag-tutorial-simple.ipynb
TARGET_SIZE_MB = 10
SAVE_LOCALLY = True
LOCAL_DATASET_PATH = 'wikipedia_dataset_10mb.json'
```

### Example 3: Production Experiments (PostgreSQL)
```python
# In wikipedia-rag-tutorial-advanced.ipynb
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

### Example 4: Compare Two Models (PostgreSQL)
```python
# First run (wikipedia-rag-tutorial-advanced.ipynb)
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
EMBEDDING_MODEL_ALIAS = 'bge_base_en_v1.5'

# Second run (copy of advanced notebook with different model)
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-small-en-v1.5-gguf'
EMBEDDING_MODEL_ALIAS = 'bge_small_en_v1.5'

# Results stored in separate tables for comparison
```

---

## Loading Embeddings in New Notebooks

### From JSON
```python
with open('wikipedia_dataset_10mb.json', 'r') as f:
    data = json.load(f)
    dataset = data['chunks']
# Re-embed or load VECTOR_DB from data
```

### From PostgreSQL
```python
db = PostgreSQLVectorDB(POSTGRES_CONFIG, 'embeddings_bge_base_en_v1_5')
results = db.similarity_search(query_embedding, top_n=3)
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: psycopg2` | Run `pip install psycopg2-binary` |
| PostgreSQL connection refused | Run `docker start pgvector-rag` |
| Table not found error | Run main notebook first to generate embeddings |
| Docker command not found | Install [Docker Desktop](https://www.docker.com/) |
| Slow embedding generation | Use smaller embedding model or smaller dataset |
| Out of memory | Reduce `TARGET_SIZE_MB` or use `STORAGE_BACKEND = 'postgresql'` |

---

## Production Migration Path

```
Local Development        →  Production
┌──────────────────────┐     ┌──────────────────────┐
│ In-Memory Vector DB  │     │ PostgreSQL/pgvector  │
│ (STORAGE_BACKEND='   │     │ + Application Server │
│  memory')            │     │ (Heroku/Railway)     │
└──────────────────────┘     └──────────────────────┘
             │                          │
             └──────────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │  Neon (PostgreSQL)   │
                 │  + Vercel (Frontend) │
                 └──────────────────────┘
```

Use PostgreSQL locally from the start to reduce migration friction.
