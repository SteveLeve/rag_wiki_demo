# PostgreSQL & pgvector Setup Guide

## Overview

This guide explains how to use PostgreSQL with pgvector to store and reuse embeddings across multiple experiments, avoiding the 50+ minute regeneration time.

## Why PostgreSQL + pgvector?

- **Durable Storage**: Embeddings persist across notebook restarts
- **Multiple Models**: Store embeddings from different models in separate tables for comparison
- **Efficient Similarity Search**: pgvector provides HNSW indexing for fast vector retrieval
- **Experiment Friendly**: Generate embeddings once, analyze them many times

## Quick Start

### 1. Start PostgreSQL with Docker

```bash
docker run -d --name pgvector-rag \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=rag_db \
  -p 5432:5432 \
  -v pgvector_data:/var/lib/postgresql/data \
  pgvector/pgvector:pg16
```

This command:
- Runs PostgreSQL with pgvector extension pre-installed
- Creates a database named `rag_db`
- Stores data in a persistent Docker volume (`pgvector_data`)
- Exposes the database on `localhost:5432`

**To stop the container later:**
```bash
docker stop pgvector-rag
```

**To restart it (data will be preserved):**
```bash
docker start pgvector-rag
```

### 2. Choose Your Learning Path

We provide two separate foundation notebooks:

**`foundation/01-basic-rag-in-memory.ipynb`** (In-Memory)
- Uses simple Python lists for storage
- No database setup required
- Perfect for learning RAG fundamentals
- Embeddings lost on notebook restart
- Ideal for: Quick experiments, learning concepts

**`foundation/02-rag-postgresql-persistent.ipynb`** (PostgreSQL + Registry)
- Uses PostgreSQL with pgvector for persistent storage
- **Automatically registers embeddings in the registry**
- Supports efficient similarity search with HNSW indexing
- Embeddings reusable across notebooks
- Ideal for: Production workflows, reproducible experiments, embedding reuse

### 3. Run the PostgreSQL Notebook

If you choose to use PostgreSQL, just open and run `foundation/02-rag-postgresql-persistent.ipynb`:

1. The notebook will automatically:
   - Create a PostgreSQL table named `embeddings_bge_base_en_v1_5` (based on your embedding model)
   - Generate embeddings and store them directly in PostgreSQL
   - Register the embeddings in the `embedding_registry` table
   - Print a success message with the registry ID

2. The notebook will ask whether to preserve existing embeddings:
   - First run: Generates new embeddings
   - Second run: Reuses existing embeddings (saves 50+ minutes!)
   - You can override via `PRESERVE_EXISTING_EMBEDDINGS` setting

### 4. Use Registered Embeddings in Advanced Notebooks

Once embeddings are registered, advanced technique notebooks can discover and reuse them:

```python
# In advanced-techniques/05-10 notebooks
from foundation.load_or_generate_pattern import load_or_generate

# This will find the registered embeddings instead of regenerating
embeddings = load_or_generate(
    db=postgres_connection,
    embedding_model='bge-base-en-v1.5',
    embedding_alias='bge_base_en_v1.5',
    preserve_existing=True  # Always use existing
)
```

See `intermediate/03-loading-and-reusing-embeddings.ipynb` for detailed examples.

## Multiple Embedding Models

You can store embeddings from different models for comparison. The registry makes this easy:

### Option 1: Run foundation/02 Multiple Times

**First time with default model:**
```python
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
EMBEDDING_MODEL_ALIAS = 'bge_base_en_v1.5'
PRESERVE_EXISTING_EMBEDDINGS = False  # Generate new
```

Creates table: `embeddings_bge_base_en_v1_5`  
Registers in: `embedding_registry` with alias `bge_base_en_v1.5`

**Second time with different model:**
```python
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
EMBEDDING_MODEL_ALIAS = 'all_minilm_l6_v2'
PRESERVE_EXISTING_EMBEDDINGS = False  # Generate new
```

Creates table: `embeddings_all_minilm_l6_v2`  
Registers in: `embedding_registry` with alias `all_minilm_l6_v2`

### Option 2: Compare Models in Advanced Notebooks

Use `intermediate/04-comparing-embedding-models.ipynb` to:

```python
# Discover both models from registry
available_models = list_available_embeddings(db)
# Shows both bge_base_en_v1.5 and all_minilm_l6_v2

# Load both from registry (reuses stored embeddings)
bge_embeddings = load_or_generate(db, model, 'bge_base_en_v1.5', preserve_existing=True)
minilm_embeddings = load_or_generate(db, model, 'all_minilm_l6_v2', preserve_existing=True)

# Compare retrieval results for the same queries
for query in test_queries:
    bge_results = retrieve_with_model(query, bge_embeddings, top_k=5)
    minilm_results = retrieve_with_model(query, minilm_embeddings, top_k=5)
    compare_results(bge_results, minilm_results)
```

## Foundation Notebook Comparison

### Foundation 01: In-Memory (Simple)
- **Storage**: Python lists and dictionaries
- **Persistence**: Lost on notebook restart
- **Setup**: None required
- **Speed**: Fastest
- **Reuse**: Not supported
- **Best for**: Learning fundamentals, quick experiments
- **Registry**: Not used

### Foundation 02: PostgreSQL (Persistent)
- **Storage**: PostgreSQL with pgvector
- **Persistence**: Durable, survives restarts
- **Setup**: Docker required
- **Speed**: Slower generation, instant reuse
- **Reuse**: Yes - embeddings registered in registry
- **Best for**: Production workflows, multiple experiments, embedding reuse
- **Registry**: Automatic registration with metadata

### When to Use Each

**Use Foundation 01 if:**
- Learning RAG concepts
- Quick one-off experiments
- No database available
- Testing without setup

**Use Foundation 02 if:**
- Regenerating 50+ minute embeddings is expensive
- Need embeddings across multiple experiments
- Building production system
- Comparing different techniques
- Setting up for advanced techniques (05-10)

## Key Concepts

### Registry Integration

The embedding registry (`embedding_registry` table) enables:
- **Discovery**: Find registered embeddings by model alias
- **Metadata Context**: Know dimension, chunk size, source dataset
- **Reuse**: Avoid 50+ minute regeneration with `load_or_generate()`
- **Reproducibility**: Track which embeddings were used in each experiment

### Data Preservation Settings

Foundation 02 asks on second run: "Embeddings already exist. Preserve or regenerate?"

```python
# Control this behavior with PRESERVE_EXISTING_EMBEDDINGS:
PRESERVE_EXISTING_EMBEDDINGS = None     # Prompt user (default)
PRESERVE_EXISTING_EMBEDDINGS = True     # Always reuse
PRESERVE_EXISTING_EMBEDDINGS = False    # Always regenerate
```

This prevents accidentally losing hours of embedding generation work!

### Performance Impact

**Foundation 01 (In-Memory):**
- Generation: 5-10 minutes
- Reuse: Not available (lost on restart)
- Regeneration: Required for new notebook

**Foundation 02 (PostgreSQL):**
- Generation: 5-10 minutes (first run)
- Reuse: Instant (subsequent runs)
- Registry: Enables discovery and reuse across all notebooks

## Installing Required Packages

```bash
pip install psycopg2-binary
```

If using a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install psycopg2-binary ollama datasets ipywidgets jupyter
```

## Troubleshooting

### Connection Refused

```
Error: could not translate host name "localhost" to address
```

**Solution**: PostgreSQL container isn't running. Start it:
```bash
docker start pgvector-rag
```

Or create a new container:
```bash
docker run -d --name pgvector-rag \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=rag_db \
  -p 5432:5432 \
  -v pgvector_data:/var/lib/postgresql/data \
  pgvector/pgvector:pg16
```

### "Table not found" Error

**Solution**: The embeddings haven't been generated yet. Run the main notebook first with `STORAGE_BACKEND = 'postgresql'`.

### Permission Denied (Windows)

If Docker requires elevated privileges on Windows:
```bash
# Run PowerShell as Administrator, then:
docker run -d --name pgvector-rag ...
```

## Data Persistence

Your embeddings are stored in a Docker volume (`pgvector_data`). They survive:
- ✅ Notebook restarts
- ✅ Container stop/start
- ✅ Python kernel restarts

To **remove** stored data:
```bash
docker volume rm pgvector_data
```

To **backup** your embeddings:
```bash
docker exec pgvector-rag pg_dump -U postgres rag_db > rag_db_backup.sql
```

To **restore** from backup:
```bash
docker exec -i pgvector-rag psql -U postgres rag_db < rag_db_backup.sql
```

## Advanced: Using Other Vector Databases

Once your embeddings are in PostgreSQL with pgvector, you can easily export them to:

- **Pinecone**: Use the export functions in the main notebook
- **Weaviate**: Standard JSON export compatible
- **Chroma**: Local vector DB alternative
- **Milvus**: Open-source distributed alternative

See "Export Dataset for Other Platforms" in the main notebook for example export functions.

## Next Steps

1. **Analyze Embeddings**: Use `embedding-analysis-template.ipynb`
2. **Compare Models**: Generate embeddings with different models, store them separately, compare results
3. **Fine-tune Retrieval**: Test different `top_n` values, similarity thresholds, chunk sizes
4. **Build Applications**: Use the persistent embeddings to power RAG applications without regeneration

## References

- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [PostgreSQL pgvector Docker Image](https://hub.docker.com/r/pgvector/pgvector)
- [psycopg2 Documentation](https://www.psycopg.org/documentation/)
