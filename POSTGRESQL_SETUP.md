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

### 2. Configure the Main Notebook

In `wikipedia-rag-tutorial.ipynb`:

Change the storage backend from:
```python
STORAGE_BACKEND = 'memory'
```

To:
```python
STORAGE_BACKEND = 'postgresql'
```

The notebook will:
1. Create a table named `embeddings_bge_base_en_v1_5` (based on your embedding model)
2. Generate embeddings and stream them directly to PostgreSQL
3. Enable reuse in other notebooks

### 3. Run the Notebook

Just run the notebook normally. Embeddings will be generated and stored in PostgreSQL as they're created.

### 4. Create Analysis Notebooks

Once embeddings are stored, create new notebooks for experiments. See `embedding-analysis-template.ipynb` for examples.

## Multiple Embedding Models

You can store embeddings from different models for comparison:

### Main Notebook (bge model)
```python
STORAGE_BACKEND = 'postgresql'
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
EMBEDDING_MODEL_ALIAS = 'bge_base_en_v1.5'
```

Creates table: `embeddings_bge_base_en_v1_5`

### Alternative Notebook (different model)
```python
STORAGE_BACKEND = 'postgresql'
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
EMBEDDING_MODEL_ALIAS = 'all_minilm_l6_v2'
```

Creates table: `embeddings_all_minilm_l6_v2`

Then in analysis notebooks, you can load and compare both:

```python
db_bge = PostgreSQLVectorDB(POSTGRES_CONFIG, 'embeddings_bge_base_en_v1_5')
db_minilm = PostgreSQLVectorDB(POSTGRES_CONFIG, 'embeddings_all_minilm_l6_v2')

# Compare retrieval results for the same query
```

## Storage Backend Options

### Memory (`'memory'`)
```python
STORAGE_BACKEND = 'memory'
```
- **Pros**: Fastest, no setup needed
- **Cons**: Lost on notebook restart
- **Best for**: Quick experiments and testing

### JSON File (`'json'`)
```python
STORAGE_BACKEND = 'json'
```
- **Pros**: Persists across restarts, portable
- **Cons**: Slower for large datasets, all data loaded into memory
- **Best for**: Small datasets, sharing/backup

### PostgreSQL (`'postgresql'`)
```python
STORAGE_BACKEND = 'postgresql'
```
- **Pros**: Durable, efficient similarity search, supports multiple models
- **Cons**: Requires Docker and psycopg2 library
- **Best for**: Production workflows, multiple experiments

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
