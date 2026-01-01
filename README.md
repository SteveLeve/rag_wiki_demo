# Wikipedia RAG Demo Project

A practical Retrieval-Augmented Generation (RAG) implementation using Simple Wikipedia articles, designed for experimentation with different embedding models, RAG evaluation techniques, and migration to production vector databases.

## 🎯 Project Goals

This project serves as a learning platform and proof-of-concept for:

1. **RAG System Fundamentals**: Understand core RAG concepts through hands-on implementation
2. **Embedding Model Comparison**: Evaluate different embedding models on the same dataset
3. **RAG Evaluation Techniques**: Experiment with retrieval quality metrics and generation assessment
4. **Production-Ready Architecture**: Develop locally with PostgreSQL, optionally migrate to hosted solutions (Neon, Supabase)
5. **Free Tier Optimization**: Maintain dataset sizes (10-50MB) that work within free tier constraints

## 📋 Features

- **Configurable Dataset Size**: Scale from 10MB to 50MB based on your needs
- **Local Development**: In-memory vector database for fast iteration
- **Smart Chunking**: Intelligent text segmentation at paragraph boundaries
- **Local Caching**: Save processed datasets to avoid re-downloading
- **Export Capabilities**: Ready-to-use exports for production vector databases
- **Progress Tracking**: Real-time feedback during data loading and embedding
- **Interactive Q&A**: Test your RAG system with various queries

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Wikipedia Dataset                         │
│              (Simple Wikipedia via HuggingFace)              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
         ┌─────────────────────────┐
         │   Data Processing       │
         │  • Filter by size       │
         │  • Chunk articles       │
         │  • Add metadata         │
         └──────────┬──────────────┘
                    │
                    ▼
         ┌─────────────────────────┐
         │   Embedding Layer       │
         │  (Ollama + BGE-base)    │
         └──────────┬──────────────┘
                    │
                    ▼
         ┌─────────────────────────┐
         │   Vector Database       │
         │  • In-memory (dev)      │
         │  • PostgreSQL + pgvector│
         │  • Neon/Supabase (opt.) │
         └──────────┬──────────────┘
                    │
                    ▼
         ┌─────────────────────────┐
         │   Retrieval Layer       │
         │  (Cosine Similarity)    │
         └──────────┬──────────────┘
                    │
                    ▼
         ┌─────────────────────────┐
         │   Generation Layer      │
         │  (Llama 3.2 1B)         │
         └─────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

1. **Install Ollama**: Download from [ollama.com](https://ollama.com/)

2. **Pull Required Models**:
   ```bash
   ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
   ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
   ```

3. **Install Python Dependencies**:
   ```bash
   pip install ollama datasets jupyter
   ```

### Running the Notebooks

This project is organized in layers by complexity. **Start with foundation/**:

#### Foundation Layer (Start Here!)
The `foundation/` directory contains 5 well-documented notebooks:

1. **01-basic-rag-in-memory.ipynb** (Recommended for beginners)
   ```bash
   jupyter notebook foundation/01-basic-rag-in-memory.ipynb
   ```
   - In-memory storage (no database needed)
   - Perfect for learning RAG fundamentals
   - Quick start section included
   - 🚀 Takes 5-10 minutes to get started

2. **02-rag-postgresql-persistent.ipynb** (For persistence & comparison)
   ```bash
   jupyter notebook foundation/02-rag-postgresql-persistent.ipynb
   ```
   - PostgreSQL + pgvector for persistent storage
   - Embeddings registered in registry for reuse
   - Multiple embedding models for comparison
   - Automatic experiment tracking

**First time setup:**
1. Run `foundation/00-setup-postgres-schema.ipynb` (one-time PostgreSQL setup)
2. Run `foundation/01-basic-rag-in-memory.ipynb` (standalone, no database)
3. Then run `foundation/02-rag-postgresql-persistent.ipynb` (if you want persistence)

For detailed setup, see **[foundation/README.md](./foundation/README.md)** or **[POSTGRESQL_SETUP.md](./POSTGRESQL_SETUP.md)**

### Next Steps

After foundation, explore:
- **intermediate/** - Registry discovery & model comparison (10-20 min)
- **advanced-techniques/** - Reranking, query expansion, hybrid search, etc. (4-6 hours)
- **evaluation-lab/** - Measure RAG quality with metrics (3-5 hours)

See **[INDEX.md](./INDEX.md)** for complete learning paths.

## 📊 Dataset Specifications

### Size Options

| Size  | Approx. Articles | Approx. Chunks | Embedding Time* |
|-------|------------------|----------------|-----------------|
| 10MB  | 200-300          | 300-500        | 2-3 minutes     |
| 20MB  | 400-600          | 600-1000       | 5-6 minutes     |
| 30MB  | 600-900          | 900-1500       | 8-10 minutes    |
| 40MB  | 800-1200         | 1200-2000      | 12-15 minutes   |
| 50MB  | 1000-1500        | 1500-2500      | 15-20 minutes   |

*Times vary based on CPU performance

### Chunk Characteristics

- **Maximum Chunk Size**: 1000 characters (configurable)
- **Chunking Strategy**: Paragraph boundaries, with sentence-level fallback
- **Metadata**: Each chunk includes article title for context
- **Format**: Plain text with title prefix

## 🔬 Experimentation Guide

### Comparing Embedding Models

1. Modify `EMBEDDING_MODEL` in the notebook
2. Clear `VECTOR_DB` and re-embed
3. Compare retrieval quality using the same queries

Example models to try:
- `hf.co/CompendiumLabs/bge-base-en-v1.5-gguf` (default)
- `hf.co/CompendiumLabs/bge-small-en-v1.5-gguf` (faster)
- `hf.co/CompendiumLabs/bge-large-en-v1.5-gguf` (better quality)

### RAG Evaluation Techniques

1. **Retrieval Quality**:
   ```python
   # Test if correct chunks are retrieved
   retrieved = retrieve("What is Python?", top_n=5)
   for chunk, similarity in retrieved:
       print(f"Similarity: {similarity:.3f}")
       print(f"Title: {chunk.split('\\n')[0]}")
   ```

2. **Answer Relevance**:
   - Does the answer address the question?
   - Is it factually correct based on retrieved context?
   - Are citations accurate?

3. **Context Utilization**:
   - Track which retrieved chunks are actually used
   - Measure answer coverage vs. retrieved content

## 🌐 PostgreSQL Setup & Deployment Options

### Option 1: Local PostgreSQL with Docker (Recommended for Learning)

**Why Start Local:**
- Complete control over data and infrastructure
- No cloud service dependencies
- Easy to understand and debug
- Ideal for learning and experimentation

**Setup:**

1. **Start PostgreSQL with Docker**:
   ```bash
   docker run --name postgres_rag \
     -e POSTGRES_PASSWORD=yourpassword \
     -e POSTGRES_DB=wikipedia_rag \
     -p 5432:5432 \
     -d postgres:16
   ```

2. **Enable pgvector Extension**:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

3. **Create Table**:
   ```sql
   CREATE TABLE wikipedia_chunks (
     id SERIAL PRIMARY KEY,
     chunk_id TEXT UNIQUE,
     title TEXT,
     text TEXT,
     embedding vector(768)
   );

   -- Create index for efficient similarity search
   CREATE INDEX ON wikipedia_chunks
   USING ivfflat (embedding vector_cosine_ops)
   WITH (lists = 100);
   ```

4. **Connect from Notebook**:
   ```python
   import psycopg2
   from psycopg2.extras import execute_values

   conn = psycopg2.connect(
       host="localhost",
       port=5432,
       database="wikipedia_rag",
       user="postgres",
       password="yourpassword"
   )

   # Insert embeddings
   with conn.cursor() as cur:
       execute_values(
           cur,
           "INSERT INTO wikipedia_chunks (chunk_id, title, text, embedding) VALUES %s",
           [(chunk_id, title, text, embedding) for chunk_id, title, text, embedding in data]
       )
   conn.commit()
   ```

### Option 2: Neon PostgreSQL (Hosted, Free Tier)

**Why Choose Neon:**
- SQL-based querying with vector similarity (pgvector)
- ACID compliance for data integrity
- Free tier: 0.5GB storage, generous compute
- Easy migration path from local

**Setup:**

1. **Create Neon Account**: [neon.tech](https://neon.tech/)

2. **Create Database**:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;

   CREATE TABLE wikipedia_chunks (
     id SERIAL PRIMARY KEY,
     chunk_id TEXT UNIQUE,
     title TEXT,
     text TEXT,
     embedding vector(768)
   );

   CREATE INDEX ON wikipedia_chunks
   USING ivfflat (embedding vector_cosine_ops)
   WITH (lists = 100);
   ```

3. **Connect from Notebook**:
   ```python
   import psycopg2

   conn = psycopg2.connect(
       host="<neon-host>",
       port=5432,
       database="wikipedia_rag",
       user="<neon-user>",
       password="<neon-password>"
   )
   ```

4. **Query Data**:
   ```sql
   SELECT chunk_id, title, text,
          1 - (embedding <=> ${queryEmbedding}::vector) as similarity
   FROM wikipedia_chunks
   ORDER BY embedding <=> ${queryEmbedding}::vector
   LIMIT 5;
   ```

### Option 3: Supabase PostgreSQL (Alternative Hosted)

**Why Choose Supabase:**
- Built on PostgreSQL with pgvector support
- Includes authentication and API layer
- Free tier: 500MB storage
- Easy to upgrade when needed

**Setup:**

1. **Create Supabase Project**: [supabase.com](https://supabase.com/)

2. **Create Table** (via SQL Editor):
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;

   CREATE TABLE wikipedia_chunks (
     id BIGSERIAL PRIMARY KEY,
     chunk_id TEXT UNIQUE,
     title TEXT,
     text TEXT,
     embedding vector(768)
   );

   CREATE INDEX ON wikipedia_chunks
   USING ivfflat (embedding vector_cosine_ops)
   WITH (lists = 100);
   ```

3. **Connect from Notebook**:
   ```python
   from supabase import create_client, Client

   url = "<your-supabase-url>"
   key = "<your-supabase-key>"
   supabase: Client = create_client(url, key)
   ```

### Option 4: Self-Hosted Alternatives

Other PostgreSQL hosting providers with pgvector support:
- **Railway**: Simple deployment, pay-as-you-go
- **Render**: Generous free tier for PostgreSQL
- **TimescaleDB Cloud**: Purpose-built for time-series + vectors

All follow the same SQL schema and connection patterns as Neon and Supabase.

### PostgreSQL Options Comparison

| Feature               | Local Docker | Neon | Supabase | Railway |
|-----------------------|--------------|------|----------|---------|
| **Setup Time**        | 5 min        | 5 min| 5 min    | 10 min  |
| **Cost**              | Free         | Free | Free     | Free    |
| **Storage (free)**    | Unlimited    | 0.5GB| 500MB    | 10GB    |
| **Best For**          | Learning     | Demo | Production| Scale   |
| **pgvector Support**  | Yes          | Yes  | Yes      | Yes     |

## 📓 Analysis & Experimentation Notebooks

Once you have embeddings stored in PostgreSQL, you can create separate analysis notebooks to:

- **Compare Embedding Models**: Generate embeddings with different models, store them separately, compare retrieval quality
- **Evaluate Retrieval Performance**: Test different queries to identify what works well
- **Statistical Analysis**: Analyze embedding properties and distributions
- **Debug Retrieval Issues**: Identify queries that return poor results
- **Benchmark Improvements**: Test changes to chunking, embeddings, or ranking strategies

See **embedding-analysis-template.ipynb** for examples and code snippets.

### Benefits of Separate Analysis Notebooks

- **Performance**: Don't regenerate embeddings for each experiment (saves 50+ minutes per run)
- **Organization**: Keep experiment code separate from the core tutorial
- **Reproducibility**: Compare results across multiple model configurations
- **Scalability**: Easily add new experiments without modifying the main notebook

Example experiments:
```
embedding-analysis-template.ipynb          # Template & examples
experiment-bge-vs-minilm.ipynb            # Compare embedding models
experiment-chunk-size-impact.ipynb        # Test different chunk sizes
experiment-top-n-threshold.ipynb          # Find optimal retrieval count
```

## 📈 Performance Optimization

### Local Development
- Cache processed datasets locally
- Use smaller embedding models for faster iteration
- Start with 10MB dataset for quick testing

### Production
- Implement batch embedding for faster indexing
- Use approximate nearest neighbor (ANN) indexes
- Cache frequent queries
- Implement query result pagination

## 🧪 Testing & Evaluation

### Create Test Sets

```python
# Define ground-truth Q&A pairs
test_cases = [
    {
        'question': 'What is the capital of France?',
        'expected_article': 'Paris',
        'expected_answer_contains': ['Paris', 'capital', 'France']
    },
    # Add more test cases...
]

# Evaluate retrieval
for test in test_cases:
    results = retrieve(test['question'], top_n=3)
    # Check if expected article is in top results
    # Measure similarity scores
    # Assess answer quality
```

### Metrics to Track

1. **Retrieval Metrics**:
   - Recall@K (Is the correct chunk in top K results?)
   - MRR (Mean Reciprocal Rank)
   - Average similarity score

2. **Generation Metrics**:
   - Faithfulness (Does answer match retrieved context?)
   - Relevance (Does answer address the question?)
   - Citation accuracy

3. **System Metrics**:
   - Query latency
   - Embedding throughput
   - Memory usage

## 🛠️ Troubleshooting

### Common Issues

**1. Ollama Model Not Found**
```bash
# Verify models are installed
ollama list

# Re-pull if needed
ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
```

**2. Out of Memory During Embedding**
- Reduce `TARGET_SIZE_MB`
- Process chunks in smaller batches
- Use a smaller embedding model

**3. Slow Retrieval**
- Reduce dataset size
- Use approximate similarity search (FAISS, Annoy)
- Pre-filter by metadata before vector search

**4. Poor Answer Quality**
- Increase `top_n` in retrieval
- Try different embedding models
- Improve chunking strategy
- Use a larger language model

## 📚 Next Steps

### Short Term
- [x] Build basic RAG pipeline
- [ ] Implement evaluation metrics
- [ ] Compare 3+ embedding models
- [ ] Test with different chunk sizes

### Medium Term
- [ ] Add reranking layer
- [ ] Implement hybrid search (vector + keyword)
- [ ] Deploy to Neon, Supabase, or self-hosted PostgreSQL
- [ ] Build simple web interface

### Long Term
- [ ] Implement query expansion
- [ ] Add citation tracking
- [ ] Build evaluation dashboard
- [ ] Scale to larger datasets

## 📖 Resources

### RAG Fundamentals
- [HuggingFace RAG Guide](https://huggingface.co/blog/ngxson/make-your-own-rag)
- [Pinecone RAG Series](https://www.pinecone.io/learn/series/rag/)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/use_cases/question_answering/)

### Vector Databases
- [Neon Documentation](https://neon.tech/docs/introduction)
- [Supabase PostgreSQL](https://supabase.com/docs/guides/database/overview)
- [pgvector Guide](https://github.com/pgvector/pgvector)

### Embedding Models
- [BGE Embeddings](https://huggingface.co/BAAI/bge-base-en-v1.5)
- [Sentence Transformers](https://www.sbert.net/)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)

### Evaluation
- [RAGAS Framework](https://docs.ragas.io/)
- [RAG Triad of Metrics](https://www.trulens.org/)

## 🤝 Contributing

This is a learning project, but improvements are welcome:

1. Better chunking strategies
2. Additional embedding model examples
3. Evaluation metric implementations
4. Production deployment examples
5. Documentation improvements

## 📝 License

This project uses:
- Wikipedia content: CC BY-SA 3.0 and GFDL
- Project code: MIT License

## 🙋 FAQ

**Q: Why Simple Wikipedia instead of full Wikipedia?**  
A: Simple Wikipedia has cleaner, more concise articles that are easier to chunk and understand, making it ideal for learning and demos.

**Q: Can I use a different dataset?**  
A: Yes! The notebook structure works with any text dataset. Just replace the loading function with your data source.

**Q: What's the minimum hardware required?**  
A: 8GB RAM recommended for 10-20MB datasets. Larger datasets may need 16GB+.

**Q: How accurate are the answers?**  
A: Accuracy depends on whether relevant articles are in your dataset sample. The 1B parameter LLM is limited but surprisingly capable for factual questions.

**Q: Can I use this project without cloud services?**
A: Yes! Start with local PostgreSQL using Docker (see Option 1 in the PostgreSQL Setup section). It's perfect for learning and requires no cloud accounts or API keys.

**Q: Which PostgreSQL option should I choose for production?**
A: Start with local PostgreSQL for development/learning. For production, choose based on your needs: Neon for simplicity, Supabase for additional features (auth, APIs), or self-hosted (Railway/Render) for cost control.

---

**Built for learning RAG systems and preparing for production deployment** 🚀
