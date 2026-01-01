# Getting Started Checklist

Choose your path and follow the checklist. Check off items as you complete them.

---

## 🚀 Path A: Quick Learning (In-Memory, No Setup)

**Time: ~1 hour | Setup: 0 minutes**

- [ ] Read [readme.md](../../README.md) - Project Overview (5 min)
- [ ] Ensure Ollama is running:
  ```bash
  ollama list
  ```
- [ ] Ensure models are downloaded:
  ```bash
  ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
  ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
  ```
- [ ] Install Python dependencies:
  ```bash
  pip install ollama datasets jupyter
  ```
- [ ] Open the **simple notebook**:
  ```bash
  jupyter notebook wikipedia-rag-tutorial-simple.ipynb
  ```
- [ ] Verify configuration cell has:
  ```python
  TARGET_SIZE_MB = 10
  SAVE_LOCALLY = True  # Optional: cache dataset
  ```
- [ ] Run all cells
- [ ] Ask test questions to verify it works
- [ ] You're done! ✅

**Next Steps**: Review [quick-reference.md](./quick-reference.md) to understand storage backends, then move to Path B or C if interested.

---

## 🐘 Path B: Single Experiment (PostgreSQL, Durable Storage)

**Time: ~2 hours | Setup: 2 minutes**

### Prerequisites
- [ ] Complete all steps from Path A OR
- [ ] Have Ollama running with models downloaded

### PostgreSQL Setup
- [ ] Verify Docker is installed:
  ```bash
  docker --version
  ```
- [ ] Start PostgreSQL container:
  ```bash
  docker run -d --name pgvector-rag \
    -e POSTGRES_PASSWORD=postgres \
    -e POSTGRES_DB=rag_db \
    -p 5432:5432 \
    -v pgvector_data:/var/lib/postgresql/data \
    pgvector/pgvector:pg16
  ```
- [ ] Wait 10 seconds for container to start
- [ ] Verify it's running:
  ```bash
  docker ps | grep pgvector-rag
  ```

### Python Setup
- [ ] Install PostgreSQL adapter:
  ```bash
  pip install psycopg2-binary
  ```

### Notebook Configuration
- [ ] Open the **advanced notebook**:
  ```bash
  jupyter notebook wikipedia-rag-tutorial-advanced.ipynb
  ```
- [ ] Find the "Configuration" cell and verify:
  ```python
  TARGET_SIZE_MB = 10
  
  POSTGRES_CONFIG = {
      'host': 'localhost',
      'port': 5432,
      'database': 'rag_db',
      'user': 'postgres',
      'password': 'postgres',
  }
  
  EMBEDDING_MODEL_ALIAS = 'bge_base_en_v1.5'
  ```
- [ ] Run all cells
- [ ] Wait for embedding generation (50 min for 10MB)
- [ ] Verify success message: "✓ Vector database ready with X embeddings in PostgreSQL!"

### Create Analysis Notebook
- [ ] Read [embedding-analysis-template.ipynb](../../embedding-analysis-template.ipynb)
- [ ] Create a copy:
  ```bash
  cp embedding-analysis-template.ipynb my-analysis.ipynb
  ```
- [ ] Run the analysis notebook to load stored embeddings
- [ ] Confirm it loads without regenerating embeddings
- [ ] You're done! ✅

**Next Steps**: 
- Try modifying the analysis notebook to run your own experiments
- See [quick-reference.md](./quick-reference.md) for advanced options

---

## 🔬 Path C: Model Comparison (PostgreSQL + Multiple Experiments)

**Time: ~3+ hours | Setup: 2 minutes**

### Prerequisites
- [ ] Complete Path B first

### Generate First Model
- [ ] Open the **advanced notebook**:
  ```bash
  jupyter notebook wikipedia-rag-tutorial-advanced.ipynb
  ```
- [ ] Use first model (already done from Path B):
  ```python
  EMBEDDING_MODEL_ALIAS = 'bge_base_en_v1.5'
  ```
- [ ] Verify embeddings exist in database:
  ```python
  db = PostgreSQLVectorDB(POSTGRES_CONFIG, 'embeddings_bge_base_en_v1_5')
  print(f"Stored: {db.get_chunk_count()} embeddings")
  ```

### Generate Second Model
- [ ] Create a copy of the advanced notebook:
  ```bash
  cp wikipedia-rag-tutorial-advanced.ipynb wikipedia-rag-model2.ipynb
  ```
- [ ] Open the copy in Jupyter
- [ ] Change configuration:
  ```python
  EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-small-en-v1.5-gguf'
  EMBEDDING_MODEL_ALIAS = 'bge_small_en_v1.5'
  ```
- [ ] Pull the new model:
  ```bash
  ollama pull hf.co/CompendiumLabs/bge-small-en-v1.5-gguf
  ```
- [ ] Run all cells in the modified notebook
- [ ] Wait for embedding generation

### Create Comparison Notebook
- [ ] Create new notebook from template:
  ```bash
  cp embedding-analysis-template.ipynb embedding-comparison.ipynb
  ```
- [ ] Open in Jupyter
- [ ] Modify to load both models:
  ```python
  db1 = PostgreSQLVectorDB(POSTGRES_CONFIG, 'embeddings_bge_base_en_v1_5')
  db2 = PostgreSQLVectorDB(POSTGRES_CONFIG, 'embeddings_bge_small_en_v1_5')
  
  # Compare retrieval results
  test_query = "What is photosynthesis?"
  ```
- [ ] Run your analysis
- [ ] You're done! ✅

**Next Steps**: 
- Try adding more models
- Implement automated evaluation metrics
- See [postgres-setup.md](./postgres-setup.md) for advanced database topics

---

## 🔧 Common Troubleshooting

### "Ollama model download hangs"
- [ ] Press Ctrl+C to cancel
- [ ] Try a smaller model first:
  ```bash
  ollama pull hf.co/CompendiumLabs/bge-small-en-v1.5-gguf
  ```
- [ ] Check Ollama is actually running with:
  ```bash
  ps aux | grep ollama
  ```

### "Docker: command not found"
- [ ] Install [Docker Desktop](https://www.docker.com/products/docker-desktop)
- [ ] Start Docker (on Mac/Windows, look for Docker icon)
- [ ] Try again:
  ```bash
  docker --version
  ```

### "PostgreSQL connection refused"
- [ ] Check container is running:
  ```bash
  docker ps | grep pgvector
  ```
- [ ] If not running, start it:
  ```bash
  docker start pgvector-rag
  ```
- [ ] If you deleted it, create it again:
  ```bash
  docker run -d --name pgvector-rag \
    -e POSTGRES_PASSWORD=postgres \
    -e POSTGRES_DB=rag_db \
    -p 5432:5432 \
    -v pgvector_data:/var/lib/postgresql/data \
    pgvector/pgvector:pg16
  ```

### "psycopg2 import error"
- [ ] Install the library:
  ```bash
  pip install psycopg2-binary
  ```

### "Jupyter notebook not found"
- [ ] Install jupyter:
  ```bash
  pip install jupyter
  ```

### Embeddings generating very slowly
- [ ] This is expected! Progress updates appear every 50 chunks
- [ ] For 10MB dataset: ~50 minutes is normal
- [ ] Use smaller dataset while testing:
  ```python
  TARGET_SIZE_MB = 5
  ```

**Can't find your issue?** See [postgres-setup.md](./postgres-setup.md) → "Troubleshooting" section

---

## 📚 Next Learning Steps

After completing your chosen path:

1. **Understand RAG Concepts** (30 min)
   - Read: [readme.md](../../README.md) → "🔬 Experimentation Guide"
   - Understand: Retrieval vs Generation trade-offs

2. **Explore Embeddings** (20 min)
   - Read: [quick-reference.md](./quick-reference.md)
   - Compare: Different embedding models
   - Experiment: Different `top_n` values in retrieval

3. **Plan Your Experiments** (1 hour)
   - Use: [embedding-analysis-template.ipynb](../../embedding-analysis-template.ipynb) as template
   - Try: Custom evaluation metrics
   - Analyze: Retrieval quality by query type

4. **Prepare for Production** (Optional)
   - Read: [postgres-setup.md](./postgres-setup.md) → "Advanced" section
   - Explore: Other vector databases (Neon, Pinecone, Weaviate)
   - Plan: Migration strategy for your use case

---

## 📖 Documentation Map

| I want to... | Read this |
|-------------|----------|
| Understand the project | [readme.md](../../README.md) |
| Get started quickly | This checklist (index.md) |
| Set up PostgreSQL | [postgres-setup.md](./postgres-setup.md) |
| Choose a storage backend | [quick-reference.md](./quick-reference.md) |
| See what's new | [enhancement-summary.md](../implementation/enhancement-summary.md) |
| Copy code examples | [embedding-analysis-template.ipynb](../../embedding-analysis-template.ipynb) |
| Learn RAG concepts | [readme.md](../../README.md) → "Experimentation Guide" |
| Troubleshoot issues | [postgres-setup.md](./postgres-setup.md) → "Troubleshooting" |

---

## 🎯 Success Criteria

You'll know you're successful when:

✅ **Path A**: You can run a query and get relevant results  
✅ **Path B**: Embeddings are stored in PostgreSQL and reloaded instantly  
✅ **Path C**: You can compare retrieval results from two different embedding models  

---

## 💡 Pro Tips

1. **Start small**: Use `TARGET_SIZE_MB = 5` to test faster
2. **Keep PostgreSQL running**: Use `docker ps` to verify it's up
3. **Save your experiments**: Create a new notebook for each experiment
4. **Monitor progress**: The notebook prints status every 50 chunks
5. **Test queries**: Try different query types to understand retrieval quality

---

## ⏱️ Time Estimates

| Activity | Time |
|----------|------|
| Read README | 5 min |
| Download models (first time) | 5 min |
| Install dependencies | 2 min |
| Set up Docker & PostgreSQL | 2 min |
| Run notebook with 5MB dataset | 15 min |
| Run notebook with 10MB dataset | 50 min |
| Create analysis notebook | 20 min |
| Run model comparison | +50 min per model |

**Total for Path A**: ~1 hour  
**Total for Path B**: ~2 hours  
**Total for Path C**: ~3+ hours  

---

**Ready to get started?** Choose your path and check off items as you go! 🚀
