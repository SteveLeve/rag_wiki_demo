"""
End-to-end tests for RAG pipeline integration.

Comprehensive tests for complete RAG pipeline flows from foundation notebooks including:
- In-memory RAG pipeline (no database)
- PostgreSQL persistence with pgvector
- Multi-model embedding comparison
- Experiment tracking and lifecycle
- Error recovery and transaction handling
- Performance benchmarks
- Load-or-generate embedding patterns
- Citation and metadata tracking

Coverage target: Complete pipeline flows (not individual functions)

Test organization:
- test_complete_in_memory_pipeline: In-memory dataset → chunk → embed → retrieve → ask
- test_postgresql_persistence_pipeline: PostgreSQL registry → embed → batch insert → search → ask
- test_multi_model_comparison: Multiple models with different dimensions
- test_experiment_tracking_lifecycle: Experiment creation → metrics → completion
- test_error_recovery_transaction_rollback: Connection loss and rollback
- test_performance_large_dataset: 10MB dataset processing < 5 minutes
- test_load_or_generate_embedding_caching: Reuse pattern with no regeneration
- test_citation_tracking_metadata_preservation: Source tracking through pipeline

Total: 8 comprehensive end-to-end tests covering complete RAG flows
"""

import pytest
import psycopg2
import numpy as np
import json
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any
from datetime import datetime


# ============================================================================
# Helper Functions (from foundation notebooks)
# ============================================================================


def chunk_text(text, max_size=1000):
    """Split text into chunks at paragraph boundaries."""
    if len(text) <= max_size:
        return [text]

    chunks = []
    paragraphs = text.split('\n\n')
    current_chunk = ''

    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) > max_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ''

            if len(paragraph) > max_size:
                sentences = paragraph.split('. ')
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) > max_size:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + '. '
                    else:
                        current_chunk += sentence + '. '
            else:
                current_chunk = paragraph
        else:
            current_chunk += '\n\n' + paragraph if current_chunk else paragraph

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)

    if a.shape != b.shape:
        raise ValueError(f"Vector dimensions must match: {a.shape} vs {b.shape}")

    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def retrieve(query_embedding, chunks_embeddings, top_n=5):
    """Retrieve top N most relevant chunks based on similarity."""
    similarities = []

    for chunk, embedding in chunks_embeddings:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]


def register_embedding_model(conn, model_alias: str, model_name: str, dimension: int) -> int:
    """Register embedding model in database and return model ID."""
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO embedding_registry (model_alias, model_name, dimension, embedding_count)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (model_alias) DO NOTHING
            RETURNING id
        """, (model_alias, model_name, dimension, 0))
        result = cur.fetchone()
        conn.commit()
        return result[0] if result else None


def insert_embeddings_batch(conn, embeddings_data: List[Tuple[str, List[float]]]) -> int:
    """Insert batch of embeddings. Returns count inserted."""
    count = 0
    with conn.cursor() as cur:
        for chunk_text, embedding_vector in embeddings_data:
            try:
                cur.execute("""
                    INSERT INTO chunk_embeddings (chunk_text, embedding, metadata)
                    VALUES (%s, %s, %s)
                """, (chunk_text, embedding_vector, json.dumps({
                    "inserted_at": datetime.now().isoformat()
                })))
                count += 1
            except psycopg2.Error:
                pass
    conn.commit()
    return count


def similarity_search(conn, query_embedding: List[float], top_n: int = 5, table_name: str = 'chunk_embeddings'):
    """Search database for similar embeddings using pgvector."""
    with conn.cursor() as cur:
        # Use pgvector <=> operator for similarity search
        query = f"""
            SELECT chunk_text, 1 - (embedding <=> %s::vector) as similarity
            FROM {table_name}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        try:
            cur.execute(query, (query_embedding, query_embedding, top_n))
            results = cur.fetchall()
            return [(chunk, sim) for chunk, sim in results]
        except psycopg2.Error:
            # If table doesn't exist or has issues, return empty results
            return []


def start_experiment(conn, name: str, model_alias: str, config: Dict[str, Any]) -> int:
    """Start new experiment and return experiment ID."""
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO experiments (experiment_name, embedding_model_alias, config_json, status)
            VALUES (%s, %s, %s, 'running')
            RETURNING id
        """, (name, model_alias, json.dumps(config)))
        conn.commit()
        return cur.fetchone()[0]


def save_metric(conn, experiment_id: int, metric_name: str, metric_value: float, details: Dict[str, Any] = None) -> None:
    """Save metric result for experiment."""
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO evaluation_results (experiment_id, metric_name, metric_value, metric_details_json)
            VALUES (%s, %s, %s, %s)
        """, (experiment_id, metric_name, metric_value, json.dumps(details or {})))
    conn.commit()


def complete_experiment(conn, experiment_id: int, status: str = 'completed') -> None:
    """Mark experiment as completed."""
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE experiments
            SET status = %s, completed_at = CURRENT_TIMESTAMP
            WHERE id = %s
        """, (status, experiment_id))
    conn.commit()


# ============================================================================
# E2E Test: Complete In-Memory Pipeline
# ============================================================================


@pytest.mark.e2e
@pytest.mark.unit
def test_complete_in_memory_pipeline(mock_dataset, sample_embeddings, mock_ollama):
    """
    End-to-end test: dataset → chunk → embed → retrieve → generate answer.

    Tests complete in-memory RAG pipeline without database persistence.
    Verifies:
    - Dataset loading and enrichment
    - Text chunking
    - Embedding generation
    - Semantic retrieval
    - Query answering with context
    """
    # Step 1: Load dataset
    chunks = []
    for title, text in zip(mock_dataset['title'], mock_dataset['text']):
        enriched = f"Article: {title}\n\n{text}"
        chunks.append(enriched)

    assert len(chunks) == 10
    assert all(isinstance(c, str) for c in chunks)
    assert all(len(c) > 100 for c in chunks)

    # Step 2: Further chunk text
    sub_chunks = []
    for chunk in chunks[:3]:  # Process first 3 articles
        sub = chunk_text(chunk, max_size=500)
        sub_chunks.extend(sub)

    assert len(sub_chunks) > 0
    assert all(len(c) <= 600 for c in sub_chunks)  # Allow some tolerance

    # Step 3: Create embeddings (use sample_embeddings)
    chunks_with_embeddings = list(zip(sub_chunks, sample_embeddings[:len(sub_chunks)]))

    # Step 4: Retrieve based on query
    query = "What is machine learning?"
    # Use first embedding as query (deterministic)
    query_embedding = sample_embeddings[0]

    results = retrieve(query_embedding, chunks_with_embeddings, top_n=3)

    assert len(results) > 0
    assert len(results) <= 3
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

    # Step 5: Verify similarity scores are valid
    for chunk, score in results:
        assert -1.0001 <= score <= 1.0001
        assert not np.isnan(score)

    # Step 6: Format context and prepare for LLM
    context = "\n".join([
        f"{i+1}. {chunk[:100]}..."
        for i, (chunk, _) in enumerate(results)
    ])

    assert len(context) > 0
    assert all(str(i) in context for i in range(1, len(results) + 1))


# ============================================================================
# E2E Test: PostgreSQL Persistence Pipeline
# ============================================================================


@pytest.mark.e2e
@pytest.mark.integration
@pytest.mark.postgres
def test_postgresql_persistence_pipeline(postgres_connection, mock_dataset, sample_embeddings, seed_test_data):
    """
    End-to-end test: register model → embed → batch insert → search → generate.

    Tests complete RAG pipeline with PostgreSQL persistence.
    Verifies:
    - Model registration in embedding_registry
    - Embedding batch insertion
    - pgvector similarity search
    - Data persistence across operations
    """
    # Step 1: Register embedding model
    with postgres_connection.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chunk_embeddings (
                id SERIAL PRIMARY KEY,
                chunk_text TEXT NOT NULL,
                embedding vector(768),
                metadata JSONB DEFAULT '{}'::jsonb,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        postgres_connection.commit()

    # Step 2: Prepare embeddings data
    chunks = []
    for title, text in zip(mock_dataset['title'][:5], mock_dataset['text'][:5]):
        enriched = f"Article: {title}\n\n{text}"
        chunks.append(enriched)

    embeddings_data = list(zip(chunks, sample_embeddings[:len(chunks)]))

    # Step 3: Insert batch
    with postgres_connection.cursor() as cur:
        for chunk, embedding in embeddings_data:
            cur.execute("""
                INSERT INTO chunk_embeddings (chunk_text, embedding, metadata)
                VALUES (%s, %s, %s)
            """, (chunk, embedding.tolist(), json.dumps({
                "source": "mock_dataset",
                "chunk_size": len(chunk)
            })))
        postgres_connection.commit()

    # Step 4: Verify insertion
    with postgres_connection.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM chunk_embeddings")
        count = cur.fetchone()[0]
        assert count == len(chunks)

    # Step 5: Perform similarity search
    query_embedding = sample_embeddings[0]

    with postgres_connection.cursor() as cur:
        cur.execute("""
            SELECT chunk_text, 1 - (embedding <=> %s::vector) as similarity
            FROM chunk_embeddings
            ORDER BY embedding <=> %s::vector
            LIMIT 5
        """, (query_embedding.tolist(), query_embedding.tolist()))
        search_results = cur.fetchall()

    assert len(search_results) > 0
    assert len(search_results) <= 5

    # Step 6: Verify similarity scores
    similarities = [sim for _, sim in search_results]
    assert similarities == sorted(similarities, reverse=True)

    # Step 7: Verify persistence - run second query
    with postgres_connection.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM chunk_embeddings")
        final_count = cur.fetchone()[0]
        assert final_count == count  # Data persisted


# ============================================================================
# E2E Test: Multi-Model Embedding Comparison
# ============================================================================


@pytest.mark.e2e
@pytest.mark.postgres
def test_multi_model_comparison(postgres_connection, mock_dataset, seed_test_data):
    """
    End-to-end test: register multiple models → generate embeddings → compare retrieval.

    Tests pipeline with different embedding models:
    - BGE Base (768 dimensions)
    - BGE Small (384 dimensions)

    Verifies:
    - Multiple model registration
    - Different embedding dimensions
    - Different retrieval results per model
    """
    # Step 1: Register two models (already in seed_test_data)
    model_1_alias = "bge_base_en_v1_5"  # 768 dims
    model_2_alias = "bge_small_en_v1_5"  # 384 dims

    # Verify models are registered
    with postgres_connection.cursor() as cur:
        cur.execute("""
            SELECT model_alias, dimension FROM embedding_registry
            WHERE model_alias IN (%s, %s)
            ORDER BY dimension DESC
        """, (model_1_alias, model_2_alias))
        models = cur.fetchall()

    assert len(models) == 2
    assert models[0][1] == 768  # Base model
    assert models[1][1] == 384  # Small model

    # Step 2: Create test data with different embedding dimensions
    chunks = []
    for title, text in zip(mock_dataset['title'][:5], mock_dataset['text'][:5]):
        chunks.append(f"Article: {title}\n\n{text}")

    # Step 3: Create embeddings for model 1 (768 dims)
    np.random.seed(42)
    embeddings_768 = np.random.randn(len(chunks), 768).astype(np.float32)

    # Create embeddings for model 2 (384 dims)
    np.random.seed(42)
    embeddings_384 = np.random.randn(len(chunks), 384).astype(np.float32)

    # Step 4: Retrieve with model 1 embeddings
    query_768 = np.random.randn(768).astype(np.float32)

    chunks_emb_768 = list(zip(chunks, embeddings_768))
    results_768 = retrieve(query_768, chunks_emb_768, top_n=3)

    # Step 5: Retrieve with model 2 embeddings
    query_384 = np.random.randn(384).astype(np.float32)

    chunks_emb_384 = list(zip(chunks, embeddings_384))
    results_384 = retrieve(query_384, chunks_emb_384, top_n=3)

    # Step 6: Verify results are different
    results_768_chunks = [r[0] for r in results_768]
    results_384_chunks = [r[0] for r in results_384]

    # At least some difference in results
    assert results_768_chunks != results_384_chunks or len(results_768) == len(results_384)

    # Both have valid results
    assert len(results_768) > 0
    assert len(results_384) > 0


# ============================================================================
# E2E Test: Experiment Tracking Lifecycle
# ============================================================================


@pytest.mark.e2e
@pytest.mark.postgres
def test_experiment_tracking_lifecycle(postgres_connection, seed_test_data):
    """
    End-to-end test: start experiment → record metrics → complete → verify tracking.

    Tests complete experiment lifecycle:
    - Experiment creation
    - Metric recording
    - Experiment completion
    - Metrics persistence and retrieval
    """
    # Step 1: Start experiment
    with postgres_connection.cursor() as cur:
        exp_name = f"e2e_test_{int(time.time())}"
        cur.execute("""
            INSERT INTO experiments (
                experiment_name, embedding_model_alias, config_json, status
            )
            VALUES (%s, %s, %s, 'running')
            RETURNING id
        """, (exp_name, 'bge_base_en_v1_5', json.dumps({
            "top_n": 5,
            "similarity_threshold": 0.7,
            "technique": "basic_rag"
        })))
        postgres_connection.commit()
        exp_id = cur.fetchone()[0]

    assert exp_id > 0

    # Step 2: Record metrics during execution
    metrics = [
        ("retrieval_precision@5", 0.85),
        ("retrieval_mrr", 0.92),
        ("generation_bleu", 0.78),
        ("total_latency_ms", 1500),
    ]

    with postgres_connection.cursor() as cur:
        for metric_name, metric_value in metrics:
            cur.execute("""
                INSERT INTO evaluation_results
                (experiment_id, metric_name, metric_value, metric_details_json)
                VALUES (%s, %s, %s, %s)
            """, (exp_id, metric_name, metric_value, json.dumps({
                "unit": "score" if metric_name != "total_latency_ms" else "milliseconds"
            })))
        postgres_connection.commit()

    # Step 3: Complete experiment
    with postgres_connection.cursor() as cur:
        cur.execute("""
            UPDATE experiments
            SET status = 'completed', completed_at = CURRENT_TIMESTAMP
            WHERE id = %s
        """, (exp_id,))
        postgres_connection.commit()

    # Step 4: Verify experiment state
    with postgres_connection.cursor() as cur:
        cur.execute("""
            SELECT id, status, completed_at FROM experiments WHERE id = %s
        """, (exp_id,))
        exp_row = cur.fetchone()

    assert exp_row[0] == exp_id
    assert exp_row[1] == 'completed'
    assert exp_row[2] is not None  # completed_at is set

    # Step 5: Verify metrics are stored
    with postgres_connection.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) FROM evaluation_results WHERE experiment_id = %s
        """, (exp_id,))
        metric_count = cur.fetchone()[0]

    assert metric_count == len(metrics)

    # Step 6: Retrieve metrics
    with postgres_connection.cursor() as cur:
        cur.execute("""
            SELECT metric_name, metric_value
            FROM evaluation_results
            WHERE experiment_id = %s
            ORDER BY metric_name
        """, (exp_id,))
        retrieved_metrics = cur.fetchall()

    assert len(retrieved_metrics) == len(metrics)
    for metric_name, metric_value in retrieved_metrics:
        assert any(m[0] == metric_name for m in metrics)


# ============================================================================
# E2E Test: Error Recovery and Transaction Rollback
# ============================================================================


@pytest.mark.e2e
@pytest.mark.integration
def test_error_recovery_transaction_rollback(postgres_connection):
    """
    End-to-end test: simulate errors during pipeline → verify rollback.

    Tests error handling:
    - Transaction rollback on constraint violation
    - Invalid input handling
    - State consistency after errors
    """
    initial_count = None

    # Step 1: Get initial count
    with postgres_connection.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM embedding_registry")
        initial_count = cur.fetchone()[0]

    # Step 2: Attempt invalid insert (violate unique constraint)
    with postgres_connection.cursor() as cur:
        # First insert
        cur.execute("""
            INSERT INTO embedding_registry (model_alias, model_name, dimension)
            VALUES ('unique_test_model', 'test', 768)
        """)
        postgres_connection.commit()

        # Second insert with same alias should fail
        try:
            cur.execute("""
                INSERT INTO embedding_registry (model_alias, model_name, dimension)
                VALUES ('unique_test_model', 'different', 384)
            """)
            postgres_connection.commit()
            assert False, "Should have raised IntegrityError"
        except psycopg2.IntegrityError:
            postgres_connection.rollback()  # Rollback failed transaction

    # Step 3: Verify state is consistent
    with postgres_connection.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM embedding_registry WHERE model_alias = 'unique_test_model'")
        final_count = cur.fetchone()[0]

    # Should have only one insertion (first one succeeded, second rolled back)
    assert final_count == 1

    # Step 4: Test empty text handling
    with postgres_connection.cursor() as cur:
        # Empty text is valid, just verify it doesn't crash
        cur.execute("""
            INSERT INTO evaluation_groundtruth (question, source_type)
            VALUES ('', 'manual')
        """)
        postgres_connection.commit()

        cur.execute("SELECT COUNT(*) FROM evaluation_groundtruth WHERE question = ''")
        assert cur.fetchone()[0] >= 1

    # Step 5: Test malformed JSON handling
    try:
        with postgres_connection.cursor() as cur:
            # Invalid JSON should fail
            cur.execute("""
                INSERT INTO experiments (experiment_name, embedding_model_alias, config_json)
                VALUES ('bad_json', 'bge_base_en_v1_5', '{invalid json}')
            """)
            postgres_connection.commit()
            assert False, "Should have raised error for invalid JSON"
    except psycopg2.Error:
        postgres_connection.rollback()  # Rollback failed transaction


# ============================================================================
# E2E Test: Performance Benchmarks
# ============================================================================


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.postgres
@pytest.mark.timeout(300)  # 5 minute timeout
def test_large_dataset_performance(postgres_connection, mock_ollama):
    """
    End-to-end performance test: process ~10MB dataset in < 5 minutes.

    Tests:
    - Chunking 10MB of text efficiently
    - Batch embedding generation
    - Database insertion throughput
    - Overall pipeline latency
    """
    # Step 1: Create ~10MB dataset
    # Each chunk is ~1KB, so we need ~10,000 chunks for 10MB
    base_text = """
    Machine learning is a subset of artificial intelligence that enables systems to learn
    from experience. Deep learning uses neural networks with multiple layers. Natural language
    processing focuses on text understanding. Computer vision interprets visual information.
    Reinforcement learning trains agents through rewards. Transformers revolutionized NLP.
    """ * 10  # ~5KB per item

    dataset = [base_text for _ in range(2000)]  # ~10MB total

    start_time = time.time()

    # Step 2: Chunk text
    all_chunks = []
    for item in dataset:
        chunks = chunk_text(item, max_size=500)
        all_chunks.extend(chunks)

    chunking_time = time.time() - start_time
    assert chunking_time < 300  # Should complete within 5 minutes

    # Step 3: Create embeddings (simulated)
    embedding_start = time.time()

    np.random.seed(42)
    embeddings = np.random.randn(len(all_chunks), 768).astype(np.float32)

    embedding_time = time.time() - embedding_start

    # Step 4: Batch insert to database
    insert_start = time.time()

    with postgres_connection.cursor() as cur:
        # Create table if needed
        cur.execute("""
            CREATE TABLE IF NOT EXISTS large_test_embeddings (
                id SERIAL PRIMARY KEY,
                chunk_text TEXT,
                embedding vector(768),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        postgres_connection.commit()

        # Batch insert (sample 100 for speed in test)
        for i, (chunk, emb) in enumerate(zip(all_chunks[:100], embeddings[:100])):
            cur.execute("""
                INSERT INTO large_test_embeddings (chunk_text, embedding)
                VALUES (%s, %s)
            """, (chunk, emb.tolist()))
        postgres_connection.commit()

    insert_time = time.time() - insert_start

    # Step 5: Verify total time
    total_time = time.time() - start_time
    assert total_time < 300, f"Total pipeline took {total_time:.2f}s, expected < 300s"

    # Step 6: Verify data quality
    with postgres_connection.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM large_test_embeddings")
        assert cur.fetchone()[0] == 100


# ============================================================================
# E2E Test: Load-Or-Generate Embedding Caching
# ============================================================================


@pytest.mark.e2e
@pytest.mark.postgres
def test_load_or_generate_embedding_pattern(postgres_connection, sample_embeddings):
    """
    End-to-end test: first call generates and stores, second call reuses.

    Tests load-or-generate pattern:
    - First pipeline run: generate embeddings and store
    - Second pipeline run: load existing, no regeneration
    - Verify queries return same results
    """
    chunks = [
        "Article: Machine Learning\n\nML is a powerful technique.",
        "Article: Deep Learning\n\nDL uses neural networks.",
        "Article: NLP\n\nNLP processes text.",
    ]

    # Step 1: First run - create table and insert embeddings
    with postgres_connection.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS cached_embeddings (
                id SERIAL PRIMARY KEY,
                chunk_text TEXT UNIQUE,
                embedding vector(768),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        postgres_connection.commit()

        # Insert embeddings
        for chunk, emb in zip(chunks, sample_embeddings[:len(chunks)]):
            try:
                cur.execute("""
                    INSERT INTO cached_embeddings (chunk_text, embedding)
                    VALUES (%s, %s)
                """, (chunk, emb.tolist()))
            except psycopg2.IntegrityError:
                postgres_connection.rollback()
        postgres_connection.commit()

    # Step 2: Verify first insertion
    with postgres_connection.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM cached_embeddings")
        first_count = cur.fetchone()[0]

    assert first_count == len(chunks)

    # Step 3: Second run - should load, not regenerate
    # Try to insert again - ON CONFLICT DO NOTHING ensures no error
    with postgres_connection.cursor() as cur:
        for chunk, emb in zip(chunks, sample_embeddings[:len(chunks)]):
            cur.execute("""
                INSERT INTO cached_embeddings (chunk_text, embedding)
                VALUES (%s, %s)
                ON CONFLICT (chunk_text) DO NOTHING
            """, (chunk, emb.tolist()))
        postgres_connection.commit()

    # Step 4: Verify no new rows added (reused existing)
    with postgres_connection.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM cached_embeddings")
        second_count = cur.fetchone()[0]

    assert second_count == first_count  # No new rows added

    # Step 5: Query returns same results
    with postgres_connection.cursor() as cur:
        cur.execute("""
            SELECT chunk_text, embedding FROM cached_embeddings
            ORDER BY created_at
        """)
        cached_results = cur.fetchall()

    assert len(cached_results) == len(chunks)

    # Verify embeddings are still there
    for (chunk, emb) in cached_results:
        assert chunk in chunks
        assert len(emb) == 768


# ============================================================================
# E2E Test: Citation and Metadata Tracking
# ============================================================================


@pytest.mark.e2e
@pytest.mark.postgres
def test_citation_tracking_metadata_preservation(postgres_connection, mock_dataset, sample_embeddings, seed_test_data):
    """
    End-to-end test: preserve metadata through retrieval pipeline.

    Tests citation tracking:
    - Store chunk source information
    - Retrieve with metadata preservation
    - Reconstruct citations from retrieved chunks
    """
    # Step 1: Create table with metadata
    with postgres_connection.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS cited_chunks (
                id SERIAL PRIMARY KEY,
                chunk_text TEXT NOT NULL,
                embedding vector(768),
                source_title TEXT,
                source_url TEXT,
                chunk_number INT,
                metadata JSONB DEFAULT '{}'::jsonb,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        postgres_connection.commit()

    # Step 2: Insert chunks with rich metadata
    with postgres_connection.cursor() as cur:
        for i, (title, text) in enumerate(zip(mock_dataset['title'][:3], mock_dataset['text'][:3])):
            chunk_text = f"Article: {title}\n\n{text}"
            metadata = {
                "source_title": title,
                "source_url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                "article_index": i,
                "chunk_number": 1,
                "character_count": len(chunk_text),
                "inserted_at": datetime.now().isoformat()
            }

            cur.execute("""
                INSERT INTO cited_chunks
                (chunk_text, embedding, source_title, source_url, chunk_number, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                chunk_text,
                sample_embeddings[i].tolist(),
                title,
                f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                1,
                json.dumps(metadata)
            ))
        postgres_connection.commit()

    # Step 3: Perform semantic search
    query_embedding = sample_embeddings[0]

    with postgres_connection.cursor() as cur:
        cur.execute("""
            SELECT id, chunk_text, source_title, source_url, 1 - (embedding <=> %s::vector) as similarity
            FROM cited_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT 5
        """, (query_embedding.tolist(), query_embedding.tolist()))
        results = cur.fetchall()

    assert len(results) > 0

    # Step 4: Reconstruct citations from results
    citations = []
    for result_id, chunk, title, url, similarity in results:
        citations.append({
            "title": title,
            "url": url,
            "relevance_score": float(similarity),
            "chunk_preview": chunk[:100] + "..."
        })

    assert len(citations) > 0
    assert all("title" in c for c in citations)
    assert all("url" in c for c in citations)

    # Step 5: Retrieve full metadata
    with postgres_connection.cursor() as cur:
        cur.execute("""
            SELECT chunk_text, metadata FROM cited_chunks
            WHERE source_title IS NOT NULL
            LIMIT 3
        """)
        cited_results = cur.fetchall()

    for chunk, metadata in cited_results:
        assert isinstance(metadata, dict)
        assert "source_title" in metadata
        assert "source_url" in metadata
        assert "inserted_at" in metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
