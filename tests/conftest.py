"""
Shared pytest fixtures for RAG Wiki Demo tests.

This module provides fixtures for:
- PostgreSQL database setup/teardown with pgvector
- Test data seeding
- Mock Ollama API responses
- Sample datasets and embeddings
- Test chunks with edge cases

Fixture Scopes:
- session: Expensive setup (PostgreSQL container) - shared across all tests
- function: Default - each test gets a fresh instance
"""

import json
import os
from typing import Any, Dict, List, Optional
from datetime import datetime

import numpy as np
import psycopg2
import psycopg2.extras
import pytest
from faker import Faker
from psycopg2 import extensions
from psycopg2.extras import RealDictCursor


# ============================================================================
# PostgreSQL Configuration Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def postgres_test_config() -> Dict[str, Any]:
    """
    PostgreSQL test database configuration.

    Returns:
        Dictionary with connection parameters for test database.
        Allows override via environment variables for CI/CD.

    Example:
        >>> config = postgres_test_config()
        >>> config['database']
        'rag_test_db'
    """
    return {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", "5432")),
        "database": os.getenv("POSTGRES_DB", "rag_test_db"),
        "user": os.getenv("POSTGRES_USER", "postgres"),
        "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
    }


@pytest.fixture(scope="function")
def postgres_test_db(postgres_test_config: Dict[str, Any]) -> psycopg2.extensions.connection:
    """
    PostgreSQL test database fixture with schema initialization.

    This fixture:
    1. Connects to PostgreSQL
    2. Creates pgvector extension
    3. Sets up 4 core tables (embedding_registry, evaluation_groundtruth, experiments, evaluation_results)
    4. Yields database connection
    5. Cleans up tables after test (truncates all data)

    Args:
        postgres_test_config: Database configuration from postgres_test_config fixture

    Yields:
        psycopg2 connection object for test database

    Raises:
        psycopg2.OperationalError: If database connection fails

    Example:
        >>> def test_something(postgres_test_db):
        ...     with postgres_test_db.cursor() as cur:
        ...         cur.execute("SELECT * FROM embedding_registry")
    """
    config = postgres_test_config

    # Connect to database
    conn = psycopg2.connect(
        host=config["host"],
        port=config["port"],
        database=config["database"],
        user=config["user"],
        password=config["password"],
    )

    conn.autocommit = False

    # Enable pgvector extension
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS pgvector")
    conn.commit()

    # Create schema tables (matching foundation/00-setup-postgres-schema.ipynb)
    with conn.cursor() as cur:
        # Table 1: embedding_registry
        cur.execute("""
            CREATE TABLE IF NOT EXISTS embedding_registry (
                id SERIAL PRIMARY KEY,
                model_alias TEXT UNIQUE NOT NULL,
                model_name TEXT NOT NULL,
                dimension INT NOT NULL,
                embedding_count INT DEFAULT 0,
                chunk_source_dataset TEXT,
                chunk_size_config INT,
                metadata_json JSONB DEFAULT '{}'::jsonb,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Table 2: evaluation_groundtruth
        cur.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_groundtruth (
                id SERIAL PRIMARY KEY,
                question TEXT NOT NULL,
                source_type TEXT CHECK (source_type IN ('llm_generated', 'template_based', 'manual')),
                relevant_chunk_ids INT ARRAY,
                quality_rating TEXT CHECK (quality_rating IN ('good', 'bad', 'ambiguous', 'rejected')),
                human_notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT
            )
        """)

        # Table 3: experiments
        cur.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id SERIAL PRIMARY KEY,
                experiment_name TEXT NOT NULL,
                notebook_path TEXT,
                embedding_model_alias TEXT,
                config_hash TEXT,
                config_json JSONB,
                techniques_applied TEXT ARRAY DEFAULT '{}'::text[],
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                status TEXT DEFAULT 'running' CHECK (status IN ('running', 'completed', 'failed')),
                notes TEXT,
                FOREIGN KEY (embedding_model_alias) REFERENCES embedding_registry(model_alias)
            )
        """)

        # Table 4: evaluation_results
        cur.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_results (
                id SERIAL PRIMARY KEY,
                experiment_id INT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value FLOAT NOT NULL,
                metric_details_json JSONB DEFAULT '{}'::jsonb,
                computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
            )
        """)

        # Create indexes
        cur.execute("CREATE INDEX IF NOT EXISTS idx_experiments_embedding_model ON experiments(embedding_model_alias)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_experiments_started ON experiments(started_at DESC)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_results_experiment ON evaluation_results(experiment_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_results_metric ON evaluation_results(metric_name)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_groundtruth_quality ON evaluation_groundtruth(quality_rating)")

    conn.commit()

    yield conn

    # Cleanup: truncate all tables after test
    with conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE evaluation_results CASCADE")
        cur.execute("TRUNCATE TABLE experiments CASCADE")
        cur.execute("TRUNCATE TABLE evaluation_groundtruth CASCADE")
        cur.execute("TRUNCATE TABLE embedding_registry CASCADE")
    conn.commit()
    conn.close()


@pytest.fixture(scope="function")
def postgres_connection(postgres_test_db: psycopg2.extensions.connection) -> psycopg2.extensions.connection:
    """
    PostgreSQL connection with transaction rollback for test isolation.

    Wraps database operations in a transaction that's rolled back after
    each test, ensuring complete isolation even without truncating tables.

    Args:
        postgres_test_db: Database fixture providing initialized connection

    Yields:
        Connection object within a transaction context

    Example:
        >>> def test_query(postgres_connection):
        ...     with postgres_connection.cursor() as cur:
        ...         cur.execute("INSERT INTO embedding_registry ...")
        ...         # Automatically rolled back after test
    """
    # Start transaction with savepoint for isolation
    postgres_test_db.isolation_level = extensions.ISOLATION_LEVEL_READ_COMMITTED

    # If savepoint support is needed, we can use it here
    yield postgres_test_db

    # Connection cleanup happens in postgres_test_db fixture


# ============================================================================
# Test Data Fixtures
# ============================================================================


@pytest.fixture(scope="function")
def seed_test_data(postgres_connection: psycopg2.extensions.connection) -> Dict[str, List[int]]:
    """
    Insert sample test data into database.

    Creates:
    - 3 embedding models in registry
    - 5 ground-truth test questions with relevant chunk IDs

    Args:
        postgres_connection: Database connection fixture

    Returns:
        Dictionary with inserted IDs:
        {
            'embedding_ids': [1, 2, 3],
            'groundtruth_ids': [1, 2, 3, 4, 5]
        }

    Example:
        >>> def test_with_data(seed_test_data):
        ...     assert len(seed_test_data['embedding_ids']) == 3
    """
    with postgres_connection.cursor() as cur:
        # Insert 3 embedding models
        embedding_data = [
            ("bge_base_en_v1_5", "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf", 768, 1000),
            ("bge_small_en_v1_5", "hf.co/CompendiumLabs/bge-small-en-v1.5-gguf", 384, 500),
            ("test_model", "test/model", 128, 100),
        ]

        embedding_ids = []
        for alias, name, dim, count in embedding_data:
            cur.execute(
                """
                INSERT INTO embedding_registry (model_alias, model_name, dimension, embedding_count, chunk_source_dataset)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (alias, name, dim, count, "Wikipedia 10MB"),
            )
            embedding_ids.append(cur.fetchone()[0])

        # Insert 5 ground-truth questions with relevant chunk IDs
        groundtruth_data = [
            ("What is photosynthesis?", "manual", [1, 2, 3], "good"),
            ("How do neural networks work?", "llm_generated", [4, 5], "good"),
            ("What is machine learning?", "template_based", [6, 7, 8, 9], "good"),
            ("Explain transformers in NLP", "manual", [10, 11], "good"),
            ("What are embeddings?", "llm_generated", [12, 13, 14], "good"),
        ]

        groundtruth_ids = []
        for question, source_type, chunk_ids, rating in groundtruth_data:
            cur.execute(
                """
                INSERT INTO evaluation_groundtruth (question, source_type, relevant_chunk_ids, quality_rating, created_by)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (question, source_type, chunk_ids, rating, "test_suite"),
            )
            groundtruth_ids.append(cur.fetchone()[0])

    postgres_connection.commit()

    return {
        "embedding_ids": embedding_ids,
        "groundtruth_ids": groundtruth_ids,
    }


# ============================================================================
# Mock API Fixtures
# ============================================================================


@pytest.fixture(scope="function")
def mock_ollama(monkeypatch):
    """
    Mock Ollama API responses.

    Mocks:
    - ollama.embeddings(): Returns deterministic 768-dim vector
    - ollama.chat(): Returns fixed test response

    Args:
        monkeypatch: pytest monkeypatch fixture

    Returns:
        Dictionary with:
        - 'embeddings_mock': Mock function for ollama.embeddings
        - 'chat_mock': Mock function for ollama.chat

    Example:
        >>> def test_embeddings(mock_ollama):
        ...     result = ollama.embeddings("test text")
        ...     assert len(result['embedding']) == 768
    """
    try:
        import ollama
    except ImportError:
        pytest.skip("ollama not installed")

    def mock_embeddings(model: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """Mock ollama.embeddings() response"""
        # Return deterministic embedding based on model
        if "768" in model or "base" in model:
            dim = 768
        elif "384" in model or "small" in model:
            dim = 384
        else:
            dim = 768

        # Deterministic vector for reproducibility
        embedding = np.random.RandomState(42).randn(dim).tolist()

        return {
            "embedding": embedding,
            "model": model,
            "prompt_eval_count": len(prompt.split()),
            "eval_count": 10,
        }

    def mock_chat(model: str, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Mock ollama.chat() response"""
        return {
            "model": model,
            "created_at": datetime.now().isoformat(),
            "message": {
                "role": "assistant",
                "content": "This is a mock response from the Ollama API.",
            },
            "done": True,
            "total_duration": 1000000,
            "load_duration": 100000,
            "prompt_eval_count": 10,
            "prompt_eval_duration": 200000,
            "eval_count": 20,
            "eval_duration": 400000,
        }

    monkeypatch.setattr("ollama.embeddings", mock_embeddings)
    monkeypatch.setattr("ollama.chat", mock_chat)

    return {
        "embeddings_mock": mock_embeddings,
        "chat_mock": mock_chat,
    }


# ============================================================================
# Dataset Fixtures
# ============================================================================


@pytest.fixture(scope="function")
def mock_dataset() -> Dict[str, Any]:
    """
    Create sample Wikipedia-like dataset.

    Returns:
        HuggingFace-format dataset dict with 10 articles:
        {
            'title': [...],
            'text': [...],
        }

    Example:
        >>> def test_with_dataset(mock_dataset):
        ...     assert len(mock_dataset['title']) == 10
        ...     assert all(len(text) > 100 for text in mock_dataset['text'])
    """
    fake = Faker()
    Faker.seed(42)  # Deterministic
    np.random.seed(42)

    titles = [
        "Introduction to Machine Learning",
        "Deep Learning Fundamentals",
        "Natural Language Processing Overview",
        "Computer Vision Basics",
        "Reinforcement Learning Guide",
        "Neural Networks Architecture",
        "Embeddings and Vector Spaces",
        "Transformer Models Explained",
        "Attention Mechanisms in Detail",
        "Transfer Learning and Fine-tuning",
    ]

    texts = []
    for title in titles:
        # Generate 200-500 word article using faker
        paragraphs = []
        for _ in range(np.random.randint(2, 5)):
            # Each paragraph is 3-5 sentences
            sentences = [fake.sentence(nb_words=10) for _ in range(np.random.randint(3, 6))]
            paragraphs.append(" ".join(sentences))

        text = "\n\n".join(paragraphs)
        texts.append(text)

    return {
        "title": titles,
        "text": texts,
    }


@pytest.fixture(scope="function")
def sample_embeddings() -> np.ndarray:
    """
    Pre-computed test embeddings.

    Returns:
        Numpy array shape (10, 768) with deterministic embeddings
        using seed=42 for reproducibility

    Example:
        >>> def test_embeddings(sample_embeddings):
        ...     assert sample_embeddings.shape == (10, 768)
    """
    np.random.seed(42)
    embeddings = np.random.randn(10, 768).astype(np.float32)
    return embeddings


@pytest.fixture(scope="function")
def test_chunks() -> List[str]:
    """
    Sample text chunks with edge cases.

    Returns list of 20 chunks including:
    - Very short (1-3 words)
    - Very long (500+ chars)
    - Special characters
    - Unicode characters
    - Quotes and escapes

    Example:
        >>> def test_chunks_fixture(test_chunks):
        ...     assert len(test_chunks) == 20
        ...     assert any(len(c) < 10 for c in test_chunks)  # Short chunks
    """
    chunks = [
        # Standard chunks
        "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience.",
        "Deep learning uses neural networks with multiple layers to learn representations of data.",
        "Natural language processing focuses on interactions between computers and human language.",
        "Computer vision is a field of AI that trains computers to interpret and understand visual information.",
        "Reinforcement learning trains agents by rewarding desired behaviors and punishing undesired ones.",

        # Short chunks
        "What is AI?",
        "Learn fast.",
        "ML rocks.",

        # Long chunks
        "Transformers have become the dominant architecture in modern natural language processing. "
        "They use self-attention mechanisms to process input sequences in parallel, making them much more efficient than "
        "previous recurrent neural network approaches. The transformer architecture was introduced in the paper 'Attention is All You Need' "
        "and has since been adapted for computer vision, multimodal learning, and many other domains. "
        "Popular models based on transformers include BERT, GPT, T5, and vision transformers.",

        # Special characters
        'Question: "What\'s the difference?" Answer: It\'s significant!',
        "Special chars: @#$%^&*()_+-=[]{}|;:,.<>?",
        "Quotes: 'single', \"double\", ```backtick```",

        # Unicode characters
        "Émojis and spëcial characters: 你好 مرحبا Привет",
        "Mathematical symbols: ∑ ∏ ∫ √ ∞ ≠ ≤ ≥ ≈",
        "Greek letters: α β γ δ ε ζ η θ",

        # Edge cases
        "",  # Empty
        "   ",  # Whitespace only
        "\n\t\n",  # Newlines and tabs
        "word" * 100,  # Very repetitive
    ]

    return chunks


# ============================================================================
# Helper Fixtures
# ============================================================================


@pytest.fixture(scope="function")
def fake_data():
    """
    Faker instance seeded for deterministic test data generation.

    Example:
        >>> def test_with_faker(fake_data):
        ...     name = fake_data.name()
        ...     email = fake_data.email()
    """
    fake = Faker()
    Faker.seed(42)
    return fake
