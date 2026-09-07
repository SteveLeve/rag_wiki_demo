"""Connections and schema.

This module owns the *only* ``vector(N)`` string in the codebase. Every embedding
table's dimension is read from the registry rather than written into DDL, which is
what stops a hardcoded 768 from surviving in a project that ships 384-, 768- and
1024-dimensional models.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import psycopg2

from . import config
from .models import canonical_alias

__all__ = [
    "connect",
    "cursor",
    "create_core_schema",
    "ensure_embedding_table",
    "table_name_for",
    "SCHEMA_VERSION",
]

SCHEMA_VERSION = 2


def table_name_for(alias_or_tag: str) -> str:
    """The embeddings table for a model. The only permitted way to build one.

    Deriving this by hand is what produced ``embeddings_all-minilm-l6-v2`` -- not
    even a legal unquoted SQL identifier -- in six notebooks while a seventh built
    ``embeddings_all_minilm_l6_v2``.
    """
    return f"embeddings_{canonical_alias(alias_or_tag)}"


def connect(**overrides: Any):
    """Open a connection using the environment-driven config.

    Defaults to port 5433, not 5432. 5432 is routinely occupied by an unrelated
    project's database, and these notebooks create and drop tables.
    """
    settings = {**config.POSTGRES_CONFIG, **overrides}
    try:
        return psycopg2.connect(**settings)
    except psycopg2.OperationalError as exc:
        raise psycopg2.OperationalError(
            f"Could not connect to PostgreSQL at "
            f"{settings['host']}:{settings['port']}/{settings['database']}.\n"
            "Start one with:\n"
            "  docker run -d --name rag-wiki-pgvector \\\n"
            "    -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=rag_db \\\n"
            "    -p 127.0.0.1:5433:5432 pgvector/pgvector:pg16\n"
            f"Original error: {exc}"
        ) from exc


@contextmanager
def cursor(conn, commit: bool = True):
    """Cursor context manager that commits on success and rolls back on error."""
    cur = conn.cursor()
    try:
        yield cur
    except Exception:
        conn.rollback()
        raise
    else:
        if commit:
            conn.commit()
    finally:
        cur.close()


CORE_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS embedding_registry (
    -- Columns 1-10 keep their v1 ordinal positions. Notebooks and tests use
    -- SELECT * with positional indexing, so inserting a column in the middle
    -- silently shifts every one of those reads. New columns go at the end.
    id                  SERIAL PRIMARY KEY,
    model_alias         TEXT UNIQUE NOT NULL,
    -- model_name IS the Ollama tag ('nomic-embed-text'). A separate ollama_tag
    -- column would just be a second place for the same fact to drift.
    model_name          TEXT NOT NULL,
    dimension           INT NOT NULL,
    embedding_count     INT DEFAULT 0,
    chunk_source_dataset TEXT,
    chunk_size_config   INT,
    metadata_json       JSONB DEFAULT '{}'::jsonb,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_accessed       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- v2 additions
    -- Derived, never supplied by callers: the table name is a pure function of
    -- the alias, so generating it removes any way for the two to disagree.
    table_name          TEXT GENERATED ALWAYS AS ('embeddings_' || model_alias) STORED,
    distance_metric     TEXT NOT NULL DEFAULT 'cosine',
    normalized          BOOLEAN NOT NULL DEFAULT TRUE,
    schema_version      INT NOT NULL DEFAULT 2,
    -- Enforced in the database, not just in Python: convention alone already
    -- failed once and produced two spellings of the same model.
    CONSTRAINT model_alias_is_identifier CHECK (model_alias ~ '^[a-z][a-z0-9_]*$'),
    CONSTRAINT dimension_is_sane CHECK (dimension BETWEEN 64 AND 4096)
);

CREATE TABLE IF NOT EXISTS evaluation_groundtruth (
    id                  SERIAL PRIMARY KEY,
    question            TEXT NOT NULL,
    source_type         TEXT CHECK (source_type IN ('llm_generated', 'template_based', 'manual')),
    relevant_chunk_ids  INT ARRAY,
    quality_rating      TEXT CHECK (quality_rating IN ('good', 'bad', 'ambiguous', 'rejected')),
    human_notes         TEXT,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by          TEXT
);

CREATE TABLE IF NOT EXISTS experiments (
    id                   SERIAL PRIMARY KEY,
    experiment_name      TEXT NOT NULL,
    notebook_path        TEXT,
    embedding_model_alias TEXT,
    config_hash          TEXT,
    config_json          JSONB,
    techniques_applied   TEXT ARRAY DEFAULT '{}'::text[],
    started_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at         TIMESTAMP,
    status               TEXT DEFAULT 'running' CHECK (status IN ('running', 'completed', 'failed')),
    notes                TEXT,
    FOREIGN KEY (embedding_model_alias) REFERENCES embedding_registry(model_alias)
);

CREATE TABLE IF NOT EXISTS evaluation_results (
    id             SERIAL PRIMARY KEY,
    experiment_id  INT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    metric_name    TEXT NOT NULL,
    metric_value   FLOAT NOT NULL,
    metric_details_json JSONB DEFAULT '{}'::jsonb,
    question_id    INT REFERENCES evaluation_groundtruth(id),
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_experiments_config_hash ON experiments(config_hash);
CREATE INDEX IF NOT EXISTS idx_results_experiment ON evaluation_results(experiment_id);
"""


def create_core_schema(conn) -> None:
    """Create the four shared tables. Idempotent."""
    with cursor(conn) as cur:
        cur.execute(CORE_SCHEMA)


def ensure_embedding_table(conn, alias: str, dimension: int) -> str:
    """Create this model's embeddings table at its own dimension. Idempotent.

    Returns the table name. This function contains the only ``vector(N)`` in the
    codebase; scripts/nb_lint.py bans the literal from notebooks so it stays that way.
    """
    alias = canonical_alias(alias)
    table = table_name_for(alias)
    if not 64 <= dimension <= 4096:
        raise ValueError(f"implausible embedding dimension {dimension} for {alias!r}")

    with cursor(conn) as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        # Table and column names cannot be parameterised; alias is constrained to
        # ^[a-z][a-z0-9_]*$ by canonical_alias, and dimension is range-checked above.
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id         SERIAL PRIMARY KEY,
                chunk_text TEXT NOT NULL,
                embedding  vector({dimension}) NOT NULL,
                metadata   JSONB DEFAULT '{{}}'::jsonb,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{table}_embedding "
            f"ON {table} USING hnsw (embedding vector_cosine_ops)"
        )
    return table


def actual_dimension(conn, table: str) -> int | None:
    """The vector width PostgreSQL actually stores for a table, or None if absent.

    Used to catch the orphaned-table case: a stale embeddings table left behind by
    a partial rename will answer queries happily with obsolete vectors.
    """
    with cursor(conn, commit=False) as cur:
        cur.execute(
            "SELECT atttypmod FROM pg_attribute "
            "WHERE attrelid = to_regclass(%s) AND attname = 'embedding'",
            (table,),
        )
        row = cur.fetchone()
    return int(row[0]) if row and row[0] and row[0] > 0 else None
