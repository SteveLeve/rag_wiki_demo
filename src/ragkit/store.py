"""Vector storage in PostgreSQL/pgvector.

Replaces the PostgreSQLVectorDB class that existed in eight drifted copies. Two
behavioural changes worth knowing about:

* The dimension comes from the registry, never from a literal. The old copies
  hardcoded ``vector(768)``, so the bge-large option the README advertised could
  not have worked.
* There is no interactive ``input()`` prompt. The old setup_table() asked
  "preserve or overwrite?" on stdin, which hangs papermill forever -- a large part
  of why notebook execution "failed" in CI.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from . import db, embed, registry

__all__ = ["VectorStore"]


class VectorStore:
    """Read/write access to one embedding model's vectors."""

    def __init__(self, conn, alias: str, *, verify: bool = True):
        self.conn = conn
        self.model = registry.resolve(conn, alias) if verify else registry.get_model(conn, alias)
        if self.model is None:
            raise LookupError(f"{alias!r} is not registered; call registry.register_model first")
        self.table = self.model.table_name
        self.dimension = self.model.dimension

    def __repr__(self) -> str:
        return f"<VectorStore {self.model.alias} dim={self.dimension} table={self.table}>"

    def count(self) -> int:
        with db.cursor(self.conn, commit=False) as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.table}")
            return cur.fetchone()[0]

    def clear(self) -> None:
        """Remove all vectors for this model. Explicit, never prompted."""
        with db.cursor(self.conn) as cur:
            cur.execute(f"TRUNCATE {self.table} RESTART IDENTITY")
        self._sync_count()

    def add_chunks(
        self,
        chunks: Sequence[str],
        *,
        metadata: Sequence[dict[str, Any]] | None = None,
        batch_size: int = 64,
    ) -> int:
        """Embed and store chunks. Returns how many were added."""
        if not chunks:
            return 0
        if metadata is not None and len(metadata) != len(chunks):
            raise ValueError(f"metadata length {len(metadata)} != chunks length {len(chunks)}")

        vectors = embed.embed_texts(chunks, model=self.model.alias, batch_size=batch_size)
        rows = [
            (text, str(vec), json.dumps(metadata[i] if metadata else {}))
            for i, (text, vec) in enumerate(zip(chunks, vectors))
        ]
        with db.cursor(self.conn) as cur:
            cur.executemany(
                f"INSERT INTO {self.table} (chunk_text, embedding, metadata) "
                f"VALUES (%s, %s, %s::jsonb)",
                rows,
            )
        self._sync_count()
        return len(rows)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Nearest neighbours to a query string, closest first.

        Distance is computed in the database by pgvector's `<=>` cosine operator,
        which uses the HNSW index. Similarity is reported as 1 - distance so it
        reads the same way as ragkit.retrieval.cosine_similarity.
        """
        query_vec = embed.embed_one(query, model=self.model.alias)
        return self.search_by_vector(query_vec, top_k=top_k)

    def search_by_vector(self, query_vec: Sequence[float], top_k: int = 5) -> list[dict[str, Any]]:
        if len(query_vec) != self.dimension:
            raise ValueError(
                f"query vector is {len(query_vec)}-dim but {self.model.alias} stores "
                f"{self.dimension}-dim vectors. Embedding models were mixed."
            )
        with db.cursor(self.conn, commit=False) as cur:
            cur.execute(
                f"SELECT id, chunk_text, metadata, 1 - (embedding <=> %s::vector) AS similarity "
                f"FROM {self.table} ORDER BY embedding <=> %s::vector LIMIT %s",
                (str(list(query_vec)), str(list(query_vec)), top_k),
            )
            return [
                {"id": r[0], "chunk_text": r[1], "metadata": r[2], "similarity": float(r[3])}
                for r in cur.fetchall()
            ]

    def _sync_count(self) -> None:
        with db.cursor(self.conn) as cur:
            cur.execute(
                "UPDATE embedding_registry SET embedding_count = "
                f"(SELECT COUNT(*) FROM {self.table}), last_accessed = CURRENT_TIMESTAMP "
                "WHERE model_alias = %s",
                (self.model.alias,),
            )
