"""The embedding registry: which models exist, where their vectors live.

The registry is the single source of truth binding an alias to its Ollama tag,
its dimension, and its table. Reading dimension back from here -- rather than
assuming 768 -- is what makes the rest of the curriculum dimension-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import db
from .models import EMBEDDING_MODELS, canonical_alias, get_embedding_model

__all__ = [
    "RegisteredModel",
    "register_model",
    "seed_catalog",
    "get_model",
    "list_models",
    "resolve",
]


@dataclass(frozen=True)
class RegisteredModel:
    alias: str
    ollama_tag: str
    table_name: str
    dimension: int
    embedding_count: int = 0
    normalized: bool = True


def register_model(
    conn,
    alias_or_tag: str,
    *,
    source_dataset: str | None = None,
    chunk_size: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> RegisteredModel:
    """Register a catalog model and create its embeddings table. Idempotent.

    Alias, tag, dimension and table name all come from the catalog, so a caller
    cannot register a model under a name that disagrees with what Ollama serves.
    """
    spec = get_embedding_model(alias_or_tag)
    table = db.table_name_for(spec.alias)

    db.ensure_embedding_table(conn, spec.alias, spec.dimension)

    with db.cursor(conn) as cur:
        cur.execute(
            """
            INSERT INTO embedding_registry
                (model_alias, model_name, dimension, normalized, schema_version,
                 chunk_source_dataset, chunk_size_config, metadata_json)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (model_alias) DO UPDATE SET
                model_name    = EXCLUDED.model_name,
                dimension     = EXCLUDED.dimension,
                last_accessed = CURRENT_TIMESTAMP
            """,
            (
                spec.alias, spec.ollama_tag, spec.dimension,
                spec.normalized, db.SCHEMA_VERSION, source_dataset, chunk_size,
                __import__("json").dumps(metadata or {}),
            ),
        )
    return RegisteredModel(spec.alias, spec.ollama_tag, table, spec.dimension)


def seed_catalog(conn) -> list[RegisteredModel]:
    """Register every catalog model.

    Called by foundation/00 so the advanced tier can never again reference an
    alias that nothing created -- the failure that made eight notebooks raise
    ValueError on a registry miss.
    """
    return [register_model(conn, alias) for alias in EMBEDDING_MODELS]


def get_model(conn, alias_or_tag: str) -> RegisteredModel | None:
    """Look up one registered model, or None if it has not been registered."""
    alias = canonical_alias(alias_or_tag)
    with db.cursor(conn, commit=False) as cur:
        cur.execute(
            "SELECT model_alias, model_name, table_name, dimension, embedding_count, normalized "
            "FROM embedding_registry WHERE model_alias = %s",
            (alias,),
        )
        row = cur.fetchone()
    return RegisteredModel(*row) if row else None


def list_models(conn) -> list[RegisteredModel]:
    """Every registered model, most recently used first."""
    with db.cursor(conn, commit=False) as cur:
        cur.execute(
            "SELECT model_alias, model_name, table_name, dimension, embedding_count, normalized "
            "FROM embedding_registry ORDER BY last_accessed DESC"
        )
        rows = cur.fetchall()
    return [RegisteredModel(*r) for r in rows]


def resolve(conn, alias_or_tag: str) -> RegisteredModel:
    """Look up a model, verifying its table really has the dimension claimed.

    Guards the orphaned-table case. A partial rename can leave a stale
    ``embeddings_*`` table that answers queries successfully with obsolete
    vectors, producing plausible-looking and wrong evaluation numbers. Better to
    fail loudly.
    """
    alias = canonical_alias(alias_or_tag)
    model = get_model(conn, alias)
    if model is None:
        known = ", ".join(m.alias for m in list_models(conn)) or "(none registered)"
        raise LookupError(
            f"Embedding model {alias!r} is not registered. Registered: {known}. "
            "Run foundation/00-setup-postgres-schema.ipynb, or "
            "ragkit.registry.seed_catalog(conn), to register the catalog."
        )

    actual = db.actual_dimension(conn, model.table_name)
    if actual is None:
        raise LookupError(
            f"{alias!r} is registered but its table {model.table_name!r} does not exist. "
            "Run scripts/reset_db.py --yes to rebuild."
        )
    if actual != model.dimension:
        raise RuntimeError(
            f"Dimension mismatch for {alias!r}: registry says {model.dimension}, "
            f"table {model.table_name!r} stores {actual}. This is usually a stale table "
            "left by a partial rename. Run scripts/reset_db.py --yes."
        )
    return model
