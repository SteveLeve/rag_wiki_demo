"""Experiment tracking.

``experiments.embedding_model_alias`` carries a foreign key to
``embedding_registry.model_alias``. That FK is why the old alias inconsistency was
a hard failure rather than cosmetic drift: notebooks calling start_experiment with
``bge_base_en_v1_5`` hit a registry holding only ``bge_base_en_v1.5``, and every
advanced-tier run died on the insert. Routing every alias through
canonical_alias() before it reaches the database is what fixes it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

from . import db
from .models import canonical_alias

__all__ = [
    "compute_config_hash",
    "start_experiment",
    "complete_experiment",
    "save_metrics",
    "compare_experiments",
]


def compute_config_hash(config: dict[str, Any]) -> str:
    """A stable SHA-256 over a config dict, for finding comparable runs.

    Sorted keys and a canonical separator, so two dicts that differ only in
    insertion order hash identically.
    """
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def start_experiment(
    conn,
    name: str,
    embedding_alias: str,
    config: dict[str, Any],
    *,
    notebook_path: str | None = None,
    techniques: Sequence[str] = (),
) -> int:
    """Open an experiment row and return its id.

    Verifies the model is registered first, so a failure names the real problem
    ("run foundation/02") instead of surfacing as a foreign-key violation.
    """
    from . import registry

    alias = canonical_alias(embedding_alias)
    registry.resolve(conn, alias)  # raises with an actionable message if missing

    with db.cursor(conn) as cur:
        cur.execute(
            """
            INSERT INTO experiments
                (experiment_name, notebook_path, embedding_model_alias,
                 config_hash, config_json, techniques_applied, status)
            VALUES (%s, %s, %s, %s, %s::jsonb, %s, 'running')
            RETURNING id
            """,
            (name, notebook_path, alias, compute_config_hash(config),
             json.dumps(config, default=str), list(techniques)),
        )
        return cur.fetchone()[0]


def complete_experiment(conn, experiment_id: int, status: str = "completed", notes: str | None = None) -> None:
    """Close an experiment row."""
    if status not in {"completed", "failed"}:
        raise ValueError(f"status must be 'completed' or 'failed', got {status!r}")
    with db.cursor(conn) as cur:
        cur.execute(
            "UPDATE experiments SET status = %s, completed_at = CURRENT_TIMESTAMP, "
            "notes = COALESCE(%s, notes) WHERE id = %s",
            (status, notes, experiment_id),
        )


def save_metrics(conn, experiment_id: int, metrics: dict[str, float], question_id: int | None = None) -> None:
    """Store one experiment's metric values."""
    if not metrics:
        return
    with db.cursor(conn) as cur:
        cur.executemany(
            "INSERT INTO evaluation_results (experiment_id, metric_name, metric_value, question_id) "
            "VALUES (%s, %s, %s, %s)",
            [(experiment_id, name, float(value), question_id) for name, value in metrics.items()],
        )


def compare_experiments(conn, experiment_ids: Sequence[int] | None = None) -> list[dict[str, Any]]:
    """Mean metric values per experiment, for the dashboard and comparison notebooks."""
    sql = """
        SELECT e.id, e.experiment_name, e.embedding_model_alias,
               r.metric_name, AVG(r.metric_value) AS mean_value, COUNT(*) AS n
        FROM experiments e
        JOIN evaluation_results r ON r.experiment_id = e.id
        {where}
        GROUP BY e.id, e.experiment_name, e.embedding_model_alias, r.metric_name
        ORDER BY e.id, r.metric_name
    """.format(where="WHERE e.id = ANY(%s)" if experiment_ids else "")
    with db.cursor(conn, commit=False) as cur:
        cur.execute(sql, (list(experiment_ids),) if experiment_ids else ())
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
