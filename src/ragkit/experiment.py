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
    "CONFIG_HASH_LENGTH",
    "compute_config_hash",
    "start_experiment",
    "complete_experiment",
    "save_metrics",
    "compare_experiments",
]


#: Length of the stored config hash. 12 hex characters is 48 bits -- ample for
#: deduplicating experiment configurations, and short enough to read in a
#: dashboard column or compare by eye. Values are stored in experiments.config_hash,
#: so changing this invalidates every existing row.
CONFIG_HASH_LENGTH = 12


def compute_config_hash(config: dict[str, Any], length: int = CONFIG_HASH_LENGTH) -> str:
    """A stable, truncated SHA-256 over a config dict, for finding comparable runs.

    Sorted keys and a canonical separator, so two dicts differing only in
    insertion order hash identically. Pass ``length=64`` for the untruncated digest.
    """
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def start_experiment(
    conn,
    experiment_name: str,
    embedding_model_alias: str,
    config: dict[str, Any] | None = None,
    notebook_path: str | None = None,
    techniques: Sequence[str] = (),
    notes: str | None = None,
) -> int:
    """Open an experiment row and return its id.

    Parameter names match what the notebooks already call this with; the library
    was the outlier, not them.

    The alias is canonicalized and its registration verified before the insert, so
    a missing model produces "not registered, run foundation/02" rather than a
    foreign-key violation from deep inside psycopg2. That FK
    (experiments.embedding_model_alias -> embedding_registry.model_alias) is why
    the old alias inconsistency broke every advanced-tier run.
    """
    from . import registry

    alias = canonical_alias(embedding_model_alias)
    registry.resolve(conn, alias)  # raises with an actionable message if missing
    config = config or {}

    with db.cursor(conn) as cur:
        cur.execute(
            """
            INSERT INTO experiments
                (experiment_name, notebook_path, embedding_model_alias,
                 config_hash, config_json, techniques_applied, notes, status)
            VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s, 'running')
            RETURNING id
            """,
            (experiment_name, notebook_path, alias, compute_config_hash(config),
             json.dumps(config, default=str), list(techniques), notes),
        )
        return cur.fetchone()[0]


def complete_experiment(
    conn, experiment_id: int, status: str = "completed", notes: str | None = None
) -> bool:
    """Close an experiment row. Returns True on success, matching the notebook contract."""
    if status not in {"completed", "failed"}:
        raise ValueError(f"status must be 'completed' or 'failed', got {status!r}")
    with db.cursor(conn) as cur:
        cur.execute(
            "UPDATE experiments SET status = %s, completed_at = CURRENT_TIMESTAMP, "
            "notes = COALESCE(%s, notes) WHERE id = %s",
            (status, notes, experiment_id),
        )
    return True


def save_metrics(
    conn,
    experiment_id: int,
    metrics: dict[str, Any],
    export_to_file: bool = True,
    export_dir: str = "data/experiment_results",
    question_id: int | None = None,
) -> tuple[bool, str]:
    """Store one experiment's metrics, optionally mirroring them to a JSON file.

    Args:
        metrics: ``{name: value}``, or ``{name: {"value": v, "details": {...}}}``
            when a metric carries supporting detail (per-question breakdowns and
            the like).
        export_to_file: Also write a JSON copy, so results survive a dropped database.

    Returns:
        ``(success, message)``.
    """
    if not metrics:
        return True, "no metrics to save"

    rows = []
    for name, data in metrics.items():
        if isinstance(data, dict):
            value, details = data.get("value", 0.0), data.get("details", {})
        else:
            value, details = data, {}
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            # evaluation_results.metric_value is a float column. Without this,
            # putting a config hash or a label in the metrics dict fails as a bare
            # "could not convert string to float", naming no metric.
            raise TypeError(
                f"metric {name!r} must be numeric, got {value!r}. "
                "Labels and identifiers belong on the experiment row or in the "
                "metric's details, not in metric_value."
            ) from exc
        rows.append((experiment_id, name, numeric, json.dumps(details or {}), question_id))

    try:
        with db.cursor(conn) as cur:
            cur.executemany(
                "INSERT INTO evaluation_results "
                "(experiment_id, metric_name, metric_value, metric_details_json, question_id) "
                "VALUES (%s, %s, %s, %s::jsonb, %s)",
                rows,
            )
    except Exception as exc:  # pragma: no cover - surfaced to the notebook user
        return False, f"failed to save metrics: {exc}"

    message = f"saved {len(rows)} metrics"
    if export_to_file:
        import os
        from datetime import datetime

        os.makedirs(export_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(export_dir, f"experiment_{experiment_id}_{stamp}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"experiment_id": experiment_id, "metrics": metrics}, fh, indent=2, default=str)
        message += f", exported to {path}"

    return True, message


def compare_experiments(
    conn,
    experiment_ids: Sequence[int] | None = None,
    metric_names: Sequence[str] | None = None,
):
    """Compare experiments side by side: experiments as rows, metrics as columns.

    Returns a pandas DataFrame.

    Experiments with no recorded embedding model show as "(unspecified)" rather
    than disappearing. The notebook version pivoted on the raw column, and
    pandas drops NaN index groups, so a run recorded without a model silently
    vanished from the comparison instead of showing up with blank metrics.
    """
    import pandas as pd

    clauses, params = [], []
    if experiment_ids:
        clauses.append("e.id = ANY(%s)")
        params.append(list(experiment_ids))
    if metric_names:
        clauses.append("r.metric_name = ANY(%s)")
        params.append(list(metric_names))
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    sql = f"""
        SELECT e.id, e.experiment_name, e.embedding_model_alias,
               r.metric_name, r.metric_value
        FROM experiments e
        LEFT JOIN evaluation_results r ON e.id = r.experiment_id
        {where}
    """
    with db.cursor(conn, commit=False) as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]

    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df

    # Fill before pivoting: pandas drops NaN index groups, which is what made
    # alias-less experiments disappear. Filling keeps the row and labels it.
    df["embedding_model_alias"] = df["embedding_model_alias"].fillna("(unspecified)")

    return df.pivot_table(
        index=["id", "experiment_name", "embedding_model_alias"],
        columns="metric_name",
        values="metric_value",
    ).reset_index()
