#!/usr/bin/env python3
"""Drop every table this curriculum creates, then rebuild the core schema.

This is *the* answer to a schema change. The data is regenerable -- re-embedding
the sample corpus on CPU takes minutes -- and the alternative is worse than it
looks: a half-migrated database can hold a registry row under one alias spelling
and an embeddings table under another. The orphaned table still answers queries,
with stale vectors, and the eval numbers come out plausible and wrong.

    python scripts/reset_db.py            # show what would be dropped
    python scripts/reset_db.py --yes      # actually drop it

Honours RAG_PG_* (see ragkit/config.py). It refuses to touch anything on port
5432 unless you insist, because 5432 is routinely another project's database.
"""
from __future__ import annotations

import argparse
import sys

from ragkit import config, db

# Everything create_core_schema builds, plus the per-model embeddings tables and
# the cache advanced-techniques/11 writes. Children before parents.
CORE_TABLES = (
    "evaluation_results",
    "experiments",
    "evaluation_groundtruth",
    "contextualized_chunks",
    "embedding_registry",
)


def embedding_tables(conn) -> list[str]:
    with db.cursor(conn, commit=False) as cur:
        cur.execute(
            "SELECT tablename FROM pg_tables "
            "WHERE schemaname = 'public' AND tablename LIKE 'embeddings\\_%' "
            "ORDER BY tablename"
        )
        return [row[0] for row in cur.fetchall()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--yes", action="store_true", help="actually drop the tables")
    ap.add_argument("--allow-5432", action="store_true",
                    help="permit a reset against port 5432 (usually someone else's database)")
    args = ap.parse_args()

    where = (f"{config.POSTGRES_CONFIG['host']}:{config.POSTGRES_CONFIG['port']}"
             f"/{config.POSTGRES_CONFIG['database']}")
    if config.POSTGRES_CONFIG["port"] == 5432 and not args.allow_5432:
        print(f"Refusing to reset {where}: port 5432 is usually another project's "
              f"database.\nSet RAG_PG_PORT=5433, or pass --allow-5432 if you are sure.",
              file=sys.stderr)
        return 2

    conn = db.connect()
    targets = embedding_tables(conn) + list(CORE_TABLES)

    print(f"target: {where}")
    for name in targets:
        print(f"  drop {name}")

    if not args.yes:
        print("\nDry run. Re-run with --yes to drop these and rebuild the core schema.")
        conn.close()
        return 0

    with db.cursor(conn) as cur:
        for name in targets:
            cur.execute(f"DROP TABLE IF EXISTS {name} CASCADE")
    db.create_core_schema(conn)
    conn.commit()
    conn.close()
    print(f"\nDropped {len(targets)} table(s) and rebuilt the core schema.")
    print("Re-run foundation/02 to regenerate embeddings.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
