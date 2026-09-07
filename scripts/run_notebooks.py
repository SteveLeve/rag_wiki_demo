#!/usr/bin/env python3
"""Execute every notebook with papermill, in curriculum order.

This is the acceptance gate. Run it with RAG_FAKE_MODELS=1 to exercise the whole
curriculum with no Ollama server and no model downloads.

    RAG_FAKE_MODELS=1 RAG_PG_PORT=5433 python scripts/run_notebooks.py
    python scripts/run_notebooks.py foundation intermediate   # a subset
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TIERS = ("foundation", "intermediate", "advanced-techniques", "evaluation-lab")

# Within a tier, prerequisites first. INDEX notebooks are markdown-only.
ORDER = {
    "foundation": ["00-setup-postgres-schema", "00-registry-and-tracking-utilities",
                   "00-load-or-generate-pattern", "01-basic-rag-in-memory",
                   "02-rag-postgresql-persistent"],
}

# The tier order is not simply the directory order. evaluation-lab/01 populates
# evaluation_groundtruth, and the advanced-techniques notebooks measure themselves
# against it -- so it has to run before them, despite living in a later folder.
# Running the tiers naively leaves ground_truth_questions empty and the advanced
# notebooks fail on an empty list.
GROUND_TRUTH_FIRST = ("evaluation-lab", "01-create-ground-truth-human-in-loop")


def ordered(tier: str) -> list[Path]:
    paths = sorted((ROOT / tier).glob("*.ipynb"))
    if tier not in ORDER:
        return paths
    rank = {name: i for i, name in enumerate(ORDER[tier])}
    return sorted(paths, key=lambda p: (rank.get(p.stem, 99), p.stem))


def main() -> int:
    import papermill

    tiers = sys.argv[1:] or list(TIERS)
    failures: list[tuple[str, str]] = []
    ran = 0


    with tempfile.TemporaryDirectory() as tmp:
        # Seed ground truth before the tiers that consume it.
        gt_tier, gt_stem = GROUND_TRUTH_FIRST
        gt_path = ROOT / gt_tier / f"{gt_stem}.ipynb"
        seeded = set()
        if gt_path.exists() and gt_tier in tiers and "advanced-techniques" in tiers:
            print(f"\n=== prerequisite " + "=" * 55)
            start = time.time()
            try:
                papermill.execute_notebook(
                    str(gt_path), str(Path(tmp) / gt_path.name),
                    kernel_name="python3", cwd=str(ROOT), progress_bar=False,
                )
            except Exception as exc:
                detail = next((l for l in reversed(str(exc).strip().splitlines()) if l.strip()), "")[:110]
                print(f"  FAIL  {gt_path.relative_to(ROOT)}  ({time.time()-start:.0f}s)  {detail}")
                failures.append((str(gt_path.relative_to(ROOT)), detail))
            else:
                print(f"  ok    {gt_path.relative_to(ROOT)}  ({time.time()-start:.0f}s)  [seeds ground truth]")
            ran += 1
            seeded.add(gt_path)

        for tier in tiers:
            print(f"\n=== {tier} " + "=" * (60 - len(tier)))
            for path in ordered(tier):
                if path in seeded:
                    continue
                rel = path.relative_to(ROOT)
                start = time.time()
                try:
                    papermill.execute_notebook(
                        str(path), str(Path(tmp) / path.name),
                        kernel_name="python3", cwd=str(ROOT), progress_bar=False,
                    )
                except Exception as exc:
                    msg = str(exc).strip().splitlines()
                    detail = next((l for l in reversed(msg) if l.strip()), "")[:110]
                    print(f"  FAIL  {rel}  ({time.time()-start:.0f}s)  {detail}")
                    failures.append((str(rel), detail))
                else:
                    print(f"  ok    {rel}  ({time.time()-start:.0f}s)")
                ran += 1

    print(f"\n{ran - len(failures)}/{ran} notebooks executed successfully")
    for name, detail in failures:
        print(f"  FAILED {name}: {detail}")
    if os.environ.get("RAG_FAKE_MODELS"):
        print("\n(ran with RAG_FAKE_MODELS=1: no Ollama server, no model downloads)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
