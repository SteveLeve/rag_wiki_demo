#!/usr/bin/env python3
"""Execute every notebook with papermill, in dependency order.

This is the acceptance gate. Run it with RAG_FAKE_MODELS=1 to exercise the whole
curriculum with no Ollama server and no model downloads.

    RAG_FAKE_MODELS=1 RAG_PG_PORT=5433 python scripts/run_notebooks.py
    python scripts/run_notebooks.py foundation intermediate   # a subset
"""
from __future__ import annotations

import os
import re
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TIERS = ("foundation", "intermediate", "advanced-techniques", "evaluation-lab")

# Within a tier, prerequisites first. INDEX notebooks are markdown-only.
WITHIN_TIER = {
    "foundation": ["00-setup-postgres-schema", "00-registry-and-tracking-utilities",
                   "00-load-or-generate-pattern", "01-basic-rag-in-memory",
                   "02-rag-postgresql-persistent"],
}

# The curriculum's dependency order is not its directory order.
# evaluation-lab/01 populates evaluation_groundtruth, and every advanced-techniques
# notebook measures itself against it -- so it must run after foundation (which
# creates the schema and embeddings it needs) but before the advanced tier.
# Walking the folders naively leaves ground_truth_questions empty, and the advanced
# notebooks then index into an empty list.
GROUND_TRUTH = ("evaluation-lab", "01-create-ground-truth-human-in-loop")


# A SyntaxError's traceback ends with a bare caret line, so "last non-empty line"
# reports `^` and tells you nothing. Prefer the last line that reads as an
# exception, and fall back to the last line carrying any word characters.
EXC_LINE = re.compile(r"^\s*(\w+(?:\.\w+)*(?:Error|Exception|Warning|Interrupt|Failure))\b")


def summarize(exc: BaseException) -> str:
    lines = [l.strip() for l in str(exc).strip().splitlines() if l.strip()]
    for line in reversed(lines):
        if EXC_LINE.match(line):
            return line
    for line in reversed(lines):
        if any(ch.isalnum() for ch in line):
            return line
    return type(exc).__name__


def ordered(tier: str) -> list[Path]:
    paths = sorted((ROOT / tier).glob("*.ipynb"))
    rank = {name: i for i, name in enumerate(WITHIN_TIER.get(tier, []))}
    return sorted(paths, key=lambda p: (rank.get(p.stem, 99), p.stem))


def plan(tiers: list[str]) -> list[tuple[str, Path]]:
    """Return (section label, notebook) pairs in the order they must execute."""
    gt_path = ROOT / GROUND_TRUTH[0] / f"{GROUND_TRUTH[1]}.ipynb"
    inject = gt_path.exists() and "advanced-techniques" in tiers and GROUND_TRUTH[0] in tiers

    steps: list[tuple[str, Path]] = []
    for tier in tiers:
        if tier == "advanced-techniques" and inject:
            steps.append(("prerequisite: ground truth", gt_path))
        for path in ordered(tier):
            if inject and path == gt_path:
                continue          # already scheduled as the prerequisite
            steps.append((tier, path))
    return steps


def main() -> int:
    import papermill

    tiers = sys.argv[1:] or list(TIERS)
    unknown = [t for t in tiers if t not in TIERS]
    if unknown:
        print(f"unknown tier(s): {unknown}. Choose from {TIERS}", file=sys.stderr)
        return 2

    failures: list[tuple[str, str]] = []
    ran = 0
    section = None

    with tempfile.TemporaryDirectory() as tmp:
        for label, path in plan(tiers):
            if label != section:
                section = label
                print(f"\n=== {label} " + "=" * max(4, 58 - len(label)))

            rel = path.relative_to(ROOT)
            start = time.time()
            try:
                papermill.execute_notebook(
                    str(path), str(Path(tmp) / path.name),
                    kernel_name="python3", cwd=str(ROOT), progress_bar=False,
                )
            except Exception as exc:
                detail = summarize(exc)[:110]
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
