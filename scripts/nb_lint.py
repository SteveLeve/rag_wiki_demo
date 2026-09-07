#!/usr/bin/env python3
"""Enforce notebook conventions. Read-only; runs in CI.

Every rule here corresponds to a bug that actually shipped in this repository.
The point is that fixing those bugs once is not enough -- nothing stopped them
recurring, and several recurred across six notebooks before anyone noticed.

    python scripts/nb_lint.py            # check all notebooks
    python scripts/nb_lint.py foundation # check one tier
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TIERS = ("foundation", "intermediate", "advanced-techniques", "evaluation-lab")

# (pattern, message) -- each one is a bug this repo actually had.
BANNED = [
    (re.compile(r"\bvector\(\s*\d+\s*\)"),
     "hardcoded vector dimension; let ragkit.db.ensure_embedding_table read it from the registry"),
    (re.compile(r"ollama\.embeddings\b"),
     "legacy Ollama API; use ragkit.embed (or ollama.embed) -- .embeddings returns "
     "non-normalized vectors for nomic-bert models"),
    (re.compile(r"\.replace\(\s*['\"]\.['\"]\s*,\s*['\"]_['\"]\s*\)"),
     "hand-rolled alias normalization; use ragkit.models.canonical_alias"),
    (re.compile(r"hf\.co/"),
     "HuggingFace GGUF path; use an Ollama-native tag from the ragkit catalog"),
    (re.compile(r"^\s*(?!#).*\binput\s*\(", re.M),
     "interactive input(); it hangs papermill forever"),
]


def cell_source(cell: dict) -> str:
    src = cell.get("source", [])
    if isinstance(src, str):
        return src
    # Tolerate notebooks whose lines were stored without trailing newlines.
    if src and not any(line.endswith("\n") for line in src[:-1]):
        return "\n".join(src)
    return "".join(src)


def check(path: Path) -> list[str]:
    problems: list[str] = []
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])

    if not cells:
        return [f"{path}: notebook has no cells (an empty .ipynb breaks nbformat and papermill)"]

    first = cells[0]
    if first.get("cell_type") != "markdown" or not cell_source(first).lstrip().startswith("# "):
        problems.append(f"{path}: first cell must be a markdown H1 title")

    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs") or cell.get("execution_count") is not None:
            problems.append(f"{path}: cell {i} has committed output; strip before commit")
        src = cell_source(cell)
        for pattern, message in BANNED:
            if pattern.search(src):
                problems.append(f"{path}: cell {i}: {message}")
    return problems


def main() -> int:
    targets = sys.argv[1:] or list(TIERS)
    paths = sorted(
        p for t in targets for p in (ROOT / t).glob("*.ipynb")
    )
    if not paths:
        print(f"no notebooks found in {targets}", file=sys.stderr)
        return 2

    all_problems: list[str] = []
    for path in paths:
        all_problems.extend(check(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path))

    for problem in all_problems:
        print(problem)
    print(f"\n{len(paths)} notebooks checked, {len(all_problems)} problem(s)")
    return 1 if all_problems else 0


if __name__ == "__main__":
    sys.exit(main())
