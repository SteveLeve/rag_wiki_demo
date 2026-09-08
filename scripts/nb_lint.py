#!/usr/bin/env python3
"""Enforce notebook conventions. Read-only; runs in CI.

Every rule here corresponds to a bug that actually shipped in this repository.
The point is that fixing those bugs once is not enough -- nothing stopped them
recurring, and several recurred across six notebooks before anyone noticed.

    python scripts/nb_lint.py            # check all notebooks
    python scripts/nb_lint.py foundation # check one tier
"""
from __future__ import annotations

import ast
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
    (re.compile(r"\bcontent\s+as\s+chunk_text\b", re.I),
     "the embeddings tables have no `content` column; ragkit.db names it chunk_text"),
]


# Helpers that live in ragkit. A notebook must import one or define it inline as
# a marked teaching copy; reading it without either is a NameError on a clean run.
SHARED_HELPERS = {
    "start_experiment", "complete_experiment", "save_metrics", "compare_experiments",
    "compute_config_hash",
    "precision_at_k", "recall_at_k", "ndcg_at_k", "mean_reciprocal_rank", "dcg_score",
    "chunk_text", "cosine_similarity", "embed_one", "embed_texts", "table_name_for",
    "reciprocal_rank_fusion",
}

# Who is allowed to write a helper out by hand. Everyone else imports it.
#
# This is the teaching-copy rule from AGENTS.md, enforced. Before it existed the
# repo carried six textually distinct ndcg_at_k implementations across the
# notebooks, all sharing one wrong ideal-DCG, and nothing compared them.
TEACHING_OWNER = {
    "chunk_text": "foundation/01-basic-rag-in-memory",
    "cosine_similarity": "foundation/01-basic-rag-in-memory",
    "start_experiment": "foundation/00-registry-and-tracking-utilities",
    "complete_experiment": "foundation/00-registry-and-tracking-utilities",
    "save_metrics": "foundation/00-registry-and-tracking-utilities",
    "compare_experiments": "foundation/00-registry-and-tracking-utilities",
    "precision_at_k": "evaluation-lab/02-evaluation-metrics-framework",
    "recall_at_k": "evaluation-lab/02-evaluation-metrics-framework",
    "mean_reciprocal_rank": "evaluation-lab/02-evaluation-metrics-framework",
    "ndcg_at_k": "evaluation-lab/02-evaluation-metrics-framework",
    "bm25_search_postgresql": "advanced-techniques/07-hybrid-search",
    "reciprocal_rank_fusion": "advanced-techniques/07-hybrid-search",
}

TOP_LEVEL_DEF = re.compile(r"^def (\w+)\s*\(", re.M)

# Statement-shaped fragments that should never appear inside a comment.
SWALLOWED = re.compile(r"(?:print\(|= \[\]|with .*:|for .* in |if .*:|cur\.execute|def )")


def cell_source(cell: dict) -> str:
    src = cell.get("source", [])
    if isinstance(src, str):
        return src
    # Tolerate notebooks whose lines were stored without trailing newlines.
    if src and not any(line.endswith("\n") for line in src[:-1]):
        return "\n".join(src)
    return "".join(src)


def undefined_constants(source: str) -> list[str]:
    """SCREAMING_CASE names a notebook reads but never assigns.

    These are configuration knobs, and four notebooks shipped referencing ones
    that existed nowhere -- RRF_K, FILTER_STATUS, LIMIT_EXPERIMENTS,
    PRIMARY_METRIC. Each was a NameError on a clean run, invisible to anyone who
    had already defined the name in their kernel from an earlier notebook.
    """
    import builtins

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    assigned: set[str] = set()
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            (assigned if isinstance(node.ctx, ast.Store) else used).add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                assigned.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            assigned.add(node.name)
            if isinstance(node, ast.FunctionDef):
                for arg in node.args.args + node.args.kwonlyargs:
                    assigned.add(arg.arg)

    missing = used - assigned - set(dir(builtins))
    return sorted(
        name for name in missing
        # SCREAMING_CASE config knobs, plus the shared helpers a notebook must
        # either import from ragkit or define inline as a teaching copy.
        if re.fullmatch(r"[A-Z][A-Z0-9_]{2,}", name) or name in SHARED_HELPERS
    )


def check(path: Path) -> list[str]:
    problems: list[str] = []
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])
    notebook_id = f"{path.parent.name}/{path.stem}"

    if not cells:
        return [f"{path}: notebook has no cells (an empty .ipynb breaks nbformat and papermill)"]

    first = cells[0]
    if first.get("cell_type") != "markdown" or not cell_source(first).lstrip().startswith("# "):
        problems.append(f"{path}: first cell must be a markdown H1 title")

    whole = "\n".join(
        (cell_source(c) if cell_source(c).endswith("\n") else cell_source(c) + "\n")
        for c in cells if c.get("cell_type") == "code"
    )
    if not any(l.lstrip().startswith(("%", "!")) for l in whole.splitlines()):
        for name in undefined_constants(whole):
            problems.append(
                f"{path}: {name} is used but never assigned; it will raise NameError "
                "on a clean kernel"
            )

    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs") or cell.get("execution_count") is not None:
            problems.append(f"{path}: cell {i} has committed output; strip before commit")
        src = cell_source(cell)

        # Every code cell must be valid Python. A codemod that emits a wrongly
        # indented replacement produces a cell that only fails at execution time,
        # which is far too late and costs a full papermill run to discover.
        if not any(line.lstrip().startswith(("%", "!")) for line in src.splitlines()):
            try:
                ast.parse(src)
            except SyntaxError as exc:
                problems.append(
                    f"{path}: cell {i}: {type(exc).__name__} on line {exc.lineno}: {exc.msg}"
                )

        # A comment line that has swallowed real code. Several cells in this repo
        # shipped with whole blocks flattened onto one line; when that line began
        # with '#', the entire block became a comment. It parses cleanly and does
        # nothing, which is the worst possible failure mode.
        for lineno, line in enumerate(src.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") and len(stripped) > 150 and len(SWALLOWED.findall(stripped)) >= 2:
                problems.append(
                    f"{path}: cell {i} line {lineno}: comment line appears to have swallowed "
                    f"code ({len(stripped)} chars); it will silently do nothing"
                )

        for name in TOP_LEVEL_DEF.findall(src):
            owner = TEACHING_OWNER.get(name)
            if owner is None:
                continue
            if notebook_id != owner:
                problems.append(
                    f"{path}: cell {i}: defines {name}() inline, but {owner} owns that "
                    f"teaching copy; import it from ragkit instead"
                )
            elif "TEACHING COPY" not in src:
                problems.append(
                    f"{path}: cell {i}: {name}() is this notebook's teaching copy but carries "
                    "no '# TEACHING COPY' marker naming the ragkit function it mirrors"
                )

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
